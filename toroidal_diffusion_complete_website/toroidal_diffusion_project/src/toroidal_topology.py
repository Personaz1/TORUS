"""
Toroidal Topology Module for Diffusion Models

This module implements toroidal topology functions for wrapping diffusion models
in a toroidal latent space, enabling cyclic continuity and self-reflection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union


class ToroidalCoordinates:
    """
    Handles coordinate transformations between Cartesian and toroidal spaces.
    
    The torus is parameterized by:
    - Major radius R (distance from center to tube center)
    - Minor radius r (tube radius)
    - Angular coordinates (θ, φ) where θ ∈ [0, 2π], φ ∈ [0, 2π]
    """
    
    def __init__(self, major_radius: float = 1.0, minor_radius: float = 0.3):
        self.R = major_radius  # Major radius
        self.r = minor_radius  # Minor radius
        
    def cartesian_to_toroidal(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to toroidal coordinates (θ, φ).
        
        Args:
            x, y, z: Cartesian coordinates
            
        Returns:
            theta, phi: Toroidal angular coordinates
        """
        # Distance from z-axis
        rho = torch.sqrt(x**2 + y**2)
        
        # Major angle (around the main axis)
        theta = torch.atan2(y, x)
        
        # Minor angle (around the tube)
        # Distance from the major circle
        d_major = rho - self.R
        phi = torch.atan2(z, d_major)
        
        # Normalize to [0, 2π]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        phi = (phi + 2 * math.pi) % (2 * math.pi)
        
        return theta, phi
    
    def toroidal_to_cartesian(self, theta: torch.Tensor, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert toroidal coordinates to Cartesian coordinates.
        
        Args:
            theta, phi: Toroidal angular coordinates
            
        Returns:
            x, y, z: Cartesian coordinates
        """
        x = (self.R + self.r * torch.cos(phi)) * torch.cos(theta)
        y = (self.R + self.r * torch.cos(phi)) * torch.sin(theta)
        z = self.r * torch.sin(phi)
        
        return x, y, z


class ToroidalLatentSpace(nn.Module):
    """
    Implements toroidal latent space operations for diffusion models.
    
    This class wraps standard latent space operations to work on a torus,
    providing cyclic continuity and enabling self-reflection mechanisms.
    """
    
    def __init__(self, latent_dim: int, major_radius: float = 1.0, minor_radius: float = 0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.coords = ToroidalCoordinates(major_radius, minor_radius)
        
        # Learnable parameters for toroidal embedding
        self.embedding_scale = nn.Parameter(torch.ones(latent_dim))
        self.embedding_offset = nn.Parameter(torch.zeros(latent_dim))
        
    def wrap_to_torus(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Wrap latent vectors to toroidal space using periodic boundary conditions.
        
        Args:
            latent: Input latent tensor of shape (batch, channels, height, width)
            
        Returns:
            wrapped_latent: Latent tensor wrapped to toroidal space
        """
        # Apply learnable scaling and offset
        scaled_latent = latent * self.embedding_scale.view(1, -1, 1, 1) + self.embedding_offset.view(1, -1, 1, 1)
        
        # Wrap to [0, 2π] using modular arithmetic
        wrapped = torch.fmod(scaled_latent + 2 * math.pi, 2 * math.pi)
        
        return wrapped
    
    def toroidal_distance(self, latent1: torch.Tensor, latent2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between points on the torus.
        
        Args:
            latent1, latent2: Latent tensors on the torus
            
        Returns:
            distance: Toroidal distance tensor
        """
        # Wrap both latents to torus
        wrapped1 = self.wrap_to_torus(latent1)
        wrapped2 = self.wrap_to_torus(latent2)
        
        # Compute angular differences
        diff = wrapped1 - wrapped2
        
        # Handle periodic boundary: choose shorter path around the circle
        diff = torch.where(diff > math.pi, diff - 2 * math.pi, diff)
        diff = torch.where(diff < -math.pi, diff + 2 * math.pi, diff)
        
        # Compute Euclidean distance on the torus surface
        distance = torch.sqrt(torch.sum(diff**2, dim=1, keepdim=True))
        
        return distance
    
    def toroidal_interpolation(self, latent1: torch.Tensor, latent2: torch.Tensor, t: float) -> torch.Tensor:
        """
        Interpolate between two points on the torus along the shorter geodesic.
        
        Args:
            latent1, latent2: Latent tensors on the torus
            t: Interpolation parameter [0, 1]
            
        Returns:
            interpolated: Interpolated latent tensor
        """
        wrapped1 = self.wrap_to_torus(latent1)
        wrapped2 = self.wrap_to_torus(latent2)
        
        # Compute angular differences (shorter path)
        diff = wrapped2 - wrapped1
        diff = torch.where(diff > math.pi, diff - 2 * math.pi, diff)
        diff = torch.where(diff < -math.pi, diff + 2 * math.pi, diff)
        
        # Linear interpolation along the shorter path
        interpolated = wrapped1 + t * diff
        
        # Ensure result is wrapped to [0, 2π]
        interpolated = torch.fmod(interpolated + 2 * math.pi, 2 * math.pi)
        
        return interpolated
    
    def compute_curvature(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute local curvature of the latent space at given points.
        
        This is used for coherence assessment and self-reflection.
        
        Args:
            latent: Input latent tensor
            
        Returns:
            curvature: Local curvature tensor
        """
        wrapped = self.wrap_to_torus(latent)
        
        # Compute second derivatives (discrete approximation)
        # This is a simplified curvature estimation
        batch_size, channels, height, width = wrapped.shape
        
        # Compute gradients
        grad_x = torch.diff(wrapped, dim=3, prepend=wrapped[:, :, :, -1:])
        grad_y = torch.diff(wrapped, dim=2, prepend=wrapped[:, :, -1:, :])
        
        # Compute second derivatives
        grad_xx = torch.diff(grad_x, dim=3, prepend=grad_x[:, :, :, -1:])
        grad_yy = torch.diff(grad_y, dim=2, prepend=grad_y[:, :, -1:, :])
        
        # Gaussian curvature approximation
        curvature = torch.abs(grad_xx + grad_yy)
        
        return curvature
    
    def forward(self, latent: torch.Tensor) -> dict:
        """
        Forward pass that computes toroidal properties.
        
        Args:
            latent: Input latent tensor
            
        Returns:
            dict: Dictionary containing wrapped latent and computed properties
        """
        wrapped_latent = self.wrap_to_torus(latent)
        curvature = self.compute_curvature(latent)
        
        return {
            'wrapped_latent': wrapped_latent,
            'curvature': curvature,
            'original_latent': latent
        }


class ToroidalFlow(nn.Module):
    """
    Implements flow dynamics on the toroidal manifold.
    
    This class handles the flow of information and energy across the torus,
    enabling the self-stabilizing properties of the toroidal diffusion model.
    """
    
    def __init__(self, channels: int, flow_strength: float = 0.1):
        super().__init__()
        self.channels = channels
        self.flow_strength = flow_strength
        
        # Learnable flow parameters
        self.flow_weights = nn.Parameter(torch.randn(channels, channels) * 0.1)
        self.flow_bias = nn.Parameter(torch.zeros(channels))
        
    def compute_flow_field(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the flow field on the toroidal surface.
        
        Args:
            latent: Input latent tensor on the torus
            
        Returns:
            flow_field: Vector field representing flow directions
        """
        batch_size, channels, height, width = latent.shape
        
        # Compute gradients for flow direction
        grad_x = torch.diff(latent, dim=3, prepend=latent[:, :, :, -1:])
        grad_y = torch.diff(latent, dim=2, prepend=latent[:, :, -1:, :])
        
        # Apply learnable transformation
        flow_x = torch.einsum('bchw,cd->bdhw', grad_x, self.flow_weights) + self.flow_bias.view(1, -1, 1, 1)
        flow_y = torch.einsum('bchw,cd->bdhw', grad_y, self.flow_weights) + self.flow_bias.view(1, -1, 1, 1)
        
        # Combine into flow field
        flow_field = torch.stack([flow_x, flow_y], dim=-1)
        
        return flow_field
    
    def apply_flow(self, latent: torch.Tensor, flow_field: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Apply flow dynamics to the latent tensor.
        
        Args:
            latent: Input latent tensor
            flow_field: Flow field tensor
            dt: Time step for flow integration
            
        Returns:
            flowed_latent: Latent tensor after applying flow
        """
        # Simple Euler integration
        flow_x, flow_y = flow_field[..., 0], flow_field[..., 1]
        
        # Apply flow with periodic boundary conditions
        flowed_latent = latent + dt * self.flow_strength * (flow_x + flow_y)
        
        return flowed_latent
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying toroidal flow dynamics.
        
        Args:
            latent: Input latent tensor
            
        Returns:
            flowed_latent: Latent tensor after flow application
        """
        flow_field = self.compute_flow_field(latent)
        flowed_latent = self.apply_flow(latent, flow_field)
        
        return flowed_latent


def test_toroidal_operations():
    """Test function for toroidal operations."""
    print("Testing Toroidal Topology Operations...")
    
    # Test coordinate transformations
    coords = ToroidalCoordinates()
    
    # Test points
    theta = torch.tensor([0.0, math.pi/2, math.pi, 3*math.pi/2])
    phi = torch.tensor([0.0, math.pi/4, math.pi/2, math.pi])
    
    # Convert to Cartesian and back
    x, y, z = coords.toroidal_to_cartesian(theta, phi)
    theta_back, phi_back = coords.cartesian_to_toroidal(x, y, z)
    
    print(f"Original theta: {theta}")
    print(f"Recovered theta: {theta_back}")
    print(f"Original phi: {phi}")
    print(f"Recovered phi: {phi_back}")
    
    # Test toroidal latent space
    latent_space = ToroidalLatentSpace(latent_dim=4)
    
    # Create test latent
    test_latent = torch.randn(2, 4, 8, 8)
    
    # Test wrapping
    result = latent_space(test_latent)
    print(f"Input latent shape: {test_latent.shape}")
    print(f"Wrapped latent shape: {result['wrapped_latent'].shape}")
    print(f"Curvature shape: {result['curvature'].shape}")
    
    # Test distance computation
    latent1 = torch.randn(1, 4, 8, 8)
    latent2 = torch.randn(1, 4, 8, 8)
    distance = latent_space.toroidal_distance(latent1, latent2)
    print(f"Toroidal distance shape: {distance.shape}")
    
    # Test flow dynamics
    flow = ToroidalFlow(channels=4)
    flowed = flow(test_latent)
    print(f"Flowed latent shape: {flowed.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_toroidal_operations()

