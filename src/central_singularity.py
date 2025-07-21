"""
Central Singularity Module

This module implements the central singularity of the toroidal diffusion model,
which acts as a self-reflective node of cognition - absorbing latent intent,
transforming internal state, and emitting structured informational jets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from einops import rearrange, reduce, repeat


class SingularityCore(nn.Module):
    """
    The core singularity that processes all information flowing through the torus center.
    
    This acts as the central cognitive node that:
    1. Absorbs latent intent from the toroidal surface
    2. Transforms and integrates internal state
    3. Emits structured informational jets back to the surface
    """
    
    def __init__(self, 
                 latent_dim: int, 
                 singularity_dim: int = 256,
                 num_jets: int = 8,
                 absorption_strength: float = 0.1,
                 emission_strength: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.singularity_dim = singularity_dim
        self.num_jets = num_jets
        self.absorption_strength = absorption_strength
        self.emission_strength = emission_strength
        
        # Absorption network - processes incoming information
        self.absorption_net = nn.Sequential(
            nn.Linear(latent_dim, singularity_dim),
            nn.LayerNorm(singularity_dim),
            nn.SiLU(),
            nn.Linear(singularity_dim, singularity_dim),
            nn.LayerNorm(singularity_dim),
            nn.SiLU()
        )
        
        # Internal state transformation - the cognitive core
        num_heads = min(8, singularity_dim // 64) if singularity_dim >= 64 else 1
        self.cognitive_core = nn.ModuleList([
            nn.MultiheadAttention(singularity_dim, num_heads=num_heads, batch_first=True),
            nn.Sequential(
                nn.Linear(singularity_dim, singularity_dim * 4),
                nn.SiLU(),
                nn.Linear(singularity_dim * 4, singularity_dim)
            )
        ])
        
        # Emission network - generates informational jets
        self.emission_net = nn.Sequential(
            nn.Linear(singularity_dim, singularity_dim * 2),
            nn.SiLU(),
            nn.Linear(singularity_dim * 2, num_jets * latent_dim),
            nn.Tanh()
        )
        
        # Learnable singularity state
        self.singularity_state = nn.Parameter(torch.randn(1, singularity_dim) * 0.1)
        
        # Jet direction embeddings (learnable)
        self.jet_directions = nn.Parameter(torch.randn(num_jets, 2) * 0.1)  # (theta, phi) for each jet
        
    def absorb_intent(self, toroidal_features: torch.Tensor) -> torch.Tensor:
        """
        Absorb latent intent from the toroidal surface into the singularity.
        
        Args:
            toroidal_features: Features from the toroidal surface [B, C, H, W]
            
        Returns:
            absorbed_intent: Absorbed and processed intent [B, singularity_dim]
        """
        batch_size, channels, height, width = toroidal_features.shape
        
        # Global average pooling to extract global intent
        global_intent = F.adaptive_avg_pool2d(toroidal_features, 1).flatten(1)
        
        # Weighted absorption based on distance from center
        center_h, center_w = height // 2, width // 2
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=toroidal_features.device),
            torch.arange(width, device=toroidal_features.device),
            indexing='ij'
        )
        
        # Distance from center (inverted for absorption weight)
        dist_from_center = torch.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
        absorption_weight = 1.0 / (1.0 + dist_from_center)
        absorption_weight = absorption_weight / absorption_weight.sum()
        
        # Weighted spatial pooling
        weighted_features = toroidal_features * absorption_weight.unsqueeze(0).unsqueeze(0)
        spatial_intent = weighted_features.sum(dim=[2, 3])
        
        # Combine global and spatial intent
        combined_intent = global_intent + spatial_intent
        
        # Process through absorption network
        absorbed_intent = self.absorption_net(combined_intent)
        
        return absorbed_intent
    
    def transform_state(self, absorbed_intent: torch.Tensor) -> torch.Tensor:
        """
        Transform the internal singularity state using absorbed intent.
        
        Args:
            absorbed_intent: Absorbed intent from toroidal surface
            
        Returns:
            transformed_state: New singularity state
        """
        batch_size = absorbed_intent.shape[0]
        
        # Expand singularity state for batch
        current_state = self.singularity_state.expand(batch_size, -1)
        
        # Combine current state with absorbed intent
        combined = torch.stack([current_state, absorbed_intent], dim=1)  # [B, 2, D]
        
        # Self-attention for cognitive processing
        attn_layer, ffn_layer = self.cognitive_core
        
        # Self-attention
        attended, _ = attn_layer(combined, combined, combined)
        attended = attended + combined  # Residual connection
        
        # Feed-forward processing
        transformed = ffn_layer(attended)
        transformed = transformed + attended  # Residual connection
        
        # Extract the transformed singularity state
        transformed_state = transformed[:, 0]  # Take the first token (singularity state)
        
        return transformed_state
    
    def emit_jets(self, transformed_state: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Emit structured informational jets from the singularity to the toroidal surface.
        
        Args:
            transformed_state: Transformed singularity state
            target_shape: Target spatial shape (H, W) for the jets
            
        Returns:
            emitted_jets: Informational jets projected onto toroidal surface [B, C, H, W]
        """
        batch_size = transformed_state.shape[0]
        height, width = target_shape
        
        # Generate jet information
        jet_info = self.emission_net(transformed_state)  # [B, num_jets * latent_dim]
        jet_info = jet_info.view(batch_size, self.num_jets, self.latent_dim)
        
        # Create spatial grid
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height, device=transformed_state.device),
            torch.linspace(-1, 1, width, device=transformed_state.device),
            indexing='ij'
        )
        
        # Convert to polar coordinates
        theta = torch.atan2(y_coords, x_coords)
        radius = torch.sqrt(x_coords**2 + y_coords**2)
        
        # Initialize emission field
        emission_field = torch.zeros(batch_size, self.latent_dim, height, width, 
                                   device=transformed_state.device)
        
        # Emit each jet
        for jet_idx in range(self.num_jets):
            # Jet direction
            jet_theta = self.jet_directions[jet_idx, 0]
            jet_phi = self.jet_directions[jet_idx, 1]
            
            # Compute jet influence based on angular distance
            angular_dist = torch.abs(theta - jet_theta)
            angular_dist = torch.min(angular_dist, 2 * math.pi - angular_dist)  # Wrap around
            
            # Jet strength decreases with angular distance and radius
            jet_strength = torch.exp(-angular_dist**2 / 0.5) * torch.exp(-radius**2 / 2.0)
            
            # Apply jet information
            jet_contribution = jet_info[:, jet_idx].unsqueeze(-1).unsqueeze(-1) * jet_strength.unsqueeze(0)
            emission_field += jet_contribution
        
        return emission_field
    
    def forward(self, toroidal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete singularity processing cycle.
        
        Args:
            toroidal_features: Input features from toroidal surface
            
        Returns:
            result: Dictionary containing all singularity outputs
        """
        # Absorption phase
        absorbed_intent = self.absorb_intent(toroidal_features)
        
        # Transformation phase
        transformed_state = self.transform_state(absorbed_intent)
        
        # Emission phase
        target_shape = toroidal_features.shape[2:]
        emitted_jets = self.emit_jets(transformed_state, target_shape)
        
        return {
            'absorbed_intent': absorbed_intent,
            'transformed_state': transformed_state,
            'emitted_jets': emitted_jets,
            'singularity_influence': emitted_jets * self.emission_strength
        }


class SingularityToroidalCoupling(nn.Module):
    """
    Manages the coupling between the central singularity and the toroidal surface.
    
    This module handles the bidirectional information flow and ensures
    proper integration of singularity effects with toroidal dynamics.
    """
    
    def __init__(self, 
                 latent_dim: int,
                 singularity_dim: int = 256,
                 coupling_strength: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.singularity_dim = singularity_dim
        self.coupling_strength = coupling_strength
        
        # Singularity core
        self.singularity = SingularityCore(latent_dim, singularity_dim)
        
        # Coupling networks
        num_groups = min(8, latent_dim) if latent_dim >= 8 else 1
        self.surface_to_singularity = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(num_groups, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )
        
        self.singularity_to_surface = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(num_groups, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim, 1)
        )
        
        # Adaptive coupling strength
        self.coupling_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(latent_dim, max(1, latent_dim // 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(1, latent_dim // 4), 1, 1),
            nn.Sigmoid()
        )
    
    def compute_coupling_strength(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive coupling strength based on feature characteristics.
        
        Args:
            features: Input features
            
        Returns:
            coupling_strength: Adaptive coupling strength
        """
        return self.coupling_modulator(features)
    
    def forward(self, toroidal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process toroidal features through singularity coupling.
        
        Args:
            toroidal_features: Features on the toroidal surface
            
        Returns:
            result: Dictionary containing coupled features and singularity outputs
        """
        # Prepare features for singularity processing
        prepared_features = self.surface_to_singularity(toroidal_features)
        
        # Process through singularity
        singularity_result = self.singularity(prepared_features)
        
        # Process singularity output for surface integration
        processed_jets = self.singularity_to_surface(singularity_result['emitted_jets'])
        
        # Compute adaptive coupling strength
        coupling_strength = self.compute_coupling_strength(toroidal_features)
        
        # Apply coupling
        coupled_influence = processed_jets * coupling_strength * self.coupling_strength
        
        # Integrate with original features
        coupled_features = toroidal_features + coupled_influence
        
        return {
            'coupled_features': coupled_features,
            'original_features': toroidal_features,
            'singularity_influence': coupled_influence,
            'coupling_strength': coupling_strength,
            **singularity_result
        }


class CognitiveFeedbackLoop(nn.Module):
    """
    Implements the cognitive feedback loop between observation, integration, and action.
    
    This creates a continuous cycle of self-reflection and adaptation.
    """
    
    def __init__(self, latent_dim: int, memory_size: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.memory_size = memory_size
        
        # Observation network
        num_groups = min(8, latent_dim) if latent_dim >= 8 else 1
        self.observer = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GroupNorm(num_groups, latent_dim),
            nn.SiLU(),
            nn.Conv2d(latent_dim, latent_dim // 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Integration network (memory + current observation)
        self.integrator = nn.Sequential(
            nn.Linear(latent_dim // 4 * (memory_size + 1), latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4)
        )
        
        # Action network
        self.actor = nn.Sequential(
            nn.Linear(latent_dim // 4, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        
        # Memory buffer
        self.register_buffer('memory', torch.zeros(memory_size, latent_dim // 4))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
    
    def observe(self, features: torch.Tensor) -> torch.Tensor:
        """
        Observe current state and extract key information.
        
        Args:
            features: Current features
            
        Returns:
            observation: Compressed observation
        """
        return self.observer(features)
    
    def update_memory(self, observation: torch.Tensor):
        """
        Update memory with new observation.
        
        Args:
            observation: New observation to store
        """
        batch_size = observation.shape[0]
        
        # Store observation in memory (simple circular buffer)
        ptr = self.memory_ptr.item()
        self.memory[ptr] = observation[0]  # Store first batch item
        self.memory_ptr[0] = (ptr + 1) % self.memory_size
    
    def integrate(self, current_observation: torch.Tensor) -> torch.Tensor:
        """
        Integrate current observation with memory.
        
        Args:
            current_observation: Current observation
            
        Returns:
            integrated_state: Integrated cognitive state
        """
        batch_size = current_observation.shape[0]
        
        # Expand memory for batch
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        memory_flat = memory_expanded.flatten(1)
        
        # Combine with current observation
        combined = torch.cat([current_observation, memory_flat], dim=1)
        
        # Integrate
        integrated_state = self.integrator(combined)
        
        return integrated_state
    
    def act(self, integrated_state: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Generate action based on integrated state.
        
        Args:
            integrated_state: Integrated cognitive state
            original_shape: Original spatial shape
            
        Returns:
            action: Action to apply to features
        """
        # Generate action vector
        action_vector = self.actor(integrated_state)
        
        # Reshape to spatial dimensions
        height, width = original_shape
        action = action_vector.unsqueeze(-1).unsqueeze(-1)
        action = action.expand(-1, -1, height, width)
        
        return action
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete cognitive feedback loop.
        
        Args:
            features: Input features
            
        Returns:
            result: Dictionary containing cognitive processing results
        """
        # Observe
        observation = self.observe(features)
        
        # Integrate with memory
        integrated_state = self.integrate(observation)
        
        # Generate action
        action = self.act(integrated_state, features.shape[2:])
        
        # Apply action
        modified_features = features + action * 0.1  # Small action strength
        
        # Update memory
        self.update_memory(observation)
        
        return {
            'modified_features': modified_features,
            'observation': observation,
            'integrated_state': integrated_state,
            'action': action,
            'original_features': features
        }


def test_central_singularity():
    """Test function for central singularity components."""
    print("Testing Central Singularity Components...")
    
    # Test parameters
    batch_size, latent_dim, height, width = 2, 64, 32, 32
    test_features = torch.randn(batch_size, latent_dim, height, width)
    
    # Test SingularityCore
    print("Testing SingularityCore...")
    singularity_core = SingularityCore(latent_dim, singularity_dim=128, num_jets=8)
    core_result = singularity_core(test_features)
    
    print(f"Absorbed intent shape: {core_result['absorbed_intent'].shape}")
    print(f"Transformed state shape: {core_result['transformed_state'].shape}")
    print(f"Emitted jets shape: {core_result['emitted_jets'].shape}")
    print(f"Singularity influence shape: {core_result['singularity_influence'].shape}")
    
    # Test SingularityToroidalCoupling
    print("\nTesting SingularityToroidalCoupling...")
    coupling = SingularityToroidalCoupling(latent_dim, singularity_dim=128)
    coupling_result = coupling(test_features)
    
    print(f"Coupled features shape: {coupling_result['coupled_features'].shape}")
    print(f"Coupling strength shape: {coupling_result['coupling_strength'].shape}")
    print(f"Coupling strength mean: {coupling_result['coupling_strength'].mean().item():.4f}")
    
    # Test CognitiveFeedbackLoop
    print("\nTesting CognitiveFeedbackLoop...")
    feedback_loop = CognitiveFeedbackLoop(latent_dim, memory_size=5)
    
    # Run multiple iterations to test memory
    for i in range(3):
        feedback_result = feedback_loop(test_features)
        print(f"Iteration {i+1}:")
        print(f"  Modified features shape: {feedback_result['modified_features'].shape}")
        print(f"  Observation shape: {feedback_result['observation'].shape}")
        print(f"  Action mean: {feedback_result['action'].mean().item():.4f}")
    
    print("\nAll central singularity tests passed!")


if __name__ == "__main__":
    test_central_singularity()

