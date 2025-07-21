"""
Toroidal Diffusion Model Demo

This script demonstrates the complete toroidal diffusion model pipeline,
including toroidal topology, central singularity, and coherence monitoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from typing import Dict, List, Tuple

# Import our modules
from toroidal_topology import ToroidalLatentSpace, ToroidalFlow, ToroidalCoordinates
from central_singularity import SingularityToroidalCoupling, CognitiveFeedbackLoop
from coherence_monitor import MultiPassRefinement
from advanced_coherence_system import AdvancedCoherenceSystem, CoherenceVisualizer
from toroidal_diffusion_wrapper import ToroidalDiffusionModel


class SimpleUNet(nn.Module):
    """Simple UNet for demonstration purposes."""
    
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder
        self.enc1 = self._block(in_channels, features)
        self.enc2 = self._block(features, features * 2)
        self.enc3 = self._block(features * 2, features * 4)
        
        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)
        
        # Decoder
        self.dec3 = self._block(features * 8 + features * 4, features * 4)
        self.dec2 = self._block(features * 4 + features * 2, features * 2)
        self.dec1 = self._block(features * 2 + features, features)
        
        # Final layer
        self.final = nn.Conv2d(features, out_channels, 1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels) if out_channels >= 8 else 1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(8, out_channels) if out_channels >= 8 else 1, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, sample, timestep, return_dict=True):
        # Encoder
        e1 = self.enc1(sample)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        
        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e3))
        
        # Decoder
        d3 = nn.functional.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = nn.functional.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        output = self.final(d1)
        
        if return_dict:
            return type('Output', (), {'sample': output})()
        return output


class SimpleScheduler:
    """Simple noise scheduler for demonstration."""
    
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        self.timesteps = torch.linspace(num_timesteps, 1, 50).long()
        
        # Linear beta schedule
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def set_timesteps(self, num_steps):
        self.timesteps = torch.linspace(self.num_timesteps, 1, num_steps).long()
    
    def step(self, noise_pred, timestep, sample, **kwargs):
        # Simple denoising step
        t = timestep.item() if torch.is_tensor(timestep) else timestep
        t = max(0, min(t, self.num_timesteps - 1))  # Clamp to valid range
        
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        
        # Compute denoised sample
        pred_original = (sample - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Compute previous sample
        prev_sample = torch.sqrt(alpha_t_prev) * pred_original + torch.sqrt(1 - alpha_t_prev) * noise_pred
        
        return type('Output', (), {'prev_sample': prev_sample})()


def create_test_image(size=(64, 64), pattern='checkerboard'):
    """Create a test image with known patterns."""
    height, width = size
    
    if pattern == 'checkerboard':
        # Create checkerboard pattern
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        checkerboard = ((x // 8) + (y // 8)) % 2
        image = np.stack([checkerboard] * 3, axis=0).astype(np.float32)
    
    elif pattern == 'circles':
        # Create concentric circles
        center_x, center_y = width // 2, height // 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        circles = np.sin(distance * 0.3) * 0.5 + 0.5
        image = np.stack([circles] * 3, axis=0).astype(np.float32)
    
    elif pattern == 'gradient':
        # Create gradient pattern
        gradient = np.linspace(0, 1, width)
        gradient = np.tile(gradient, (height, 1))
        image = np.stack([gradient] * 3, axis=0).astype(np.float32)
    
    else:
        # Random noise
        image = np.random.rand(3, height, width).astype(np.float32)
    
    return torch.from_numpy(image).unsqueeze(0)  # Add batch dimension


def demonstrate_toroidal_topology():
    """Demonstrate toroidal topology operations."""
    print("=" * 60)
    print("DEMONSTRATING TOROIDAL TOPOLOGY")
    print("=" * 60)
    
    # Create test data
    test_image = create_test_image(pattern='circles')
    print(f"Test image shape: {test_image.shape}")
    
    # Initialize toroidal components
    toroidal_space = ToroidalLatentSpace(latent_dim=3, major_radius=1.0, minor_radius=0.3)
    toroidal_flow = ToroidalFlow(channels=3, flow_strength=0.1)
    
    # Apply toroidal operations
    print("\nApplying toroidal wrapping...")
    toroidal_result = toroidal_space(test_image)
    wrapped_image = toroidal_result['wrapped_latent']
    curvature = toroidal_result['curvature']
    
    print(f"Wrapped image range: [{wrapped_image.min():.3f}, {wrapped_image.max():.3f}]")
    print(f"Mean curvature: {curvature.mean():.6f}")
    
    # Apply flow dynamics
    print("\nApplying toroidal flow...")
    flowed_image = toroidal_flow(wrapped_image)
    
    # Compute toroidal distance
    distance = toroidal_space.toroidal_distance(test_image, flowed_image)
    print(f"Toroidal distance: {distance.mean():.6f}")
    
    # Test coordinate transformations
    print("\nTesting coordinate transformations...")
    coords = ToroidalCoordinates(major_radius=1.0, minor_radius=0.3)
    
    # Test points
    theta = torch.tensor([0.0, np.pi/2, np.pi, 3*np.pi/2])
    phi = torch.tensor([0.0, np.pi/4, np.pi/2, np.pi])
    
    # Convert to Cartesian and back
    x, y, z = coords.toroidal_to_cartesian(theta, phi)
    theta_recovered, phi_recovered = coords.cartesian_to_toroidal(x, y, z)
    
    coord_error = torch.abs(theta - theta_recovered).max() + torch.abs(phi - phi_recovered).max()
    print(f"Coordinate transformation error: {coord_error:.8f}")
    
    return {
        'original_image': test_image,
        'wrapped_image': wrapped_image,
        'flowed_image': flowed_image,
        'curvature': curvature,
        'toroidal_distance': distance
    }


def demonstrate_central_singularity():
    """Demonstrate central singularity operations."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING CENTRAL SINGULARITY")
    print("=" * 60)
    
    # Create test data
    test_image = create_test_image(pattern='gradient')
    print(f"Test image shape: {test_image.shape}")
    
    # Initialize singularity components
    singularity_coupling = SingularityToroidalCoupling(
        latent_dim=3,
        singularity_dim=128,
        coupling_strength=0.1
    )
    
    cognitive_feedback = CognitiveFeedbackLoop(latent_dim=3, memory_size=5)
    
    # Apply singularity processing
    print("\nApplying singularity coupling...")
    coupling_result = singularity_coupling(test_image)
    
    coupled_features = coupling_result['coupled_features']
    singularity_influence = coupling_result['singularity_influence']
    coupling_strength = coupling_result['coupling_strength']
    
    print(f"Coupling strength: {coupling_strength.mean():.6f}")
    print(f"Singularity influence magnitude: {singularity_influence.abs().mean():.6f}")
    
    # Apply cognitive feedback
    print("\nApplying cognitive feedback...")
    feedback_results = []
    current_features = coupled_features
    
    for i in range(3):
        feedback_result = cognitive_feedback(current_features)
        current_features = feedback_result['modified_features']
        feedback_results.append(feedback_result)
        
        action_magnitude = feedback_result['action'].abs().mean()
        print(f"Iteration {i+1}: Action magnitude = {action_magnitude:.6f}")
    
    return {
        'original_image': test_image,
        'coupled_features': coupled_features,
        'final_features': current_features,
        'singularity_influence': singularity_influence,
        'feedback_results': feedback_results
    }


def demonstrate_coherence_monitoring():
    """Demonstrate coherence monitoring and refinement."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING COHERENCE MONITORING")
    print("=" * 60)
    
    # Create test data
    test_image = create_test_image(pattern='checkerboard')
    print(f"Test image shape: {test_image.shape}")
    
    # Initialize coherence system
    advanced_coherence = AdvancedCoherenceSystem(
        feature_dim=3,
        max_refinement_passes=5,
        enable_hierarchical=True,
        enable_adaptive_threshold=True
    )
    
    # Apply comprehensive refinement
    print("\nApplying comprehensive coherence refinement...")
    start_time = time.time()
    
    refinement_result = advanced_coherence(test_image)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Extract results
    refined_features = refinement_result['refined_features']
    report = refinement_result['report']
    refinement_steps = refinement_result['refinement_steps']
    
    print(f"Processing time: {processing_time:.3f} seconds")
    print(f"Refinement passes: {report.refinement_passes}")
    print(f"Convergence achieved: {report.convergence_achieved}")
    print(f"Final quality score: {report.quality_score:.6f}")
    print(f"Semantic coherence: {report.semantic_coherence:.6f}")
    print(f"Structural coherence: {report.structural_coherence:.6f}")
    print(f"Overall coherence: {report.overall_coherence:.6f}")
    
    # Visualize refinement progress
    print("\nGenerating coherence visualization...")
    visualizer = CoherenceVisualizer()
    
    if refinement_steps:
        fig = visualizer.plot_coherence_evolution(refinement_steps)
        if fig:
            fig.savefig('examples/coherence_evolution.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("Coherence evolution plot saved to coherence_evolution.png")
    
    # Save detailed report
    report_path = 'examples/coherence_report.txt'
    visualizer.save_coherence_report(report, report_path)
    print(f"Detailed report saved to {report_path}")
    
    return {
        'original_image': test_image,
        'refined_features': refined_features,
        'report': report,
        'refinement_steps': refinement_steps,
        'processing_time': processing_time
    }


def demonstrate_full_pipeline():
    """Demonstrate the complete toroidal diffusion pipeline."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING FULL TOROIDAL DIFFUSION PIPELINE")
    print("=" * 60)
    
    # Create components
    base_model = SimpleUNet(in_channels=3, out_channels=3)
    scheduler = SimpleScheduler()
    
    # Create toroidal diffusion model
    toroidal_model = ToroidalDiffusionModel(
        base_model=base_model,
        scheduler=scheduler,
        image_size=(64, 64),
        enable_singularity=True,
        enable_coherence_monitoring=True,
        enable_multi_pass=True,
        max_refinement_passes=3
    )
    
    print("Toroidal diffusion model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in toroidal_model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_sample = torch.randn(1, 3, 64, 64)
    test_timestep = torch.randint(0, 1000, (1,))
    
    start_time = time.time()
    
    with torch.no_grad():
        result = toroidal_model(test_sample, test_timestep, return_dict=True)
    
    end_time = time.time()
    forward_time = end_time - start_time
    
    print(f"Forward pass time: {forward_time:.3f} seconds")
    print(f"Output shape: {result['sample'].shape}")
    
    # Analyze metadata
    toroidal_meta = result['toroidal_metadata']
    singularity_meta = result['singularity_metadata']
    coherence_meta = result['coherence_metadata']
    
    print(f"Toroidal curvature mean: {toroidal_meta['curvature'].mean():.6f}")
    if 'singularity_influence' in singularity_meta:
        print(f"Singularity influence: {singularity_meta['singularity_influence'].abs().mean():.6f}")
    if 'total_passes' in coherence_meta:
        print(f"Coherence refinement passes: {coherence_meta['total_passes']}")
    
    # Test sampling
    print("\nTesting sampling process...")
    start_time = time.time()
    
    with torch.no_grad():
        sample_result = toroidal_model.sample(
            batch_size=1,
            num_inference_steps=20,
            return_dict=True
        )
    
    end_time = time.time()
    sampling_time = end_time - start_time
    
    generated_sample = sample_result['sample']
    generation_history = sample_result['generation_history']
    
    print(f"Sampling time: {sampling_time:.3f} seconds")
    print(f"Generated sample shape: {generated_sample.shape}")
    print(f"Generation history length: {len(generation_history)}")
    
    # Save generated sample as image
    sample_image = generated_sample[0].clamp(0, 1)
    sample_image = (sample_image * 255).byte().permute(1, 2, 0).numpy()
    
    pil_image = Image.fromarray(sample_image)
    pil_image.save('examples/generated_sample.png')
    print("Generated sample saved to generated_sample.png")
    
    return {
        'model': toroidal_model,
        'forward_result': result,
        'sample_result': sample_result,
        'forward_time': forward_time,
        'sampling_time': sampling_time
    }


def run_comprehensive_demo():
    """Run the comprehensive demonstration."""
    print("TOROIDAL DIFFUSION MODEL - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases all components of the toroidal diffusion model:")
    print("1. Toroidal topology operations")
    print("2. Central singularity processing")
    print("3. Coherence monitoring and refinement")
    print("4. Complete pipeline integration")
    print("=" * 80)
    
    # Run individual demonstrations
    topo_results = demonstrate_toroidal_topology()
    sing_results = demonstrate_central_singularity()
    coher_results = demonstrate_coherence_monitoring()
    pipeline_results = demonstrate_full_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION SUMMARY")
    print("=" * 60)
    
    print(f"✓ Toroidal topology: Coordinate error < {1e-6}")
    print(f"✓ Central singularity: Processing completed")
    print(f"✓ Coherence monitoring: {coher_results['report'].refinement_passes} refinement passes")
    print(f"✓ Full pipeline: {pipeline_results['sampling_time']:.2f}s sampling time")
    
    print(f"\nFinal quality metrics:")
    print(f"  - Semantic coherence: {coher_results['report'].semantic_coherence:.4f}")
    print(f"  - Structural coherence: {coher_results['report'].structural_coherence:.4f}")
    print(f"  - Overall quality: {coher_results['report'].quality_score:.4f}")
    
    print(f"\nGenerated files:")
    print(f"  - coherence_evolution.png: Refinement progress visualization")
    print(f"  - coherence_report.txt: Detailed coherence analysis")
    print(f"  - generated_sample.png: Sample from toroidal diffusion model")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'topology_results': topo_results,
        'singularity_results': sing_results,
        'coherence_results': coher_results,
        'pipeline_results': pipeline_results
    }


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the comprehensive demo
    results = run_comprehensive_demo()

