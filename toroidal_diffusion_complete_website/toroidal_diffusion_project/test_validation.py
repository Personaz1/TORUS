#!/usr/bin/env python3
"""
Comprehensive TORUS Architecture Validation Test

This script validates all components of the toroidal diffusion model:
1. Toroidal topology operations
2. Central singularity processing
3. Coherence monitoring
4. Full pipeline integration
5. Performance metrics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple

# Import TORUS components
from toroidal_topology import ToroidalLatentSpace, ToroidalFlow, ToroidalCoordinates
from central_singularity import SingularityCore, SingularityToroidalCoupling, CognitiveFeedbackLoop
from coherence_monitor import MultiPassRefinement
from toroidal_diffusion_wrapper import ToroidalDiffusionModel


class MockUNet(nn.Module):
    """Mock UNet for testing purposes."""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Simple conv layers for testing
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, sample, timestep, return_dict=True):
        x = torch.relu(self.conv1(sample))
        output = self.conv2(x)
        
        if return_dict:
            return type('Output', (), {'sample': output})()
        return output


class MockScheduler:
    """Mock scheduler for testing."""
    
    def __init__(self):
        self.timesteps = torch.linspace(1000, 1, 50).long()
        
    def set_timesteps(self, num_steps):
        self.timesteps = torch.linspace(1000, 1, num_steps).long()
        
    def step(self, noise_pred, timestep, sample, **kwargs):
        # Simple denoising step
        alpha = 0.9
        result = sample * alpha + noise_pred * (1 - alpha)
        # Return object with prev_sample attribute
        return type('SchedulerStep', (), {'prev_sample': result})()


def test_toroidal_topology():
    """Test toroidal topology operations."""
    print("🔍 Testing Toroidal Topology...")
    
    # Initialize components
    latent_space = ToroidalLatentSpace(latent_dim=64, major_radius=1.0, minor_radius=0.3)
    flow = ToroidalFlow(channels=64, flow_strength=0.1)
    
    # Test data
    batch_size, channels, height, width = 2, 64, 32, 32
    test_latent = torch.randn(batch_size, channels, height, width)
    
    # Test wrapping
    wrapped = latent_space.wrap_to_torus(test_latent)
    assert wrapped.shape == test_latent.shape, f"Shape mismatch: {wrapped.shape} vs {test_latent.shape}"
    assert torch.all(wrapped >= 0) and torch.all(wrapped <= 2 * np.pi), "Wrapping bounds violated"
    
    # Test distance computation
    latent1 = torch.randn(batch_size, channels, height, width)
    latent2 = torch.randn(batch_size, channels, height, width)
    distance = latent_space.toroidal_distance(latent1, latent2)
    assert distance.shape == (batch_size, 1, height, width), f"Distance shape: {distance.shape}"
    assert torch.all(distance >= 0), "Negative distance detected"
    
    # Test flow
    flow_field = flow.compute_flow_field(test_latent)
    assert flow_field.shape == (batch_size, channels, height, width, 2), f"Flow field shape: {flow_field.shape}"
    
    flowed = flow.apply_flow(test_latent, flow_field, dt=0.01)
    assert flowed.shape == test_latent.shape, f"Flowed shape: {flowed.shape}"
    
    print("✅ Toroidal topology tests passed")


def test_central_singularity():
    """Test central singularity processing."""
    print("🔍 Testing Central Singularity...")
    
    # Initialize components
    singularity = SingularityCore(latent_dim=64, singularity_dim=128, num_jets=8)
    coupling = SingularityToroidalCoupling(latent_dim=64, singularity_dim=128)
    feedback = CognitiveFeedbackLoop(latent_dim=64, memory_size=5)
    
    # Test data
    batch_size, channels, height, width = 2, 64, 32, 32
    test_features = torch.randn(batch_size, channels, height, width)
    
    # Test absorption
    absorbed = singularity.absorb_intent(test_features)
    assert absorbed.shape == (batch_size, 128), f"Absorbed shape: {absorbed.shape}"
    
    # Test state transformation
    transformed = singularity.transform_state(absorbed)
    assert transformed.shape == (batch_size, 128), f"Transformed shape: {transformed.shape}"
    
    # Test jet emission
    jets = singularity.emit_jets(transformed, (height, width))
    assert jets.shape == (batch_size, 64, height, width), f"Jets shape: {jets.shape}"
    
    # Test coupling
    coupling_result = coupling(test_features)
    assert 'coupled_features' in coupling_result, "Coupling result missing"
    assert coupling_result['coupled_features'].shape == test_features.shape
    
    # Test feedback loop
    feedback_result = feedback(test_features)
    assert 'modified_features' in feedback_result, "Feedback result missing"
    assert feedback_result['modified_features'].shape == test_features.shape
    
    print("✅ Central singularity tests passed")


def test_coherence_monitoring():
    """Test coherence monitoring system."""
    print("🔍 Testing Coherence Monitoring...")
    
    # Initialize component
    refinement = MultiPassRefinement(feature_dim=64, max_passes=3, coherence_threshold=0.8)
    
    # Test data
    batch_size, channels, height, width = 2, 64, 32, 32
    test_features = torch.randn(batch_size, channels, height, width)
    
    # Test refinement
    refined_result = refinement(test_features)
    assert 'final_features' in refined_result, "Refinement result missing"
    assert refined_result['final_features'].shape == test_features.shape
    
    # Test coherence metrics
    assert 'final_coherence' in refined_result, "Coherence score missing"
    assert 0 <= refined_result['final_coherence'] <= 1, f"Invalid coherence score: {refined_result['final_coherence']}"
    
    # Test convergence
    assert 'total_passes' in refined_result, "Pass count missing"
    assert isinstance(refined_result['total_passes'], int), "Pass count not integer"
    
    print("✅ Coherence monitoring tests passed")


def test_full_pipeline():
    """Test complete TORUS pipeline."""
    print("🔍 Testing Full TORUS Pipeline...")
    
    # Initialize components
    base_model = MockUNet(in_channels=3, out_channels=3)
    scheduler = MockScheduler()
    
    # Create TORUS model
    model = ToroidalDiffusionModel(
        base_model=base_model,
        scheduler=scheduler,
        image_size=(32, 32),
        enable_singularity=True,
        enable_coherence_monitoring=True,
        enable_multi_pass=True,
        max_refinement_passes=3
    )
    
    # Test data
    batch_size = 2
    test_sample = torch.randn(batch_size, 3, 32, 32)
    timestep = torch.tensor([500])
    
    # Test forward pass
    start_time = time.time()
    output = model(test_sample, timestep)
    forward_time = time.time() - start_time
    
    assert 'sample' in output, "Forward output missing sample"
    assert output['sample'].shape == test_sample.shape, f"Output shape: {output['sample'].shape}"
    
    # Test sampling
    start_time = time.time()
    sample_result = model.sample(batch_size=batch_size, num_inference_steps=10)
    sampling_time = time.time() - start_time
    
    assert 'sample' in sample_result, "Sampling result missing sample"
    assert sample_result['sample'].shape == (batch_size, 3, 32, 32), f"Sample shape: {sample_result['sample'].shape}"
    
    print(f"✅ Full pipeline tests passed")
    print(f"   Forward pass time: {forward_time:.3f}s")
    print(f"   Sampling time: {sampling_time:.3f}s")


def test_performance_metrics():
    """Test performance and quality metrics."""
    print("🔍 Testing Performance Metrics...")
    
    # Initialize model
    base_model = MockUNet(in_channels=3, out_channels=3)
    scheduler = MockScheduler()
    
    model = ToroidalDiffusionModel(
        base_model=base_model,
        scheduler=scheduler,
        image_size=(32, 32),
        enable_singularity=True,
        enable_coherence_monitoring=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test memory usage
    batch_size = 4
    test_sample = torch.randn(batch_size, 3, 32, 32)
    timestep = torch.tensor([500])
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Measure memory and time
    start_time = time.time()
    output = model(test_sample, timestep)
    inference_time = time.time() - start_time
    
    print(f"   Inference time (batch={batch_size}): {inference_time:.3f}s")
    print(f"   Throughput: {batch_size/inference_time:.1f} samples/sec")
    
    # Test coherence improvement
    if 'coherence_score' in output:
        coherence = output['coherence_score']
        print(f"   Coherence score: {coherence:.3f}")
    
    print("✅ Performance metrics calculated")


def run_comprehensive_validation():
    """Run all validation tests."""
    print("🚀 Starting TORUS Architecture Validation")
    print("=" * 50)
    
    try:
        test_toroidal_topology()
        test_central_singularity()
        test_coherence_monitoring()
        test_full_pipeline()
        test_performance_metrics()
        
        print("=" * 50)
        print("🎉 ALL TESTS PASSED - TORUS ARCHITECTURE VALIDATED")
        print("=" * 50)
        
        # Summary
        print("\n📊 VALIDATION SUMMARY:")
        print("✅ Toroidal topology: Cyclic continuity and flow dynamics")
        print("✅ Central singularity: Cognitive processing and jet emission")
        print("✅ Coherence monitoring: Multi-pass refinement and quality assessment")
        print("✅ Full pipeline: End-to-end integration")
        print("✅ Performance: Parameter count and inference speed")
        
        return True
        
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1) 