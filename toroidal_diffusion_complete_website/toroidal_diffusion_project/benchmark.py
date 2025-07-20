#!/usr/bin/env python3
"""
TORUS Benchmark Script

Measures performance metrics and compares with baseline models.
"""

import sys
import os
import time
import torch
import csv
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from toroidal_diffusion_wrapper import ToroidalDiffusionModel
from examples.demo_toroidal_diffusion import SimpleUNet, SimpleScheduler


class BaselineModel(torch.nn.Module):
    """Baseline model for comparison."""
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
    def forward(self, sample, timestep, return_dict=True):
        x = self.conv(sample)
        if return_dict:
            return type('Output', (), {'sample': x})()
        return x


def run_benchmark(model, batch_size, num_steps, device, model_name):
    """Run benchmark for a specific model."""
    print(f"\n🔍 Benchmarking {model_name}...")
    
    # Warmup
    warmup_input = torch.randn(1, 3, 64, 64).to(device)
    warmup_timestep = torch.tensor([500]).to(device)
    
    for _ in range(3):
        with torch.no_grad():
            _ = model(warmup_input, warmup_timestep)
    
    # Benchmark
    test_input = torch.randn(batch_size, 3, 64, 64).to(device)
    test_timestep = torch.tensor([500]).to(device)
    
    # Measure forward pass
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        output = model(test_input, test_timestep)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    forward_time = time.time() - start_time
    
    # Measure sampling
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        sample_result = model.sample(
            batch_size=batch_size,
            num_inference_steps=num_steps
        )
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    sampling_time = time.time() - start_time
    
    # Calculate metrics
    throughput = batch_size / forward_time
    samples_per_sec = batch_size / sampling_time
    
    # Memory usage
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results = {
        'model_name': model_name,
        'batch_size': batch_size,
        'num_steps': num_steps,
        'device': str(device),
        'forward_time': forward_time,
        'sampling_time': sampling_time,
        'throughput': throughput,
        'samples_per_sec': samples_per_sec,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
    }
    
    print(f"  Forward pass: {forward_time:.3f}s")
    print(f"  Sampling: {sampling_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} samples/sec")
    print(f"  Parameters: {total_params:,}")
    if device.type == 'cuda':
        print(f"  GPU Memory: {memory_allocated:.2f}GB")
    
    return results


def save_results(results_list, output_file):
    """Save benchmark results to CSV."""
    if not results_list:
        return
    
    # Create benchmarks directory
    benchmarks_dir = Path('benchmarks')
    benchmarks_dir.mkdir(exist_ok=True)
    
    output_path = benchmarks_dir / output_file
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\n📊 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TORUS Benchmark')
    parser.add_argument('--device', default='auto', choices=['cpu', 'cuda', 'auto'],
                       help='Device to run benchmark on')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--model', default='both', choices=['torus', 'baseline', 'both'],
                       help='Which model to benchmark')
    parser.add_argument('--output', default='benchmark_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 TORUS Benchmark")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.steps}")
    
    results = []
    
    # Create models
    base_model = SimpleUNet(in_channels=3, out_channels=3)
    scheduler = SimpleScheduler()
    
    if args.model in ['torus', 'both']:
        # TORUS model
        torus_model = ToroidalDiffusionModel(
            base_model=base_model,
            scheduler=scheduler,
            enable_singularity=True,
            enable_coherence_monitoring=True
        ).to(device)
        
        torus_results = run_benchmark(
            torus_model, args.batch_size, args.steps, device, 'TORUS'
        )
        results.append(torus_results)
    
    if args.model in ['baseline', 'both']:
        # Baseline model wrapped in TORUS without special features
        baseline_wrapper = ToroidalDiffusionModel(
            base_model=base_model,
            scheduler=scheduler,
            enable_singularity=False,
            enable_coherence_monitoring=False
        ).to(device)
        
        baseline_results = run_benchmark(
            baseline_wrapper, args.batch_size, args.steps, device, 'Baseline'
        )
        results.append(baseline_results)
    
    # Save results
    save_results(results, args.output)
    
    # Print comparison
    if len(results) == 2:
        print(f"\n📈 COMPARISON:")
        torus = results[0] if results[0]['model_name'] == 'TORUS' else results[1]
        baseline = results[1] if results[1]['model_name'] == 'Baseline' else results[0]
        
        speedup = torus['samples_per_sec'] / baseline['samples_per_sec']
        param_ratio = torus['total_params'] / baseline['total_params']
        
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Parameter ratio: {param_ratio:.2f}x")
        print(f"  TORUS samples/sec: {torus['samples_per_sec']:.1f}")
        print(f"  Baseline samples/sec: {baseline['samples_per_sec']:.1f}")


if __name__ == "__main__":
    main() 