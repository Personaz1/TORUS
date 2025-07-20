"""
Performance Benchmark Suite for DEF Toroidal Architecture
=========================================================

Comprehensive performance testing and optimization analysis
for the enhanced DEF (Diffusion-Embedding-Flow) architecture.

Author: Stepan Egoshin (ΔΣ-Foundation)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import psutil
import gc
from dataclasses import dataclass

from toroidal_diffusion_core_def import ToroidalCore, JetHead
from enhanced_toroidal_wrapper import EnhancedToroidalDiffusionModel, ToroidalConfig

@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    execution_time: float
    memory_usage: float
    gpu_memory: float
    throughput: float
    convergence_steps: int
    final_coherence: float
    geometric_stability: float

class DEFPerformanceBenchmark:
    """Performance benchmark suite for DEF architecture."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results: List[BenchmarkResult] = []
        
        print(f"🚀 DEF Performance Benchmark Suite")
        print(f"Device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
        print("-" * 60)
    
    def measure_memory(self) -> Tuple[float, float]:
        """Measure CPU and GPU memory usage."""
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        return cpu_memory, gpu_memory
    
    def benchmark_core_diffusion(self, config: Dict) -> BenchmarkResult:
        """Benchmark core toroidal diffusion performance."""
        print(f"📊 Testing core diffusion with geometry {config['N_theta']}x{config['N_phi']}")
        
        # Setup
        geom = {k: v for k, v in config.items() if k in ['N_theta', 'N_phi', 'R', 'r_base', 'alpha', 'h', 'phi_c']}
        core = ToroidalCore(geom, self.device)
        
        # Warm up
        for _ in range(3):
            core(steps=10, D=0.05, dt=0.1)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark
        start_memory, start_gpu = self.measure_memory()
        start_time = time.time()
        
        deltas, final_state, metadata = core(
            steps=config.get('steps', 100),
            D=config.get('D', 0.05),
            dt=config.get('dt', 0.15),
            return_history=True
        )
        
        end_time = time.time()
        end_memory, end_gpu = self.measure_memory()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        gpu_memory = end_gpu - start_gpu
        throughput = config.get('steps', 100) / execution_time
        
        # Convergence analysis
        convergence_threshold = 1e-5
        convergence_steps = len(deltas)
        for i, delta in enumerate(deltas):
            if abs(delta.item()) < convergence_threshold:
                convergence_steps = i
                break
        
        # Geometric analysis
        geometric_analysis = core.get_geometric_analysis()
        geometric_stability = 1.0 / (1.0 + geometric_analysis['total_energy'])
        
        result = BenchmarkResult(
            test_name=f"Core_Diffusion_{config['N_theta']}x{config['N_phi']}",
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            throughput=throughput,
            convergence_steps=convergence_steps,
            final_coherence=abs(deltas[-1].item()),
            geometric_stability=geometric_stability
        )
        
        self.results.append(result)
        print(f"  ⏱️  Time: {execution_time:.3f}s | 🧠 Memory: {memory_usage:.1f}MB | 🎯 Throughput: {throughput:.1f} steps/s")
        return result
    
    def benchmark_enhanced_wrapper(self, config: ToroidalConfig) -> BenchmarkResult:
        """Benchmark enhanced wrapper performance."""
        print(f"📊 Testing enhanced wrapper with SBERT: {config.enable_sbert}")
        
        # Setup
        model = EnhancedToroidalDiffusionModel(config=config, device=self.device)
        
        # Warm up
        for _ in range(3):
            model(return_dict=True)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark
        start_memory, start_gpu = self.measure_memory()
        start_time = time.time()
        
        result_dict = model(return_dict=True)
        
        end_time = time.time()
        end_memory, end_gpu = self.measure_memory()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        gpu_memory = end_gpu - start_gpu
        throughput = config.steps / execution_time
        
        result = BenchmarkResult(
            test_name=f"Enhanced_Wrapper_SBERT_{config.enable_sbert}",
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            throughput=throughput,
            convergence_steps=len(result_dict['deltas']),
            final_coherence=result_dict['coherence_loss'].item(),
            geometric_stability=result_dict['geometric_analysis']['total_energy']
        )
        
        self.results.append(result)
        print(f"  ⏱️  Time: {execution_time:.3f}s | 🧠 Memory: {memory_usage:.1f}MB | 🎯 Throughput: {throughput:.1f} steps/s")
        return result
    
    def benchmark_sampling_performance(self, config: ToroidalConfig) -> BenchmarkResult:
        """Benchmark sampling performance."""
        print(f"📊 Testing sampling performance")
        
        # Setup
        model = EnhancedToroidalDiffusionModel(config=config, device=self.device)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark
        start_memory, start_gpu = self.measure_memory()
        start_time = time.time()
        
        samples = model.sample(batch_size=4, return_history=True)
        
        end_time = time.time()
        end_memory, end_gpu = self.measure_memory()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        gpu_memory = end_gpu - start_gpu
        throughput = 4 / execution_time  # samples per second
        
        result = BenchmarkResult(
            test_name="Sampling_Performance",
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            throughput=throughput,
            convergence_steps=len(samples['history']) if samples['history'] else 0,
            final_coherence=0.0,  # Not applicable for sampling
            geometric_stability=samples['final_geometric_analysis']['total_energy']
        )
        
        self.results.append(result)
        print(f"  ⏱️  Time: {execution_time:.3f}s | 🧠 Memory: {memory_usage:.1f}MB | 🎯 Throughput: {throughput:.2f} samples/s")
        return result
    
    def benchmark_scalability(self) -> List[BenchmarkResult]:
        """Test scalability across different geometry sizes."""
        print(f"📊 Testing scalability across geometry sizes")
        
        geometries = [
            {'N_theta': 16, 'N_phi': 32},
            {'N_theta': 32, 'N_phi': 64},
            {'N_theta': 64, 'N_phi': 128},
            {'N_theta': 128, 'N_phi': 256}
        ]
        
        base_config = {
            'R': 1.0, 'r_base': 0.4, 'alpha': 0.48, 'h': 0.22, 'phi_c': 0.18,
            'steps': 50, 'D': 0.05, 'dt': 0.15
        }
        
        scalability_results = []
        for geom in geometries:
            config = {**base_config, **geom}
            try:
                result = self.benchmark_core_diffusion(config)
                scalability_results.append(result)
            except Exception as e:
                print(f"  ❌ Failed for {geom}: {e}")
        
        return scalability_results
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive performance benchmark suite."""
        print("🎯 Starting Comprehensive DEF Performance Benchmark")
        print("=" * 60)
        
        # Test configurations
        base_config = ToroidalConfig(
            N_theta=32,
            N_phi=64,
            steps=50,
            enable_sbert=False
        )
        
        sbert_config = ToroidalConfig(
            N_theta=32,
            N_phi=64,
            steps=50,
            enable_sbert=True
        )
        
        # Run benchmarks
        print("\n1️⃣ Core Diffusion Performance")
        core_result = self.benchmark_core_diffusion({
            'N_theta': 32, 'N_phi': 64, 'R': 1.0, 'r_base': 0.4,
            'alpha': 0.48, 'h': 0.22, 'phi_c': 0.18,
            'steps': 50, 'D': 0.05, 'dt': 0.15
        })
        
        print("\n2️⃣ Enhanced Wrapper (No SBERT)")
        wrapper_result = self.benchmark_enhanced_wrapper(base_config)
        
        print("\n3️⃣ Enhanced Wrapper (With SBERT)")
        sbert_result = self.benchmark_enhanced_wrapper(sbert_config)
        
        print("\n4️⃣ Sampling Performance")
        sampling_result = self.benchmark_sampling_performance(base_config)
        
        print("\n5️⃣ Scalability Analysis")
        scalability_results = self.benchmark_scalability()
        
        # Generate summary
        summary = self.generate_summary()
        
        print("\n" + "=" * 60)
        print("🏁 Benchmark Complete!")
        return summary
    
    def generate_summary(self) -> Dict:
        """Generate performance summary."""
        if not self.results:
            return {}
        
        summary = {
            'total_tests': len(self.results),
            'average_execution_time': np.mean([r.execution_time for r in self.results]),
            'average_memory_usage': np.mean([r.memory_usage for r in self.results]),
            'average_throughput': np.mean([r.throughput for r in self.results]),
            'best_performance': min(self.results, key=lambda x: x.execution_time),
            'worst_performance': max(self.results, key=lambda x: x.execution_time),
            'results': self.results
        }
        
        print(f"\n📈 Performance Summary:")
        print(f"  Tests run: {summary['total_tests']}")
        print(f"  Average execution time: {summary['average_execution_time']:.3f}s")
        print(f"  Average memory usage: {summary['average_memory_usage']:.1f}MB")
        print(f"  Average throughput: {summary['average_throughput']:.1f} ops/s")
        print(f"  Best performance: {summary['best_performance'].test_name} ({summary['best_performance'].execution_time:.3f}s)")
        print(f"  Worst performance: {summary['worst_performance'].test_name} ({summary['worst_performance'].execution_time:.3f}s)")
        
        return summary
    
    def save_results(self, filename: str = "def_benchmark_results.txt"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            f.write("DEF Architecture Performance Benchmark Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"Test: {result.test_name}\n")
                f.write(f"  Execution Time: {result.execution_time:.3f}s\n")
                f.write(f"  Memory Usage: {result.memory_usage:.1f}MB\n")
                f.write(f"  GPU Memory: {result.gpu_memory:.1f}MB\n")
                f.write(f"  Throughput: {result.throughput:.2f} ops/s\n")
                f.write(f"  Convergence Steps: {result.convergence_steps}\n")
                f.write(f"  Final Coherence: {result.final_coherence:.6f}\n")
                f.write(f"  Geometric Stability: {result.geometric_stability:.6f}\n")
                f.write("-" * 30 + "\n")
        
        print(f"📄 Results saved to {filename}")

def main():
    """Run the performance benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benchmark = DEFPerformanceBenchmark(device)
    
    try:
        summary = benchmark.run_comprehensive_benchmark()
        benchmark.save_results()
        
        print(f"\n✅ Benchmark completed successfully!")
        return summary
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()

