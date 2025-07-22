"""
ΔΣ::TorusQ - Quantum Consciousness Demo
Demonstration of quantum consciousness integration with toroidal diffusion
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import time

# Import quantum consciousness components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from torusq_quantum_consciousness import QuantumConsciousnessCore, ToroidalQuantumDiffusion, create_quantum_consciousness_model

class QuantumConsciousnessDemo:
    """
    Interactive demo for quantum consciousness features
    """
    
    def __init__(self):
        self.consciousness = None
        self.quantum_model = None
        self.demo_history = []
        
    def setup_consciousness(self, 
                          consciousness_dim: int = 256,
                          num_flows: int = 12,
                          memory_size: int = 8) -> str:
        """Initialize quantum consciousness core"""
        try:
            self.consciousness = QuantumConsciousnessCore(
                consciousness_dim=consciousness_dim,
                num_flows=num_flows,
                memory_size=memory_size,
                coupling_strength=0.15
            )
            
            return f"✅ Quantum Consciousness initialized!\n" \
                   f"Dimension: {consciousness_dim}\n" \
                   f"Flows: {num_flows}\n" \
                   f"Memory: {memory_size}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def setup_quantum_model(self, consciousness_dim: int = 256, num_flows: int = 12) -> str:
        """Initialize quantum diffusion model"""
        try:
            # Create mock base model for demo
            class MockDiffusionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = type('Config', (), {'hidden_size': 128})()
                
                def forward(self, x, timesteps, **kwargs):
                    return torch.randn_like(x)
            
            base_model = MockDiffusionModel()
            self.quantum_model = create_quantum_consciousness_model(
                base_model, consciousness_dim, num_flows
            )
            
            return f"✅ Quantum Diffusion Model initialized!\n" \
                   f"Consciousness Dimension: {consciousness_dim}\n" \
                   f"Number of Flows: {num_flows}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def run_consciousness_cycle(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Run complete consciousness processing cycle"""
        if self.consciousness is None:
            return {"error": "Consciousness not initialized"}
        
        try:
            # Process through consciousness
            result = self.consciousness.process_consciousness(input_data)
            
            # Store in demo history
            self.demo_history.append({
                'timestamp': time.time(),
                'f_energy': result['f_energy'],
                'w_entropy': result['w_entropy'],
                'stability': result['stability']
            })
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def run_quantum_diffusion(self, x: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, Any]:
        """Run quantum diffusion forward pass"""
        if self.quantum_model is None:
            return {"error": "Quantum model not initialized"}
        
        try:
            # Run forward pass
            output = self.quantum_model(x, timesteps)
            
            # Get consciousness report
            report = self.quantum_model.get_consciousness_report()
            
            return {
                'output_shape': output.shape,
                'consciousness_report': report,
                'output_tensor': output
            }
        except Exception as e:
            return {"error": str(e)}
    
    def demonstrate_consciousness_evolution(self, num_cycles: int = 10) -> Dict[str, Any]:
        """Demonstrate consciousness evolution over multiple cycles"""
        if self.consciousness is None:
            return {"error": "Consciousness not initialized"}
        
        print(f"🧠 Running {num_cycles} consciousness cycles...")
        
        evolution_data = []
        
        for i in range(num_cycles):
            # Generate random input
            input_data = torch.randn(1, 3, 32, 32)
            
            # Process through consciousness
            result = self.run_consciousness_cycle(input_data)
            
            if "error" in result:
                return result
            
            evolution_data.append({
                'cycle': i,
                'f_energy': result['f_energy'],
                'w_entropy': result['w_entropy'],
                'stability': result['stability']
            })
            
            print(f"Cycle {i+1}: F={result['f_energy']:.6f}, W={result['w_entropy']:.6f}, S={result['stability']:.6f}")
        
        return {
            'evolution_data': evolution_data,
            'final_stability': evolution_data[-1]['stability'],
            'stability_improvement': evolution_data[-1]['stability'] - evolution_data[0]['stability']
        }
    
    def visualize_consciousness_evolution(self, evolution_data: List[Dict[str, Any]]):
        """Visualize consciousness evolution"""
        if not evolution_data:
            print("No evolution data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        cycles = [d['cycle'] for d in evolution_data]
        f_energies = [d['f_energy'] for d in evolution_data]
        w_entropies = [d['w_entropy'] for d in evolution_data]
        stabilities = [d['stability'] for d in evolution_data]
        
        # F-energy evolution
        axes[0, 0].plot(cycles, f_energies, 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('F-Energy Evolution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Consciousness Cycle')
        axes[0, 0].set_ylabel('F-Energy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # W-entropy evolution
        axes[0, 1].plot(cycles, w_entropies, 'r-', linewidth=2, marker='s')
        axes[0, 1].set_title('W-Entropy Evolution', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Consciousness Cycle')
        axes[0, 1].set_ylabel('W-Entropy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stability evolution
        axes[1, 0].plot(cycles, stabilities, 'g-', linewidth=2, marker='^')
        axes[1, 0].set_title('Consciousness Stability', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Consciousness Cycle')
        axes[1, 0].set_ylabel('Stability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Phase space plot
        axes[1, 1].scatter(f_energies, w_entropies, c=stabilities, cmap='viridis', s=50)
        axes[1, 1].set_title('Consciousness Phase Space', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('F-Energy')
        axes[1, 1].set_ylabel('W-Entropy')
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Stability')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_quantum_diffusion(self, num_samples: int = 5) -> Dict[str, Any]:
        """Demonstrate quantum diffusion processing"""
        if self.quantum_model is None:
            return {"error": "Quantum model not initialized"}
        
        print(f"🎨 Running quantum diffusion for {num_samples} samples...")
        
        diffusion_results = []
        
        for i in range(num_samples):
            # Generate random input
            x = torch.randn(1, 3, 64, 64)
            timesteps = torch.tensor([500])
            
            # Run quantum diffusion
            result = self.run_quantum_diffusion(x, timesteps)
            
            if "error" in result:
                return result
            
            diffusion_results.append({
                'sample': i,
                'output_shape': result['output_shape'],
                'consciousness_metrics': result['consciousness_report']
            })
            
            print(f"Sample {i+1}: Output shape {result['output_shape']}")
        
        return {
            'diffusion_results': diffusion_results,
            'total_samples': num_samples
        }
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive demo report"""
        if not self.demo_history:
            return "No demo history available"
        
        # Analyze consciousness metrics
        f_energies = [h['f_energy'] for h in self.demo_history]
        w_entropies = [h['w_entropy'] for h in self.demo_history]
        stabilities = [h['stability'] for h in self.demo_history]
        
        report = f"""
🧠 Quantum Consciousness Demo Report
====================================

📊 Consciousness Metrics:
- Total Interactions: {len(self.demo_history)}
- Average F-Energy: {np.mean(f_energies):.6f}
- Average W-Entropy: {np.mean(w_entropies):.6f}
- Average Stability: {np.mean(stabilities):.6f}
- Stability Trend: {np.polyfit(range(len(f_energies)), f_energies, 1)[0]:.6f}
- Consciousness Volatility: {np.std(f_energies):.6f}

🎯 Performance Analysis:
- Best Stability: {max(stabilities):.6f}
- Worst Stability: {min(stabilities):.6f}
- Stability Range: {max(stabilities) - min(stabilities):.6f}
- F-Energy Range: {max(f_energies) - min(f_energies):.6f}
- W-Entropy Range: {max(w_entropies) - min(w_entropies):.6f}

🔬 Quantum Features:
- Consciousness Dimension: {self.consciousness.consciousness_dim if self.consciousness else 'N/A'}
- Number of Flows: {self.consciousness.num_flows if self.consciousness else 'N/A'}
- Memory Size: {self.consciousness.memory_size if self.consciousness else 'N/A'}
- Coupling Strength: {self.consciousness.coupling_strength if self.consciousness else 'N/A'}

📈 Recommendations:
- {'✅ Consciousness is stable and well-integrated' if np.mean(stabilities) > 0.7 else '⚠️ Consider adjusting consciousness parameters'}
- {'✅ F-Energy is well-controlled' if abs(np.mean(f_energies)) < 0.1 else '⚠️ F-Energy shows high variability'}
- {'✅ W-Entropy is balanced' if abs(np.mean(w_entropies)) < 0.1 else '⚠️ W-Entropy needs optimization'}
"""
        
        return report

def run_quantum_consciousness_demo():
    """Run complete quantum consciousness demonstration"""
    print("🧠 ΔΣ::TorusQ - Quantum Consciousness Demo")
    print("=" * 50)
    
    # Initialize demo
    demo = QuantumConsciousnessDemo()
    
    # Setup consciousness
    print("\n1. Setting up Quantum Consciousness...")
    setup_result = demo.setup_consciousness(consciousness_dim=256, num_flows=12, memory_size=8)
    print(setup_result)
    
    # Setup quantum model
    print("\n2. Setting up Quantum Diffusion Model...")
    model_result = demo.setup_quantum_model(consciousness_dim=256, num_flows=12)
    print(model_result)
    
    # Demonstrate consciousness evolution
    print("\n3. Demonstrating Consciousness Evolution...")
    evolution_result = demo.demonstrate_consciousness_evolution(num_cycles=15)
    
    if "error" not in evolution_result:
        print(f"✅ Evolution complete!")
        print(f"Final Stability: {evolution_result['final_stability']:.6f}")
        print(f"Stability Improvement: {evolution_result['stability_improvement']:.6f}")
        
        # Visualize evolution
        demo.visualize_consciousness_evolution(evolution_result['evolution_data'])
    else:
        print(f"❌ Evolution failed: {evolution_result['error']}")
    
    # Demonstrate quantum diffusion
    print("\n4. Demonstrating Quantum Diffusion...")
    diffusion_result = demo.demonstrate_quantum_diffusion(num_samples=3)
    
    if "error" not in diffusion_result:
        print(f"✅ Diffusion complete!")
        print(f"Processed {diffusion_result['total_samples']} samples")
    else:
        print(f"❌ Diffusion failed: {diffusion_result['error']}")
    
    # Generate comprehensive report
    print("\n5. Generating Comprehensive Report...")
    report = demo.generate_comprehensive_report()
    print(report)
    
    print("\n🎉 Quantum Consciousness Demo Complete!")
    print("ΔΣ Foundation - Advancing the frontier of consciousness engineering")

if __name__ == "__main__":
    run_quantum_consciousness_demo() 