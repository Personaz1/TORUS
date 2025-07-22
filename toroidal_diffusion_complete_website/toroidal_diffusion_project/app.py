#!/usr/bin/env python3
"""
TORUS Gradio App for Hugging Face Space
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from toroidal_diffusion_wrapper import ToroidalDiffusionModel
from examples.demo_toroidal_diffusion import SimpleUNet, SimpleScheduler
from toroidal_diffusion_core_def import ToroidalCore, GEOM, HYPER


class TORUSApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.toroidal_core = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize TORUS model with real geometry."""
        print("Initializing TORUS model...")
        
        # Create base components
        base_model = SimpleUNet(in_channels=3, out_channels=3)
        scheduler = SimpleScheduler()
        
        # Load demo checkpoint if available
        checkpoint_path = "weights/demo_ckpt.pt"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        
        # Create TORUS model
        self.model = ToroidalDiffusionModel(
            base_model=base_model,
            scheduler=scheduler,
            enable_singularity=True,
            enable_coherence_monitoring=True,
            max_refinement_passes=3
        ).to(self.device)
        
        # Initialize real toroidal geometry
        self.toroidal_core = ToroidalCore(GEOM, self.device)
        print(f"✅ Real toroidal geometry initialized")
        print(f"   Grid: {GEOM['N_theta']}×{GEOM['N_phi']}")
        print(f"   Geometry: R={GEOM['R']}, r={GEOM['r_base']}, α={GEOM['alpha']}")
        
        print(f"TORUS model initialized on {self.device}")
    
    def generate_image(self, prompt, num_steps, batch_size, enable_singularity, enable_coherence):
        """Generate image using TORUS with real geometry."""
        try:
            # Update model settings
            self.model.enable_singularity = enable_singularity
            self.model.enable_coherence_monitoring = enable_coherence
            
            # Run toroidal diffusion with real geometry
            if self.toroidal_core is not None:
                print("🔄 Running toroidal diffusion with real geometry...")
                deltas, final_state, metadata = self.toroidal_core(
                    steps=HYPER['steps'],
                    D=HYPER['D'],
                    dt=HYPER['dt'],
                    return_history=True
                )
                
                # Get geometric analysis
                geom_analysis = self.toroidal_core.get_geometric_analysis()
                throat_state = self.toroidal_core.get_throat_state()
                
                print(f"   Gaussian curvature: {geom_analysis['mean_gaussian_curvature']:.4f}")
                print(f"   Mean curvature: {geom_analysis['mean_surface_curvature']:.4f}")
                print(f"   Throat activity: {throat_state.abs().mean():.4f}")
            
            # Generate sample with base model
            with torch.no_grad():
                sample_result = self.model.sample(
                    batch_size=batch_size,
                    num_inference_steps=num_steps
                )
            
            # Convert to image
            sample = sample_result['sample'][0]  # Take first image
            sample = sample.clamp(0, 1)
            sample = (sample * 255).byte().permute(1, 2, 0).cpu().numpy()
            
            # Create PIL image
            image = Image.fromarray(sample)
            
            # Enhanced metadata with geometry
            metadata_info = {
                "prompt": prompt,
                "steps": num_steps,
                "batch_size": batch_size,
                "singularity": enable_singularity,
                "coherence": enable_coherence,
                "device": str(self.device)
            }
            
            if self.toroidal_core is not None:
                metadata_info.update({
                    "gaussian_curvature": f"{geom_analysis['mean_gaussian_curvature']:.4f}",
                    "mean_curvature": f"{geom_analysis['mean_surface_curvature']:.4f}",
                    "throat_activity": f"{throat_state.abs().mean():.4f}",
                    "total_energy": f"{geom_analysis['total_energy']:.6f}",
                    "toroidal_deltas": f"{deltas.mean():.4f}"
                })
            
            return image, str(metadata_info)
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def benchmark_performance(self):
        """Run performance benchmark with geometry."""
        try:
            import time
            
            # Warmup
            test_input = torch.randn(1, 3, 64, 64).to(self.device)
            timestep = torch.tensor([500]).to(self.device)
            
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(test_input, timestep)
            
            # Benchmark base model
            start_time = time.time()
            with torch.no_grad():
                sample_result = self.model.sample(
                    batch_size=4,
                    num_inference_steps=10
                )
            end_time = time.time()
            
            sampling_time = end_time - start_time
            throughput = 4 / sampling_time
            
            # Benchmark toroidal geometry
            geom_start = time.time()
            if self.toroidal_core is not None:
                deltas, final_state, _ = self.toroidal_core(
                    steps=50,  # Reduced for benchmark
                    D=HYPER['D'],
                    dt=HYPER['dt'],
                    return_history=False
                )
            geom_time = time.time() - geom_start
            
            result = f"Base model: {throughput:.1f} samples/sec\n"
            result += f"Sampling time: {sampling_time:.3f}s\n"
            
            if self.toroidal_core is not None:
                result += f"Toroidal geometry: {geom_time:.3f}s\n"
                result += f"Geometry deltas: {deltas.mean():.4f}"
            
            return result
            
        except Exception as e:
            return f"Benchmark error: {str(e)}"
    
    def get_geometry_metrics(self):
        """Get real-time geometry metrics."""
        if self.toroidal_core is None:
            return "Geometry not initialized"
        
        try:
            geom_analysis = self.toroidal_core.get_geometric_analysis()
            throat_state = self.toroidal_core.get_throat_state()
            
            metrics = f"🔬 Real Geometry Metrics:\n"
            metrics += f"Gaussian curvature: {geom_analysis['mean_gaussian_curvature']:.4f}\n"
            metrics += f"Mean curvature: {geom_analysis['mean_surface_curvature']:.4f}\n"
            metrics += f"Flow magnitude: {geom_analysis['flow_magnitude']:.4f}\n"
            metrics += f"Throat activity: {throat_state.abs().mean():.4f}\n"
            metrics += f"Total energy: {geom_analysis['total_energy']:.6f}"
            
            return metrics
        except Exception as e:
            return f"Geometry error: {str(e)}"


# Initialize app
app = TORUSApp()

# Create Gradio interface with real geometry
def create_compatible_interface():
    """Create Gradio interface with real toroidal geometry."""
    
    # Function wrappers
    def generate_wrapper(prompt, num_steps, batch_size, enable_singularity, enable_coherence):
        return app.generate_image(prompt, num_steps, batch_size, enable_singularity, enable_coherence)
    
    def benchmark_wrapper():
        return app.benchmark_performance()
    
    def geometry_wrapper():
        return app.get_geometry_metrics()
    
    # Create interface with geometry features - Fixed schema compatibility
    interface = gr.Interface(
        fn=generate_wrapper,
        inputs=[
            gr.Textbox(label="Prompt", value="A serene landscape with mountains"),
            gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Steps"),
            gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Batch Size"),
            gr.Checkbox(label="Singularity", value=True, interactive=True),
            gr.Checkbox(label="Coherence", value=True, interactive=True)
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Info", lines=5)
        ],
        title="🌀 TORUS: Real Toroidal Geometry",
        description="Self-Stabilizing, Self-Reflective Generative Architecture with Real Differential Geometry",
        allow_flagging="never"
    )
    
    return interface

def create_simple_fallback_interface():
    """Create a simple fallback interface without complex parameters."""
    
    def simple_generation(prompt, steps, batch_size):
        """Simple generation without complex parameters."""
        try:
            # Create a simple test image
            size = 256
            img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            
            # Add some variation based on prompt
            if "mountain" in prompt.lower():
                for i in range(size):
                    for j in range(size):
                        height = int(255 * (1 - i/size))
                        img_array[i, j] = [height//3, height//2, height]
            
            image = Image.fromarray(img_array)
            
            metadata = {
                "prompt": prompt,
                "steps": steps,
                "batch_size": batch_size,
                "status": "Generated successfully (fallback mode)",
                "device": str(app.device)
            }
            
            return image, str(metadata)
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    # Simple interface without problematic components
    interface = gr.Interface(
        fn=simple_generation,
        inputs=[
            gr.Textbox(label="Prompt", value="A serene landscape with mountains"),
            gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Steps"),
            gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Batch Size")
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Info", lines=5)
        ],
        title="🌀 TORUS: Fallback Interface",
        description="Self-Stabilizing Generative Architecture - Fallback Mode",
        allow_flagging="never"
    )
    
    return interface

# Create and launch interface
if __name__ == "__main__":
    try:
        # Import gradio with error handling
        import gradio as gr
        
        # Check gradio version
        import pkg_resources
        gradio_version = pkg_resources.get_distribution("gradio").version
        print(f"Gradio version: {gradio_version}")
        
        # Try main interface first
        try:
            print("Attempting to create main interface...")
            demo = create_compatible_interface()
            print("✅ Main interface created successfully!")
        except Exception as e:
            print(f"Main interface failed: {e}")
            print("Falling back to simple interface...")
            demo = create_simple_fallback_interface()
            print("✅ Fallback interface created successfully!")
        
        # Launch with proper settings for HF Spaces
        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860,
            share=True,
            show_error=True,
            debug=True,
            quiet=False
        )
    except Exception as e:
        print(f"Failed to launch Gradio interface: {e}")
        import traceback
        traceback.print_exc()
        
        print("Falling back to command-line demo...")
        
        # Fallback to command-line demo with geometry
        print("\n=== TORUS Command Line Demo with Real Geometry ===")
        print("Generating sample image...")
        
        try:
            result = app.generate_image(
                "A serene landscape with mountains", 
                20, 1, True, True
            )
            if result[0] is not None:
                print("✅ Generation successful!")
                print(f"Metadata: {result[1]}")
            else:
                print(f"❌ Generation failed: {result[1]}")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc() 