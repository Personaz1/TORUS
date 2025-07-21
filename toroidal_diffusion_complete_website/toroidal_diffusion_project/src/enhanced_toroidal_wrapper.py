"""
Enhanced Toroidal Diffusion Wrapper with DEF Architecture
=========================================================

Integrates the advanced DEF (Diffusion-Embedding-Flow) toroidal architecture
with existing diffusion models and provides a unified API.

Features:
- Double-sheet toroidal geometry with throat synchronization
- SBERT semantic embeddings for coherence monitoring
- Jet decoder for structured output generation
- Integration with Hugging Face Diffusers
- Real-time geometric analysis and flow statistics

Author: Stepan Solncev (ΔΣ-Foundation)
License: MIT
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import warnings

# Import our DEF core
from toroidal_diffusion_core_def import ToroidalCore, JetHead, GEOM, HYPER

@dataclass
class ToroidalConfig:
    """Configuration for toroidal diffusion model."""
    # Geometry parameters
    N_theta: int = 64
    N_phi: int = 128
    R: float = 1.0
    r_base: float = 0.4
    alpha: float = 0.48
    h: float = 0.22
    phi_c: float = 0.18
    
    # Diffusion parameters
    D: float = 0.05
    dt: float = 0.15
    steps: int = 160
    tau_fixed: float = 5e-3
    tau_stop: float = 1e-4
    
    # Model parameters
    enable_sbert: bool = True
    sbert_model: str = 'all-MiniLM-L6-v2'
    jet_vocab_size: int = 50257
    jet_hidden_dim: int = 256
    
    # Integration parameters
    coherence_weight: float = 1.0
    geometric_weight: float = 1e-3
    jet_weight: float = 1e-4

class EnhancedToroidalDiffusionModel(nn.Module):
    """Enhanced toroidal diffusion model with DEF architecture integration."""
    
    def __init__(self, 
                 base_model: Optional[nn.Module] = None,
                 scheduler: Optional[Any] = None,
                 config: Optional[ToroidalConfig] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        
        self.config = config or ToroidalConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert config to geometry and hyperparameter dictionaries
        self.geom = {
            'N_theta': self.config.N_theta,
            'N_phi': self.config.N_phi,
            'R': self.config.R,
            'r_base': self.config.r_base,
            'alpha': self.config.alpha,
            'h': self.config.h,
            'phi_c': self.config.phi_c
        }
        
        self.hyper = {
            'D': self.config.D,
            'dt': self.config.dt,
            'steps': self.config.steps,
            'tau_fixed': self.config.tau_fixed,
            'tau_stop': self.config.tau_stop
        }
        
        # Initialize DEF core components
        self.toroidal_core = ToroidalCore(self.geom, self.device)
        self.jet_head = JetHead(
            throat_size=2,  # Two sheets
            vocab_size=self.config.jet_vocab_size,
            hidden_dim=self.config.jet_hidden_dim
        )
        
        # Base model integration (optional)
        self.base_model = base_model
        self.scheduler = scheduler
        
        # Integration layers
        if base_model is not None:
            self._setup_integration_layers()
        
        # Move to device
        self.to(self.device)
        
        # Training history
        self.training_history: List[Dict] = []
        self.generation_history: List[Dict] = []
    
    def _setup_integration_layers(self):
        """Setup layers for integrating with base diffusion models."""
        # Projection layers for base model integration
        base_dim = getattr(self.base_model, 'config', {}).get('sample_size', 64) ** 2 * 4
        torus_dim = 2 * self.config.N_theta * self.config.N_phi
        
        self.base_to_torus = nn.Sequential(
            nn.Linear(base_dim, torus_dim // 2),
            nn.GELU(),
            nn.Linear(torus_dim // 2, torus_dim)
        )
        
        self.torus_to_base = nn.Sequential(
            nn.Linear(torus_dim, torus_dim // 2),
            nn.GELU(),
            nn.Linear(torus_dim // 2, base_dim)
        )
        
        # Attention mechanism for cross-modal integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=min(512, torus_dim // 4),
            num_heads=8,
            batch_first=True
        )
        
        self.norm_layer = nn.LayerNorm(min(512, torus_dim // 4))
    
    def forward(self, 
                x: Optional[torch.Tensor] = None,
                timestep: Optional[torch.Tensor] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through enhanced toroidal diffusion model."""
        
        # If base model input is provided, integrate it
        if x is not None and self.base_model is not None:
            return self._integrated_forward(x, timestep, return_dict, **kwargs)
        else:
            return self._pure_toroidal_forward(return_dict, **kwargs)
    
    def _pure_toroidal_forward(self, return_dict: bool = True, **kwargs) -> Union[torch.Tensor, Dict]:
        """Pure toroidal diffusion forward pass."""
        # Run toroidal diffusion
        deltas, final_state, metadata = self.toroidal_core(
            steps=self.hyper['steps'],
            D=self.hyper['D'],
            dt=self.hyper['dt'],
            return_history=True
        )
        
        # Generate jet output
        throat_state = self.toroidal_core.get_throat_state()
        jet_logits = self.jet_head(throat_state)
        
        # Compute losses
        coherence_loss = deltas[-1] + deltas.mean() * 1e-2
        geometric_analysis = self.toroidal_core.get_geometric_analysis()
        geometric_loss = torch.tensor(geometric_analysis['total_energy'], device=self.device)
        jet_loss = jet_logits.pow(2).mean()
        
        total_loss = (self.config.coherence_weight * coherence_loss + 
                     self.config.geometric_weight * geometric_loss + 
                     self.config.jet_weight * jet_loss)
        
        if return_dict:
            return {
                'loss': total_loss,
                'coherence_loss': coherence_loss,
                'geometric_loss': geometric_loss,
                'jet_loss': jet_loss,
                'final_state': final_state,
                'deltas': deltas,
                'jet_logits': jet_logits,
                'throat_state': throat_state,
                'geometric_analysis': geometric_analysis,
                'metadata': metadata
            }
        else:
            return total_loss
    
    def _integrated_forward(self, 
                           x: torch.Tensor, 
                           timestep: Optional[torch.Tensor] = None,
                           return_dict: bool = True,
                           **kwargs) -> Union[torch.Tensor, Dict]:
        """Integrated forward pass with base model."""
        batch_size = x.shape[0]
        
        # Project base model input to toroidal space
        x_flat = x.flatten(start_dim=1)
        torus_input = self.base_to_torus(x_flat)
        
        # Reshape to toroidal format
        torus_input = torus_input.view(batch_size, 2, self.config.N_theta, self.config.N_phi)
        
        # Update toroidal core state
        self.toroidal_core.u.data = torus_input.mean(dim=0)  # Average over batch
        
        # Run toroidal diffusion
        deltas, final_state, metadata = self.toroidal_core(
            steps=self.hyper['steps'] // 4,  # Reduced steps for integration
            D=self.hyper['D'],
            dt=self.hyper['dt'],
            return_history=True
        )
        
        # Project back to base model space
        torus_flat = final_state.flatten().unsqueeze(0).repeat(batch_size, 1)
        base_output = self.torus_to_base(torus_flat)
        base_output = base_output.view_as(x)
        
        # Apply base model if available
        if self.base_model is not None and timestep is not None:
            try:
                base_result = self.base_model(base_output, timestep, **kwargs)
                if hasattr(base_result, 'sample'):
                    base_output = base_result.sample
                elif isinstance(base_result, torch.Tensor):
                    base_output = base_result
            except Exception as e:
                warnings.warn(f"Base model forward failed: {e}")
        
        # Generate jet output
        throat_state = self.toroidal_core.get_throat_state()
        jet_logits = self.jet_head(throat_state)
        
        # Compute integrated loss
        coherence_loss = deltas[-1] + deltas.mean() * 1e-2
        geometric_analysis = self.toroidal_core.get_geometric_analysis()
        geometric_loss = torch.tensor(geometric_analysis['total_energy'], device=self.device)
        jet_loss = jet_logits.pow(2).mean()
        
        total_loss = (self.config.coherence_weight * coherence_loss + 
                     self.config.geometric_weight * geometric_loss + 
                     self.config.jet_weight * jet_loss)
        
        if return_dict:
            return {
                'sample': base_output,
                'loss': total_loss,
                'coherence_loss': coherence_loss,
                'geometric_loss': geometric_loss,
                'jet_loss': jet_loss,
                'final_state': final_state,
                'deltas': deltas,
                'jet_logits': jet_logits,
                'throat_state': throat_state,
                'geometric_analysis': geometric_analysis,
                'metadata': metadata
            }
        else:
            return base_output
    
    def sample(self, 
               batch_size: int = 1,
               num_inference_steps: int = 50,
               guidance_scale: float = 7.5,
               return_history: bool = False,
               **kwargs) -> Dict[str, Any]:
        """Generate samples using enhanced toroidal diffusion."""
        
        self.eval()
        with torch.no_grad():
            # Initialize random noise if using base model
            if self.base_model is not None:
                # Get sample size from base model config
                sample_size = getattr(self.base_model, 'config', {}).get('sample_size', 64)
                in_channels = getattr(self.base_model, 'config', {}).get('in_channels', 4)
                
                latents = torch.randn(
                    batch_size, in_channels, sample_size, sample_size,
                    device=self.device, dtype=torch.float32
                )
                
                # Use scheduler if available
                if self.scheduler is not None:
                    self.scheduler.set_timesteps(num_inference_steps)
                    timesteps = self.scheduler.timesteps
                else:
                    timesteps = torch.linspace(1000, 0, num_inference_steps, device=self.device)
                
                # Denoising loop with toroidal enhancement
                history = []
                for i, t in enumerate(timesteps):
                    timestep = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    
                    # Enhanced forward pass
                    result = self._integrated_forward(latents, timestep, return_dict=True)
                    
                    if self.scheduler is not None:
                        latents = self.scheduler.step(result['sample'], t, latents).prev_sample
                    else:
                        latents = result['sample']
                    
                    if return_history:
                        history.append({
                            'step': i,
                            'timestep': t.item(),
                            'coherence_delta': result['deltas'][-1].item(),
                            'geometric_analysis': result['geometric_analysis'],
                            'throat_activity': result['throat_state'].abs().mean().item()
                        })
                
                return {
                    'samples': latents,
                    'history': history if return_history else None,
                    'final_throat_state': self.toroidal_core.get_throat_state(),
                    'final_geometric_analysis': self.toroidal_core.get_geometric_analysis()
                }
            
            else:
                # Pure toroidal sampling
                result = self._pure_toroidal_forward(return_dict=True)
                
                # Generate multiple samples by running diffusion multiple times
                samples = []
                histories = []
                
                for _ in range(batch_size):
                    sample_result = self._pure_toroidal_forward(return_dict=True)
                    samples.append(sample_result['final_state'])
                    
                    if return_history:
                        histories.append({
                            'deltas': sample_result['deltas'],
                            'geometric_analysis': sample_result['geometric_analysis'],
                            'metadata': sample_result['metadata']
                        })
                
                return {
                    'samples': torch.stack(samples) if samples else result['final_state'].unsqueeze(0),
                    'history': histories if return_history else None,
                    'jet_tokens': torch.argmax(result['jet_logits'], dim=-1),
                    'final_throat_state': result['throat_state'],
                    'final_geometric_analysis': result['geometric_analysis']
                }
    
    def get_coherence_metrics(self) -> Dict[str, float]:
        """Get current coherence and geometric metrics."""
        with torch.no_grad():
            geometric_analysis = self.toroidal_core.get_geometric_analysis()
            throat_state = self.toroidal_core.get_throat_state()
            
            return {
                **geometric_analysis,
                'throat_magnitude': throat_state.abs().mean().item(),
                'throat_variance': throat_state.var().item(),
                'sheet_correlation': F.cosine_similarity(
                    self.toroidal_core.u[0].flatten(),
                    self.toroidal_core.u[1].flatten(),
                    dim=0
                ).item()
            }
    
    def visualize_toroidal_state(self) -> Dict[str, torch.Tensor]:
        """Get visualization data for toroidal state."""
        with torch.no_grad():
            return {
                'upper_sheet': self.toroidal_core.u[0].cpu(),
                'lower_sheet': self.toroidal_core.u[1].cpu(),
                'throat_mask': self.toroidal_core.mask.cpu(),
                'gaussian_curvature': self.toroidal_core.gaussian_curvature.cpu(),
                'mean_curvature': self.toroidal_core.mean_curvature.cpu(),
                'surface_element': self.toroidal_core.surface_element.cpu()
            }
    
    def reset_state(self):
        """Reset toroidal core to initial state."""
        self.toroidal_core.u.data = 0.1 * torch.randn_like(self.toroidal_core.u.data)
        self.training_history.clear()
        self.generation_history.clear()

# Utility functions for integration
def create_enhanced_model(base_model=None, scheduler=None, config=None, device=None):
    """Factory function to create enhanced toroidal diffusion model."""
    return EnhancedToroidalDiffusionModel(
        base_model=base_model,
        scheduler=scheduler,
        config=config,
        device=device
    )

def load_pretrained_base_model(model_name: str = "runwayml/stable-diffusion-v1-5"):
    """Load a pretrained base model for integration."""
    try:
        from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
        
        # Load UNet and scheduler
        unet = UNet2DConditionModel.from_pretrained(
            model_name, subfolder="unet", torch_dtype=torch.float32
        )
        scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        
        return unet, scheduler
        
    except ImportError:
        warnings.warn("diffusers not available, using mock base model")
        return None, None
    except Exception as e:
        warnings.warn(f"Failed to load pretrained model: {e}")
        return None, None

# Demo function
def demo_enhanced_wrapper():
    """Demonstrate the enhanced toroidal wrapper."""
    print("=== Enhanced Toroidal Diffusion Wrapper Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    config = ToroidalConfig(
        N_theta=32,  # Smaller for demo
        N_phi=64,
        steps=50,
        enable_sbert=True
    )
    
    # Create model
    model = create_enhanced_model(config=config, device=device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test pure toroidal mode
    print("\n--- Pure Toroidal Mode ---")
    result = model(return_dict=True)
    print(f"Coherence loss: {result['coherence_loss'].item():.6f}")
    print(f"Geometric analysis: {result['geometric_analysis']}")
    
    # Test sampling
    print("\n--- Sampling ---")
    samples = model.sample(batch_size=2, return_history=True)
    print(f"Generated {len(samples['samples'])} samples")
    if samples['history']:
        print(f"History length: {len(samples['history'])}")
    
    # Test metrics
    print("\n--- Coherence Metrics ---")
    metrics = model.get_coherence_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n✅ Enhanced wrapper demo completed successfully!")
    return model

if __name__ == '__main__':
    demo_enhanced_wrapper()

