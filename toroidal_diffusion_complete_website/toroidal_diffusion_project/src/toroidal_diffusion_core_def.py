# -*- coding: utf-8 -*-
"""
Toroidal Diffusion Core — full DEF prototype (geometry + learnable + SBERT + Jet)
===========================================================================
• D – autograd-enabled diffusion on double-torus lattice (CPU / CUDA)
• E – real semantic embeddings via Sentence-Transformer (optional fall-back)
• F – single-jet decoder head for text tokens

Author: Stepan Solncev (ΔΣ-Foundation) + Enhanced Implementation 2025-07-14
License: MIT (prototype)

Instructions
------------
1.  `pip install torch sentence-transformers` (SBERT is optional; see `USE_SBERT`).
2.  Adjust GEOM and HYPER dictionaries below.
3.  Run: `python toroidal_diffusion_core_def.py` — prints Δ-curve & sample jet.
4.  CUDA is auto-detected. To force CPU: `CUDA_VISIBLE_DEVICES='' python ...`.

This file is intentionally single-module for quick experimentation; split into
packages once design stabilises.
"""

from typing import Dict, Tuple, Optional, List
import math, random, time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
GEOM: Dict[str, float] = dict(
    N_theta=64,          # lat grid
    N_phi=128,           # long grid
    R=1.0,               # major radius
    r_base=0.4,          # tube radius at equator
    alpha=0.48,          # tube squashing along θ
    h=0.22,              # neck thickness (for double torus)
    phi_c=0.18           # throat angular half-width (rad)
)

HYPER: Dict[str, float] = dict(
    D=0.05,              # diffusion coefficient
    dt=0.15,             # Euler step
    steps=160,           # steps per forward
    tau_fixed=5e-3,      # trigger snapshot
    tau_stop=1e-4,       # early stop
)

USE_SBERT = True         # set False to keep random projection
SBERT_MODEL = 'all-MiniLM-L6-v2'

# ------------------------------------------------------------
# Helper — geometry-dependent coefficients
# ------------------------------------------------------------

def build_coeff_tensors(N_theta: int, N_phi: int, R: float, r_base: float,
                         alpha: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return coeff_theta, coeff_phi (buffers for discrete Laplacian)."""
    theta = torch.linspace(0, 2*math.pi, N_theta, device=device, dtype=torch.float32, requires_grad=False)
    phi   = torch.linspace(0, 2*math.pi, N_phi,   device=device, dtype=torch.float32, requires_grad=False)
    Θ, Φ  = torch.meshgrid(theta, phi, indexing='ij')
    r_tube = r_base * (1.0 - alpha*torch.cos(Θ)**2)
    coeff_theta = 1.0 / (r_tube**2)
    coeff_phi   = 1.0 / (R + r_tube*torch.cos(Θ))**2
    return coeff_theta, coeff_phi

def compute_geometric_properties(N_theta: int, N_phi: int, R: float, r_base: float,
                                alpha: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """Compute additional geometric properties for analysis."""
    theta = torch.linspace(0, 2*math.pi, N_theta, device=device, dtype=torch.float32)
    phi   = torch.linspace(0, 2*math.pi, N_phi,   device=device, dtype=torch.float32)
    Θ, Φ  = torch.meshgrid(theta, phi, indexing='ij')
    
    r_tube = r_base * (1.0 - alpha*torch.cos(Θ)**2)
    
    # Gaussian curvature approximation
    K = 1.0 / (r_tube * (R + r_tube*torch.cos(Θ)))
    
    # Mean curvature
    H = (R + 2*r_tube*torch.cos(Θ)) / (2*r_tube*(R + r_tube*torch.cos(Θ)))
    
    # Surface area element
    dS = r_tube * (R + r_tube*torch.cos(Θ))
    
    return {
        'gaussian_curvature': K,
        'mean_curvature': H,
        'surface_element': dS,
        'tube_radius': r_tube
    }

# ------------------------------------------------------------
# Core module
# ------------------------------------------------------------

class ToroidalCore(nn.Module):
    """Double-sheet toroidal diffusion with learnable state & semantic Δ."""

    def __init__(self, geom: Dict[str, float], device: torch.device):
        super().__init__()
        self.geom = geom
        self.device = device

        Nt, Np = geom['N_theta'], geom['N_phi']
        self.Nt, self.Np = Nt, Np
        
        # learnable 2×Nt×Np field (upper & lower sheet)
        self.u = nn.Parameter(0.1*torch.randn(2, Nt, Np, device=device))

        # geometry buffers
        coeff_theta, coeff_phi = build_coeff_tensors(Nt, Np, geom['R'], geom['r_base'], geom['alpha'], device)
        self.register_buffer('coeff_theta', coeff_theta)
        self.register_buffer('coeff_phi', coeff_phi)
        self.register_buffer('mask', self._build_mask())
        
        # geometric properties for analysis
        geom_props = compute_geometric_properties(Nt, Np, geom['R'], geom['r_base'], geom['alpha'], device)
        for name, tensor in geom_props.items():
            self.register_buffer(name, tensor)

        # random projection for synthetic embedding (overridden if SBERT)
        proj_len = 2*Nt*Np
        self.register_buffer('rand_proj', torch.randn(proj_len, device=device))
        
        # history tracking
        self.delta_history: List[float] = []
        self.embedding_history: List[torch.Tensor] = []

    # ---------------------------- private ----------------------------
    def _build_mask(self) -> torch.Tensor:
        """Build throat mask for synchronization between sheets."""
        phi_c = self.geom['phi_c']
        θ = torch.linspace(0, 2*math.pi, self.Nt,  device=self.device, dtype=torch.float32)
        φ = torch.linspace(0, 2*math.pi, self.Np,  device=self.device, dtype=torch.float32)
        Θ, Φ = torch.meshgrid(θ, φ, indexing='ij')
        return ((torch.abs(((Φ-math.pi+math.pi)%(2*math.pi))-math.pi) < phi_c)).bool()

    def _laplace(self, f: torch.Tensor) -> torch.Tensor:
        """Discrete Laplacian on torus with geometric coefficients."""
        return self.coeff_theta*(torch.roll(f,-1,0)+torch.roll(f,1,0)-2*f) + \
               self.coeff_phi  *(torch.roll(f,-1,1)+torch.roll(f,1,1)-2*f)

    def _embed(self, state: torch.Tensor) -> torch.Tensor:
        """Semantic embedding of 2-sheet field → 384-dim (SBERT) or scalar (rand)."""
        if USE_SBERT and SBERT_LOADER is not None:
            txt = encode_state_to_text(state.detach())
            with torch.no_grad():
                return SBERT_LOADER.encode(txt, convert_to_tensor=True).to(self.device)
        else:
            flat = state.flatten()
            return torch.tanh((flat @ self.rand_proj) / (flat.norm() + 1e-8))  # scalar with numerical stability

    def _throat_sync(self, u: torch.Tensor) -> torch.Tensor:
        """Synchronize upper and lower sheets through throat region."""
        mask = self.mask
        avg = 0.5*(u[0][mask] + u[1][mask])          # throat sync
        u_sync = u.clone()
        u_sync[0][mask] = avg
        u_sync[1][mask] = avg
        return u_sync

    def _compute_flow_statistics(self, u: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute flow and geometric statistics."""
        # Gradient magnitude (flow strength)
        grad_theta = torch.roll(u, -1, dims=1) - torch.roll(u, 1, dims=1)
        grad_phi = torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)
        flow_magnitude = torch.sqrt(grad_theta**2 + grad_phi**2)
        
        # Sheet coupling strength
        coupling_strength = torch.abs(u[0] - u[1]).mean()
        
        # Throat activity
        throat_activity = u[:, self.mask].abs().mean()
        
        return {
            'flow_magnitude': flow_magnitude.mean(),
            'coupling_strength': coupling_strength,
            'throat_activity': throat_activity,
            'total_energy': (u**2).mean()
        }

    # ---------------------------- public ----------------------------
    def forward(self, steps: int, D: float, dt: float, 
                return_history: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Return sequence of Δ and final state with optional detailed history."""
        deltas = []
        flow_stats = []
        prev_emb = self._embed(self.u.detach())
        u = self.u
        
        for step in range(steps):
            # Throat synchronization
            u = self._throat_sync(u)
            
            # Diffusion step
            laplace_u = torch.stack([self._laplace(u[0]), self._laplace(u[1])])
            u = u + dt * D * laplace_u
            
            # Compute semantic change
            new_emb = self._embed(u)
            if new_emb.dim() > 0:  # SBERT case
                delta = 1.0 - F.cosine_similarity(new_emb, prev_emb, dim=0).mean()
            else:  # scalar case
                delta = 1.0 - F.cosine_similarity(new_emb.unsqueeze(0), prev_emb.unsqueeze(0), dim=0)
            
            deltas.append(delta.unsqueeze(0))
            prev_emb = new_emb
            
            # Track flow statistics if requested
            if return_history:
                stats = self._compute_flow_statistics(u)
                flow_stats.append(stats)
        
        # Update parameter for optimizer
        self.u = nn.Parameter(u)
        
        # Prepare return data
        delta_tensor = torch.cat(deltas)
        metadata = None
        if return_history:
            metadata = {
                'flow_statistics': flow_stats,
                'final_embedding': prev_emb,
                'throat_mask': self.mask,
                'geometric_properties': {
                    'gaussian_curvature': self.gaussian_curvature,
                    'mean_curvature': self.mean_curvature,
                    'surface_element': self.surface_element
                }
            }
        
        return delta_tensor, u, metadata

    def get_throat_state(self) -> torch.Tensor:
        """Extract current state at throat region."""
        return self.u[:, self.mask].mean(dim=1)
    
    def get_geometric_analysis(self) -> Dict[str, float]:
        """Get geometric analysis of current state."""
        with torch.no_grad():
            stats = self._compute_flow_statistics(self.u)
            return {
                'flow_magnitude': stats['flow_magnitude'].item(),
                'coupling_strength': stats['coupling_strength'].item(),
                'throat_activity': stats['throat_activity'].item(),
                'total_energy': stats['total_energy'].item(),
                'mean_gaussian_curvature': self.gaussian_curvature.mean().item(),
                'mean_surface_curvature': self.mean_curvature.mean().item()
            }

# ------------------------------------------------------------
# Jet-decoder F
# ------------------------------------------------------------

class JetHead(nn.Module):
    """Enhanced jet decoder with attention mechanism."""
    
    def __init__(self, throat_size: int, vocab_size: int = 50257, hidden_dim: int = 256):
        super().__init__()
        self.throat_size = throat_size
        self.hidden_dim = hidden_dim
        
        # Multi-layer projection with residual connections
        self.input_proj = nn.Linear(throat_size, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # Attention mechanism for throat state
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, throat_vec: torch.Tensor) -> torch.Tensor:
        """Forward pass through jet decoder."""
        # Ensure proper shape
        if throat_vec.dim() == 1:
            throat_vec = throat_vec.unsqueeze(0)
        
        # Input projection
        x = self.input_proj(throat_vec)
        
        # Hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = x
            x = layer(x) + residual
        
        # Self-attention (treating each element as a sequence element)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        
        # Output projection
        x = x.squeeze(1) if x.size(1) == 1 else x.mean(dim=1)
        logits = self.output_proj(x)
        
        return logits

# ------------------------------------------------------------
# Utility — encode state to short string (enhanced)
# ------------------------------------------------------------

def encode_state_to_text(state: torch.Tensor, topk: int = 4) -> str:
    """Enhanced state encoding with more semantic information."""
    out = []
    for i, sheet in enumerate(state):
        flat = sheet.flatten()
        vals, idx = torch.topk(flat.abs(), topk)
        
        # Convert to 2D coordinates
        coords_2d = [(idx_val.item() // sheet.size(1), idx_val.item() % sheet.size(1)) 
                     for idx_val in idx]
        
        # Create more semantic description
        descriptors = []
        for (theta_idx, phi_idx), val in zip(coords_2d, vals):
            theta_norm = theta_idx / sheet.size(0)
            phi_norm = phi_idx / sheet.size(1)
            intensity = flat[idx[len(descriptors)]].item()
            
            # Semantic regions
            if theta_norm < 0.25:
                region = "north"
            elif theta_norm < 0.75:
                region = "equator"
            else:
                region = "south"
                
            descriptors.append(f"{region}_{phi_norm:.2f}:{intensity:+.3f}")
        
        sheet_desc = f"sheet_{i}[{','.join(descriptors)}]"
        out.append(sheet_desc)
    
    return ' | '.join(out)

# ------------------------------------------------------------
# SBERT Integration
# ------------------------------------------------------------

SBERT_LOADER = None
if USE_SBERT:
    try:
        from sentence_transformers import SentenceTransformer
        SBERT_LOADER = SentenceTransformer(SBERT_MODEL)
        print(f"[SBERT] Loaded model: {SBERT_MODEL}")
    except Exception as e:
        print(f'[SBERT] fallback to random projection → {e}')
        USE_SBERT = False

# ------------------------------------------------------------
# Enhanced Demo / Train loop
# ------------------------------------------------------------

def enhanced_demo():
    """Enhanced demonstration with detailed analysis."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    core = ToroidalCore(GEOM, device).to(device)
    jet = JetHead(throat_size=2, vocab_size=1000, hidden_dim=128).to(device)
    
    # Optimizer
    opt = torch.optim.Adam(list(core.parameters()) + list(jet.parameters()), lr=3e-4)
    
    print("\n=== Training Enhanced DEF Architecture ===")
    print(f"Geometry: {GEOM}")
    print(f"Hyperparameters: {HYPER}")
    print(f"SBERT enabled: {USE_SBERT}")
    
    training_history = []
    
    for epoch in range(30):
        opt.zero_grad()
        
        # Forward pass with detailed tracking
        deltas, final_state, metadata = core(
            steps=HYPER['steps'], 
            D=HYPER['D'], 
            dt=HYPER['dt'],
            return_history=True
        )

        # Jet processing
        throat_state = core.get_throat_state()
        logits = jet(throat_state)
        jet_loss = logits.pow(2).mean() * 1e-4  # regularization

        # Enhanced loss function
        delta_loss = deltas[-1] + deltas.mean() * 1e-2  # coherence objective
        
        # Geometric regularization
        geom_analysis = core.get_geometric_analysis()
        geom_loss = torch.tensor(geom_analysis['total_energy'], device=device) * 1e-3
        
        total_loss = delta_loss + jet_loss + geom_loss
        total_loss.backward()
        opt.step()

        # Track training progress
        training_history.append({
            'epoch': epoch,
            'delta_final': deltas[-1].item(),
            'delta_mean': deltas.mean().item(),
            'jet_loss': jet_loss.item(),
            'geom_loss': geom_loss.item(),
            'total_loss': total_loss.item(),
            'geometric_analysis': geom_analysis
        })

        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} | Δ_last={deltas[-1].item():.4e} | "
                  f"loss={total_loss.item():.4e} | "
                  f"throat_activity={geom_analysis['throat_activity']:.4f}")

    # Final analysis
    print("\n=== Final Analysis ===")
    with torch.no_grad():
        # Generate sample
        final_deltas, final_state, final_metadata = core(
            steps=50, D=HYPER['D'], dt=HYPER['dt'], return_history=True
        )
        
        # Jet output
        throat_state = core.get_throat_state()
        final_logits = jet(throat_state)
        token_id = torch.argmax(final_logits).item()
        
        # Geometric analysis
        final_geom = core.get_geometric_analysis()
        
        print(f"Sample jet-token id: {token_id}")
        print(f"Final coherence delta: {final_deltas[-1].item():.6f}")
        print(f"Geometric properties:")
        for key, value in final_geom.items():
            print(f"  {key}: {value:.6f}")
        
        # State encoding
        if USE_SBERT:
            state_text = encode_state_to_text(final_state)
            print(f"State encoding: {state_text}")

    return training_history, core, jet

def main():
    """Main execution function."""
    try:
        history, core, jet = enhanced_demo()
        print("\n✅ Enhanced DEF architecture demonstration completed successfully!")
        return history, core, jet
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    main()

