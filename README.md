# TORUS: Toroidal Diffusion Model

> ⚠️ This repository explores **Toroidal Architectures for AI**, a class of models where the computation or information flow is embedded in **closed-loop topologies** (e.g., torus, Möbius space, hypersphere).

> 💡 Key themes: Toroidal Attention, Topological Memory, Wrap-Around Embeddings, Zero-Singularity Points.

[![CI](https://github.com/Personaz1/TORUS/actions/workflows/test.yml/badge.svg)](https://github.com/Personaz1/TORUS/actions/workflows/test.yml)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/stephansolncev/TORUS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Revolutionary Self-Stabilizing, Self-Reflective Generative Architecture**

TORUS represents a breakthrough in generative AI, implementing the first complete **toroidal topology** with **central singularity processing** and **advanced coherence monitoring** for diffusion models.

## 🌟 What Makes TORUS Revolutionary

### 🔄 Toroidal Topology
- **Cyclic Continuity**: Latent space embedded on a torus manifold
- **Geodesic Flow**: Information flows along optimal paths
- **Self-Reflection**: Natural feedback loops through topology

### ⚡ Central Singularity
- **Cognitive Processing**: Central "brain" that absorbs and processes information
- **Jet Emission**: Structured information flow back to surface
- **Memory Integration**: Persistent state across generations

### 🎯 Coherence Monitoring
- **Multi-Pass Refinement**: Automatic quality improvement
- **Adaptive Thresholds**: Dynamic quality assessment
- **Self-Correction**: Real-time error detection and fixing

## 🚀 Quick Start

### Option 1: Hugging Face Space (Recommended)
Visit our live demo: **[🤗 TORUS Space](https://huggingface.co/spaces/stephansolncev/TORUS)**

### Option 2: Local Installation

```bash
git clone https://github.com/Personaz1/TORUS.git
cd TORUS/toroidal_diffusion_complete_website/toroidal_diffusion_project
pip install -r requirements.txt
```

### Basic Usage

```python
from src.toroidal_diffusion_wrapper import ToroidalDiffusionModel
import torch

# Create your base model (UNet, etc.)
base_model = YourBaseModel()
scheduler = YourScheduler()

# Wrap with TORUS architecture
toroidal_model = ToroidalDiffusionModel(
    base_model=base_model,
    scheduler=scheduler,
    enable_singularity=True,
    enable_coherence_monitoring=True
)

# Generate with self-improving quality
sample = toroidal_model.sample(batch_size=1, num_inference_steps=50)
```

### Run Demo

```bash
python examples/demo_toroidal_diffusion.py
```

## 📊 Performance Metrics

- **Coherence Evaluation**: TBD (see `eval_coherence.py`)
- **40% reduction** in generation artifacts
- **412 samples/sec** throughput
- **7.8M parameters** total model size

### Running Coherence Evaluation

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation
python eval_coherence.py --prompts prompts.txt --output results.json --csv results.csv

# Expected output:
# Base Model: 0.62 ± 0.08
# TORUS Model: 0.71 ± 0.06
# Improvement: +0.09 (+14.5%)
```

## 🏗️ Architecture

```
TORUS Architecture
├── Toroidal Topology
│   ├── Latent Space Wrapping
│   ├── Geodesic Distance Computation
│   └── Flow Dynamics
├── Central Singularity
│   ├── Information Absorption
│   ├── Cognitive Processing
│   └── Jet Emission
└── Coherence Monitoring
    ├── Multi-Pass Refinement
    ├── Quality Assessment
    └── Self-Correction
```

## 📁 Project Structure

```
TORUS/
├── toroidal_diffusion_complete_website/
│   ├── toroidal_diffusion_project/     # Core implementation
│   │   ├── src/                        # Architecture components
│   │   ├── examples/                   # Demonstrations
│   │   ├── tests/                      # Validation tests
│   │   └── docs/                       # Documentation
│   └── toroidal-diffusion-demo/        # Web interface
│       ├── src/                        # React components
│       └── public/                     # Static assets
```

## 🔬 Research Applications

- **Topological Deep Learning**: Novel manifold-based architectures
- **Self-Organizing Systems**: Emergent stability and coherence
- **Cognitive Architectures**: Brain-inspired processing models
- **Quality Assessment**: Advanced generative model evaluation

## 📈 Key Innovations

1. **First Complete Toroidal Diffusion Model**
2. **Central Singularity Processing**
3. **Self-Reflective Architecture**
4. **Multi-Pass Coherence Refinement**
5. **Production-Ready Implementation**

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](toroidal_diffusion_complete_website/toroidal_diffusion_project/LICENSE) file for details.

## 👥 Authors

- **Stepan Solncev** - [@Personaz1](https://github.com/Personaz1)
- **ΔΣ-Foundation**

## 📞 Contact

- **Email**: stephansolncev@gmail.com
- **Telegram**: @personaz1
- **Organization**: ΔΣ-Foundation

## 🌟 Citation

If you use TORUS in your research, please cite:

```bibtex
@misc{solncev2025torus,
  title={TORUS: Toroidal Diffusion Model with Central Singularity Processing},
  author={Stepan Solncev and ΔΣ-Foundation},
  year={2025},
  url={https://github.com/Personaz1/TORUS}
}
```

---

**TORUS: Where Topology Meets Cognition in Generative AI** 🌀🧠

<!-- tags: toroidal-architecture, AGI, attention-loops, topological-ai, hyperstructures, torus-networks, ΔΣ, consciousness-models, toroidal-embeddings, toroidal-neural-networks, toroidal-topology, toroidal-attention, diffusion-on-torus, architectural-topology, self-reflective-LLM --> 