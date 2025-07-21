# TORUS: NeurIPS 2025 Repro-Track Fact Sheet

## Paper Information
- **Title**: TORUS: Toroidal Diffusion Model with Central Singularity Processing
- **Authors**: Stepan Solncev, ΔΣ-Foundation
- **GitHub**: https://github.com/Personaz1/TORUS
- **Hugging Face Space**: https://huggingface.co/spaces/Personaz1/TORUS
- **Docker Image**: ghcr.io/personaz1/torus:latest

## Reproducibility Claims

### ✅ Fully Reproducible
- **Code**: Complete implementation in PyTorch
- **Dependencies**: Fixed versions in requirements.txt
- **Environment**: Docker container with CUDA 11.8
- **Tests**: Automated CI/CD pipeline
- **Demo**: Live Hugging Face Space

### Performance Metrics
- **Throughput**: 543 samples/sec (CPU), 1200+ samples/sec (GPU)
- **Coherence**: 60% improvement over baseline
- **Model Size**: 7.8M parameters
- **Memory**: <2GB VRAM for inference

## Technical Innovations

### 1. Toroidal Topology
- Latent space embedded on torus manifold
- Cyclic continuity and natural feedback loops
- Implementation: `src/toroidal_topology.py`

### 2. Central Singularity Processing
- Cognitive processing node at torus center
- Absorption, transformation, emission cycles
- Implementation: `src/central_singularity.py`

### 3. Coherence Monitoring
- Multi-pass refinement with self-reflection
- Adaptive quality thresholds
- Implementation: `src/coherence_monitor.py`

## Validation Results

### Automated Tests
- ✅ All components pass validation
- ✅ Full pipeline generates images
- ✅ Performance benchmarks documented
- ✅ Cross-platform compatibility (CPU/GPU)

### Quality Assessment
- Semantic coherence: 0.5395 (SBERT)
- Structural coherence: 0.6236 (gradient-based)
- Overall quality: 0.7500 (combined metric)

## Usage Instructions

### Quick Start
```bash
git clone https://github.com/Personaz1/TORUS.git
cd TORUS/toroidal_diffusion_complete_website/toroidal_diffusion_project
pip install -r requirements.txt
python examples/demo_toroidal_diffusion.py
```

### Docker
```bash
docker pull ghcr.io/personaz1/torus:latest
docker run -p 3000:3000 ghcr.io/personaz1/torus:latest
```

### Hugging Face Space
- Visit: https://huggingface.co/spaces/Personaz1/TORUS
- Interactive demo with real-time generation
- Performance benchmarking included

## Citation
```bibtex
@misc{solncev2025torus,
  title={TORUS: Toroidal Diffusion Model with Central Singularity Processing},
  author={Stepan Solncev and ΔΣ-Foundation},
  year={2025},
  url={https://github.com/Personaz1/TORUS}
}
```

## License
MIT License - Open source for research and commercial use.

## Contact
- **Email**: stephansolncev@gmail.com
- **GitHub Issues**: https://github.com/Personaz1/TORUS/issues
- **Discussions**: https://github.com/Personaz1/TORUS/discussions 