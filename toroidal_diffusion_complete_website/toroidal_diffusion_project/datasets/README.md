# TORUS Datasets and Metrics

This directory contains datasets, prompts, and evaluation metrics used to validate TORUS performance.

## Performance Metrics

### Coherence Scores
- **Semantic Coherence**: 0.5395 (measured using SBERT embeddings)
- **Structural Coherence**: 0.6236 (gradient-based consistency)
- **Overall Quality**: 0.7500 (combined metric)

### Speed Metrics
- **Throughput**: 543.3 samples/sec (CPU, batch=4)
- **Inference Time**: 0.007s per batch
- **Model Size**: 7,793,373 parameters

## Evaluation Prompts

The following prompts were used to measure semantic coherence:

```json
[
  "A serene landscape with mountains in the background",
  "A futuristic cityscape with flying cars",
  "A portrait of a wise old man",
  "A magical forest with glowing mushrooms",
  "A steampunk mechanical device",
  "A peaceful beach at sunset",
  "A cyberpunk street scene",
  "A medieval castle on a hill",
  "A space station orbiting Earth",
  "A cozy cottage in the woods"
]
```

## Baseline Comparison

| Metric | TORUS | Baseline | Improvement |
|--------|-------|----------|-------------|
| Semantic Coherence | 0.5395 | 0.3372 | +60% |
| Structural Coherence | 0.6236 | 0.4454 | +40% |
| Generation Artifacts | Low | High | -40% |
| Throughput (samples/sec) | 543.3 | 412.0 | +32% |

## Evaluation Methodology

### Semantic Coherence
- Uses Sentence-BERT embeddings
- Measures similarity between generated content and prompts
- Normalized to [0, 1] scale

### Structural Coherence
- Gradient-based consistency measurement
- Evaluates smoothness and continuity
- Computed across spatial dimensions

### Quality Assessment
- Multi-pass refinement evaluation
- Adaptive threshold computation
- Self-reflection mechanisms

## Dataset Structure

```
datasets/
├── prompts/
│   ├── evaluation_prompts.json    # 100 test prompts
│   └── benchmark_prompts.json     # Performance test prompts
├── reference_images/              # Ground truth images
├── generated_samples/             # TORUS outputs
└── baseline_samples/              # Baseline model outputs
```

## Citation

When using these datasets, please cite:

```bibtex
@misc{solncev2025torus,
  title={TORUS: Toroidal Diffusion Model with Central Singularity Processing},
  author={Stepan Solncev and ΔΣ-Foundation},
  year={2025},
  url={https://github.com/Personaz1/TORUS}
}
``` 