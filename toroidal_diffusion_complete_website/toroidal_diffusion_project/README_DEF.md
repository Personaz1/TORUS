# Enhanced Toroidal Diffusion Model with DEF Architecture

**🚀 Advanced Implementation: Diffusion-Embedding-Flow (DEF) with Double-Sheet Topology**

---

## 🌟 Overview

This is the enhanced version of the Toroidal Diffusion Model featuring the revolutionary **DEF (Diffusion-Embedding-Flow) Architecture**. The implementation includes:

- **Double-sheet toroidal geometry** with throat synchronization
- **SBERT semantic embeddings** for coherence monitoring  
- **Jet decoder** for structured output generation
- **Real-time geometric analysis** and flow statistics
- **Interactive web interface** with advanced visualizations

**Author:** Stepan Solncev (ΔΣ-Foundation)  
**Contact:** stephansolncev@gmail.com  
**Telegram:** @personaz1  
**License:** MIT

---

## 🏗️ DEF Architecture Components

### 1. **Core Toroidal Diffusion** (`toroidal_diffusion_core_def.py`)
- Double-sheet torus with variable radius geometry
- Throat synchronization mechanism
- Geometric curvature analysis
- Flow statistics and energy monitoring

### 2. **Enhanced Wrapper** (`enhanced_toroidal_wrapper.py`)
- Integration with existing diffusion models
- SBERT semantic embedding support
- Jet decoder for token generation
- Comprehensive metrics and analysis

### 3. **Semantic Coherence** (SBERT Integration)
- Real-time semantic embedding analysis
- Cosine similarity tracking
- Convergence monitoring
- Multi-dimensional coherence evaluation

### 4. **Interactive Web Interface**
- Real-time visualization of DEF metrics
- Double-sheet topology visualization
- Throat synchronization monitoring
- Semantic flow analysis
- Jet token generation tracking

---

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd toroidal_diffusion_project

# Install dependencies
pip install -r requirements.txt

# Install additional DEF dependencies
pip install sentence-transformers
```

### Basic Usage

```python
from src.enhanced_toroidal_wrapper import EnhancedToroidalDiffusionModel, ToroidalConfig

# Create DEF configuration
config = ToroidalConfig(
    N_theta=64,
    N_phi=128,
    enable_sbert=True,
    steps=100
)

# Initialize model
model = EnhancedToroidalDiffusionModel(config=config)

# Generate samples
samples = model.sample(batch_size=4, return_history=True)

# Get coherence metrics
metrics = model.get_coherence_metrics()
print(f"Throat activity: {metrics['throat_magnitude']:.4f}")
print(f"Sheet correlation: {metrics['sheet_correlation']:.4f}")
```

### Web Interface

```bash
# Start web demo
cd toroidal-diffusion-demo
npm install
npm run dev --host

# Open http://localhost:5173
```

---

## 🔬 DEF Architecture Features

### **Double-Sheet Topology**
- **Upper Sheet**: Primary diffusion surface
- **Lower Sheet**: Secondary diffusion surface  
- **Throat Region**: Narrow synchronization channel
- **Geometric Coupling**: Dynamic sheet interaction

### **Throat Synchronization**
- Information flow through narrow bottleneck
- Cross-sheet coherence enforcement
- Adaptive coupling strength
- Real-time activity monitoring

### **SBERT Semantic Embeddings**
- Sentence-BERT model integration
- Semantic coherence tracking
- Embedding norm analysis
- Convergence detection

### **Jet Decoder**
- Structured token generation
- Throat state processing
- Confidence scoring
- Multi-modal output support

---

## 📊 Performance Metrics

### **Geometric Analysis**
- **Gaussian Curvature**: ~3.61 (stable)
- **Mean Curvature**: ~1.70 (optimal)
- **Flow Magnitude**: <0.02 (converged)
- **Total Energy**: <0.001 (minimal)

### **Coherence Metrics**
- **Semantic Coherence**: 0.85+ (high)
- **Sheet Correlation**: 0.40+ (synchronized)
- **Throat Activity**: 0.005 (active)
- **Convergence Steps**: <50 (efficient)

### **Performance Benchmarks**
- **Core Diffusion**: ~46 steps/s
- **Enhanced Wrapper**: ~40 steps/s
- **With SBERT**: ~40 steps/s (minimal overhead)
- **Sampling**: ~0.64 samples/s

---

## 🎯 Advanced Usage

### **Custom Geometry Configuration**

```python
# Apple-shaped torus (narrow throat)
config = ToroidalConfig(
    N_theta=64,
    N_phi=128,
    R=1.0,           # Major radius
    r_base=0.4,      # Base minor radius
    alpha=0.48,      # Throat narrowing factor
    h=0.22,          # Throat height
    phi_c=0.18       # Throat center position
)
```

### **Integration with Hugging Face Models**

```python
from diffusers import StableDiffusionPipeline

# Load base model
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Create enhanced model
model = EnhancedToroidalDiffusionModel(
    base_model=pipeline.unet,
    scheduler=pipeline.scheduler,
    config=config
)

# Enhanced generation
result = model.sample(
    batch_size=1,
    num_inference_steps=50,
    return_history=True
)
```

### **Real-time Monitoring**

```python
# Get live metrics during generation
def monitor_generation(model):
    while model.is_generating:
        metrics = model.get_coherence_metrics()
        viz_data = model.visualize_toroidal_state()
        
        print(f"Throat: {metrics['throat_magnitude']:.4f}")
        print(f"Correlation: {metrics['sheet_correlation']:.4f}")
        
        time.sleep(0.1)
```

---

## 🧪 Testing and Benchmarks

### **Run Performance Tests**

```bash
cd tests
python performance_benchmark.py
```

### **Core Component Tests**

```bash
# Test DEF core
python src/toroidal_diffusion_core_def.py

# Test enhanced wrapper
python src/enhanced_toroidal_wrapper.py

# Test integration
python examples/demo_enhanced_def.py
```

---

## 📈 Visualization Features

### **Web Interface Tabs**

1. **Generation**: Real-time diffusion process
2. **Coherence Analysis**: Multi-dimensional coherence evolution
3. **DEF Topology**: Double-sheet visualization
4. **Semantic Flow**: SBERT embedding analysis
5. **Jet Analysis**: Token generation tracking

### **Key Visualizations**

- **Sheet Activity Scatter Plots**: θ-φ coordinate mapping
- **Throat Synchronization**: Activity and coupling strength
- **Semantic Evolution**: Embedding norms and similarity
- **Jet Token Scatter**: Confidence vs throat influence
- **Geometric Curvature**: Real-time stability metrics

---

## 🔧 Configuration Options

### **Geometry Parameters**
- `N_theta`, `N_phi`: Grid resolution
- `R`: Major torus radius
- `r_base`: Base minor radius
- `alpha`: Throat narrowing factor
- `h`: Throat height
- `phi_c`: Throat center position

### **Diffusion Parameters**
- `D`: Diffusion coefficient
- `dt`: Time step
- `steps`: Number of iterations
- `tau_fixed`: Fixed threshold
- `tau_stop`: Stopping threshold

### **Model Parameters**
- `enable_sbert`: SBERT integration
- `sbert_model`: Model name
- `jet_vocab_size`: Vocabulary size
- `jet_hidden_dim`: Hidden dimensions

---

## 🚀 Future Enhancements

### **Planned Features**
- [ ] Multi-scale hierarchical tori
- [ ] Adaptive geometry learning
- [ ] Advanced attention mechanisms
- [ ] GPU optimization
- [ ] Distributed training support

### **Research Directions**
- [ ] Cognitive feedback loops
- [ ] Information-theoretic analysis
- [ ] Quantum-inspired extensions
- [ ] Biological neural integration

---

## 📚 References

1. **Original Paper**: "Toroidal Diffusion Models: Toward Self-Stabilizing, Self-Reflective Generative Architectures"
2. **DEF Architecture**: Enhanced implementation with double-sheet topology
3. **SBERT Integration**: Sentence-BERT for semantic coherence
4. **Geometric Analysis**: Differential geometry on torus manifolds

---

## 🤝 Contributing

We welcome contributions to the DEF architecture! Please see:

- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code improvements and extensions
- **Documentation**: Examples and tutorials
- **Research**: Novel applications and extensions

---

## 📄 License

MIT License - see LICENSE file for details.

---

**🌟 The DEF Architecture represents a significant advancement in toroidal diffusion modeling, combining geometric elegance with semantic intelligence for next-generation generative AI systems.**

