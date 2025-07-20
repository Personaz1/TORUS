# Installation Guide

## System Requirements

- **Python**: 3.8+ (tested with 3.11)
- **PyTorch**: 2.0+ with CUDA support (optional)
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space
- **OS**: Linux, macOS, or Windows

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv toroidal_env
source toroidal_env/bin/activate  # Linux/macOS
# or
toroidal_env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd toroidal_diffusion_project

# Install Python dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation

```bash
# Test core components
cd src
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from toroidal_topology import ToroidalLatentSpace; print('✓ Toroidal topology loaded')"
python -c "from central_singularity import SingularityToroidalCoupling; print('✓ Central singularity loaded')"
python -c "from coherence_monitor import MultiPassRefinement; print('✓ Coherence monitor loaded')"
```

### 4. Run Tests

```bash
# Test individual components
python toroidal_topology.py
python central_singularity.py
python coherence_monitor.py
python advanced_coherence_system.py
python toroidal_diffusion_wrapper.py

# Run comprehensive demo
cd ../examples
python demo_toroidal_diffusion.py
```

### 5. Web Interface Setup

```bash
# Install Node.js dependencies
cd ../toroidal-diffusion-demo
npm install

# Start development server
npm run dev --host

# Open browser to http://localhost:5173
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation
```bash
# If PyTorch installation fails
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Memory Issues
```bash
# Reduce batch size in examples
# Edit demo_toroidal_diffusion.py:
# batch_size = 1  # instead of 2
```

#### 3. Import Errors
```bash
# Ensure you're in the correct directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 4. Web Interface Issues
```bash
# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Performance Optimization

#### GPU Acceleration
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Move models to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

#### Memory Management
```python
# Enable memory efficient attention
import torch
torch.backends.cuda.enable_flash_sdp(True)

# Use gradient checkpointing
model.enable_gradient_checkpointing()
```

## Development Setup

### For Contributors

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
pylint src/
```

### IDE Configuration

#### VS Code
```json
{
    "python.defaultInterpreterPath": "./toroidal_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm
- Set interpreter to `./toroidal_env/bin/python`
- Enable code style: Black
- Configure project structure to mark `src/` as source root

## Docker Setup (Optional)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5173

CMD ["python", "examples/demo_toroidal_diffusion.py"]
```

```bash
# Build and run
docker build -t toroidal-diffusion .
docker run -p 5173:5173 toroidal-diffusion
```

## Cloud Deployment

### Google Colab
```python
# Install in Colab
!git clone <repository-url>
%cd toroidal_diffusion_project
!pip install -r requirements.txt

# Run demo
!python examples/demo_toroidal_diffusion.py
```

### Jupyter Notebook
```python
# Install kernel
python -m ipykernel install --user --name=toroidal_env

# Start notebook
jupyter notebook
```

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Core components import successfully
- [ ] Individual tests pass
- [ ] Comprehensive demo runs
- [ ] Web interface loads
- [ ] GPU acceleration working (if available)

## Next Steps

After successful installation:

1. **Explore Examples**: Run `demo_toroidal_diffusion.py`
2. **Try Web Interface**: Open http://localhost:5173
3. **Read Documentation**: Review README.md and code comments
4. **Experiment**: Modify parameters and observe results
5. **Contribute**: Submit improvements and bug fixes

## Support

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Search existing issues on GitHub
4. Contact: stephansolncev@gmail.com or @personaz1 on Telegram

---

**Happy experimenting with Toroidal Diffusion Models!**

