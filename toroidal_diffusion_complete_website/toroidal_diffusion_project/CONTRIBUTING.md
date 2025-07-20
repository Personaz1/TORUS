# Contributing to TORUS

Thank you for your interest in contributing to TORUS! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/TORUS.git
   cd TORUS/toroidal_diffusion_complete_website/toroidal_diffusion_project
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests**
   ```bash
   python test_validation.py
   python quick_test.py
   ```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests
```bash
python test_validation.py
python benchmark.py --device cpu --batch-size 2 --steps 5
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google style
- **Type hints**: Include where helpful
- **Line length**: 88 characters (Black formatter)

## Testing

### Running Tests
```bash
# Full validation suite
python test_validation.py

# Quick functionality test
python quick_test.py

# Performance benchmark
python benchmark.py --device cpu --batch-size 4 --steps 10
```

### Adding Tests
- Add tests in `tests/` directory
- Use descriptive test names
- Test both success and failure cases

## Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Include type hints
- Provide usage examples

### README Updates
- Update README.md for new features
- Add installation instructions if needed
- Update performance metrics

## Issue Guidelines

### Bug Reports
- Use the bug report template
- Include steps to reproduce
- Provide error messages and stack traces
- Specify your environment (OS, Python version, etc.)

### Feature Requests
- Use the feature request template
- Describe the problem you're solving
- Provide use cases and examples
- Consider implementation complexity

## Pull Request Guidelines

### Before Submitting
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Performance impact is considered

### PR Description
- Describe the changes clearly
- Link to related issues
- Include performance benchmarks if applicable
- Add screenshots for UI changes

## Performance Guidelines

### Benchmarking
- Run benchmarks before and after changes
- Document performance impact
- Consider both CPU and GPU performance
- Test with different batch sizes

### Optimization
- Profile code for bottlenecks
- Consider memory usage
- Optimize for both inference and training
- Maintain code readability

## Release Process

### Versioning
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Update version in relevant files
- Create release notes

### Pre-release Checklist
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] Performance benchmarks are updated
- [ ] Release notes are written

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact stephansolncev@gmail.com for private matters

## License

By contributing to TORUS, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to TORUS! 🌀🧠 