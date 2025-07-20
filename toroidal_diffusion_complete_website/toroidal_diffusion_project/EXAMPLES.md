# Usage Examples

## Basic Examples

### 1. Simple Generation

```python
import torch
from src.toroidal_diffusion_wrapper import ToroidalDiffusionModel
from diffusers import UNet2DModel, DDPMScheduler

# Create base model
base_model = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D", 
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D", 
        "UpBlock2D",
    ),
)

# Create scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000)

# Create toroidal model
model = ToroidalDiffusionModel(
    base_model=base_model,
    scheduler=scheduler,
    image_size=(64, 64)
)

# Generate sample
sample = model.sample(batch_size=1, num_inference_steps=50)
print(f"Generated sample shape: {sample['sample'].shape}")
```

### 2. Advanced Configuration

```python
# Advanced toroidal model with all features
advanced_model = ToroidalDiffusionModel(
    base_model=base_model,
    scheduler=scheduler,
    image_size=(64, 64),
    enable_singularity=True,
    enable_coherence_monitoring=True,
    enable_multi_pass=True,
    max_refinement_passes=5,
    toroidal_config={
        'major_radius': 1.0,
        'minor_radius': 0.3,
        'flow_strength': 0.1
    },
    singularity_config={
        'singularity_dim': 128,
        'coupling_strength': 0.1
    }
)

# Generate with detailed monitoring
result = advanced_model.sample(
    batch_size=2,
    num_inference_steps=50,
    return_dict=True
)

# Access metadata
print(f"Toroidal curvature: {result['toroidal_metadata']['curvature'].mean()}")
print(f"Coherence passes: {result['coherence_metadata']['total_passes']}")
```

### 3. Custom Topology

```python
from src.toroidal_topology import ToroidalLatentSpace, ToroidalFlow

# Create custom toroidal space
custom_torus = ToroidalLatentSpace(
    latent_dim=3,
    major_radius=1.5,  # Larger torus
    minor_radius=0.2   # Thinner tube
)

# Apply to latent
latent = torch.randn(1, 3, 64, 64)
result = custom_torus(latent)

print(f"Wrapped latent range: [{result['wrapped_latent'].min():.3f}, {result['wrapped_latent'].max():.3f}]")
print(f"Curvature statistics: mean={result['curvature'].mean():.6f}, std={result['curvature'].std():.6f}")
```

## Advanced Examples

### 4. Coherence Analysis

```python
from src.advanced_coherence_system import AdvancedCoherenceSystem, CoherenceVisualizer

# Create coherence system
coherence_system = AdvancedCoherenceSystem(
    feature_dim=64,
    max_refinement_passes=10,
    enable_hierarchical=True,
    enable_adaptive_threshold=True
)

# Analyze features
features = torch.randn(2, 64, 32, 32)
analysis = coherence_system(features)

# Print results
report = analysis['report']
print(f"Semantic coherence: {report.semantic_coherence:.4f}")
print(f"Structural coherence: {report.structural_coherence:.4f}")
print(f"Quality score: {report.quality_score:.4f}")
print(f"Convergence achieved: {report.convergence_achieved}")

# Visualize evolution
visualizer = CoherenceVisualizer()
if analysis['refinement_steps']:
    fig = visualizer.plot_coherence_evolution(analysis['refinement_steps'])
    fig.savefig('coherence_analysis.png')
```

### 5. Singularity Processing

```python
from src.central_singularity import SingularityToroidalCoupling, CognitiveFeedbackLoop

# Create singularity coupling
coupling = SingularityToroidalCoupling(
    latent_dim=64,
    singularity_dim=256,
    coupling_strength=0.2
)

# Process through singularity
features = torch.randn(1, 64, 32, 32)
result = coupling(features)

print(f"Coupling strength: {result['coupling_strength'].mean():.6f}")
print(f"Singularity influence magnitude: {result['singularity_influence'].abs().mean():.6f}")

# Add cognitive feedback
feedback = CognitiveFeedbackLoop(latent_dim=64, memory_size=10)

# Iterative processing
current_features = result['coupled_features']
for i in range(5):
    feedback_result = feedback(current_features)
    current_features = feedback_result['modified_features']
    print(f"Iteration {i+1}: Action magnitude = {feedback_result['action'].abs().mean():.6f}")
```

### 6. Custom Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup training
model = ToroidalDiffusionModel(base_model, scheduler, (64, 64))
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (images, _) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Add noise
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 1000, (images.shape[0],))
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # Forward pass with toroidal processing
        result = model(noisy_images, timesteps, return_dict=True)
        
        # Compute loss
        loss = F.mse_loss(result['sample'], noise)
        
        # Add coherence regularization
        if 'coherence_metadata' in result:
            coherence_loss = 1.0 - result['coherence_metadata'].get('final_coherence', 0.0)
            loss += 0.1 * coherence_loss
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
```

## Specialized Use Cases

### 7. Image-to-Image Translation

```python
# Load source image
source_image = load_image("source.jpg").resize((64, 64))
source_tensor = torch.from_numpy(np.array(source_image)).permute(2, 0, 1).float() / 255.0
source_tensor = source_tensor.unsqueeze(0)

# Encode to latent space
with torch.no_grad():
    # Add small amount of noise
    noise_level = 0.1
    noise = torch.randn_like(source_tensor) * noise_level
    noisy_source = source_tensor + noise
    
    # Process through toroidal space
    toroidal_space = ToroidalLatentSpace(latent_dim=3)
    toroidal_result = toroidal_space(noisy_source)
    
    # Generate variations
    variations = []
    for i in range(4):
        # Apply different flow patterns
        flow = ToroidalFlow(channels=3, flow_strength=0.05 * (i + 1))
        varied = flow(toroidal_result['wrapped_latent'])
        variations.append(varied)

# Save variations
for i, var in enumerate(variations):
    save_image(var, f"variation_{i}.png")
```

### 8. Quality Assessment

```python
from src.coherence_monitor import CoherenceMetrics

# Assess generated samples
generated_samples = model.sample(batch_size=10, num_inference_steps=50)['sample']

# Compute coherence metrics
coherence_metrics = CoherenceMetrics(feature_dim=3)
quality_scores = []

for sample in generated_samples:
    sample_batch = sample.unsqueeze(0)
    metrics = coherence_metrics(sample_batch)
    
    semantic_score = metrics['semantic_coherence'].mean().item()
    structural_score = metrics['structural_coherence'].mean().item()
    overall_score = metrics['overall_coherence'].mean().item()
    
    quality_scores.append({
        'semantic': semantic_score,
        'structural': structural_score,
        'overall': overall_score
    })

# Analyze quality distribution
import numpy as np
overall_scores = [score['overall'] for score in quality_scores]
print(f"Quality statistics:")
print(f"  Mean: {np.mean(overall_scores):.4f}")
print(f"  Std:  {np.std(overall_scores):.4f}")
print(f"  Min:  {np.min(overall_scores):.4f}")
print(f"  Max:  {np.max(overall_scores):.4f}")
```

### 9. Batch Processing

```python
# Process multiple images efficiently
def process_batch(images, model, batch_size=4):
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_tensor = torch.stack(batch)
        
        with torch.no_grad():
            # Apply toroidal processing
            result = model.apply_toroidal_processing(batch_tensor)
            
            # Apply singularity processing
            singularity_result = model.apply_singularity_processing(result['toroidal_features'])
            
            # Apply coherence refinement
            coherence_result = model.apply_coherence_refinement(singularity_result['coupled_features'])
            
            results.extend(coherence_result['refined_features'])
    
    return results

# Example usage
image_list = [torch.randn(3, 64, 64) for _ in range(20)]
processed = process_batch(image_list, model)
print(f"Processed {len(processed)} images")
```

### 10. Parameter Sensitivity Analysis

```python
# Analyze sensitivity to toroidal parameters
parameter_ranges = {
    'major_radius': [0.5, 1.0, 1.5, 2.0],
    'minor_radius': [0.1, 0.2, 0.3, 0.4],
    'flow_strength': [0.01, 0.05, 0.1, 0.2]
}

results = {}

for param_name, values in parameter_ranges.items():
    param_results = []
    
    for value in values:
        # Create model with specific parameter
        config = {'major_radius': 1.0, 'minor_radius': 0.3, 'flow_strength': 0.1}
        config[param_name] = value
        
        toroidal_space = ToroidalLatentSpace(
            latent_dim=3,
            major_radius=config['major_radius'],
            minor_radius=config['minor_radius']
        )
        
        # Test with sample data
        test_data = torch.randn(5, 3, 32, 32)
        result = toroidal_space(test_data)
        
        # Compute metrics
        curvature_mean = result['curvature'].mean().item()
        distance_variance = toroidal_space.toroidal_distance(test_data, result['wrapped_latent']).var().item()
        
        param_results.append({
            'value': value,
            'curvature_mean': curvature_mean,
            'distance_variance': distance_variance
        })
    
    results[param_name] = param_results

# Print sensitivity analysis
for param_name, param_results in results.items():
    print(f"\n{param_name} sensitivity:")
    for result in param_results:
        print(f"  {result['value']:.2f}: curvature={result['curvature_mean']:.6f}, variance={result['distance_variance']:.6f}")
```

## Integration Examples

### 11. With Hugging Face Diffusers

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pretrained pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Wrap with toroidal processing
toroidal_wrapper = ToroidalDiffusionModel(
    base_model=pipe.unet,
    scheduler=pipe.scheduler,
    image_size=(512, 512),
    enable_singularity=True,
    enable_coherence_monitoring=True
)

# Replace UNet in pipeline
pipe.unet = toroidal_wrapper

# Generate with toroidal enhancement
prompt = "A beautiful landscape with mountains and lakes"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save("toroidal_enhanced_landscape.png")
```

### 12. Custom Loss Functions

```python
def toroidal_coherence_loss(pred_noise, target_noise, coherence_metadata):
    """Custom loss incorporating coherence metrics."""
    
    # Standard MSE loss
    mse_loss = F.mse_loss(pred_noise, target_noise)
    
    # Coherence regularization
    coherence_score = coherence_metadata.get('final_coherence', 0.0)
    coherence_loss = torch.clamp(0.8 - coherence_score, min=0.0)
    
    # Singularity influence regularization
    singularity_influence = coherence_metadata.get('singularity_influence', 0.0)
    singularity_loss = torch.abs(singularity_influence).mean()
    
    # Combined loss
    total_loss = mse_loss + 0.1 * coherence_loss + 0.05 * singularity_loss
    
    return total_loss, {
        'mse': mse_loss.item(),
        'coherence': coherence_loss.item(),
        'singularity': singularity_loss.item()
    }
```

## Performance Tips

### Memory Optimization
```python
# Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    result = model(noisy_images, timesteps)
    loss = compute_loss(result, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Speed Optimization
```python
# Compile model for faster inference (PyTorch 2.0+)
model = torch.compile(model)

# Reduce refinement passes for faster generation
fast_model = ToroidalDiffusionModel(
    base_model=base_model,
    scheduler=scheduler,
    image_size=(64, 64),
    max_refinement_passes=2  # Reduced from default 5
)
```

These examples demonstrate the flexibility and power of the Toroidal Diffusion Model architecture. Experiment with different configurations to find the optimal setup for your specific use case!

