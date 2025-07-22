#!/usr/bin/env python3
"""
Minimal test script for TORUS Gradio interface
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image

def test_generation(prompt, steps, batch_size, enable_singularity, enable_coherence):
    """Test function that returns a simple image and metadata."""
    try:
        # Create a simple test image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(img_array)
        
        # Create metadata
        metadata = {
            "prompt": prompt,
            "steps": steps,
            "batch_size": batch_size,
            "singularity": enable_singularity,
            "coherence": enable_coherence,
            "status": "Test successful"
        }
        
        return image, str(metadata)
    except Exception as e:
        return None, f"Test error: {str(e)}"

def create_test_interface():
    """Create a minimal test interface."""
    interface = gr.Interface(
        fn=test_generation,
        inputs=[
            gr.Textbox(label="Prompt", value="Test prompt"),
            gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Steps"),
            gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Batch Size"),
            gr.Checkbox(label="Singularity", value=True, interactive=True),
            gr.Checkbox(label="Coherence", value=True, interactive=True)
        ],
        outputs=[
            gr.Image(label="Generated Image"),
            gr.Textbox(label="Info", lines=5)
        ],
        title="🌀 TORUS Test Interface",
        description="Test interface for schema compatibility",
        allow_flagging="never"
    )
    return interface

if __name__ == "__main__":
    try:
        print("Testing TORUS Gradio interface...")
        
        # Check gradio version
        import pkg_resources
        gradio_version = pkg_resources.get_distribution("gradio").version
        print(f"Gradio version: {gradio_version}")
        
        # Create and test interface
        demo = create_test_interface()
        
        print("✅ Interface created successfully!")
        print("Launching test interface...")
        
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True,
            debug=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 