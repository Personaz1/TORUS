#!/usr/bin/env python3
"""
TORUS Minimal Gradio App - Schema-Safe Version
"""

import gradio as gr
import numpy as np
from PIL import Image

def generate_image(prompt):
    """Minimal generation function with no complex parameters."""
    try:
        # Create a simple image
        size = 256
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        
        # Simple pattern based on prompt
        if "mountain" in prompt.lower():
            for i in range(size):
                for j in range(size):
                    height = int(255 * (1 - i/size))
                    img_array[i, j] = [height//3, height//2, height]
        elif "abstract" in prompt.lower():
            # Abstract pattern
            for i in range(size):
                for j in range(size):
                    img_array[i, j] = [(i+j)%255, (i*2)%255, (j*2)%255]
        
        image = Image.fromarray(img_array)
        return image, f"Generated: {prompt}"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create minimal interface with no problematic components
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Prompt", value="A serene landscape"),
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Status")
    ],
    title="🌀 TORUS Minimal",
    description="Self-Stabilizing Generative Architecture"
)

if __name__ == "__main__":
    print("=== TORUS Minimal App ===")
    print("Launching minimal interface...")
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True
        )
    except Exception as e:
        print(f"Launch failed: {e}")
        import traceback
        traceback.print_exc() 