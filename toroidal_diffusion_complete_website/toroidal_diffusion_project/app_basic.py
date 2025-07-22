#!/usr/bin/env python3
"""
TORUS Basic Gradio App - Ultra-Minimal Version
"""

import gradio as gr
import numpy as np

def generate(prompt):
    """Ultra-simple generation."""
    # Create basic image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return img, f"Generated: {prompt}"

# Ultra-minimal interface
demo = gr.Interface(
    fn=generate,
    inputs="text",
    outputs=["image", "text"],
    title="TORUS Basic"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True) 