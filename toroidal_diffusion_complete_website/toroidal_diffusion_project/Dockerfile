FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy TORUS code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for web interface
EXPOSE 3000

# Default command
CMD ["python", "examples/demo_toroidal_diffusion.py"] 