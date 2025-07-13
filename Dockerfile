# Dockerfile for Cartoon Animation Worker on RunPod

FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies in stages for better error handling
RUN pip install --no-cache-dir --upgrade pip

# Install core dependencies first
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    torchvision \
    transformers>=4.40.0 \
    diffusers>=0.30.0 \
    accelerate>=0.30.0 \
    runpod>=1.7.0 \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    requests>=2.31.0 \
    soundfile>=0.13.1

# Install web framework dependencies
RUN pip install --no-cache-dir \
    gradio>=5.25.2 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.11.3

# Install optional dependencies (ignore failures)
RUN pip install --no-cache-dir \
    huggingface-hub>=0.30.2 \
    safetensors>=0.5.3 \
    descript-audio-codec>=1.0.0 \
    tqdm \
    gdown \
    || echo "Some optional dependencies failed to install"

# Try to install xformers (ignore if it fails)
RUN pip install --no-cache-dir xformers>=0.0.20 || echo "xformers not available"

# Create necessary directories
RUN mkdir -p /workspace/models/sdxl-turbo /workspace/models/animatediff/motion_adapter /workspace/lora_models /workspace/outputs /workspace/temp

# Download models (will use environment variables if set)
RUN python download_models.py || echo "Model download script failed, ensure models are pre-loaded or URLs are set."

# Expose ports for both interfaces
EXPOSE 7860 8000

# Set environment variables for interface mode
ENV INTERFACE_MODE=web
ENV HOST=0.0.0.0
ENV PORT=7860

# Set entrypoint based on deployment mode
# For RunPod serverless: python handler.py
# For web interface: python launch.py web
# For API server: python launch.py api
CMD python -u handler.py