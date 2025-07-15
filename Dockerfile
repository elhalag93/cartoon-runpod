# Simplified Dockerfile for Cartoon Animation Worker - No Virtual Environment Needed

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . .

# Install PyTorch with CUDA support - using latest compatible version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    diffusers>=0.30.0 \
    accelerate>=0.30.0 \
    runpod>=1.7.0 \
    gradio>=5.25.2 \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    huggingface-hub>=0.30.2 \
    numpy>=1.24.0 \
    pillow>=10.0.0 \
    soundfile>=0.13.1 \
    pydantic>=2.11.3 \
    safetensors>=0.5.3 \
    requests>=2.31.0   \
    gdown>=4.6.0
# Create necessary directories
RUN mkdir -p /workspace/models/sdxl-turbo \
    /workspace/models/animatediff/motion_adapter \
    /workspace/lora_models \
    /workspace/outputs \
    /workspace/temp

# Expose ports
EXPOSE 7860 8000

# Set environment variables
ENV PYTHONPATH="/workspace"
ENV INTERFACE_MODE=web
ENV HOST=0.0.0.0
ENV PORT=7860

# Disable RunPod connections in container mode (standalone by default)
ENV RUNPOD_STANDALONE_MODE=true
ENV STANDALONE_WORKER=true
ENV RUNPOD_DISABLE=true
ENV LOCAL_DEVELOPMENT=true

# Download models (will use environment variables if set)
#RUN python download_models.py || echo "Model download deferred - will download at runtime"
RUN python download_models.py

# CRITICAL: Completely disable RunPod aiapi service to prevent connection errors
RUN echo '#!/bin/bash\necho "🚫 RunPod aiapi disabled in standalone mode"\nexit 0' > /bin/aiapi && chmod +x /bin/aiapi
RUN echo '#!/bin/bash\necho "🚫 RunPod service disabled in standalone mode"\nexit 0' > /usr/local/bin/runpod && chmod +x /usr/local/bin/runpod || true

# Disable any RunPod system services
RUN systemctl disable runpod 2>/dev/null || true
RUN systemctl mask runpod 2>/dev/null || true

# Default command - use standalone mode to prevent RunPod connections
CMD ["python","src/start_standalone.py"]