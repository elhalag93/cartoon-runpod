# Multi-stage Dockerfile for Cartoon Animation Worker - Optimized for Space

# Stage 1: Build dependencies
FROM python:3.10-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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
    requests>=2.31.0

# Stage 2: Runtime image with CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy application code
COPY . .

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
RUN python download_models.py || echo "Model download deferred - will download at runtime"

# Default command - use standalone mode to prevent RunPod connections
CMD ["python", "-u", "start_standalone.py"]