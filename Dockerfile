# Dockerfile for Cartoon Animation Worker on RunPod

FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir torch diffusers transformers runpod numpy pillow soundfile tqdm requests

# Create necessary directories
RUN mkdir -p /workspace/models/sdxl-turbo /workspace/models/animatediff/motion_adapter /workspace/lora_models /workspace/outputs /workspace/temp

# Download models (will use environment variables if set)
RUN python download_models.py || echo "Model download script failed, ensure models are pre-loaded or URLs are set."

# Expose port for RunPod worker
EXPOSE 8000

# Set entrypoint to start the worker
ENTRYPOINT ["python", "src/handler.py"] 