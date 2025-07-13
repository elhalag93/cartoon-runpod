# Dockerfile for Cartoon Animation Worker on RunPod

FROM python:3.10-slim
RUN mkdir -p /workspace
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
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir torch diffusers transformers runpod numpy pillow soundfile tqdm requests gradio fastapi uvicorn

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
ENTRYPOINT ["python"]
CMD ["handler.py"] 