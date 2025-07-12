# 🐳 Docker Deployment Guide - Cartoon Animation Studio

> Complete Docker setup for cartoon character animation generation with multiple interface options

## 🚀 Quick Start with Docker

### Option 1: Web Interface (Recommended for Users)
```bash
# Build and start the web interface
docker-compose up cartoon-web

# Access at: http://localhost:7860
```

### Option 2: API Server (For Developers)
```bash
# Build and start the API server
docker-compose up cartoon-api

# Access at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### Option 3: All Services
```bash
# Start all services simultaneously
docker-compose up

# Web Interface: http://localhost:7860
# API Server: http://localhost:8000
# RunPod Worker: http://localhost:8001
```

## 🏗️ Docker Architecture

### Services Overview
- **`cartoon-web`**: Interactive Gradio web interface (Port 7860)
- **`cartoon-api`**: FastAPI server with REST endpoints (Port 8000)  
- **`cartoon-worker`**: Original RunPod handler (Port 8001)

### Container Structure
```
cartoon-animation-container/
├── /workspace/
│   ├── models/           # AI models (mounted volume)
│   ├── lora_models/      # Character LoRA weights (mounted volume)
│   ├── outputs/          # Generated animations (mounted volume)
│   ├── temp/             # Temporary files (mounted volume)
│   ├── web_interface.py  # Web interface
│   ├── api_server.py     # API server
│   ├── launch.py         # Service launcher
│   └── src/handler.py    # RunPod handler
```

## 🔧 Configuration

### Environment Variables
Set these in your `.env` file or docker-compose.yml:

```bash
# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all

# Debug Mode
RUNPOD_DEBUG=true

# Interface Mode
INTERFACE_MODE=web  # or 'api'

# Network Configuration
HOST=0.0.0.0
PORT=7860  # or 8000 for API

# Model URLs (optional)
TEMO_LORA_URL=https://your-lora-url.com/temo.pt
FELFEL_LORA_URL=https://your-lora-url.com/felfel.pt
```

### Volume Mounts
```yaml
volumes:
  - ./models:/workspace/models           # AI models
  - ./lora_models:/workspace/lora_models # Character weights
  - ./outputs:/workspace/outputs         # Generated content
  - ./temp:/workspace/temp               # Temporary files
```

## 🎯 Usage Examples

### Web Interface Container
```bash
# Start web interface
docker-compose up cartoon-web

# With custom port
docker-compose run -p 8080:7860 cartoon-web web --port 7860

# With sharing enabled
docker-compose run cartoon-web web --share
```

### API Server Container
```bash
# Start API server
docker-compose up cartoon-api

# With auto-reload (development)
docker-compose run cartoon-api api --reload

# Custom port
docker-compose run -p 9000:8000 cartoon-api api --port 8000
```

### Direct Docker Commands
```bash
# Build the image
docker build -t cartoon-animation .

# Run web interface
docker run -p 7860:7860 --gpus all cartoon-animation web

# Run API server
docker run -p 8000:8000 --gpus all cartoon-animation api

# Run with volume mounts
docker run -p 7860:7860 --gpus all \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/lora_models:/workspace/lora_models \
  -v $(pwd)/outputs:/workspace/outputs \
  cartoon-animation web
```

## 🔌 API Usage from Host

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get available characters
curl http://localhost:8000/api/characters

# Generate animation
curl -X POST http://localhost:8000/api/animation \
  -H "Content-Type: application/json" \
  -d '{
    "character": "temo",
    "prompt": "temo character exploring space",
    "num_frames": 16,
    "seed": 42
  }'
```

### Python Client Example
```python
import requests
import base64

# Connect to containerized API
api_url = "http://localhost:8000"

# Generate animation
response = requests.post(f"{api_url}/api/animation", json={
    "character": "temo",
    "prompt": "temo character dancing on moon",
    "num_frames": 16,
    "seed": 42
})

result = response.json()

# Save generated files
if result["gif"]:
    gif_data = base64.b64decode(result["gif"])
    with open("animation.gif", "wb") as f:
        f.write(gif_data)
```

## 🚀 Production Deployment

### Docker Swarm
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml cartoon-stack
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cartoon-web
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cartoon-web
  template:
    metadata:
      labels:
        app: cartoon-web
    spec:
      containers:
      - name: cartoon-web
        image: cartoon-animation:latest
        args: ["web", "--host", "0.0.0.0"]
        ports:
        - containerPort: 7860
        resources:
          limits:
            nvidia.com/gpu: 1
```

### RunPod Deployment
```bash
# Build and push image
docker build -t your-registry/cartoon-animation:latest .
docker push your-registry/cartoon-animation:latest

# Deploy on RunPod using the image
# Use environment variables for configuration
```

## 🔍 Monitoring & Debugging

### Container Logs
```bash
# View web interface logs
docker-compose logs cartoon-web

# View API server logs
docker-compose logs cartoon-api

# Follow logs in real-time
docker-compose logs -f cartoon-web
```

### Container Shell Access
```bash
# Access web interface container
docker-compose exec cartoon-web bash

# Access API server container
docker-compose exec cartoon-api bash

# Check GPU access
docker-compose exec cartoon-web nvidia-smi
```

### Health Checks
```bash
# Check container status
docker-compose ps

# Test web interface
curl http://localhost:7860

# Test API server
curl http://localhost:8000/health
```

## 📊 Performance Optimization

### GPU Memory Management
```dockerfile
# Add to Dockerfile for memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0
```

### Container Resources
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '4'
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🛠️ Development Setup

### Development with Hot Reload
```bash
# API server with auto-reload
docker-compose run -p 8000:8000 cartoon-api api --reload

# Mount source code for development
docker-compose run -v $(pwd):/workspace cartoon-api api --reload
```

### Testing in Container
```bash
# Run tests inside container
docker-compose exec cartoon-web python test_interface.py

# Run specific test
docker-compose exec cartoon-api python test_interface.py api
```

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check build logs
docker-compose build --no-cache

# Check container logs
docker-compose logs cartoon-web
```

#### GPU Not Available
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Verify GPU access in container
docker-compose exec cartoon-web nvidia-smi
```

#### Port Already in Use
```bash
# Check what's using the port
netstat -tulpn | grep :7860

# Use different port
docker-compose run -p 8080:7860 cartoon-web web
```

#### Model Loading Issues
```bash
# Check model directory
docker-compose exec cartoon-web ls -la /workspace/models/

# Download models manually
docker-compose exec cartoon-web python download_models.py
```

## 📋 Docker Commands Reference

### Build & Run
```bash
docker build -t cartoon-animation .
docker run -p 7860:7860 --gpus all cartoon-animation web
```

### Compose Commands
```bash
docker-compose up                    # Start all services
docker-compose up cartoon-web        # Start web interface only
docker-compose up cartoon-api        # Start API server only
docker-compose down                  # Stop all services
docker-compose build                 # Rebuild images
docker-compose logs                  # View logs
docker-compose ps                    # Check status
```

### Cleanup
```bash
docker-compose down -v              # Stop and remove volumes
docker system prune                 # Clean up unused containers
docker image prune                  # Remove unused images
```

## 🎉 Ready for Docker Deployment!

Your cartoon animation system is now fully containerized with:

- ✅ **Web Interface Container** (Port 7860)
- ✅ **API Server Container** (Port 8000)  
- ✅ **RunPod Worker Container** (Port 8001)
- ✅ **GPU Support** with NVIDIA runtime
- ✅ **Volume Mounts** for persistent data
- ✅ **Environment Configuration**
- ✅ **Production Ready** deployment options

Use `docker-compose up` to start your preferred interface! 