# 🎬 Cartoon Animation Studio - Interface Guide

> Enhanced interfaces for generating cartoon character animations with voice using AI

## 🚀 Quick Start

### Option 1: Web Interface (Recommended for beginners)
```bash
# Start the interactive web interface
python launch.py web

# Or with sharing enabled
python launch.py web --share
```

### Option 2: API Server (For developers)
```bash
# Start the REST API server
python launch.py api

# Or with auto-reload for development
python launch.py api --reload
```

### Option 3: Direct Launch
```bash
# Web interface directly
python web_interface.py --host 0.0.0.0 --port 7860

# API server directly
python api_server.py --host 0.0.0.0 --port 8000
```

## 🎭 Web Interface Features

### 🖥️ Interactive Dashboard
- **Real-time Generation**: Watch your animations generate in real-time
- **Character Selection**: Easy dropdown to choose between Temo and Felfel
- **Parameter Controls**: Intuitive sliders and inputs for all settings
- **Progress Tracking**: See generation progress with detailed status updates
- **Result Preview**: Immediate preview of generated GIFs, videos, and audio

### 📱 Three Generation Modes

#### 1. 🎬 Animation Only
- Generate character animations without voice
- Perfect for creating silent animations
- Outputs: GIF + MP4 files

#### 2. 🎵 Text-to-Speech Only
- Convert text to realistic speech
- Support for multiple speakers using `[S1]` and `[S2]` tags
- Outputs: Audio file

#### 3. 🎬🎵 Combined Generation
- Generate animation + synchronized speech
- Perfect for creating complete animated scenes
- Outputs: GIF + MP4 + Audio files

### 🎨 Advanced Features
- **Character-specific Prompts**: Auto-suggested prompts for each character
- **Seed Control**: Reproducible results with custom seeds
- **Quality Settings**: Adjust frames, resolution, and inference steps
- **Memory Optimization**: Automatic memory management for different GPU sizes

## 🔌 API Server Features

### 📡 REST API Endpoints

#### Animation Generation
```http
POST /api/animation
Content-Type: application/json

{
  "character": "temo",
  "prompt": "temo character walking on moon surface",
  "num_frames": 16,
  "fps": 8,
  "width": 512,
  "height": 512,
  "guidance_scale": 7.5,
  "num_inference_steps": 15,
  "seed": 42
}
```

#### TTS Generation
```http
POST /api/tts
Content-Type: application/json

{
  "dialogue_text": "[S1] Hello from the moon! [S2] What an adventure!",
  "max_new_tokens": 3072,
  "guidance_scale": 3.0,
  "temperature": 1.8,
  "seed": 42
}
```

#### Combined Generation
```http
POST /api/combined
Content-Type: application/json

{
  "character": "temo",
  "prompt": "temo character waving hello",
  "dialogue_text": "[S1] Greetings from space!",
  "num_frames": 16,
  "fps": 8,
  "seed": 42
}
```

### 📋 Response Format
```json
{
  "task_type": "animation",
  "gif": "base64_encoded_gif_data",
  "mp4": "base64_encoded_mp4_data",
  "audio": "base64_encoded_audio_data",
  "seed": 42,
  "memory_usage": {
    "allocated_gb": 4.2,
    "total_gb": 24.0
  }
}
```

### 🛠️ Utility Endpoints
- `GET /` - API status and model information
- `GET /health` - Health check
- `GET /api/characters` - Available characters and their descriptions
- `GET /api/examples` - Example requests for each endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## 🎯 Usage Examples

### Web Interface Examples

#### Basic Animation
1. Open web interface at `http://localhost:7860`
2. Go to "🎬 Animation Generation" tab
3. Select character: "temo"
4. Enter prompt: "temo character exploring alien planet"
5. Click "🎬 Generate Animation"

#### TTS with Multiple Speakers
1. Go to "🎵 Text-to-Speech" tab
2. Enter dialogue: `[S1] Welcome to our space station! [S2] Thank you for having me here!`
3. Adjust temperature for voice variation
4. Click "🎵 Generate Speech"

#### Combined Scene
1. Go to "🎬🎵 Animation + Speech" tab
2. Character: "felfel"
3. Prompt: "felfel character discovering magical crystal"
4. Dialogue: `[S1] Look at this amazing crystal! [S2] It's absolutely beautiful!`
5. Click "🎬🎵 Generate Combined"

### API Examples

#### Python Client
```python
import requests
import base64

# Animation generation
response = requests.post("http://localhost:8000/api/animation", json={
    "character": "temo",
    "prompt": "temo character dancing on moon",
    "num_frames": 16,
    "seed": 42
})

result = response.json()

# Save GIF
if result["gif"]:
    gif_data = base64.b64decode(result["gif"])
    with open("animation.gif", "wb") as f:
        f.write(gif_data)

# Save MP4
if result["mp4"]:
    mp4_data = base64.b64decode(result["mp4"])
    with open("animation.mp4", "wb") as f:
        f.write(mp4_data)
```

#### JavaScript Client
```javascript
const generateAnimation = async () => {
  const response = await fetch('http://localhost:8000/api/animation', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      character: 'temo',
      prompt: 'temo character waving hello',
      num_frames: 16,
      seed: 42
    })
  });
  
  const result = await response.json();
  
  // Create download links
  if (result.gif) {
    const gifBlob = new Blob([Uint8Array.from(atob(result.gif), c => c.charCodeAt(0))]);
    const gifUrl = URL.createObjectURL(gifBlob);
    // Use gifUrl for download or display
  }
};
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/animation" \
  -H "Content-Type: application/json" \
  -d '{
    "character": "temo",
    "prompt": "temo character exploring space",
    "num_frames": 16,
    "seed": 42
  }'
```

## ⚙️ Configuration

### Environment Variables
```bash
# Model paths (optional, will use defaults)
export MODELS_DIR="/workspace/models"
export LORA_DIR="/workspace/lora_models"
export OUTPUT_DIR="/workspace/outputs"
export TEMP_DIR="/workspace/temp"

# Model URLs (optional)
export TEMO_LORA_URL="https://your-lora-url.com/temo.pt"
export FELFEL_LORA_URL="https://your-lora-url.com/felfel.pt"
```

### Custom Launch Options
```bash
# Web interface with custom settings
python launch.py web --host 0.0.0.0 --port 8080 --share --debug

# API server with custom settings
python launch.py api --host 127.0.0.1 --port 9000 --reload
```

## 🔧 Advanced Features

### Memory Optimization
Both interfaces automatically apply memory optimizations:
- Sequential CPU offload for large models
- Attention slicing for reduced VRAM usage
- VAE slicing and tiling for high-resolution outputs
- Automatic garbage collection between generations

### Error Handling
- Graceful error messages in web interface
- Detailed error responses in API
- Automatic retry mechanisms for common failures
- Memory cleanup on errors

### Performance Monitoring
- Real-time memory usage tracking
- Generation time measurements
- Model loading status indicators
- System resource monitoring

## 🚀 Deployment Options

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start web interface
python launch.py web --debug

# Start API server with auto-reload
python launch.py api --reload
```

### Production Deployment
```bash
# Web interface (production)
python launch.py web --host 0.0.0.0 --port 80

# API server (production)
python launch.py api --host 0.0.0.0 --port 8000
```

### Docker Deployment
```dockerfile
# Use existing Dockerfile
FROM python:3.10-slim

# Copy interface files
COPY web_interface.py api_server.py launch.py ./
COPY src/ ./src/

# Install dependencies
RUN pip install -r requirements.txt

# Expose ports
EXPOSE 7860 8000

# Start launcher
CMD ["python", "launch.py", "web"]
```

### RunPod Deployment
```yaml
# runpod.yml
version: 1
image: your-image:latest
ports:
  - 7860:7860  # Web interface
  - 8000:8000  # API server
env:
  - MODELS_DIR=/workspace/models
  - LORA_DIR=/workspace/lora_models
command: ["python", "launch.py", "web", "--host", "0.0.0.0"]
```

## 🎨 Customization

### Adding New Characters
1. Add LoRA weights to `/workspace/lora_models/{character}_lora/`
2. Update `CHARACTERS` list in `web_interface.py`
3. Add default prompt in `DEFAULT_PROMPTS` dictionary
4. Update API character endpoint

### Custom Themes
```python
# In web_interface.py
interface = gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(),  # or gr.themes.Glass(), gr.themes.Monochrome()
    title="Your Custom Title"
)
```

### API Extensions
```python
# Add custom endpoint in api_server.py
@app.post("/api/custom")
async def custom_endpoint(request: CustomRequest):
    # Your custom logic here
    return {"result": "success"}
```

## 📊 Performance Benchmarks

### Web Interface
- **Startup Time**: ~30-60 seconds (model loading)
- **Generation Time**: 15-45 seconds (depending on frames)
- **Memory Usage**: 4-8GB VRAM (depending on settings)

### API Server
- **Request Latency**: <100ms (excluding generation)
- **Concurrent Requests**: 1 (due to GPU limitations)
- **Throughput**: ~2-4 generations/minute

## 🐛 Troubleshooting

### Common Issues

#### Web Interface Not Loading
```bash
# Check if port is available
netstat -an | grep :7860

# Try different port
python launch.py web --port 8080
```

#### API Server Errors
```bash
# Check logs
python launch.py api --debug

# Verify model loading
curl http://localhost:8000/health
```

#### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi

# Reduce settings
# - Lower num_frames (8-12)
# - Reduce resolution (256x256)
# - Fewer inference steps (10-12)
```

#### Model Loading Failures
```bash
# Check model paths
ls -la /workspace/models/
ls -la /workspace/lora_models/

# Download models manually
python download_models.py
```

## 📚 Additional Resources

- **API Documentation**: `http://localhost:8000/docs`
- **Model Information**: See `src/handler.py` for model details
- **Character LoRAs**: Available at Google Drive (see main README)
- **Examples**: Check `test_input.json` for sample requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test both web and API interfaces
4. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**🎬 Ready to create amazing cartoon animations with an intuitive interface!** 🚀✨ 