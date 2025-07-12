# Cartoon Animation Worker v1.0.0

## 🎬 Features
- Generate cartoon character animations using AnimateDiff + SDXL Turbo
- Text-to-speech generation using Dia TTS
- Combined animation + voice generation
- Support for custom characters (Temo, Felfel) with LoRA weights
- Web interface and REST API
- RunPod serverless deployment ready

## 🚀 Usage

### RunPod Deployment
```bash
# Use this Docker image on RunPod:
your-dockerhub-username/cartoon-animation:1.0.0

# Set container command to:
python src/handler.py  # For worker mode
python launch.py web   # For web interface
python launch.py api   # For API server
```

### Local Docker
```bash
docker run -p 7860:7860 --gpus all your-dockerhub-username/cartoon-animation:1.0.0 web
```

## 📋 Input Format
```json
{
  "input": {
    "task_type": "combined",
    "character": "temo",
    "prompt": "temo character walking on moon surface",
    "dialogue_text": "[S1] Hello from the moon!",
    "num_frames": 16,
    "seed": 42
  }
}
```

## 🔧 Requirements
- GPU with 16GB+ VRAM (recommended)
- CUDA 12.1+
- Environment variables for LoRA weights (if using private URLs)

## 🐛 Bug Fixes
- See commit history for detailed changes

## 📝 Documentation
- See README.md for complete setup instructions
- Check INPUT_OUTPUT_GUIDE.md for usage examples
