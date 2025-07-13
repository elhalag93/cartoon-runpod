# 🚀 SIMPLE RunPod Deployment Guide

## **Option 1: Direct Docker Deployment (EASIEST)**

### 1. Build and Push Your Image
```bash
# Build the image
docker build -t your-username/cartoon-animation:latest .

# Push to Docker Hub
docker push your-username/cartoon-animation:latest
```

### 2. Deploy on RunPod
1. Go to RunPod.io
2. Create new Pod
3. Use your Docker image: `your-username/cartoon-animation:latest`
4. Set container command: `python handler.py`
5. Expose port: `8000` (for API) or `7860` (for web)
6. Select GPU: RTX 4090 or A6000 (16GB+ VRAM)

### 3. Test Your Deployment
Send a POST request to your RunPod endpoint:
```json
{
  "input": {
    "task_type": "animation",
    "character": "temo",
    "prompt": "temo character waving hello",
    "num_frames": 8,
    "seed": 42
  }
}
```

## **Option 2: GitHub Integration (ALTERNATIVE)**

### 1. Update Repository URL
In `.runpod/hub.json`, change:
```json
"repository": "https://github.com/elhalag93/cartoon-runpod"
```

### 2. Deploy via GitHub
1. Go to RunPod Hub
2. Connect GitHub
3. Select your repository
4. RunPod will auto-deploy

## **Option 3: Manual File Upload (SIMPLEST)**

### 1. Create RunPod Pod
- Select GPU: RTX 4090
- Use base image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`

### 2. Upload Files
Upload these key files to `/workspace/`:
- `handler.py` (root entry point)
- `src/handler.py` (main implementation)
- `requirements.txt`
- `lora_models/` (your character weights)

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Handler
```bash
python handler.py
```

## **Troubleshooting**

### If "handler is required" error:
1. Make sure `handler.py` exists in root directory
2. Check that it imports from `src/handler.py`
3. Verify RunPod can find the handler function

### If models not found:
1. Upload LoRA weights to `/workspace/lora_models/`
2. Set environment variables for model URLs
3. Run `python download_models.py`

### If CUDA errors:
1. Use GPU with 16GB+ VRAM
2. Reduce `num_frames` to 8
3. Lower resolution to 256x256

## **Quick Test Commands**

```bash
# Test handler import
python -c "from handler import handler; print('✅ Handler works')"

# Test with minimal input
python -c "
import json
from handler import handler
result = handler({'input': {'task_type': 'animation', 'character': 'temo', 'prompt': 'test'}})
print('✅ Result:', type(result))
"
```

## **Expected Response**
```json
{
  "task_type": "animation",
  "gif": "base64_encoded_data...",
  "mp4": "base64_encoded_data...",
  "seed": 42,
  "generation_time": 25.3,
  "memory_usage": {"allocated_gb": 4.2, "total_gb": 24.0}
}
```

## **🎯 SUCCESS INDICATORS**

✅ **Handler loads without errors**  
✅ **Models download successfully**  
✅ **Generation completes in 15-60 seconds**  
✅ **Returns base64-encoded GIF/MP4**  
✅ **Memory usage reported correctly**

**That's it! Your cartoon animation worker should now work on RunPod!** 🎬✨ 