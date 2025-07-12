# 📋 Input/Output Guide - Cartoon Animation Studio

> Complete guide to input parameters, output formats, and usage examples

## 🎯 **Task Types**

The application supports three main task types:

1. **`animation`** - Generate character animations only
2. **`tts`** - Generate text-to-speech audio only  
3. **`combined`** - Generate both animation and speech

## 📥 **Input Formats**

### 🎬 **Animation Input**

```json
{
  "input": {
    "task_type": "animation",
    "character": "temo",
    "prompt": "temo character walking on moon surface, detailed cartoon style",
    "negative_prompt": "blurry, low quality, distorted",
    "num_frames": 16,
    "fps": 8,
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5,
    "num_inference_steps": 15,
    "seed": 42
  }
}
```

#### Animation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `character` | string | **required** | Character name: `"temo"` or `"felfel"` |
| `prompt` | string | **required** | Animation description |
| `negative_prompt` | string | `"blurry, low quality"` | What to avoid |
| `num_frames` | integer | `16` | Number of frames (8-32) |
| `fps` | integer | `8` | Frames per second (4-12) |
| `width` | integer | `512` | Video width (256-768) |
| `height` | integer | `512` | Video height (256-768) |
| `guidance_scale` | float | `7.5` | How closely to follow prompt (1.0-15.0) |
| `num_inference_steps` | integer | `15` | Quality vs speed (10-30) |
| `seed` | integer | `null` | Random seed for reproducibility |

### 🎵 **TTS Input**

```json
{
  "input": {
    "task_type": "tts",
    "dialogue_text": "[S1] Hello from the moon! [S2] What an amazing adventure!",
    "max_new_tokens": 3072,
    "guidance_scale": 3.0,
    "temperature": 1.8,
    "top_p": 0.9,
    "top_k": 45,
    "seed": 42
  }
}
```

#### TTS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dialogue_text` | string | **required** | Text with speaker tags `[S1]`, `[S2]` |
| `max_new_tokens` | integer | `3072` | Maximum audio tokens (1024-4096) |
| `guidance_scale` | float | `3.0` | TTS guidance strength (1.0-10.0) |
| `temperature` | float | `1.8` | Voice variation (0.1-2.0) |
| `top_p` | float | `0.9` | Nucleus sampling (0.1-1.0) |
| `top_k` | integer | `45` | Top-k sampling (1-100) |
| `seed` | integer | `null` | Random seed for reproducibility |

### 🎬🎵 **Combined Input**

```json
{
  "input": {
    "task_type": "combined",
    "character": "temo",
    "prompt": "temo character waving hello from moon surface",
    "dialogue_text": "[S1] Greetings from the lunar surface! [S2] This is incredible!",
    "num_frames": 16,
    "fps": 8,
    "width": 512,
    "height": 512,
    "guidance_scale": 7.5,
    "num_inference_steps": 15,
    "max_new_tokens": 3072,
    "tts_guidance_scale": 3.0,
    "temperature": 1.8,
    "seed": 42
  }
}
```

#### Combined Parameters

Combines all animation and TTS parameters above.

## 📤 **Output Formats**

### 🎬 **Animation Output**

```json
{
  "task_type": "animation",
  "gif": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mp4": "AAAAIGZ0eXBpc29tAAACAGlzb2...",
  "gif_path": "/workspace/outputs/temo_1234567890.gif",
  "mp4_path": "/workspace/outputs/temo_1234567890.mp4",
  "seed": 42,
  "memory_usage": {
    "allocated_gb": 4.2,
    "total_gb": 24.0
  }
}
```

### 🎵 **TTS Output**

```json
{
  "task_type": "tts",
  "audio": "UklGRnoGAABXQVZFZm10IBAAAAABAAEA...",
  "audio_path": "/workspace/temp/tts_output_1234567890.wav",
  "seed": 42,
  "dialogue_text": "[S1] Hello from the moon! [S2] What an amazing adventure!",
  "memory_usage": {
    "allocated_gb": 2.1,
    "total_gb": 24.0
  }
}
```

### 🎬🎵 **Combined Output**

```json
{
  "task_type": "combined",
  "gif": "iVBORw0KGgoAAAANSUhEUgAA...",
  "mp4": "AAAAIGZ0eXBpc29tAAACAGlzb2...",
  "audio": "UklGRnoGAABXQVZFZm10IBAAAAABAAEA...",
  "gif_path": "/workspace/outputs/temo_1234567890.gif",
  "mp4_path": "/workspace/outputs/temo_1234567890.mp4",
  "audio_path": "/workspace/temp/tts_output_1234567890.wav",
  "seed": 42,
  "memory_usage": {
    "allocated_gb": 6.3,
    "total_gb": 24.0
  }
}
```

#### Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | string | Type of generation performed |
| `gif` | string | Base64-encoded GIF animation |
| `mp4` | string | Base64-encoded MP4 video |
| `audio` | string | Base64-encoded WAV audio |
| `gif_path` | string | Container path to GIF file |
| `mp4_path` | string | Container path to MP4 file |
| `audio_path` | string | Container path to audio file |
| `seed` | integer | Seed used for generation |
| `memory_usage` | object | GPU memory statistics |

## 🚀 **Usage Examples**

### 🐳 **Docker Container Usage**

#### Start the Services

```bash
# Web Interface
docker-compose up cartoon-web
# → Access at: http://localhost:7860

# API Server
docker-compose up cartoon-api
# → Access at: http://localhost:8000
```

#### Test with cURL

```bash
# Animation generation
curl -X POST http://localhost:8000/api/animation \
  -H "Content-Type: application/json" \
  -d '{
    "character": "temo",
    "prompt": "temo character exploring alien planet",
    "num_frames": 16,
    "seed": 42
  }'

# TTS generation
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{
    "dialogue_text": "[S1] Hello from space! [S2] Amazing view!",
    "seed": 42
  }'

# Combined generation
curl -X POST http://localhost:8000/api/combined \
  -H "Content-Type: application/json" \
  -d '{
    "character": "felfel",
    "prompt": "felfel character discovering crystal cave",
    "dialogue_text": "[S1] Look at this magical crystal! [S2] Its so beautiful!",
    "seed": 42
  }'
```

### 🐍 **Python Client Usage**

```python
import requests
import base64
import json

# API endpoint
api_url = "http://localhost:8000"

def save_base64_file(base64_data, filename):
    """Save base64 encoded data to file"""
    if base64_data:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(base64_data))
        print(f"✅ Saved {filename}")

def generate_animation(character, prompt, num_frames=16, seed=42):
    """Generate character animation"""
    payload = {
        "character": character,
        "prompt": prompt,
        "num_frames": num_frames,
        "seed": seed
    }
    
    response = requests.post(f"{api_url}/api/animation", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Save files
        save_base64_file(result.get("gif"), f"{character}_animation.gif")
        save_base64_file(result.get("mp4"), f"{character}_animation.mp4")
        
        print(f"🎬 Animation generated with seed: {result.get('seed')}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def generate_tts(dialogue_text, seed=42):
    """Generate text-to-speech"""
    payload = {
        "dialogue_text": dialogue_text,
        "seed": seed
    }
    
    response = requests.post(f"{api_url}/api/tts", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Save audio
        save_base64_file(result.get("audio"), "dialogue.wav")
        
        print(f"🎵 TTS generated with seed: {result.get('seed')}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

def generate_combined(character, prompt, dialogue_text, seed=42):
    """Generate combined animation and TTS"""
    payload = {
        "character": character,
        "prompt": prompt,
        "dialogue_text": dialogue_text,
        "num_frames": 16,
        "seed": seed
    }
    
    response = requests.post(f"{api_url}/api/combined", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        
        # Save all files
        save_base64_file(result.get("gif"), f"{character}_combined.gif")
        save_base64_file(result.get("mp4"), f"{character}_combined.mp4")
        save_base64_file(result.get("audio"), f"{character}_dialogue.wav")
        
        print(f"🎬🎵 Combined generation with seed: {result.get('seed')}")
        return result
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    # Generate Temo animation
    generate_animation(
        character="temo",
        prompt="temo character walking on moon surface, space suit, lunar landscape",
        num_frames=16,
        seed=42
    )
    
    # Generate TTS
    generate_tts(
        dialogue_text="[S1] Welcome to the moon base! [S2] This is amazing!",
        seed=42
    )
    
    # Generate combined
    generate_combined(
        character="felfel",
        prompt="felfel character discovering magical portal",
        dialogue_text="[S1] Look at this portal! [S2] Where does it lead?",
        seed=84
    )
```

### 🌐 **Web Interface Usage**

1. **Start Web Interface**:
   ```bash
   docker-compose up cartoon-web
   ```

2. **Open Browser**: Navigate to `http://localhost:7860`

3. **Use Interface**:
   - **Animation Tab**: Generate character animations
   - **TTS Tab**: Generate speech from text
   - **Combined Tab**: Generate both animation and speech

## 🎭 **Character Guidelines**

### Available Characters

#### **Temo** (Space Explorer)
- **Best Prompts**: Space themes, moon exploration, sci-fi adventures
- **Examples**:
  - `"temo character walking on moon surface, space suit, lunar landscape"`
  - `"temo character floating in space station, zero gravity"`
  - `"temo character discovering alien artifacts"`

#### **Felfel** (Adventure Seeker)
- **Best Prompts**: Fantasy themes, magical adventures, exploration
- **Examples**:
  - `"felfel character exploring magical forest, fantasy adventure"`
  - `"felfel character discovering crystal cave, magical lighting"`
  - `"felfel character climbing mountain peak, epic adventure"`

### Prompt Writing Tips

✅ **Good Prompts**:
- Be specific about actions: "walking", "running", "waving"
- Include environment: "on moon surface", "in magical forest"
- Add style keywords: "detailed cartoon style", "smooth animation"
- Mention character by name: "temo character" or "felfel character"

❌ **Avoid**:
- Vague descriptions: "character doing something"
- Complex scenes with multiple characters
- Realistic or photographic styles
- Violent or inappropriate content

## 🎵 **TTS Guidelines**

### Speaker Tags

Use `[S1]` and `[S2]` to indicate different speakers:

```
[S1] Hello, I'm the first speaker.
[S2] And I'm the second speaker!
[S1] We can have a conversation like this.
```

### Non-Verbal Sounds

Include natural sounds for more realistic speech:

```
[S1] Wow, this is amazing! (laughs)
[S2] I know, right? (gasps) Look at that!
[S1] (whispers) This is incredible.
```

Supported non-verbals:
- `(laughs)`, `(chuckle)`, `(giggles)`
- `(gasps)`, `(sighs)`, `(whispers)`
- `(coughs)`, `(clears throat)`, `(sniffs)`

### Text Length Guidelines

- **Minimum**: 10-15 words (too short sounds unnatural)
- **Optimal**: 20-100 words (best quality)
- **Maximum**: 200 words (may sound rushed)

## ⚙️ **Performance Guidelines**

### Recommended Settings

#### **Fast Generation** (Testing)
```json
{
  "num_frames": 8,
  "num_inference_steps": 10,
  "width": 256,
  "height": 256,
  "max_new_tokens": 1024
}
```

#### **Balanced Quality** (Default)
```json
{
  "num_frames": 16,
  "num_inference_steps": 15,
  "width": 512,
  "height": 512,
  "max_new_tokens": 3072
}
```

#### **High Quality** (Slow)
```json
{
  "num_frames": 24,
  "num_inference_steps": 20,
  "width": 512,
  "height": 512,
  "max_new_tokens": 4096
}
```

### Memory Considerations

| Setting | VRAM Usage | Generation Time |
|---------|------------|-----------------|
| 8 frames, 256x256 | ~3GB | 15-30 seconds |
| 16 frames, 512x512 | ~6GB | 30-60 seconds |
| 24 frames, 512x512 | ~8GB | 60-120 seconds |

## 🔧 **Troubleshooting**

### Common Issues

#### **"Character not found"**
- Ensure character name is exactly `"temo"` or `"felfel"`
- Check that LoRA models are properly mounted in container

#### **"CUDA out of memory"**
- Reduce `num_frames` (try 8-12)
- Reduce resolution (`width: 256, height: 256`)
- Reduce `num_inference_steps` (try 10-12)

#### **"Generation takes too long"**
- Use recommended "Fast Generation" settings
- Ensure GPU is properly configured
- Check Docker GPU access with `nvidia-smi`

#### **"Poor quality output"**
- Increase `num_inference_steps` (15-20)
- Improve prompt specificity
- Use higher resolution (512x512)
- Adjust `guidance_scale` (7.5-10.0)

### Error Response Format

```json
{
  "error": "Error message describing what went wrong",
  "task_type": "animation",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🎉 **Ready to Create!**

Your cartoon animation system is ready to generate amazing content:

- ✅ **Input formats** clearly defined
- ✅ **Output formats** with base64 encoding
- ✅ **Docker integration** for easy deployment
- ✅ **Multiple interfaces** (Web + API)
- ✅ **Character customization** with LoRA weights
- ✅ **Performance optimization** guidelines

Start creating your cartoon animations with the examples above! 🚀🎬🎵 