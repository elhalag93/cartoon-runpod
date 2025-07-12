#!/usr/bin/env python3
"""
Create a GitHub release for the cartoon animation worker
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def get_version():
    """Get version from pyproject.toml"""
    try:
        with open("pyproject.toml", "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line.startswith("version ="):
                    return line.split('"')[1]
    except Exception:
        pass
    return "1.0.0"

def create_release_notes(version):
    """Create release notes"""
    notes = f"""# Cartoon Animation Worker v{version}

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
your-dockerhub-username/cartoon-animation:{version}

# Set container command to:
python src/handler.py  # For worker mode
python launch.py web   # For web interface
python launch.py api   # For API server
```

### Local Docker
```bash
docker run -p 7860:7860 --gpus all your-dockerhub-username/cartoon-animation:{version} web
```

## 📋 Input Format
```json
{{
  "input": {{
    "task_type": "combined",
    "character": "temo",
    "prompt": "temo character walking on moon surface",
    "dialogue_text": "[S1] Hello from the moon!",
    "num_frames": 16,
    "seed": 42
  }}
}}
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
"""
    return notes

def main():
    parser = argparse.ArgumentParser(description="Create GitHub release")
    parser.add_argument("--version", help="Version to release (auto-detected if not provided)")
    parser.add_argument("--tag", help="Git tag to create (defaults to v{version})")
    parser.add_argument("--draft", action="store_true", help="Create as draft release")
    parser.add_argument("--prerelease", action="store_true", help="Mark as prerelease")
    
    args = parser.parse_args()
    
    # Get version
    version = args.version or get_version()
    tag = args.tag or f"v{version}"
    
    print(f"Creating release {tag} (version {version})")
    
    # Create release notes
    notes = create_release_notes(version)
    
    # Write to file
    notes_file = Path("release_notes.md")
    with open(notes_file, "w", encoding="utf-8") as f:
        f.write(notes)
    
    print(f"Release notes written to {notes_file}")
    print("\nNext steps:")
    print("1. Review the release notes")
    print("2. Commit and push any final changes")
    print("3. Create the release on GitHub:")
    print(f"   - Tag: {tag}")
    print(f"   - Title: Cartoon Animation Worker {version}")
    print(f"   - Body: Copy from {notes_file}")
    print("4. GitHub Actions will automatically build and push the Docker image")

if __name__ == "__main__":
    main() 