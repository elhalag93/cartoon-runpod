#!/usr/bin/env python3
"""
RunPod Entry Point for Cartoon Animation Worker
This file serves as the main entry point for RunPod serverless deployment
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the actual handler
from handler import handler

# Import RunPod
try:
    import runpod
    print("✅ RunPod module loaded successfully")
except ImportError as e:
    print(f"❌ RunPod module not found: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting RunPod Cartoon Animation Worker...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path}")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler}) 