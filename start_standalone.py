#!/usr/bin/env python3
"""
Standalone Launcher for Cartoon Animation Studio
Starts the system without any RunPod connections or external dependencies
"""

import os
import sys
import subprocess

def main():
    print("🎬 Cartoon Animation Studio - Standalone Mode")
    print("=" * 50)
    
    # Set environment variables to prevent RunPod connections
    os.environ["RUNPOD_STANDALONE_MODE"] = "true"
    os.environ["STANDALONE_WORKER"] = "true"
    os.environ["RUNPOD_DISABLE"] = "true"
    os.environ["LOCAL_DEVELOPMENT"] = "true"
    
    print("🔧 Environment configured for standalone operation")
    print("🚫 RunPod connections disabled")
    print("✅ Starting web interface...")
    print()
    
    # Start the web interface in standalone mode
    try:
        cmd = [
            sys.executable, "web_interface.py",
            "--host", "0.0.0.0",
            "--port", "7860"
        ]
        
        print("🚀 Launching Cartoon Animation Web Interface...")
        print("📡 Interface will be available at: http://localhost:7860")
        print("🎭 Multi-character support enabled!")
        print("💡 No RunPod connections will be made")
        print()
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Standalone mode stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting standalone mode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 