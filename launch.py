#!/usr/bin/env python3
"""
Launcher Script for Cartoon Animation Studio
Choose between Web Interface, API Server, or Standalone Worker
"""

import argparse
import sys
import subprocess
import os
import signal
import time
from pathlib import Path

def kill_runpod_processes():
    """Kill any existing RunPod processes"""
    try:
        subprocess.run(["pkill", "-f", "aiapi"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "runpod"], stderr=subprocess.DEVNULL)
        time.sleep(1)
        print("🔪 Terminated any existing RunPod processes")
    except Exception:
        pass

def launch_web_interface(host="0.0.0.0", port=7860, share=False, debug=False):
    """Launch the Gradio web interface"""
    print("🚀 Starting Cartoon Animation Web Interface...")
    
    # Kill any RunPod processes first
    kill_runpod_processes()
    
    # Set environment variable to prevent RunPod connections
    env = os.environ.copy()
    env["RUNPOD_STANDALONE_MODE"] = "true"
    env["STANDALONE_WORKER"] = "true"
    env["RUNPOD_DISABLE"] = "true"
    env["LOCAL_DEVELOPMENT"] = "true"
    
    cmd = [
        sys.executable, "web_interface.py",
        "--host", host,
        "--port", str(port)
    ]
    
    if share:
        cmd.append("--share")
    if debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web interface: {e}")

def launch_api_server(host="0.0.0.0", port=8000, reload=False):
    """Launch the FastAPI server"""
    print("🚀 Starting Cartoon Animation API Server...")
    
    # Kill any RunPod processes first
    kill_runpod_processes()
    
    # Set environment variable to prevent RunPod connections
    env = os.environ.copy()
    env["RUNPOD_STANDALONE_MODE"] = "true"
    env["STANDALONE_WORKER"] = "true"
    env["RUNPOD_DISABLE"] = "true"
    env["LOCAL_DEVELOPMENT"] = "true"
    
    cmd = [
        sys.executable, "api_server.py",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API server: {e}")

def launch_standalone_worker():
    """Launch the worker in standalone mode (no RunPod connections)"""
    print("🚀 Starting Cartoon Animation Worker in Standalone Mode...")
    
    # Kill any RunPod processes first
    kill_runpod_processes()
    
    # Set environment variable to prevent RunPod connections
    env = os.environ.copy()
    env["RUNPOD_STANDALONE_MODE"] = "true"
    env["STANDALONE_WORKER"] = "true"
    env["RUNPOD_DISABLE"] = "true"
    env["LOCAL_DEVELOPMENT"] = "true"
    
    cmd = [sys.executable, "-c", """
import os
os.environ["RUNPOD_STANDALONE_MODE"] = "true"
os.environ["STANDALONE_WORKER"] = "true"
os.environ["RUNPOD_DISABLE"] = "true"
os.environ["LOCAL_DEVELOPMENT"] = "true"

from handler import MODELS, generate_cartoon, setup_directories
import json

print("🎬 Cartoon Animation Worker - Standalone Mode")
print("✅ Models loaded and ready")
print("💡 This mode loads models but doesn't start RunPod serverless")
print("📋 Use this for testing the worker without RunPod connections")

# Setup directories
setup_directories()

print("🎉 Worker ready! Models are loaded and available.")
print("💡 To test the worker, import and call generate_cartoon() from another script")

# Keep running
try:
    import time
    while True:
        print("⏰ Worker running... (Ctrl+C to stop)")
        time.sleep(30)
except KeyboardInterrupt:
    print("\\n👋 Worker stopped by user")
"""]
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n👋 Standalone worker stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting standalone worker: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="🎬 Cartoon Animation Studio Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py web                    # Start web interface
  python launch.py api                    # Start API server
  python launch.py standalone             # Start worker without RunPod
  python launch.py web --share            # Start web interface with sharing
  python launch.py api --reload           # Start API server with auto-reload
  python launch.py web --port 8080        # Start web interface on port 8080
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["web", "api", "standalone"],
        help="Launch mode: 'web' for Gradio interface, 'api' for FastAPI server, 'standalone' for worker without RunPod"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0) - for web/api modes only"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default: 7860 for web, 8000 for api) - for web/api modes only"
    )
    
    # Web interface specific options
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio sharing (web mode only)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (web mode only)"
    )
    
    # API server specific options
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (api mode only)"
    )
    
    args = parser.parse_args()
    
    # Set default ports if not specified (for web/api modes)
    if args.mode in ["web", "api"] and args.port is None:
        args.port = 7860 if args.mode == "web" else 8000
    
    # Print banner
    print("=" * 60)
    print("🎬 CARTOON ANIMATION STUDIO")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    
    if args.mode == "web":
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"URL: http://{args.host}:{args.port}")
        print("Features: Interactive web interface with real-time generation")
        if args.share:
            print("Sharing: Enabled (public URL will be generated)")
        print("=" * 60)
        launch_web_interface(args.host, args.port, args.share, args.debug)
    
    elif args.mode == "api":
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"API URL: http://{args.host}:{args.port}")
        print(f"Docs URL: http://{args.host}:{args.port}/docs")
        print("Features: REST API for programmatic access")
        if args.reload:
            print("Auto-reload: Enabled")
        print("=" * 60)
        launch_api_server(args.host, args.port, args.reload)
    
    elif args.mode == "standalone":
        print("Features: Worker with models loaded, no RunPod connections")
        print("Use: Testing and development without external dependencies")
        print("=" * 60)
        launch_standalone_worker()

if __name__ == "__main__":
    main() 