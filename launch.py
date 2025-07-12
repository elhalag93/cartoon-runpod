#!/usr/bin/env python3
"""
Launcher Script for Cartoon Animation Studio
Choose between Web Interface or API Server
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path

def launch_web_interface(host="0.0.0.0", port=7860, share=False, debug=False):
    """Launch the Gradio web interface"""
    print("🚀 Starting Cartoon Animation Web Interface...")
    
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
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web interface: {e}")

def launch_api_server(host="0.0.0.0", port=8000, reload=False):
    """Launch the FastAPI server"""
    print("🚀 Starting Cartoon Animation API Server...")
    
    cmd = [
        sys.executable, "api_server.py",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting API server: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="🎬 Cartoon Animation Studio Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py web                    # Start web interface
  python launch.py api                    # Start API server
  python launch.py web --share            # Start web interface with sharing
  python launch.py api --reload           # Start API server with auto-reload
  python launch.py web --port 8080        # Start web interface on port 8080
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["web", "api"],
        help="Launch mode: 'web' for Gradio interface, 'api' for FastAPI server"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default: 7860 for web, 8000 for api)"
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
    
    # Set default ports if not specified
    if args.port is None:
        args.port = 7860 if args.mode == "web" else 8000
    
    # Print banner
    print("=" * 60)
    print("🎬 CARTOON ANIMATION STUDIO")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    
    if args.mode == "web":
        print(f"URL: http://{args.host}:{args.port}")
        print("Features: Interactive web interface with real-time generation")
        if args.share:
            print("Sharing: Enabled (public URL will be generated)")
        print("=" * 60)
        launch_web_interface(args.host, args.port, args.share, args.debug)
    
    elif args.mode == "api":
        print(f"API URL: http://{args.host}:{args.port}")
        print(f"Docs URL: http://{args.host}:{args.port}/docs")
        print("Features: REST API for programmatic access")
        if args.reload:
            print("Auto-reload: Enabled")
        print("=" * 60)
        launch_api_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main() 