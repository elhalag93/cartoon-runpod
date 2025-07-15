#!/usr/bin/env python3
"""
Test script to verify Docker image builds and basic functionality works
"""

import sys
import subprocess
import time

def test_docker_build():
    """Test if Docker image can be built"""
    print("🐳 Testing Docker image build...")
    
    try:
        # Build the Docker image
        result = subprocess.run([
            "docker", "build", "-t", "cartoon-test", "."
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("✅ Docker image built successfully!")
            return True
        else:
            print("❌ Docker build failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def test_docker_run():
    """Test if Docker image can run basic imports"""
    print("🧪 Testing Docker image imports...")
    
    try:
        # Test basic Python imports
        result = subprocess.run([
            "docker", "run", "--rm", "cartoon-test", 
            "python", "-c", 
            "import torch; import transformers; import diffusers; print('✅ All imports successful')"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Docker image imports work!")
            print(result.stdout)
            return True
        else:
            print("❌ Docker imports failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker run timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Docker run error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Docker Image Testing")
    print("=" * 40)
    
    # Test build
    if not test_docker_build():
        print("❌ Docker build failed - stopping tests")
        return 1
    
    # Test run
    if not test_docker_run():
        print("❌ Docker run failed")
        return 1
    
    print("\n🎉 All Docker tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 