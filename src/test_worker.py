"""
Test script for the cartoon animation RunPod worker
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from handler import handler

def test_animation_only():
    """Test animation generation only"""
    test_input = {
        "input": {
            "task_type": "animation",
            "character": "temo",
            "prompt": "temo character walking on moon surface, cartoon style",
            "num_frames": 8,  # Smaller for faster testing
            "fps": 8,
            "guidance_scale": 7.5,
            "num_inference_steps": 10,  # Fewer steps for faster testing
            "seed": 42
        }
    }
    
    print("🧪 Testing animation generation...")
    result = handler(test_input)
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False
    else:
        print("✅ Animation test passed!")
        print(f"Generated files: {result.get('gif_path')}, {result.get('mp4_path')}")
        return True

def test_tts_only():
    """Test TTS generation only"""
    test_input = {
        "input": {
            "task_type": "tts",
            "dialogue_text": "[S1] Hello, this is a test of the TTS system. [S2] It sounds pretty good!",
            "max_new_tokens": 1024,  # Smaller for faster testing
            "tts_guidance_scale": 3.0,
            "temperature": 1.8
        }
    }
    
    print("🧪 Testing TTS generation...")
    result = handler(test_input)
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False
    else:
        print("✅ TTS test passed!")
        print(f"Generated audio: {result.get('audio_path')}")
        return True

def test_combined():
    """Test combined animation + TTS generation"""
    test_input = {
        "input": {
            "task_type": "combined",
            "character": "felfel",
            "prompt": "felfel character exploring moon, cartoon style",
            "dialogue_text": "[S1] Felfel is on the moon! [S2] What an adventure!",
            "num_frames": 8,
            "fps": 8,
            "guidance_scale": 7.5,
            "num_inference_steps": 10,
            "seed": 84,
            "max_new_tokens": 1024,
            "tts_guidance_scale": 3.0,
            "temperature": 1.8
        }
    }
    
    print("🧪 Testing combined generation...")
    result = handler(test_input)
    
    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False
    else:
        print("✅ Combined test passed!")
        print(f"Generated animation: {result.get('gif_path')}, {result.get('mp4_path')}")
        print(f"Generated audio: {result.get('audio_path')}")
        return True

def main():
    """Run all tests"""
    print("🚀 Starting cartoon animation worker tests...")
    print("=" * 50)
    
    tests = [
        ("Animation Only", test_animation_only),
        ("TTS Only", test_tts_only),
        ("Combined", test_combined)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("❌ Some tests failed. Check the logs above.")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main()) 