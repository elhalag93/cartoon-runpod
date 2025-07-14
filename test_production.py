#!/usr/bin/env python3
"""
Production Testing Script for Cartoon Animation Worker
Tests the handler with real production inputs and validates responses

This script validates that the system can handle:
- Animation generation with high quality settings
- TTS generation with professional audio
- Combined animation + TTS workflows
- Edge cases and error handling

Usage:
    python test_production.py                    # Run all tests
    python test_production.py --test animation   # Run only animation tests
    python test_production.py --test tts         # Run only TTS tests
    python test_production.py --test combined    # Run only combined tests
    python test_production.py --quick            # Run quick tests only
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

def load_test_cases() -> Dict[str, Any]:
    """Load test cases from test_input.json"""
    test_file = Path("test_input.json")
    if not test_file.exists():
        raise FileNotFoundError(f"Test input file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        return json.load(f)

def validate_response(result: Dict[str, Any], expected_task_type: str) -> bool:
    """Validate that response has expected format"""
    if not isinstance(result, dict):
        print(f"❌ Response is not a dictionary: {type(result)}")
        return False
    
    if "error" in result:
        print(f"❌ Error in response: {result['error']}")
        return False
    
    # In CI environment, check for mock response
    if result.get("ci_test", False):
        if result.get("status") == "success":
            print("✅ CI test passed - Handler validates inputs correctly")
            print(f"   Task Type: {result.get('task_type', 'N/A')}")
            print(f"   Validated: {result.get('validated_input', False)}")
            return True
        else:
            print(f"❌ CI test failed: {result}")
            return False
    
    # Production environment validation
    required_fields = ["task_type", "seed", "generation_time", "memory_usage"]
    for field in required_fields:
        if field not in result:
            print(f"❌ Missing required field: {field}")
            return False
    
    if result["task_type"] != expected_task_type:
        print(f"❌ Task type mismatch: expected {expected_task_type}, got {result['task_type']}")
        return False
    
    # Check for output files based on task type
    if expected_task_type in ["animation", "combined"]:
        if "gif" not in result and "gif_path" not in result:
            print("❌ Missing animation output (gif)")
            return False
        if "mp4" not in result and "mp4_path" not in result:
            print("❌ Missing animation output (mp4)")
            return False
    
    if expected_task_type in ["tts", "combined"]:
        if "audio" not in result and "audio_path" not in result:
            print("❌ Missing audio output")
            return False
    
    print("✅ Response validation passed")
    return True

def run_test_case(test_name: str, test_data: Dict[str, Any]) -> bool:
    """Run a single test case"""
    print(f"\n🧪 Running test: {test_name}")
    print("=" * 50)
    
    try:
        # Import handler
        from handler import generate_cartoon
        
        # Prepare job
        job = {
            "input": test_data["input"],
            "id": f"test-{test_name}-{int(time.time())}"
        }
        
        # Display test details
        task_type = test_data["input"]["task_type"]
        print(f"📋 Test Details:")
        print(f"   Task Type: {task_type}")
        
        if "character" in test_data["input"]:
            print(f"   Character: {test_data['input']['character']}")
        
        if "prompt" in test_data["input"]:
            prompt = test_data["input"]["prompt"]
            print(f"   Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        
        if "dialogue_text" in test_data["input"]:
            dialogue = test_data["input"]["dialogue_text"]
            print(f"   Dialogue: {dialogue[:60]}{'...' if len(dialogue) > 60 else ''}")
        
        if "num_frames" in test_data["input"]:
            print(f"   Frames: {test_data['input']['num_frames']}")
        
        if "width" in test_data["input"] and "height" in test_data["input"]:
            print(f"   Resolution: {test_data['input']['width']}x{test_data['input']['height']}")
        
        # Run generation
        print(f"\n🚀 Starting generation...")
        start_time = time.time()
        
        result = generate_cartoon(job)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        
        # Validate response
        if validate_response(result, task_type):
            print(f"✅ Test '{test_name}' PASSED")
            
            # Display additional info if available
            if "generation_time" in result:
                print(f"   Generation time: {result['generation_time']} seconds")
            
            if "memory_usage" in result:
                memory = result["memory_usage"]
                print(f"   Memory usage: {memory.get('allocated_gb', 0):.1f}GB / {memory.get('total_gb', 0):.1f}GB")
            
            if "seed" in result:
                print(f"   Seed: {result['seed']}")
            
            return True
        else:
            print(f"❌ Test '{test_name}' FAILED - Response validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test '{test_name}' FAILED - Exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Production testing for cartoon animation worker")
    parser.add_argument("--test", choices=["animation", "tts", "combined", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests (minimum viable)")
    parser.add_argument("--ci", action="store_true",
                       help="Run embedded CI tests")
    
    args = parser.parse_args()
    
    print("🎬 CARTOON ANIMATION PRODUCTION TESTING")
    print("=" * 60)
    
    try:
        # Load test cases
        test_data = load_test_cases()
        
        if args.ci:
            # Run embedded CI tests
            print("🔄 Running embedded CI tests...")
            test_cases = test_data.get("embedded_ci_tests", [])
            tests_to_run = [(test["name"], test) for test in test_cases]
        else:
            # Run production tests
            production_tests = test_data.get("production_tests", {})
            
            if args.quick:
                # Run only quick tests
                tests_to_run = [
                    ("minimum_viable", production_tests.get("minimum_viable")),
                    ("quick_test", production_tests.get("quick_test"))
                ]
                tests_to_run = [(name, test) for name, test in tests_to_run if test is not None]
            elif args.test == "animation":
                # Run animation tests only
                tests_to_run = [
                    ("high_quality_animation", production_tests.get("high_quality_animation")),
                    ("standard_animation", production_tests.get("standard_animation")),
                    ("minimum_viable", production_tests.get("minimum_viable"))
                ]
                tests_to_run = [(name, test) for name, test in tests_to_run if test is not None]
            elif args.test == "tts":
                # Run TTS tests only
                tests_to_run = [
                    ("premium_tts", production_tests.get("premium_tts"))
                ]
                tests_to_run = [(name, test) for name, test in tests_to_run if test is not None]
            elif args.test == "combined":
                # Run combined tests only
                tests_to_run = [
                    ("ultra_quality_combined", production_tests.get("ultra_quality_combined")),
                    ("quick_test", production_tests.get("quick_test"))
                ]
                tests_to_run = [(name, test) for name, test in tests_to_run if test is not None]
            else:
                # Run all tests
                tests_to_run = [(name, test) for name, test in production_tests.items()]
        
        if not tests_to_run:
            print("❌ No test cases found!")
            return 1
        
        print(f"📋 Found {len(tests_to_run)} test cases")
        
        # Run tests
        passed = 0
        failed = 0
        
        for test_name, test_data in tests_to_run:
            if test_data is None:
                print(f"⚠️ Skipping {test_name} - test data not found")
                continue
                
            if run_test_case(test_name, test_data):
                passed += 1
            else:
                failed += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 PRODUCTION TEST RESULTS")
        print("=" * 60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {passed + failed}")
        
        if failed == 0:
            print("\n🎉 ALL PRODUCTION TESTS PASSED!")
            print("✅ System is ready for production deployment")
            print("🚀 Ready for RunPod with confidence!")
            return 0
        else:
            print(f"\n⚠️ {failed} tests failed - please review and fix issues")
            return 1
            
    except Exception as e:
        print(f"❌ Test runner failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 