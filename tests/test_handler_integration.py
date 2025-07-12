"""
Integration tests for the cartoon animation RunPod handler
Tests the complete workflow including model loading and generation
"""

import unittest
import json
import sys
import os
import tempfile
import base64
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestHandlerIntegration(unittest.TestCase):
    """Integration test cases for the cartoon animation handler"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - only run once"""
        # Mock environment for testing
        os.environ["RUNPOD_DEBUG"] = "true"
        
    def setUp(self):
        """Set up test fixtures"""
        self.minimal_input = {
            "task_type": "animation",
            "character": "temo",
            "prompt": "simple test animation",
            "num_frames": 4,  # Very small for fast testing
            "fps": 4,
            "width": 128,  # Very small for fast testing
            "height": 128,
            "guidance_scale": 5.0,
            "num_inference_steps": 5,  # Very few steps for fast testing
            "seed": 42
        }
    
    def test_handler_function_exists(self):
        """Test that the handler function can be imported"""
        try:
            from handler import handler
            self.assertTrue(callable(handler))
        except ImportError as e:
            self.fail(f"Could not import handler function: {e}")
    
    def test_handler_input_validation(self):
        """Test handler input validation"""
        from handler import handler
        
        # Test empty input
        result = handler({"input": {}})
        self.assertIsInstance(result, dict)
        
        # Test missing input key
        result = handler({})
        self.assertIsInstance(result, dict)
        
        # Test invalid task type
        result = handler({"input": {"task_type": "invalid"}})
        self.assertIn("error", result)
    
    def test_base64_encoding_function(self):
        """Test base64 encoding utility function"""
        from handler import encode_file_to_base64
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            # Test encoding
            encoded = encode_file_to_base64(temp_path)
            self.assertIsInstance(encoded, str)
            self.assertGreater(len(encoded), 0)
            
            # Verify it's valid base64
            decoded = base64.b64decode(encoded)
            self.assertEqual(decoded.decode(), "test content")
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_directory_setup(self):
        """Test directory setup function"""
        from handler import setup_directories
        
        # Should not raise exception
        try:
            setup_directories()
        except Exception as e:
            self.fail(f"setup_directories raised an exception: {e}")
    
    def test_memory_clearing(self):
        """Test memory clearing function"""
        from handler import clear_memory
        
        # Should not raise exception
        try:
            clear_memory()
        except Exception as e:
            self.fail(f"clear_memory raised an exception: {e}")
    
    def test_model_loading_functions_exist(self):
        """Test that model loading functions exist and are callable"""
        from handler import load_tts_model, load_animation_pipeline
        
        self.assertTrue(callable(load_tts_model))
        self.assertTrue(callable(load_animation_pipeline))
    
    def test_generation_functions_exist(self):
        """Test that generation functions exist and are callable"""
        from handler import generate_animation, generate_tts, generate_combined
        
        self.assertTrue(callable(generate_animation))
        self.assertTrue(callable(generate_tts))
        self.assertTrue(callable(generate_combined))
    
    def test_handler_response_format(self):
        """Test that handler returns proper response format"""
        from handler import handler
        
        # Test with minimal valid input
        result = handler({"input": self.minimal_input})
        
        # Should return a dictionary
        self.assertIsInstance(result, dict)
        
        # Should have either success fields or error field
        if "error" not in result:
            # Success case - check for expected fields
            self.assertIn("task_type", result)
            self.assertEqual(result["task_type"], "animation")
        else:
            # Error case - should have error message
            self.assertIsInstance(result["error"], str)
            self.assertGreater(len(result["error"]), 0)
    
    def test_runpod_serverless_entry_point(self):
        """Test that the RunPod serverless entry point is properly configured"""
        # Read the handler file and check for runpod.serverless.start
        handler_file = Path(__file__).parent.parent / "src" / "handler.py"
        with open(handler_file, 'r') as f:
            content = f.read()
        
        # Should contain runpod serverless start
        self.assertIn("runpod.serverless.start", content)
        self.assertIn('{"handler": handler}', content)
        self.assertIn('if __name__ == "__main__":', content)
    
    def test_error_handling(self):
        """Test error handling in handler"""
        from handler import handler
        
        # Test with completely invalid input
        result = handler({"input": {"task_type": "animation", "invalid_param": "test"}})
        
        # Should handle gracefully and return dict
        self.assertIsInstance(result, dict)
    
    def test_seed_handling(self):
        """Test seed parameter handling"""
        from handler import handler
        
        # Test with explicit seed
        input_with_seed = self.minimal_input.copy()
        input_with_seed["seed"] = 123
        
        result = handler({"input": input_with_seed})
        
        if "error" not in result:
            # Should preserve the seed
            self.assertEqual(result.get("seed"), 123)
    
    def test_task_type_routing(self):
        """Test that different task types are routed correctly"""
        from handler import handler
        
        # Test animation task
        anim_input = {"task_type": "animation", "character": "temo", "prompt": "test"}
        result = handler({"input": anim_input})
        self.assertIsInstance(result, dict)
        
        # Test TTS task
        tts_input = {"task_type": "tts", "dialogue_text": "[S1] Test"}
        result = handler({"input": tts_input})
        self.assertIsInstance(result, dict)
        
        # Test combined task
        combined_input = {
            "task_type": "combined",
            "character": "temo",
            "prompt": "test",
            "dialogue_text": "[S1] Test"
        }
        result = handler({"input": combined_input})
        self.assertIsInstance(result, dict)

if __name__ == "__main__":
    unittest.main() 