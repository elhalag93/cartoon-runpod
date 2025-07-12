"""
Tests for the cartoon animation RunPod handler
"""

import unittest
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from handler import handler

class TestCartoonAnimationHandler(unittest.TestCase):
    """Test cases for the cartoon animation handler"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_input = {
            "task_type": "animation",
            "character": "temo",
            "prompt": "temo character walking on moon surface, detailed cartoon style",
            "num_frames": 8,  # Small for faster testing
            "fps": 8,
            "width": 256,  # Small for faster testing
            "height": 256,
            "guidance_scale": 7.5,
            "num_inference_steps": 10,  # Few steps for faster testing
            "seed": 42
        }
    
    def test_animation_generation(self):
        """Test animation generation"""
        job = {"input": self.base_input}
        result = handler(job)
        
        # Check that no error occurred
        self.assertNotIn("error", result)
        
        # Check required fields
        self.assertEqual(result["task_type"], "animation")
        self.assertIn("gif", result)
        self.assertIn("mp4", result)
        self.assertIn("seed", result)
        self.assertIn("generation_time", result)
        self.assertIn("memory_usage", result)
        self.assertEqual(result["seed"], 42)
        
        # Check base64 encoded data
        self.assertIsInstance(result["gif"], str)
        self.assertIsInstance(result["mp4"], str)
        self.assertGreater(len(result["gif"]), 0)
        self.assertGreater(len(result["mp4"]), 0)
    
    def test_tts_generation(self):
        """Test TTS generation"""
        tts_input = {
            "task_type": "tts",
            "dialogue_text": "[S1] Hello, this is a test. [S2] Testing TTS generation.",
            "max_new_tokens": 1024,
            "tts_guidance_scale": 3.0,
            "temperature": 1.8,
            "seed": 42
        }
        
        job = {"input": tts_input}
        result = handler(job)
        
        # Check that no error occurred
        self.assertNotIn("error", result)
        
        # Check required fields
        self.assertEqual(result["task_type"], "tts")
        self.assertIn("audio", result)
        self.assertIn("seed", result)
        self.assertIn("generation_time", result)
        self.assertIn("memory_usage", result)
        
        # Check base64 encoded audio
        self.assertIsInstance(result["audio"], str)
        self.assertGreater(len(result["audio"]), 0)
    
    def test_combined_generation(self):
        """Test combined animation + TTS generation"""
        combined_input = {
            "task_type": "combined",
            "character": "felfel",
            "prompt": "felfel character waving hello",
            "dialogue_text": "[S1] Hello everyone! [S2] Nice to meet you!",
            "num_frames": 8,
            "fps": 8,
            "width": 256,
            "height": 256,
            "guidance_scale": 7.5,
            "num_inference_steps": 10,
            "max_new_tokens": 1024,
            "tts_guidance_scale": 3.0,
            "temperature": 1.8,
            "seed": 84
        }
        
        job = {"input": combined_input}
        result = handler(job)
        
        # Check that no error occurred
        self.assertNotIn("error", result)
        
        # Check required fields
        self.assertEqual(result["task_type"], "combined")
        self.assertIn("gif", result)
        self.assertIn("mp4", result)
        self.assertIn("audio", result)
        self.assertIn("seed", result)
        self.assertIn("generation_time", result)
        self.assertIn("memory_usage", result)
        
        # Check all outputs are present
        self.assertGreater(len(result["gif"]), 0)
        self.assertGreater(len(result["mp4"]), 0)
        self.assertGreater(len(result["audio"]), 0)
    
    def test_invalid_task_type(self):
        """Test handling of invalid task type"""
        invalid_input = {
            "task_type": "invalid_task",
            "character": "temo",
            "prompt": "test prompt"
        }
        
        job = {"input": invalid_input}
        result = handler(job)
        
        # Should return error
        self.assertIn("error", result)
        self.assertIn("Unknown task_type", result["error"])
    
    def test_missing_character(self):
        """Test handling of missing character for animation"""
        missing_char_input = {
            "task_type": "animation",
            "prompt": "test prompt without character"
        }
        
        job = {"input": missing_char_input}
        result = handler(job)
        
        # Should handle gracefully (either error or use default)
        # This depends on your handler implementation
        self.assertIsInstance(result, dict)
    
    def test_invalid_character(self):
        """Test handling of invalid character"""
        invalid_char_input = {
            "task_type": "animation",
            "character": "invalid_character",
            "prompt": "test prompt"
        }
        
        job = {"input": invalid_char_input}
        result = handler(job)
        
        # Should return error or handle gracefully
        self.assertIsInstance(result, dict)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test with extreme values
        extreme_input = {
            "task_type": "animation",
            "character": "temo",
            "prompt": "test prompt",
            "num_frames": 100,  # Too many
            "width": 2048,      # Too large
            "height": 2048,     # Too large
            "guidance_scale": 50.0,  # Too high
            "num_inference_steps": 100  # Too many
        }
        
        job = {"input": extreme_input}
        result = handler(job)
        
        # Should either clamp values or return error
        self.assertIsInstance(result, dict)
    
    def test_empty_input(self):
        """Test handling of empty input"""
        job = {"input": {}}
        result = handler(job)
        
        # Should handle gracefully
        self.assertIsInstance(result, dict)
    
    def test_memory_usage_reporting(self):
        """Test that memory usage is reported"""
        job = {"input": self.base_input}
        result = handler(job)
        
        if "memory_usage" in result:
            self.assertIn("allocated_gb", result["memory_usage"])
            self.assertIn("total_gb", result["memory_usage"])
            self.assertIsInstance(result["memory_usage"]["allocated_gb"], (int, float))
            self.assertIsInstance(result["memory_usage"]["total_gb"], (int, float))

if __name__ == "__main__":
    unittest.main() 