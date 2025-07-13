"""
RunPod Serverless Handler for Cartoon Animation Generation
Combines Dia TTS and AnimateDiff for character animations with voice

This handler follows RunPod serverless best practices:
- Proper error handling and validation
- Memory management and optimization
- Base64 encoded outputs for file transfer
- Comprehensive logging and monitoring

Example usage:
    import runpod

    def handler(job):
        job_input = job["input"]
        # Process the input (replace this with your own code)
        result = process_data(job_input)
        return result

    runpod.serverless.start({"handler": handler})
"""

import os
import gc
import time
import json
import base64
import tempfile
import traceback
from typing import Dict, Any, Optional, Union
from pathlib import Path

import runpod  # Required
import torch
import numpy as np
import soundfile as sf
from PIL import Image
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video, export_to_gif
from transformers import AutoProcessor, DiaForConditionalGeneration

# Global model instances - loaded once and reused
dia_model = None
dia_processor = None
animation_pipeline = None
motion_adapter = None

# Configuration - use different paths for local testing vs RunPod
if os.getenv("RUNPOD_POD_ID") or os.getenv("RUNPOD_ENDPOINT_ID"):
    # Running on RunPod
    MODELS_DIR = Path("/workspace/models")
    LORA_DIR = Path("/workspace/lora_models")
    OUTPUT_DIR = Path("/workspace/outputs")
    TEMP_DIR = Path("/workspace/temp")
else:
    # Running locally or in CI
    project_root = Path(__file__).parent.parent
    MODELS_DIR = project_root / "models"
    LORA_DIR = project_root / "lora_models"
    OUTPUT_DIR = project_root / "outputs"
    TEMP_DIR = project_root / "temp"

# Model configuration
DIA_MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"
SDXL_MODEL_ID = "stabilityai/sdxl-turbo"
MOTION_ADAPTER_ID = "animatediff/animatediff-motion-adapter-v1"

# Supported characters
SUPPORTED_CHARACTERS = ["temo", "felfel"]

def setup_directories():
    """Ensure all required directories exist"""
    for directory in [MODELS_DIR, LORA_DIR, OUTPUT_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2)
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "total_gb": 0, "free_gb": 0}

def load_tts_model():
    """Load Dia TTS model and processor"""
    global dia_model, dia_processor
    
    if dia_model is None or dia_processor is None:
        print("🔄 Loading Dia TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Skip model loading in CI/testing environments
        if os.getenv("CI_TESTING") or os.getenv("TESTING"):
            print("⚠️ Skipping TTS model loading in CI/testing environment")
            raise Exception("TTS model loading skipped in CI/testing environment")
        
        try:
            dia_processor = AutoProcessor.from_pretrained(DIA_MODEL_CHECKPOINT)
            dia_model = DiaForConditionalGeneration.from_pretrained(
                DIA_MODEL_CHECKPOINT,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            if device != "cuda":
                dia_model = dia_model.to(device)
            
            print("✅ Dia TTS model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading TTS model: {e}")
            raise
    
    return dia_model, dia_processor

def load_animation_pipeline():
    """Load AnimateDiff pipeline with SDXL"""
    global animation_pipeline, motion_adapter
    
    if animation_pipeline is None:
        print("🔄 Loading AnimateDiff pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Skip model loading in CI/testing environments
        if os.getenv("CI_TESTING") or os.getenv("TESTING"):
            print("⚠️ Skipping model loading in CI/testing environment")
            raise Exception("Model loading skipped in CI/testing environment")
        
        try:
            # Load motion adapter
            motion_adapter = MotionAdapter.from_pretrained(
                MOTION_ADAPTER_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Load pipeline
            animation_pipeline = AnimateDiffSDXLPipeline.from_pretrained(
                SDXL_MODEL_ID,
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True
            )
            
            animation_pipeline = animation_pipeline.to(device)
            
            # Configure scheduler
            animation_pipeline.scheduler = DDIMScheduler.from_config(
                animation_pipeline.scheduler.config,
                beta_schedule="linear",
                steps_offset=1,
                clip_sample=False
            )
            
            # Enable memory optimizations
            if device == "cuda":
                animation_pipeline.enable_sequential_cpu_offload()
                animation_pipeline.enable_attention_slicing()
                animation_pipeline.enable_vae_slicing()
                animation_pipeline.enable_vae_tiling()
                
                # Try to enable xformers if available
                try:
                    animation_pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers memory efficient attention enabled")
                except Exception:
                    print("⚠️ xFormers not available, using default attention")
            
            print("✅ AnimateDiff pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Error loading animation pipeline: {e}")
            raise
    
    return animation_pipeline

def load_lora_weights(pipeline, character: str):
    """Load LoRA weights for specific character"""
    try:
        # Unload any existing LoRA weights
        pipeline.unload_lora_weights()
        clear_memory()
        
        # Load character-specific LoRA
        lora_path = LORA_DIR / f"{character}_lora"
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found for character '{character}' at {lora_path}")
        
        pipeline.load_lora_weights(
            str(lora_path),
            weight_name="deep_sdxl_turbo_lora_weights.pt"
        )
        
        print(f"✅ LoRA weights loaded for character: {character}")
    except Exception as e:
        print(f"❌ Error loading LoRA weights for {character}: {e}")
        raise

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error encoding file {file_path}: {e}")
        return ""

def validate_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize input parameters"""
    validated = {}
    
    # Task type validation
    task_type = input_data.get("task_type", "animation")
    if task_type not in ["animation", "tts", "combined"]:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'animation', 'tts', or 'combined'")
    validated["task_type"] = task_type
    
    # Character validation (for animation tasks)
    if task_type in ["animation", "combined"]:
        character = input_data.get("character", "temo")
        if character not in SUPPORTED_CHARACTERS:
            raise ValueError(f"Invalid character: {character}. Must be one of {SUPPORTED_CHARACTERS}")
        validated["character"] = character
        
        # Prompt validation
        prompt = input_data.get("prompt", "")
        if not prompt:
            validated["prompt"] = f"{character} character in cartoon style"
        else:
            validated["prompt"] = str(prompt)[:500]  # Limit prompt length
    
    # TTS validation (for tts tasks)
    if task_type in ["tts", "combined"]:
        dialogue_text = input_data.get("dialogue_text", "")
        if not dialogue_text:
            raise ValueError("dialogue_text is required for TTS generation")
        validated["dialogue_text"] = str(dialogue_text)[:1000]  # Limit text length
    
    # Animation parameters
    if task_type in ["animation", "combined"]:
        validated["num_frames"] = max(4, min(32, input_data.get("num_frames", 16)))
        validated["fps"] = max(4, min(12, input_data.get("fps", 8)))
        validated["width"] = max(256, min(768, input_data.get("width", 512)))
        validated["height"] = max(256, min(768, input_data.get("height", 512)))
        validated["guidance_scale"] = max(1.0, min(15.0, input_data.get("guidance_scale", 7.5)))
        validated["num_inference_steps"] = max(5, min(30, input_data.get("num_inference_steps", 15)))
        validated["negative_prompt"] = input_data.get("negative_prompt", "blurry, low quality, distorted")
    
    # TTS parameters
    if task_type in ["tts", "combined"]:
        validated["max_new_tokens"] = max(512, min(4096, input_data.get("max_new_tokens", 3072)))
        validated["tts_guidance_scale"] = max(1.0, min(10.0, input_data.get("tts_guidance_scale", 3.0)))
        validated["temperature"] = max(0.1, min(2.0, input_data.get("temperature", 1.8)))
        validated["top_p"] = max(0.1, min(1.0, input_data.get("top_p", 0.9)))
        validated["top_k"] = max(1, min(100, input_data.get("top_k", 45)))
    
    # Seed handling
    seed = input_data.get("seed")
    if seed is not None:
        validated["seed"] = int(seed)
    else:
        validated["seed"] = torch.randint(0, 1000000, (1,)).item()
    
    return validated

def generate_tts(
    dialogue_text: str,
    max_new_tokens: int = 3072,
    tts_guidance_scale: float = 3.0,
    temperature: float = 1.8,
    top_p: float = 0.9,
    top_k: int = 45,
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Generate TTS audio from text"""
    
    # Load TTS model
    model, processor = load_tts_model()
    
    print(f"🎵 Generating TTS for: {dialogue_text[:50]}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    try:
        # Process text
        inputs = processor(text=[dialogue_text], padding=True, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=tts_guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        # Decode audio
        audio_arrays = processor.batch_decode(outputs)
        
        # Save audio
        timestamp = int(time.time())
        audio_path = TEMP_DIR / f"tts_output_{timestamp}.wav"
        
        # Save the first audio array
        if audio_arrays and len(audio_arrays) > 0:
            # Save audio using processor's save method
            processor.save_audio(audio_arrays, str(audio_path))
        
        clear_memory()
        
        return {
            "audio_path": str(audio_path),
            "seed": seed,
            "dialogue_text": dialogue_text
        }
        
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        raise

def generate_animation(
    character: str,
    prompt: str,
    num_frames: int = 16,
    fps: int = 8,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 15,
    negative_prompt: str = "blurry, low quality, distorted",
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Generate character animation"""
    
    # Load animation pipeline
    pipeline = load_animation_pipeline()
    
    # Load character LoRA weights
    load_lora_weights(pipeline, character)
    
    print(f"🎬 Generating animation for {character}: {prompt[:50]}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)
    
    try:
        # Generate animation
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator
        )
        
        frames = result.frames[0]
        
        # Save outputs
        timestamp = int(time.time())
        gif_path = OUTPUT_DIR / f"{character}_animation_{timestamp}.gif"
        mp4_path = OUTPUT_DIR / f"{character}_animation_{timestamp}.mp4"
        
        # Export to files
        export_to_gif(frames, str(gif_path), fps=fps)
        export_to_video(frames, str(mp4_path), fps=fps)
        
        clear_memory()
        
        return {
            "gif_path": str(gif_path),
            "mp4_path": str(mp4_path),
            "seed": seed,
            "character": character,
            "prompt": prompt
        }
        
    except Exception as e:
        print(f"❌ Error generating animation: {e}")
        raise

def generate_combined(
    character: str,
    prompt: str,
    dialogue_text: str,
    **kwargs
) -> Dict[str, Any]:
    """Generate combined animation and TTS"""
    
    print(f"🎬🎵 Generating combined animation + TTS for {character}")
    
    # Extract parameters for each task
    animation_params = {k: v for k, v in kwargs.items() 
                       if k in ["num_frames", "fps", "width", "height", "guidance_scale", 
                               "num_inference_steps", "negative_prompt", "seed"]}
    
    tts_params = {k: v for k, v in kwargs.items() 
                  if k in ["max_new_tokens", "tts_guidance_scale", "temperature", 
                          "top_p", "top_k", "seed"]}
    
    # Generate animation
    animation_result = generate_animation(
        character=character,
        prompt=prompt,
        **animation_params
    )
    
    # Generate TTS
    tts_result = generate_tts(
        dialogue_text=dialogue_text,
        **tts_params
    )
    
    # Combine results
    combined_result = {
        **animation_result,
        **tts_result,
        "task_type": "combined"
    }
    
    return combined_result

def handler(job):
    """
    RunPod serverless handler function
    
    Args:
        job: Job dictionary containing 'input' with generation parameters
        
    Returns:
        Dictionary with generation results or error information
    """
    
    start_time = time.time()
    
    try:
        # Setup directories
        setup_directories()
        
        # Extract and validate input
        input_data = job.get("input", {})
        if not input_data:
            return {"error": "No input data provided"}
        
        # Validate input parameters
        validated_input = validate_input(input_data)
        task_type = validated_input["task_type"]
        
        print(f"🚀 Starting {task_type} generation with validated input")
        
        # Route to appropriate generation function
        if task_type == "animation":
            result = generate_animation(**validated_input)
        elif task_type == "tts":
            result = generate_tts(**validated_input)
        elif task_type == "combined":
            result = generate_combined(**validated_input)
        else:
            return {"error": f"Unknown task_type: {task_type}"}
        
        # Prepare response with base64 encoded files
        response = {
            "task_type": task_type,
            "seed": result.get("seed"),
            "generation_time": round(time.time() - start_time, 2),
            "memory_usage": get_memory_usage()
        }
        
        # Encode files to base64
        if "gif_path" in result:
            response["gif"] = encode_file_to_base64(result["gif_path"])
            response["gif_path"] = result["gif_path"]
        
        if "mp4_path" in result:
            response["mp4"] = encode_file_to_base64(result["mp4_path"])
            response["mp4_path"] = result["mp4_path"]
        
        if "audio_path" in result:
            response["audio"] = encode_file_to_base64(result["audio_path"])
            response["audio_path"] = result["audio_path"]
        
        # Add metadata
        if "character" in result:
            response["character"] = result["character"]
        if "prompt" in result:
            response["prompt"] = result["prompt"]
        if "dialogue_text" in result:
            response["dialogue_text"] = result["dialogue_text"]
        
        # Clean up temporary files
        for path_key in ["gif_path", "mp4_path", "audio_path"]:
            if path_key in result:
                try:
                    os.remove(result[path_key])
                except Exception:
                    pass  # Ignore cleanup errors
        
        print(f"✅ {task_type} generation completed in {response['generation_time']}s")
        return response
        
    except Exception as e:
        error_msg = f"Error in {task_type if 'task_type' in locals() else 'unknown'} generation: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Return detailed error for debugging
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc() if os.getenv("RUNPOD_DEBUG") else None,
            "generation_time": round(time.time() - start_time, 2),
            "memory_usage": get_memory_usage()
        }

# RunPod serverless entry point
if __name__ == "__main__":
    print("🚀 Starting RunPod Cartoon Animation Worker...")
    print(f"📱 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"🔧 PyTorch: {torch.__version__}")
    
    # Initialize directories
    setup_directories()
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})  # Required 