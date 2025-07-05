"""
RunPod Worker for Cartoon Animation Generation
Combines Dia TTS and AnimateDiff for character animations
"""

import io
import os
import time
import json
import tempfile
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path

import runpod
import torch
import numpy as np
import soundfile as sf
from PIL import Image
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video, export_to_gif
from transformers import AutoProcessor, DiaForConditionalGeneration

# Global variables for models
dia_model = None
dia_processor = None
animation_pipeline = None
motion_adapter = None

# Configuration
MODELS_DIR = Path("/workspace/models")
LORA_DIR = Path("/workspace/lora_models")
OUTPUT_DIR = Path("/workspace/outputs")
TEMP_DIR = Path("/workspace/temp")

# Model paths
SDXL_MODEL_PATH = str(MODELS_DIR / "sdxl-turbo")
MOTION_ADAPTER_PATH = str(MODELS_DIR / "animatediff" / "motion_adapter")
DIA_MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"

def setup_directories():
    """Ensure all required directories exist"""
    for directory in [OUTPUT_DIR, TEMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        return allocated, total
    return 0, 0

def load_dia_model():
    """Load Dia TTS model"""
    global dia_model, dia_processor
    
    if dia_model is not None:
        return dia_model, dia_processor
    
    print("Loading Dia TTS model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load using transformers (HuggingFace approach)
        dia_processor = AutoProcessor.from_pretrained(DIA_MODEL_CHECKPOINT)
        dia_model = DiaForConditionalGeneration.from_pretrained(
            DIA_MODEL_CHECKPOINT,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        
        print("✅ Dia TTS model loaded successfully")
        return dia_model, dia_processor
        
    except Exception as e:
        print(f"❌ Error loading Dia model: {e}")
        raise

def load_animation_pipeline():
    """Load AnimateDiff pipeline"""
    global animation_pipeline, motion_adapter
    
    if animation_pipeline is not None:
        return animation_pipeline
    
    print("Loading AnimateDiff pipeline...")
    try:
        # Load motion adapter
        motion_adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER_PATH,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        
        # Create pipeline
        animation_pipeline = AnimateDiffSDXLPipeline.from_pretrained(
            SDXL_MODEL_PATH,
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True
        ).to("cuda")
        
        animation_pipeline.scheduler = DDIMScheduler.from_config(animation_pipeline.scheduler.config)
        
        # Optimize for memory efficiency
        animation_pipeline.enable_sequential_cpu_offload()
        animation_pipeline.enable_attention_slicing("max")
        animation_pipeline.enable_vae_slicing()
        animation_pipeline.enable_vae_tiling()
        
        print("✅ AnimateDiff pipeline loaded successfully")
        return animation_pipeline
        
    except Exception as e:
        print(f"❌ Error loading animation pipeline: {e}")
        raise

def generate_tts(text: str, audio_prompt: Optional[str] = None, **kwargs) -> str:
    """Generate speech using Dia TTS"""
    model, processor = load_dia_model()
    device = next(model.parameters()).device
    
    print(f"Generating TTS for: {text[:50]}...")
    
    try:
        # Process text input
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)
        
        # Generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 3072)
        guidance_scale = kwargs.get("guidance_scale", 3.0)
        temperature = kwargs.get("temperature", 1.8)
        top_p = kwargs.get("top_p", 0.90)
        top_k = kwargs.get("top_k", 45)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        # Decode and save audio
        audio_outputs = processor.batch_decode(outputs)
        
        # Save to temporary file
        timestamp = int(time.time())
        audio_path = TEMP_DIR / f"tts_output_{timestamp}.mp3"
        processor.save_audio(audio_outputs, str(audio_path))
        
        print(f"✅ TTS generated and saved to {audio_path}")
        return str(audio_path)
        
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        raise

def generate_animation(
    character: str,
    prompt: str,
    num_frames: int = 16,
    fps: int = 8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 15,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, str]:
    """Generate character animation"""
    pipeline = load_animation_pipeline()
    
    print(f"Generating animation for {character}...")
    clear_memory()
    
    try:
        # Load character LoRA
        lora_path = LORA_DIR / f"{character}_lora"
        if not lora_path.exists():
            raise ValueError(f"LoRA for character '{character}' not found at {lora_path}")
        
        # Unload any previous LoRA
        try:
            pipeline.unload_lora_weights()
            clear_memory()
        except:
            pass
        
        # Load character LoRA
        pipeline.load_lora_weights(str(lora_path), weight_name="deep_sdxl_turbo_lora_weights.pt")
        
        # Set up generation
        if seed is None:
            seed = torch.randint(0, 1000000, (1,)).item()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Generate animation
        video_frames = pipeline(
            prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=kwargs.get("height", 512),
            width=kwargs.get("width", 512),
            generator=generator
        ).frames[0]
        
        # Save outputs
        timestamp = int(time.time())
        gif_path = OUTPUT_DIR / f"{character}_animation_{timestamp}.gif"
        mp4_path = OUTPUT_DIR / f"{character}_animation_{timestamp}.mp4"
        
        export_to_gif(video_frames, str(gif_path), fps=fps)
        export_to_video(video_frames, str(mp4_path), fps=fps)
        
        clear_memory()
        
        print(f"✅ Animation generated for {character}")
        return {
            "gif_path": str(gif_path),
            "mp4_path": str(mp4_path),
            "seed": seed
        }
        
    except Exception as e:
        print(f"❌ Error generating animation for {character}: {e}")
        raise

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler function"""
    try:
        setup_directories()
        job_input = job.get("input", {})
        
        # Extract parameters
        task_type = job_input.get("task_type", "animation")  # "animation", "tts", or "combined"
        character = job_input.get("character", "temo")
        prompt = job_input.get("prompt", "character walking on moon surface")
        
        # Animation parameters
        num_frames = job_input.get("num_frames", 16)
        fps = job_input.get("fps", 8)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        num_inference_steps = job_input.get("num_inference_steps", 15)
        seed = job_input.get("seed")
        
        # TTS parameters
        dialogue_text = job_input.get("dialogue_text", "")
        max_new_tokens = job_input.get("max_new_tokens", 3072)
        tts_guidance_scale = job_input.get("tts_guidance_scale", 3.0)
        temperature = job_input.get("temperature", 1.8)
        
        result = {"task_type": task_type}
        
        if task_type in ["tts", "combined"]:
            if not dialogue_text:
                return {"error": "dialogue_text is required for TTS generation"}
            
            print("🎵 Generating TTS...")
            audio_path = generate_tts(
                text=dialogue_text,
                max_new_tokens=max_new_tokens,
                guidance_scale=tts_guidance_scale,
                temperature=temperature
            )
            
            result["audio"] = encode_file_to_base64(audio_path)
            result["audio_path"] = audio_path
        
        if task_type in ["animation", "combined"]:
            print("🎬 Generating animation...")
            animation_result = generate_animation(
                character=character,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            
            result.update({
                "gif": encode_file_to_base64(animation_result["gif_path"]),
                "mp4": encode_file_to_base64(animation_result["mp4_path"]),
                "gif_path": animation_result["gif_path"],
                "mp4_path": animation_result["mp4_path"],
                "seed": animation_result["seed"]
            })
        
        # Memory cleanup
        clear_memory()
        allocated, total = check_gpu_memory()
        result["memory_usage"] = {"allocated_gb": allocated, "total_gb": total}
        
        return result
        
    except Exception as e:
        print(f"❌ Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# RunPod serverless handler
runpod.serverless.start({"handler": handler}) 