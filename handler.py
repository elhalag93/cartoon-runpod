#!/usr/bin/env python3
"""
RunPod Serverless Handler for Cartoon Animation Generation
This is the main entry point for RunPod deployment
"""

import os
import gc
import time
import json
import base64
import tempfile
import traceback
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

try:
    import runpod
    from runpod.serverless.utils import rp_cleanup
except ImportError:
    print("⚠️ RunPod module not available - this is expected in local development")
    runpod = None

import torch
import numpy as np
import soundfile as sf
from PIL import Image
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video, export_to_gif
from transformers import AutoProcessor, DiaForConditionalGeneration

# Clear GPU cache at startup
torch.cuda.empty_cache()

# Configuration - use safe paths for CI environment
if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
    # Use current directory for CI (safe and writable)
    project_root = Path.cwd()
    MODELS_DIR = project_root / "models"
    LORA_DIR = project_root / "lora_models"
    OUTPUT_DIR = project_root / "outputs"
    TEMP_DIR = project_root / "temp"
else:
    # Use workspace paths for production
    MODELS_DIR = Path("/workspace/models")
    LORA_DIR = Path("/workspace/lora_models")
    OUTPUT_DIR = Path("/workspace/outputs")
    TEMP_DIR = Path("/workspace/temp")

# Model configuration - HIGHEST QUALITY MODELS
DIA_MODEL_CHECKPOINT = "nari-labs/Dia-1.6B-0626"  # Keep the best TTS model
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # Full SDXL for maximum quality
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"  # Stable non-beta version
CONTROLNET_MODEL_ID = "diffusers/controlnet-openpose-sdxl-1.0"  # For better character consistency

# ULTRA HIGH QUALITY Generation Settings - 1024x1024
HIGH_QUALITY_DEFAULTS = {
    "num_frames": 32,           # Maximum frames for ultra smooth motion
    "fps": 16,                  # Highest framerate for ultra fluidity
    "width": 1024,              # Ultra high resolution
    "height": 1024,             # Ultra high resolution
    "guidance_scale": 12.0,     # Maximum guidance for perfect prompt following
    "num_inference_steps": 50,  # Maximum steps for ultimate quality
    "tts_guidance_scale": 5.0,  # Maximum TTS guidance
    "max_new_tokens": 4096,     # Maximum tokens for best TTS quality
    "temperature": 1.4,         # Optimized for consistency
}

# Supported characters
SUPPORTED_CHARACTERS = ["temo", "felfel"]

def setup_directories():
    """Ensure all required directories exist"""
    # Skip directory creation in CI environment
    if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
        print("⚠️ CI environment detected - skipping directory creation")
        return
    
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

def verify_gpu_setup():
    """Verify GPU is available and being used"""
    if not torch.cuda.is_available():
        # In CI/testing environment, just warn but don't fail
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            print("⚠️ WARNING: Running in CI environment without GPU - this is expected")
            return False
        else:
            raise RuntimeError("🚨 CRITICAL: CUDA/GPU not available! This will be extremely slow on CPU!")
    
    gpu_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"🎮 GPU DETECTED: {gpu_name}")
    print(f"💾 GPU Memory: {memory_gb:.1f}GB")
    print(f"🔥 CUDA Version: {torch.version.cuda}")
    
    if memory_gb < 8:
        print("⚠️ WARNING: GPU has less than 8GB VRAM - may run out of memory")
    elif memory_gb >= 16:
        print("✅ EXCELLENT: GPU has 16GB+ VRAM - perfect for RunPod deployment")
    else:
        print("✅ GOOD: GPU has sufficient VRAM for generation")
    
    return True

class ModelHandler:
    """Model handler following RunPod best practices"""
    
    def __init__(self):
        self.dia_model = None
        self.dia_processor = None
        self.animation_pipeline = None
        self.motion_adapter = None
        self.load_models()
    
    def load_tts_model(self):
        """Load Dia TTS model and processor"""
        print("🔄 Loading Dia TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.dia_processor = AutoProcessor.from_pretrained(DIA_MODEL_CHECKPOINT)
            self.dia_model = DiaForConditionalGeneration.from_pretrained(
                DIA_MODEL_CHECKPOINT,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            if device != "cuda":
                self.dia_model = self.dia_model.to(device)
            
            print("✅ Dia TTS model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading TTS model: {e}")
            raise
        
        return self.dia_model, self.dia_processor
    
    def load_animation_pipeline(self):
        """Load HIGH QUALITY AnimateDiff pipeline with full SDXL"""
        print("🔄 Loading HIGH QUALITY AnimateDiff pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load motion adapter (stable version)
            print("📥 Loading stable motion adapter...")
            self.motion_adapter = MotionAdapter.from_pretrained(
                MOTION_ADAPTER_ID,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Load FULL SDXL AnimateDiff pipeline for maximum quality
            print("🔄 Creating HIGH QUALITY AnimateDiff pipeline with full SDXL...")
            self.animation_pipeline = AnimateDiffSDXLPipeline.from_pretrained(
                SDXL_MODEL_ID,
                motion_adapter=self.motion_adapter,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=True
            )
            
            self.animation_pipeline = self.animation_pipeline.to(device)
            
            # Configure scheduler for MAXIMUM QUALITY
            self.animation_pipeline.scheduler = DDIMScheduler.from_config(
                self.animation_pipeline.scheduler.config,
                beta_schedule="scaled_linear",  # Better for quality
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
                skip_prk_steps=True,
                rescale_betas_zero_snr=True  # Improved quality
            )
            
            # Quality-focused optimizations for HIGH-END GPU
            if device == "cuda":
                try:
                    self.animation_pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers memory efficient attention enabled")
                except Exception:
                    print("⚠️ xFormers not available, using default attention")
                
                # QUALITY-FIRST GPU optimizations 
                # Keep models on GPU, use minimal memory optimizations
                self.animation_pipeline.enable_vae_slicing()
                self.animation_pipeline.enable_vae_tiling()
                
                # Only use attention slicing if necessary (high VRAM usage)
                # self.animation_pipeline.enable_attention_slicing("max")  # Comment out for max quality
                
                print("🚀 HIGH QUALITY mode enabled - maximum quality generation")
            
            print("✅ HIGH QUALITY AnimateDiff pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Error loading animation pipeline: {e}")
            raise
        
        return self.animation_pipeline
    
    def load_models(self):
        """Load all models at startup"""
        # VERIFY GPU IS AVAILABLE FIRST
        gpu_available = verify_gpu_setup()
        
        # Skip model loading only in CI to save time and resources
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            print("⚠️ CI environment detected - skipping model loading (models will load in production)")
            return
        
        print("🚀 Loading models on GPU for maximum performance...")
        self.load_tts_model()
        self.load_animation_pipeline()
        
        # Verify models are on GPU
        if torch.cuda.is_available():
            print(f"🎮 TTS Model device: {next(self.dia_model.parameters()).device}")
            print(f"🎮 Animation Pipeline device: {self.animation_pipeline.device}")
            print("✅ ALL MODELS LOADED ON GPU - READY FOR FAST GENERATION!")

# Initialize models globally (following working example pattern)
MODELS = ModelHandler()

def load_lora_weights(pipeline, characters: Union[str, List[str]]):
    """Load LoRA weights for single or multiple characters"""
    try:
        # Unload any existing LoRA weights
        pipeline.unload_lora_weights()
        clear_memory()
        
        # Handle single character (backward compatibility)
        if isinstance(characters, str):
            characters = [characters]
        
        # Load multiple character LoRAs
        lora_weights = {}
        adapter_names = []
        
        for character in characters:
            lora_path = LORA_DIR / f"{character}_lora"
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA weights not found for character '{character}' at {lora_path}")
            
            # Load each character's LoRA with unique adapter name
            adapter_name = f"{character}_adapter"
            pipeline.load_lora_weights(
                str(lora_path),
                weight_name="deep_sdxl_turbo_lora_weights.pt",
                adapter_name=adapter_name
            )
            adapter_names.append(adapter_name)
            lora_weights[character] = adapter_name
            print(f"✅ LoRA weights loaded for character: {character} (adapter: {adapter_name})")
        
        # If multiple characters, set adapter weights for blending
        if len(characters) > 1:
            # Equal weight blending for balanced representation
            adapter_weights = [1.0 / len(characters)] * len(characters)
            pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
            print(f"🎭 Multi-character mode: {characters} with equal blending ({adapter_weights})")
        else:
            # Single character mode
            pipeline.set_adapters(adapter_names[0])
            print(f"🎭 Single character mode: {characters[0]}")
        
        return lora_weights
        
    except Exception as e:
        print(f"❌ Error loading LoRA weights for {characters}: {e}")
        raise

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error encoding file {file_path}: {e}")
        return ""

def _save_and_upload_files(result, job_id):
    """Save and upload generated files, following working example pattern"""
    os.makedirs(f"/{job_id}", exist_ok=True)
    response = {}
    
    # Handle GIF
    if "gif_path" in result:
        gif_local_path = os.path.join(f"/{job_id}", "animation.gif")
        os.rename(result["gif_path"], gif_local_path)
        
        with open(gif_local_path, "rb") as f:
            gif_data = base64.b64encode(f.read()).decode("utf-8")
            response["gif"] = f"data:image/gif;base64,{gif_data}"
    
    # Handle MP4
    if "mp4_path" in result:
        mp4_local_path = os.path.join(f"/{job_id}", "animation.mp4")
        os.rename(result["mp4_path"], mp4_local_path)
        
        with open(mp4_local_path, "rb") as f:
            mp4_data = base64.b64encode(f.read()).decode("utf-8")
            response["mp4"] = f"data:video/mp4;base64,{mp4_data}"
    
    # Handle Audio
    if "audio_path" in result:
        audio_local_path = os.path.join(f"/{job_id}", "audio.wav")
        os.rename(result["audio_path"], audio_local_path)
        
        with open(audio_local_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
            response["audio"] = f"data:audio/wav;base64,{audio_data}"
    
    # Cleanup
    if runpod:
        rp_cleanup.clean([f"/{job_id}"])
    
    return response

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
        characters = input_data.get("characters", input_data.get("character", "temo"))
        
        # Handle both single character and multiple characters
        if isinstance(characters, str):
            if characters not in SUPPORTED_CHARACTERS:
                raise ValueError(f"Invalid character: {characters}. Must be one of {SUPPORTED_CHARACTERS}")
            validated["characters"] = [characters]
        elif isinstance(characters, list):
            for char in characters:
                if char not in SUPPORTED_CHARACTERS:
                    raise ValueError(f"Invalid character: {char}. Must be one of {SUPPORTED_CHARACTERS}")
            validated["characters"] = characters
        else:
            raise ValueError(f"Characters must be string or list, got {type(characters)}")
        
        # Keep backward compatibility with single "character" field
        validated["character"] = validated["characters"][0] if len(validated["characters"]) == 1 else "multi"
        
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
    
    # Animation parameters - ULTRA HIGH QUALITY DEFAULTS (1024x1024)
    if task_type in ["animation", "combined"]:
        validated["num_frames"] = max(16, min(64, input_data.get("num_frames", HIGH_QUALITY_DEFAULTS["num_frames"])))
        validated["fps"] = max(12, min(24, input_data.get("fps", HIGH_QUALITY_DEFAULTS["fps"])))
        validated["width"] = max(768, min(1536, input_data.get("width", HIGH_QUALITY_DEFAULTS["width"])))
        validated["height"] = max(768, min(1536, input_data.get("height", HIGH_QUALITY_DEFAULTS["height"])))
        validated["guidance_scale"] = max(8.0, min(25.0, input_data.get("guidance_scale", HIGH_QUALITY_DEFAULTS["guidance_scale"])))
        validated["num_inference_steps"] = max(30, min(75, input_data.get("num_inference_steps", HIGH_QUALITY_DEFAULTS["num_inference_steps"])))
        validated["negative_prompt"] = input_data.get("negative_prompt", "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands, bad proportions, gross proportions, short arm, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot")
    
    # TTS parameters - ULTRA HIGH QUALITY DEFAULTS  
    if task_type in ["tts", "combined"]:
        validated["max_new_tokens"] = max(2048, min(8192, input_data.get("max_new_tokens", HIGH_QUALITY_DEFAULTS["max_new_tokens"])))
        validated["tts_guidance_scale"] = max(2.0, min(10.0, input_data.get("tts_guidance_scale", HIGH_QUALITY_DEFAULTS["tts_guidance_scale"])))
        validated["temperature"] = max(0.8, min(2.0, input_data.get("temperature", HIGH_QUALITY_DEFAULTS["temperature"])))
        validated["top_p"] = max(0.85, min(0.95, input_data.get("top_p", 0.9)))
        validated["top_k"] = max(40, min(100, input_data.get("top_k", 60)))
    
    # Seed handling
    seed = input_data.get("seed")
    if seed is not None:
        validated["seed"] = int(seed)
    else:
        validated["seed"] = torch.randint(0, 1000000, (1,)).item()
    
    return validated

@torch.inference_mode()
def generate_tts(
    dialogue_text: str,
    max_new_tokens: int = 4096,    # Ultra high quality default
    tts_guidance_scale: float = 5.0,  # Ultra high quality default - maximum guidance
    temperature: float = 1.4,      # Ultra high quality default - optimized consistency
    top_p: float = 0.9,
    top_k: int = 60,               # Ultra high quality default - optimized sampling
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Generate TTS audio from text"""
    
    print(f"🎵 Generating TTS for: {dialogue_text[:50]}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    try:
        # Process text
        inputs = MODELS.dia_processor(text=[dialogue_text], padding=True, return_tensors="pt").to(device)
        
        # Generate audio
        with torch.no_grad():
            outputs = MODELS.dia_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=tts_guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        
        # Decode audio
        audio_arrays = MODELS.dia_processor.batch_decode(outputs)
        
        # Save audio
        timestamp = int(time.time())
        audio_path = TEMP_DIR / f"tts_output_{timestamp}.wav"
        
        # Save the first audio array
        if audio_arrays and len(audio_arrays) > 0:
            MODELS.dia_processor.save_audio(audio_arrays, str(audio_path))
        
        clear_memory()
        
        return {
            "audio_path": str(audio_path),
            "seed": seed,
            "dialogue_text": dialogue_text
        }
        
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        raise

@torch.inference_mode()
def generate_animation(
    characters: Union[str, List[str]] = None,
    character: str = None,  # Backward compatibility
    prompt: str = "",
    num_frames: int = 32,   # Ultra high quality default - maximum frames
    fps: int = 16,          # Ultra high quality default - maximum framerate
    width: int = 1024,      # Ultra high quality default - 1024x1024
    height: int = 1024,     # Ultra high quality default - 1024x1024
    guidance_scale: float = 12.0,     # Ultra high quality default - maximum guidance
    num_inference_steps: int = 50,    # Ultra high quality default - maximum steps
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, censored, distorted hands, amputation, missing hands, obese, doubled face, double hands, bad proportions, gross proportions, short arm, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot",
    seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    """Generate character animation with single or multiple characters"""
    
    # Handle backward compatibility
    if characters is None:
        characters = character if character else "temo"
    
    # Load character LoRA weights (single or multiple)
    lora_weights = load_lora_weights(MODELS.animation_pipeline, characters)
    
    # Format characters for display
    chars_display = characters if isinstance(characters, str) else "_".join(characters)
    chars_list = [characters] if isinstance(characters, str) else characters
    
    print(f"🎬 Generating animation for {chars_display}: {prompt[:50]}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)
    
    try:
        # Generate animation
        result = MODELS.animation_pipeline(
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
        gif_path = OUTPUT_DIR / f"{chars_display}_animation_{timestamp}.gif"
        mp4_path = OUTPUT_DIR / f"{chars_display}_animation_{timestamp}.mp4"
        
        # Export to files
        export_to_gif(frames, str(gif_path), fps=fps)
        export_to_video(frames, str(mp4_path), fps=fps)
        
        clear_memory()
        
        return {
            "gif_path": str(gif_path),
            "mp4_path": str(mp4_path),
            "seed": seed,
            "characters": chars_list,
            "character": chars_display,  # For backward compatibility
            "prompt": prompt,
            "lora_adapters": lora_weights
        }
        
    except Exception as e:
        print(f"❌ Error generating animation: {e}")
        raise

def generate_combined(
    characters: Union[str, List[str]] = None,
    character: str = None,  # Backward compatibility
    prompt: str = "",
    dialogue_text: str = "",
    **kwargs
) -> Dict[str, Any]:
    """Generate combined animation and TTS with single or multiple characters"""
    
    # Handle backward compatibility
    if characters is None:
        characters = character if character else "temo"
    
    chars_display = characters if isinstance(characters, str) else "_".join(characters)
    print(f"🎬🎵 Generating combined animation + TTS for {chars_display}")
    
    # Extract parameters for each task
    animation_params = {k: v for k, v in kwargs.items() 
                       if k in ["num_frames", "fps", "width", "height", "guidance_scale", 
                               "num_inference_steps", "negative_prompt", "seed"]}
    
    tts_params = {k: v for k, v in kwargs.items() 
                  if k in ["max_new_tokens", "tts_guidance_scale", "temperature", 
                          "top_p", "top_k", "seed"]}
    
    # Generate animation
    animation_result = generate_animation(
        characters=characters,
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

@torch.inference_mode()
def generate_cartoon(job):
    """
    Generate cartoon animation with voice using RunPod serverless
    Following the working SDXL example pattern
    """
    # Debug logging (following working example)
    import json, pprint
    
    print("[generate_cartoon] RAW job dict:")
    try:
        print(json.dumps(job, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job, depth=4, compact=False)
    
    start_time = time.time()
    
    try:
        # Setup directories
        setup_directories()
        
        # Only skip model loading in CI, not the entire generation process
        # This ensures the handler works properly in production
        
        # Extract input following working example pattern
        job_input = job["input"]
        
        print("[generate_cartoon] job['input'] payload:")
        try:
            print(json.dumps(job_input, indent=2, default=str), flush=True)
        except Exception:
            pprint.pprint(job_input, depth=4, compact=False)
        
        # Validate input parameters
        validated_input = validate_input(job_input)
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
        
        # Save and upload files (following working example)
        file_urls = _save_and_upload_files(result, job["id"])
        
        # Prepare response
        response = {
            "task_type": task_type,
            "seed": result.get("seed"),
            "generation_time": round(time.time() - start_time, 2),
            "memory_usage": get_memory_usage(),
            **file_urls
        }
        
        # Add metadata
        if "character" in result:
            response["character"] = result["character"]
        if "prompt" in result:
            response["prompt"] = result["prompt"]
        if "dialogue_text" in result:
            response["dialogue_text"] = result["dialogue_text"]
        
        print(f"✅ {task_type} generation completed in {response['generation_time']}s")
        return response
        
    except RuntimeError as err:
        print(f"[ERROR] RuntimeError in generation pipeline: {err}", flush=True)
        return {
            "error": f"RuntimeError: {err}",
            "refresh_worker": True,
        }
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

# RunPod serverless entry point (following working example exactly)
if __name__ == "__main__":
    print("🚀 Starting RunPod Cartoon Animation Worker...")
    print(f"🔧 PyTorch: {torch.__version__}")
    
    # CRITICAL: Verify GPU is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {gpu_name} ({memory_gb:.1f}GB)")
        print("✅ GPU DETECTED - FAST GENERATION ENABLED")
    else:
        print("🚨 WARNING: NO GPU DETECTED - THIS WILL BE EXTREMELY SLOW!")
        print("🚨 Make sure you're running on RunPod with GPU enabled!")
    
    # Initialize directories
    setup_directories()
    
    # Start RunPod serverless worker
    if runpod is not None:
        print("🎬 Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": generate_cartoon})
    else:
        print("⚠️ RunPod module not available - running in development mode")
        print("✅ Handler function is ready for RunPod deployment") 