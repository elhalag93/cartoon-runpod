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
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler, ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import export_to_video, export_to_gif
from transformers import AutoProcessor, DiaForConditionalGeneration
import cv2
import openpose_pytorch as openpose

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

def load_tts_model():
    """Load TTS model and processor"""
    global dia_model, dia_processor
    
    if dia_model is None:
        print("🔄 Loading Dia TTS model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dia_processor = AutoProcessor.from_pretrained(DIA_MODEL_CHECKPOINT)
            dia_model = DiaForConditionalGeneration.from_pretrained(DIA_MODEL_CHECKPOINT).to(device)
            print("✅ Dia TTS model loaded")
        except Exception as e:
            print(f"❌ Error loading TTS model: {e}")
            raise
    
    return dia_model, dia_processor

def load_animation_pipeline():
    """Load animation pipeline"""
    global animation_pipeline, motion_adapter
    
    if animation_pipeline is None:
        print("🔄 Loading AnimateDiff pipeline...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load motion adapter
            motion_adapter = MotionAdapter.from_pretrained(
                "animatediff/animatediff-motion-adapter-v1",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            # Load pipeline
            animation_pipeline = AnimateDiffSDXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                motion_adapter=motion_adapter,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None
            )
            
            animation_pipeline = animation_pipeline.to(device)
            
            # Enable memory optimizations
            if device == "cuda":
                animation_pipeline.enable_sequential_cpu_offload()
                animation_pipeline.enable_attention_slicing()
                animation_pipeline.enable_vae_slicing()
                animation_pipeline.enable_vae_tiling()
            
            # Set scheduler
            animation_pipeline.scheduler = DDIMScheduler.from_config(
                animation_pipeline.scheduler.config,
                beta_schedule="linear",
                steps_offset=1,
                clip_sample=False
            )
            
            print("✅ AnimateDiff pipeline loaded")
        except Exception as e:
            print(f"❌ Error loading animation pipeline: {e}")
            raise
    
    return animation_pipeline

def load_controlnet_pose_model():
    """Load ControlNet pose model"""
    controlnet_path = os.path.join("models", "controlnet", "controlnet_pose.pth")
    if not os.path.exists(controlnet_path):
        raise FileNotFoundError(f"ControlNet pose model not found at {controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return controlnet

def extract_pose_map(image: np.ndarray) -> np.ndarray:
    """Extract pose map from an image using OpenPose"""
    # Convert to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Use OpenPose to extract pose
    pose_map = openpose.infer(image)
    # Convert pose map to uint8 for ControlNet
    pose_map = (pose_map * 255).astype(np.uint8)
    return pose_map

def encode_file_to_base64(file_path: str) -> str:
    """Encode file to base64 string"""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding file {file_path}: {e}")
        return ""

def generate_tts(
    dialogue_text: str,
    max_new_tokens: int = 3072,
    guidance_scale: float = 3.0,
    temperature: float = 1.8,
    top_p: float = 0.9,
    top_k: int = 45,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate TTS audio"""
    
    # Load TTS model if not loaded
    model, processor = load_tts_model()
    
    print(f"🎵 Generating TTS for: {dialogue_text[:50]}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set seed
    if seed is None:
        seed = torch.randint(0, 1000000, (1,)).item()
    
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
                guidance_scale=guidance_scale,
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
            # Assuming the processor returns audio data that can be saved
            sf.write(str(audio_path), audio_arrays[0], 44100)
        
        clear_memory()
        
        return {
            "audio_path": str(audio_path),
            "seed": seed,
            "dialogue_text": dialogue_text
        }
        
    except Exception as e:
        print(f"❌ Error generating TTS: {e}")
        raise

def generate_combined(
    character: str,
    prompt: str,
    dialogue_text: str,
    num_frames: int = 16,
    fps: int = 8,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 15,
    max_new_tokens: int = 3072,
    tts_guidance_scale: float = 3.0,
    temperature: float = 1.8,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate combined animation and TTS"""
    
    print(f"🎬🎵 Generating combined animation + TTS for {character}")
    
    # Set seed
    if seed is None:
        seed = torch.randint(0, 1000000, (1,)).item()
    
    # Generate animation
    animation_result = generate_animation(
        character=character,
        prompt=prompt,
        num_frames=num_frames,
        fps=fps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    # Generate TTS
    tts_result = generate_tts(
        dialogue_text=dialogue_text,
        max_new_tokens=max_new_tokens,
        guidance_scale=tts_guidance_scale,
        temperature=temperature,
        seed=seed
    )
    
    # Combine results
    combined_result = {
        **animation_result,
        **tts_result,
        "task_type": "combined"
    }
    
    return combined_result

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler for animation generation"""
    try:
        # Extract input parameters
        input_data = job.get("input", {})
        task_type = input_data.get("task_type", "animation")
        
        print(f"🎬 Starting {task_type} generation...")
        
        # Setup directories
        setup_directories()
        
        if task_type == "animation":
            result = generate_animation(**input_data)
        elif task_type == "tts":
            result = generate_tts(**input_data)
        elif task_type == "combined":
            result = generate_combined(**input_data)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        # Encode files to base64 for download
        response = {"task_type": task_type}
        
        if "gif_path" in result:
            response["gif"] = encode_file_to_base64(result["gif_path"])
            response["gif_path"] = result["gif_path"]
        
        if "mp4_path" in result:
            response["mp4"] = encode_file_to_base64(result["mp4_path"])
            response["mp4_path"] = result["mp4_path"]
        
        if "audio_path" in result:
            response["audio"] = encode_file_to_base64(result["audio_path"])
            response["audio_path"] = result["audio_path"]
        
        response["seed"] = result.get("seed")
        
        # Clean up files after encoding
        for path_key in ["gif_path", "mp4_path", "audio_path"]:
            if path_key in result:
                try:
                    os.remove(result[path_key])
                except:
                    pass
        
        # Memory usage info
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            response["memory_usage"] = {
                "allocated_gb": round(allocated, 2),
                "total_gb": round(total, 2)
            }
        
        return response
        
    except Exception as e:
        error_msg = f"Error in handler: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}

# Update generate_animation to return proper result format
def generate_animation(
    character: str,
    prompt: str,
    num_frames: int = 16,
    fps: int = 8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 15,
    seed: Optional[int] = None,
    width: int = 512,
    height: int = 512,
    **kwargs
) -> Dict[str, str]:
    """Generate character animation using frame-to-frame conditioning with ControlNet pose"""
    # Load AnimateDiff pipeline and ControlNet pose model
    pipeline = load_animation_pipeline()
    controlnet = load_controlnet_pose_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionControlNetPipeline(
        vae=pipeline.vae,
        text_encoder=pipeline.text_encoder,
        tokenizer=pipeline.tokenizer,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        safety_checker=None,
        feature_extractor=None,
        controlnet=controlnet,
        requires_safety_checker=False
    ).to(device)

    # Load LoRA weights as before
    felfel_lora_path = LORA_DIR / "felfel_lora"
    temo_lora_path = LORA_DIR / "temo_lora"
    if not felfel_lora_path.exists():
        raise ValueError(f"Felfel LoRA not found at {felfel_lora_path}")
    if not temo_lora_path.exists():
        raise ValueError(f"Temo LoRA not found at {temo_lora_path}")
    pipeline.load_lora_weights(
        [str(felfel_lora_path), str(temo_lora_path)],
        weight_name=["deep_sdxl_turbo_lora_weights.pt", "deep_sdxl_turbo_lora_weights.pt"],
        adapter_name=["felfel", "temo"]
    )
    pipeline.set_adapters(["felfel", "temo"], adapter_weights=[0.8, 0.8])

    # Set seed
    if seed is None:
        seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device).manual_seed(seed)

    # Frame-to-frame conditioning loop
    frames = []
    pose_map = None  # Initial pose map (could be user-provided or generated)
    for i in range(num_frames):
        if i == 0:
            # For the first frame, use a blank or user-provided pose map
            pose_map = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Extract pose from previous frame
            prev_frame = np.array(frames[-1])
            pose_map = extract_pose_map(prev_frame)
        # Generate frame with ControlNet pose conditioning
        result = pipeline(
            prompt,
            image=pose_map,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator
        )
        frame = result.images[0]
        frames.append(frame)

    # Save outputs
    timestamp = int(time.time())
    gif_path = OUTPUT_DIR / f"{character}_{timestamp}.gif"
    mp4_path = OUTPUT_DIR / f"{character}_{timestamp}.mp4"
    export_to_gif(frames, str(gif_path), fps=fps)
    export_to_video(frames, str(mp4_path), fps=fps)
    clear_memory()
    return {
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path),
        "seed": seed
    }

# RunPod serverless entry point
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler}) 