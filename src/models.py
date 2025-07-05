"""
Model management utilities for the cartoon animation worker
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Any
from transformers import AutoProcessor, DiaForConditionalGeneration
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler

# Global model instances
_dia_model = None
_dia_processor = None
_animation_pipeline = None
_motion_adapter = None

def get_device() -> torch.device:
    """Get the appropriate device for model inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_dia_models(model_checkpoint: str = "nari-labs/Dia-1.6B-0626") -> Tuple[Any, Any]:
    """
    Load Dia TTS model and processor
    
    Args:
        model_checkpoint: HuggingFace model checkpoint path
        
    Returns:
        Tuple of (model, processor)
    """
    global _dia_model, _dia_processor
    
    if _dia_model is not None and _dia_processor is not None:
        return _dia_model, _dia_processor
    
    device = get_device()
    print(f"Loading Dia TTS model on {device}...")
    
    try:
        # Load processor and model
        _dia_processor = AutoProcessor.from_pretrained(model_checkpoint)
        _dia_model = DiaForConditionalGeneration.from_pretrained(
            model_checkpoint,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        
        if device.type != "cuda":
            _dia_model = _dia_model.to(device)
        
        print(f"✅ Dia TTS model loaded on {device}")
        return _dia_model, _dia_processor
        
    except Exception as e:
        print(f"❌ Error loading Dia model: {e}")
        raise

def load_animation_models(
    sdxl_path: str,
    motion_adapter_path: str,
    optimize_memory: bool = True
) -> Any:
    """
    Load AnimateDiff pipeline with SDXL and motion adapter
    
    Args:
        sdxl_path: Path to SDXL model
        motion_adapter_path: Path to motion adapter
        optimize_memory: Whether to apply memory optimizations
        
    Returns:
        AnimateDiff pipeline
    """
    global _animation_pipeline, _motion_adapter
    
    if _animation_pipeline is not None:
        return _animation_pipeline
    
    device = get_device()
    print(f"Loading AnimateDiff pipeline on {device}...")
    
    try:
        # Load motion adapter
        print("Loading motion adapter...")
        _motion_adapter = MotionAdapter.from_pretrained(
            motion_adapter_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            local_files_only=True
        )
        
        # Create pipeline
        print("Creating AnimateDiff pipeline...")
        _animation_pipeline = AnimateDiffSDXLPipeline.from_pretrained(
            sdxl_path,
            motion_adapter=_motion_adapter,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device.type == "cuda" else None,
            local_files_only=True
        )
        
        # Move to device
        _animation_pipeline = _animation_pipeline.to(device)
        
        # Set scheduler
        _animation_pipeline.scheduler = DDIMScheduler.from_config(
            _animation_pipeline.scheduler.config
        )
        
        # Apply memory optimizations if requested
        if optimize_memory and device.type == "cuda":
            print("Applying memory optimizations...")
            _animation_pipeline.enable_sequential_cpu_offload()
            _animation_pipeline.enable_attention_slicing("max")
            _animation_pipeline.enable_vae_slicing()
            _animation_pipeline.enable_vae_tiling()
            
            # Enable memory efficient attention if available
            try:
                _animation_pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers memory efficient attention enabled")
            except Exception:
                print("⚠️ xFormers not available, using default attention")
        
        print(f"✅ AnimateDiff pipeline loaded on {device}")
        return _animation_pipeline
        
    except Exception as e:
        print(f"❌ Error loading animation pipeline: {e}")
        raise

def unload_models():
    """Unload all models to free memory"""
    global _dia_model, _dia_processor, _animation_pipeline, _motion_adapter
    
    print("Unloading models...")
    
    # Move models to CPU and delete references
    if _dia_model is not None:
        _dia_model.cpu()
        del _dia_model
        _dia_model = None
    
    if _dia_processor is not None:
        del _dia_processor
        _dia_processor = None
    
    if _animation_pipeline is not None:
        _animation_pipeline.cpu()
        del _animation_pipeline
        _animation_pipeline = None
    
    if _motion_adapter is not None:
        _motion_adapter.cpu()
        del _motion_adapter
        _motion_adapter = None
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("✅ Models unloaded")

def get_model_info() -> dict:
    """Get information about loaded models"""
    device = get_device()
    
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "dia_model": _dia_model is not None,
            "dia_processor": _dia_processor is not None,
            "animation_pipeline": _animation_pipeline is not None,
            "motion_adapter": _motion_adapter is not None,
        }
    }
    
    if torch.cuda.is_available():
        info["gpu_memory"] = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    return info 