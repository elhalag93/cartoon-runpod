import torch
import os
import gc
from datetime import datetime
from diffusers import AnimateDiffSDXLPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif, export_to_video

# Configuration
SDXL_MODEL_PATH = "/workspace/models/sdxl-turbo"  # Place SDXL Turbo here
MOTION_ADAPTER_PATH = "/workspace/models/animatediff"  # Place AnimateDiff motion adapter here
OUTPUT_DIR = "/workspace/outputs"  # Generated videos will be saved here
LORA_DIR = "/workspace/lora_models"  # Place your LoRA models here

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

def check_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        print(f"\n💻 GPU: {gpu_name}")
        print(f"📊 Memory: {allocated:.2f}GB used, {free:.2f}GB free, {total:.2f}GB total")
        print(f"🔄 Reserved: {reserved:.2f}GB")
        return allocated, total
    return 0, 0

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def setup_pipeline_optimizations(pipe, high_vram_mode=False):
    """Configure pipeline optimizations based on available VRAM"""
    print("\n🔧 Configuring pipeline optimizations...")
    
    try:
        # Basic performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TensorFloat-32: Enabled")
        
        if not high_vram_mode:
            print("📉 Running with memory optimizations")
            pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing("max")
            pipe.enable_vae_slicing()
            try:
                pipe.enable_vae_tiling()
                print("✅ VAE optimizations: Enabled")
            except:
                pass
            
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ xFormers: Enabled")
            except:
                print("ℹ️ xFormers not available")
        else:
            print("📈 Running in high-performance mode")
            
        return True
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        return False

def generate_animation(
    pipe,
    character_name,
    lora_path,
    prompt,
    num_frames=16,
    seed=None,
    guidance_scale=7.5,
    num_inference_steps=15,
    height=512,
    width=512
):
    """Generate animated video for a character"""
    print(f"\n🎬 Generating animation for {character_name}")
    print(f"🎯 Prompt: {prompt}")
    check_gpu_memory()
    
    # Clean memory before generation
    clear_memory()
    
    try:
        # Unload any previous LoRA
        pipe.unload_lora_weights()
        clear_memory()
    except:
        pass
    
    # Load character LoRA
    print(f"📥 Loading {character_name} LoRA...")
    pipe.load_lora_weights(lora_path, weight_name="deep_sdxl_turbo_lora_weights.pt")
    print("✅ LoRA loaded")
    
    # Set up generation parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None:
        seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device).manual_seed(seed)
    
    print(f"🎲 Using seed: {seed}")
    print(f"🖼️ Generating {num_frames} frames...")
    
    # Generate frames
    video_frames = pipe(
        prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator
    ).frames[0]
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = f"{OUTPUT_DIR}/{character_name}_{timestamp}.gif"
    mp4_path = f"{OUTPUT_DIR}/{character_name}_{timestamp}.mp4"
    
    print("💾 Saving outputs...")
    export_to_gif(video_frames, gif_path, fps=8)
    export_to_video(video_frames, mp4_path, fps=8)
    
    print(f"✅ Generation complete!")
    print(f"📁 Saved to:")
    print(f"   GIF: {gif_path}")
    print(f"   MP4: {mp4_path}")
    
    clear_memory()
    return gif_path, mp4_path, seed

def main():
    print("\n🚀 Starting RunPod Animation Pipeline")
    print("=====================================")
    
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU is required but not available!")
    
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🎮 CUDA version: {torch.version.cuda}")
    check_gpu_memory()
    
    try:
        # Load motion adapter
        print(f"\n📥 Loading motion adapter...")
        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER_PATH,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        print("✅ Motion adapter loaded")
        
        # Create pipeline
        print("\n🔄 Creating AnimateDiff pipeline...")
        pipe = AnimateDiffSDXLPipeline.from_pretrained(
            SDXL_MODEL_PATH,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            local_files_only=True
        ).to("cuda")
        
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        print("✅ Pipeline created")
        
        # Configure optimizations based on available VRAM
        _, total_vram = check_gpu_memory()
        high_vram_mode = total_vram >= 24  # Enable high performance mode for 24GB+ GPUs
        setup_pipeline_optimizations(pipe, high_vram_mode)
        
        # Animation prompts
        prompts = {
            "temo": "temo character walking confidently on moon surface, detailed cartoon style, space helmet, lunar landscape, smooth animation, high quality",
            "felfel": "felfel character exploring moon surface, detailed cartoon style, space suit, moon craters, smooth walking animation, high quality"
        }
        
        # Generate animations
        results = []
        for character, prompt in prompts.items():
            lora_path = f"{LORA_DIR}/{character}_lora"
            if os.path.exists(lora_path):
                try:
                    gif_path, mp4_path, seed = generate_animation(
                        pipe=pipe,
                        character_name=character,
                        lora_path=lora_path,
                        prompt=prompt,
                        seed=42 if character == "temo" else 84
                    )
                    results.append({
                        "character": character,
                        "gif": gif_path,
                        "mp4": mp4_path,
                        "seed": seed
                    })
                except Exception as e:
                    print(f"❌ Error generating {character} animation: {e}")
                    continue
        
        # Print summary
        print("\n📊 Generation Summary")
        print("===================")
        print(f"✅ Total animations generated: {len(results)}")
        for result in results:
            print(f"\n🎭 {result['character'].title()}:")
            print(f"   🎲 Seed: {result['seed']}")
            print(f"   🎞️ GIF: {result['gif']}")
            print(f"   🎬 MP4: {result['mp4']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        clear_memory()
        print("\n💾 Final memory state:")
        check_gpu_memory()
        print("\n✨ Pipeline complete!")

if __name__ == "__main__":
    main() 