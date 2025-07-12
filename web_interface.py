#!/usr/bin/env python3
"""
Enhanced Web Interface for Cartoon Animation Generation
Combines Animation Generation + TTS with intuitive UI
"""

import gradio as gr
import json
import base64
import tempfile
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import requests
import torch
import numpy as np
from PIL import Image
import io

# Import local modules
from src.handler import generate_animation, generate_tts, generate_combined
from src.handler import load_animation_pipeline, load_tts_model

# Global variables
animation_pipeline = None
tts_model = None
tts_processor = None

# Configuration
CHARACTERS = ["both"]  # Always use both characters together
MAX_FRAMES = 32
DEFAULT_PROMPTS = {
    "both": "felfel and temo characters together, cartoon adventure, detailed animation style"
}

def initialize_models():
    """Initialize models on startup"""
    global animation_pipeline, tts_model, tts_processor
    
    try:
        print("🔄 Loading animation pipeline...")
        animation_pipeline = load_animation_pipeline()
        print("✅ Animation pipeline loaded")
        
        print("🔄 Loading TTS model...")
        tts_model, tts_processor = load_tts_model()
        print("✅ TTS model loaded")
        
        return "✅ All models loaded successfully!"
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}"
        print(error_msg)
        return error_msg

def generate_animation_interface(
    character: str,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int],
    progress=gr.Progress()
) -> Tuple[str, str, str, Dict]:
    """Generate animation with progress tracking"""
    
    progress(0.1, "Starting animation generation...")
    
    try:
        # Prepare parameters
        params = {
            "character": character,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed
        }
        
        progress(0.3, f"Generating {num_frames} frames for {character}...")
        
        # Generate animation
        result = generate_animation(**params)
        
        progress(0.9, "Finalizing outputs...")
        
        # Prepare results
        gif_path = result.get("gif_path", "")
        mp4_path = result.get("mp4_path", "")
        used_seed = result.get("seed", seed)
        
        # Create result info
        result_info = {
            "character": character,
            "prompt": prompt,
            "frames": num_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "seed": used_seed,
            "guidance_scale": guidance_scale,
            "inference_steps": num_inference_steps
        }
        
        progress(1.0, "Animation complete!")
        
        return (
            gif_path,
            mp4_path,
            f"✅ Animation generated successfully!\n🎲 Seed: {used_seed}",
            result_info
        )
        
    except Exception as e:
        error_msg = f"❌ Error generating animation: {str(e)}"
        return None, None, error_msg, {}

def generate_tts_interface(
    dialogue_text: str,
    max_new_tokens: int,
    guidance_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int],
    progress=gr.Progress()
) -> Tuple[str, str, Dict]:
    """Generate TTS with progress tracking"""
    
    progress(0.1, "Starting TTS generation...")
    
    try:
        # Prepare parameters
        params = {
            "dialogue_text": dialogue_text,
            "max_new_tokens": max_new_tokens,
            "guidance_scale": guidance_scale,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed
        }
        
        progress(0.5, "Generating speech...")
        
        # Generate TTS
        result = generate_tts(**params)
        
        progress(0.9, "Finalizing audio...")
        
        # Prepare results
        audio_path = result.get("audio_path", "")
        used_seed = result.get("seed", seed)
        
        # Create result info
        result_info = {
            "dialogue_text": dialogue_text,
            "max_tokens": max_new_tokens,
            "guidance_scale": guidance_scale,
            "temperature": temperature,
            "seed": used_seed
        }
        
        progress(1.0, "TTS complete!")
        
        return (
            audio_path,
            f"✅ TTS generated successfully!\n🎲 Seed: {used_seed}",
            result_info
        )
        
    except Exception as e:
        error_msg = f"❌ Error generating TTS: {str(e)}"
        return None, error_msg, {}

def generate_combined_interface(
    character: str,
    prompt: str,
    dialogue_text: str,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    max_new_tokens: int,
    tts_guidance_scale: float,
    temperature: float,
    seed: Optional[int],
    progress=gr.Progress()
) -> Tuple[str, str, str, str, Dict]:
    """Generate combined animation + TTS"""
    
    progress(0.1, "Starting combined generation...")
    
    try:
        # Prepare parameters
        params = {
            "character": character,
            "prompt": prompt,
            "dialogue_text": dialogue_text,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "max_new_tokens": max_new_tokens,
            "tts_guidance_scale": tts_guidance_scale,
            "temperature": temperature,
            "seed": seed
        }
        
        progress(0.3, f"Generating animation for {character}...")
        progress(0.6, "Generating speech...")
        
        # Generate combined
        result = generate_combined(**params)
        
        progress(0.9, "Finalizing outputs...")
        
        # Prepare results
        gif_path = result.get("gif_path", "")
        mp4_path = result.get("mp4_path", "")
        audio_path = result.get("audio_path", "")
        used_seed = result.get("seed", seed)
        
        # Create result info
        result_info = {
            "character": character,
            "prompt": prompt,
            "dialogue_text": dialogue_text,
            "frames": num_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "seed": used_seed,
            "animation_guidance": guidance_scale,
            "tts_guidance": tts_guidance_scale,
            "temperature": temperature
        }
        
        progress(1.0, "Combined generation complete!")
        
        return (
            gif_path,
            mp4_path,
            audio_path,
            f"✅ Combined generation successful!\n🎲 Seed: {used_seed}",
            result_info
        )
        
    except Exception as e:
        error_msg = f"❌ Error in combined generation: {str(e)}"
        return None, None, None, error_msg, {}

def update_prompt_suggestion(character: str) -> str:
    """Update prompt suggestion for both characters"""
    return DEFAULT_PROMPTS.get("both", "felfel and temo characters together, cartoon adventure, detailed animation style")

def create_interface():
    """Create the main Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .character-card {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .result-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .status-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=css, title="🎬 Cartoon Animation Studio", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎬 Cartoon Animation Studio</h1>
            <p>Generate high-quality cartoon animations with voice using AI</p>
            <p><strong>Characters:</strong> Felfel & Temo Together - Adventure Duo!</p>
        </div>
        """)
        
        # Model initialization status
        with gr.Row():
            with gr.Column():
                init_status = gr.Textbox(
                    label="🔧 System Status",
                    value="Click 'Initialize Models' to start",
                    interactive=False
                )
                init_btn = gr.Button("🚀 Initialize Models", variant="primary")
        
        # Main tabs
        with gr.Tabs():
            
            # Animation Only Tab
            with gr.TabItem("🎬 Animation Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎭 Character & Prompt</h3></div>')
                        
                        anim_character = gr.Dropdown(
                            choices=CHARACTERS,
                            value="both",
                            label="Characters",
                            info="Using both felfel and temo together"
                        )
                        
                        anim_prompt = gr.Textbox(
                            label="Animation Prompt",
                            placeholder="Describe what you want both characters to do...",
                            lines=3,
                            value=DEFAULT_PROMPTS["both"]
                        )
                        
                        anim_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the animation...",
                            value="blurry, low quality, distorted, ugly",
                            lines=2
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Generation Settings</h3></div>')
                        
                        with gr.Row():
                            anim_frames = gr.Slider(8, MAX_FRAMES, 16, label="Frames", step=1)
                            anim_fps = gr.Slider(4, 12, 8, label="FPS", step=1)
                        
                        with gr.Row():
                            anim_width = gr.Slider(256, 768, 512, label="Width", step=64)
                            anim_height = gr.Slider(256, 768, 512, label="Height", step=64)
                        
                        with gr.Row():
                            anim_guidance = gr.Slider(1.0, 15.0, 7.5, label="Guidance Scale", step=0.5)
                            anim_steps = gr.Slider(10, 30, 15, label="Inference Steps", step=1)
                        
                        anim_seed = gr.Number(
                            label="Seed (optional)",
                            placeholder="Leave empty for random",
                            precision=0
                        )
                        
                        anim_generate_btn = gr.Button("🎬 Generate Animation", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>📺 Results</h3></div>')
                        
                        anim_gif_output = gr.Image(label="Generated GIF", type="filepath")
                        anim_video_output = gr.Video(label="Generated Video")
                        anim_status = gr.Textbox(label="Status", interactive=False)
                        anim_info = gr.JSON(label="Generation Info", visible=False)
            
            # TTS Only Tab
            with gr.TabItem("🎵 Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎤 Speech Generation</h3></div>')
                        
                        tts_dialogue = gr.Textbox(
                            label="Dialogue Text",
                            placeholder="[S1] Hello! [S2] How are you today?",
                            lines=4,
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ TTS Settings</h3></div>')
                        
                        tts_max_tokens = gr.Slider(1024, 4096, 3072, label="Max Tokens", step=256)
                        tts_guidance = gr.Slider(1.0, 10.0, 3.0, label="Guidance Scale", step=0.5)
                        tts_temperature = gr.Slider(0.1, 2.0, 1.8, label="Temperature", step=0.1)
                        tts_top_p = gr.Slider(0.1, 1.0, 0.9, label="Top P", step=0.05)
                        tts_top_k = gr.Slider(1, 100, 45, label="Top K", step=1)
                        
                        tts_seed = gr.Number(
                            label="Seed (optional)",
                            placeholder="Leave empty for random",
                            precision=0
                        )
                        
                        tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>🔊 Results</h3></div>')
                        
                        tts_audio_output = gr.Audio(label="Generated Speech")
                        tts_status = gr.Textbox(label="Status", interactive=False)
                        tts_info = gr.JSON(label="Generation Info", visible=False)
            
            # Combined Tab
            with gr.TabItem("🎬🎵 Animation + Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎭 Character & Content</h3></div>')
                        
                        comb_character = gr.Dropdown(
                            choices=CHARACTERS,
                            value="both",
                            label="Characters"
                        )
                        
                        comb_prompt = gr.Textbox(
                            label="Animation Prompt",
                            placeholder="Describe both characters' actions...",
                            lines=2,
                            value=DEFAULT_PROMPTS["both"]
                        )
                        
                        comb_dialogue = gr.Textbox(
                            label="Dialogue Text",
                            placeholder="[S1] Character speech here...",
                            lines=3,
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Combined Settings</h3></div>')
                        
                        with gr.Row():
                            comb_frames = gr.Slider(8, MAX_FRAMES, 16, label="Frames", step=1)
                            comb_fps = gr.Slider(4, 12, 8, label="FPS", step=1)
                        
                        with gr.Row():
                            comb_width = gr.Slider(256, 768, 512, label="Width", step=64)
                            comb_height = gr.Slider(256, 768, 512, label="Height", step=64)
                        
                        with gr.Row():
                            comb_anim_guidance = gr.Slider(1.0, 15.0, 7.5, label="Animation Guidance", step=0.5)
                            comb_steps = gr.Slider(10, 30, 15, label="Inference Steps", step=1)
                        
                        with gr.Row():
                            comb_max_tokens = gr.Slider(1024, 4096, 3072, label="Max Tokens", step=256)
                            comb_tts_guidance = gr.Slider(1.0, 10.0, 3.0, label="TTS Guidance", step=0.5)
                        
                        comb_temperature = gr.Slider(0.1, 2.0, 1.8, label="Temperature", step=0.1)
                        
                        comb_seed = gr.Number(
                            label="Seed (optional)",
                            placeholder="Leave empty for random",
                            precision=0
                        )
                        
                        comb_generate_btn = gr.Button("🎬🎵 Generate Combined", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>📺🔊 Results</h3></div>')
                        
                        comb_gif_output = gr.Image(label="Generated GIF", type="filepath")
                        comb_video_output = gr.Video(label="Generated Video")
                        comb_audio_output = gr.Audio(label="Generated Speech")
                        comb_status = gr.Textbox(label="Status", interactive=False)
                        comb_info = gr.JSON(label="Generation Info", visible=False)
            
            # API Tab
            with gr.TabItem("🔌 API Reference"):
                gr.HTML("""
                <div style="padding: 20px;">
                    <h3>🔌 API Endpoints</h3>
                    <p>Use these endpoints for programmatic access:</p>
                    
                    <h4>Animation Generation</h4>
                    <pre><code>POST /api/animation
{
  "character": "temo",
  "prompt": "character walking on moon",
  "num_frames": 16,
  "fps": 8,
  "width": 512,
  "height": 512,
  "guidance_scale": 7.5,
  "num_inference_steps": 15,
  "seed": 42
}</code></pre>
                    
                    <h4>TTS Generation</h4>
                    <pre><code>POST /api/tts
{
  "dialogue_text": "[S1] Hello world!",
  "max_new_tokens": 3072,
  "guidance_scale": 3.0,
  "temperature": 1.8,
  "seed": 42
}</code></pre>
                    
                    <h4>Combined Generation</h4>
                    <pre><code>POST /api/combined
{
  "character": "temo",
  "prompt": "character waving hello",
  "dialogue_text": "[S1] Hello everyone!",
  "num_frames": 16,
  "fps": 8,
  "seed": 42
}</code></pre>
                    
                    <h4>Response Format</h4>
                    <pre><code>{
  "gif": "base64_encoded_gif",
  "mp4": "base64_encoded_mp4",
  "audio": "base64_encoded_audio",
  "seed": 42,
  "status": "success"
}</code></pre>
                </div>
                """)
        
        # Event handlers
        init_btn.click(
            fn=initialize_models,
            outputs=[init_status]
        )
        
        # Character change updates prompt
        anim_character.change(
            fn=update_prompt_suggestion,
            inputs=[anim_character],
            outputs=[anim_prompt]
        )
        
        comb_character.change(
            fn=update_prompt_suggestion,
            inputs=[comb_character],
            outputs=[comb_prompt]
        )
        
        # Animation generation
        anim_generate_btn.click(
            fn=generate_animation_interface,
            inputs=[
                anim_character, anim_prompt, anim_negative,
                anim_frames, anim_fps, anim_width, anim_height,
                anim_guidance, anim_steps, anim_seed
            ],
            outputs=[anim_gif_output, anim_video_output, anim_status, anim_info]
        )
        
        # TTS generation
        tts_generate_btn.click(
            fn=generate_tts_interface,
            inputs=[
                tts_dialogue, tts_max_tokens, tts_guidance,
                tts_temperature, tts_top_p, tts_top_k, tts_seed
            ],
            outputs=[tts_audio_output, tts_status, tts_info]
        )
        
        # Combined generation
        comb_generate_btn.click(
            fn=generate_combined_interface,
            inputs=[
                comb_character, comb_prompt, comb_dialogue,
                comb_frames, comb_fps, comb_width, comb_height,
                comb_anim_guidance, comb_steps, comb_max_tokens,
                comb_tts_guidance, comb_temperature, comb_seed
            ],
            outputs=[comb_gif_output, comb_video_output, comb_audio_output, comb_status, comb_info]
        )
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cartoon Animation Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_interface()
    
    print("🚀 Starting Cartoon Animation Studio...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    ) 