#!/usr/bin/env python3
"""
Web Interface for Cartoon Animation Studio
Interactive Gradio interface for generating animations and TTS
"""

import gradio as gr
import json
import time
import traceback
import os
import base64
from typing import Optional, Dict, Any, Tuple, Union, List
from pathlib import Path

# Import local modules
from src.handler import generate_animation, generate_tts, generate_combined
from src.handler import load_animation_pipeline, load_tts_model

# Global variables
animation_pipeline = None
tts_model = None
tts_processor = None

# Configuration
CHARACTERS = [
    "Temo", 
    "Felfel", 
    "Both (Multi-Character)"
]

MAX_FRAMES = 32

DEFAULT_PROMPTS = {
    "Temo": "temo character exploring space station with epic lighting, detailed cartoon style, space adventure",
    "Felfel": "felfel character discovering magical crystal cave with epic lighting, detailed cartoon style, fantasy adventure", 
    "Both (Multi-Character)": "temo and felfel characters working together on moon base, both characters clearly visible, temo in space suit on left, felfel in adventure gear on right, epic lighting, detailed cartoon style"
}

DEFAULT_DIALOGUES = {
    "Temo": "[S1] Greetings from the space station! [S2] What an incredible view of Earth!",
    "Felfel": "[S1] Look at these amazing crystal formations! [S2] This cave is truly magical!",
    "Both (Multi-Character)": "[S1] Temo: Welcome to our lunar base, Felfel! [S2] Felfel: This technology is incredible, Temo! [S1] Temo: Let's explore together!"
}

def initialize_models():
    """Initialize models on startup"""
    global animation_pipeline, tts_model, tts_processor
    
    try:
        gr.Info("🔄 Loading animation pipeline...")
        animation_pipeline = load_animation_pipeline()
        gr.Info("✅ Animation pipeline loaded")
        
        gr.Info("🔄 Loading TTS model...")
        tts_model, tts_processor = load_tts_model()
        gr.Info("✅ TTS model loaded")
        
        return "✅ All models loaded successfully! Ready to generate animations."
        
    except Exception as e:
        error_msg = f"❌ Error loading models: {str(e)}"
        gr.Error(error_msg)
        return error_msg

def convert_character_selection(character_selection: str) -> Union[str, List[str]]:
    """Convert character selection to format expected by backend"""
    if character_selection == "Both (Multi-Character)":
        return ["temo", "felfel"]
    else:
        return character_selection.lower()

def generate_animation_interface(
    character_selection: str,
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
    """Generate animation through the interface"""
    
    try:
        progress(0.1, desc="Starting animation generation...")
        
        # Convert character selection
        characters = convert_character_selection(character_selection)
        
        progress(0.3, desc="Setting up parameters...")
        
        # Prepare parameters for backend
        params = {
            "characters": characters,
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
        
        progress(0.5, desc="Generating animation...")
        
        # Generate animation
        result = generate_animation(**params)
        
        progress(0.9, desc="Processing results...")
        
        # Handle file paths and create outputs
        gif_output = None
        mp4_output = None
        
        if "gif_path" in result and os.path.exists(result["gif_path"]):
            gif_output = result["gif_path"]
        
        if "mp4_path" in result and os.path.exists(result["mp4_path"]):
            mp4_output = result["mp4_path"]
        
        # Create status message
        if isinstance(characters, list) and len(characters) > 1:
            status = f"✅ Multi-Character Animation Generated!\n" \
                     f"Characters: {' + '.join([c.upper() for c in characters])}\n" \
                     f"Frames: {num_frames} at {fps} FPS\n" \
                     f"Resolution: {width}x{height}\n" \
                     f"Seed: {result.get('seed', 'N/A')}\n" \
                     f"🎭 Both characters appear together!"
        else:
            char_name = characters if isinstance(characters, str) else characters[0]
            status = f"✅ Animation Generated!\n" \
                     f"Character: {char_name.upper()}\n" \
                     f"Frames: {num_frames} at {fps} FPS\n" \
                     f"Resolution: {width}x{height}\n" \
                     f"Seed: {result.get('seed', 'N/A')}"
        
        progress(1.0, desc="Complete!")
        
        # Return results
        return gif_output, mp4_output, status, result
        
    except Exception as e:
        error_msg = f"❌ Error generating animation: {str(e)}"
        gr.Error(error_msg)
        return None, None, error_msg, {"error": str(e)}

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
    """Generate TTS through the interface"""
    
    try:
        progress(0.1, desc="Starting TTS generation...")
        
        params = {
            "dialogue_text": dialogue_text,
            "max_new_tokens": max_new_tokens,
            "tts_guidance_scale": guidance_scale,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed
        }
        
        progress(0.5, desc="Generating speech...")
        
        result = generate_tts(**params)
        
        progress(0.9, desc="Processing audio...")
        
        # Handle audio output
        audio_output = None
        if "audio_path" in result and os.path.exists(result["audio_path"]):
            audio_output = result["audio_path"]
        
        status = f"✅ TTS Generated!\n" \
                 f"Text length: {len(dialogue_text)} characters\n" \
                 f"Tokens: {max_new_tokens}\n" \
                 f"Temperature: {temperature}\n" \
                 f"Seed: {result.get('seed', 'N/A')}"
        
        progress(1.0, desc="Complete!")
        
        return audio_output, status, result
        
    except Exception as e:
        error_msg = f"❌ Error generating TTS: {str(e)}"
        gr.Error(error_msg)
        return None, error_msg, {"error": str(e)}

def generate_combined_interface(
    character_selection: str,
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
    """Generate combined animation and TTS through the interface"""
    
    try:
        progress(0.1, desc="Starting combined generation...")
        
        # Convert character selection
        characters = convert_character_selection(character_selection)
        
        params = {
            "characters": characters,
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
        
        progress(0.5, desc="Generating animation and speech...")
        
        result = generate_combined(**params)
        
        progress(0.9, desc="Processing results...")
        
        # Handle outputs
        gif_output = None
        mp4_output = None
        audio_output = None
        
        if "gif_path" in result and os.path.exists(result["gif_path"]):
            gif_output = result["gif_path"]
        
        if "mp4_path" in result and os.path.exists(result["mp4_path"]):
            mp4_output = result["mp4_path"]
            
        if "audio_path" in result and os.path.exists(result["audio_path"]):
            audio_output = result["audio_path"]
        
        # Create status message
        if isinstance(characters, list) and len(characters) > 1:
            status = f"✅ Multi-Character Combined Generation Complete!\n" \
                     f"Characters: {' + '.join([c.upper() for c in characters])}\n" \
                     f"Animation: {num_frames} frames at {fps} FPS\n" \
                     f"Resolution: {width}x{height}\n" \
                     f"Audio tokens: {max_new_tokens}\n" \
                     f"Seed: {result.get('seed', 'N/A')}\n" \
                     f"🎭 Both characters with synchronized speech!"
        else:
            char_name = characters if isinstance(characters, str) else characters[0]
            status = f"✅ Combined Generation Complete!\n" \
                     f"Character: {char_name.upper()}\n" \
                     f"Animation: {num_frames} frames at {fps} FPS\n" \
                     f"Resolution: {width}x{height}\n" \
                     f"Audio tokens: {max_new_tokens}\n" \
                     f"Seed: {result.get('seed', 'N/A')}"
        
        progress(1.0, desc="Complete!")
        
        return gif_output, mp4_output, audio_output, status, result
        
    except Exception as e:
        error_msg = f"❌ Error in combined generation: {str(e)}"
        gr.Error(error_msg)
        return None, None, None, error_msg, {"error": str(e)}

def update_prompt_suggestion(character_selection: str) -> str:
    """Update prompt suggestion based on character selection"""
    return DEFAULT_PROMPTS.get(character_selection, "")

def update_dialogue_suggestion(character_selection: str) -> str:
    """Update dialogue suggestion based on character selection"""
    return DEFAULT_DIALOGUES.get(character_selection, "")

def create_interface():
    """Create the main Gradio interface"""
    
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
    .multi-character-card {
        border: 2px solid #FF6B35;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
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
            <p><strong>Characters:</strong> Temo (Space Explorer) & Felfel (Adventure Seeker)</p>
            <p><strong>✨ Multi-Character Support:</strong> Both characters together in one scene!</p>
        </div>
        """)
        
        # Model initialization
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div style="background-color: #e7f3ff; padding: 15px; border-radius: 10px; border: 1px solid #b3d9ff; margin: 10px 0;">
                    <h3 style="color: #1976d2; margin: 0;">🚀 Model Initialization</h3>
                    <p style="margin: 5px 0 0 0;">Click to load the AI models (required before generating)</p>
                </div>
                """)
                init_status = gr.Textbox(
                    label="Initialization Status",
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
                            value="Both (Multi-Character)",
                            label="Character Selection",
                            info="Choose single character or both together"
                        )
                        
                        anim_prompt = gr.Textbox(
                            label="Animation Prompt",
                            placeholder="Describe what you want the character(s) to do...",
                            lines=3,
                            value=DEFAULT_PROMPTS["Both (Multi-Character)"]
                        )
                        
                        anim_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the animation...",
                            value="blurry, low quality, distorted, ugly",
                            lines=2
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Animation Settings</h3></div>')
                        
                        with gr.Row():
                            anim_frames = gr.Slider(8, MAX_FRAMES, 16, label="Frames", step=1)
                            anim_fps = gr.Slider(4, 12, 8, label="FPS", step=1)
                        
                        with gr.Row():
                            anim_width = gr.Slider(256, 768, 512, label="Width", step=64)
                            anim_height = gr.Slider(256, 768, 512, label="Height", step=64)
                        
                        with gr.Row():
                            anim_guidance = gr.Slider(1.0, 15.0, 7.5, label="Guidance Scale", step=0.5)
                            anim_steps = gr.Slider(10, 30, 15, label="Inference Steps", step=1)
                        
                        anim_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        anim_generate_btn = gr.Button("🎬 Generate Animation", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>📺 Animation Results</h3></div>')
                        
                        anim_gif_output = gr.Image(label="Generated GIF", type="filepath")
                        anim_video_output = gr.Video(label="Generated MP4")
                        anim_status = gr.Textbox(label="Status", interactive=False, lines=6)
                        anim_info = gr.JSON(label="Generation Info", visible=False)
            
            # TTS Only Tab
            with gr.TabItem("🎵 Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎤 Speech Content</h3></div>')
                        
                        tts_dialogue = gr.Textbox(
                            label="Dialogue Text",
                            placeholder="Enter dialogue with [S1] and [S2] speaker tags...",
                            lines=4,
                            value="[S1] Welcome to our amazing animation studio! [S2] Create incredible voices with AI technology!",
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Voice Settings</h3></div>')
                        
                        tts_max_tokens = gr.Slider(1024, 4096, 3072, label="Max Tokens", step=256)
                        tts_guidance = gr.Slider(1.0, 10.0, 3.0, label="Guidance Scale", step=0.5)
                        tts_temperature = gr.Slider(0.1, 2.0, 1.8, label="Temperature", step=0.1)
                        
                        with gr.Row():
                            tts_top_p = gr.Slider(0.1, 1.0, 0.9, label="Top P", step=0.05)
                            tts_top_k = gr.Slider(1, 100, 45, label="Top K", step=5)
                        
                        tts_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>🔊 Audio Results</h3></div>')
                        
                        tts_audio_output = gr.Audio(label="Generated Audio")
                        tts_status = gr.Textbox(label="Status", interactive=False, lines=4)
                        tts_info = gr.JSON(label="Generation Info", visible=False)
            
            # Combined Tab
            with gr.TabItem("🎬🎵 Animation + Speech (BEST)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="multi-character-card"><h3>🎭 Combined Generation</h3></div>')
                        
                        comb_character = gr.Dropdown(
                            choices=CHARACTERS,
                            value="Both (Multi-Character)",
                            label="Character Selection",
                            info="Choose single character or both together"
                        )
                        
                        comb_prompt = gr.Textbox(
                            label="Animation Prompt",
                            placeholder="Describe the visual scene...",
                            lines=2,
                            value=DEFAULT_PROMPTS["Both (Multi-Character)"]
                        )
                        
                        comb_dialogue = gr.Textbox(
                            label="Dialogue Text",
                            placeholder="Enter speech with [S1] [S2] tags...",
                            lines=3,
                            value=DEFAULT_DIALOGUES["Both (Multi-Character)"],
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Quality Settings</h3></div>')
                        
                        with gr.Row():
                            comb_frames = gr.Slider(8, MAX_FRAMES, 16, label="Animation Frames", step=1)
                            comb_fps = gr.Slider(4, 12, 8, label="FPS", step=1)
                        
                        with gr.Row():
                            comb_width = gr.Slider(256, 768, 512, label="Width", step=64)
                            comb_height = gr.Slider(256, 768, 512, label="Height", step=64)
                        
                        with gr.Row():
                            comb_guidance = gr.Slider(1.0, 15.0, 7.5, label="Animation Guidance", step=0.5)
                            comb_steps = gr.Slider(10, 30, 15, label="Inference Steps", step=1)
                        
                        with gr.Row():
                            comb_max_tokens = gr.Slider(1024, 4096, 3072, label="Audio Tokens", step=256)
                            comb_tts_guidance = gr.Slider(1.0, 10.0, 3.0, label="TTS Guidance", step=0.5)
                        
                        comb_temperature = gr.Slider(0.1, 2.0, 1.8, label="Voice Temperature", step=0.1)
                        comb_seed = gr.Number(label="Seed (optional)", precision=0)
                        
                        comb_generate_btn = gr.Button("🎬🎵 Generate Animation + Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>🎉 Combined Results</h3></div>')
                        
                        comb_gif_output = gr.Image(label="Generated GIF")
                        comb_video_output = gr.Video(label="Generated MP4")
                        comb_audio_output = gr.Audio(label="Generated Audio")
                        comb_status = gr.Textbox(label="Status", interactive=False, lines=8)
                        comb_info = gr.JSON(label="Generation Info", visible=False)
        
        # Event handlers for character selection updates
        anim_character.change(
            update_prompt_suggestion,
            inputs=[anim_character],
            outputs=[anim_prompt]
        )
        
        comb_character.change(
            update_prompt_suggestion,
            inputs=[comb_character],
            outputs=[comb_prompt]
        )
        
        comb_character.change(
            update_dialogue_suggestion,
            inputs=[comb_character],
            outputs=[comb_dialogue]
        )
        
        # Event handlers
        init_btn.click(
            initialize_models,
            outputs=[init_status]
        )
        
        anim_generate_btn.click(
            generate_animation_interface,
            inputs=[anim_character, anim_prompt, anim_negative, anim_frames, anim_fps, 
                   anim_width, anim_height, anim_guidance, anim_steps, anim_seed],
            outputs=[anim_gif_output, anim_video_output, anim_status, anim_info]
        )
        
        tts_generate_btn.click(
            generate_tts_interface,
            inputs=[tts_dialogue, tts_max_tokens, tts_guidance, tts_temperature, 
                   tts_top_p, tts_top_k, tts_seed],
            outputs=[tts_audio_output, tts_status, tts_info]
        )
        
        comb_generate_btn.click(
            generate_combined_interface,
            inputs=[comb_character, comb_prompt, comb_dialogue, comb_frames, comb_fps,
                   comb_width, comb_height, comb_guidance, comb_steps, comb_max_tokens,
                   comb_tts_guidance, comb_temperature, comb_seed],
            outputs=[comb_gif_output, comb_video_output, comb_audio_output, comb_status, comb_info]
        )
        
        # Footer with instructions
        gr.HTML("""
        <div style="text-align: center; padding: 20px; border-top: 1px solid #ddd; margin-top: 30px;">
            <h3>🎯 How to Use</h3>
            <p><strong>1. Initialize Models:</strong> Click the "Initialize Models" button first</p>
            <p><strong>2. Choose Characters:</strong> Select single character or "Both" for multi-character scenes</p>
            <p><strong>3. Enter Prompts:</strong> Describe what you want to see and hear</p>
            <p><strong>4. Adjust Settings:</strong> More frames/steps = higher quality (slower generation)</p>
            <p><strong>5. Generate:</strong> Click the generate button and wait for results</p>
            <hr>
            <h4>🎭 Multi-Character Mode:</h4>
            <p><strong>• Both Characters:</strong> Temo AND Felfel appear together in the same animation</p>
            <p><strong>• Character Interactions:</strong> Perfect for conversations and collaborations</p>
            <p><strong>• Equal Representation:</strong> Both characters get equal visual prominence</p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cartoon Animation Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Enable sharing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting Cartoon Animation Web Interface...")
    print(f"📡 Interface will be available at: http://{args.host}:{args.port}")
    print("🎭 Multi-character support: Both Temo and Felfel together!")
    
    interface = create_interface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        inbrowser=True
    ) 