#!/usr/bin/env python3
"""
Demo GUI for Cartoon Animation Studio
Shows all the controls for entering prompts and parameters
"""

import gradio as gr
import json
import time
from typing import Optional, Dict, Any, Tuple

def demo_generate_animation(
    character: str,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    fps: int,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: Optional[int]
) -> Tuple[str, str, str]:
    """Demo animation generation - shows what inputs you can control"""
    
    # Simulate processing time
    time.sleep(2)
    
    # Create demo response
    result = {
        "task_type": "animation",
        "character": character,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed if seed else 42,
        "status": "Demo mode - showing parameter controls"
    }
    
    status = f"✅ Demo Complete! Your settings:\n" \
             f"Character: {character}\n" \
             f"Prompt: {prompt[:50]}...\n" \
             f"Frames: {num_frames} at {fps} FPS\n" \
             f"Resolution: {width}x{height}\n" \
             f"Guidance: {guidance_scale}\n" \
             f"Steps: {num_inference_steps}\n" \
             f"Seed: {result['seed']}"
    
    info = json.dumps(result, indent=2)
    
    return None, None, status, info

def demo_generate_tts(
    dialogue_text: str,
    max_new_tokens: int,
    guidance_scale: float,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int]
) -> Tuple[str, str, str]:
    """Demo TTS generation"""
    
    time.sleep(1.5)
    
    result = {
        "task_type": "tts",
        "dialogue_text": dialogue_text,
        "max_new_tokens": max_new_tokens,
        "guidance_scale": guidance_scale,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed if seed else 84,
        "status": "Demo mode - showing TTS controls"
    }
    
    status = f"✅ Demo TTS Complete!\n" \
             f"Dialogue: {dialogue_text[:50]}...\n" \
             f"Max Tokens: {max_new_tokens}\n" \
             f"Guidance: {guidance_scale}\n" \
             f"Temperature: {temperature}\n" \
             f"Top P: {top_p}, Top K: {top_k}\n" \
             f"Seed: {result['seed']}"
    
    info = json.dumps(result, indent=2)
    
    return None, status, info

def demo_generate_combined(
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
    seed: Optional[int]
) -> Tuple[str, str, str, str, str]:
    """Demo combined generation"""
    
    time.sleep(3)
    
    result = {
        "task_type": "combined",
        "character": character,
        "prompt": prompt,
        "dialogue_text": dialogue_text,
        "animation_settings": {
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps
        },
        "tts_settings": {
            "max_new_tokens": max_new_tokens,
            "guidance_scale": tts_guidance_scale,
            "temperature": temperature
        },
        "seed": seed if seed else 168,
        "status": "Demo mode - showing combined controls"
    }
    
    status = f"✅ Demo Combined Complete!\n" \
             f"Character: {character}\n" \
             f"Animation: {prompt[:30]}...\n" \
             f"Dialogue: {dialogue_text[:30]}...\n" \
             f"Resolution: {width}x{height}\n" \
             f"Frames: {num_frames} at {fps} FPS\n" \
             f"Animation Guidance: {guidance_scale}\n" \
             f"TTS Guidance: {tts_guidance_scale}\n" \
             f"Seed: {result['seed']}"
    
    info = json.dumps(result, indent=2)
    
    return None, None, None, status, info

def create_demo_interface():
    """Create the demo interface showing all controls"""
    
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
    .demo-notice {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    """
    
    with gr.Blocks(css=css, title="🎬 Cartoon Animation Studio - Demo", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🎬 Cartoon Animation Studio - Demo Mode</h1>
            <p>See all the controls for entering prompts and parameters</p>
            <p><strong>Characters:</strong> Temo (Space Explorer) & Felfel (Adventure Seeker)</p>
        </div>
        """)
        
        # Demo notice
        gr.HTML("""
        <div class="demo-notice">
            <strong>📋 DEMO MODE:</strong> This shows all the parameter controls you'll have in the full system.
            In production, these controls will generate actual animations and audio!
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Animation Only Tab
            with gr.TabItem("🎬 Animation Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎭 Character & Prompt Controls</h3></div>')
                        
                        anim_character = gr.Dropdown(
                            choices=["temo", "felfel"],
                            value="temo",
                            label="Character",
                            info="Choose Temo (space) or Felfel (adventure)"
                        )
                        
                        anim_prompt = gr.Textbox(
                            label="🎬 Animation Prompt (YOUR INPUT HERE)",
                            placeholder="Enter your animation description here...\nExample: temo character walking on moon surface with epic lighting, detailed cartoon style",
                            lines=3,
                            value="temo character walking confidently on moon surface with epic cinematic lighting, detailed cartoon style, space helmet reflecting Earth"
                        )
                        
                        anim_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the animation...",
                            value="blurry, low quality, distorted, ugly, bad anatomy",
                            lines=2
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Quality & Performance Controls</h3></div>')
                        
                        with gr.Row():
                            anim_frames = gr.Slider(8, 48, 32, label="🎞️ Frames (More = Smoother)", step=1)
                            anim_fps = gr.Slider(8, 24, 16, label="🎬 FPS (Speed)", step=1)
                        
                        with gr.Row():
                            anim_width = gr.Slider(512, 1024, 1024, label="📐 Width (Ultra Quality)", step=64)
                            anim_height = gr.Slider(512, 1024, 1024, label="📐 Height (Ultra Quality)", step=64)
                        
                        with gr.Row():
                            anim_guidance = gr.Slider(8.0, 15.0, 12.0, label="🎯 Guidance (Follow Prompt)", step=0.5)
                            anim_steps = gr.Slider(30, 75, 50, label="🔧 Inference Steps (Quality)", step=5)
                        
                        anim_seed = gr.Number(
                            label="🎲 Seed (Reproducibility)",
                            info="Leave empty for random, or enter number like 42",
                            precision=0
                        )
                        
                        anim_generate_btn = gr.Button("🎬 Generate Animation", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>📺 Animation Results</h3></div>')
                        
                        anim_gif_output = gr.Image(label="Generated GIF", type="filepath")
                        anim_video_output = gr.Video(label="Generated MP4 Video")
                        anim_status = gr.Textbox(label="📊 Generation Status", interactive=False, lines=8)
                        anim_info = gr.JSON(label="🔍 Detailed Settings", visible=True)
            
            # TTS Only Tab
            with gr.TabItem("🎵 Text-to-Speech"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎤 Speech Content Controls</h3></div>')
                        
                        tts_dialogue = gr.Textbox(
                            label="🎵 Dialogue Text (YOUR SPEECH HERE)",
                            placeholder="Enter your dialogue with speaker tags...\nExample: [S1] Hello from the moon! [S2] What an amazing adventure!",
                            lines=4,
                            value="[S1] Welcome to the ultra high quality text-to-speech system! [S2] Listen to the crystal clear audio generation with perfect pronunciation.",
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Voice Quality Controls</h3></div>')
                        
                        tts_max_tokens = gr.Slider(2048, 8192, 4096, label="🎙️ Max Tokens (Audio Length)", step=256)
                        tts_guidance = gr.Slider(2.0, 10.0, 5.0, label="🎯 TTS Guidance (Voice Quality)", step=0.5)
                        tts_temperature = gr.Slider(0.8, 2.0, 1.4, label="🌡️ Temperature (Voice Variation)", step=0.1)
                        
                        with gr.Row():
                            tts_top_p = gr.Slider(0.85, 0.95, 0.9, label="📊 Top P", step=0.01)
                            tts_top_k = gr.Slider(40, 100, 60, label="📊 Top K", step=5)
                        
                        tts_seed = gr.Number(
                            label="🎲 Seed (Voice Consistency)",
                            info="Leave empty for random",
                            precision=0
                        )
                        
                        tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>🔊 Audio Results</h3></div>')
                        
                        tts_audio_output = gr.Audio(label="Generated Speech Audio")
                        tts_status = gr.Textbox(label="📊 Generation Status", interactive=False, lines=6)
                        tts_info = gr.JSON(label="🔍 TTS Settings", visible=True)
            
            # Combined Tab
            with gr.TabItem("🎬🎵 Animation + Speech (BEST)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="character-card"><h3>🎭 Combined Content Controls</h3></div>')
                        
                        comb_character = gr.Dropdown(
                            choices=["temo", "felfel"],
                            value="felfel",
                            label="Character"
                        )
                        
                        comb_prompt = gr.Textbox(
                            label="🎬 Animation Prompt (YOUR VISUAL)",
                            placeholder="Describe the character's actions...",
                            lines=2,
                            value="felfel character exploring magical crystal cave with epic cinematic lighting, ultra detailed cartoon style, masterpiece quality"
                        )
                        
                        comb_dialogue = gr.Textbox(
                            label="🎵 Dialogue Text (YOUR AUDIO)",
                            placeholder="Enter speech with [S1] [S2] tags...",
                            lines=3,
                            value="[S1] Felfel discovers an incredible crystal cave! [S2] Look at these magnificent formations sparkling in the light. [S1] (gasps in wonder) This is absolutely breathtaking!",
                            info="Use [S1] and [S2] tags for different speakers"
                        )
                        
                        gr.HTML('<div class="character-card"><h3>⚙️ Combined Quality Controls</h3></div>')
                        
                        with gr.Row():
                            comb_frames = gr.Slider(16, 48, 32, label="🎞️ Animation Frames", step=1)
                            comb_fps = gr.Slider(12, 24, 16, label="🎬 FPS", step=1)
                        
                        with gr.Row():
                            comb_width = gr.Slider(768, 1024, 1024, label="📐 Width (Ultra)", step=64)
                            comb_height = gr.Slider(768, 1024, 1024, label="📐 Height (Ultra)", step=64)
                        
                        with gr.Row():
                            comb_guidance = gr.Slider(8.0, 15.0, 12.0, label="🎯 Animation Guidance", step=0.5)
                            comb_steps = gr.Slider(30, 75, 50, label="🔧 Inference Steps", step=5)
                        
                        with gr.Row():
                            comb_max_tokens = gr.Slider(2048, 8192, 4096, label="🎙️ Audio Tokens", step=256)
                            comb_tts_guidance = gr.Slider(2.0, 10.0, 5.0, label="🎯 TTS Guidance", step=0.5)
                        
                        comb_temperature = gr.Slider(0.8, 2.0, 1.4, label="🌡️ Voice Temperature", step=0.1)
                        
                        comb_seed = gr.Number(
                            label="🎲 Seed (Full Reproducibility)",
                            info="Same seed = identical results",
                            precision=0
                        )
                        
                        comb_generate_btn = gr.Button("🎬🎵 Generate Animation + Speech", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.HTML('<div class="result-container"><h3>🎉 Combined Results</h3></div>')
                        
                        comb_gif_output = gr.Image(label="Generated GIF Animation")
                        comb_video_output = gr.Video(label="Generated MP4 Video")
                        comb_audio_output = gr.Audio(label="Generated Speech Audio")
                        comb_status = gr.Textbox(label="📊 Generation Status", interactive=False, lines=10)
                        comb_info = gr.JSON(label="🔍 All Settings", visible=True)
        
        # Event handlers
        anim_generate_btn.click(
            demo_generate_animation,
            inputs=[anim_character, anim_prompt, anim_negative, anim_frames, anim_fps, 
                   anim_width, anim_height, anim_guidance, anim_steps, anim_seed],
            outputs=[anim_gif_output, anim_video_output, anim_status, anim_info]
        )
        
        tts_generate_btn.click(
            demo_generate_tts,
            inputs=[tts_dialogue, tts_max_tokens, tts_guidance, tts_temperature, 
                   tts_top_p, tts_top_k, tts_seed],
            outputs=[tts_audio_output, tts_status, tts_info]
        )
        
        comb_generate_btn.click(
            demo_generate_combined,
            inputs=[comb_character, comb_prompt, comb_dialogue, comb_frames, comb_fps,
                   comb_width, comb_height, comb_guidance, comb_steps, comb_max_tokens,
                   comb_tts_guidance, comb_temperature, comb_seed],
            outputs=[comb_gif_output, comb_video_output, comb_audio_output, comb_status, comb_info]
        )
        
        # Footer with instructions
        gr.HTML("""
        <div style="text-align: center; padding: 20px; border-top: 1px solid #ddd; margin-top: 30px;">
            <h3>🎯 How to Use These Controls</h3>
            <p><strong>1. Enter Your Prompt:</strong> Describe what you want the character to do</p>
            <p><strong>2. Add Dialogue:</strong> Use [S1] and [S2] tags for different speakers</p>
            <p><strong>3. Adjust Quality:</strong> More frames & steps = higher quality (slower generation)</p>
            <p><strong>4. Set Resolution:</strong> 1024x1024 for ultra quality, 512x512 for speed</p>
            <p><strong>5. Use Seeds:</strong> Same seed = identical results for reproducibility</p>
            <hr>
            <p><em>💡 In production mode, these controls generate real videos and audio!</em></p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("🎬 Starting Cartoon Animation Studio Demo...")
    print("📋 This demo shows all the parameter controls you'll have")
    print("🚀 In production, these will generate real animations!")
    
    demo_interface = create_demo_interface()
    demo_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    ) 