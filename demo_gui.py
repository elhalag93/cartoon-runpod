#!/usr/bin/env python3
"""
Demo GUI for Cartoon Animation Studio
Shows all the controls for entering prompts and parameters
"""

import gradio as gr
import json
import time
from typing import Optional, Dict, Any, Tuple, Union, List

def demo_generate_animation(
    character_selection: str,
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
    
    # Handle character selection
    if character_selection == "Both (Multi-Character)":
        characters = ["temo", "felfel"]
        character_display = "temo_and_felfel"
    else:
        characters = [character_selection.lower()]
        character_display = character_selection.lower()
    
    # Create demo response
    result = {
        "task_type": "animation",
        "characters": characters,
        "character": character_display,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed if seed else 42,
        "status": "Demo mode - showing parameter controls",
        "multi_character": len(characters) > 1
    }
    
    if len(characters) > 1:
        status = f"✅ Demo Complete! MULTI-CHARACTER Mode:\n" \
                 f"Characters: {' + '.join(characters).upper()}\n" \
                 f"Prompt: {prompt[:50]}...\n" \
                 f"Frames: {num_frames} at {fps} FPS\n" \
                 f"Resolution: {width}x{height}\n" \
                 f"Guidance: {guidance_scale}\n" \
                 f"Steps: {num_inference_steps}\n" \
                 f"Seed: {result['seed']}\n" \
                 f"🎭 BOTH characters will appear together!"
    else:
        status = f"✅ Demo Complete! Single Character Mode:\n" \
                 f"Character: {character_display.upper()}\n" \
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
    seed: Optional[int]
) -> Tuple[str, str, str, str, str]:
    """Demo combined generation"""
    
    time.sleep(3)
    
    # Handle character selection
    if character_selection == "Both (Multi-Character)":
        characters = ["temo", "felfel"]
        character_display = "temo_and_felfel"
    else:
        characters = [character_selection.lower()]
        character_display = character_selection.lower()
    
    result = {
        "task_type": "combined",
        "characters": characters,
        "character": character_display,
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
        "status": "Demo mode - showing combined controls",
        "multi_character": len(characters) > 1
    }
    
    if len(characters) > 1:
        status = f"✅ Demo Combined Complete! MULTI-CHARACTER:\n" \
                 f"Characters: {' + '.join(characters).upper()}\n" \
                 f"Animation: {prompt[:30]}...\n" \
                 f"Dialogue: {dialogue_text[:30]}...\n" \
                 f"Resolution: {width}x{height}\n" \
                 f"Frames: {num_frames} at {fps} FPS\n" \
                 f"Animation Guidance: {guidance_scale}\n" \
                 f"TTS Guidance: {tts_guidance_scale}\n" \
                 f"Seed: {result['seed']}\n" \
                 f"🎭 BOTH characters appear together!"
    else:
        status = f"✅ Demo Combined Complete!\n" \
                 f"Character: {character_display.upper()}\n" \
                 f"Animation: {prompt[:30]}...\n" \
                 f"Dialogue: {dialogue_text[:30]}...\n" \
                 f"Resolution: {width}x{height}\n" \
                 f"Frames: {num_frames} at {fps} FPS\n" \
                 f"Animation Guidance: {guidance_scale}\n" \
                 f"TTS Guidance: {tts_guidance_scale}\n" \
                 f"Seed: {result['seed']}"
    
    info = json.dumps(result, indent=2)
    
    return None, None, None, status, info

def update_prompt_for_characters(character_selection: str) -> str:
    """Update prompt suggestions based on character selection"""
    if character_selection == "Both (Multi-Character)":
        return "temo and felfel characters working together on moon base, both characters clearly visible, temo in space suit on left, felfel in adventure gear on right, epic lighting, detailed cartoon style"
    elif character_selection == "Temo":
        return "temo character walking confidently on moon surface with epic cinematic lighting, detailed cartoon style, space helmet reflecting Earth"
    elif character_selection == "Felfel":
        return "felfel character exploring magical crystal cave with epic cinematic lighting, ultra detailed cartoon style, masterpiece quality"
    else:
        return ""

def update_dialogue_for_characters(character_selection: str) -> str:
    """Update dialogue suggestions based on character selection"""
    if character_selection == "Both (Multi-Character)":
        return "[S1] Temo: Welcome to our lunar base, Felfel! [S2] Felfel: This technology is incredible, Temo! [S1] Temo: Let's explore together!"
    elif character_selection == "Temo":
        return "[S1] Greetings from the lunar surface with crystal clear audio! [S2] What an amazing space adventure!"
    elif character_selection == "Felfel":
        return "[S1] Look at this magical crystal cave! [S2] The formations are absolutely breathtaking!"
    else:
        return ""

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
    .demo-notice {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
    }
    .multi-character-notice {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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
            <p><strong>✨ NEW:</strong> Multi-Character Support - Both characters together!</p>
        </div>
        """)
        
        # Demo notice
        gr.HTML("""
        <div class="demo-notice">
            <strong>📋 DEMO MODE:</strong> This shows all the parameter controls you'll have in the full system.
            In production, these controls will generate actual animations and audio!
        </div>
        """)
        
        # Multi-character notice
        gr.HTML("""
        <div class="multi-character-notice">
            <strong>🎭 MULTI-CHARACTER FEATURE:</strong> You can now select "Both" to have Temo AND Felfel appear together in the same animation! 
            Perfect for character interactions and conversations.
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
                            choices=["Temo", "Felfel", "Both (Multi-Character)"],
                            value="Temo",
                            label="Character Selection",
                            info="Choose single character or both together"
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
                        gr.HTML('<div class="multi-character-card"><h3>🎭 Combined Content Controls</h3></div>')
                        
                        comb_character = gr.Dropdown(
                            choices=["Temo", "Felfel", "Both (Multi-Character)"],
                            value="Both (Multi-Character)",
                            label="Character Selection",
                            info="Choose single character or both together"
                        )
                        
                        comb_prompt = gr.Textbox(
                            label="🎬 Animation Prompt (YOUR VISUAL)",
                            placeholder="Describe the character's actions...",
                            lines=2,
                            value="temo and felfel characters working together on moon base, both characters clearly visible, temo in space suit on left, felfel in adventure gear on right, epic lighting, detailed cartoon style"
                        )
                        
                        comb_dialogue = gr.Textbox(
                            label="🎵 Dialogue Text (YOUR AUDIO)",
                            placeholder="Enter speech with [S1] [S2] tags...",
                            lines=3,
                            value="[S1] Temo: Welcome to our lunar base, Felfel! [S2] Felfel: This technology is incredible, Temo! [S1] Temo: Let's explore together!",
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
        
        # Event handlers for character selection updates
        anim_character.change(
            update_prompt_for_characters,
            inputs=[anim_character],
            outputs=[anim_prompt]
        )
        
        comb_character.change(
            update_prompt_for_characters,
            inputs=[comb_character],
            outputs=[comb_prompt]
        )
        
        comb_character.change(
            update_dialogue_for_characters,
            inputs=[comb_character],
            outputs=[comb_dialogue]
        )
        
        # Event handlers for generation
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
            <p><strong>1. Choose Characters:</strong> Select single character or "Both" for multi-character scenes</p>
            <p><strong>2. Enter Your Prompt:</strong> Describe what you want the character(s) to do</p>
            <p><strong>3. Add Dialogue:</strong> Use [S1] and [S2] tags for different speakers</p>
            <p><strong>4. Adjust Quality:</strong> More frames & steps = higher quality (slower generation)</p>
            <p><strong>5. Set Resolution:</strong> 1024x1024 for ultra quality, 512x512 for speed</p>
            <p><strong>6. Use Seeds:</strong> Same seed = identical results for reproducibility</p>
            <hr>
            <h4>🎭 Multi-Character Mode:</h4>
            <p><strong>• Both Characters:</strong> Temo AND Felfel appear together in the same animation</p>
            <p><strong>• Character Interactions:</strong> Perfect for conversations and collaborations</p>
            <p><strong>• Equal Representation:</strong> Both characters get equal visual prominence</p>
            <hr>
            <p><em>💡 In production mode, these controls generate real videos and audio!</em></p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("🎬 Starting Cartoon Animation Studio Demo...")
    print("📋 This demo shows all the parameter controls you'll have")
    print("🎭 NEW: Multi-character support - both Temo and Felfel together!")
    print("🚀 In production, these will generate real animations!")
    
    demo_interface = create_demo_interface()
    demo_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    ) 