#!/usr/bin/env python3
"""
Simple GUI for Cartoon Animation Studio
Shows all the controls for entering prompts and parameters
"""

import gradio as gr
import json
import time

def demo_animation(character, prompt, frames, fps, width, height, guidance, steps, seed):
    """Demo animation generation"""
    time.sleep(1)
    
    result = {
        "character": character,
        "prompt": prompt,
        "frames": frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "guidance_scale": guidance,
        "inference_steps": steps,
        "seed": seed if seed else 42
    }
    
    status = f"✅ Demo Animation Settings:\n" \
             f"Character: {character}\n" \
             f"Prompt: {prompt[:60]}...\n" \
             f"Frames: {frames} at {fps} FPS\n" \
             f"Resolution: {width}x{height}\n" \
             f"Guidance Scale: {guidance}\n" \
             f"Inference Steps: {steps}\n" \
             f"Seed: {result['seed']}"
    
    return status

def demo_tts(dialogue, max_tokens, guidance, temperature, seed):
    """Demo TTS generation"""
    time.sleep(1)
    
    result = {
        "dialogue": dialogue,
        "max_tokens": max_tokens,
        "guidance_scale": guidance,
        "temperature": temperature,
        "seed": seed if seed else 84
    }
    
    status = f"✅ Demo TTS Settings:\n" \
             f"Dialogue: {dialogue[:50]}...\n" \
             f"Max Tokens: {max_tokens}\n" \
             f"Guidance Scale: {guidance}\n" \
             f"Temperature: {temperature}\n" \
             f"Seed: {result['seed']}"
    
    return status

def demo_combined(character, prompt, dialogue, frames, fps, width, height, 
                 anim_guidance, steps, max_tokens, tts_guidance, temperature, seed):
    """Demo combined generation"""
    time.sleep(2)
    
    status = f"✅ Demo Combined Settings:\n" \
             f"Character: {character}\n" \
             f"Animation: {prompt[:40]}...\n" \
             f"Dialogue: {dialogue[:40]}...\n" \
             f"Resolution: {width}x{height}\n" \
             f"Frames: {frames} at {fps} FPS\n" \
             f"Animation Guidance: {anim_guidance}\n" \
             f"Inference Steps: {steps}\n" \
             f"TTS Guidance: {tts_guidance}\n" \
             f"Temperature: {temperature}\n" \
             f"Max Tokens: {max_tokens}\n" \
             f"Seed: {seed if seed else 168}"
    
    return status

# Create interface
with gr.Blocks(title="🎬 Cartoon Animation Controls") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🎬 Cartoon Animation Studio - Parameter Controls</h1>
        <p>This shows all the controls you'll have for entering prompts and adjusting quality</p>
        <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px;">
            <strong>📋 DEMO MODE:</strong> In production, these controls generate real animations and audio!
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # Animation Tab
        with gr.TabItem("🎬 Animation Generation"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>🎭 Animation Controls</h3>")
                    
                    anim_character = gr.Dropdown(
                        choices=["temo", "felfel"],
                        value="temo",
                        label="Character"
                    )
                    
                    anim_prompt = gr.Textbox(
                        label="🎬 YOUR ANIMATION PROMPT HERE",
                        value="temo character walking confidently on moon surface with epic cinematic lighting, detailed cartoon style, space helmet reflecting Earth",
                        lines=3,
                        info="Describe what you want the character to do"
                    )
                    
                    with gr.Row():
                        anim_frames = gr.Slider(8, 48, 32, label="Frames (More = Smoother)", step=1)
                        anim_fps = gr.Slider(8, 24, 16, label="FPS", step=1)
                    
                    with gr.Row():
                        anim_width = gr.Slider(512, 1024, 1024, label="Width", step=64)
                        anim_height = gr.Slider(512, 1024, 1024, label="Height", step=64)
                    
                    with gr.Row():
                        anim_guidance = gr.Slider(8.0, 15.0, 12.0, label="Guidance Scale", step=0.5)
                        anim_steps = gr.Slider(30, 75, 50, label="Inference Steps", step=5)
                    
                    anim_seed = gr.Number(label="Seed (optional)", precision=0)
                    
                    anim_btn = gr.Button("🎬 Generate Animation", variant="primary")
                
                with gr.Column():
                    gr.HTML("<h3>📊 Results</h3>")
                    anim_output = gr.Textbox(label="Demo Output", lines=10)
            
            anim_btn.click(
                demo_animation,
                inputs=[anim_character, anim_prompt, anim_frames, anim_fps, 
                       anim_width, anim_height, anim_guidance, anim_steps, anim_seed],
                outputs=anim_output
            )
        
        # TTS Tab
        with gr.TabItem("🎵 Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>🎤 TTS Controls</h3>")
                    
                    tts_dialogue = gr.Textbox(
                        label="🎵 YOUR DIALOGUE HERE",
                        value="[S1] Welcome to the ultra high quality text-to-speech system! [S2] Listen to the crystal clear audio generation.",
                        lines=3,
                        info="Use [S1] and [S2] for different speakers"
                    )
                    
                    tts_max_tokens = gr.Slider(2048, 8192, 4096, label="Max Tokens", step=256)
                    tts_guidance = gr.Slider(2.0, 10.0, 5.0, label="TTS Guidance Scale", step=0.5)
                    tts_temperature = gr.Slider(0.8, 2.0, 1.4, label="Temperature", step=0.1)
                    tts_seed = gr.Number(label="Seed (optional)", precision=0)
                    
                    tts_btn = gr.Button("🎵 Generate Speech", variant="primary")
                
                with gr.Column():
                    gr.HTML("<h3>📊 Results</h3>")
                    tts_output = gr.Textbox(label="Demo Output", lines=8)
            
            tts_btn.click(
                demo_tts,
                inputs=[tts_dialogue, tts_max_tokens, tts_guidance, tts_temperature, tts_seed],
                outputs=tts_output
            )
        
        # Combined Tab
        with gr.TabItem("🎬🎵 Animation + Speech"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<h3>🎭 Combined Controls</h3>")
                    
                    comb_character = gr.Dropdown(
                        choices=["temo", "felfel"],
                        value="felfel",
                        label="Character"
                    )
                    
                    comb_prompt = gr.Textbox(
                        label="🎬 Animation Prompt",
                        value="felfel character exploring magical crystal cave with epic lighting",
                        lines=2
                    )
                    
                    comb_dialogue = gr.Textbox(
                        label="🎵 Dialogue Text",
                        value="[S1] Felfel discovers an incredible crystal cave! [S2] Look at these magnificent formations!",
                        lines=2
                    )
                    
                    with gr.Row():
                        comb_frames = gr.Slider(16, 48, 32, label="Frames", step=1)
                        comb_fps = gr.Slider(12, 24, 16, label="FPS", step=1)
                    
                    with gr.Row():
                        comb_width = gr.Slider(768, 1024, 1024, label="Width", step=64)
                        comb_height = gr.Slider(768, 1024, 1024, label="Height", step=64)
                    
                    with gr.Row():
                        comb_anim_guidance = gr.Slider(8.0, 15.0, 12.0, label="Animation Guidance", step=0.5)
                        comb_steps = gr.Slider(30, 75, 50, label="Inference Steps", step=5)
                    
                    with gr.Row():
                        comb_max_tokens = gr.Slider(2048, 8192, 4096, label="Audio Tokens", step=256)
                        comb_tts_guidance = gr.Slider(2.0, 10.0, 5.0, label="TTS Guidance", step=0.5)
                    
                    comb_temperature = gr.Slider(0.8, 2.0, 1.4, label="Voice Temperature", step=0.1)
                    comb_seed = gr.Number(label="Seed (optional)", precision=0)
                    
                    comb_btn = gr.Button("🎬🎵 Generate Combined", variant="primary")
                
                with gr.Column():
                    gr.HTML("<h3>📊 Results</h3>")
                    comb_output = gr.Textbox(label="Demo Output", lines=12)
            
            comb_btn.click(
                demo_combined,
                inputs=[comb_character, comb_prompt, comb_dialogue, comb_frames, comb_fps,
                       comb_width, comb_height, comb_anim_guidance, comb_steps, 
                       comb_max_tokens, comb_tts_guidance, comb_temperature, comb_seed],
                outputs=comb_output
            )
    
    # Instructions
    gr.HTML("""
    <div style="text-align: center; padding: 20px; border-top: 1px solid #ddd; margin-top: 20px;">
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

if __name__ == "__main__":
    print("🎬 Starting Simple Animation Studio Demo...")
    print("📋 This shows all the parameter controls you'll have")
    print("🚀 Open your browser to see the interface!")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    ) 