#!/usr/bin/env python3
"""
FastAPI Server for Cartoon Animation Generation
Provides REST API endpoints for animation and TTS generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import tempfile
import base64
from pathlib import Path
import asyncio
import logging

# Import local modules
from src.handler import generate_animation, generate_tts, generate_combined
from src.handler import load_animation_pipeline, load_tts_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🎬 Cartoon Animation API",
    description="Generate cartoon character animations with voice using AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AnimationRequest(BaseModel):
    character: Optional[str] = Field(None, description="Single character name (temo, felfel) - for backward compatibility")
    characters: Optional[List[str]] = Field(None, description="List of characters for multi-character animations")
    prompt: str = Field(..., description="Animation prompt")
    negative_prompt: str = Field("blurry, low quality", description="Negative prompt")
    num_frames: int = Field(32, ge=16, le=64, description="Number of frames (ultra quality)")
    fps: int = Field(16, ge=12, le=24, description="Frames per second (ultra quality)")
    width: int = Field(1024, ge=768, le=1536, description="Video width (ultra quality 1024x1024)")
    height: int = Field(1024, ge=768, le=1536, description="Video height (ultra quality 1024x1024)")
    guidance_scale: float = Field(12.0, ge=8.0, le=25.0, description="Guidance scale (ultra quality)")
    num_inference_steps: int = Field(50, ge=30, le=75, description="Inference steps (ultra quality)")
    seed: Optional[int] = Field(None, description="Random seed")

class TTSRequest(BaseModel):
    dialogue_text: str = Field(..., description="Text to convert to speech")
    max_new_tokens: int = Field(4096, ge=2048, le=8192, description="Max tokens (ultra quality)")
    guidance_scale: float = Field(5.0, ge=2.0, le=10.0, description="Guidance scale (ultra quality)")
    temperature: float = Field(1.4, ge=0.8, le=2.0, description="Temperature (ultra quality)")
    top_p: float = Field(0.9, ge=0.85, le=0.95, description="Top P (ultra quality)")
    top_k: int = Field(60, ge=40, le=100, description="Top K (ultra quality)")
    seed: Optional[int] = Field(None, description="Random seed")

class CombinedRequest(BaseModel):
    character: Optional[str] = Field(None, description="Single character name - for backward compatibility")
    characters: Optional[List[str]] = Field(None, description="List of characters for multi-character animations")
    prompt: str = Field(..., description="Animation prompt")
    dialogue_text: str = Field(..., description="Text to convert to speech")
    num_frames: int = Field(32, ge=16, le=64, description="Number of frames (ultra quality)")
    fps: int = Field(16, ge=12, le=24, description="Frames per second (ultra quality)")
    width: int = Field(1024, ge=768, le=1536, description="Video width (ultra quality 1024x1024)")
    height: int = Field(1024, ge=768, le=1536, description="Video height (ultra quality 1024x1024)")
    guidance_scale: float = Field(12.0, ge=8.0, le=25.0, description="Animation guidance (ultra quality)")
    num_inference_steps: int = Field(50, ge=30, le=75, description="Inference steps (ultra quality)")
    max_new_tokens: int = Field(4096, ge=2048, le=8192, description="Max tokens (ultra quality)")
    tts_guidance_scale: float = Field(5.0, ge=2.0, le=10.0, description="TTS guidance (ultra quality)")
    temperature: float = Field(1.4, ge=0.8, le=2.0, description="Temperature (ultra quality)")
    seed: Optional[int] = Field(None, description="Random seed")

class GenerationResponse(BaseModel):
    task_type: str
    gif: Optional[str] = None
    mp4: Optional[str] = None
    audio: Optional[str] = None
    seed: Optional[int] = None
    memory_usage: Optional[Dict[str, float]] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    models_loaded: Dict[str, bool]

# Global variables for model status
models_status = {
    "animation_pipeline": False,
    "tts_model": False
}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Cartoon Animation API...")
    
    # Initialize models in background
    try:
        logger.info("🔄 Loading animation pipeline...")
        load_animation_pipeline()
        models_status["animation_pipeline"] = True
        logger.info("✅ Animation pipeline loaded")
        
        logger.info("🔄 Loading TTS model...")
        load_tts_model()
        models_status["tts_model"] = True
        logger.info("✅ TTS model loaded")
        
        logger.info("🎉 All models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")

@app.get("/", response_model=StatusResponse)
async def root():
    """API status endpoint"""
    return StatusResponse(
        status="running",
        message="Cartoon Animation API is running",
        models_loaded=models_status
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models": models_status}

@app.post("/api/animation", response_model=GenerationResponse)
async def generate_animation_endpoint(request: AnimationRequest):
    """Generate character animation"""
    try:
        if not models_status["animation_pipeline"]:
            raise HTTPException(status_code=503, detail="Animation pipeline not loaded")
        
        # Handle character selection (backward compatibility + multi-character support)
        if request.characters:
            characters = request.characters
            char_display = " + ".join(characters)
        elif request.character:
            characters = [request.character]
            char_display = request.character
        else:
            raise HTTPException(status_code=400, detail="Either 'character' or 'characters' must be provided")
        
        logger.info(f"🎬 Generating animation for {char_display}")
        
        # Convert request to dict and update characters
        params = request.dict()
        params["characters"] = characters
        # Remove the old single character field to avoid conflicts
        if "character" in params:
            del params["character"]
        
        # Generate animation
        result = generate_animation(**params)
        
        # Encode files to base64
        response = GenerationResponse(
            task_type="animation",
            seed=result.get("seed")
        )
        
        if "gif_path" in result:
            with open(result["gif_path"], "rb") as f:
                response.gif = base64.b64encode(f.read()).decode("utf-8")
        
        if "mp4_path" in result:
            with open(result["mp4_path"], "rb") as f:
                response.mp4 = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info("✅ Animation generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating animation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts", response_model=GenerationResponse)
async def generate_tts_endpoint(request: TTSRequest):
    """Generate text-to-speech"""
    try:
        if not models_status["tts_model"]:
            raise HTTPException(status_code=503, detail="TTS model not loaded")
        
        logger.info(f"🎵 Generating TTS for: {request.dialogue_text[:50]}...")
        
        # Convert request to dict
        params = request.dict()
        
        # Generate TTS
        result = generate_tts(**params)
        
        # Encode audio to base64
        response = GenerationResponse(
            task_type="tts",
            seed=result.get("seed")
        )
        
        if "audio_path" in result:
            with open(result["audio_path"], "rb") as f:
                response.audio = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info("✅ TTS generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/combined", response_model=GenerationResponse)
async def generate_combined_endpoint(request: CombinedRequest):
    """Generate combined animation and TTS"""
    try:
        if not models_status["animation_pipeline"] or not models_status["tts_model"]:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Handle character selection (backward compatibility + multi-character support)
        if request.characters:
            characters = request.characters
            char_display = " + ".join(characters)
        elif request.character:
            characters = [request.character]
            char_display = request.character
        else:
            raise HTTPException(status_code=400, detail="Either 'character' or 'characters' must be provided")
        
        logger.info(f"🎬🎵 Generating combined for {char_display}")
        
        # Convert request to dict and update characters
        params = request.dict()
        params["characters"] = characters
        # Remove the old single character field to avoid conflicts
        if "character" in params:
            del params["character"]
        
        # Generate combined
        result = generate_combined(**params)
        
        # Encode files to base64
        response = GenerationResponse(
            task_type="combined",
            seed=result.get("seed")
        )
        
        if "gif_path" in result:
            with open(result["gif_path"], "rb") as f:
                response.gif = base64.b64encode(f.read()).decode("utf-8")
        
        if "mp4_path" in result:
            with open(result["mp4_path"], "rb") as f:
                response.mp4 = base64.b64encode(f.read()).decode("utf-8")
        
        if "audio_path" in result:
            with open(result["audio_path"], "rb") as f:
                response.audio = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info("✅ Combined generation successful")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in combined generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/characters")
async def get_characters():
    """Get available characters"""
    return {
        "characters": [
            {
                "name": "temo",
                "description": "Space explorer character",
                "default_prompt": "temo character walking on moon surface, space adventure, detailed cartoon style"
            },
            {
                "name": "felfel",
                "description": "Adventure character",
                "default_prompt": "felfel character exploring magical forest, fantasy adventure, detailed cartoon style"
            }
        ],
        "multi_character_support": {
            "description": "Both characters can appear together in the same animation",
            "usage": "Use 'characters': ['temo', 'felfel'] for multi-character scenes",
            "default_prompt": "temo and felfel characters working together on moon base, both characters clearly visible, epic lighting, detailed cartoon style"
        }
    }

@app.get("/api/examples")
async     def get_examples():
        """Get example requests"""
        return {
            "single_character_animation": {
                "character": "temo",
                "prompt": "temo character walking on moon surface, space adventure, ultra high quality, 4K resolution",
                "num_frames": 32,
                "fps": 16,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 12.0,
                "num_inference_steps": 50,
                "seed": 42
            },
            "multi_character_animation": {
                "characters": ["temo", "felfel"],
                "prompt": "temo and felfel characters working together on moon base, both characters clearly visible, temo in space suit on left, felfel in adventure gear on right, epic lighting, detailed cartoon style",
                "num_frames": 32,
                "fps": 16,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 12.0,
                "num_inference_steps": 50,
                "seed": 42
            },
            "tts_example": {
                "dialogue_text": "[S1] Hello from the moon with crystal clear audio! [S2] What an amazing ultra quality adventure!",
                "max_new_tokens": 4096,
                "guidance_scale": 5.0,
                "temperature": 1.4,
                "seed": 42
            },
            "single_character_combined": {
                "character": "temo",
                "prompt": "temo character waving hello from moon, ultra high quality, masterpiece animation",
                "dialogue_text": "[S1] Greetings from the lunar surface with perfect audio!",
                "num_frames": 32,
                "fps": 16,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 12.0,
                "num_inference_steps": 50,
                "max_new_tokens": 4096,
                "tts_guidance_scale": 5.0,
                "temperature": 1.4,
                "seed": 42
            },
            "multi_character_combined": {
                "characters": ["temo", "felfel"],
                "prompt": "temo and felfel characters working together on moon base, both characters clearly visible, epic lighting, detailed cartoon style",
                "dialogue_text": "[S1] Temo: Welcome to our lunar base, Felfel! [S2] Felfel: This technology is incredible, Temo! [S1] Temo: Let's explore together!",
                "num_frames": 32,
                "fps": 16,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 12.0,
                "num_inference_steps": 50,
                "max_new_tokens": 4096,
                "tts_guidance_scale": 5.0,
                "temperature": 1.4,
                "seed": 42
            }
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cartoon Animation API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print("🚀 Starting Cartoon Animation API Server...")
    print(f"📡 API will be available at: http://{args.host}:{args.port}")
    print(f"📚 Documentation at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    ) 