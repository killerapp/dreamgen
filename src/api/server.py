"""
FastAPI server for Continuous Image Generation
Provides REST API and WebSocket endpoints for the Next.js frontend
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.generators.mock_image_generator import MockImageGenerator
from src.generators.prompt_generator import PromptGenerator
from src.plugins import ensure_initialized, plugin_manager
from src.utils.config import Config
from src.utils.storage import save_image_and_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Continuous Image Generator API",
    description="API for AI-powered image generation with plugin architecture",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",  # Next.js on custom port
        "http://localhost:3000",  # Next.js default dev server
        "http://localhost:3001",  # Alternative port
        "https://imagegen.agenticinsights.com",  # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = Config()
ensure_initialized(config)
state = {"use_mock": False}  # Use real Flux generation with GPU

# Register plugins - simplified for now
# TODO: Properly integrate plugins once their interfaces are standardized

# Output directory setup
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving generated images
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")


# Pydantic models
class GenerateRequest(BaseModel):
    """Request model for image generation"""

    prompt: Optional[str] = Field(None, description="Optional custom prompt")
    use_mock: bool = Field(False, description="Use mock generator for testing")
    enable_plugins: bool = Field(True, description="Enable plugin enhancements")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerateResponse(BaseModel):
    """Response model for image generation"""

    id: str = Field(..., description="Unique generation ID")
    prompt: str = Field(..., description="Final prompt used")
    image_path: str = Field(..., description="Path to generated image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    created_at: str = Field(..., description="ISO timestamp")


class PluginInfo(BaseModel):
    """Plugin information model"""

    name: str
    enabled: bool
    description: str


class PluginToggleRequest(BaseModel):
    """Request model for enabling or disabling a plugin"""

    enabled: bool


class SystemStatus(BaseModel):
    """System status model"""

    status: str = Field(..., description="System status (ready, busy, error)")
    backend: str = Field(..., description="Active backend (mock, flux)")
    plugins_enabled: bool
    active_plugins: List[str]
    gpu_available: bool
    ollama_available: bool


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass  # Handle disconnected clients


manager = ConnectionManager()


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Continuous Image Generator API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "by": "Agentic Insights",
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status and configuration"""
    # Check if CUDA/MPS is available
    try:
        import torch

        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    except:
        gpu_available = False

    # Check Ollama availability
    try:
        import ollama

        ollama_available = True
    except:
        ollama_available = False

    # Determine which Flux model is being used
    if state["use_mock"]:
        backend_name = "mock"
    else:
        flux_model = config.model.flux_model
        if "schnell" in flux_model.lower():
            backend_name = "flux-schnell"
        elif "dev" in flux_model.lower():
            backend_name = "flux-dev"
        else:
            backend_name = "flux"

    return SystemStatus(
        status="ready",
        backend=backend_name,
        plugins_enabled=True,
        active_plugins=[name for name, info in plugin_manager.plugins.items() if info.enabled],
        gpu_available=gpu_available,
        ollama_available=ollama_available,
    )


@app.get("/api/plugins", response_model=List[PluginInfo])
async def get_plugins():
    """Get list of available plugins and their states"""
    plugins = []
    for name, info in plugin_manager.plugins.items():
        plugins.append(PluginInfo(name=name, enabled=info.enabled, description=info.description))
    return plugins


@app.post("/api/plugins/{plugin_name}", response_model=PluginInfo)
async def set_plugin_state(plugin_name: str, request: PluginToggleRequest):
    """Enable or disable a plugin"""
    if plugin_name not in plugin_manager.plugins:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not found")

    if request.enabled:
        plugin_manager.enable_plugin(plugin_name)
        if plugin_name not in config.plugins.enabled_plugins:
            config.plugins.enabled_plugins.append(plugin_name)
    else:
        plugin_manager.disable_plugin(plugin_name)
        if plugin_name in config.plugins.enabled_plugins:
            config.plugins.enabled_plugins.remove(plugin_name)

    info = plugin_manager.plugins[plugin_name]
    return PluginInfo(name=plugin_name, enabled=info.enabled, description=info.description)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate a single image"""
    generation_id = str(uuid.uuid4())

    try:
        # Broadcast start event
        await manager.broadcast(
            json.dumps(
                {
                    "type": "generation_started",
                    "id": generation_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        )

        # Generate prompt if not provided
        if request.prompt:
            final_prompt = request.prompt
        else:
            prompt_gen = PromptGenerator(config)
            final_prompt = await prompt_gen.generate_prompt()

            # Broadcast prompt generated event
            await manager.broadcast(
                json.dumps(
                    {"type": "prompt_generated", "id": generation_id, "prompt": final_prompt}
                )
            )

        # Generate image
        logger.info(
            f"Generation mode - request.use_mock: {request.use_mock}, state['use_mock']: {state['use_mock']}"
        )

        if request.use_mock or state["use_mock"]:
            logger.info("Using MOCK image generator")
            image_gen = MockImageGenerator(config)
        else:
            logger.info("Using REAL Flux image generator")

            # Broadcast model loading event
            await manager.broadcast(
                json.dumps(
                    {
                        "type": "model_loading",
                        "id": generation_id,
                        "message": "Loading Flux model (this may take several minutes on first run)...",
                    }
                )
            )

            # Import real generator only if needed
            from src.generators.image_generator import ImageGenerator

            try:
                image_gen = ImageGenerator(config)
            except MemoryError as e:
                error_msg = "Insufficient memory to load Flux model. This model requires significant RAM/VRAM."
                logger.error(f"Memory error loading Flux model: {str(e)}")
                await manager.broadcast(
                    json.dumps(
                        {"type": "generation_error", "id": generation_id, "error": error_msg}
                    )
                )
                raise HTTPException(status_code=507, detail=error_msg)
            except Exception as e:
                error_msg = f"Failed to load Flux model: {str(e)}"
                logger.error(error_msg)
                await manager.broadcast(
                    json.dumps(
                        {"type": "generation_error", "id": generation_id, "error": error_msg}
                    )
                )
                raise HTTPException(status_code=500, detail=error_msg)

        # Generate the image
        image = await image_gen.generate(final_prompt, seed=request.seed)

        # Save image and prompt
        image_path = save_image_and_prompt(image, final_prompt)

        # Create relative path for API response
        relative_path = f"/images/{image_path.relative_to(OUTPUT_DIR).as_posix()}"

        # Broadcast completion event
        await manager.broadcast(
            json.dumps(
                {
                    "type": "generation_completed",
                    "id": generation_id,
                    "image_path": relative_path,
                    "prompt": final_prompt,
                }
            )
        )

        # Determine backend name
        if request.use_mock or state["use_mock"]:
            backend_name = "mock"
        else:
            flux_model = config.model.flux_model
            if "schnell" in flux_model.lower():
                backend_name = "flux-schnell"
            elif "dev" in flux_model.lower():
                backend_name = "flux-dev"
            else:
                backend_name = "flux"

        return GenerateResponse(
            id=generation_id,
            prompt=final_prompt,
            image_path=relative_path,
            metadata={
                "backend": backend_name,
                "plugins_used": [
                    name for name, info in plugin_manager.plugins.items() if info.enabled
                ],
                "seed": request.seed,
            },
            created_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")

        # Broadcast error event
        await manager.broadcast(
            json.dumps({"type": "generation_error", "id": generation_id, "error": str(e)})
        )

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gallery")
async def get_gallery(limit: int = 50, offset: int = 0):
    """Get list of generated images"""
    images = []

    # Get all image files from output directory
    image_files = sorted(OUTPUT_DIR.glob("**/*.png"), key=lambda x: x.stat().st_mtime, reverse=True)

    # Apply pagination
    paginated_files = image_files[offset : offset + limit]

    for image_file in paginated_files:
        # Check if corresponding prompt file exists
        prompt_file = image_file.with_suffix(".txt")
        prompt = ""
        if prompt_file.exists():
            async with aiofiles.open(prompt_file, "r") as f:
                prompt = await f.read()

        images.append(
            {
                "path": f"/images/{image_file.relative_to(OUTPUT_DIR).as_posix()}",
                "prompt": prompt.strip(),
                "created_at": datetime.fromtimestamp(image_file.stat().st_mtime).isoformat(),
                "size": image_file.stat().st_size,
            }
        )

    return {"images": images, "total": len(image_files), "limit": limit, "offset": offset}


@app.delete("/api/gallery/{image_path:path}")
async def delete_image(image_path: str):
    """Delete an image from the gallery"""
    full_path = (OUTPUT_DIR / image_path).resolve()
    output_root = OUTPUT_DIR.resolve()

    if not full_path.is_relative_to(output_root):
        raise HTTPException(status_code=400, detail="Invalid image path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete image and prompt files
    full_path.unlink()
    prompt_path = full_path.with_suffix(".txt")
    if prompt_path.exists():
        prompt_path.unlink()

    return {"message": "Image deleted successfully"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Echo back or handle commands
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/batch")
async def batch_generate(count: int = 5, delay: int = 0):
    """Generate multiple images in batch"""
    batch_id = str(uuid.uuid4())
    results = []

    for i in range(count):
        if delay > 0 and i > 0:
            await asyncio.sleep(delay)

        try:
            # Generate each image
            request = GenerateRequest(use_mock=state["use_mock"])
            result = await generate_image(request)
            results.append(result.dict())
        except Exception as e:
            logger.error(f"Batch generation {i+1}/{count} failed: {str(e)}")
            results.append({"error": str(e)})

    return {"batch_id": batch_id, "count": count, "results": results}


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
