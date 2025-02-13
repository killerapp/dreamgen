"""
Image generator using Flux 1.1 transformers model.
Includes support for meme text overlay.
"""
from pathlib import Path
from typing import Optional, Tuple, List
import os
import time
import logging
import re
import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont

from ..utils.error_handler import handle_errors, ModelError, ResourceError
from ..utils.memory_manager import MemoryManager
from ..utils.config import Config
from ..utils.metrics import GenerationMetrics

# Configure logging to suppress verbose output
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)


class ImageGenerator:
    def __init__(self, config: Config):
        """Initialize the image generator with configurable parameters.
        
        Args:
            model_variant: Which Flux model variant to use (full model path)
            cpu_only: Whether to force CPU-only mode
            height: Height of generated images (default from env or 768)
            width: Width of generated images (default from env or 1360)
            num_inference_steps: Number of denoising steps (default from env or 50)
            guidance_scale: Guidance scale for generation (default from env or 7.5)
            true_cfg_scale: True classifier-free guidance scale (default from env or 1.0)
            max_sequence_length: Max sequence length for text processing (default from env or 512)
        """
        # Font setup for meme text
        self.font_size = 72
        self.font = None
        try:
            # Try to load Impact font (standard meme font)
            self.font = ImageFont.truetype("impact.ttf", self.font_size)
        except OSError:
            # Fallback to default font if Impact not available
            print("Impact font not found, using default font")
            self.font = ImageFont.load_default()
        self.config = config
        self.model_name = config.model.flux_model
        self.height = config.image.height
        self.width = config.image.width
        self.num_inference_steps = config.image.num_inference_steps
        self.guidance_scale = config.image.guidance_scale
        self.true_cfg_scale = config.image.true_cfg_scale
        self.max_sequence_length = config.model.max_sequence_length
        self.pipe = None
        
        if not config.system.cpu_only and not torch.cuda.is_available():
            raise ResourceError(
                "GPU (CUDA) is not available. "
                "This model requires a GPU for efficient processing. "
                "If you want to run on CPU anyway, use --cpu-only flag"
            )
            
        self.device = "cpu" if config.system.cpu_only else "cuda"
        self.memory_manager = MemoryManager(self.device)
        
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            torch.cuda.set_device(0)
            self.memory_manager.optimize_memory_usage()
        else:
            print("WARNING: Running on CPU. This will be significantly slower.")
        
    def initialize(self, force_reinit: bool = False):
        """Initialize the Flux diffusion pipeline."""
        if force_reinit and self.pipe is not None:
            self.cleanup()
            
        if self.pipe is None:
            # Check and optimize memory before loading
            is_critical, status = self.memory_manager.check_memory_pressure()
            if is_critical:
                print(f"Memory status: {status}")
                self.memory_manager.optimize_memory_usage()
                
            print(f"Loading model on {self.device}...")
            
            # Set memory management environment variables
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            
            # Load model with memory optimizations
            self.pipe = DiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="balanced"
            )
            
            if self.device == "cuda":
                self._setup_gpu_optimizations()
    
    def _setup_gpu_optimizations(self):
        """Set up GPU-specific optimizations for the pipeline."""
        self.pipe.enable_attention_slicing()
        print("Enabled attention slicing")
        
        self.pipe.enable_vae_tiling()
        print("Enabled VAE tiling")
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception:
            print("Xformers optimization not available")
            
        allocated, reserved, total = self.memory_manager.get_gpu_memory_info()
        print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved (Total: {total:.2f} GB)")

    def _parse_meme_text(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse top and bottom meme text from the prompt."""
        # Look for text positioned at top/bottom
        top_match = re.search(r"text ['\"](.*?)['\"] (?:positioned |written |)at the top", prompt)
        bottom_match = re.search(r"text ['\"](.*?)['\"] (?:positioned |written |)at the bottom", prompt)
        
        top_text = top_match.group(1) if top_match else None
        bottom_text = bottom_match.group(1) if bottom_match else None
        
        return top_text, bottom_text
    
    def _add_text_outline(self, draw: ImageDraw, x: int, y: int, text: str, font: ImageFont, stroke_width: int = 3) -> None:
        """Add black outline to text for better visibility."""
        # Draw black outline
        for adj_x in range(-stroke_width, stroke_width + 1):
            for adj_y in range(-stroke_width, stroke_width + 1):
                draw.text((x + adj_x, y + adj_y), text, font=font, fill="black")
        # Draw white text on top
        draw.text((x, y), text, font=font, fill="white")
    
    def _add_text_overlay(self, image: Image, top_text: Optional[str], bottom_text: Optional[str]) -> Image:
        """Add meme text overlay to the image."""
        if not (top_text or bottom_text):
            return image
            
        # Create draw object
        draw = ImageDraw.Draw(image)
        
        # Calculate text sizes and positions
        padding = 20
        
        if top_text:
            # Adjust font size to fit width
            font_size = self.font_size
            while True:
                font = ImageFont.truetype("impact.ttf", font_size) if self.font != ImageFont.load_default() else self.font
                text_width = draw.textlength(top_text, font=font)
                if text_width <= image.width - 2 * padding or font_size <= 30:
                    break
                font_size -= 2
            
            # Draw top text
            text_height = font.getsize(top_text)[1]
            x = (image.width - text_width) // 2
            y = padding
            self._add_text_outline(draw, x, y, top_text, font)
        
        if bottom_text:
            # Adjust font size to fit width
            font_size = self.font_size
            while True:
                font = ImageFont.truetype("impact.ttf", font_size) if self.font != ImageFont.load_default() else self.font
                text_width = draw.textlength(bottom_text, font=font)
                if text_width <= image.width - 2 * padding or font_size <= 30:
                    break
                font_size -= 2
            
            # Draw bottom text
            text_height = font.getsize(bottom_text)[1]
            x = (image.width - text_width) // 2
            y = image.height - text_height - padding
            self._add_text_outline(draw, x, y, bottom_text, font)
        
        return image
    
    @handle_errors(error_type=ModelError, retries=1, cleanup_func=lambda: self.memory_manager.optimize_memory_usage())
    async def generate_image(self, prompt: str, output_path: Path, force_reinit: bool = False) -> Tuple[Path, float, str]:
        """Generate an image from the given prompt."""
        metrics = GenerationMetrics(prompt=prompt, model_name=self.model_name)
        start_time = time.time()
        
        try:
            # Check memory and initialize
            is_critical, _ = self.memory_manager.check_memory_pressure()
            if is_critical:
                force_reinit = True
            self.initialize(force_reinit)
            
            # Generate image
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.device=="cuda"):
                image = self.pipe(
                    prompt=prompt,
                    prompt_2=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    true_cfg_scale=self.true_cfg_scale,
                    height=self.height,
                    width=self.width,
                    max_sequence_length=self.max_sequence_length,
                ).images[0]
            
            # Parse and add meme text if present
            top_text, bottom_text = self._parse_meme_text(prompt)
            if top_text or bottom_text:
                image = self._add_text_overlay(image, top_text, bottom_text)
            
            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, quality=95)
            
            # Update metrics
            metrics.generation_time = time.time() - start_time
            if self.device == "cuda":
                metrics.gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**3
            
            return output_path, metrics.generation_time, self.model_name.split('/')[-1]
            
        except Exception as e:
            metrics.success = False
            metrics.error = str(e)
            raise
        finally:
            self.memory_manager.optimize_memory_usage()
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.memory_manager.optimize_memory_usage()
