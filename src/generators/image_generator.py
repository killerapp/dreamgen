"""
Image generator using Flux 1.1 transformers model.
"""
from pathlib import Path
from typing import Optional
import os
import torch
from diffusers import DiffusionPipeline
from PIL import Image

class ImageGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('FLUX_MODEL', 'black-forest-labs/FLUX.1-dev')
        self.pipe = None
        
    def initialize(self):
        """Initialize the Flux diffusion pipeline."""
        if self.pipe is None:
            try:
                self.pipe = DiffusionPipeline.from_pretrained(self.model_name)
            except Exception as e:
                raise Exception(f"Error initializing Flux model: {str(e)}")
    
    async def generate_image(self, prompt: str, output_path: Path) -> Path:
        """Generate an image from the given prompt and save it to the specified path."""
        self.initialize()
        
        try:
            # Generate the image
            image = self.pipe(prompt).images[0]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the image
            image.save(output_path)
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None
