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
    def __init__(self, model_name: str = None, cpu_only: bool = False):
        self.model_name = model_name or os.getenv('FLUX_MODEL', 'black-forest-labs/FLUX.1-dev')
        self.pipe = None
        
        if not cpu_only and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU (CUDA) is not available. "
                "This model requires a GPU for efficient processing. "
                "If you want to run on CPU anyway, use --cpu-only flag"
            )
            
        self.device = "cpu" if cpu_only else "cuda"
        
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            torch.cuda.set_device(0)  # Use first GPU
            torch.cuda.empty_cache()  # Initial cache clear
        else:
            print("WARNING: Running on CPU. This will be significantly slower.")
        
    def initialize(self):
        """Initialize the Flux diffusion pipeline."""
        if self.pipe is None:
            try:
                # Clear CUDA cache before loading model
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                print(f"Loading model on {self.device}...")
                
                # Load model with memory optimizations
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                )
                
                if self.device == "cuda":
                    # Enable memory optimizations before moving to GPU
                    self.pipe.enable_attention_slicing(slice_size=1)
                    print("Enabled attention slicing")
                    
                    self.pipe.enable_vae_slicing()
                    print("Enabled VAE slicing")
                    
                    # Move to device
                    print("Moving pipeline to GPU...")
                    self.pipe = self.pipe.to(self.device)
                    print("Pipeline successfully loaded on GPU with memory optimizations")
                    
                    # Monitor memory usage
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                    
            except Exception as e:
                raise Exception(f"Error initializing Flux model: {str(e)}")
    
    async def generate_image(self, prompt: str, output_path: Path) -> Path:
        """Generate an image from the given prompt and save it to the specified path."""
        self.initialize()
        
        try:
            # Generate the image
            print(f"Generating image on {self.device}...")
            
            if self.device == "cuda":
                # Clear cache before generation
                torch.cuda.empty_cache()
                print(f"GPU Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.device=="cuda"):
                image = self.pipe(
                    prompt,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                ).images[0]
            
            if self.device == "cuda":
                torch.cuda.synchronize()  # Ensure GPU operations are complete
                print(f"GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the image
            image.save(output_path)
            
            return output_path
                
        except Exception as e:
            if self.device == "cuda":
                print(f"GPU Memory at error: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            raise Exception(f"Error generating image: {str(e)}")
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.pipe = None
