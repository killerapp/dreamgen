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
    FLUX_MODELS = {
        "dev": "black-forest-labs/FLUX.1-dev",      # High quality, ~50 steps
        "schnell": "black-forest-labs/FLUX.1-schnell"  # Fast generation, ~4 steps
    }

    def __init__(self, 
                 model_variant: str = None, 
                 cpu_only: bool = False,
                 height: int = None,
                 width: int = None,
                 num_inference_steps: int = None,
                 guidance_scale: float = None,
                 true_cfg_scale: float = None,
                 max_sequence_length: int = None):
        """Initialize the image generator with configurable parameters.
        
        Args:
            model_variant: Which Flux model variant to use ('dev' or 'schnell')
            cpu_only: Whether to force CPU-only mode
            height: Height of generated images (default from env or 768)
            width: Width of generated images (default from env or 1360)
            num_inference_steps: Number of denoising steps (default from env or 50)
            guidance_scale: Guidance scale for generation (default from env or 7.5)
            true_cfg_scale: True classifier-free guidance scale (default from env or 1.0)
            max_sequence_length: Max sequence length for text processing (default from env or 512)
        """
        env_model = os.getenv('FLUX_MODEL', 'dev')
        model_variant = model_variant or env_model
        
        if model_variant not in self.FLUX_MODELS:
            raise ValueError(f"Invalid model variant '{model_variant}'. Must be one of: {', '.join(self.FLUX_MODELS.keys())}")
            
        self.model_name = self.FLUX_MODELS[model_variant]
        self.height = height or int(os.getenv('IMAGE_HEIGHT', 768))
        self.width = width or int(os.getenv('IMAGE_WIDTH', 1360))
        self.num_inference_steps = num_inference_steps or int(os.getenv('NUM_INFERENCE_STEPS', 50))
        self.guidance_scale = guidance_scale or float(os.getenv('GUIDANCE_SCALE', 7.5))
        self.true_cfg_scale = true_cfg_scale or float(os.getenv('TRUE_CFG_SCALE', 1.0))
        self.max_sequence_length = max_sequence_length or int(os.getenv('MAX_SEQUENCE_LENGTH', 512))
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
                
                # Set memory management environment variables
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
                
                # Load model with memory optimizations
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="balanced",  # Enable balanced device placement
                )
                
                if self.device == "cuda":
                    # Enable memory optimizations
                    self.pipe.enable_attention_slicing()
                    print("Enabled attention slicing")
                    
                    self.pipe.enable_vae_tiling()
                    print("Enabled VAE tiling")
                    
                    # Enable xformers memory efficient attention if available
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("Enabled xformers memory efficient attention")
                    except Exception:
                        print("Xformers optimization not available")
                    
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
            # Track generation time
            import time
            start_time = time.time()
            
            # Generate the image
            print(f"Generating image on {self.device}...")
            
            if self.device == "cuda":
                # Clear cache before generation
                torch.cuda.empty_cache()
                print(f"GPU Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self.device=="cuda"):
                # Let pipeline handle both encoders internally
                image = self.pipe(
                    prompt=prompt,
                    prompt_2=prompt,  # Pass same prompt to T5
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    true_cfg_scale=self.true_cfg_scale,
                    height=self.height,
                    width=self.width,
                    max_sequence_length=self.max_sequence_length,
                ).images[0]
            
            if self.device == "cuda":
                torch.cuda.synchronize()  # Ensure GPU operations are complete
                print(f"GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the image
            image.save(output_path)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            return output_path, generation_time, self.model_name.split('/')[-1]
                
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
