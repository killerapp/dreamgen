"""
Storage utilities for managing image output directories and files.
"""
from datetime import datetime
from pathlib import Path
import hashlib
from PIL import Image

class StorageManager:
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        
    def get_weekly_directory(self) -> Path:
        """Get the directory path for the current week of the year."""
        now = datetime.now()
        year = now.year
        week = now.isocalendar()[1]
        
        # Create path: output/[year]/week_[XX]
        weekly_dir = self.base_dir / str(year) / f"week_{week:02d}"
        weekly_dir.mkdir(parents=True, exist_ok=True)
        
        return weekly_dir
    
    def get_output_path(self, prompt: str) -> Path:
        """Generate a unique output path for an image based on timestamp and prompt."""
        # Create a short hash of the prompt for uniqueness
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        # Generate filename with timestamp and prompt hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}_{prompt_hash}.png"
        
        # Get weekly directory and return full path
        weekly_dir = self.get_weekly_directory()
        
        # Save prompt text alongside image
        prompt_file = weekly_dir / f"image_{timestamp}_{prompt_hash}.txt"
        prompt_file.write_text(prompt)
        
        return weekly_dir / filename
    
    def cleanup_old_files(self, max_age_days: int = None):
        """Optional: Clean up old files beyond a certain age."""
        if max_age_days is None:
            return
            
        now = datetime.now()
        
        for image_file in self.base_dir.rglob("*.png"):
            # Get file age in days
            age = (now - datetime.fromtimestamp(image_file.stat().st_mtime)).days
            
            if age > max_age_days:
                # Remove both image and its associated prompt file
                prompt_file = image_file.with_suffix(".txt")
                if prompt_file.exists():
                    prompt_file.unlink()
                image_file.unlink()

def save_image_and_prompt(image: Image.Image, prompt: str, base_dir: str = "output") -> Path:
    """Save an image and its prompt to the output directory."""
    storage = StorageManager(base_dir)
    output_path = storage.get_output_path(prompt)
    
    # Save the image
    image.save(output_path, "PNG")
    
    return output_path
