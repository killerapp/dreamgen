"""
Storage utilities for managing image output directories and files.
"""
from datetime import datetime
from pathlib import Path
import hashlib
import os
import logging
import shutil

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        # Ensure base directory exists and is writable
        self._ensure_directory_is_writable(self.base_dir)
        
    def _ensure_directory_is_writable(self, directory: Path) -> bool:
        """
        Ensure a directory exists and is writable.
        
        Args:
            directory: Directory path to check
            
        Returns:
            bool: True if directory is writable, False otherwise
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            
            # Test write access by creating and removing a test file
            test_file = directory / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except (PermissionError, OSError) as e:
            logger.error(f"Directory not writable: {directory} - {str(e)}")
            return False
        
    def get_weekly_directory(self) -> Path:
        """Get the directory path for the current week of the year."""
        now = datetime.now()
        year = now.year
        week = now.isocalendar()[1]
        
        # Create path: output/[year]/week_[XX]
        weekly_dir = self.base_dir / str(year) / f"week_{week:02d}"
        weekly_dir.mkdir(parents=True, exist_ok=True)
        
        return weekly_dir
    
    def get_output_path(self, prompt: str) -> tuple[Path, Path]:
        """
        Generate unique output paths for an image and its prompt text file.
        
        Note: This method DOES NOT write the prompt file yet, only returns paths.
        
        Args:
            prompt: The prompt text
            
        Returns:
            tuple[Path, Path]: (image_path, prompt_file_path)
        """
        # Create a short hash of the prompt for uniqueness
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        # Generate filename with timestamp and prompt hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"image_{timestamp}_{prompt_hash}.png"
        prompt_filename = f"image_{timestamp}_{prompt_hash}.txt"
        
        # Get weekly directory
        weekly_dir = self.get_weekly_directory()
        
        # Return both image and prompt paths
        return weekly_dir / image_filename, weekly_dir / prompt_filename
    
    def save_prompt_file(self, prompt_path: Path, prompt: str) -> bool:
        """
        Save prompt text to file.
        
        Args:
            prompt_path: Path to save the prompt text
            prompt: The prompt text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            prompt_path.write_text(prompt)
            return True
        except Exception as e:
            logger.error(f"Failed to save prompt file: {str(e)}")
            return False
    
    def get_file_pairs(self) -> list[tuple[Path, Path]]:
        """
        Get all matching image/prompt file pairs.
        
        Returns:
            list[tuple[Path, Path]]: List of (image_path, prompt_path) pairs
        """
        pairs = []
        for image_file in self.base_dir.rglob("*.png"):
            prompt_file = image_file.with_suffix(".txt")
            if prompt_file.exists():
                pairs.append((image_file, prompt_file))
        return pairs
    
    def cleanup_orphaned_prompt_files(self) -> int:
        """
        Remove prompt text files that don't have a corresponding image file.
        
        Returns:
            int: Number of files removed
        """
        count = 0
        for prompt_file in self.base_dir.rglob("*.txt"):
            image_file = prompt_file.with_suffix(".png")
            if not image_file.exists():
                try:
                    prompt_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to remove orphaned prompt file {prompt_file}: {str(e)}")
        return count
    
    def cleanup_old_files(self, max_age_days: int = None) -> int:
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
