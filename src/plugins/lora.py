"""
Plugin for loading and managing Lora models.
"""
from pathlib import Path
from typing import List, Optional, NamedTuple
import logging
import random

from ..utils.config import Config

logger = logging.getLogger(__name__)

class SelectedLora(NamedTuple):
    """Container for selected Lora information."""
    name: str
    path: Path
    keyword: str  # The keyword that must be used in the prompt

def get_available_loras(lora_dir: Path) -> List[str]:
    """Get list of available Lora models in the specified directory."""
    if not lora_dir.exists():
        logger.warning(f"Lora directory not found: {lora_dir}")
        return []
    
    # Look for subdirectories that contain .safetensors files
    lora_names = []
    for subdir in lora_dir.iterdir():
        if subdir.is_dir():
            if list(subdir.glob("*.safetensors")):
                lora_names.append(subdir.name)
                logger.info(f"Found Lora directory: {subdir.name}")
    
    logger.info(f"Found {len(lora_names)} Lora directories in {lora_dir}")
    return lora_names

def get_lora_path(lora_name: str, config: Config) -> Optional[Path]:
    """Get the full path to a Lora model file."""
    lora_dir = config.model.lora.lora_dir / lora_name
    
    if not lora_dir.exists():
        logger.warning(f"Lora directory not found: {lora_dir}")
        return None
    
    # Get all .safetensors files in the directory
    lora_files = list(lora_dir.glob("*.safetensors"))
    if not lora_files:
        logger.warning(f"No .safetensors files found in {lora_dir}")
        return None
    
    # Sort by version number and get the latest
    latest_lora = sorted(lora_files, key=lambda x: int(x.stem.split('-')[-1]) if '-' in x.stem else 0)[-1]
    logger.info(f"Selected latest Lora version: {latest_lora}")
    return latest_lora

def get_lora_keyword(lora_name: str) -> str:
    """Get the keyword that must be used in the prompt for this Lora."""
    # For v4skin, the keyword is "v4skin"
    # You can add more mappings here if needed
    return lora_name

def select_random_lora(config: Config) -> Optional[SelectedLora]:
    """
    Randomly select a Lora from enabled Loras based on configuration.
    Returns None if no Lora should be applied based on probability.
    """
    # First check if we should apply a Lora at all
    if random.random() > config.model.lora.application_probability:
        logger.info("Skipping Lora application based on probability")
        return None
    
    # Get available and enabled Loras
    available_loras = get_available_loras(config.model.lora.lora_dir)
    logger.info(f"Enabled Loras: {config.model.lora.enabled_loras}")
    logger.info(f"Available Loras: {available_loras}")
    
    enabled_loras = [lora for lora in config.model.lora.enabled_loras if lora in available_loras]
    
    if not enabled_loras:
        logger.warning("No enabled Loras found in the available Loras list")
        return None
    
    # Randomly select one Lora
    selected_name = random.choice(enabled_loras)
    selected_path = get_lora_path(selected_name, config)
    
    if selected_path:
        keyword = get_lora_keyword(selected_name)
        logger.info(f"Selected Lora: {selected_name} at path: {selected_path}")
        logger.info(f"Required keyword for prompt: {keyword}")
        return SelectedLora(name=selected_name, path=selected_path, keyword=keyword)
    
    return None

def apply_lora(config: Config) -> Optional[str]:
    """
    Plugin entry point. Randomly selects a Lora and returns the required keyword
    for the prompt if a Lora is selected.
    """
    selected = select_random_lora(config)
    if selected:
        # Return the keyword that must be used in the prompt
        return selected.keyword
    return None
