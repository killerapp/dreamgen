import json
import random
import logging
from pathlib import Path
from typing import NamedTuple, Optional, List

logger = logging.getLogger(__name__)

class ArtStyle(NamedTuple):
    """Container for art style information."""
    name: str
    description: str

class ArtStylePlugin:
    """Plugin for managing and selecting art styles."""
    _instance = None
    _styles: list[ArtStyle] = []
    _last_style: Optional[ArtStyle] = None

    def __new__(cls):
        """Singleton pattern to ensure styles are loaded only once."""
        if cls._instance is None:
            cls._instance = super(ArtStylePlugin, cls).__new__(cls)
            cls._instance._load_styles()
        return cls._instance

    def _load_styles(self) -> None:
        """Load art styles from JSON file."""
        try:
            styles_path = Path(__file__).parent.parent.parent / "data" / "art_styles.json"
            
            # Check if the file exists
            if not styles_path.exists():
                logger.error(f"Art styles file not found: {styles_path}")
                self._load_default_styles()
                return
                
            # Attempt to load and parse the file
            with open(styles_path, 'r') as f:
                data = json.load(f)
                if "styles" not in data:
                    logger.error("Invalid art styles format: missing 'styles' key")
                    self._load_default_styles()
                    return
                    
                self._styles = [
                    ArtStyle(name=style["name"], description=style["description"])
                    for style in data["styles"]
                    if "name" in style and "description" in style
                ]
                
            if not self._styles:
                logger.warning("No valid art styles loaded from file")
                self._load_default_styles()
            else:
                logger.info(f"Successfully loaded {len(self._styles)} art styles")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing art styles file: {str(e)}")
            self._load_default_styles()
        except Exception as e:
            logger.error(f"Error loading art styles: {str(e)}")
            self._load_default_styles()

    def _load_default_styles(self) -> None:
        """Load fallback default styles if the JSON file can't be loaded."""
        logger.info("Loading default art styles")
        self._styles = [
            ArtStyle(name="Impressionist", description="light, airy brushstrokes with emphasis on light and color"),
            ArtStyle(name="Cubist", description="geometric shapes, multiple viewpoints, and abstract forms"),
            ArtStyle(name="Surrealist", description="dreamlike, unexpected juxtapositions and symbolic elements"),
            ArtStyle(name="Minimalist", description="sparse, simple elements with focus on form and negative space"),
            ArtStyle(name="Pixel Art", description="digital art created using precise, limited resolution blocks of color"),
        ]

    def get_available_styles(self) -> List[ArtStyle]:
        """
        Get the list of all available art styles.
        
        Returns:
            List[ArtStyle]: List of all loaded art styles
        """
        return self._styles.copy()
        
    def get_style_by_name(self, name: str) -> Optional[ArtStyle]:
        """
        Get a specific art style by name.
        
        Args:
            name: Name of the art style to find (case-insensitive)
            
        Returns:
            Optional[ArtStyle]: The art style if found, None otherwise
        """
        name_lower = name.lower()
        for style in self._styles:
            if style.name.lower() == name_lower:
                return style
        return None
        
    def get_random_style(self, avoid_last: bool = True) -> Optional[ArtStyle]:
        """
        Get a random art style, optionally avoiding the last used style.
        
        Args:
            avoid_last: If True, won't return the same style twice in a row
            
        Returns:
            Optional[ArtStyle]: A randomly selected art style, or None if no styles are available
        """
        if not self._styles:
            return None

        available_styles = self._styles
        if avoid_last and self._last_style and len(self._styles) > 1:
            available_styles = [s for s in self._styles if s != self._last_style]

        style = random.choice(available_styles)
        self._last_style = style
        return style

def get_art_style() -> str:
    """
    Get a random art style as a formatted string.
    
    Returns:
        str: A string combining the style name and description,
             or an empty string if no styles are available
    """
    plugin = ArtStylePlugin()
    style = plugin.get_random_style()
    
    if style:
        return f"in the style of {style.name} ({style.description})"
    return ""
