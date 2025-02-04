import json
import random
from pathlib import Path
from typing import NamedTuple, Optional

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
            with open(styles_path, 'r') as f:
                data = json.load(f)
                self._styles = [
                    ArtStyle(name=style["name"], description=style["description"])
                    for style in data["styles"]
                ]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading art styles: {str(e)}")
            self._styles = []

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
