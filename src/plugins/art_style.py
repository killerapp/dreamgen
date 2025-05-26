import json
import logging
import random
from pathlib import Path
from typing import NamedTuple, Optional

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
            with open(styles_path, 'r') as f:
                data = json.load(f)
                self._styles = [
                    ArtStyle(name=style["name"], description=style["description"])
                    for style in data["styles"]
                ]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading art styles: {str(e)}")
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

        # Start with all styles as candidates
        candidate_styles = self._styles

        if avoid_last and self._last_style and len(self._styles) > 1:
            # If we need to avoid the last style and there's more than one style overall,
            # try to pick from styles that are not the last one.
            styles_excluding_last = [s for s in self._styles if s != self._last_style]
            if styles_excluding_last:
                # If this filtering results in a non-empty list, these are our candidates
                candidate_styles = styles_excluding_last
            # If styles_excluding_last is empty, it means the only style available is the one
            # we are trying to avoid. In this case, candidate_styles remains self._styles,
            # so we will pick the only available style.

        # At this point, candidate_styles is guaranteed not to be empty if self._styles was not empty.
        chosen_style = random.choice(candidate_styles)
        self._last_style = chosen_style
        return chosen_style

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
