"""Meme plugin for generating meme-style prompts."""
import random
from typing import NamedTuple, Optional

class MemeStyle(NamedTuple):
    """Container for meme style information."""
    template: str
    top_text: str
    bottom_text: Optional[str] = None

class MemePlugin:
    """Plugin for generating meme context following singleton pattern."""
    _instance = None
    _last_template = None
    _last_phrase = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # Meme templates that work with Flux model's text placement
    TEMPLATES = [
        "classic meme with white text '{text}' {position}, on a simple background",
        "meme format with impact-style text '{text}' {position}",
        "minimalist meme with '{text}' {position}",
        "internet meme showing '{text}' {position}",
        "social media meme displaying '{text}' {position}",
    ]
    
    # Common meme phrases that can be adapted
    PHRASES = [
        ("When you forget to commit your changes", "And lose all your work"),
        ("Me debugging at 3am", "Finding it was just a typo"),
        ("Code in production", "Code in development"),
        ("What I think AI does", "What AI actually does"),
        ("Nobody:", "AI generating memes"),
        ("My code before code review", "My code after code review"),
        ("Writing clean code", "Maintaining legacy code"),
        ("Expectation:", "Reality:"),
        ("First day learning to code", "One year into coding"),
        ("How I explain my code", "How it actually works"),
        ("When the bug appears in prod", "But works fine locally"),
        ("Senior dev reviewing my PR", "Me explaining my spaghetti code"),
    ]
    
    def get_meme_style(self) -> MemeStyle:
        """Generate a meme style with template and text."""
        # Avoid repeating the last template and phrase
        available_templates = [t for t in self.TEMPLATES if t != self._last_template]
        available_phrases = [p for p in self.PHRASES if p != self._last_phrase]
        
        template = random.choice(available_templates)
        phrase_pair = random.choice(available_phrases)
        
        self._last_template = template
        self._last_phrase = phrase_pair
        
        top_phrase, bottom_phrase = phrase_pair
        
        return MemeStyle(
            template=template,
            top_text=top_phrase,
            bottom_text=bottom_phrase
        )

def get_meme_context() -> str:
    """
    Creates a meme-style prompt incorporating template and text positioning.
    Uses singleton instance to avoid template/phrase repetition.
    
    Returns:
        str: A descriptive string for generating a meme-style image
    """
    plugin = MemePlugin()
    meme = plugin.get_meme_style()
    
    # Build the complete meme prompt
    prompt_parts = []
    
    # Add top text
    prompt_parts.append(
        meme.template.format(text=meme.top_text, position="at the top")
    )
    
    # Add bottom text if present
    if meme.bottom_text:
        prompt_parts.append(
            meme.template.format(text=meme.bottom_text, position="at the bottom")
        )
    
    return ", ".join(prompt_parts)
