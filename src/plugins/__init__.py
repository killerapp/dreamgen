from typing import NamedTuple, Optional

from .time_of_day import get_time_of_day
from .nearest_holiday import get_nearest_holiday
from .holiday_fact import get_holiday_fact
from .art_style import get_art_style
from .weather import get_weather_context
from .meme import get_meme_context

class TemporalContext(NamedTuple):
    """Container for all temporal information and art style."""
    time_of_day: str
    holiday: Optional[str]
    holiday_fact: Optional[str]
    art_style: str
    weather: Optional[str]
    meme: Optional[str]

def get_temporal_context() -> TemporalContext:
    """
    Retrieves the complete temporal context including time of day,
    holiday information, art style, weather conditions, and meme style.
    
    Returns:
        TemporalContext: Named tuple containing all temporal information and styles
    """
    return TemporalContext(
        time_of_day=get_time_of_day(),
        holiday=get_nearest_holiday(),
        holiday_fact=get_holiday_fact(),
        art_style=get_art_style(),
        weather=get_weather_context(),
        meme=get_meme_context()
    )

def get_temporal_descriptor() -> str:
    """
    Creates a human-readable string combining all temporal information, art style, and meme context.
    
    Returns:
        str: A descriptive string that can include meme formatting if meme context is present
    """
    context = get_temporal_context()
    
    # If meme context is present, use it as the primary descriptor
    if context.meme:
        return context.meme
        
    # Otherwise use standard temporal context
    temporal_parts = [
        context.time_of_day,
        context.holiday
    ]
    # Filter out None values and join temporal parts with commas
    temporal_desc = ", ".join(part for part in temporal_parts if part is not None)
    
    # Add holiday fact if available
    if context.holiday_fact:
        temporal_desc = f"{temporal_desc} ({context.holiday_fact})"
    
    # Add weather if available
    if context.weather:
        temporal_desc = f"{temporal_desc} {context.weather}"
    
    # Add art style if available
    if context.art_style:
        return f"{temporal_desc}, {context.art_style}"
    return temporal_desc
