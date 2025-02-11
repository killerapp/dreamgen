from typing import NamedTuple, Optional

from .time_of_day import get_time_of_day
from .nearest_holiday import get_nearest_holiday
from .holiday_fact import get_holiday_fact
from .art_style import get_art_style

class TemporalContext(NamedTuple):
    """Container for all temporal information and art style."""
    time_of_day: str
    holiday: Optional[str]
    holiday_fact: Optional[str]
    art_style: str

def get_temporal_context() -> TemporalContext:
    """
    Retrieves the complete temporal context including time of day,
    holiday information, and art style.
    
    Returns:
        TemporalContext: Named tuple containing all temporal information and art style
    """
    return TemporalContext(
        time_of_day=get_time_of_day(),
        holiday=get_nearest_holiday(),
        holiday_fact=get_holiday_fact(),
        art_style=get_art_style()
    )

def get_temporal_descriptor() -> str:
    """
    Creates a human-readable string combining all temporal information and art style.
    
    Returns:
        str: A descriptive string like "night, approaching Christmas (and today is World Photography Day!), 
             in the style of Impressionism"
    """
    context = get_temporal_context()
    temporal_parts = [
        context.time_of_day,
        context.holiday
    ]
    # Filter out None values and join temporal parts with commas
    temporal_desc = ", ".join(part for part in temporal_parts if part is not None)
    
    # Add holiday fact if available
    if context.holiday_fact:
        temporal_desc = f"{temporal_desc} ({context.holiday_fact})"
    
    # Add art style if available
    if context.art_style:
        return f"{temporal_desc}, {context.art_style}"
    return temporal_desc
