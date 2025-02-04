from datetime import datetime
from typing import Literal

TimeOfDay = Literal["morning", "afternoon", "evening", "night"]

def get_time_of_day() -> TimeOfDay:
    """
    Determines the time of day based on the current hour.
    
    Returns:
        str: One of "morning", "afternoon", "evening", or "night"
    """
    hour = datetime.now().hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"
