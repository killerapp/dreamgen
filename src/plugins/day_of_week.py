from datetime import datetime
from typing import Literal

DayOfWeek = Literal[
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

def get_day_of_week() -> DayOfWeek:
    """
    Gets the current day of the week.
    
    Returns:
        str: The current day of the week (e.g., "Monday", "Tuesday", etc.)
    """
    return datetime.now().strftime("%A")  # type: ignore
