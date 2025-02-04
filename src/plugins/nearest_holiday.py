import json
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

class Holiday(TypedDict):
    month: int
    day: int
    name: str

def get_nearest_holiday() -> Optional[str]:
    """
    Determines the nearest upcoming holiday that hasn't passed yet.
    If all holidays for the current year have passed, returns the earliest holiday of next year.
    
    Returns:
        Optional[str]: A string describing the nearest holiday (e.g., "approaching Christmas") 
                      or None if no holidays are defined
    """
    try:
        holidays_path = Path(__file__).parent.parent.parent / "data" / "holidays.json"
        with open(holidays_path, 'r') as f:
            holidays: list[Holiday] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not holidays:
        return None

    current_date = datetime.now()
    current_month = current_date.month
    current_day = current_date.day
    
    # Convert holidays to (month, day, name) tuples for easier comparison
    holiday_tuples = [(h["month"], h["day"], h["name"]) for h in holidays]
    
    # Sort holidays by month and day
    holiday_tuples.sort()
    
    # Find the next upcoming holiday
    upcoming_holiday = None
    
    # First, check remaining holidays this year
    for month, day, name in holiday_tuples:
        if (month > current_month) or (month == current_month and day >= current_day):
            upcoming_holiday = (month, day, name)
            break
    
    # If no upcoming holidays this year, take the first holiday of next year
    if not upcoming_holiday and holiday_tuples:
        upcoming_holiday = holiday_tuples[0]
    
    if upcoming_holiday:
        return f"approaching {upcoming_holiday[2]}"
    
    return None
