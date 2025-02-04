import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypedDict

class Holiday(TypedDict):
    month: int
    day: int
    name: str
    type: str

class HolidayFact:
    """Plugin for getting information about today's holidays and observances."""
    _instance = None
    _holidays: List[Holiday] = []

    def __new__(cls):
        """Singleton pattern to ensure holidays are loaded only once."""
        if cls._instance is None:
            cls._instance = super(HolidayFact, cls).__new__(cls)
            cls._instance._load_holidays()
        return cls._instance

    def _load_holidays(self) -> None:
        """Load holidays from JSON file."""
        try:
            holidays_path = Path(__file__).parent.parent.parent / "data" / "holidays.json"
            with open(holidays_path, 'r') as f:
                self._holidays = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading holidays: {str(e)}")
            self._holidays = []

    def get_todays_holidays(self) -> List[Holiday]:
        """Get all holidays for today's date."""
        if not self._holidays:
            return []

        current_date = datetime.now()
        return [
            holiday for holiday in self._holidays
            if holiday["month"] == current_date.month and holiday["day"] == current_date.day
        ]

def get_holiday_fact() -> Optional[str]:
    """
    Get an interesting fact about today's holidays or observances.
    
    Returns:
        Optional[str]: A string mentioning today's special observances,
                      or None if there are no holidays today
    """
    plugin = HolidayFact()
    todays_holidays = plugin.get_todays_holidays()
    
    if not todays_holidays:
        return None
        
    if len(todays_holidays) == 1:
        holiday = todays_holidays[0]
        if holiday["type"] == "fun":
            return f"Did you know today is {holiday['name']}? A fun day to celebrate!"
        elif holiday["type"] == "observance":
            return f"Today is {holiday['name']}, a day for awareness and reflection."
        elif holiday["type"] == "cultural":
            return f"Today we celebrate {holiday['name']}, a cultural celebration!"
        else:  # public_holiday
            return f"Today is {holiday['name']}, a public holiday!"
    else:
        holiday_names = [h["name"] for h in todays_holidays]
        if len(holiday_names) == 2:
            return f"Today is special: it's both {holiday_names[0]} and {holiday_names[1]}!"
        else:
            names = ", ".join(holiday_names[:-1]) + f", and {holiday_names[-1]}"
            return f"Today is packed with celebrations: {names}!"
