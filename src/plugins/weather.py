"""Weather plugin for fetching current weather conditions."""
import json
from typing import Optional
import urllib.request
from urllib.error import URLError

class WeatherPlugin:
    _instance = None
    _cached_data = None
    _default_location = {"lat": 41.8781, "lon": -87.6298}  # Chicago coordinates

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _fetch_weather(self) -> Optional[dict]:
        """Fetch current weather data from Open-Meteo API."""
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={self._default_location['lat']}"
                f"&longitude={self._default_location['lon']}"
                f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
            )
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read())
        except URLError:
            return None

    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to descriptive text."""
        weather_codes = {
            0: "clear sky",
            1: "mainly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "depositing rime fog",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            77: "snow grains",
            80: "slight rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "slight snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with slight hail",
            99: "thunderstorm with heavy hail",
        }
        return weather_codes.get(code, "unknown weather")

    def get_context(self) -> Optional[str]:
        """
        Get current weather conditions as context string.

        Returns:
            Optional[str]: Weather context string or None if weather data cannot be fetched
        """
        data = self._fetch_weather()
        if not data or "current" not in data:
            return None

        current = data["current"]
        weather_desc = self._weather_code_to_description(current["weather_code"])
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]

        return (
            f"with {weather_desc}, {temp}°C temperature, "
            f"{humidity}% humidity, and {wind_speed} km/h winds"
        )

def get_weather_context() -> Optional[str]:
    """Get weather context from the WeatherPlugin singleton."""
    context = WeatherPlugin().get_context()
    print(f"Weather context: {context}")  # Debug logging
    return context
