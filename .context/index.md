# Continuous Image Generation

## Overview

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux 1.1 for image generation.

## Architecture Details

### Components

- **Prompt Generator**: Uses Ollama for local LLM inference to generate creative prompts
- **Image Generator**: Uses Flux 1.1 transformers model for image generation
- **Storage Manager**: Handles weekly directory organization for outputs
- **CLI Interface**: Provides interactive and continuous generation modes
- **Plugin System**: Modular system for enhancing prompts with contextual information

### Design Decisions

- **Local LLM Usage**: Ollama was chosen for local inference to avoid API costs and latency
- **Weekly Organization**: Images are organized by year/week for better file management
- **Async Processing**: Used for efficient handling of both prompt and image generation
- **Interactive Mode**: Allows prompt refinement before image generation
- **Plugin Architecture**: Modular design for extensible prompt enhancement

### Plugin System Architecture

The plugin system follows these key design principles:

1. **Modularity**
   - Each plugin is a standalone Python module
   - Plugins are independent and can be enabled/disabled without affecting others
   - Common interface through standardized function signatures

2. **Data Management**
   - Plugins can have associated data files (e.g., holidays.json, art_styles.json)
   - Data files are stored in the data/ directory
   - JSON format used for easy maintenance and updates

3. **Singleton Pattern**
   - Plugins that need to maintain state or cache data use the Singleton pattern
   - Ensures efficient resource usage and consistent state

4. **Type Safety**
   - TypedDict and NamedTuple used for structured data
   - Optional types for nullable values
   - Literal types for enumerated values

#### Plugin Implementation Guide

1. **Basic Plugin Structure**
   ```python
   from typing import Optional

   def get_context() -> Optional[str]:
       """
       Get contextual information for prompt enhancement.
       Returns:
           Optional[str]: Context string or None if not applicable
       """
       return "context information"
   ```

2. **Stateful Plugin Template**
   ```python
   class MyPlugin:
       _instance = None
       _cached_data = None

       def __new__(cls):
           if cls._instance is None:
               cls._instance = super().__new__(cls)
               cls._instance._load_data()
           return cls._instance

       def _load_data(self) -> None:
           # Load and cache data
           pass

       def get_context(self) -> Optional[str]:
           # Use cached data to generate context
           pass
   ```

3. **Data-Driven Plugin Example**
   ```python
   from pathlib import Path
   import json
   from typing import TypedDict, List

   class DataItem(TypedDict):
       name: str
       description: str

   class DataDrivenPlugin:
       _instance = None
       _items: List[DataItem] = []

       def __new__(cls):
           if cls._instance is None:
               cls._instance = super().__new__(cls)
               cls._instance._load_data()
           return cls._instance

       def _load_data(self) -> None:
           data_path = Path(__file__).parent.parent.parent / "data" / "items.json"
           with open(data_path, 'r') as f:
               self._items = json.load(f)

       def get_context(self) -> str:
           # Process items to generate context
           pass
   ```

4. **Plugin Registration**
   - Add import to src/plugins/__init__.py
   - Update TemporalContext NamedTuple with new field
   - Include in get_temporal_context() function
   - Add to get_temporal_descriptor() string formatting

### Current Plugin Implementations

1. **Time of Day Plugin**
   - Location: src/plugins/time_of_day.py
   - Purpose: Provides time-based context
   - Implementation: Uses datetime to categorize current hour
   - Returns: "morning", "afternoon", "evening", or "night"

2. **Day of Week Plugin**
   - Location: src/plugins/day_of_week.py
   - Purpose: Provides current day context
   - Implementation: Uses datetime.strftime
   - Returns: Full day name (e.g., "Monday")

3. **Holiday Plugin**
   - Location: src/plugins/nearest_holiday.py
   - Purpose: Tracks upcoming holidays
   - Implementation: Reads from data/holidays.json
   - Returns: String describing approaching holiday

4. **Holiday Fact Plugin**
   - Location: src/plugins/holiday_fact.py
   - Purpose: Provides information about current day's holidays
   - Implementation: Reads from data/holidays.json
   - Returns: Formatted string about today's special observances

5. **Art Style Plugin**
   - Location: src/plugins/art_style.py
   - Purpose: Provides varied artistic styles
   - Implementation: Reads from data/art_styles.json
   - Features:
     - 90+ distinct art styles
     - Detailed style descriptions
     - Avoids consecutive repetition
     - Singleton pattern for efficient data loading

### Plugin Integration

The prompt generator integrates plugins through these steps:

1. **Context Collection**
   ```python
   temporal_context = get_temporal_descriptor()
   ```

2. **System Context Enhancement**
   ```python
   system_context = "\n".join([
       "You are a creative prompt generator for image generation.",
       f"\nCurrent temporal context: {temporal_context}",
       "Incorporate this temporal context naturally into the generated prompt"
   ])
   ```

3. **Prompt Generation**
   - LLM receives enhanced system context
   - Generates prompt incorporating all plugin contexts
   - Maintains coherence while respecting token limits

