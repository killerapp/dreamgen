"""Mock image generator for testing and development.

This class emulates the interface of :class:`ImageGenerator` but does not
require any ML models. It simply creates a placeholder image using PIL.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple
from PIL import Image

from ..utils.config import Config


class MockImageGenerator:
    """Lightweight stand-in for the real :class:`ImageGenerator`."""

    def __init__(self, config: Config):
        self.config = config
        self.model_name = "mock"

    def initialize(self, force_reinit: bool = False) -> None:  # noqa: D401 - stub
        """No-op initialize to mirror real generator."""
        # Nothing to initialize for the mock
        return None

    async def generate_image(self, prompt: str, output_path: Path, force_reinit: bool = False) -> Tuple[Path, float, str]:
        """Generate a simple placeholder image.

        Args:
            prompt: Prompt text (unused but saved alongside image).
            output_path: Where to save the generated image.
            force_reinit: Unused compatibility flag.
        Returns:
            Tuple of (output_path, generation_time_seconds, model_name)
        """
        start = time.time()
        width = self.config.image.width
        height = self.config.image.height
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a simple colored image
        img = Image.new("RGB", (width, height), color=(200, 200, 200))
        img.save(output_path)

        with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        gen_time = time.time() - start
        return output_path, gen_time, self.model_name

    def cleanup(self) -> None:  # noqa: D401 - stub
        """No-op cleanup to mirror real generator."""
        return None
