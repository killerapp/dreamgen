"""Simple web interface for image generation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import gradio as gr

from ..generators.image_generator import ImageGenerator
from ..generators.mock_image_generator import MockImageGenerator
from .config import Config
from .storage import StorageManager


def launch_web_ui(config: Config, mock: bool = False, _launch: bool = True) -> gr.Interface:
    """Create and optionally launch the Gradio web UI.

    Args:
        config: Application configuration.
        mock: Use the mock image generator instead of the real one.
        _launch: If True, start the server. Set to False for testing.

    Returns:
        The configured :class:`gr.Interface` instance.
    """

    storage = StorageManager(str(config.system.output_dir))
    generator_cls = MockImageGenerator if mock else ImageGenerator

    def generate(prompt: str) -> Path:
        """Generate an image from ``prompt`` and return the filepath."""
        gen = generator_cls(config)
        output_path = storage.get_output_path(prompt)
        result_path, _, _ = asyncio.run(gen.generate_image(prompt, output_path))
        gen.cleanup()
        return result_path

    demo = gr.Interface(
        fn=generate,
        inputs=gr.Textbox(label="Prompt"),
        outputs=gr.Image(type="filepath"),
        title="Continuous Image Generator",
        description="Enter a prompt to generate an image.",
    )

    if _launch:
        demo.launch()
    return demo
