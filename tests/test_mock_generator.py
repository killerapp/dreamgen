import asyncio
from pathlib import Path

from src.generators.mock_image_generator import MockImageGenerator
from src.utils.config import Config


def test_mock_image_generator(tmp_path: Path):
    config = Config()
    generator = MockImageGenerator(config)

    prompt = "a test prompt"
    output_path = tmp_path / "test.png"

    result_path, gen_time, model_name = asyncio.run(
        generator.generate_image(prompt, output_path)
    )

    assert result_path.exists()
    assert model_name == "mock"
    assert gen_time >= 0

    text_file = result_path.with_suffix(".txt")
    assert text_file.exists()
    assert text_file.read_text() == prompt

    generator.cleanup()
