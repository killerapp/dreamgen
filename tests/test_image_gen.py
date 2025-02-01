"""
Test script for generating an image with a long prompt.
"""
import asyncio
from pathlib import Path
from src.generators.image_generator import ImageGenerator

async def main():
    # Create a very long prompt to test the new embedding handling
    base_prompt = "A stunning photograph of a magical forest with ancient trees, "
    details = [
        "mystical glowing orbs floating between the branches, ",
        "rays of golden sunlight filtering through the dense canopy, ",
        "a crystal-clear stream winding through moss-covered rocks, ",
        "delicate wildflowers in vibrant purples and blues dotting the forest floor, ",
        "ethereal mist swirling around the tree trunks, ",
        "tiny fairies with gossamer wings darting between the flowers, ",
        "ancient runes carved into the bark of the oldest trees, ",
        "bioluminescent mushrooms growing in clusters at the base of the trees, ",
        "a family of deer grazing peacefully in a nearby clearing, ",
        "butterflies with iridescent wings fluttering through the scene"
    ] * 3  # Repeat details to make the prompt extra long
    
    long_prompt = base_prompt + "".join(details)
    print(f"Prompt length (words): {len(long_prompt.split())}")
    
    # Initialize generator
    generator = ImageGenerator()
    
    # Generate image
    output_path = Path("test_output.png")
    try:
        result = await generator.generate_image(long_prompt, output_path)
        print(f"Image generated successfully at: {result}")
    finally:
        generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
