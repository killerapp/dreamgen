"""
Command-line interface for the image generation system.
"""
import asyncio
from pathlib import Path
import typer
from typing import Optional
from ..generators.prompt_generator import PromptGenerator
from ..generators.image_generator import ImageGenerator
from .storage import StorageManager

app = typer.Typer()

@app.command()
def generate(
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enable interactive mode with prompt feedback"),
    model: str = typer.Option("llama2", "--model", "-m", help="Ollama model to use for prompt generation"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Force CPU-only mode (not recommended)"),
):
    """Generate a single image using AI-generated prompts."""
    async def _generate():
        try:
            # Initialize components
            prompt_gen = PromptGenerator(model_name=model)
            image_gen = ImageGenerator(cpu_only=cpu_only)
            storage = StorageManager()

            # Generate prompt (with or without feedback)
            if interactive:
                prompt = await prompt_gen.get_prompt_with_feedback()
            else:
                prompt = await prompt_gen.generate_prompt()
                print("\nGenerated prompt:")
                print("-" * 80)
                print(prompt)
                print("-" * 80)

            # Get output path
            output_path = storage.get_output_path(prompt)
            
            print(f"\nGenerating image...")
            
            # Generate image
            await image_gen.generate_image(prompt, output_path)
            
            print(f"\nImage generated successfully!")
            print(f"Saved to: {output_path}")
            print(f"Prompt saved to: {output_path.with_suffix('.txt')}")
            
            # Cleanup
            image_gen.cleanup()
            
        except Exception as e:
            typer.echo(f"Error: {str(e)}", err=True)
            raise typer.Exit(1)

    # Run the async function
    asyncio.run(_generate())

@app.command()
def loop(
    interval: Optional[int] = typer.Option(None, "--interval", "-n", help="Interval in seconds between generations (default: immediate)"),
    model: str = typer.Option("llama2", "--model", "-m", help="Ollama model to use for prompt generation"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Force CPU-only mode (not recommended)"),
):
    """Continuously generate images in a loop."""
    async def _loop():
        try:
            # Initialize components
            prompt_gen = PromptGenerator(model_name=model)
            image_gen = ImageGenerator(cpu_only=cpu_only)
            storage = StorageManager()

            print(f"Starting continuous generation loop...")
            print(f"Press Ctrl+C to stop")
            
            while True:
                try:
                    # Generate prompt
                    prompt = await prompt_gen.generate_prompt()
                    print("\nGenerated prompt:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)

                    # Get output path
                    output_path = storage.get_output_path(prompt)
                    
                    print(f"\nGenerating image...")
                    
                    # Generate image
                    await image_gen.generate_image(prompt, output_path)
                    
                    print(f"\nImage generated successfully!")
                    print(f"Saved to: {output_path}")
                    print(f"Prompt saved to: {output_path.with_suffix('.txt')}")
                    
                    # Wait if interval is specified
                    if interval:
                        print(f"\nWaiting {interval} seconds...")
                        await asyncio.sleep(interval)
                        
                except KeyboardInterrupt:
                    print("\nStopping generation loop...")
                    break
                except Exception as e:
                    print(f"Error in generation cycle: {str(e)}")
                    print("Continuing with next cycle...")
                    if interval:
                        await asyncio.sleep(interval)
            
            # Final cleanup
            image_gen.cleanup()
            
        except Exception as e:
            typer.echo(f"Error: {str(e)}", err=True)
            raise typer.Exit(1)

    # Run the async function
    asyncio.run(_loop())

if __name__ == "__main__":
    app()
