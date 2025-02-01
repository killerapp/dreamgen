"""
Command-line interface for the image generation system.
"""
import asyncio
import os
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
    model: str = typer.Option(os.getenv('OLLAMA_MODEL', 'phi4:latest'), "--model", "-m", help="Ollama model to use for prompt generation"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Force CPU-only mode (not recommended)"),
) -> None:
    """Generate a single image using AI-generated prompts."""
    async def _generate() -> None:
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

    try:
        asyncio.run(_generate())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command()
def loop(
    batch_size: int = typer.Option(5, "--batch-size", "-b", help="Number of images to generate per run"),
    interval: Optional[int] = typer.Option(None, "--interval", "-n", help="Interval in seconds between generations (default: immediate)"),
    model: str = typer.Option(os.getenv('OLLAMA_MODEL', 'phi4:latest'), "--model", "-m", help="Ollama model to use for prompt generation"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="Force CPU-only mode (not recommended)"),
) -> None:
    """Generate a batch of images with unique prompts."""
    async def _loop() -> None:
        try:
            # Initialize components once
            print("Initializing models...")
            prompt_gen = PromptGenerator(model_name=model)
            image_gen = ImageGenerator(cpu_only=cpu_only)
            storage = StorageManager()
            
            print(f"\nStarting batch generation of {batch_size} images...")
            
            for i in range(batch_size):
                try:
                    print(f"\nGenerating image {i+1}/{batch_size}...")
                    
                    # Generate prompt in conversation context
                    prompt = await prompt_gen.generate_prompt()
                    print("\nGenerated prompt:")
                    print("-" * 80)
                    print(prompt)
                    print("-" * 80)

                    # Get output path
                    output_path = storage.get_output_path(prompt)
                    
                    # Generate image
                    await image_gen.generate_image(prompt, output_path)
                    
                    print(f"\nImage {i+1} generated successfully!")
                    print(f"Saved to: {output_path}")
                    print(f"Prompt saved to: {output_path.with_suffix('.txt')}")
                    
                    # Wait if interval is specified
                    if interval and i < batch_size - 1:
                        print(f"\nWaiting {interval} seconds...")
                        await asyncio.sleep(interval)
                        
                except Exception as e:
                    print(f"Error generating image {i+1}: {str(e)}")
                    if i < batch_size - 1:
                        print("Continuing with next image...")
                        if interval:
                            await asyncio.sleep(interval)
                        continue
                    raise
                    
            # Final cleanup
            prompt_gen.cleanup()
            image_gen.cleanup()
            print(f"\nBatch generation complete! {batch_size} images created.")
            
        except Exception as e:
            typer.echo(f"Error: {str(e)}", err=True)
            raise typer.Exit(1)
        finally:
            # Ensure cleanup happens even if there's an error
            try:
                prompt_gen.cleanup()
                image_gen.cleanup()
            except:
                pass

    try:
        asyncio.run(_loop())
        raise typer.Exit(0)  # Explicitly exit with success code
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)
