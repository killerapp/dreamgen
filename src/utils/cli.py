"""
Command-line interface for the continuous image generation system.
Provides a unified interface for generating AI images with various options.
"""
import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

from ..generators.prompt_generator import PromptGenerator
from ..generators.image_generator import ImageGenerator
from .storage import StorageManager

# Initialize rich console for better output
console = Console()
app = typer.Typer(
    help="Continuous Image Generation CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

def version_callback(value: bool):
    """Display version information."""
    if value:
        console.print(
            Panel.fit(
                "[bold green]Continuous Image Generator[/bold green]\n"
                "Version: 0.1.0\n"
                "Using: Ollama for prompts, Flux 1.1 for images"
            )
        )
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", 
        callback=version_callback,
        help="Show version information and exit",
        is_eager=True
    ),
):
    """
    🎨 Continuous Image Generation System

    Generate AI images using Ollama for prompts and Flux for image generation.
    Run with 'uv run imagegen' followed by a command.
    """
    pass

@app.command(help="Generate a single image with optional interactive prompt refinement")
def generate(
    interactive: bool = typer.Option(
        False, "--interactive", "-i", 
        help="Enable interactive mode with prompt feedback"
    ),
    model: str = typer.Option(
        os.getenv('OLLAMA_MODEL', 'phi4:latest'), 
        "--model", "-m", 
        help="Ollama model to use for prompt generation"
    ),
    flux_model: str = typer.Option(
        os.getenv('FLUX_MODEL', 'dev'),
        "--flux-model", "-f",
        help="Flux model variant to use: 'dev' (high quality) or 'schnell' (fast)",
        case_sensitive=False
    ),
    height: int = typer.Option(
        int(os.getenv('IMAGE_HEIGHT', 768)),
        "--height",
        help="Height of generated image in pixels",
        min=128, max=2048
    ),
    width: int = typer.Option(
        int(os.getenv('IMAGE_WIDTH', 1360)),
        "--width",
        help="Width of generated image in pixels",
        min=128, max=2048
    ),
    steps: int = typer.Option(
        int(os.getenv('NUM_INFERENCE_STEPS', 50)),
        "--steps", "-s",
        help="Number of inference steps (more = higher quality but slower)",
        min=1, max=150
    ),
    guidance: float = typer.Option(
        float(os.getenv('GUIDANCE_SCALE', 7.5)),
        "--guidance", "-g",
        help="Guidance scale (how closely to follow the prompt)",
        min=1.0, max=30.0
    ),
    true_cfg: float = typer.Option(
        float(os.getenv('TRUE_CFG_SCALE', 1.0)),
        "--true-cfg",
        help="True classifier-free guidance scale",
        min=1.0, max=10.0
    ),
    max_seq_len: int = typer.Option(
        int(os.getenv('MAX_SEQUENCE_LENGTH', 512)),
        "--max-seq-len",
        help="Maximum sequence length for text processing",
        min=64, max=2048
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", 
        help="Force CPU-only mode (not recommended)"
    ),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", 
        help="Provide a custom prompt for direct inference"
    ),
) -> None:
    """
    Generate a single image using AI-generated prompts or a custom prompt.
    
    Examples:
        uv run imagegen generate
        uv run imagegen generate --interactive
        uv run imagegen generate --prompt "your custom prompt"
    """
    async def _generate() -> None:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                try:
                    # Initialize components
                    init_task = progress.add_task("[cyan]Initializing components...", total=None)
                    prompt_gen = PromptGenerator(model_name=model)
                    image_gen = ImageGenerator(
                        model_variant=flux_model.lower(),
                        cpu_only=cpu_only,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        true_cfg_scale=true_cfg,
                        max_sequence_length=max_seq_len
                    )
                    storage = StorageManager()
                    progress.remove_task(init_task)

                    # Use provided prompt or generate one
                    if prompt:
                        generated_prompt = prompt
                        console.print(Panel(
                            f"[bold]Using provided prompt:[/bold]\n\n{generated_prompt}",
                            title="Custom Prompt",
                            border_style="blue"
                        ))
                    else:
                        prompt_task = progress.add_task(
                            "[cyan]Generating creative prompt...", 
                            total=None
                        )
                        if interactive:
                            generated_prompt = await prompt_gen.get_prompt_with_feedback()
                        else:
                            generated_prompt = await prompt_gen.generate_prompt()
                        progress.remove_task(prompt_task)
                        console.print(Panel(
                            f"[bold]Generated prompt:[/bold]\n\n{generated_prompt}",
                            title="AI Prompt",
                            border_style="green"
                        ))

                    # Get output path
                    output_path = storage.get_output_path(generated_prompt)
                    
                    # Generate image
                    image_task = progress.add_task("[cyan]Generating image...", total=None)
                    output_path, gen_time, model_name = await image_gen.generate_image(generated_prompt, output_path)
                    progress.remove_task(image_task)
                    
                    # Show success message with details
                    console.print(Panel(
                        f"[bold green]Image generated successfully![/bold green]\n\n"
                        f"📁 Saved to: {output_path}\n"
                        f"📝 Prompt saved to: {output_path.with_suffix('.txt')}\n\n"
                        f"[dim]Model: {model_name}\n"
                        f"Time: {gen_time:.1f}s\n"
                        f"Prompt: {generated_prompt}[/dim]",
                        title="Success",
                        border_style="green"
                    ))
                    
                    # Cleanup
                    image_gen.cleanup()
                except Exception as e:
                    console.print(f"[red]Error: {str(e)}[/red]", err=True)
                    raise
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]", err=True)
            raise typer.Exit(1)

    try:
        asyncio.run(_generate())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

@app.command(help="Generate multiple images in a batch with configurable settings")
def loop(
    batch_size: int = typer.Option(
        5, "--batch-size", "-b", 
        help="Number of images to generate per run",
        min=1, max=100
    ),
    interval: Optional[int] = typer.Option(
        None, "--interval", "-n", 
        help="Interval in seconds between generations",
        min=0
    ),
    model: str = typer.Option(
        os.getenv('OLLAMA_MODEL', 'phi4:latest'), 
        "--model", "-m", 
        help="Ollama model to use for prompt generation"
    ),
    flux_model: str = typer.Option(
        os.getenv('FLUX_MODEL', 'dev'),
        "--flux-model", "-f",
        help="Flux model variant to use: 'dev' (high quality) or 'schnell' (fast)",
        case_sensitive=False
    ),
    height: int = typer.Option(
        int(os.getenv('IMAGE_HEIGHT', 768)),
        "--height",
        help="Height of generated image in pixels",
        min=128, max=2048
    ),
    width: int = typer.Option(
        int(os.getenv('IMAGE_WIDTH', 1360)),
        "--width",
        help="Width of generated image in pixels",
        min=128, max=2048
    ),
    steps: int = typer.Option(
        int(os.getenv('NUM_INFERENCE_STEPS', 50)),
        "--steps", "-s",
        help="Number of inference steps (more = higher quality but slower)",
        min=1, max=150
    ),
    guidance: float = typer.Option(
        float(os.getenv('GUIDANCE_SCALE', 7.5)),
        "--guidance", "-g",
        help="Guidance scale (how closely to follow the prompt)",
        min=1.0, max=30.0
    ),
    true_cfg: float = typer.Option(
        float(os.getenv('TRUE_CFG_SCALE', 1.0)),
        "--true-cfg",
        help="True classifier-free guidance scale",
        min=1.0, max=10.0
    ),
    max_seq_len: int = typer.Option(
        int(os.getenv('MAX_SEQUENCE_LENGTH', 512)),
        "--max-seq-len",
        help="Maximum sequence length for text processing",
        min=64, max=2048
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", 
        help="Force CPU-only mode (not recommended)"
    ),
) -> None:
    """
    Generate a batch of images with unique prompts.
    
    Examples:
        uv run imagegen loop
        uv run imagegen loop --batch-size 10
        uv run imagegen loop --interval 60
    """
    async def _loop() -> None:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                try:
                    # Initialize components once
                    init_task = progress.add_task("[cyan]Initializing models...", total=None)
                    prompt_gen = PromptGenerator(model_name=model)
                    image_gen = ImageGenerator(
                        model_variant=flux_model.lower(),
                        cpu_only=cpu_only,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        true_cfg_scale=true_cfg,
                        max_sequence_length=max_seq_len
                    )
                    storage = StorageManager()
                    progress.remove_task(init_task)
                    
                    console.print(f"\n[bold]Starting batch generation of {batch_size} images...[/bold]")
                    
                    batch_task = progress.add_task(
                        "[cyan]Generating batch", 
                        total=batch_size
                    )
                    
                    for i in range(batch_size):
                        try:
                            progress.update(
                                batch_task,
                                description=f"[cyan]Generating image {i+1}/{batch_size}..."
                            )
                            
                            # Generate prompt
                            prompt = await prompt_gen.generate_prompt()
                            console.print(Panel(
                                f"[bold]Generated prompt for image {i+1}:[/bold]\n\n{prompt}",
                                title=f"Prompt {i+1}/{batch_size}",
                                border_style="blue"
                            ))

                            # Get output path and generate
                            output_path = storage.get_output_path(prompt)
                            output_path, gen_time, model_name = await image_gen.generate_image(prompt, output_path)
                            
                            console.print(
                                f"[green]✓[/green] Image {i+1} generated in {gen_time:.1f}s using {model_name}\n"
                                f"   📁 {output_path}"
                            )
                            
                            progress.update(batch_task, advance=1)
                            
                            # Wait if interval is specified
                            if interval and i < batch_size - 1:
                                wait_task = progress.add_task(
                                    f"[yellow]Waiting {interval}s before next generation...", 
                                    total=interval
                                )
                                for _ in range(interval):
                                    await asyncio.sleep(1)
                                    progress.update(wait_task, advance=1)
                                progress.remove_task(wait_task)
                                
                        except Exception as e:
                            console.print(f"[red]Error generating image {i+1}: {str(e)}[/red]")
                            if i < batch_size - 1:
                                console.print("[yellow]Continuing with next image...[/yellow]")
                                if interval:
                                    await asyncio.sleep(interval)
                                continue
                            raise
                        
                    # Final cleanup
                    prompt_gen.cleanup()
                    image_gen.cleanup()
                    console.print(Panel(
                        f"[bold green]Batch generation complete![/bold green]\n"
                        f"Successfully created {batch_size} images using {model_name}\n\n"
                        f"[dim]Model: {model_name}\n"
                        f"Steps: {steps}\n"
                        f"Guidance: {guidance}\n"
                        f"Resolution: {width}x{height}[/dim]",
                        title="Success",
                        border_style="green"
                    ))
                except Exception as e:
                    console.print(f"[red]Error: {str(e)}[/red]", err=True)
                    raise
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]", err=True)
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
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]", err=True)
        raise typer.Exit(1)
