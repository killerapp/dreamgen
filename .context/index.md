# Continuous Image Generation

## Overview

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux 1.1 for image generation.

## Architecture Details

### Components

- **Prompt Generator**: Uses Ollama for local LLM inference to generate creative prompts
- **Image Generator**: Uses Flux 1.1 transformers model for image generation
- **Storage Manager**: Handles weekly directory organization for outputs
- **CLI Interface**: Provides interactive and continuous generation modes

### Design Decisions

- **Local LLM Usage**: Ollama was chosen for local inference to avoid API costs and latency
- **Weekly Organization**: Images are organized by year/week for better file management
- **Async Processing**: Used for efficient handling of both prompt and image generation
- **Interactive Mode**: Allows prompt refinement before image generation

## Development Guide

### Environment Setup

This project uses `uv` for all Python operations. Dependencies are managed through `pyproject.toml`.

1. Initial Setup:
   ```bash
   uv venv                           # Create virtual environment
   uv add --editable .              # Install package in development mode
   uv sync                          # Install/update all dependencies from pyproject.toml
   ```

2. Dependency Management:
   ```bash
   uv add package-name              # Add a new dependency to pyproject.toml
   uv add --dev package-name        # Add a new dev dependency to pyproject.toml
   uv sync                          # Sync dependencies after pyproject.toml changes
   ```

   Note: Always use `uv add` to manage dependencies as it automatically updates pyproject.toml.
   The project uses modern Python packaging with `pyproject.toml` as the single source
   of truth for dependencies. The `uv.lock` file ensures reproducible installations
   across different environments.

3. Running the Application:
   The CLI is implemented using Typer with rich progress indicators and proper error handling:
   ```bash
   uv run imagegen [command] [options]  # Main entry point
   ```

   Key CLI Features:
   - Rich progress bars and spinners for visual feedback
   - Proper error handling with nested try/except blocks
   - Cleanup handlers to ensure resource release
   - Version information via --version flag
   - Command groups for better organization
   - Input validation (e.g., batch size limits)
   - Interactive progress updates during batch operations

### Project Structure

```
continuous-image-gen/
├── src/
│   ├── generators/
│   │   ├── prompt_generator.py    # Ollama integration for prompt generation
│   │   └── image_generator.py     # Flux 1.1 integration for image creation
│   ├── utils/
│   │   ├── storage.py            # Weekly directory management
│   │   └── cli.py               # Typer-based CLI with rich formatting
│   └── main.py                  # Entry point with proper package imports
├── output/
│   └── [year]/
│       └── week_[XX]/           # Auto-created weekly directories
└── prompts/
    └── examples.py              # Example prompts for testing

CLI Architecture:
├── main.py                      # Exports app from cli.py
└── utils/
    └── cli.py                   # Core CLI implementation
        ├── app                  # Typer application instance
        ├── version_callback     # Version info handler
        ├── generate            # Single image generation
        └── loop                # Batch generation with progress
```

### CLI Implementation Details

The CLI is built using Typer and Rich libraries with several key design decisions:

1. Command Structure:
   - Root command: `imagegen`
   - Subcommands: `generate`, `loop`
   - Global options: `--version`

2. Progress Handling:
   ```python
   with Progress(
       SpinnerColumn(),
       TextColumn("[progress.description]{task.description}"),
       TimeElapsedColumn(),
   ) as progress:
       # Task tracking with visual feedback
   ```

3. Error Handling Strategy:
   - Nested try/except blocks for granular error control
   - Proper cleanup in finally blocks
   - Rich formatting for error messages
   - Graceful handling of KeyboardInterrupt

4. Resource Management:
   - Automatic cleanup of generators
   - Progress bar task cleanup
   - Proper async/await usage

5. User Feedback:
   - Rich panels for prompts and results
   - Color-coded status messages
   - Progress spinners for indeterminate tasks
   - Batch progress tracking

### Testing Strategy

- Unit tests for business logic
- Integration tests for API endpoints
- E2E tests for critical flows

## Contributing

### Guidelines

- Use `uv` for all Python operations
- Follow PEP 8 style guide
- Write tests for new features
- Document significant changes

### Common Tasks

1. Adding new features:
   - Create feature branch
   - Implement changes
   - Add tests
   - Update documentation

2. Development workflow:
   ```bash
   uv add --editable ".[dev]"  # Install package in editable mode with dev dependencies
   uv python -m black src/     # Format code
   uv python -m pylint src/    # Run linting
   uv python -m pytest         # Run tests
