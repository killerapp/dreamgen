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
   ```bash
   uv python -m src.main generate --interactive  # Interactive mode
   uv python -m src.main generate               # Single generation
   uv python -m src.main loop                   # Continuous generation
   ```

### Project Structure

```
continuous-image-gen/
├── src/
│   ├── generators/
│   │   ├── prompt_generator.py    # Ollama integration
│   │   └── image_generator.py     # Flux 1.1 integration
│   ├── utils/
│   │   ├── storage.py            # Directory management
│   │   └── cli.py               # CLI interface
│   └── main.py                  # Entry point
├── output/
│   └── [year]/
│       └── week_[XX]/           # Auto-created weekly dirs
└── prompts/
    └── examples.py              # Example prompts
```

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
