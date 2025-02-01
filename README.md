# Continuous Image Generation

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux 1.1 for image generation.

## Features

- AI-powered prompt generation using Ollama
- Image generation using Flux 1.1 transformers model
- Interactive mode for prompt feedback and editing
- Automatic weekly directory organization
- Continuous generation loop with optional intervals
- Prompt history saved alongside generated images

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- CUDA-capable GPU recommended for faster image generation
- uv package manager (`pip install uv`)

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd continuous-image-gen
```

2. Install dependencies using uv:
```bash
uv pip install -e .
```

For development, install with extra dependencies:
```bash
uv pip install -e ".[dev]"
```

## Usage

### Single Image Generation

Generate a single image:
```bash
uv python -m src.main generate
```

Generate with interactive prompt feedback:
```bash
uv python -m src.main generate --interactive
```

Use a different Ollama model:
```bash
uv python -m src.main generate --model mistral
```

### Continuous Generation

Run continuous generation (immediate):
```bash
uv python -m src.main loop
```

Run with interval:
```bash
uv python -m src.main loop --interval 300  # 5 minutes between generations
```

### Output

Generated images and their prompts are organized in weekly directories:
```
output/
└── [year]/
    └── week_[XX]/
        ├── image_[timestamp]_[prompt_hash].png
        └── image_[timestamp]_[prompt_hash].txt
```

## Development

Format code:
```bash
uv python -m black src/
uv python -m isort src/
```

Run linting:
```bash
uv python -m pylint src/
```

Run tests:
```bash
uv python -m pytest
```

## License

[Your chosen license]
