# Continuous Image Generation

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux 1.1 for image generation.
![ComfyUI_00196_](https://github.com/user-attachments/assets/c5534dd2-878f-484f-932b-79df132c9481)

## Features

- AI-powered prompt generation using Ollama
- Image generation using Flux 1.1 transformers model
- Interactive mode for prompt feedback and editing
- Automatic weekly directory organization for outputs
- Continuous generation loop with optional intervals and batch settings
- Prompt history saved alongside generated images
- Plugin system for dynamic prompt enhancement:
  - Time of day context (morning/afternoon/evening/night)
  - Day of week awareness
  - Holiday detection and theming
  - Art style variation (90+ distinct styles)
  - Extensible plugin architecture for custom enhancements

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- CUDA-capable GPU with at least 8GB VRAM recommended
- uv package manager (install using `pip install uv`)

## Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/killerapp/continuous-image-gen
   cd continuous-image-gen
   ```

2. Set up the project using uv:
   ```powershell
   # Initialize virtual environment and install dependencies
   uv sync
   
   # For development with extra dependencies
   uv sync --dev
   ```

Note: Always use `uv sync` when updating dependencies or switching branches to ensure your environment matches the project requirements.

## Usage

The application provides an intuitive CLI powered by Typer. All commands are executed using `uv run imagegen`.

### Quick Start

```powershell
# Generate a single image
uv run imagegen generate

# Generate with interactive prompt feedback
uv run imagegen generate --interactive

# Generate multiple images
uv run imagegen loop --batch-size 10 --interval 300

# Show help and available options
uv run imagegen --help
```

### Available Commands

1. `generate` - Create a single image
   ```powershell
   uv run imagegen generate [OPTIONS]
   
   Options:
   -i, --interactive      Enable interactive mode with prompt feedback
   -m, --model TEXT      Ollama model to use (default: phi4:latest)
   -f, --flux-model TEXT Choose model variant: 'dev' (quality) or 'schnell' (speed)
   -p, --prompt TEXT     Provide a custom prompt
   --height INT         Image height in pixels (128-2048, default: 768)
   --width INT          Image width in pixels (128-2048, default: 1360)
   -s, --steps INT      Number of inference steps (1-150, default: 50 for dev, 4 for schnell)
   -g, --guidance FLOAT Guidance scale (1.0-30.0, default: 7.5 for dev, 0.0 for schnell)
   --true-cfg FLOAT    True classifier-free guidance scale (1.0-10.0, default: 1.0)
   --max-seq-len INT   Maximum sequence length (64-2048, default: 512)
   --cpu-only          Force CPU-only mode (not recommended)
   ```

2. `loop` - Generate multiple images in a batch
   ```powershell
   uv run imagegen loop [OPTIONS]
   
   Options:
   -b, --batch-size INT  Number of images to generate (1-100)
   -n, --interval INT   Seconds between generations
   -m, --model TEXT     Ollama model to use (default: phi4:latest)
   -f, --flux-model TEXT Choose model variant: 'dev' (quality) or 'schnell' (speed)
   --height INT         Image height in pixels (128-2048, default: 768)
   --width INT          Image width in pixels (128-2048, default: 1360)
   -s, --steps INT      Number of inference steps (1-150, default: 50 for dev, 4 for schnell)
   -g, --guidance FLOAT Guidance scale (1.0-30.0, default: 7.5 for dev, 0.0 for schnell)
   --true-cfg FLOAT    True classifier-free guidance scale (1.0-10.0, default: 1.0)
   --max-seq-len INT   Maximum sequence length (64-2048, default: 512)
   --cpu-only          Force CPU-only mode (not recommended)
   ```

3. Version Information
   ```powershell
   uv run imagegen --version
   ```

### Plugin System

The application features a modular plugin system that enhances prompt generation with contextual information:

1. Time of Day Plugin
   - Automatically detects current time
   - Categorizes into morning/afternoon/evening/night
   - Influences scene lighting and atmosphere in prompts

2. Day of Week Plugin
   - Adds current day context
   - Useful for generating scene-appropriate activities

3. Holiday Plugin
   - Tracks upcoming holidays
   - Provides "holiday facts" about current day
   - Automatically themes prompts around nearby celebrations

4. Art Style Plugin
   - Database of 90+ distinct art styles
   - Ranges from classical to contemporary
   - Each style includes detailed characteristics
   - Randomly selects styles to vary artistic approach

Example prompt with plugin enhancements:
```
night, Monday, approaching Christmas, in the style of Art Nouveau (organic forms, flowing lines, nature-inspired patterns)
```

### Creating Custom Plugins

Plugins can be created by:
1. Adding a new Python module in `src/plugins/`
2. Implementing a function that returns contextual information
3. Registering the plugin in `src/plugins/__init__.py`

Basic plugin template:
```python
def get_my_context() -> str:
    """
    Get contextual information for prompt enhancement.
    
    Returns:
        str: Context to be incorporated into the prompt
    """
    # Your implementation here
    return "context string"
```

### Interactive Mode Details

Interactive Mode enables you to provide feedback on the generated prompt before finalizing image creation. When using the `--interactive` flag:
- The system will generate a prompt and then allow you to review and provide feedback.
- This mode is ideal if you want to refine or adjust the prompt before image processing begins.
- After confirmation, the image generation proceeds with your input incorporated.

### Image Generation Models

The system supports two Flux model variants, each optimized for different use cases:

1. **Dev Model** (`-f dev`): High-quality generation
   - Default for detailed, high-fidelity images
   - Uses 50 inference steps
   - Guidance scale of 7.5 for strong prompt adherence
   - Best for final, production-quality images
   - Example: `uv run imagegen generate -f dev --height 1024 --width 1024`

2. **Schnell Model** (`-f schnell`): Fast generation
   - Optimized for rapid prototyping
   - Uses only 4 inference steps
   - Guidance scale of 0.0 for faster processing
   - Best for testing prompts or quick iterations
   - Example: `uv run imagegen generate -f schnell --steps 4 --guidance 0.0`

### Environment Variables

You can customize the behavior using these environment variables:

- `OLLAMA_MODEL`: Default Ollama model for prompt generation (default: phi4:latest)
- `OLLAMA_TEMPERATURE`: Temperature for prompt generation (default: 0.7)
- `FLUX_MODEL`: Model variant for image generation (default: dev)
- `IMAGE_HEIGHT`: Output image height in pixels (default: 768)
- `IMAGE_WIDTH`: Output image width in pixels (default: 1360)
- `NUM_INFERENCE_STEPS`: Number of denoising steps (default: 50 for dev, 4 for schnell)
- `GUIDANCE_SCALE`: Guidance scale for image generation (default: 7.5 for dev, 0.0 for schnell)
- `TRUE_CFG_SCALE`: True classifier-free guidance scale (default: 1.0)
- `MAX_SEQUENCE_LENGTH`: Maximum sequence length for text processing (default: 512)

For example, to set a custom model in PowerShell:
```powershell
$env:OLLAMA_MODEL = "mistral"
uv run imagegen generate
```

### Output

Generated images and their corresponding prompt files are organized in weekly directories:
```
output/
└── [year]/
    └── week_[XX]/
        ├── image_[timestamp]_[prompt_hash].png
        └── image_[timestamp]_[prompt_hash].txt
```

## Known Issues

- CLIP Token Limit: The current implementation uses experimental embeddings to work around CLIP's 77 token limit. This may affect prompt processing for very long descriptions.
- Plugin Ordering: The order of plugin-provided context in the final prompt is fixed. Future versions may allow customizable ordering.

## Development

Format code:
```powershell
uv run -m black src/
uv run -m isort src/
```

Run linting:
```powershell
uv run -m pylint src/
```

Run tests:
```powershell
uv run -m pytest
```

## License

MIT License

Copyright (c) 2024 killerapp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
