# 🤖 Do LLMs Dream of electric sheep?

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux for image generation. Let your machine dream up endless artistic possibilities! ✨

Like electric sheep in the dreams of androids, this project explores the boundaries between human and artificial creativity. What does AI imagine when we let it dream? 🌠

Built by [Agentic Insights](https://agenticinsights.com)

![Do androids dream of electric sheep?](https://host-image.agentic.workers.dev/)

## 🔑 Key Benefits

- **100% Local Processing**: Everything runs locally on your machine - no cloud APIs, no usage limits!
- **Privacy-First**: Your prompts and generated images never leave your computer
- **Internet-Optional**: Only connects to the internet to download model weights
- **Extensible Plugin System**: Enhance your prompts with local or remote data sources
- **No Subscription Fees**: Generate unlimited images without ongoing costs

## 🚀 Quick Start

1. Install prerequisites:
   - uv Python manager (install using [astral](https://astral.sh/uv/install))
   - Ollama (from [ollama.ai](https://ollama.ai))
   - CUDA-capable GPU (8GB+ VRAM recommended) or Apple Silicon Mac (M1/M2/M3/M4)
   - Hugging Face account with access token (for downloading models)

2. Set up the project:
   ```bash
   git clone https://github.com/killerapp/continuous-image-gen
   cd continuous-image-gen
   
   # Set your Hugging Face token (required to download models)
   # You must ungate the dev or schnell models on Hugging Face first
   export HUGGINGFACE_TOKEN=your_token_here
   
   # Install dependencies
   uv sync
   
   # For NVIDIA GPU support (CUDA), install PyTorch separately:
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # Note: uv sync may revert PyTorch to CPU version. After running uv sync,
   # always reinstall PyTorch with CUDA if you have an NVIDIA GPU.
   ```

3. Let the magic happen! ✨
   ```bash
   # Single image
   uv run imagegen generate

   # With interactive prompt feedback
   uv run imagegen generate --interactive

   # Multiple images (perfect for coffee breaks ☕)
   uv run imagegen loop --batch-size 10 --interval 300

   # Force a specific prompt (bypass Ollama)
   uv run imagegen generate -p "your custom prompt here"

   # Run without downloading large models (saves a placeholder image)
   uv run imagegen generate --mock

   # Enable verbose backend logging
   uv run imagegen --debug generate
   ```

4. Launch the modern web interface:
   ```bash
   # Start the FastAPI backend
   uv run uvicorn src.api.server:app --reload --port 8000
   
   # In a new terminal, start the Next.js frontend
   cd web-ui
   npm install
   npm run dev
   
   # Open http://localhost:7860 in your browser
   ```

## ✨ Features

- **Modern Web Interface**:
  - IDE-style dark theme with VS Code aesthetics
  - Real-time generation with WebSocket updates
  - Plugin management interface
  - Gallery view for browsing generated images
  - Built with Next.js, TypeScript, and Tailwind CSS
- **RESTful API with FastAPI**:
  - Full REST API for programmatic access
  - WebSocket support for real-time updates
  - Batch generation endpoints
  - Plugin management API
- **Powerful Plugin System** for dynamic prompt enhancement:
  - Time of day context (morning/afternoon/evening/night)
  - Holiday detection and theming (because every day is special 🎉)
  - Art style variation (90+ distinct styles)
  - Lora integration (custom model fine-tuning)
  - Extensible plugin architecture (PRs welcome! 🙌)
- AI-powered prompt generation using Ollama (runs 100% locally)
- Image generation using Flux transformers (runs 100% locally)
- Interactive mode for prompt feedback (be the art director!)
- Lora support for custom model fine-tuning

## 🔌 Plugin System

The plugin system is the heart of what makes this project special. It dynamically enhances prompts with contextual information to create more creative, relevant, and diverse images.

### How It Works

1. **Modular Architecture**: Each plugin is a standalone Python module that can be enabled/disabled independently
2. **Context Injection**: Plugins provide contextual information that gets seamlessly integrated into prompts
3. **Local & Remote Sources**: Plugins can use local data files or connect to remote APIs (while respecting your privacy settings)
4. **Easy Extensibility**: Create your own plugins with minimal code to add custom functionality

### Included Plugins

- **Time of Day**: Adapts prompts to morning, afternoon, evening, or night themes
- **Holiday Awareness**: Detects upcoming holidays and incorporates them into prompts
- **Art Style Variation**: Rotates through 90+ distinct art styles to keep generations fresh
- **Lora Integration**: Seamlessly incorporates your custom Lora models as subjects
- **Day of Week**: Adjusts prompts based on the current day of the week

### Creating Custom Plugins

The plugin system follows a simple interface pattern, making it easy to create your own:

```python
def get_context() -> Optional[str]:
    """Return contextual information to enhance prompts"""
    return "your custom context here"
```

Place your plugin in the `src/plugins/` directory and expose a `get_plugin` function. It will be discovered automatically. Your plugin can:
- Read from local data files
- Connect to APIs (with proper authentication)
- Use system information
- Implement caching for performance
- Maintain state between generations

### Managing Plugins

You can list, enable, or disable plugins from the CLI:

```bash
uv run imagegen plugins list
uv run imagegen plugins disable time_of_day
uv run imagegen plugins enable time_of_day
```

The API also exposes plugin management:

- `GET /api/plugins` – list all plugins and their status
- `POST /api/plugins/{name}` with `{ "enabled": true|false }` – update plugin state

And the web UI includes a sidebar where you can toggle plugins on or off.

## 🎮 Command Reference

### Generate Single Image
```bash
uv run imagegen generate [OPTIONS]

Options:
-i, --interactive      Enable interactive mode
-m, --model TEXT      Ollama model (default: phi4:latest)
-f, --flux-model TEXT Model variant: 'dev' or 'schnell'
-p, --prompt TEXT     Custom prompt (bypass Ollama generation)
--height INT         Image height (128-2048, default: 768)
--width INT          Image width (128-2048, default: 1360)
-s, --steps INT      Inference steps (1-150)
-g, --guidance FLOAT Guidance scale (1.0-30.0)
--true-cfg FLOAT    True CFG scale (1.0-10.0)
--cpu-only          Force CPU mode (slower but hey, it works! 🐌)
--mps-use-fp16      Use float16 precision on Apple Silicon (may improve performance for some models)
--mock              Use placeholder image generator (no models required)
```

### Generate Multiple Images
```bash
uv run imagegen loop [OPTIONS]

Options:
-b, --batch-size INT Number of images (1-100)
-n, --interval INT  Seconds between generations
[+ same options as generate command]
```

### Run System Diagnostics
```bash
uv run imagegen diagnose [OPTIONS]

Options:
-v, --verbose        Show detailed diagnostic information
--check-env/--no-check-env  Check environment variables (default: True)
--fix                Attempt to fix common issues automatically
```

### Launch Web UI
```bash
uv run imagegen web [OPTIONS]

Options:
--mock              Use placeholder image generator (no models required)
```

## 🎭 Model Variants

Flux offers two model variants with different licensing terms:

1. **Dev Model** (`-f dev`)
   ```bash
   uv run imagegen generate -f dev --height 1024 --width 1024
   ```
   - Non-commercial use only
   - High-quality output (for when you're feeling fancy 🎩)
   - 50 inference steps
   - 7.5 guidance scale
   - Best for personal projects and experimentation

2. **Schnell Model** (`-f schnell`)
   ```bash
   uv run imagegen generate -f schnell --steps 4 --guidance 0.0
   ```
   - Commercial-friendly license
   - Optimized for speed (zoom zoom! 🏃‍♂️)
   - 4 inference steps
   - 0.0 guidance scale
   - Suitable for production environments

Choose the appropriate model based on your use case and licensing requirements.

## 🍎 Apple Silicon Support

This project now supports Apple Silicon (M1/M2/M3/M4) Macs using PyTorch's Metal Performance Shaders (MPS) backend. The system will automatically detect Apple Silicon and use the appropriate GPU acceleration.

### Apple Silicon Tips

- Performance is generally good on Apple Silicon, but may vary depending on model complexity
- By default, the system uses float32 precision on MPS for better compatibility
- You can enable float16 precision with the `--mps-use-fp16` flag for potentially better performance
- Memory management on Apple Silicon is handled automatically through the unified memory architecture
- For best results on Apple Silicon, consider using the Schnell model variant which is optimized for speed

```bash
# Example: Running on Apple Silicon with float16 precision
uv run imagegen generate --mps-use-fp16

# Example: Running the faster Schnell model on Apple Silicon
uv run imagegen generate -f schnell --mps-use-fp16
```

## 🎨 Lora Support

The system supports Lora models for custom fine-tuning. Loras are loaded from subdirectories in your Lora directory, with automatic version selection.

Loras can be used to add specific likenesses (people, characters) or artistic styles to your generated images. The plugin system **automatically integrates Loras into your prompts** when they are enabled, making it seamless to add your favorite characters or styles to generated images.

### Lora Sources
- [Fal.ai](https://fal.ai/) - Offers high-quality Loras for various styles and subjects
- [CivitAI](https://civitai.com/) - Large community library of Loras for characters and styles
- [Hugging Face](https://huggingface.co/) - Many open-source Loras with various licenses

### Configuration
```bash
# Lora Configuration in .env
LORA_DIR=C:/ComfyUI/ComfyUI/models/loras
ENABLED_LORAS=your_lora_name
LORA_APPLICATION_PROBABILITY=0.99
```

### Directory Structure
```
loras/
└── your_lora_name/
    ├── your_lora_name-000004.safetensors
    ├── your_lora_name-000008.safetensors
    └── your_lora_name-000012.safetensors  # Latest version used
```

### Using Loras

#### Automatic Integration (Recommended)
The system will automatically:
1. Randomly select from your enabled Loras based on the configured probability
2. Integrate the selected Lora as a central character/subject in the generated prompt
3. Format the Lora keyword properly with single quotes (e.g., 'your_lora_name')

Simply run:
```bash
uv run imagegen generate
# or
uv run imagegen loop --batch-size 10
```

#### Manual Prompt with Lora
If you prefer to craft your own prompt with a specific Lora:
```bash
uv run imagegen generate -p "Evening scene with 'your_lora_name' as the main character walking through a cyberpunk city"
```

> **How it works**: The Lora plugin detects enabled Loras, selects one based on your configuration, and instructs the prompt generator to make the Lora a central subject in the scene. This happens automatically in continuous generation mode.

## 🌐 Host-Image Feature

Share your AI-generated masterpieces with the world! This feature allows you to have your latest generated image available on a public endpoint using Cloudflare Workers and R2 storage.

> **Note**: This feature is already in use to serve the image at the top of this README!

### How It Works
1. Your generated images are uploaded to a Cloudflare R2 bucket
2. A Cloudflare Worker serves the latest image via a public URL
3. You can embed this URL anywhere (websites, social media, etc.)

### Requirements
- Cloudflare account with Workers and R2 access
- Basic knowledge of Cloudflare Workers deployment

### Setup Instructions
1. Clone the host-image directory
2. Configure your R2 bucket in wrangler.jsonc
3. Deploy using `wrangler deploy`

### Usage
Once deployed, your image will be available at your worker's URL:
```
https://host-image.yourdomain.workers.dev/
```

Perfect for embedding in websites, sharing on social media, or creating an always-updating display of your AI art!

## ⚙️ Environment Configuration

Set these environment variables before running:
```bash
# Default values shown
export HUGGINGFACE_TOKEN=your_token_here  # Required for downloading models
export OLLAMA_MODEL=phi4:latest
export OLLAMA_TEMPERATURE=0.7
export FLUX_MODEL=dev  # Must ungate this model on Hugging Face first
export IMAGE_HEIGHT=768
export IMAGE_WIDTH=1360
export NUM_INFERENCE_STEPS=50  # 50 for dev, 4 for schnell
export GUIDANCE_SCALE=7.5      # 7.5 for dev, 0.0 for schnell
export TRUE_CFG_SCALE=1.0
export MAX_SEQUENCE_LENGTH=512

# Lora Configuration
export LORA_DIR=C:/ComfyUI/ComfyUI/models/loras
export ENABLED_LORAS=your_lora_name
export LORA_APPLICATION_PROBABILITY=0.99
```
