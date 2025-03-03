# 🤖 Do LLMs Dream of electric sheep?

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux for image generation. Let your machine dream up endless artistic possibilities! ✨

Like electric sheep in the dreams of androids, this project explores the boundaries between human and artificial creativity. What does AI imagine when we let it dream? 🌠

![Do androids dream of electric sheep?](https://host-image.agentic.workers.dev/)

## 🚀 Quick Start

1. Install prerequisites:
   - uv Python manager (install using [astral](https://astral.sh/uv/install))
   - Ollama (from [ollama.ai](https://ollama.ai))
   - CUDA-capable GPU (8GB+ VRAM recommended) or Apple Silicon Mac (M1/M2/M3/M4)

2. Set up the project:
   ```bash
   git clone https://github.com/killerapp/continuous-image-gen
   cd continuous-image-gen
   uv sync  # Install dependencies
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
   ```

## ✨ Features

- AI-powered prompt generation using Ollama
- Image generation using Flux transformers
- Interactive mode for prompt feedback (be the art director!)
- Lora support for custom model fine-tuning
- Plugin system for dynamic prompt enhancement:
  - Time of day context (morning/afternoon/evening/night)
  - Holiday detection and theming (because every day is special 🎉)
  - Art style variation (90+ distinct styles)
  - Lora integration (custom model fine-tuning)
  - Extensible plugin architecture (PRs welcome! 🙌)

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
```

### Generate Multiple Images
```bash
uv run imagegen loop [OPTIONS]

Options:
-b, --batch-size INT Number of images (1-100)
-n, --interval INT  Seconds between generations
[+ same options as generate command]
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
1. **Automatic Mode**: Let the system generate prompts with your Lora
   ```bash
   uv run imagegen generate
   ```

2. **Manual Mode**: Force a prompt with your Lora as the subject
   ```bash
   uv run imagegen generate -p "Evening scene with 'your_lora_name' as the main character walking through a cyberpunk city"
   ```

Note: When using Loras, always make the Lora keyword a central subject in your prompt using single quotes, e.g., 'your_lora_name'.

## ⚙️ Environment Configuration

Set these environment variables before running:
```bash
# Default values shown
export OLLAMA_MODEL=phi4:latest
export OLLAMA_TEMPERATURE=0.7
export FLUX_MODEL=dev
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

[Rest of README remains unchanged...]
