# 🤖 Do AIs Dream of Creative Prompts?

A Python application that continuously generates creative images using AI. It uses Ollama for generating creative prompts and Flux for image generation. Let your machine dream up endless artistic possibilities! ✨

Like electric sheep in the dreams of androids, this project explores the boundaries between human and artificial creativity. What does AI imagine when we let it dream? 🌠

![ComfyUI_00196_](https://github.com/user-attachments/assets/c5534dd2-878f-484f-932b-79df132c9481)

## 🚀 Quick Start

1. Install prerequisites:
   - Python 3.11 or higher
   - Ollama (from [ollama.ai](https://ollama.ai))
   - CUDA-capable GPU (8GB+ VRAM recommended)
   - uv package manager (install using [astral](https://astral.sh/uv/install))

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
   ```

## ✨ Features

- AI-powered prompt generation using Ollama
- Image generation using Flux transformers
- Interactive mode for prompt feedback (be the art director!)
- Weekly organized outputs with prompt history
- Plugin system for dynamic prompt enhancement:
  - Time of day context (morning/afternoon/evening/night)
  - Holiday detection and theming (because every day is special 🎉)
  - Art style variation (90+ distinct styles)
  - Extensible plugin architecture (PRs welcome! 🙌)

## 🎮 Command Reference

### Generate Single Image
```bash
uv run imagegen generate [OPTIONS]

Options:
-i, --interactive      Enable interactive mode
-m, --model TEXT      Ollama model (default: phi4:latest)
-f, --flux-model TEXT Model variant: 'dev' or 'schnell'
-p, --prompt TEXT     Custom prompt
--height INT         Image height (128-2048, default: 768)
--width INT          Image width (128-2048, default: 1360)
-s, --steps INT      Inference steps (1-150)
-g, --guidance FLOAT Guidance scale (1.0-30.0)
--true-cfg FLOAT    True CFG scale (1.0-10.0)
--cpu-only          Force CPU mode (slower but hey, it works! 🐌)
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
```

## 🔌 Plugin Development

Got a cool idea for a plugin? We'd love to see it! PRs are very welcome for new plugins that add creative context to our prompts. 

Create new plugins in `src/plugins/`:
```python
def get_my_context() -> str:
    """Add custom context to prompts."""
    return "your awesome context here ✨"
```

Register in `src/plugins/__init__.py`

Some plugin ideas we'd love to see:
- Music mood integration 🎵
- Local events awareness 🎪
- Color palette themes 🎨
- Cultural celebrations 🌍
- Astronomy conditions 🌟

## 📁 Output Structure
```
output/
└── [year]/
    └── week_[XX]/
        ├── image_[timestamp]_[prompt_hash].png
        └── image_[timestamp]_[prompt_hash].txt
```

## 🐛 Known Issues

- CLIP Token Limit: Using experimental embeddings for >77 tokens
- Fixed plugin context ordering in prompts (but hey, we're working on it! 🔧)

## 📜 License

MIT License - See LICENSE file for details

Note: The Flux models have their own separate licensing terms. The dev variant is for non-commercial use only, while the schnell variant includes a commercial-friendly license.

## 🤝 Contributing

Contributions are super welcome! Whether it's:
- New plugins (the more creative, the better!)
- Bug fixes (squash those bugs! 🐞)
- Documentation improvements (help others learn!)
- Feature suggestions (dream big! ✨)

Let's make image generation more fun and creative together! 🎨✨
