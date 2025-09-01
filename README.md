# ✨ DreamGen

Generate unlimited AI images locally with no subscriptions, no cloud APIs, and complete privacy. Your machine dreams with you! ✨

![Do androids dream of electric sheep?](https://host-image.agentic.workers.dev/)

## ✨ Modern Web Interface

Beautiful, VS Code-inspired dark theme with real-time generation and organized galleries. The web interface features:

- **🎨 Smart Generation Dashboard** - AI-enhanced prompts with contextual plugins
- **🖼️ Weekly Gallery Organization** - Browse your creations by week with thumbnail previews  
- **⚙️ Plugin Management** - Configure time-aware and artistic enhancement plugins
- **📊 Real-time Status** - Monitor API, GPU, and generation progress

## 🚀 Quick Install

```bash
# Install the CLI tool
uv tool install dreamgen

# Clone for web interface (optional)
git clone https://github.com/killerapp/dreamgen
cd dreamgen/web-ui && npm install

# Set up your environment
export HUGGINGFACE_TOKEN=your_token_here

# Start generating images!
dreamgen generate

# Launch the web interface
npm run dev
```

## 🔑 Why Choose This?

- **🏠 100% Local**: No cloud APIs, no usage limits, complete privacy
- **🧠 Smart Prompts**: AI-enhanced prompts with time, holidays, and art styles  
- **🌐 Modern UI**: Professional web interface with galleries and real-time updates
- **💰 Zero Cost**: Generate unlimited images after initial setup
- **🔌 Extensible**: Plugin system for custom prompt enhancements

## 🎮 Quick Commands

```bash
# Generate a single image
dreamgen generate

# Generate with interactive prompt refinement  
dreamgen generate --interactive

# Generate multiple images in a batch
dreamgen loop --batch-size 10 --interval 300

# Use mock mode (no GPU required)
dreamgen generate --mock

# Get help
dreamgen --help
```

## 🔧 Requirements

- **Python 3.11+** with uv package manager
- **Ollama** for prompt generation ([ollama.ai](https://ollama.ai))
- **Hugging Face Token** for model access
- **GPU recommended**: NVIDIA (8GB+ VRAM) or Apple Silicon

## 📖 Full Documentation

For detailed setup, plugin development, and advanced usage, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

Built by [Agentic Insights](https://agenticinsights.com) • [Report Issues](https://github.com/killerapp/dreamgen/issues)
