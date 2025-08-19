#!/bin/sh
set -e

echo "Starting Continuous Image Generator Backend..."

# Create necessary directories if they don't exist
mkdir -p /app/output /app/models /app/logs

# Check if we're in mock mode
if [ "$USE_MOCK_GENERATOR" = "true" ] || [ "$MOCK_MODE" = "true" ]; then
    echo "Running in MOCK mode (no GPU required)"
else
    echo "Running in PRODUCTION mode with Flux models"
    
    # Check for HF token if not in mock mode
    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN not set. Model downloads may fail."
    fi
    
    # Check if models are mounted
    if [ ! -d "/app/models" ] || [ -z "$(ls -A /app/models)" ]; then
        echo "Warning: Models directory is empty. Models will be downloaded on first use."
        echo "This may take significant time and bandwidth (~15GB)."
    else
        echo "Models directory found with existing content."
    fi
fi

# Check Ollama connectivity
if [ ! -z "$OLLAMA_HOST" ]; then
    echo "Checking Ollama connectivity at $OLLAMA_HOST..."
    if curl -f -s "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "Ollama is reachable at $OLLAMA_HOST"
    else
        echo "Warning: Cannot reach Ollama at $OLLAMA_HOST. Prompt generation may use fallback."
    fi
fi

# Export Python path
export PYTHONPATH=/app:$PYTHONPATH

# Start the application
echo "Starting API server on port ${PORT:-8000}..."
exec "$@"