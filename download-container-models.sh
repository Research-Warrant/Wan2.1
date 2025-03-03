#!/bin/bash

# This script helps download models directly into the container

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <model_choice>"
    echo "Where <model_choice> is one of:"
    echo "  1: Wan2.1-T2V-1.3B (1.3B Text-to-Video and Video-to-Video model, 3.5GB+ VRAM)"
    echo "  2: Wan2.1-I2V-14B-720P (Image to Video 720p)"
    echo "  3: Wan2.1-I2V-14B-480P (Image to Video 480p)"
    echo "  4: Wan2.1-T2V-14B (14B Text to Video)"
    echo "  5: Qwen2.5-14B-Instruct (for prompt enhancement)"
    echo "  6: All models"
    exit 1
fi

CONTAINER_NAME="wan-flask-api"
MODEL_CHOICE="$1"
HF_TOKEN="$2"  # Optional HuggingFace token

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    echo "Please start the container with: docker compose up -d"
    exit 1
fi

# Download the model
echo "Starting download for model choice $MODEL_CHOICE..."
if [ -n "$HF_TOKEN" ]; then
    docker exec -e HF_TOKEN="$HF_TOKEN" "$CONTAINER_NAME" /app/Download.sh "$MODEL_CHOICE"
else
    docker exec "$CONTAINER_NAME" /app/Download.sh "$MODEL_CHOICE"
fi

echo "Download process completed. Restart the container to use the downloaded models:"
echo "docker compose restart"