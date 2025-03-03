#!/bin/bash

# Simple test script for the Wan2.1 Flask API

# Define API URL
API_URL="http://localhost:5000"

# Test the API health endpoint
echo "Testing API health..."
curl -s "$API_URL/health"
echo

# Test the text-to-video endpoint with a simple prompt
echo
echo "Testing text-to-video endpoint with a simple prompt..."
echo "This may take some time to generate the video..."
curl -X POST "$API_URL/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat running on a beach at sunset",
    "model_size": "1.3B",
    "num_frames": 81,
    "aspect_ratio": "16:9",
    "seed": 42
  }' --output test_output.mp4

echo
echo "If the API is working correctly and models are loaded, a video file named 'test_output.mp4' should be created."
echo "Otherwise, check the container logs: docker compose logs"