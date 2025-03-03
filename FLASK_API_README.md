# Wan2.1 Flask API Service

This project provides a Flask-based API service for Wan2.1 video generation models. It exposes endpoints for text-to-video, image-to-video, and video-to-video generation, allowing you to integrate Wan2.1's capabilities into your own applications or workflows.

## Features

- REST API for Wan2.1 models (1.3B and 14B models)
- Supports all generation modes:
  - Text-to-Video
  - Image-to-Video
  - Video-to-Video
- Intelligent model caching to minimize VRAM usage
- Docker integration for easy deployment
- Configurable via environment variables

## Requirements

- NVIDIA GPU with sufficient VRAM (at least 8GB, 24GB+ recommended for larger models)
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Setup

1. Clone this repository
2. Build and run the Docker container:

```bash
docker compose up -d
```

3. Download the models using one of two methods:

**Method 1: Download directly into the container**
```bash
# Download the 1.3B Text-to-Video model (lowest VRAM requirements)
./download-container-models.sh 1

# Or for other models:
# 2: Image-to-Video 720p (14B model)
# 3: Image-to-Video 480p (14B model)
# 4: Text-to-Video 14B model
# 5: Qwen2.5-14B-Instruct (for prompt enhancement)
# 6: All models
./download-container-models.sh 6  # Downloads all models (very large download)
```

**Method 2: Download manually and mount to container**
1. Download the required model files from HuggingFace
2. Place them in the `models/` directory according to the structure below
3. Create a `.env` file (if needed) from the `.env.example` template

## Model Directory Structure

The models should be organized in this structure:

```
models/
  Wan-AI/
    Wan2.1-T2V-1.3B/
      diffusion_pytorch_model.safetensors
      models_t5_umt5-xxl-enc-bf16.pth
      Wan2.1_VAE.pth
    Wan2.1-T2V-14B/
      diffusion_pytorch_model-00001-of-00006.safetensors
      ...
      diffusion_pytorch_model-00006-of-00006.safetensors
      models_t5_umt5-xxl-enc-bf16.pth
      Wan2.1_VAE.pth
    Wan2.1-I2V-14B-720P/
      ...
    Wan2.1-I2V-14B-480P/
      ...
```

## API Usage

### Text-to-Video Generation

```bash
curl -X POST http://localhost:5000/text-to-video \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat riding a skateboard in a city street",
    "negative_prompt": "blurry, distorted",
    "model_size": "1.3B",
    "num_frames": 81,
    "aspect_ratio": "16:9",
    "inference_steps": 50,
    "seed": 42
  }' \
  --output output_video.mp4
```

### Image-to-Video Generation

```bash
curl -X POST http://localhost:5000/image-to-video \
  -F "prompt=A cat riding a skateboard in a city street" \
  -F "image=@/path/to/input/image.jpg" \
  -F "model_quality=480p" \
  -F "num_frames=81" \
  -F "aspect_ratio=16:9" \
  --output output_video.mp4
```

### Video-to-Video Generation

```bash
curl -X POST http://localhost:5000/video-to-video \
  -F "prompt=A cat riding a skateboard in a city street" \
  -F "video=@/path/to/input/video.mp4" \
  -F "denoising_strength=0.7" \
  -F "num_frames=81" \
  -F "aspect_ratio=16:9" \
  --output output_video.mp4
```

## API Endpoints

### GET /

Returns basic information about the API.

### GET /health

Health check endpoint.

### POST /text-to-video

Generate a video from text.

Parameters:
- `prompt` (string, required): The text description of the video to generate
- `negative_prompt` (string): What to avoid in the generation
- `model_size` (string): Either "1.3B" or "14B"
- `num_frames` (integer): Number of frames to generate (default: 81)
- `inference_steps` (integer): Number of diffusion steps (default: 50)
- `seed` (integer): Random seed for reproducibility
- `cfg_scale` (float): Classifier-free guidance scale (default: 5.0)
- `sigma_shift` (float): Sigma shift value (default: 5.0)
- `aspect_ratio` (string): Predefined aspect ratio (e.g., "16:9", "1:1")
- `width_height` (string): Custom resolution in format "width*height"
- `vram_setting` (string): VRAM allocation for model (default: 12000000000)
- `torch_dtype` (string): Either "torch.float8_e4m3fn" or "torch.bfloat16"
- `tiled` (boolean): Use tiled processing (default: true)
- `enhance_prompt` (boolean): Use prompt enhancement (default: false)
- `enhance_language` (string): Target language for prompt enhancement

### POST /image-to-video

Generate a video from an image.

Parameters:
- `prompt` (string, required): The text description of the video to generate
- `image` (file, required): Input image file
- ...plus many of the same parameters as text-to-video

### POST /video-to-video

Generate a new video from an existing video.

Parameters:
- `prompt` (string, required): The text description of the video to generate
- `video` (file, required): Input video file
- `denoising_strength` (float): How much to change the original video (0.0-1.0)
- ...plus many of the same parameters as text-to-video

## VRAM Considerations

The models in Wan2.1 require significant GPU VRAM:
- 1.3B model: ~8GB minimum
- 14B models: ~24GB minimum (48GB+ recommended)

You can adjust VRAM usage with the `vram_setting` parameter.