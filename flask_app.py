# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import sys
import random
import time
import uuid
import tempfile
from datetime import datetime

import torch
import flask
from flask import Flask, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_video, cache_image, str2bool

app = Flask(__name__)

# Global variables to store loaded models
loaded_pipeline = None
loaded_pipeline_config = {}
prompt_expander = None

# Model configuration mapping
ASPECT_RATIOS_1_3b = {
    "1:1":  (640, 640),
    "4:3":  (736, 544),
    "3:4":  (544, 736),
    "3:2":  (768, 512),
    "2:3":  (512, 768),
    "16:9": (832, 480),
    "9:16": (480, 832),
    "21:9": (960, 416),
    "9:21": (416, 960),
    "4:5":  (560, 704),
    "5:4":  (704, 560),
}

ASPECT_RATIOS_14b = {
    "1:1":  (960, 960),
    "4:3":  (1104, 832),
    "3:4":  (832, 1104),
    "3:2":  (1152, 768),
    "2:3":  (768, 1152),
    "16:9": (1280, 720),
    "16:9_low": (832, 480),
    "9:16": (720, 1280),
    "9:16_low": (832, 480),
    "21:9": (1472, 624),
    "9:21": (624, 1472),
    "4:5":  (864, 1072),
    "5:4":  (1072, 864),
}

def auto_crop_image(image, target_width, target_height):
    """
    Crops and downscales the image to exactly the target resolution.
    
    The function first crops the image centrally to match the target aspect ratio,
    then resizes it to the target dimensions.
    """
    w, h = image.size
    target_ratio = target_width / target_height
    current_ratio = w / h

    # Crop the image to the desired aspect ratio.
    if current_ratio > target_ratio:
        # Image is too wide: crop the left and right.
        new_width = int(h * target_ratio)
        left = (w - new_width) // 2
        right = left + new_width
        image = image.crop((left, 0, right, h))
    elif current_ratio < target_ratio:
        # Image is too tall: crop the top and bottom.
        new_height = int(w / target_ratio)
        top = (h - new_height) // 2
        bottom = top + new_height
        image = image.crop((0, top, w, bottom))

    # Resize to the target resolution.
    image = image.resize((target_width, target_height), Image.LANCZOS)
    return image

def load_wan_pipeline(model_choice, torch_dtype_str, num_persistent):
    """
    Loads the appropriate WAN pipeline based on:
      - model_choice: one of "1.3B", "14B_text", "14B_image_720p", or "14B_image_480p"
      - torch_dtype_str: either "torch.float8_e4m3fn" or "torch.bfloat16"
      - num_persistent: VRAM related parameter (can be an integer or None)
    """
    app.logger.info(f"Loading model: {model_choice} with torch dtype: {torch_dtype_str} and num_persistent_param_in_dit: {num_persistent}")
    device = "cuda"
    torch_dtype = torch.float8_e4m3fn if torch_dtype_str == "torch.float8_e4m3fn" else torch.bfloat16

    from diffsynth import ModelManager, WanVideoPipeline
    model_manager = ModelManager(device="cpu")
    if model_choice == "1.3B":
        model_manager.load_models(
            [
                "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    elif model_choice == "14B_text":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors",
                    "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors"
                ],
                "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    elif model_choice == "14B_image_720p":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    elif model_choice == "14B_image_480p":
        model_manager.load_models(
            [
                [
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
                    "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
                ],
                "models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
            ],
            torch_dtype=torch_dtype,
        )
        pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device=device)
    else:
        raise ValueError("Invalid model choice")
    
    if str(num_persistent).strip().lower() == "none":
        num_persistent_val = None
    else:
        try:
            num_persistent_val = int(num_persistent)
        except Exception as e:
            app.logger.warning("Could not parse num_persistent_param_in_dit value, defaulting to 6000000000")
            num_persistent_val = 6000000000
    
    pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent_val)
    app.logger.info("Model loaded successfully")
    return pipe

def extend_prompt(prompt, tar_lang="en", input_image=None):
    """Enhances the prompt using the prompt expander model"""
    global prompt_expander
    
    if prompt_expander is None:
        # Use local Qwen prompt expander by default
        app.logger.info("Initializing Qwen prompt expander")
        prompt_expander = QwenPromptExpander(is_vl=input_image is not None, device=0)
    
    prompt_output = prompt_expander(prompt, tar_lang=tar_lang.lower(), image=input_image)
    result = prompt if not prompt_output.status else prompt_output.prompt
    return result

@app.route('/')
def index():
    return jsonify({
        "name": "Wan2.1 API Service",
        "description": "API for generating videos using Wan2.1 models",
        "endpoints": [
            "/text-to-video",
            "/image-to-video",
            "/video-to-video", 
            "/health"
        ]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/text-to-video', methods=['POST'])
def text_to_video_endpoint():
    global loaded_pipeline, loaded_pipeline_config
    
    # Parse request data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.json
    
    # Extract parameters with defaults
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    negative_prompt = data.get('negative_prompt', '')
    model_size = data.get('model_size', '1.3B')  # '1.3B' or '14B'
    num_frames = int(data.get('num_frames', 81))
    inference_steps = int(data.get('inference_steps', 50))
    seed = int(data.get('seed', random.randint(0, 2**32 - 1)))
    cfg_scale = float(data.get('cfg_scale', 5.0))
    sigma_shift = float(data.get('sigma_shift', 5.0))
    aspect_ratio = data.get('aspect_ratio', '16:9')
    width_height = data.get('width_height', None)  # Optional manual override
    vram_setting = data.get('vram_setting', '12000000000')
    torch_dtype = data.get('torch_dtype', 'torch.bfloat16')
    tiled = data.get('tiled', True)
    enhance_prompt = data.get('enhance_prompt', False)
    enhance_language = data.get('enhance_language', 'en')
    
    # Determine model choice
    if model_size == '1.3B':
        model_choice = "1.3B"
        aspect_ratio_dict = ASPECT_RATIOS_1_3b
    else:
        model_choice = "14B_text"
        aspect_ratio_dict = ASPECT_RATIOS_14b
    
    # Determine width and height
    if width_height:
        # Manual override
        try:
            width, height = map(int, width_height.split('*'))
        except:
            return jsonify({"error": "Invalid width_height format. Use '720*1280' format"}), 400
    else:
        # Use aspect ratio
        if aspect_ratio not in aspect_ratio_dict:
            return jsonify({"error": f"Invalid aspect ratio. Choose from: {list(aspect_ratio_dict.keys())}"}), 400
        width, height = aspect_ratio_dict[aspect_ratio]
    
    # Log the request
    app.logger.info(f"T2V request: prompt='{prompt}', size={width}x{height}, frames={num_frames}")
    
    # Enhance prompt if requested
    if enhance_prompt:
        app.logger.info(f"Enhancing prompt: '{prompt}' to {enhance_language}")
        prompt = extend_prompt(prompt, tar_lang=enhance_language)
        app.logger.info(f"Enhanced prompt: '{prompt}'")
    
    # Load or update model if needed
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_setting,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        app.logger.info(f"Loading model with config: {current_config}")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_setting)
        loaded_pipeline_config = current_config
    
    # Set generation parameters
    generation_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": inference_steps,
        "seed": seed,
        "tiled": tiled,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
    }
    
    try:
        # Start generation
        start_time = time.time()
        app.logger.info(f"Starting T2V generation with params: {generation_args}")
        
        video_data = loaded_pipeline(**generation_args)
        
        # Create unique output filename
        output_path = f"outputs/t2v_{uuid.uuid4().hex}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        # Save video with parameters from the config
        from diffsynth import save_video
        save_video(video_data, output_path, fps=16, quality=5)
        
        app.logger.info(f"Saved video to {output_path}, generation took {time.time() - start_time:.2f}s")
        
        # Optionally unload model to save VRAM
        if data.get('unload_after_generation', False):
            loaded_pipeline = None
            loaded_pipeline_config = {}
            app.logger.info("Unloaded model after generation")
        
        # Return the video file
        return send_file(output_path, mimetype='video/mp4')
        
    except Exception as e:
        app.logger.error(f"Error in generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/image-to-video', methods=['POST'])
def image_to_video_endpoint():
    global loaded_pipeline, loaded_pipeline_config
    
    # Handle form data with image file
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "Image file is required"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image file"}), 400
    
    try:
        # Save uploaded image
        image = Image.open(image_file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
    
    # Extract parameters with defaults
    negative_prompt = request.form.get('negative_prompt', '')
    model_quality = request.form.get('model_quality', '480p')  # '480p' or '720p'
    num_frames = int(request.form.get('num_frames', 81))
    inference_steps = int(request.form.get('inference_steps', 40))  # Default is lower for I2V
    seed = int(request.form.get('seed', random.randint(0, 2**32 - 1)))
    cfg_scale = float(request.form.get('cfg_scale', 5.0))
    sigma_shift = float(request.form.get('sigma_shift', 3.0))  # Default is lower for I2V
    aspect_ratio = request.form.get('aspect_ratio', '16:9')
    width_height = request.form.get('width_height', None)  # Optional manual override
    vram_setting = request.form.get('vram_setting', '12000000000')
    torch_dtype = request.form.get('torch_dtype', 'torch.bfloat16')
    tiled = request.form.get('tiled', 'true').lower() == 'true'
    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
    enhance_prompt = request.form.get('enhance_prompt', 'false').lower() == 'true'
    enhance_language = request.form.get('enhance_language', 'en')
    
    # Determine model choice based on quality setting
    if model_quality == '720p':
        model_choice = "14B_image_720p"
        aspect_ratio_dict = ASPECT_RATIOS_14b
    else:
        model_choice = "14B_image_480p"
        aspect_ratio_dict = ASPECT_RATIOS_1_3b
    
    # Determine width and height
    if width_height:
        # Manual override
        try:
            width, height = map(int, width_height.split('*'))
        except:
            return jsonify({"error": "Invalid width_height format. Use '720*1280' format"}), 400
    else:
        # Use aspect ratio
        if aspect_ratio not in aspect_ratio_dict:
            return jsonify({"error": f"Invalid aspect ratio. Choose from: {list(aspect_ratio_dict.keys())}"}), 400
        width, height = aspect_ratio_dict[aspect_ratio]
    
    # Log the request
    app.logger.info(f"I2V request: prompt='{prompt}', size={width}x{height}, frames={num_frames}")
    
    # Process the image if auto-crop is enabled
    if auto_crop:
        app.logger.info(f"Auto-cropping image to {width}x{height}")
        processed_image = auto_crop_image(image, width, height)
    else:
        processed_image = image
    
    # Enhance prompt if requested
    if enhance_prompt:
        app.logger.info(f"Enhancing prompt with image: '{prompt}' to {enhance_language}")
        prompt = extend_prompt(prompt, tar_lang=enhance_language, input_image=processed_image)
        app.logger.info(f"Enhanced prompt: '{prompt}'")
    
    # Load or update model if needed
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_setting,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        app.logger.info(f"Loading model with config: {current_config}")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_setting)
        loaded_pipeline_config = current_config
    
    # Set generation parameters
    generation_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": inference_steps,
        "seed": seed,
        "tiled": tiled,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "input_image": processed_image,
    }
    
    try:
        # Start generation
        start_time = time.time()
        app.logger.info(f"Starting I2V generation with params (excluding image): {generation_args}")
        
        video_data = loaded_pipeline(**generation_args)
        
        # Create unique output filename
        output_path = f"outputs/i2v_{uuid.uuid4().hex}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        # Save video with parameters from the config
        from diffsynth import save_video
        save_video(video_data, output_path, fps=16, quality=5)
        
        app.logger.info(f"Saved video to {output_path}, generation took {time.time() - start_time:.2f}s")
        
        # Return the video file
        return send_file(output_path, mimetype='video/mp4')
        
    except Exception as e:
        app.logger.error(f"Error in generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/video-to-video', methods=['POST'])
def video_to_video_endpoint():
    global loaded_pipeline, loaded_pipeline_config
    
    # Handle form data with video file
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    # Check if video was uploaded
    if 'video' not in request.files:
        return jsonify({"error": "Video file is required"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video file"}), 400
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        video_path = tmp.name
        video_file.save(video_path)
    
    # Extract parameters with defaults
    negative_prompt = request.form.get('negative_prompt', '')
    denoising_strength = float(request.form.get('denoising_strength', 0.7))
    num_frames = int(request.form.get('num_frames', 81))
    inference_steps = int(request.form.get('inference_steps', 50))
    seed = int(request.form.get('seed', random.randint(0, 2**32 - 1)))
    cfg_scale = float(request.form.get('cfg_scale', 5.0))
    sigma_shift = float(request.form.get('sigma_shift', 5.0))
    aspect_ratio = request.form.get('aspect_ratio', '16:9')
    width_height = request.form.get('width_height', None)  # Optional manual override
    vram_setting = request.form.get('vram_setting', '12000000000')
    torch_dtype = request.form.get('torch_dtype', 'torch.bfloat16')
    tiled = request.form.get('tiled', 'true').lower() == 'true'
    auto_crop = request.form.get('auto_crop', 'true').lower() == 'true'
    enhance_prompt = request.form.get('enhance_prompt', 'false').lower() == 'true'
    enhance_language = request.form.get('enhance_language', 'en')
    
    # For V2V we only support 1.3B model
    model_choice = "1.3B"
    aspect_ratio_dict = ASPECT_RATIOS_1_3b
    
    # Determine width and height
    if width_height:
        # Manual override
        try:
            width, height = map(int, width_height.split('*'))
        except:
            os.unlink(video_path)  # Clean up temp file
            return jsonify({"error": "Invalid width_height format. Use '720*1280' format"}), 400
    else:
        # Use aspect ratio
        if aspect_ratio not in aspect_ratio_dict:
            os.unlink(video_path)  # Clean up temp file
            return jsonify({"error": f"Invalid aspect ratio. Choose from: {list(aspect_ratio_dict.keys())}"}), 400
        width, height = aspect_ratio_dict[aspect_ratio]
    
    # Log the request
    app.logger.info(f"V2V request: prompt='{prompt}', size={width}x{height}, frames={num_frames}, denoising={denoising_strength}")
    
    # Process video if auto-crop is enabled
    if auto_crop:
        app.logger.info(f"Auto-cropping video to {width}x{height}")
        from diffsynth import VideoData
        # Use the auto_crop_video function to process the video
        processed_video_path = auto_crop_video(video_path, width, height, num_frames, desired_fps=16)
        video_obj = VideoData(processed_video_path, height=height, width=width)
    else:
        from diffsynth import VideoData
        video_obj = VideoData(video_path, height=height, width=width)
    
    # Enhance prompt if requested
    if enhance_prompt:
        app.logger.info(f"Enhancing prompt: '{prompt}' to {enhance_language}")
        prompt = extend_prompt(prompt, tar_lang=enhance_language)
        app.logger.info(f"Enhanced prompt: '{prompt}'")
    
    # Load or update model if needed
    current_config = {
        "model_choice": model_choice,
        "torch_dtype": torch_dtype,
        "num_persistent": vram_setting,
    }
    if loaded_pipeline is None or loaded_pipeline_config != current_config:
        app.logger.info(f"Loading model with config: {current_config}")
        loaded_pipeline = load_wan_pipeline(model_choice, torch_dtype, vram_setting)
        loaded_pipeline_config = current_config
    
    # Set generation parameters
    generation_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": inference_steps,
        "seed": seed,
        "tiled": tiled,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "sigma_shift": sigma_shift,
        "input_video": video_obj,
        "denoising_strength": denoising_strength,
    }
    
    try:
        # Start generation
        start_time = time.time()
        app.logger.info(f"Starting V2V generation with params (excluding video): {generation_args}")
        
        video_data = loaded_pipeline(**generation_args)
        
        # Create unique output filename
        output_path = f"outputs/v2v_{uuid.uuid4().hex}.mp4"
        os.makedirs("outputs", exist_ok=True)
        
        # Save video with parameters from the config
        from diffsynth import save_video
        save_video(video_data, output_path, fps=16, quality=5)
        
        app.logger.info(f"Saved video to {output_path}, generation took {time.time() - start_time:.2f}s")
        
        # Clean up temp files
        os.unlink(video_path)
        if auto_crop and os.path.exists(processed_video_path):
            os.unlink(processed_video_path)
        
        # Return the video file
        return send_file(output_path, mimetype='video/mp4')
        
    except Exception as e:
        # Clean up temp files
        os.unlink(video_path)
        if auto_crop and os.path.exists(processed_video_path):
            os.unlink(processed_video_path)
            
        app.logger.error(f"Error in generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

def auto_crop_video(video_path, target_width, target_height, desired_frame_count, desired_fps=16):
    """
    Reads a video from disk, and for each frame:
      - Downscales if the frame is larger than target dimensions.
      - Performs center crop to get exactly the target resolution.
      - Processes only a number of frames equal to desired_frame_count.
    Saves to a new file (with a _cropped suffix) and returns its path.
    
    The output video FPS is set to desired_fps.
    The output video duration will be desired_frame_count / desired_fps seconds.
    """
    app.logger.info(f"Starting video processing for file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        app.logger.error(f"Failed to open video: {video_path}")
        return video_path
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    app.logger.info(f"Original video FPS: {orig_fps}")
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.logger.info(f"Original video resolution: {orig_width}x{orig_height}")
    scale = min(1.0, target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    app.logger.info(f"Scaling factor: {scale}. New intermediate resolution: {new_width}x{new_height}")
    
    base, ext = os.path.splitext(video_path)
    out_path = base + "_cropped.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, desired_fps, (target_width, target_height))
    
    frame_count = 0
    while frame_count < desired_frame_count:
        ret, frame = cap.read()
        if not ret:
            app.logger.info("No more frames available from the video.")
            break
        # Downscale if needed.
        if scale < 1.0:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        if w > target_width or h > target_height:
            left = (w - target_width) // 2
            top = (h - target_height) // 2
            frame = frame[top:top+target_height, left:left+target_width]
            app.logger.debug(f"Cropped frame {frame_count+1}: left={left}, top={top}")
        else:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            app.logger.debug(f"Resized frame {frame_count+1} to target resolution.")
        out.write(frame)
        frame_count += 1
    cap.release()
    out.release()
    app.logger.info(f"Finished processing video. Total frames written: {frame_count}")
    app.logger.info(f"Set output FPS to {desired_fps}. Final video duration: {frame_count/desired_fps:.2f} seconds.")
    app.logger.info(f"Output video saved to: {out_path}")
    return out_path

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)