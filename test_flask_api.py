#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Test script for Wan2.1 Flask API
This script demonstrates how to use the Flask API for Text-to-Video,
Image-to-Video, and Video-to-Video generation.
"""

import argparse
import json
import requests
import os
from PIL import Image

def test_health():
    """Test the health endpoint"""
    response = requests.get('http://localhost:5000/health')
    print(f"Health check status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("-" * 50)

def test_text_to_video(prompt, output_path):
    """Test the text-to-video endpoint"""
    print(f"Generating Text-to-Video with prompt: '{prompt}'")
    
    data = {
        "prompt": prompt,
        "negative_prompt": "blurry, distorted, bad quality",
        "model_size": "1.3B",  # Use 1.3B as it requires less VRAM
        "num_frames": 81,
        "aspect_ratio": "16:9",
        "inference_steps": 50,
        "seed": 42
    }
    
    response = requests.post(
        'http://localhost:5000/text-to-video', 
        json=data,
        stream=True
    )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video saved to {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_image_to_video(prompt, image_path, output_path):
    """Test the image-to-video endpoint"""
    print(f"Generating Image-to-Video with prompt: '{prompt}' and image: {image_path}")
    
    with open(image_path, 'rb') as img:
        files = {'image': (os.path.basename(image_path), img, 'image/jpeg')}
        data = {
            "prompt": prompt,
            "model_quality": "480p",  # Use 480p as it requires less VRAM
            "num_frames": 81,
            "aspect_ratio": "16:9",
            "auto_crop": "true"
        }
        
        response = requests.post(
            'http://localhost:5000/image-to-video',
            files=files,
            data=data,
            stream=True
        )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video saved to {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print("-" * 50)

def test_video_to_video(prompt, video_path, output_path):
    """Test the video-to-video endpoint"""
    print(f"Generating Video-to-Video with prompt: '{prompt}' and video: {video_path}")
    
    with open(video_path, 'rb') as vid:
        files = {'video': (os.path.basename(video_path), vid, 'video/mp4')}
        data = {
            "prompt": prompt,
            "denoising_strength": "0.7",
            "num_frames": 81,
            "aspect_ratio": "16:9",
            "auto_crop": "true"
        }
        
        response = requests.post(
            'http://localhost:5000/video-to-video',
            files=files,
            data=data,
            stream=True
        )
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video saved to {output_path}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    print("-" * 50)

def parse_args():
    parser = argparse.ArgumentParser(description='Test the Wan2.1 Flask API')
    parser.add_argument('--mode', type=str, default='health', 
                        choices=['health', 't2v', 'i2v', 'v2v', 'all'],
                        help='Test mode: health, t2v (text-to-video), i2v (image-to-video), v2v (video-to-video), or all')
    parser.add_argument('--prompt', type=str, default='A cute cat playing with a ball of yarn',
                        help='Prompt for video generation')
    parser.add_argument('--image', type=str, default='examples/i2v_input.JPG',
                        help='Image path for image-to-video test')
    parser.add_argument('--video', type=str, default=None,
                        help='Video path for video-to-video test')
    parser.add_argument('--output-dir', type=str, default='test_outputs',
                        help='Directory to save output videos')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'health' or args.mode == 'all':
        test_health()
    
    if args.mode == 't2v' or args.mode == 'all':
        t2v_output = os.path.join(args.output_dir, 't2v_test_output.mp4')
        test_text_to_video(args.prompt, t2v_output)
    
    if args.mode == 'i2v' or args.mode == 'all':
        if not os.path.exists(args.image):
            print(f"Warning: Image file {args.image} not found, skipping i2v test")
        else:
            i2v_output = os.path.join(args.output_dir, 'i2v_test_output.mp4')
            test_image_to_video(args.prompt, args.image, i2v_output)
    
    if args.mode == 'v2v' or args.mode == 'all':
        if args.video is None or not os.path.exists(args.video):
            print(f"Warning: Video file not provided or not found, skipping v2v test")
        else:
            v2v_output = os.path.join(args.output_dir, 'v2v_test_output.mp4')
            test_video_to_video(args.prompt, args.video, v2v_output)

if __name__ == '__main__':
    main()