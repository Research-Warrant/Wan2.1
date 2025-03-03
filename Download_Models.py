import sys
import subprocess
import os
import platform
import shutil
from huggingface_hub import snapshot_download
import hf_transfer

def download_model(model_name):
    if model_name == "1":
        print("Downloading Wan2.1-T2V-1.3B...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-T2V-1.3B"
        )
        print("Download complete!")
    elif model_name == "2":
        print("Downloading Wan2.1-I2V-14B-720P...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-I2V-14B-720P"
        )
        print("Download complete!")
    elif model_name == "3":
        print("Downloading Wan2.1-I2V-14B-480P...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-I2V-14B-480P"
        )
        print("Download complete!")
    elif model_name == "4":
        print("Downloading Wan2.1-T2V-14B...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-T2V-14B"
        )
        print("Download complete!")
    elif model_name == "5":
        print("Downloading Qwen2.5-14B-Instruct...")
        snapshot_download(
            repo_id="Qwen/Qwen2.5-14B-Instruct"
        )
        print("Download complete!")
    elif model_name == "6":
        print("Downloading all models...")
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-T2V-1.3B"
        )
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-720P",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-I2V-14B-720P"
        )
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-I2V-14B-480P"
        )
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            local_dir="Wan2.1/models/Wan-AI/Wan2.1-T2V-14B"
        )
        snapshot_download(
            repo_id="Qwen/Qwen2.5-14B-Instruct"
        )
        print("All downloads complete!")
    else:
        print("Invalid selection. No models downloaded.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_model(sys.argv[1])
