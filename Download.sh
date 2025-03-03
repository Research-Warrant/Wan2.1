pip install huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$HF_TOKEN  # Use token from environment if provided

echo "========================================================================"
echo "                       Wan2.1 Model Downloader                          "
echo "========================================================================"
echo
echo "Please select which model(s) to download:"
echo
echo "1. Wan2.1-T2V-1.3B : 1.3B Wan 2.1 Text-to-Video and Video-to-Video model"
echo "   - As low as 3.5 GB VRAM"
echo "   - 480x832px or 832x480px resolution"
echo
echo "2. Wan2.1-I2V-14B-720P : 14B Wan 2.1 Image to Video model (720p)"
echo "   - 720x1280px or 1280x720px resolution"
echo
echo "3. Wan2.1-I2V-14B-480P : 14B Wan 2.1 Image to Video model (480p)"
echo "   - 480x832px or 832x480px resolution"
echo
echo "4. Wan2.1-T2V-14B : 14B Wan 2.1 Text to Video model"
echo "   - 720x1280px or 1280x720px resolution"
echo
echo "5. Qwen2.5-14B-Instruct : For Prompt Enhance (downloads to default Hugging Face cache)"
echo
echo "6. Download all models"
echo
echo "========================================================================"
echo

if [ -t 0 ]; then  # Check if running in interactive terminal
    read -p "Enter your choice (1-6): " choice
else
    # If not in interactive terminal, use default choice
    choice=${1:-1}
    echo "Non-interactive mode detected. Using choice: $choice"
fi

echo "Starting download for choice $choice..."
python3 /app/Download_Models.py $choice

# If download is successful, run Flask app
echo "Download complete. Restart the container to use the new models."

echo "Press Enter to exit..."
read
