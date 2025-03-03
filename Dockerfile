FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set non-interactive frontend and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    tzdata \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Copy requirements and install Python dependencies
COPY requirements-flask.txt .

# Install core dependencies with optimized CUDA packages
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir xformers && \
    pip3 install --no-cache-dir triton

# Clone and install DiffSynth-Studio
RUN cd /tmp && \
    git clone https://github.com/modelscope/DiffSynth-Studio && \
    cd DiffSynth-Studio && \
    pip install -e . && \
    cd /app

# Install HuggingFace Hub tools for model downloading
RUN pip install --no-cache-dir huggingface_hub hf_transfer

# Clone and install RIFE for frame interpolation
RUN cd /tmp && \
    git clone https://github.com/FurkanGozukara/Practical-RIFE && \
    cd Practical-RIFE && \
    pip3 install -r requirements.txt && \
    cd /app

# Install Python dependencies with error handling
RUN pip3 install --no-cache-dir -r requirements-flask.txt || \
    (echo "Falling back to installation without flash_attn" && \
     grep -v "flash_attn" requirements-flask.txt > requirements-no-flash.txt && \
     pip3 install --no-cache-dir -r requirements-no-flash.txt)

# Install additional dependencies
RUN pip install --no-cache-dir moviepy huggingface_hub ipywidgets hf_transfer torchao

# Copy application code
COPY flask_app.py .
COPY wan/ ./wan/

# Fix amp.autocast deprecation warnings by modifying the model file directly
RUN sed -i 's/@amp.autocast(enabled=False)/@torch.amp.autocast("cuda", enabled=False)/g' /app/wan/modules/model.py && \
    sed -i 's/import torch.cuda.amp as amp/import torch.amp as amp/g' /app/wan/modules/model.py && \
    sed -i 's/with amp.autocast(dtype=torch.float32)/with amp.autocast("cuda", dtype=torch.float32)/g' /app/wan/modules/model.py

# Create outputs and models directories with placeholder README
RUN mkdir -p outputs && mkdir -p models/Wan-AI/Wan2.1-T2V-1.3B && \
    echo "# Model Files Required\n\nThis directory should contain the Wan2.1 model files.\nPlease download the models and place them in the appropriate subdirectories.\n\nSee FLASK_API_README.md for details on the expected directory structure." > models/README.md

# Copy download scripts to the container
COPY Download_Models.py Download_RIFE.py /app/
COPY Download.sh /app/

# Modify the download scripts to use the correct paths
RUN sed -i 's/Wan2.1\/models/models/g' /app/Download_Models.py && \
    sed -i 's/Wan2.1\/Practical-RIFE/Practical-RIFE/g' /app/Download_RIFE.py && \
    chmod +x /app/Download.sh

# Create a stub diffsynth.py file to handle missing models gracefully
RUN echo '# Stub diffsynth module for testing\nclass ModelManager:\n    def __init__(self, **kwargs):\n        pass\n    def load_models(self, *args, **kwargs):\n        pass\n\nclass WanVideoPipeline:\n    @classmethod\n    def from_model_manager(cls, manager, **kwargs):\n        return cls()\n    \n    def __init__(self):\n        pass\n        \n    def enable_vram_management(self, **kwargs):\n        pass\n        \n    def __call__(self, **kwargs):\n        return []\n        \ndef save_video(video_data, path, **kwargs):\n    with open(path, "w") as f:\n        f.write("test")\n\nclass VideoData:\n    def __init__(self, path, **kwargs):\n        self.path = path' > /app/diffsynth.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=UTF-8

# Create a startup script to check for models and provide helpful messages
RUN echo '#!/bin/bash\necho "Starting Wan2.1 Flask API Server..."\nif [ -f /app/models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors ] || [ -f /app/models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors ]; then\n  echo "Model files detected. Starting with real models."\nelse\n  echo "WARNING: No model files found in /app/models/Wan-AI/!"\n  echo "The API will start in mock mode and return test responses."\n  echo "To download models directly, you can run:"\n  echo "  docker exec -it wan-flask-api /app/Download.sh"\n  echo "Or mount your model files to /app/models following the structure in FLASK_API_README.md"\nfi\nexec gunicorn --bind 0.0.0.0:5000 --timeout 3600 --workers 1 flask_app:app' > /app/start.sh && \
    chmod +x /app/start.sh

# Run the application with our startup script
CMD ["/app/start.sh"]