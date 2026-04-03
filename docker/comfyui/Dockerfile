# ComfyUI Docker Image
# Base: CUDA 12.8 + cuDNN runtime on Ubuntu 22.04 (matches cu128 PyTorch wheels)
# Container OS: Linux (standard for all Docker containers, even on Windows hosts)

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Clone latest ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

# Install PyTorch with CUDA 12.8 support (cu128 wheels)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# Install ComfyUI requirements
RUN pip install -r requirements.txt

# Volumes — models and output are mounted at runtime (see docker-compose.yml)
# These are created as empty dirs so ComfyUI starts cleanly without mounted volumes too
RUN mkdir -p \
    models/checkpoints \
    models/clip \
    models/controlnet \
    models/loras \
    models/vae \
    models/unet \
    models/text_encoders \
    output \
    input

EXPOSE 8189

# --listen 0.0.0.0 required to accept connections from outside the container
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8189"]
