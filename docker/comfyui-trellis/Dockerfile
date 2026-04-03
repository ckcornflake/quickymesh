# ComfyUI + Trellis2 Docker Image
#
# Base: CUDA 12.8 runtime on Ubuntu 24.04 (Python 3.12 default — required for cp312 wheels)
# All CUDA extensions installed as prebuilt wheels from camenduru/wheels (no compilation).
#
# Wheel sources: https://github.com/camenduru/wheels/releases/tag/trellis2

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Ubuntu 24.04 marks system Python as externally managed — this flag lets pip install anyway.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PATH="/home/comfy/.local/bin:${PATH}"

# System dependencies
# build-essential + cmake needed by some pip packages (open3d, pymeshlab) at install time
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    python3-dev \
    build-essential \
    cmake \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    ca-certificates \
    libgl1 \
    libopengl0 \
    libglib2.0-0 \
    libegl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root (avoids permission headaches with mounted volumes)
RUN useradd -m -s /bin/bash comfy
USER comfy
WORKDIR /app

# --- PyTorch ---
RUN pip install --upgrade pip && \
    pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# --- Trellis2 compiled extension wheels (prebuilt Linux x86_64 cp312 — no compilation) ---
# flash_attn and nvdiffrec_render come from camenduru (built for Torch 2.8+cu128).
# cumesh, flex_gemm, nvdiffrast, o_voxel come from the cloned ComfyUI-Trellis2 repo
# (Torch270 Linux set — proven compatible with Torch 2.8.0 in testing).
ENV WHEELS=https://github.com/camenduru/wheels/releases/download/trellis2
RUN pip install --no-cache-dir ${WHEELS}/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${WHEELS}/nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl

# --- Core Trellis2 Python dependencies ---
RUN pip install \
    triton \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    trimesh \
    transformers \
    zstandard \
    kornia \
    timm \
    scipy \
    plotly \
    "rembg[gpu]" \
    onnxruntime-gpu \
    requests \
    plyfile

# --- ComfyUI-Trellis2 specific deps (heavier, separate layer for caching) ---
RUN pip install \
    open3d \
    pymeshlab \
    meshlib

# --- ComfyUI ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# --- ComfyUI-Trellis2 custom node ---
# git-lfs is installed above so the wheels/ directory (cumesh, o_voxel etc.) is fetched too.
RUN git clone --depth 1 https://github.com/visualbruno/ComfyUI-Trellis2.git \
        custom_nodes/ComfyUI-Trellis2

# --- Install wheels from visualbruno's repo (updated APIs required by latest node code) ---
# Torch270 Linux wheels — ABI-compatible with our Torch 2.8.0.
# cumesh 1.0: adds remove_inner_faces to remesh_narrow_band_dc (Trellis2Remesh)
# o_voxel 0.0.1: adds tiled_flexible_dual_grid_to_mesh (shape decoder)
ENV VB_WHEELS=custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch270
RUN pip install --no-cache-dir ${VB_WHEELS}/cumesh-1.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${VB_WHEELS}/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl
# nvdiffrast: use camenduru wheel (Torch270 version has ABI incompatibility with Torch 2.8.0)
RUN pip install --no-cache-dir ${WHEELS}/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir --no-deps ${VB_WHEELS}/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl

# --- Pre-download rembg u2net model (used by Trellis2PreProcessImage remove_background) ---
# Without this it downloads from GitHub at runtime and times out.
RUN mkdir -p /home/comfy/.u2net && \
    wget -q -O /home/comfy/.u2net/u2net.onnx \
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

# --- Pre-download BiRefNet (used by some Trellis2 nodes) ---
RUN python3 -c "\
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)"

# Create placeholder dirs so ComfyUI starts cleanly even without volume mounts
RUN mkdir -p \
    models/checkpoints \
    models/microsoft \
    models/facebook \
    output \
    input

EXPOSE 8190

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8190"]
