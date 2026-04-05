# quickymesh runtime image
#
# Contains:
#   - ComfyUI + Trellis2 custom nodes  (port 8190)
#   - quickymesh FastAPI server         (port 8000)
#   - Blender 4.3.2                    (headless, for screenshots + mesh cleanup)
#
# Build context: repo root  (set in docker-compose.yml)
# Dockerfile location: docker/comfyui-trellis/Dockerfile
#
# Base: CUDA 12.8 runtime on Ubuntu 24.04 (Python 3.12 — required for cp312 wheels)
# Wheel sources: https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels/Linux/Torch291

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Ubuntu 24.04 marks system Python as externally managed — this flag lets pip install anyway.
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PATH="/home/comfy/.local/bin:${PATH}"

# System dependencies
# build-essential + cmake needed by some pip packages (open3d, pymeshlab) at install time
# libx11-6 + libxi6 + libxxf86vm1 etc. needed for Blender headless rendering
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
    xz-utils \
    ffmpeg \
    ca-certificates \
    libgl1 \
    libopengl0 \
    libglib2.0-0 \
    libegl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Blender 4.3.2 (headless) -----------------------------------------------
# Installed as root into /opt/blender before switching to the comfy user.
RUN wget -q -O /tmp/blender.tar.xz \
        "https://download.blender.org/release/Blender4.3/blender-4.3.2-linux-x64.tar.xz" && \
    mkdir -p /opt/blender && \
    tar -xf /tmp/blender.tar.xz -C /opt/blender && \
    rm /tmp/blender.tar.xz

# Run as non-root (avoids permission headaches with mounted volumes)
RUN useradd -m -s /bin/bash comfy
USER comfy
WORKDIR /app

# ---- PyTorch ---
RUN pip uninstall torch torchvision torchaudio
RUN pip install --upgrade pip && \
    pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# ---- Core Trellis2 Python dependencies ---
RUN pip install \
    triton \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    trimesh \
    "transformers==4.57.1" \
    zstandard \
    kornia \
    timm \
    scipy \
    plotly \
    "rembg[gpu]" \
    onnxruntime-gpu \
    requests \
    plyfile

# ---- ComfyUI-Trellis2 specific deps (heavier, separate layer for caching) ---
RUN pip install \
    open3d \
    pymeshlab \
    meshlib

# ---- quickymesh API server dependencies ---
# Pillow, requests, trimesh, numpy already installed above.
RUN pip install \
    "pydantic>=2.7.0" \
    "python-dotenv>=1.0.0" \
    "PyYAML>=6.0" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "sse-starlette>=2.1.0" \
    "python-multipart>=0.0.9" \
    "httpx>=0.27.0" \
    "google-genai>=1.10.0" \
    "python-json-logger>=3.2.0"

# ---- ComfyUI ---
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt

# ---- ComfyUI-Trellis2 custom node ---
# git-lfs is installed above so the wheels/ directory (cumesh, o_voxel etc.) is fetched too.
RUN git clone https://github.com/visualbruno/ComfyUI-Trellis2.git \
        custom_nodes/ComfyUI-Trellis2 && \
    git -C custom_nodes/ComfyUI-Trellis2 checkout e7c959b

# ---- Trellis2 compiled extension wheels (prebuilt Linux x86_64 cp312 for Torch 2.9.1) ---
# These come directly from the cloned ComfyUI-Trellis2 repo (visualbruno's Torch291 Linux set).
# All 6 wheels are built together for Torch 2.9.1+cu128 — no ABI mismatches.
# Note: flash_attn is NOT installed separately; PyTorch 2.9+ includes native SDPA which
# Trellis2 uses via the 'sdpa' backend (added by visualbruno).
ENV VB_WHEELS=custom_nodes/ComfyUI-Trellis2/wheels/Linux/Torch291
RUN pip install --no-cache-dir ${VB_WHEELS}/cumesh-1.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${VB_WHEELS}/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${VB_WHEELS}/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${VB_WHEELS}/nvdiffrec_render-0.0.0-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir --no-deps ${VB_WHEELS}/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl
RUN pip install --no-cache-dir ${VB_WHEELS}/custom_rasterizer-0.1-cp312-cp312-linux_x86_64.whl

# ---- Install ControlNet Auxiliary Preprocessors ---
RUN git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git \
    /app/custom_nodes/comfyui_controlnet_aux

# Install requirements
RUN pip install --no-cache-dir -r /app/custom_nodes/comfyui_controlnet_aux/requirements.txt

# Ensure ONNX Runtime GPU for your GPU
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# flash_attn — required by Trellis2 attention backend.
# Pre-built wheel for torch 2.9 + cu12 from the official flash-attention GitHub releases.
RUN pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# From https://www.youtube.com/watch?v=OkK-BfLiS2Q
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu128

# Re-pin torchvision — requirements.txt installs above can override it with a CPU-only build
RUN pip install --force-reinstall torchvision==0.24.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ---- Pre-download rembg u2net model (used by Trellis2PreProcessImage remove_background) ---
# Without this it downloads from GitHub at runtime and times out.
RUN mkdir -p /home/comfy/.u2net && \
    wget -q -O /home/comfy/.u2net/u2net.onnx \
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

# ---- Pre-download BiRefNet (used by some Trellis2 nodes) ---
RUN python3 -c "\
from transformers import AutoModelForImageSegmentation; \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)"

# Create placeholder dirs so ComfyUI starts cleanly even without volume mounts
RUN mkdir -p models output input

# ---- quickymesh source code --------------------------------------------------
# Build context is the repo root — copy source and config files.
COPY --chown=comfy:comfy src /quickymesh/src
COPY --chown=comfy:comfy api_server.py /quickymesh/api_server.py
COPY --chown=comfy:comfy qm_cli.py /quickymesh/qm_cli.py
COPY --chown=comfy:comfy defaults.yaml /quickymesh/defaults.yaml
COPY --chown=comfy:comfy comfyui_workflows /quickymesh/comfyui_workflows
COPY --chown=comfy:comfy blender_scripts /quickymesh/blender_scripts

# Persistent data directories (overridden by volume mounts at runtime)
RUN mkdir -p /quickymesh/pipeline_root /quickymesh/logs

# ---- Startup script ----------------------------------------------------------
COPY --chown=comfy:comfy docker/comfyui-trellis/startup.sh /startup.sh
RUN chmod +x /startup.sh

# ---- Runtime environment variables ------------------------------------------
# These are defaults; all can be overridden via docker-compose environment:.
ENV BLENDER_PATH=/opt/blender/blender-4.3.2-linux-x64/blender
ENV COMFYUI_URL=http://localhost:8190
ENV COMFYUI_OUTPUT_DIR=/app/output
ENV OUTPUT_ROOT=/quickymesh/pipeline_root

EXPOSE 8190 8000

CMD ["/startup.sh"]
