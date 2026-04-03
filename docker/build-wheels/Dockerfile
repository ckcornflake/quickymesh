# Wheel Builder for ComfyUI-Trellis2
#
# Compiles CUDA extension wheels for PyTorch 2.8.0+cu128 / Python 3.12 / Linux x86_64.
# Output wheels are written to /wheels — mount a host directory there to extract them.
#
# Usage:
#   docker build -t trellis2-wheel-builder .
#   docker run --gpus all -v $(pwd)/output:/wheels trellis2-wheel-builder
#
# Wheels produced (copy to docker/wheels/linux/torch280/ in this repo):
#   cumesh-*.whl
#   flex_gemm-*.whl
#   nvdiffrast-*.whl
#   o_voxel-*.whl

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CUDA_HOME=/usr/local/cuda
# Target SM 8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100)
# sm_120 (RTX 5090/Blackwell) excluded — o-voxel rasterizer is incompatible.
# Required during docker build where no GPU is visible to auto-detect.
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    python3-dev \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    libeigen3-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch 2.8.0+cu128 ---
RUN pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# Build tools needed by CUDA extensions
RUN pip install setuptools wheel ninja packaging

WORKDIR /build

# --- FlexGEMM (Triton-based sparse ops — no CUDA compilation needed) ---
RUN git clone --depth 1 https://github.com/visualbruno/FlexGEMM.git && \
    pip wheel FlexGEMM/ --no-build-isolation -w /wheels

# --- CuMesh (CUDA mesh utilities) ---
# cubvh has a nested eigen submodule. Fetch cubvh with --depth 1, then
# symlink system libeigen3-dev in place of its empty eigen submodule.
RUN git clone --depth 1 https://github.com/visualbruno/CuMesh.git && \
    cd CuMesh && \
    git submodule update --init --depth 1 && \
    mkdir -p third_party/cubvh/third_party && \
    rm -rf third_party/cubvh/third_party/eigen && \
    ln -s /usr/include/eigen3 third_party/cubvh/third_party/eigen && \
    cd /build && \
    pip wheel CuMesh/ --no-build-isolation -w /wheels

# --- nvdiffrast (differentiable rasterizer) ---
RUN git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git && \
    pip wheel nvdiffrast/ --no-build-isolation -w /wheels

# --- o_voxel (from visualbruno's TRELLIS.2 fork — includes tiled_flexible_dual_grid_to_mesh) ---
# o-voxel also has a third_party/eigen submodule that shallow clone leaves empty.
RUN git clone --depth 1 https://github.com/visualbruno/TRELLIS.2.git && \
    mkdir -p TRELLIS.2/o-voxel/third_party && \
    rm -rf TRELLIS.2/o-voxel/third_party/eigen && \
    ln -s /usr/include/eigen3 TRELLIS.2/o-voxel/third_party/eigen && \
    pip wheel TRELLIS.2/o-voxel/ --no-build-isolation -w /wheels

# List produced wheels
RUN echo "Built wheels:" && ls -lh /wheels/
