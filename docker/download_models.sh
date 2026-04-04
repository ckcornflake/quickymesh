#!/usr/bin/env bash
# download_models.sh — Download all models required by the quickymesh Docker containers.
#
# All workflows (FLUX, ControlNet restyle, and Trellis) run inside the
# comfyui-trellis container, so all models go into TRELLIS_MODELS_DIR.
#
# Run this once before starting the containers for the first time.
# Models are large (~40 GB total) and are NOT included in the repo.
#
# Usage:
#   bash docker/download_models.sh              # download everything
#   bash docker/download_models.sh flux         # FLUX.1-dev FP8 only
#   bash docker/download_models.sh restyle      # ControlNet restyle models only
#   bash docker/download_models.sh trellis      # Trellis2 models only
#
# NOTE: Trellis model weights are downloaded automatically by the container
# on first run via HuggingFace Hub. You only need to run the trellis target
# if you want to pre-populate a local folder (e.g. to reuse existing downloads
# or avoid downloading inside the container).
#
# FLUX.1-dev requires a HuggingFace account and license acceptance at:
#   https://huggingface.co/black-forest-labs/FLUX.1-dev
# Log in first:  huggingface-cli login
#
# Model destination controlled by docker/.env:
#   TRELLIS_MODELS_DIR — mounted into the comfyui-trellis container at /app/models
#   (defaults to docker/models/trellis if not set)
#
# Requirements: pip install huggingface_hub

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load docker/.env if present so TRELLIS_MODELS_DIR can be overridden
[[ -f "$SCRIPT_DIR/.env" ]] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Default to repo-relative path if env var not set
TRELLIS_MODELS_DIR="${TRELLIS_MODELS_DIR:-$SCRIPT_DIR/models/trellis}"

# ---- helpers ---------------------------------------------------------------

check_hf_cli() {
  if ! command -v huggingface-cli &>/dev/null; then
    echo "Installing huggingface_hub ..."
    pip install -q huggingface_hub
  fi
}

hf_download() {
  local repo="$1"
  local local_dir="$2"
  local filename="${3:-}"   # optional: single file within the repo

  echo "  Downloading $repo${filename:+ ($filename)} ..."
  if [[ -n "$filename" ]]; then
    huggingface-cli download "$repo" "$filename" --local-dir "$local_dir"
  else
    huggingface-cli download "$repo" --local-dir "$local_dir"
  fi
}

# ---- FLUX.1-dev FP8 (~8 GB) ------------------------------------------------
# Used by: comfyui-trellis container, flux_dev_fp8.json workflow

download_flux() {
  echo ""
  echo "==> FLUX.1-dev FP8 (~8 GB) → $TRELLIS_MODELS_DIR/checkpoints"

  local ckpt_dir="$TRELLIS_MODELS_DIR/checkpoints"
  mkdir -p "$ckpt_dir"

  if [[ -f "$ckpt_dir/flux1-dev-fp8.safetensors" ]]; then
    echo "  flux1-dev-fp8.safetensors already present, skipping."
  else
    hf_download \
      "Comfy-Org/flux1-dev" \
      "$ckpt_dir" \
      "flux1-dev-fp8.safetensors"
  fi

  echo "  FLUX done."
}

# ---- ControlNet restyle models (~8 GB) -------------------------------------
# Used by: comfyui-trellis container, controlnet_restyle.json workflow

download_restyle() {
  echo ""
  echo "==> ControlNet restyle models (~8 GB) → $TRELLIS_MODELS_DIR"

  local ckpt_dir="$TRELLIS_MODELS_DIR/checkpoints"
  local cn_dir="$TRELLIS_MODELS_DIR/controlnet"
  mkdir -p "$ckpt_dir" "$cn_dir"

  if [[ -f "$ckpt_dir/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" ]]; then
    echo "  Juggernaut-XL v9 already present, skipping."
  else
    hf_download \
      "RunDiffusion/Juggernaut-XL-v9" \
      "$ckpt_dir" \
      "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
  fi

  if [[ -f "$cn_dir/controlnet-canny-sdxl-1.0.safetensors" ]]; then
    echo "  controlnet-canny-sdxl-1.0 already present, skipping."
  else
    hf_download \
      "xinsir/controlnet-canny-sdxl-1.0" \
      "$cn_dir" \
      "diffusion_pytorch_model.safetensors"
    mv "$cn_dir/diffusion_pytorch_model.safetensors" \
       "$cn_dir/controlnet-canny-sdxl-1.0.safetensors"
  fi

  echo "  Restyle done."
}

# ---- Trellis models (~25 GB) -----------------------------------------------
# Used by: comfyui-trellis container, trellis_generate.json + trellis_texture.json
#
# NOTE: The container downloads these automatically on first run via HuggingFace
# Hub into a persistent Docker volume. Only run this target if you want to
# pre-populate TRELLIS_MODELS_DIR from an existing local copy.

download_trellis() {
  echo ""
  echo "==> Trellis models (~25 GB) → $TRELLIS_MODELS_DIR"
  echo "  (Skip this if you prefer the container to download automatically on first run)"

  # TRELLIS.2-4B — main 3D generation model
  local trellis_dir="$TRELLIS_MODELS_DIR/microsoft/TRELLIS.2-4B"
  if [[ -d "$trellis_dir" && -n "$(ls -A "$trellis_dir" 2>/dev/null)" ]]; then
    echo "  TRELLIS.2-4B already present, skipping."
  else
    mkdir -p "$trellis_dir"
    hf_download "microsoft/TRELLIS.2-4B" "$trellis_dir"
  fi

  # DINOv2 ViT-L14 — vision encoder used by Trellis image conditioning
  local dino_dir="$TRELLIS_MODELS_DIR/facebook/dinov2-vitl14"
  if [[ -d "$dino_dir" && -n "$(ls -A "$dino_dir" 2>/dev/null)" ]]; then
    echo "  DINOv2 ViT-L14 already present, skipping."
  else
    mkdir -p "$dino_dir"
    hf_download "facebook/dinov2-vitl14" "$dino_dir"
  fi

  echo "  Trellis done."
}

# ---- main ------------------------------------------------------------------

check_hf_cli

TARGET="${1:-all}"

case "$TARGET" in
  flux)    download_flux ;;
  restyle) download_restyle ;;
  trellis) download_trellis ;;
  all)     download_flux; download_restyle; download_trellis ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash docker/download_models.sh [flux|restyle|trellis|all]"
    exit 1
    ;;
esac

echo ""
echo "==> All done."
echo ""
echo "Next steps:"
echo "  1. Copy docker/.env.example to docker/.env and set TRELLIS_MODELS_DIR"
echo "     if you downloaded to a custom location."
echo "  2. cd docker && docker compose up comfyui-trellis"
