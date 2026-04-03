#!/usr/bin/env bash
# download_models.sh — Download all models required by the Docker containers.
#
# Run this once before starting the containers for the first time.
# Models are saved to docker/models/ which is gitignored (weights only, ~25GB total).
#
# Usage:
#   bash docker/download_models.sh           # download everything
#   bash docker/download_models.sh sdxl      # SDXL models only
#   bash docker/download_models.sh trellis   # Trellis models only
#
# Requirements: pip install huggingface_hub
#   (or: pip install huggingface_hub[cli] for the hf CLI)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present so SDXL_MODELS_DIR / TRELLIS_MODELS_DIR can be overridden
[[ -f "$SCRIPT_DIR/.env" ]] && set -a && source "$SCRIPT_DIR/.env" && set +a

# Default to repo-relative paths if env vars not set
SDXL_MODELS_DIR="${SDXL_MODELS_DIR:-$SCRIPT_DIR/models/sdxl}"
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

# ---- SDXL models (~7 GB) ---------------------------------------------------

download_sdxl() {
  echo ""
  echo "==> SDXL models (~7 GB) → $SDXL_MODELS_DIR"

  local ckpt_dir="$SDXL_MODELS_DIR/checkpoints"

  if [[ -f "$ckpt_dir/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" ]]; then
    echo "  Juggernaut-XL v9 already present, skipping."
  else
    hf_download \
      "RunDiffusion/Juggernaut-XL-v9" \
      "$ckpt_dir" \
      "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
  fi

  echo "  SDXL done."
}

# ---- Trellis models (~17 GB) -----------------------------------------------

download_trellis() {
  echo ""
  echo "==> Trellis models (~17 GB) → $TRELLIS_MODELS_DIR"

  # TRELLIS.2-4B — the main 3D generation model
  local trellis_dir="$TRELLIS_MODELS_DIR/microsoft/TRELLIS.2-4B"
  if [[ -d "$trellis_dir" && -n "$(ls -A "$trellis_dir" 2>/dev/null)" ]]; then
    echo "  TRELLIS.2-4B already present, skipping."
  else
    mkdir -p "$trellis_dir"
    hf_download "microsoft/TRELLIS.2-4B" "$trellis_dir"
  fi

  # DINOv3 ViT-L16 — vision encoder used by Trellis
  local dino_dir="$TRELLIS_MODELS_DIR/facebook/dinov3-vitl16-pretrain-lvd1689m"
  if [[ -d "$dino_dir" && -n "$(ls -A "$dino_dir" 2>/dev/null)" ]]; then
    echo "  DINOv3 already present, skipping."
  else
    mkdir -p "$dino_dir"
    hf_download "facebook/dinov2-vitl14" "$dino_dir"
  fi

  # RMBG-2.0 — background removal used during preprocessing
  local rmbg_dir="$TRELLIS_MODELS_DIR/checkpoints/RMBG-2.0"
  if [[ -d "$rmbg_dir" && -n "$(ls -A "$rmbg_dir" 2>/dev/null)" ]]; then
    echo "  RMBG-2.0 already present, skipping."
  else
    mkdir -p "$rmbg_dir"
    hf_download "briaai/RMBG-2.0" "$rmbg_dir"
  fi

  echo "  Trellis done."
}

# ---- main ------------------------------------------------------------------

check_hf_cli

TARGET="${1:-all}"

case "$TARGET" in
  sdxl)    download_sdxl ;;
  trellis) download_trellis ;;
  all)     download_sdxl; download_trellis ;;
  *)
    echo "Unknown target: $TARGET"
    echo "Usage: bash docker/download_models.sh [sdxl|trellis|all]"
    exit 1
    ;;
esac

echo ""
echo "==> All done. Run 'bash docker/build_run.sh' to start the containers."
