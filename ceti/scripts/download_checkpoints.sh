#!/usr/bin/env bash
# Download Depth Anything pretrained checkpoints
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_DIR="$REPO_ROOT/checkpoints"
mkdir -p "$CKPT_DIR"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

download_hf() {
    local repo="$1"
    local filename="$2"
    local dest_name="${3:-$2}"

    local dest="$CKPT_DIR/$dest_name"

    if [ -f "$dest" ]; then
        echo "  ✓ $dest_name (already exists)"
        return 0
    fi

    echo "  ↓ Downloading $filename from $repo → $dest_name..."
    "$PYTHON" -c "
import os
from huggingface_hub import hf_hub_download
import shutil
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
path = hf_hub_download(repo_id='$repo', filename='$filename', token=token)
shutil.copy(path, '$dest')
print(f'  ✓ Saved to $dest')
" || {
        echo "  ✗ Failed — try: huggingface-cli download $repo $filename --local-dir $CKPT_DIR"
        return 1
    }
}

echo "Downloading Depth Anything checkpoints..."
echo ""

# HF model repos store weights as pytorch_model.bin
download_hf "LiheYoung/depth_anything_vits14" "pytorch_model.bin" "depth_anything_vits14.pth" || true
download_hf "LiheYoung/depth_anything_vitb14" "pytorch_model.bin" "depth_anything_vitb14.pth" || true
download_hf "LiheYoung/depth_anything_vitl14" "pytorch_model.bin" "depth_anything_vitl14.pth" || true

# Metric depth checkpoints live in the HF *Space* (not the model repo)
download_metric() {
    local filename="$1"
    local dest_name="$2"
    local dest="$CKPT_DIR/$dest_name"
    if [ -f "$dest" ]; then
        echo "  ✓ $dest_name (already exists)"
        return 0
    fi
    echo "  ↓ Downloading $filename (HF Space) → $dest_name..."
    "$PYTHON" -c "
import os, shutil
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
path = hf_hub_download(
    repo_id='LiheYoung/Depth-Anything',
    filename='$filename',
    repo_type='space',
    token=token,
)
shutil.copy(path, '$dest')
print('  ✓ Saved to $dest')
" 2>/dev/null || {
        echo "  ⚠ Optional metric checkpoint skipped: $dest_name (Track A only; relative depth train does not need it)"
        return 0
    }
}

download_metric "checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt" "depth_anything_metric_depth_indoor.pt"
download_metric "checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt" "depth_anything_metric_depth_outdoor.pt"

echo ""
echo "Checkpoint directory: $CKPT_DIR"
ls -lh "$CKPT_DIR"/ 2>/dev/null || true
echo ""
echo "Note: Models also load automatically via DepthAnything.from_pretrained() at runtime."
