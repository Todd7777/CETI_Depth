#!/usr/bin/env bash
# Minimal Linux setup: CETI fine-tuned depth → 3D point clouds via web portal.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt"
PYTHON="${REPO_ROOT}/.venv/bin/python"
PIP="${REPO_ROOT}/.venv/bin/pip"

echo "============================================"
echo " CETI Depth — Point Cloud Portal (Linux)"
echo "============================================"

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
fi

"$PIP" install -U pip wheel
"$PIP" install -r requirements.txt
"$PIP" install -r ceti/requirements.txt
"$PIP" install -q 'flask>=3.0.0'

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA detected — using GPU PyTorch (cu124)."
  "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "No CUDA — installing CPU PyTorch."
  "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

mkdir -p checkpoints/ceti_whale_depth ceti/inbox/uploads ceti/inbox/results

if [ ! -f "$CKPT" ]; then
  echo "Fine-tuned checkpoint not found — downloading from Hugging Face…"
  bash ceti/scripts/download_ceti_checkpoint.sh || {
    echo ""
    echo "ERROR: could not obtain $CKPT"
    echo "  bash ceti/scripts/download_ceti_checkpoint.sh"
    echo "  Or copy best.pt manually into checkpoints/ceti_whale_depth/"
    exit 1
  }
fi

"$PYTHON" -c "
import torch
from pathlib import Path
print('PyTorch', torch.__version__)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('Checkpoint:', Path('$CKPT').resolve())
"

echo ""
echo "Setup complete. Start the portal:"
echo "  bash ceti/scripts/launch_portal.sh"
echo "  → http://127.0.0.1:7860"
