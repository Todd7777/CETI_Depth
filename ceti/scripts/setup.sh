#!/usr/bin/env bash
# CETI Depth — environment setup (Linux + macOS).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
PIP="${REPO_ROOT}/.venv/bin/pip"
CKPT="${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt"

echo "============================================"
echo " CETI Depth — Point Cloud Setup"
echo "============================================"

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtual environment…"
  python3 -m venv .venv
fi

"$PIP" install -U pip wheel
"$PIP" install -r requirements.txt

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "macOS — using default PyTorch (MPS when available)."
elif python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA detected — installing GPU PyTorch (cu124)."
  "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "No CUDA — installing CPU PyTorch."
  "$PIP" install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

mkdir -p checkpoints/ceti_whale_depth ceti/inbox/uploads ceti/inbox/results

if [ ! -f "$CKPT" ]; then
  echo ""
  echo "Checkpoint not found. Download with:"
  echo "  bash ceti/scripts/download_checkpoint.sh"
fi

"$PYTHON" -c "
from ceti.bootstrap import ensure_paths
ensure_paths()
import torch
from pathlib import Path
print('PyTorch', torch.__version__)
if torch.cuda.is_available():
    print('Device: cuda')
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    print('Device: mps')
else:
    print('Device: cpu')
print('Checkpoint:', Path('$CKPT').resolve(), '(exists:', Path('$CKPT').is_file(), ')')
"

echo ""
echo "Next steps:"
echo "  bash ceti/scripts/download_checkpoint.sh   # if checkpoint missing"
echo "  bash ceti/scripts/launch_portal.sh"
echo "  bash ceti/scripts/run_batch.sh               # batch process inbox/uploads"
