#!/usr/bin/env bash
# Download CETI fine-tuned depth checkpoint (best.pt) from Hugging Face.
#
# Default repo: Todd7777/ceti-whale-depth
# Override: export CETI_HF_CHECKPOINT_REPO=your-org/your-repo
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${CETI_HF_CHECKPOINT_REPO:-Todd7777/ceti-whale-depth}"
DEST_DIR="${REPO_ROOT}/checkpoints/ceti_whale_depth"
DEST="${DEST_DIR}/best.pt"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

mkdir -p "$DEST_DIR"

if [ -f "$DEST" ]; then
  echo "Already present: $DEST"
  ls -lh "$DEST"
  exit 0
fi

echo "Downloading CETI checkpoint from Hugging Face…"
echo "  Repo: ${HF_REPO}"
echo "  File: best.pt → ${DEST}"
echo ""

"$PYTHON" -c "
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ.get('CETI_HF_CHECKPOINT_REPO', '${HF_REPO}')
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
dest = Path('${DEST}')
path = hf_hub_download(
    repo_id=repo,
    filename='best.pt',
    token=token,
)
shutil.copy(path, dest)
print(f'Saved {dest} ({dest.stat().st_size / (1024**3):.2f} GB)')
"

echo ""
echo "Done. Start the portal:"
echo "  bash ceti/scripts/launch_portal.sh"
