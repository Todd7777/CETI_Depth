#!/usr/bin/env bash
# Download CETI fine-tuned checkpoint from Hugging Face.
# Override: export CETI_HF_CHECKPOINT_REPO=your-org/your-repo
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${CETI_HF_CHECKPOINT_REPO:-Todd7777/ceti-whale-depth}"
DEST_DIR="${REPO_ROOT}/checkpoints/ceti_whale_depth"
DEST="${DEST_DIR}/best.pt"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

mkdir -p "$DEST_DIR"

if [ -f "$DEST" ]; then
  echo "Checkpoint already present: $DEST"
  exit 0
fi

echo "============================================"
echo " Download CETI checkpoint"
echo "============================================"
echo "  Repo: ${HF_REPO}"
echo "  Dest: ${DEST}"
echo ""

"$PYTHON" - <<PY
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = os.environ.get("CETI_HF_CHECKPOINT_REPO", "${HF_REPO}")
dest = Path("${DEST}")
dest.parent.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=repo, filename="best.pt", local_dir=str(dest.parent), local_dir_use_symlinks=False)
final = dest.parent / "best.pt"
if Path(path).resolve() != final.resolve() and Path(path).is_file():
    Path(path).replace(final)
print("Saved:", final, f"({final.stat().st_size // (1024*1024)} MB)")
PY
