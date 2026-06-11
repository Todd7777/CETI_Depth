#!/usr/bin/env bash
# One-time upload of checkpoints/ceti_whale_depth/best.pt to Hugging Face.
#
# Prerequisites:
#   1. Create a token at https://huggingface.co/settings/tokens (Write access)
#   2. export HF_TOKEN=hf_...
#   3. bash ceti/scripts/upload_ceti_checkpoint_to_hf.sh
#
# Colleagues then run:
#   bash ceti/scripts/download_ceti_checkpoint.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${CETI_HF_CHECKPOINT_REPO:-Todd7777/ceti-whale-depth}"
SRC="${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(command -v python3)"

if [ ! -f "$SRC" ]; then
  echo "ERROR: missing $SRC"
  exit 1
fi

TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  echo "ERROR: set HF_TOKEN (https://huggingface.co/settings/tokens, Write access)"
  exit 1
fi

echo "Uploading to Hugging Face model repo: ${HF_REPO}"
echo "  Source: ${SRC} ($(du -h "$SRC" | cut -f1))"
echo ""

"$PYTHON" <<PY
import os
from pathlib import Path
from huggingface_hub import HfApi

repo = os.environ.get("CETI_HF_CHECKPOINT_REPO", "${HF_REPO}")
token = os.environ["HF_TOKEN"] if os.environ.get("HF_TOKEN") else os.environ["HUGGING_FACE_HUB_TOKEN"]
src = Path("${SRC}")

api = HfApi(token=token)
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_file(
    path_or_fileobj=str(src),
    path_in_repo="best.pt",
    repo_id=repo,
    repo_type="model",
    commit_message="CETI fine-tuned Depth Anything ViT-L checkpoint (tank / payload)",
)
print(f"Uploaded: https://huggingface.co/{repo}/tree/main")
PY

echo ""
echo "Share with colleagues:"
echo "  bash ceti/scripts/download_ceti_checkpoint.sh"
