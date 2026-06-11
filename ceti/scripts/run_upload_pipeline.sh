#!/usr/bin/env bash
# Drop-folder pipeline: process all media in ceti/inbox/uploads/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

export CETI_DEVICE="${CETI_DEVICE:-$(
  if [[ "$(uname -s)" == "Darwin" ]]; then echo mps; else echo cuda; fi
)}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi
export CETI_TANK_PRESET="${CETI_TANK_PRESET:-tank_roi_ceti_full}"

UPLOAD_DIR="${CETI_INBOX:-${REPO_ROOT}/ceti/inbox/uploads}"
CKPT="${CETI_DEPTH_CKPT:-${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt}"

echo "============================================"
echo " CETI Point Cloud — Batch Pipeline"
echo "============================================"
echo "  Uploads:    ${UPLOAD_DIR}"
echo "  Checkpoint: ${CKPT}"
echo ""

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  echo "  bash ceti/scripts/download_checkpoint.sh"
  exit 1
fi

N=$(find "$UPLOAD_DIR" -maxdepth 1 -type f \( \
  -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
  -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$N" -eq 0 ]; then
  echo "No files in ${UPLOAD_DIR}"
  echo "  cp your tank images there, then re-run this script."
  exit 1
fi

echo "  Processing ${N} file(s)…"
echo ""

"$PYTHON" -u -c "
from ceti.bootstrap import ensure_paths
ensure_paths()
from pathlib import Path
from ceti.depth.upload_pipeline import run_inbox_drop_folder

manifest = run_inbox_drop_folder(Path('${UPLOAD_DIR}'), move_processed=False)
print('RUN_DIR=' + manifest['run_dir'])
"

RUN_DIR=$(ls -td "${REPO_ROOT}/ceti/inbox/results"/*/ 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
  echo ""
  echo "Results: ${RUN_DIR}"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "${RUN_DIR}index.html" 2>/dev/null || true
  fi
fi
