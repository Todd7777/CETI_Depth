#!/usr/bin/env bash
# Professor drop-folder pipeline:
#   1. Copy JPEGs / MP4s into ceti/inbox/uploads/
#   2. Run this script
#   3. Open ceti/inbox/results/<timestamp>/index.html
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export CETI_DEPTH_MODE="${CETI_DEPTH_MODE:-pointcloud}"
export CETI_TANK_PRESET="${CETI_TANK_PRESET:-tank_roi_ceti_full}"

UPLOAD_DIR="${CETI_INBOX:-${REPO_ROOT}/ceti/inbox/uploads}"
CKPT="${CETI_DEPTH_CKPT:-${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt}"

echo "============================================"
echo " CETI Depth — Professor Upload Pipeline"
echo "============================================"
echo "  Drop zone:  ${UPLOAD_DIR}"
echo "  Checkpoint: ${CKPT}"
echo ""

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

N=$(find "$UPLOAD_DIR" -maxdepth 1 -type f \( \
  -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
  -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' \) 2>/dev/null | wc -l | tr -d ' ')

if [ "$N" -eq 0 ]; then
  echo "No files in upload folder."
  echo ""
  echo "  1. Copy images/videos into:"
  echo "       ${UPLOAD_DIR}"
  echo "  2. Re-run:"
  echo "       bash ceti/scripts/run_professor_pipeline.sh"
  echo ""
  echo "  Or use the web portal:"
  echo "       bash ceti/scripts/launch_professor_portal.sh"
  exit 1
fi

echo "  Found ${N} file(s). Processing…"
echo ""

"$PYTHON" -u -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from pathlib import Path
from ceti.depth.upload_pipeline import run_inbox_drop_folder, INBOX_UPLOADS

manifest = run_inbox_drop_folder(Path('${UPLOAD_DIR}'), move_processed=False)
print('RUN_DIR=' + manifest['run_dir'])
"

RUN_DIR=$(ls -td "${REPO_ROOT}/ceti/inbox/results"/*/ 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
  echo ""
  echo "Opening results…"
  open "${RUN_DIR}index.html" 2>/dev/null || open "${RUN_DIR}" 2>/dev/null || true
fi
