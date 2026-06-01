#!/usr/bin/env bash
# Ensure data/underwater_field/rgb exists and train lists point at real files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

RGB_DIR="${REPO_ROOT}/data/underwater_field/rgb"
TRAIN_LIST="${REPO_ROOT}/ceti/data/whale_depth_train.txt"
MIN_IMAGES="${CETI_MIN_TRAIN_IMAGES:-500}"

count_ok() {
  "$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${REPO_ROOT}')
from ceti.depth.whale_depth_dataset import load_image_paths
n = sum(1 for p in load_image_paths('${TRAIN_LIST}') if p.is_file())
print(n)
" 2>/dev/null || echo 0
}

EXISTING="$(count_ok)"
echo "Training images on disk: ${EXISTING} (need >= ${MIN_IMAGES})"

if [ "$EXISTING" -lt "$MIN_IMAGES" ]; then
  echo "Downloading phase-1 underwater RGB (this takes a while)…"
  bash "${REPO_ROOT}/ceti/scripts/download_all_online_data.sh"
  EXISTING="$(count_ok)"
fi

if [ "$EXISTING" -lt "$MIN_IMAGES" ]; then
  echo "ERROR: Still only ${EXISTING} images under ${RGB_DIR}"
  echo "  Check network / HF_TOKEN, then re-run:"
  echo "  bash ceti/scripts/download_all_online_data.sh"
  exit 1
fi

echo "Rebuilding train/val lists from files on disk…"
"$PYTHON" -c "
import sys
sys.path.insert(0, '${REPO_ROOT}')
from pathlib import Path
from ceti.data_curation.underwater_real import build_train_val_lists, REPO_ROOT
rgb = REPO_ROOT / 'data/underwater_field/rgb'
if not rgb.exists():
    raise SystemExit('Missing data/underwater_field/rgb')
build_train_val_lists(rgb, REPO_ROOT/'ceti/data/whale_depth_train.txt', REPO_ROOT/'ceti/data/whale_depth_val.txt')
build_train_val_lists(rgb, REPO_ROOT/'ceti/data/underwater_field_train.txt', REPO_ROOT/'ceti/data/underwater_field_val.txt')
from ceti.depth.whale_depth_dataset import load_image_paths
n = sum(1 for p in load_image_paths(REPO_ROOT/'ceti/data/whale_depth_train.txt') if p.is_file())
print(f'  Train images verified: {n}')
"

echo "Data ready."
