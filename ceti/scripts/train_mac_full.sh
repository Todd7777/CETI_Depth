#!/usr/bin/env bash
# Full CETI pipeline for Apple M5 Max 128GB: MPS verify → data → train → prove
# Aligned with Professor Stephanie Gil / AVATARS (Harvard SEAS, Project CETI)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

CONFIG="${CETI_TRAIN_CONFIG:-ceti/configs/whale_depth_m5max_128gb.yaml}"

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"

echo "============================================"
echo " Whale-Depth-Anything — M5 Max Full Train"
echo "============================================"
echo "  Config:     $CONFIG"
echo "  Device:     $CETI_DEVICE (require MPS=$CETI_REQUIRE_MPS)"
echo "  CPU threads: $OMP_NUM_THREADS"

"$PYTHON" ceti/scripts/verify_mps.py

bash ceti/scripts/ensure_training_data.sh

"$PYTHON" ceti/depth/train_whale_depth.py \
  --config "$CONFIG" \
  "$@"

bash ceti/scripts/prove_pipeline.sh --skip-metric-train

echo ""
echo "Done."
echo "  Checkpoint: checkpoints/ceti_whale_depth/best.pt"
echo "  Proof:      ceti/outputs/proof/"
