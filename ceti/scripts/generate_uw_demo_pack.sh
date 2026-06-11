#!/usr/bin/env bash
# Generate many underwater demo stills + videos (baseline vs CETI fine-tuned).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "Generating demo pack (this takes ~15–45 min on M5 Max)…"
"$PYTHON" -u ceti/scripts/generate_demo_pack.py \
  --max-stills "${CETI_DEMO_STILLS:-36}" \
  --frames-per-video "${CETI_DEMO_FRAMES:-10}" \
  "$@"

echo ""
echo "Open outputs:"
echo "  open ${REPO_ROOT}/ceti/outputs/demo_pack/stills"
echo "  open ${REPO_ROOT}/ceti/outputs/demo_pack/videos"
echo "  open ${REPO_ROOT}/ceti/outputs/demo_pack/videos_compare"
