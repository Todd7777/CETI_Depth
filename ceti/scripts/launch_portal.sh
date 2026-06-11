#!/usr/bin/env bash
# CETI research portal (Flask).
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
export CETI_PORTAL_PORT="${CETI_PORTAL_PORT:-7860}"
export CETI_TANK_PRESET="${CETI_TANK_PRESET:-tank_roi_ceti_full}"
export CETI_DEPTH_MODE="${CETI_DEPTH_MODE:-pointcloud}"

CKPT="${REPO_ROOT}/checkpoints/ceti_whale_depth/best.pt"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: missing ${CKPT}"
  exit 1
fi

"$PYTHON" -c "import flask" 2>/dev/null || {
  echo "Installing Flask…"
  "$PYTHON" -m pip install -q 'flask>=3.0.0'
}

if command -v lsof >/dev/null 2>&1; then
  OLD_PIDS="$(lsof -ti tcp:"${CETI_PORTAL_PORT}" 2>/dev/null || true)"
  if [ -n "$OLD_PIDS" ]; then
    echo "Stopping previous portal on port ${CETI_PORTAL_PORT}…"
    kill $OLD_PIDS 2>/dev/null || true
    sleep 1
  fi
fi

echo "============================================"
echo " CETI Research Portal"
echo "============================================"
echo "  URL:    http://127.0.0.1:${CETI_PORTAL_PORT}"
echo "  Health: http://127.0.0.1:${CETI_PORTAL_PORT}/health"
echo "  Preset: ${CETI_TANK_PRESET}"
echo "  CLI:    bash ceti/scripts/run_upload_pipeline.sh"
echo ""

exec "$PYTHON" -u ceti/scripts/ceti_depth_portal_web.py
