#!/usr/bin/env bash
# Resume whale depth training — picks best.pt when last.pt diverged (OOM / bad resume)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3

CONFIG="${CETI_TRAIN_CONFIG:-ceti/configs/whale_depth_m5max_128gb.yaml}"
SAVE_DIR="${REPO_ROOT}/checkpoints/ceti_whale_depth"
BEST="${SAVE_DIR}/best.pt"
LAST="${SAVE_DIR}/last.pt"

export CETI_DEVICE=mps
export CETI_REQUIRE_MPS=1
export CETI_UNIFIED_MEMORY_GB=128
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ -n "${CETI_RESUME_CKPT:-}" ]; then
  CKPT="$CETI_RESUME_CKPT"
else
  CKPT="$("$PYTHON" - <<'PY'
import torch
from pathlib import Path

save = Path("checkpoints/ceti_whale_depth")
best = save / "best.pt"
last = save / "last.pt"

def loss_of(p):
    if not p.exists():
        return None
    c = torch.load(p, map_location="cpu", weights_only=False)
    return float(c.get("loss", 1e9))

b = loss_of(best)
l = loss_of(last)
# Prefer last only if it exists and is not much worse than best (divergence guard)
if l is not None and b is not None and l <= b * 1.35:
    print(last)
elif best.exists():
    print(best)
elif last.exists():
    print(last)
else:
    raise SystemExit(1)
PY
)"
fi

echo "============================================"
echo " CETI — Resume Mac Training (safe)"
echo "============================================"
echo "  Config:     $CONFIG"
echo "  Checkpoint: $CKPT"
echo "  (Set CETI_RESUME_CKPT to override auto-pick)"
echo ""

"$PYTHON" ceti/scripts/verify_mps.py

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

# Keep diverged last.pt for forensics, don't train from it by default
if [ -f "$LAST" ] && [ "$CKPT" = "$BEST" ] && [ -f "$BEST" ]; then
  if ! "$PYTHON" -c "
import torch
b=torch.load('$BEST',map_location='cpu',weights_only=False)
l=torch.load('$LAST',map_location='cpu',weights_only=False)
exit(0 if float(l.get('loss',1e9)) <= float(b.get('loss',1e9))*1.35 else 1)
"; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    cp -f "$LAST" "${SAVE_DIR}/last_diverged_${STAMP}.pt"
    echo "  Backed up diverged last.pt → last_diverged_${STAMP}.pt"
  fi
fi

# Stop any stale training job
pkill -f "ceti/depth/train_whale_depth.py" 2>/dev/null || true
sleep 1

"$PYTHON" -u ceti/depth/train_whale_depth.py \
  --config "$CONFIG" \
  --resume "$CKPT" \
  "$@"
