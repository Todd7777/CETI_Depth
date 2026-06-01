#!/usr/bin/env bash
# Whale-Depth-Anything — M5 Max 128GB: setup → data → train (no broken git pull)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"

echo "============================================"
echo " Whale-Depth-Anything — M5 Max full pipeline"
echo " Repo: $REPO_ROOT"
echo "============================================"

# --- 1) Optional git pull (never fail the pipeline) ---
echo "[1/7] Git update (optional)…"
if [ "${CETI_SKIP_GIT_PULL:-0}" = "1" ]; then
  echo "  Skipped (CETI_SKIP_GIT_PULL=1)"
elif git rev-parse --is-inside-work-tree &>/dev/null; then
  PULLED=0
  for remote in whale origin; do
    if git remote get-url "$remote" &>/dev/null 2>&1; then
      if git ls-remote --heads "$remote" main &>/dev/null 2>&1; then
        echo "  git pull $remote main"
        git pull "$remote" main && PULLED=1 && break || true
      fi
    fi
  done
  if [ "$PULLED" = "0" ]; then
    echo "  No working git remote — continuing with local copy"
  fi
else
  echo "  Not a git repo — skip"
fi

# --- 2) Venv + deps + MPS ---
echo "[2/7] Mac MPS setup…"
bash ceti/scripts/setup_mac_mps.sh

# --- 3) Checkpoints ---
echo "[3/7] Download checkpoints…"
bash ceti/scripts/download_checkpoints.sh

# --- 4) Smoke test ---
echo "[4/7] Smoke test…"
"${REPO_ROOT}/.venv/bin/python" ceti/scripts/smoke_test.py

# --- 5) Training images MUST exist on disk ---
echo "[5/7] Ensure training data on disk…"
bash ceti/scripts/ensure_training_data.sh

# --- 6) Full training + proof ---
echo "[6/7] Full training (ViT-L, MPS)…"
bash ceti/scripts/train_mac_full.sh

echo ""
echo "============================================"
echo " DONE"
echo "  Checkpoint: checkpoints/ceti_whale_depth/best.pt"
echo "  Proof:      ceti/outputs/proof/"
echo "============================================"
