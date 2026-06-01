#!/usr/bin/env bash
# M5 Max: skip setup — only ensure data + train (after setup already passed)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

export CETI_DEVICE="${CETI_DEVICE:-mps}"
export CETI_REQUIRE_MPS="${CETI_REQUIRE_MPS:-1}"
export CETI_UNIFIED_MEMORY_GB="${CETI_UNIFIED_MEMORY_GB:-128}"
export CETI_SKIP_GIT_PULL=1

bash ceti/scripts/ensure_training_data.sh
bash ceti/scripts/train_mac_full.sh
