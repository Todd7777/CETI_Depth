#!/usr/bin/env bash
# Process every image/video in ceti/inbox/uploads/ in one pipeline run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

exec bash ceti/scripts/run_upload_pipeline.sh "$@"
