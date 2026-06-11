#!/usr/bin/env bash
# Push main to https://github.com/Todd7777/CETI_Depth
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if ! ssh -T git@github.com -o BatchMode=yes 2>&1 | grep -qi 'successfully authenticated'; then
  echo "GitHub SSH auth failed."
  echo ""
  echo "Add your public key at:"
  echo "  https://github.com/settings/ssh/new"
  echo ""
  echo "Public key file:"
  echo "  ~/.ssh/id_ed25519.pub"
  echo ""
  exit 1
fi

git push -u ceti-depth main
echo ""
echo "Done: https://github.com/Todd7777/CETI_Depth"
