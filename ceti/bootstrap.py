"""Ensure repo root and vendored third-party packages are importable."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "3rd_party"


def ensure_paths() -> Path:
    for path in (REPO_ROOT, THIRD_PARTY):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO_ROOT
