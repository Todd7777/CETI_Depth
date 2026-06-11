"""Minimal web API manifest for portal integration."""

from __future__ import annotations

import json
from pathlib import Path


def write_web_api_manifest(run_dir: Path, assets: list[dict]) -> Path:
    api = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "unit": "relative",
        "assets": assets,
    }
    path = run_dir / "web_api.json"
    path.write_text(json.dumps(api, indent=2), encoding="utf-8")
    return path
