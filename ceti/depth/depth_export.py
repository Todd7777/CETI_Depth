"""
Export depth-in-meters as standalone files for web / API integration.

Outputs (under run_dir/distances/):
  - {stem}.json   — summary + downsampled grid (website-friendly)
  - {stem}.csv    — sampled distances (row, col, distance_m)
  - {stem}.npy    — full H×W float32 meters

Previews (RGB|depth panels) stay in run_dir/previews/ only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def depth_summary(
    depth_m: np.ndarray,
    subject_mask: np.ndarray | None = None,
) -> dict[str, float]:
    h, w = depth_m.shape
    cy, cx = h // 2, w // 2
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if subject_mask is not None:
        valid &= subject_mask.astype(bool)
    if subject_mask is not None and valid.any():
        ys, xs = np.where(valid)
        cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    if not valid.any():
        return {
            "center_m": float(depth_m[h // 2, w // 2]),
            "min_m": 0.0,
            "max_m": 0.0,
            "median_m": 0.0,
            "p10_m": 0.0,
            "p90_m": 0.0,
        }
    v = depth_m[valid].astype(np.float64)
    return {
        "center_m": float(depth_m[cy, cx]),
        "min_m": float(v.min()),
        "max_m": float(v.max()),
        "median_m": float(np.median(v)),
        "p10_m": float(np.percentile(v, 10)),
        "p90_m": float(np.percentile(v, 90)),
    }


def downsample_depth(
    depth_m: np.ndarray,
    *,
    max_width: int = 160,
    max_height: int = 120,
) -> tuple[np.ndarray, int, int]:
    """Return (grid, grid_width, grid_height) for JSON embedding."""
    import cv2

    h, w = depth_m.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    gw = max(1, int(round(w * scale)))
    gh = max(1, int(round(h * scale)))
    if gw == w and gh == h:
        grid = depth_m.astype(np.float32)
    else:
        grid = cv2.resize(depth_m.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return grid, gw, gh


def build_depth_payload(
    depth_m: np.ndarray,
    *,
    source_file: str,
    mode: str = "accurate",
    alignment: dict | None = None,
    roi: dict | None = None,
    subject_mask: np.ndarray | None = None,
    max_grid_width: int = 160,
    max_grid_height: int = 120,
) -> dict[str, Any]:
    grid, gw, gh = downsample_depth(
        depth_m, max_width=max_grid_width, max_height=max_grid_height
    )
    h, w = depth_m.shape[:2]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "unit": "meters",
        "mode": mode,
        "width": int(w),
        "height": int(h),
        "alignment": alignment or {},
        "roi": roi,
        "summary": depth_summary(depth_m, subject_mask),
        "grid": {
            "width": gw,
            "height": gh,
            "depth_m": grid.round(4).tolist(),
        },
    }


def write_depth_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_depth_csv(
    path: Path,
    depth_m: np.ndarray,
    *,
    sample_grid: int = 24,
) -> Path:
    """
    CSV: row,col,distance_m on a regular grid (default 24×24 samples).
    Maps row/col to original image coordinates.
    """
    import csv

    h, w = depth_m.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "distance_m"])
        for ri in range(sample_grid):
            for ci in range(sample_grid):
                y = int(round((ri + 0.5) / sample_grid * h - 0.5))
                x = int(round((ci + 0.5) / sample_grid * w - 0.5))
                y = min(max(y, 0), h - 1)
                x = min(max(x, 0), w - 1)
                writer.writerow([y, x, f"{float(depth_m[y, x]):.6f}"])
    return path


def write_depth_npy(path: Path, depth_m: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth_m.astype(np.float32))
    return path


def export_distance_files(
    distances_dir: Path,
    stem: str,
    depth_m: np.ndarray,
    *,
    source_file: str,
    mode: str = "accurate",
    alignment: dict | None = None,
    roi: dict | None = None,
    subject_mask: np.ndarray | None = None,
) -> dict[str, str]:
    """
    Write json, csv, npy under distances_dir. Returns relative filenames (within distances_dir).
    """
    distances_dir.mkdir(parents=True, exist_ok=True)
    payload = build_depth_payload(
        depth_m,
        source_file=source_file,
        mode=mode,
        alignment=alignment,
        roi=roi,
        subject_mask=subject_mask,
    )
    json_name = f"{stem}.json"
    csv_name = f"{stem}.csv"
    npy_name = f"{stem}.npy"
    write_depth_json(distances_dir / json_name, payload)
    write_depth_csv(distances_dir / csv_name, depth_m)
    write_depth_npy(distances_dir / npy_name, depth_m)
    return {
        "distance_json": json_name,
        "distance_csv": csv_name,
        "distance_npy": npy_name,
    }


def append_video_frame_record(
    records: list[dict],
    frame_index: int,
    depth_m: np.ndarray,
    info: dict,
    *,
    subject_mask: np.ndarray | None = None,
) -> None:
    records.append(
        {
            "frame": frame_index,
            **depth_summary(depth_m, subject_mask),
            "scale": info.get("scale"),
            "shift": info.get("shift"),
        }
    )


def write_video_depth_json(
    path: Path,
    *,
    source_file: str,
    fps: float,
    frames: list[dict],
    mode: str = "accurate",
) -> Path:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_file": source_file,
        "type": "video",
        "unit": "meters",
        "mode": mode,
        "fps": fps,
        "frame_count": len(frames),
        "frames": frames,
    }
    return write_depth_json(path, payload)


def write_web_api_manifest(run_dir: Path, assets: list[dict]) -> Path:
    """Single index for frontend: list all distance + preview paths."""
    api = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "unit": "meters",
        "assets": assets,
    }
    path = run_dir / "web_api.json"
    path.write_text(json.dumps(api, indent=2), encoding="utf-8")
    return path
