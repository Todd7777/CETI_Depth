#!/usr/bin/env python3
"""
Depth noise sensitivity study for CETI tank payload point clouds.

Gaussian noise on payload depth at 5, 10, 20, 30, 40% of depth σ.
Generates researcher comparison boards: segmentation + clean/noised depth + delta + 3D.

Usage:
  python ceti/scripts/run_depth_noise_study.py
  python ceti/scripts/run_depth_noise_study.py --images ceti/inbox/uploads/Distance_1*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

NOISE_LEVELS_PCT = (0, 5, 10, 20, 30, 40)


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def add_depth_noise(
    depth: np.ndarray,
    mask: np.ndarray,
    noise_pct: float,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Returns (noisy depth, sigma applied)."""
    out = depth.astype(np.float32).copy()
    if noise_pct <= 0:
        return out, 0.0
    vals = out[mask.astype(bool)]
    if vals.size < 10:
        return out, 0.0
    sigma = (noise_pct / 100.0) * float(np.std(vals))
    rng = np.random.default_rng(seed)
    noise = np.zeros_like(out)
    ys, xs = np.where(mask)
    noise[ys, xs] = rng.normal(0.0, sigma, size=ys.shape[0]).astype(np.float32)
    out[mask] = np.maximum(out[mask] + noise[mask], 1e-3)
    return out, sigma


def displacement_stats(baseline: np.ndarray, noisy: np.ndarray) -> dict:
    if baseline.shape != noisy.shape or baseline.shape[0] < 3:
        return {"mean": None, "median": None, "max": None, "rms": None}
    b = baseline - np.median(baseline, axis=0)
    n = noisy - np.median(noisy, axis=0)
    d = np.linalg.norm(n - b, axis=1)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
        "rms": float(np.sqrt(np.mean(d**2))),
    }


def process_image(
    image_path: Path,
    out_dir: Path,
    *,
    rel_model,
    transform,
    device: str,
    noise_levels: tuple[int, ...] = NOISE_LEVELS_PCT,
) -> dict:
    from ceti.depth.infer_robot import predict_depth_raw
    from ceti.depth.upload_pipeline import _pointcloud_settings
    from ceti.geometry.dji_action4 import dji_action4_intrinsics
    from ceti.geometry.point_cloud import depth_rgb_to_point_cloud, plane_fit_rms, write_ply
    from ceti.geometry.showcase import (
        compose_noise_research_board,
        compose_noise_research_row,
        compose_showcase_panel,
        render_point_cloud_views,
    )
    from ceti.preprocessing.payload_segmentation import (
        build_payload_mask,
        render_payload_overlay,
    )
    from ceti.preprocessing.tank_roi import load_roi_preset
    from ceti.preprocessing.underwater import preprocess_underwater

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    h, w = frame.shape[:2]
    preset = load_roi_preset() or {}
    proc = preprocess_underwater(frame, method="combined")
    depth_clean = predict_depth_raw(rel_model, transform, proc, device)
    mask, mask_mode = build_payload_mask(frame, depth_clean, preset=preset)
    if mask is None:
        raise RuntimeError(f"Payload segmentation failed for {image_path.name}")

    settings = _pointcloud_settings()
    intrinsics = dji_action4_intrinsics(w, h, model=settings["model"])
    stem = image_path.stem
    img_dir = out_dir / stem
    img_dir.mkdir(parents=True, exist_ok=True)

    seg_path = img_dir / f"{stem}_segmentation.jpg"
    cv2.imwrite(
        str(seg_path),
        render_payload_overlay(frame, mask),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    cv2.imwrite(str(img_dir / f"{stem}_payload_mask.png"), (mask.astype(np.uint8) * 255))

    baseline_pts = None
    records = []
    level_data = []
    depth_std = float(np.std(depth_clean[mask]))

    for pct in noise_levels:
        depth_n, sigma = add_depth_noise(depth_clean, mask, float(pct), seed=42 + int(pct))
        pts, colors = depth_rgb_to_point_cloud(
            depth_n,
            frame,
            intrinsics,
            mask=mask,
            depth_scale=settings["depth_scale"],
            subsample=settings["subsample"],
        )
        rms = plane_fit_rms(pts)
        tag = f"{pct:02d}pct"

        ply_path = img_dir / f"{stem}_noise_{tag}.ply"
        write_ply(ply_path, pts, colors, binary=settings["binary_ply"])
        np.save(img_dir / f"{stem}_depth_noise_{tag}.npy", depth_n)

        row_path = img_dir / f"{stem}_research_row_{tag}.jpg"
        row_img = compose_noise_research_row(
            frame,
            mask,
            depth_clean,
            depth_n,
            pts,
            colors,
            noise_pct=float(pct),
        )
        cv2.imwrite(str(row_path), row_img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])

        snap = img_dir / f"{stem}_3d_{tag}.jpg"
        render_point_cloud_views(pts, colors, snap)
        showcase = img_dir / f"{stem}_showcase_{tag}.jpg"
        compose_showcase_panel(frame, mask, depth_n, pts, colors, showcase)

        if pct == 0:
            baseline_pts = pts.copy()
        disp = None
        if pct > 0 and baseline_pts is not None and pts.shape == baseline_pts.shape:
            disp = displacement_stats(baseline_pts, pts)

        level_data.append(
            {"noise_pct": pct, "depth_noisy": depth_n, "points": pts, "colors": colors}
        )
        records.append(
            {
                "noise_pct": pct,
                "noise_sigma": sigma,
                "point_count": int(pts.shape[0]),
                "plane_fit_rms": rms,
                "displacement_vs_0pct": disp,
                "ply": f"{stem}/{stem}_noise_{tag}.ply",
                "showcase": f"{stem}/{stem}_showcase_{tag}.jpg",
                "research_row": f"{stem}/{stem}_research_row_{tag}.jpg",
                "snapshot_3d": f"{stem}/{stem}_3d_{tag}.jpg",
            }
        )

    board_path = img_dir / f"{stem}_research_board.jpg"
    compose_noise_research_board(frame, mask, depth_clean, level_data, board_path)

    summary = {
        "source": image_path.name,
        "stem": stem,
        "mask_mode": mask_mode,
        "mask_pixels": int(mask.sum()),
        "depth_std_payload": depth_std,
        "intrinsics": intrinsics.to_dict(),
        "noise_levels_pct": list(noise_levels),
        "segmentation_preview": f"{stem}/{stem}_segmentation.jpg",
        "research_board": f"{stem}/{stem}_research_board.jpg",
        "payload_mask": f"{stem}/{stem}_payload_mask.png",
        "results": records,
    }
    (img_dir / f"{stem}_noise_study.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def run_noise_study(
    images: list[Path],
    *,
    output: Path | None = None,
) -> dict:
    from ceti.depth.upload_pipeline import _load_models
    from ceti.utils.device import configure_compute, device_name, get_device
    from ceti.web.researcher_ui import render_noise_study_index, write_static_html

    configure_compute()
    device = str(get_device())
    ckpt = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"
    model, transform, _ = _load_models(ckpt, device)

    run_id = _utc_id()
    folder_name = output.name if output else f"noise_study_{run_id}"
    out_dir = output or (REPO_ROOT / "ceti/inbox/results" / folder_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": folder_name,
        "device": device_name(get_device()),
        "noise_levels_pct": list(NOISE_LEVELS_PCT),
        "method": "gaussian_on_depth_sigma_pct",
        "images": [],
    }

    for img in images:
        img = Path(img)
        if not img.is_file():
            continue
        print(f"  Noise study: {img.name}")
        summary = process_image(
            img,
            out_dir,
            rel_model=model,
            transform=transform,
            device=device,
        )
        manifest["images"].append(summary)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def study_url(rid: str, _stem: str = "") -> str:
        return f"noise/results/{rid}"

    def asset_url(rid: str, rel: str) -> str:
        return f"noise/results/{rid}/{rel}"

    html = render_noise_study_index(
        manifest,
        study_url=lambda rid, stem="": (
            f"/noise/results/{rid}/view/{stem}" if stem else f"/noise/results/{rid}"
        ),
        asset_url=lambda rid, rel: f"/noise/results/{rid}/files/{rel}",
        zip_url=f"/noise/results/{manifest['run_id']}/download.zip",
    )
    write_static_html(out_dir / "index.html", html)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="CETI depth noise sensitivity study")
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=list((REPO_ROOT / "ceti/inbox/uploads").glob("Distance_*Check*.png")),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = run_noise_study([Path(p) for p in args.images], output=args.output)
    out_name = manifest["run_id"]
    out_dir = args.output or (REPO_ROOT / "ceti/inbox/results" / out_name)
    print(f"\nDone. Open: {out_dir / 'index.html'}")
    print(f"Portal:  http://127.0.0.1:7860/noise/results/{out_name}")


if __name__ == "__main__":
    main()
