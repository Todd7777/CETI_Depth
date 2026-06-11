"""
CETI tank upload pipeline — fine-tuned depth → payload + scene point clouds.

Used by:
  - ceti/scripts/run_upload_pipeline.sh
  - ceti/scripts/run_batch.sh
  - ceti/scripts/ceti_depth_portal_web.py
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ceti.bootstrap import REPO_ROOT, ensure_paths

ensure_paths()

DEFAULT_CKPT = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"
INBOX_UPLOADS = REPO_ROOT / "ceti/inbox/uploads"
INBOX_RESULTS = REPO_ROOT / "ceti/inbox/results"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def discover_media(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    expanded: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                    expanded.append(f)
        elif p.is_file():
            expanded.append(p)
    images = sorted({p for p in expanded if p.suffix.lower() in IMAGE_EXTS})
    videos = sorted({p for p in expanded if p.suffix.lower() in VIDEO_EXTS})
    return images, videos


def _tank_roi_margins() -> tuple[float, float, float, float] | None:
    from ceti.preprocessing.tank_roi import DEFAULT_MARGINS, load_roi_preset, margins_from_env

    env = margins_from_env()
    if env is not None:
        return env
    preset = load_roi_preset()
    if preset and "margins" in preset:
        m = preset["margins"]
        if len(m) == 4:
            return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))
    return DEFAULT_MARGINS


def _load_models(checkpoint: Path, device: str):
    from ceti.depth.infer import build_depth_model, resolve_encoder

    enc = resolve_encoder(checkpoint)
    model, transform = build_depth_model(enc, device, str(checkpoint))
    return model, transform, enc


def _pointcloud_settings() -> dict:
    binary_env = os.environ.get("CETI_POINTCLOUD_BINARY", "1").strip().lower()
    return {
        "subsample": max(1, int(os.environ.get("CETI_POINTCLOUD_SUBSAMPLE", "1"))),
        "depth_scale": float(os.environ.get("CETI_POINTCLOUD_DEPTH_SCALE", "1.0")),
        "model": os.environ.get("CETI_UNPROJECT_MODEL", "").strip() or None,
        "binary_ply": binary_env in ("1", "true", "yes"),
    }


def _process_image_pointcloud(
    image_path: Path,
    previews_dir: Path,
    pointclouds_dir: Path,
    rel_model,
    rel_transform,
    device: str,
    *,
    underwater_preprocess: bool = True,
    tank_roi: bool = True,
) -> dict:
    from ceti.depth.infer import predict_depth_raw
    from ceti.geometry.dji_action4 import dji_action4_intrinsics, write_intrinsics_json
    from ceti.geometry.point_cloud import (
        adaptive_scene_subsample,
        depth_rgb_to_payload_point_cloud,
        depth_rgb_to_scene_point_cloud,
        plane_fit_rms,
        write_ply,
        write_pointcloud_readme,
    )
    from ceti.geometry.showcase import (
        compose_3d_contrast_panel,
        compose_showcase_panel,
        render_point_cloud_views,
    )
    from ceti.preprocessing.payload_segmentation import (
        render_mask_border_overlay,
        render_payload_overlay,
        segment_ceti_payload,
    )
    from ceti.preprocessing.tank_roi import build_scene_contrast_mask, load_roi_preset, resolve_tank_roi
    from ceti.preprocessing.underwater import preprocess_underwater

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    h, w = frame.shape[:2]
    preset = load_roi_preset() or {}
    roi = resolve_tank_roi(frame, enabled=tank_roi, margins=_tank_roi_margins()) if tank_roi else None

    proc = preprocess_underwater(frame, method="combined") if underwater_preprocess else frame.copy()
    depth = predict_depth_raw(rel_model, rel_transform, proc, device)
    seg_result = segment_ceti_payload(frame, depth, preset=preset)
    if seg_result is None:
        raise RuntimeError(
            f"Payload segmentation failed for {image_path.name}. "
            "Check tank ROI preset (ceti/configs/tank_roi_ceti_full.json)."
        )
    mask_full, mask_used = seg_result.mask, seg_result.mode
    settings = _pointcloud_settings()
    intrinsics = dji_action4_intrinsics(w, h, model=settings["model"])

    pts, colors = depth_rgb_to_payload_point_cloud(
        depth,
        frame,
        intrinsics,
        mask_full,
        depth_scale=settings["depth_scale"],
        subsample=settings["subsample"],
    )
    plane_rms = plane_fit_rms(pts)
    if pts.shape[0] == 0:
        raise RuntimeError(f"Zero payload points for {image_path.name}.")

    scene_sub = adaptive_scene_subsample(h, w, settings["subsample"])
    scene_mask = build_scene_contrast_mask(frame, depth, roi, preset=preset, payload_mask=mask_full)
    pts_scene, colors_scene = depth_rgb_to_scene_point_cloud(
        depth,
        frame,
        intrinsics,
        scene_mask,
        mask_full,
        depth_scale=settings["depth_scale"],
        subsample=scene_sub,
        smooth=True,
        outlier_filter=True,
    )

    stem = image_path.stem
    pointclouds_dir.mkdir(parents=True, exist_ok=True)
    ply_path = pointclouds_dir / f"{stem}.ply"
    ply_scene_path = pointclouds_dir / f"{stem}_scene.ply"
    write_ply(ply_path, pts, colors, binary=settings["binary_ply"])
    write_ply(ply_scene_path, pts_scene, colors_scene, binary=settings["binary_ply"])
    np.save(pointclouds_dir / f"{stem}_depth.npy", depth.astype(np.float32))
    write_intrinsics_json(
        pointclouds_dir / f"{stem}_intrinsics.json",
        intrinsics,
        extra={
            "source_file": image_path.name,
            "point_count": int(pts.shape[0]),
            "mask_mode": mask_used,
            "plane_fit_rms_relative": plane_rms,
            "aspect_mode": intrinsics.aspect_mode,
            "depth_semantics": intrinsics.depth_semantics,
        },
    )
    write_pointcloud_readme(
        pointclouds_dir / f"{stem}_readme.txt",
        source=image_path.name,
        intrinsics=intrinsics,
        point_count=int(pts.shape[0]),
        depth_scale=settings["depth_scale"],
        plane_rms=plane_rms,
        units_note=(
            "Depth is CETI RELATIVE units (not meters). See ceti/docs/POINTCLOUD_ACCURACY.md."
        ),
    )

    previews_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(pointclouds_dir / f"{stem}_payload_mask.png"),
        mask_full.astype(np.uint8) * 255,
    )
    overlay_path = previews_dir / f"{stem}_segmentation.jpg"
    cv2.imwrite(str(overlay_path), render_payload_overlay(frame, mask_full), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    contour_path = previews_dir / f"{stem}_contour.jpg"
    cv2.imwrite(str(contour_path), render_mask_border_overlay(frame, mask_full), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    cloud3d_payload_path = previews_dir / f"{stem}_pointcloud_3d_payload.jpg"
    cloud3d_scene_path = previews_dir / f"{stem}_pointcloud_3d_scene.jpg"
    cloud3d_contrast_path = previews_dir / f"{stem}_pointcloud_3d_contrast.jpg"
    render_point_cloud_views(pts, colors, cloud3d_payload_path, suptitle="Payload only")
    render_point_cloud_views(pts_scene, colors_scene, cloud3d_scene_path, suptitle="Tank scene")
    compose_3d_contrast_panel(pts, colors, pts_scene, colors_scene, cloud3d_contrast_path)
    showcase_path = previews_dir / f"{stem}_showcase.jpg"
    compose_showcase_panel(
        frame,
        mask_full,
        depth,
        pts,
        colors,
        showcase_path,
        scene_points=pts_scene,
        scene_colors=colors_scene,
    )

    return {
        "preview": f"previews/{showcase_path.name}",
        "showcase": f"previews/{showcase_path.name}",
        "segmentation_preview": f"previews/{overlay_path.name}",
        "contour_preview": f"previews/{contour_path.name}",
        "pointcloud_3d_preview": f"previews/{cloud3d_payload_path.name}",
        "pointcloud_3d_payload": f"previews/{cloud3d_payload_path.name}",
        "pointcloud_3d_scene": f"previews/{cloud3d_scene_path.name}",
        "pointcloud_3d_contrast": f"previews/{cloud3d_contrast_path.name}",
        "pointcloud_ply": f"pointclouds/{ply_path.name}",
        "pointcloud_ply_scene": f"pointclouds/{ply_scene_path.name}",
        "point_count_scene": int(pts_scene.shape[0]),
        "pointcloud_depth_npy": f"pointclouds/{stem}_depth.npy",
        "pointcloud_intrinsics": f"pointclouds/{stem}_intrinsics.json",
        "point_count": int(pts.shape[0]),
        "intrinsics_model": intrinsics.model,
        "mask_mode": mask_used,
        "mask_pixels": int(seg_result.pixel_count),
        "mask_coverage": float(seg_result.coverage_frac),
        "seed_found": bool(seg_result.seed_found),
        "payload_mask": f"pointclouds/{stem}_payload_mask.png",
        "plane_fit_rms": plane_rms,
        "aspect_mode": intrinsics.aspect_mode,
        "depth_semantics": intrinsics.depth_semantics,
    }


def write_results_index(run_dir: Path, manifest: dict) -> Path:
    rows = []
    for item in manifest.get("outputs", []):
        rel = item.get("showcase") or item.get("preview", "")
        name = item["source"]
        chips = []
        if item.get("point_count"):
            chips.append(f'<span class="chip">{item["point_count"]:,} points</span>')
        if item.get("mask_mode"):
            chips.append(f'<span class="chip">{item["mask_mode"]}</span>')
        chip_line = f'<div class="chips">{"".join(chips)}</div>' if chips else ""
        links = []
        if item.get("pointcloud_ply"):
            links.append(f'<a href="{item["pointcloud_ply"]}">Payload PLY</a>')
        if item.get("pointcloud_ply_scene"):
            links.append(f'<a href="{item["pointcloud_ply_scene"]}">Scene PLY</a>')
        if item.get("pointcloud_3d_preview"):
            links.append(f'<a href="{item["pointcloud_3d_preview"]}">3D snapshot</a>')
        link_line = f'<p class="links">{" · ".join(links)}</p>' if links else ""
        if item.get("type") == "image":
            rows.append(
                f'<article class="card"><header><h2>{name}</h2>{chip_line}</header>'
                f'<a class="hero" href="{rel}"><img src="{rel}" alt="{name}"/></a>{link_line}</article>'
            )
        else:
            rows.append(f'<article class="card"><header><h2>{name}</h2></header>{link_line}</article>')

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CETI Point Cloud — {manifest.get("run_id", "")}</title>
<style>
:root {{ --bg:#0b1016; --panel:#141d29; --text:#edf2f7; --muted:#9fb0c4; --link:#7eb8ff; }}
body {{ margin:0; font-family:system-ui,sans-serif; background:var(--bg); color:var(--text); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:2rem 1.25rem; }}
.card {{ background:var(--panel); border-radius:14px; padding:1rem; margin:1.25rem 0; }}
.hero img {{ width:100%; border-radius:10px; margin-top:0.75rem; }}
a {{ color:var(--link); }}
</style></head><body><div class="wrap">
<h1>CETI Point Cloud Results</h1>
<p style="color:var(--muted)">Run {manifest.get("run_id","")} · {manifest.get("device","")}</p>
{"".join(rows)}
</div></body></html>"""
    index = run_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def run_pipeline(
    inputs: list[Path],
    *,
    output_root: Path | None = None,
    checkpoint: Path | None = None,
    underwater_preprocess: bool = True,
    copy_inputs: bool = False,
    tank_roi: bool = True,
) -> dict:
    from ceti.depth.web_export import write_web_api_manifest
    from ceti.utils.device import configure_compute, device_name, get_device

    images, videos = discover_media(inputs)
    images = [p.resolve() for p in images]
    videos = [p.resolve() for p in videos]
    if not images and not videos:
        raise ValueError("No JPEG/PNG/MP4 files found in inputs.")

    configure_compute()
    device = str(get_device())
    ckpt = Path(checkpoint or DEFAULT_CKPT)
    if not ckpt.is_file():
        raise FileNotFoundError(f"CETI checkpoint not found: {ckpt}\nRun: bash ceti/scripts/download_checkpoint.sh")

    print(f"  Loading model ({ckpt.name})…")
    model, transform, enc = _load_models(ckpt, device)

    run_id = _utc_run_id()
    run_dir = (output_root or INBOX_RESULTS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = run_dir / "previews"
    pointclouds_dir = run_dir / "pointclouds"
    previews_dir.mkdir(parents=True, exist_ok=True)
    pointclouds_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name(get_device()),
        "depth_mode": "pointcloud",
        "description": "CETI relative depth → colored 3D point cloud (DJI Osmo Action 4 intrinsics).",
        "underwater_preprocess": underwater_preprocess,
        "tank_roi": tank_roi,
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "encoder": enc,
        "camera": "DJI Osmo Action 4",
        "intrinsics_config": "ceti/configs/dji_action4.yaml",
        "outputs": [],
    }
    if tank_roi:
        margins = _tank_roi_margins()
        from ceti.preprocessing.tank_roi import DEFAULT_MARGINS

        manifest["tank_roi_margins"] = list(margins) if margins else list(DEFAULT_MARGINS)

    if copy_inputs:
        inputs_dir = run_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for p in images + videos:
            dest = inputs_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)

    print(f"CETI point cloud pipeline — run {run_id}")
    print(f"  Device: {manifest['device']}")
    print(f"  Images: {len(images)}  Videos: {len(videos)}")
    print(f"  Output: {run_dir}")

    for img in images:
        print(f"  → {img.name}")
        out_paths = _process_image_pointcloud(
            img,
            previews_dir,
            pointclouds_dir,
            model,
            transform,
            device,
            underwater_preprocess=underwater_preprocess,
            tank_roi=tank_roi,
        )
        manifest["outputs"].append({"source": img.name, "type": "image", "kind": "pointcloud", **out_paths})

    for vid in videos:
        print(f"  → {vid.name} (first frame)")
        cap = cv2.VideoCapture(str(vid))
        ok, frame = cap.read()
        cap.release()
        pc_out: dict = {}
        if ok:
            tmp = run_dir / f"_frame_{vid.stem}.jpg"
            cv2.imwrite(str(tmp), frame)
            pc_out = _process_image_pointcloud(
                tmp,
                previews_dir,
                pointclouds_dir,
                model,
                transform,
                device,
                underwater_preprocess=underwater_preprocess,
                tank_roi=tank_roi,
            )
            tmp.unlink(missing_ok=True)
        manifest["outputs"].append({"source": vid.name, "type": "video", "kind": "pointcloud", **pc_out})

    web_assets = []
    for item in manifest["outputs"]:
        web_assets.append({
            k: v
            for k, v in {
                "source": item["source"],
                "type": item["type"],
                "kind": "pointcloud",
                "preview": item.get("preview"),
                "pointcloud_ply": item.get("pointcloud_ply"),
                "pointcloud_ply_scene": item.get("pointcloud_ply_scene"),
                "pointcloud_intrinsics": item.get("pointcloud_intrinsics"),
                "point_count": item.get("point_count"),
                "point_count_scene": item.get("point_count_scene"),
            }.items()
            if v is not None
        })
    write_web_api_manifest(run_dir, web_assets)
    manifest["web_api"] = "web_api.json"

    manifest_path = run_dir / "manifest.json"
    index_path = write_results_index(run_dir, manifest)
    manifest["index_html"] = str(index_path.relative_to(REPO_ROOT))
    manifest["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone: {run_dir}")
    print(f"  open {index_path}")
    return manifest


def run_inbox_drop_folder(
    upload_dir: Path | None = None,
    *,
    move_processed: bool = False,
) -> dict:
    upload_dir = upload_dir or INBOX_UPLOADS
    upload_dir.mkdir(parents=True, exist_ok=True)
    INBOX_RESULTS.mkdir(parents=True, exist_ok=True)

    media = sorted(
        p
        for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    )
    if not media:
        raise FileNotFoundError(f"No media in {upload_dir}. Drop .jpg / .png files there first.")

    manifest = run_pipeline(media, copy_inputs=True)
    if move_processed:
        done = upload_dir / "_processed" / manifest["run_id"]
        done.mkdir(parents=True, exist_ok=True)
        for p in media:
            shutil.move(str(p), str(done / p.name))
    return manifest
