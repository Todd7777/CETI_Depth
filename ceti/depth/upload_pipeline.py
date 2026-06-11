"""
Batch depth inference for lab uploads (JPEG, PNG, MP4, …).

Used by:
  - ceti/scripts/run_upload_pipeline.sh
  - ceti/scripts/ceti_depth_portal_web.py
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"
DEFAULT_METRIC_CKPT = REPO_ROOT / "checkpoints/depth_anything_metric_depth_outdoor.pt"
INBOX_UPLOADS = REPO_ROOT / "ceti/inbox/uploads"
INBOX_RESULTS = REPO_ROOT / "ceti/inbox/results"
DEPTH_MODES = ("pointcloud", "relative", "accurate", "metric", "both")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def discover_media(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Expand directories; return (images, videos) sorted by name."""
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


def _prepare_tank_frame(
    bgr: np.ndarray,
    *,
    tank_roi: bool,
) -> tuple[np.ndarray, np.ndarray, object | None, np.ndarray | None]:
    """
    Returns (bgr_for_inference, full_bgr, roi, subject_mask_on_crop).
    """
    from ceti.preprocessing.tank_roi import (
        build_subject_mask,
        crop_to_roi,
        load_roi_preset,
        resolve_tank_roi,
    )

    preset = load_roi_preset() or {}
    margins = _tank_roi_margins()
    roi = resolve_tank_roi(bgr, enabled=tank_roi, margins=margins)
    if roi is None:
        return bgr, bgr, None, None
    infer = crop_to_roi(bgr, roi)
    mask_full = build_subject_mask(
        bgr,
        roi,
        exclude_bright=bool(preset.get("exclude_bright", True)),
        center_fraction=float(preset.get("center_fraction", 0.72)),
        preset=preset,
    )
    mask_crop = mask_full[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    return infer, bgr, roi, mask_crop


def _depth_to_full_frame(
    depth_crop: np.ndarray,
    roi: object | None,
    full_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    from ceti.preprocessing.tank_roi import build_subject_mask, paste_crop_to_canvas

    if roi is None:
        return depth_crop, None
    full_h, full_w = full_shape[:2]
    full_depth = paste_crop_to_canvas(depth_crop, roi, (full_h, full_w))
    # mask for export stats on full frame
    return full_depth, None


def _load_models(checkpoint: Path, device: str):
    from ceti.depth.infer_robot import build_depth_model, resolve_encoder

    enc = resolve_encoder(checkpoint)
    model, transform = build_depth_model(enc, device, str(checkpoint))
    return model, transform, enc


def process_image_file(
    image_path: Path,
    out_dir: Path,
    model,
    transform,
    device: str,
    *,
    underwater_preprocess: bool = True,
    preprocess_method: str = "combined",
    tank_roi: bool = True,
) -> Path:
    from ceti.depth.infer_robot import process_frame

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    infer, full, roi, _ = _prepare_tank_frame(frame, tank_roi=tank_roi)
    vis, _ = process_frame(
        infer,
        model,
        transform,
        None,
        device,
        underwater_preprocess,
        preprocess_method,
        0.5,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_rgb_depth.jpg"
    if roi is not None:
        from ceti.preprocessing.tank_roi import compose_tank_preview

        vis = compose_tank_preview(full, roi, vis)
    cv2.imwrite(str(out_path), vis)
    return out_path


def process_video_file(
    video_path: Path,
    out_dir: Path,
    model,
    transform,
    device: str,
    *,
    underwater_preprocess: bool = True,
    preprocess_method: str = "combined",
) -> tuple[Path, int, float]:
    from ceti.depth.infer_robot import process_frame

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_rgb_depth.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, _ = process_frame(
            frame,
            model,
            transform,
            None,
            device,
            underwater_preprocess,
            preprocess_method,
            0.5,
        )
        writer.write(vis)
        n += 1

    elapsed = time.time() - t0
    cap.release()
    writer.release()
    fps_out = n / elapsed if elapsed > 0 else 0.0
    return out_path, n, fps_out


def process_image_metric(
    image_path: Path,
    out_dir: Path,
    *,
    metric_checkpoint: Path | None = None,
    underwater_preprocess: bool = True,
) -> tuple[Path, Path, Path]:
    from ceti.depth.metric_infer import (
        predict_metric_depth_meters,
        visualize_metric_panel,
        write_depth_readme,
    )

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    depth_m, _proc = predict_metric_depth_meters(
        frame,
        checkpoint=metric_checkpoint,
        underwater_preprocess=underwater_preprocess,
    )
    panel = visualize_metric_panel(_proc, depth_m)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / f"{image_path.stem}_metric_meters.jpg"
    npy_path = out_dir / f"{image_path.stem}_depth_meters.npy"
    txt_path = out_dir / f"{image_path.stem}_depth_readme.txt"
    cv2.imwrite(str(panel_path), panel)
    np.save(npy_path, depth_m)
    ckpt_name = str(metric_checkpoint or DEFAULT_METRIC_CKPT)
    write_depth_readme(txt_path, depth_m=depth_m, source_name=image_path.name, checkpoint=ckpt_name)
    return panel_path, npy_path, txt_path


def process_video_metric(
    video_path: Path,
    out_dir: Path,
    *,
    metric_checkpoint: Path | None = None,
    underwater_preprocess: bool = True,
) -> tuple[Path, int, float]:
    from ceti.depth.metric_infer import predict_metric_depth_meters, visualize_metric_panel

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_metric_meters.mp4"

    n = 0
    t0 = time.time()
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        depth_m, proc = predict_metric_depth_meters(
            frame,
            checkpoint=metric_checkpoint,
            underwater_preprocess=underwater_preprocess,
        )
        panel = visualize_metric_panel(proc, depth_m)
        if writer is None:
            ph, pw = panel.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)
            )
        writer.write(panel)
        n += 1

    elapsed = time.time() - t0
    cap.release()
    if writer is not None:
        writer.release()
    fps_out = n / elapsed if elapsed > 0 else 0.0
    return out_path, n, fps_out


def _pointcloud_settings() -> dict:
    import os

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
    import os

    from ceti.depth.infer_robot import predict_depth_raw
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
        build_payload_mask,
        render_mask_border_overlay,
        render_payload_overlay,
        segment_ceti_payload,
    )
    from ceti.preprocessing.tank_roi import (
        build_scene_contrast_mask,
        load_roi_preset,
        resolve_tank_roi,
    )
    from ceti.preprocessing.underwater import preprocess_underwater

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    h, w = frame.shape[:2]
    preset = load_roi_preset() or {}
    roi = None
    if tank_roi:
        roi = resolve_tank_roi(frame, enabled=True, margins=_tank_roi_margins())

    proc = (
        preprocess_underwater(frame, method="combined")
        if underwater_preprocess
        else frame.copy()
    )
    depth = predict_depth_raw(rel_model, rel_transform, proc, device)
    seg_result = segment_ceti_payload(frame, depth, preset=preset)
    if seg_result is None:
        mask_full, mask_used = None, "payload_failed"
    else:
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
    if mask_full is None or pts.shape[0] == 0:
        raise RuntimeError(
            f"Payload segmentation produced zero points for {image_path.name}. "
            "Adjust subject_zone or payload_segmentation in the tank ROI preset."
        )

    scene_sub = adaptive_scene_subsample(h, w, settings["subsample"])
    scene_mask = build_scene_contrast_mask(
        frame, depth, roi, preset=preset, payload_mask=mask_full
    )
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
            "Depth is CETI RELATIVE units (not meters). Points are payload-only. "
            "Geometry is qualitative: monocular depth + approximate intrinsics. "
            "See ceti/docs/POINTCLOUD_ACCURACY.md."
        ),
    )

    previews_dir.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask_full.astype(np.uint8) * 255)
    cv2.imwrite(str(pointclouds_dir / f"{stem}_payload_mask.png"), mask_u8)

    overlay_path = previews_dir / f"{stem}_segmentation.jpg"
    cv2.imwrite(
        str(overlay_path),
        render_payload_overlay(frame, mask_full),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    contour_path = previews_dir / f"{stem}_contour.jpg"
    cv2.imwrite(
        str(contour_path),
        render_mask_border_overlay(frame, mask_full),
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    cloud3d_payload_path = previews_dir / f"{stem}_pointcloud_3d_payload.jpg"
    cloud3d_scene_path = previews_dir / f"{stem}_pointcloud_3d_scene.jpg"
    cloud3d_contrast_path = previews_dir / f"{stem}_pointcloud_3d_contrast.jpg"
    render_point_cloud_views(pts, colors, cloud3d_payload_path, suptitle="Payload only")
    render_point_cloud_views(pts_scene, colors_scene, cloud3d_scene_path, suptitle="Tank scene (payload + interior)")
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
    preview_path = previews_dir / f"{stem}_preview.jpg"
    cv2.imwrite(str(preview_path), cv2.imread(str(showcase_path)), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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


def _process_image_accurate(
    image_path: Path,
    previews_dir: Path,
    distances_dir: Path,
    rel_model,
    rel_transform,
    device: str,
    metric_runner,
    *,
    underwater_preprocess: bool = True,
    reference_center_m: float | None = None,
    tank_roi: bool = True,
) -> dict:
    from ceti.depth.accurate_depth import (
        predict_accurate_depth_meters,
        visualize_accurate_panel,
    )
    from ceti.depth.depth_export import export_distance_files
    from ceti.preprocessing.tank_roi import build_subject_mask, compose_tank_preview

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)

    infer, full, roi, mask_crop = _prepare_tank_frame(frame, tank_roi=tank_roi)
    depth_m, proc, info = predict_accurate_depth_meters(
        infer,
        rel_model,
        rel_transform,
        device,
        metric_runner,
        underwater_preprocess=underwater_preprocess,
        reference_center_m=reference_center_m,
        sample_mask=mask_crop,
        metric_bgr=infer,
    )
    depth_export, _ = _depth_to_full_frame(depth_m, roi, full.shape)
    subject_mask = (
        build_subject_mask(full, roi) if roi is not None else mask_crop
    )
    stem = image_path.stem
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"{stem}_preview.jpg"
    panel = visualize_accurate_panel(proc, depth_m, info)
    preview_img = compose_tank_preview(full, roi, panel) if roi else panel
    cv2.imwrite(str(preview_path), preview_img)

    dist_files = export_distance_files(
        distances_dir,
        stem,
        depth_export,
        source_file=image_path.name,
        mode="accurate",
        alignment={
            "scale": info.get("scale"),
            "shift": info.get("shift"),
            "inverted_relative": info.get("inverted_relative"),
            "calibration_factor": info.get("calibration_factor"),
        },
        roi=roi.to_dict() if roi is not None else None,
        subject_mask=subject_mask,
    )
    return {
        "preview": f"previews/{preview_path.name}",
        "distance_json": f"distances/{dist_files['distance_json']}",
        "distance_csv": f"distances/{dist_files['distance_csv']}",
        "distance_npy": f"distances/{dist_files['distance_npy']}",
        "center_m": info.get("center_m"),
    }


def _process_video_accurate(
    video_path: Path,
    previews_dir: Path,
    distances_dir: Path,
    rel_model,
    rel_transform,
    device: str,
    metric_runner,
    *,
    underwater_preprocess: bool = True,
    reference_center_m: float | None = None,
    tank_roi: bool = True,
) -> dict:
    from ceti.depth.accurate_depth import predict_accurate_depth_meters, visualize_accurate_panel
    from ceti.depth.depth_export import (
        append_video_frame_record,
        export_distance_files,
        write_video_depth_json,
    )
    from ceti.preprocessing.tank_roi import compose_tank_preview

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    stem = video_path.stem
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"{stem}_preview.mp4"
    import os

    export_per_frame = os.environ.get("CETI_EXPORT_VIDEO_FRAMES", "").strip() in ("1", "true", "yes")
    frame_records: list[dict] = []
    n = 0
    t0 = time.time()
    writer = None
    fixed_roi = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        infer, full, roi, mask_crop = _prepare_tank_frame(frame, tank_roi=tank_roi)
        if tank_roi and fixed_roi is None and roi is not None:
            fixed_roi = roi
        if fixed_roi is not None:
            from ceti.preprocessing.tank_roi import build_subject_mask, crop_to_roi

            infer = crop_to_roi(frame, fixed_roi)
            mask_crop = build_subject_mask(frame, fixed_roi)[
                fixed_roi.y : fixed_roi.y + fixed_roi.h,
                fixed_roi.x : fixed_roi.x + fixed_roi.w,
            ]
            roi = fixed_roi
            full = frame

        depth_m, proc, info = predict_accurate_depth_meters(
            infer,
            rel_model,
            rel_transform,
            device,
            metric_runner,
            underwater_preprocess=underwater_preprocess,
            reference_center_m=reference_center_m,
            sample_mask=mask_crop,
            metric_bgr=infer,
        )
        panel = visualize_accurate_panel(proc, depth_m, info)
        panel = compose_tank_preview(full, roi, panel) if roi else panel
        if writer is None:
            ph, pw = panel.shape[:2]
            writer = cv2.VideoWriter(
                str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)
            )
        writer.write(panel)
        append_video_frame_record(frame_records, n, depth_m, info, subject_mask=mask_crop)
        if export_per_frame:
            export_distance_files(
                distances_dir / stem,
                f"frame_{n:05d}",
                depth_m,
                source_file=f"{video_path.name}#frame={n}",
                mode="accurate",
                alignment={
                    "scale": info.get("scale"),
                    "shift": info.get("shift"),
                    "inverted_relative": info.get("inverted_relative"),
                },
            )
        n += 1

    elapsed = time.time() - t0
    cap.release()
    if writer is not None:
        writer.release()
    fps_out = n / elapsed if elapsed > 0 else 0.0

    distances_dir.mkdir(parents=True, exist_ok=True)
    write_video_depth_json(
        distances_dir / f"{stem}.json",
        source_file=video_path.name,
        fps=fps,
        frames=frame_records,
        mode="accurate",
    )

    csv_path = distances_dir / f"{stem}_frames.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame,center_m,median_m,min_m,max_m\n")
        for rec in frame_records:
            f.write(
                f"{rec['frame']},{rec['center_m']:.6f},{rec['median_m']:.6f},"
                f"{rec['min_m']:.6f},{rec['max_m']:.6f}\n"
            )

    out: dict = {
        "preview": f"previews/{preview_path.name}",
        "distance_json": f"distances/{stem}.json",
        "distance_csv": f"distances/{stem}_frames.csv",
        "frames": n,
        "fps": round(fps_out, 2),
    }
    if export_per_frame:
        out["distance_frames_dir"] = f"distances/{stem}/"
    return out


def write_results_index(run_dir: Path, manifest: dict) -> Path:
    """HTML gallery for browser presentation."""
    is_pointcloud = manifest.get("depth_mode") == "pointcloud"
    rows = []
    for item in manifest.get("outputs", []):
        rel = item.get("showcase") or item.get("preview") or item.get("file", "")
        name = item["source"]
        kind = item["type"]
        chips = []
        if item.get("point_count"):
            chips.append(f'<span class="chip">{item["point_count"]:,} points</span>')
        if item.get("mask_mode"):
            chips.append(f'<span class="chip">{item["mask_mode"]}</span>')
        if item.get("intrinsics_model"):
            chips.append(f'<span class="chip">{item["intrinsics_model"]}</span>')
        if item.get("depth_semantics"):
            chips.append(f'<span class="chip">{item["depth_semantics"]}</span>')
        if item.get("aspect_mode"):
            chips.append(f'<span class="chip">{item["aspect_mode"]}</span>')
        if item.get("plane_fit_rms") is not None:
            chips.append(f'<span class="chip">RMS {item["plane_fit_rms"]:.2f}</span>')
        chip_line = f'<div class="chips">{"".join(chips)}</div>' if chips else ""
        links = []
        if item.get("pointcloud_ply"):
            links.append(f'<a href="{item["pointcloud_ply"]}">Download PLY</a>')
        if item.get("payload_mask"):
            links.append(f'<a href="{item["payload_mask"]}">Segmentation mask</a>')
        if item.get("pointcloud_intrinsics"):
            links.append(f'<a href="{item["pointcloud_intrinsics"]}">Camera intrinsics</a>')
        if item.get("pointcloud_3d_preview"):
            links.append(f'<a href="{item["pointcloud_3d_preview"]}">3D snapshot</a>')
        if item.get("distance_json"):
            links.append(f'<a href="{item["distance_json"]}">distances.json</a>')
        link_line = (
            f'<p class="links">{" · ".join(links)}</p>'
            if links
            else ""
        )
        if kind == "image":
            rows.append(
                f'<article class="card"><header><h2>{name}</h2>{chip_line}</header>'
                f'<a class="hero" href="{rel}"><img src="{rel}" alt="{name} showcase"/></a>'
                f"{link_line}</article>"
            )
        else:
            rows.append(
                f'<article class="card"><header><h2>{name}</h2></header>{link_line}'
                f'<video controls width="100%" src="{rel}"></video></article>'
            )

    api_link = ""
    if manifest.get("web_api"):
        api_link = (
            f'<p class="meta">API index: <a href="{manifest["web_api"]}">web_api.json</a></p>'
        )
    subtitle = (
        "Payload-only 3D reconstruction · DJI Osmo Action 4 intrinsics · CETI fine-tuned depth"
        if is_pointcloud
        else manifest.get("description", "")
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CETI Point Cloud — {manifest.get("run_id", "")}</title>
<style>
:root {{
  --bg: #0b1016; --panel: #141d29; --text: #edf2f7; --muted: #9fb0c4; --accent: #5fd38d; --link: #7eb8ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  background: radial-gradient(1200px 600px at 10% -10%, #1a2b3f 0%, var(--bg) 55%);
  color: var(--text); line-height: 1.45;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }}
.hero-box {{
  background: linear-gradient(135deg, rgba(95,211,141,0.12), rgba(126,184,255,0.08));
  border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 1.4rem 1.5rem; margin-bottom: 1.5rem;
}}
h1 {{ margin: 0 0 0.35rem; font-size: 1.75rem; letter-spacing: -0.02em; }}
.lead {{ margin: 0; color: var(--muted); max-width: 70ch; }}
.meta {{ color: var(--muted); font-size: 0.92rem; margin: 0.75rem 0 0; }}
.card {{
  background: var(--panel); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px; padding: 1rem 1rem 1.1rem; margin: 1.25rem 0;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}}
.card h2 {{ margin: 0; font-size: 1.05rem; font-weight: 600; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.55rem; }}
.chip {{
  font-size: 0.78rem; color: #d8ffe7; background: rgba(95,211,141,0.14);
  border: 1px solid rgba(95,211,141,0.35); border-radius: 999px; padding: 0.2rem 0.65rem;
}}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.hero img {{ width: 100%; border-radius: 10px; margin-top: 0.85rem; display: block; }}
.links {{ margin: 0.75rem 0 0; color: var(--muted); font-size: 0.92rem; }}
</style></head><body>
<div class="wrap">
  <section class="hero-box">
    <h1>CETI Underwater Point Cloud</h1>
    <p class="lead">{subtitle}</p>
    <p class="meta">Run {manifest.get("run_id", "")} · Device {manifest.get("device", "")} · Mode {manifest.get("depth_mode", "")}</p>
    {api_link}
    <p class="meta">Checkpoint: {manifest.get("checkpoint", "n/a")}</p>
  </section>
  {"".join(rows)}
</div>
</body></html>"""
    index = run_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def run_pipeline(
    inputs: list[Path],
    *,
    output_root: Path | None = None,
    checkpoint: Path | None = None,
    metric_checkpoint: Path | None = None,
    depth_mode: str = "pointcloud",
    underwater_preprocess: bool = True,
    copy_inputs: bool = False,
    reference_center_m: float | None = None,
    tank_roi: bool = True,
) -> dict:
    """
    Process all images/videos; write timestamped folder under inbox/results.

    depth_mode:
      - pointcloud: CETI relative depth → 3D PLY (DJI Action 4 intrinsics) — default
      - relative: RGB|depth preview only
      - accurate: metric alignment (legacy)
      - metric: outdoor ZoeDepth only

    Returns manifest dict with paths and stats.
    """
    from ceti.utils.device import configure_compute, device_name, get_device

    depth_mode = depth_mode.lower().strip()
    if depth_mode == "both":
        depth_mode = "accurate"
    if depth_mode not in DEPTH_MODES:
        raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")

    images, videos = discover_media(inputs)
    images = [p.resolve() for p in images]
    videos = [p.resolve() for p in videos]
    if not images and not videos:
        raise ValueError("No JPEG/PNG/MP4 (etc.) files found in inputs.")

    configure_compute()
    device = str(get_device())

    ckpt = Path(checkpoint or DEFAULT_CKPT)
    metric_ckpt = Path(metric_checkpoint or DEFAULT_METRIC_CKPT)
    if depth_mode in ("relative", "accurate", "pointcloud") and not ckpt.is_file():
        raise FileNotFoundError(f"CETI relative checkpoint not found: {ckpt}")
    if depth_mode in ("metric", "accurate") and not metric_ckpt.is_file():
        raise FileNotFoundError(
            f"Metric checkpoint not found: {metric_ckpt}\n"
            "Run: bash ceti/scripts/download_checkpoints.sh"
        )

    model = transform = enc = None
    if depth_mode in ("relative", "accurate", "pointcloud"):
        print(f"  Loading CETI fine-tuned model ({ckpt.name})…")
        model, transform, enc = _load_models(ckpt, device)

    metric_runner = None
    if depth_mode in ("metric", "accurate"):
        from ceti.depth.metric_infer import get_metric_runner

        print(f"  Loading metric scale model (meters)…")
        metric_runner = get_metric_runner(metric_ckpt)

    run_id = _utc_run_id()
    run_dir = (output_root or INBOX_RESULTS) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = run_dir / "inputs"
    previews_dir = run_dir / "previews"
    pointclouds_dir = run_dir / "pointclouds"
    distances_dir = run_dir / "distances"
    legacy_outputs_dir = run_dir / "outputs"
    previews_dir.mkdir(parents=True, exist_ok=True)
    pointclouds_dir.mkdir(parents=True, exist_ok=True)

    desc = {
        "pointcloud": "CETI relative depth → colored 3D point cloud (DJI Osmo Action 4 intrinsics).",
        "accurate": "CETI underwater structure aligned to metric scale — depth in METERS.",
        "relative": "Colormap preview only.",
        "metric": "Outdoor metric prior only (often poor underwater).",
    }[depth_mode]
    desc_extra = ""

    manifest: dict = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name(get_device()),
        "depth_mode": depth_mode,
        "description": desc,
        "underwater_preprocess": underwater_preprocess,
        "tank_roi": tank_roi,
        "outputs": [],
    }
    if tank_roi:
        margins = _tank_roi_margins()
        from ceti.preprocessing.tank_roi import DEFAULT_MARGINS

        manifest["tank_roi_margins"] = list(margins) if margins else list(DEFAULT_MARGINS)

    desc_extra = " Tank ROI crop excludes window/edges." if tank_roi else ""
    manifest["description"] = desc + desc_extra
    if depth_mode in ("relative", "accurate", "pointcloud"):
        manifest["checkpoint"] = str(ckpt.relative_to(REPO_ROOT))
        manifest["encoder"] = enc
    if depth_mode == "pointcloud":
        import os

        manifest["camera"] = "DJI Osmo Action 4"
        manifest["unproject_model"] = os.environ.get("CETI_UNPROJECT_MODEL", "pinhole")
        manifest["depth_semantics"] = os.environ.get("CETI_DEPTH_SEMANTICS", "z_depth")
        manifest["aspect_mode"] = os.environ.get("CETI_ASPECT_MODE", "auto")
        manifest["intrinsics_config"] = "ceti/configs/dji_action4.yaml"
        manifest["accuracy_doc"] = "ceti/docs/POINTCLOUD_ACCURACY.md"
    if depth_mode in ("metric", "accurate"):
        manifest["metric_checkpoint"] = str(metric_ckpt.relative_to(REPO_ROOT))

    if copy_inputs:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for p in images + videos:
            dest = inputs_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)

    print(f"CETI depth pipeline — run {run_id}")
    print(f"  Device: {manifest['device']}")
    print(f"  Mode:   {depth_mode}")
    print(f"  Tank ROI crop: {tank_roi}")
    print(f"  Images: {len(images)}  Videos: {len(videos)}")
    print(f"  Output: {run_dir}")

    for img in images:
        print(f"  Image: {img.name}")
        if depth_mode == "pointcloud":
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
            manifest["outputs"].append(
                {"source": img.name, "type": "image", "kind": "pointcloud", **out_paths}
            )
        elif depth_mode == "accurate":
            out_paths = _process_image_accurate(
                img,
                previews_dir,
                distances_dir,
                model,
                transform,
                device,
                metric_runner,
                underwater_preprocess=underwater_preprocess,
                reference_center_m=reference_center_m,
                tank_roi=tank_roi,
            )
            manifest["outputs"].append(
                {"source": img.name, "type": "image", "kind": "accurate", **out_paths}
            )
        elif depth_mode == "relative":
            legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
            out = process_image_file(
                img,
                legacy_outputs_dir,
                model,
                transform,
                device,
                underwater_preprocess=underwater_preprocess,
                tank_roi=tank_roi,
            )
            manifest["outputs"].append(
                {
                    "source": img.name,
                    "type": "image",
                    "kind": "relative",
                    "file": str(out.relative_to(run_dir)),
                }
            )
        else:
            legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
            panel, npy, txt = process_image_metric(
                img,
                legacy_outputs_dir,
                metric_checkpoint=metric_ckpt,
                underwater_preprocess=underwater_preprocess,
            )
            manifest["outputs"].append(
                {
                    "source": img.name,
                    "type": "image",
                    "kind": "metric",
                    "file": str(panel.relative_to(run_dir)),
                    "depth_npy": str(npy.relative_to(run_dir)),
                    "readme": str(txt.relative_to(run_dir)),
                }
            )

    for vid in videos:
        print(f"  Video: {vid.name}")
        if depth_mode == "pointcloud":
            legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
            out, nframes, fps = process_video_file(
                vid,
                legacy_outputs_dir,
                model,
                transform,
                device,
                underwater_preprocess=underwater_preprocess,
                tank_roi=tank_roi,
            )
            cap = cv2.VideoCapture(str(vid))
            ok, frame = cap.read()
            cap.release()
            pc_out = {}
            if ok:
                tmp = run_dir / "_video_frame0.jpg"
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
            manifest["outputs"].append(
                {
                    "source": vid.name,
                    "type": "video",
                    "kind": "pointcloud",
                    "file": str(out.relative_to(run_dir)),
                    "frames": nframes,
                    "fps": round(fps, 2),
                    **pc_out,
                }
            )
        elif depth_mode == "accurate":
            out_paths = _process_video_accurate(
                vid,
                previews_dir,
                distances_dir,
                model,
                transform,
                device,
                metric_runner,
                underwater_preprocess=underwater_preprocess,
                reference_center_m=reference_center_m,
                tank_roi=tank_roi,
            )
            manifest["outputs"].append(
                {"source": vid.name, "type": "video", "kind": "accurate", **out_paths}
            )
        elif depth_mode == "relative":
            legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
            out, nframes, fps = process_video_file(
                vid,
                legacy_outputs_dir,
                model,
                transform,
                device,
                underwater_preprocess=underwater_preprocess,
                tank_roi=tank_roi,
            )
            manifest["outputs"].append(
                {
                    "source": vid.name,
                    "type": "video",
                    "kind": "relative",
                    "file": str(out.relative_to(run_dir)),
                    "frames": nframes,
                    "fps": round(fps, 2),
                }
            )
        else:
            legacy_outputs_dir.mkdir(parents=True, exist_ok=True)
            out, nframes, fps = process_video_metric(
                vid,
                legacy_outputs_dir,
                metric_checkpoint=metric_ckpt,
                underwater_preprocess=underwater_preprocess,
            )
            manifest["outputs"].append(
                {
                    "source": vid.name,
                    "type": "video",
                    "kind": "metric",
                    "file": str(out.relative_to(run_dir)),
                    "frames": nframes,
                    "fps": round(fps, 2),
                }
            )

    if depth_mode in ("accurate", "pointcloud"):
        from ceti.depth.depth_export import write_web_api_manifest

        web_assets = []
        for item in manifest["outputs"]:
            asset = {
                "source": item["source"],
                "type": item["type"],
                "kind": item.get("kind", depth_mode),
                "preview": item.get("preview"),
                "pointcloud_ply": item.get("pointcloud_ply"),
                "pointcloud_intrinsics": item.get("pointcloud_intrinsics"),
                "point_count": item.get("point_count"),
                "distance_json": item.get("distance_json"),
                "distance_csv": item.get("distance_csv"),
                "distance_npy": item.get("distance_npy"),
                "center_m": item.get("center_m"),
                "frames": item.get("frames"),
                "fps": item.get("fps"),
            }
            web_assets.append({k: v for k, v in asset.items() if v is not None})
        write_web_api_manifest(run_dir, web_assets)
        manifest["web_api"] = "web_api.json"

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    index_path = write_results_index(run_dir, manifest)
    manifest["index_html"] = str(index_path.relative_to(REPO_ROOT))
    manifest["run_dir"] = str(run_dir.relative_to(REPO_ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Open results:")
    print(f"  open {run_dir}")
    print(f"  open {index_path}")
    return manifest


def run_inbox_drop_folder(
    upload_dir: Path | None = None,
    *,
    move_processed: bool = False,
) -> dict:
    """Process every file in ceti/inbox/uploads/."""
    upload_dir = upload_dir or INBOX_UPLOADS
    upload_dir.mkdir(parents=True, exist_ok=True)
    INBOX_RESULTS.mkdir(parents=True, exist_ok=True)

    inputs = sorted(upload_dir.iterdir())
    media = [p for p in inputs if p.is_file() and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    if not media:
        raise FileNotFoundError(
            f"No media in {upload_dir}. Drop .jpg / .mp4 files there first."
        )

    import os

    mode = os.environ.get("CETI_DEPTH_MODE", "pointcloud").lower().strip()
    manifest = run_pipeline(media, copy_inputs=True, depth_mode=mode)

    if move_processed:
        done = upload_dir / "_processed" / manifest["run_id"]
        done.mkdir(parents=True, exist_ok=True)
        for p in media:
            shutil.move(str(p), str(done / p.name))

    return manifest
