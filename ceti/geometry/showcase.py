"""
Presentation-quality panels and 3D snapshots for CETI point clouds.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _fit_panel(img: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    scale = min(width / iw, height / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _label_panel(img: np.ndarray, title: str, *, bar_h: int = 36) -> np.ndarray:
    out = np.zeros((img.shape[0] + bar_h, img.shape[1], 3), dtype=np.uint8)
    out[bar_h:] = img
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (18, 24, 34), -1)
    cv2.putText(
        out,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 236, 245),
        1,
        cv2.LINE_AA,
    )
    return out


def depth_colormap_panel(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Depth visualization restricted to payload pixels."""
    d = depth.astype(np.float32).copy()
    valid = mask & np.isfinite(d)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin, vmax = np.percentile(d[valid], [2, 98])
    span = max(vmax - vmin, 1e-6)
    norm = np.clip((d - vmin) / span, 0.0, 1.0)
    vis = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(vis, colormap)
    colored[~valid] = (18, 24, 34)
    return colored


def render_point_cloud_views(
    points: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    *,
    max_points: int = 12000,
    figsize: tuple[float, float] = (10.0, 4.8),
    dpi: int = 150,
    suptitle: str = "",
) -> Path:
    """Save a 2-view matplotlib snapshot of the colored point cloud."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = points.shape[0]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        pts = points[idx].astype(np.float32)
        cols = colors[idx] / 255.0
    else:
        pts = points.astype(np.float32)
        cols = colors / 255.0

    # Center for presentation (shape preserved; scale unchanged).
    pts -= np.median(pts, axis=0)

    fig = plt.figure(figsize=figsize, facecolor="#0f1419")
    views = [
        (20, -58, "Perspective"),
        (0, 0, "Front"),
    ]
    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d", facecolor="#0f1419")
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=cols,
            s=1.2,
            linewidths=0,
            alpha=0.95,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, color="#e7ecf1", fontsize=11, pad=8)
        ax.set_axis_off()
        rng = np.ptp(pts, axis=0).max()
        mid = pts.mean(axis=0)
        if rng <= 0:
            rng = 1.0
        ax.set_xlim(mid[0] - rng / 2, mid[0] + rng / 2)
        ax.set_ylim(mid[1] - rng / 2, mid[1] + rng / 2)
        ax.set_zlim(mid[2] - rng / 2, mid[2] + rng / 2)

    if suptitle:
        fig.suptitle(suptitle, color="#9fb3c8", fontsize=12, y=0.98)
    fig.tight_layout(pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="#0f1419", bbox_inches="tight")
    plt.close(fig)
    return out_path


def compose_3d_contrast_panel(
    payload_pts: np.ndarray,
    payload_colors: np.ndarray,
    scene_pts: np.ndarray,
    scene_colors: np.ndarray,
    out_path: Path,
    *,
    cell_w: int = 512,
    cell_h: int = 320,
) -> Path:
    """Side-by-side 3D snapshots: payload-only vs full scene (background contrast)."""
    import tempfile

    payload_snap = out_path.with_suffix(".payload_tmp.png")
    scene_snap = out_path.with_suffix(".scene_tmp.png")
    try:
        render_point_cloud_views(
            payload_pts,
            payload_colors,
            payload_snap,
            suptitle="Payload only",
            figsize=(7.0, 3.4),
            dpi=130,
        )
        render_point_cloud_views(
            scene_pts,
            scene_colors,
            scene_snap,
            suptitle="Tank scene (payload + interior)",
            figsize=(7.0, 3.4),
            dpi=130,
        )
        payload_bgr = cv2.imread(str(payload_snap))
        scene_bgr = cv2.imread(str(scene_snap))
    finally:
        payload_snap.unlink(missing_ok=True)
        scene_snap.unlink(missing_ok=True)

    if payload_bgr is None:
        payload_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if scene_bgr is None:
        scene_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    left = _label_panel(_fit_panel(payload_bgr, cell_w, cell_h), "3D - Payload only")
    right = _label_panel(_fit_panel(scene_bgr, cell_w, cell_h), "3D - Tank scene")
    board = np.concatenate([left, right], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return out_path


def depth_delta_panel(
    depth_clean: np.ndarray,
    depth_noisy: np.ndarray,
    mask: np.ndarray,
    *,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Absolute depth change on payload (noisy − clean)."""
    valid = mask.astype(bool) & np.isfinite(depth_clean) & np.isfinite(depth_noisy)
    if not valid.any():
        return np.zeros((*depth_clean.shape[:2], 3), dtype=np.uint8)

    delta = np.zeros_like(depth_clean, dtype=np.float32)
    delta[valid] = np.abs(depth_noisy[valid] - depth_clean[valid])
    vmax = float(np.percentile(delta[valid], 98))
    vmax = max(vmax, 1e-6)
    norm = np.clip(delta / vmax, 0.0, 1.0)
    vis = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(vis, colormap)
    base = np.full((*depth_clean.shape[:2], 3), 18, dtype=np.uint8)
    base[valid] = colored[valid]
    return base


def compose_noise_research_row(
    bgr: np.ndarray,
    mask: np.ndarray,
    depth_clean: np.ndarray,
    depth_noisy: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    noise_pct: float,
    cell_w: int = 360,
    cell_h: int = 220,
) -> np.ndarray:
    """
    One noise level for researchers:
    segmentation | clean depth | noised depth | |Δdepth| | 3D
    """
    from ceti.preprocessing.payload_segmentation import render_payload_overlay

    seg = _fit_panel(render_payload_overlay(bgr, mask), cell_w, cell_h)
    clean = _fit_panel(depth_colormap_panel(depth_clean, mask), cell_w, cell_h)
    noisy = _fit_panel(depth_colormap_panel(depth_noisy, mask), cell_w, cell_h)
    delta = _fit_panel(depth_delta_panel(depth_clean, depth_noisy, mask), cell_w, cell_h)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        snap_path = Path(tmp.name)
    try:
        render_point_cloud_views(points, colors, snap_path, figsize=(7.2, 3.2), dpi=120)
        cloud_bgr = cv2.imread(str(snap_path))
    finally:
        snap_path.unlink(missing_ok=True)
    if cloud_bgr is None:
        cloud_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cloud = _fit_panel(cloud_bgr, cell_w, cell_h)

    panels = [
        _label_panel(seg, "Payload segmentation"),
        _label_panel(clean, "Clean depth (0%)"),
        _label_panel(noisy, f"Noised depth ({noise_pct:g}%)"),
        _label_panel(delta, f"|Δdepth| ({noise_pct:g}%)"),
        _label_panel(cloud, f"3D cloud ({noise_pct:g}%)"),
    ]
    return np.concatenate(panels, axis=1)


def compose_noise_research_board(
    bgr: np.ndarray,
    mask: np.ndarray,
    depth_clean: np.ndarray,
    level_data: list[dict],
    out_path: Path,
    *,
    cell_w: int = 300,
    cell_h: int = 200,
) -> Path:
    """
    Vertical stack of noise rows. Each level_data item:
      noise_pct, depth_noisy, points, colors
    """
    rows = []
    for item in level_data:
        row = compose_noise_research_row(
            bgr,
            mask,
            depth_clean,
            item["depth_noisy"],
            item["points"],
            item["colors"],
            noise_pct=float(item["noise_pct"]),
            cell_w=cell_w,
            cell_h=cell_h,
        )
        rows.append(row)
    board = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    return out_path


def compose_showcase_panel(
    bgr: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
    *,
    scene_points: np.ndarray | None = None,
    scene_colors: np.ndarray | None = None,
    cell_w: int = 512,
    cell_h: int = 320,
) -> Path:
    """Four-up board: RGB | segmentation | masked depth | dual 3D contrast."""
    from ceti.preprocessing.payload_segmentation import render_payload_overlay

    rgb_panel = _fit_panel(bgr, cell_w, cell_h)
    seg_panel = _fit_panel(render_payload_overlay(bgr, mask), cell_w, cell_h)
    depth_panel = _fit_panel(depth_colormap_panel(depth, mask), cell_w, cell_h)

    half_w = max(256, cell_w // 2)
    if scene_points is not None and scene_colors is not None:
        contrast_path = out_path.with_suffix(".contrast_tmp.jpg")
        compose_3d_contrast_panel(
            points,
            colors,
            scene_points,
            scene_colors,
            contrast_path,
            cell_w=half_w,
            cell_h=cell_h,
        )
        contrast_bgr = cv2.imread(str(contrast_path))
        contrast_path.unlink(missing_ok=True)
        if contrast_bgr is None:
            contrast_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cloud_panel = _fit_panel(contrast_bgr, cell_w, cell_h)
    else:
        snap_path = out_path.with_suffix(".3d_tmp.png")
        render_point_cloud_views(points, colors, snap_path, suptitle="Payload only")
        cloud_bgr = cv2.imread(str(snap_path))
        if cloud_bgr is None:
            cloud_bgr = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cloud_panel = _fit_panel(cloud_bgr, cell_w, cell_h)
        snap_path.unlink(missing_ok=True)

    panels = [
        _label_panel(rgb_panel, "Original (DJI Action 4)"),
        _label_panel(seg_panel, "Payload segmentation"),
        _label_panel(depth_panel, "CETI depth (payload only)"),
        _label_panel(cloud_panel, "3D contrast - payload vs full scene"),
    ]
    top = np.concatenate([panels[0], panels[1]], axis=1)
    bottom = np.concatenate([panels[2], panels[3]], axis=1)
    board = np.concatenate([top, bottom], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), board, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return out_path
