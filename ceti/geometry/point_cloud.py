"""
Depth map + camera intrinsics → colored 3D point cloud (PLY).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ceti.geometry.dji_action4 import CameraIntrinsics


def normalize_depth_for_cloud(depth: np.ndarray) -> np.ndarray:
    """Positive depth values; relative scale preserved."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if not valid.any():
        return d
    if np.nanmedian(d) < 0:
        d = -d
    d = np.maximum(d, 1e-6)
    return d


def plane_fit_rms(points: np.ndarray) -> float:
    """RMS residual to best-fit plane (shape diagnostic, not ground truth)."""
    if points.shape[0] < 10:
        return float("nan")
    pts = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return float("nan")
    center = np.median(pts, axis=0)
    centered = pts - center
    # Stabilize SVD for large relative-depth coordinates.
    scale = float(np.max(np.abs(centered)))
    if scale < 1e-9:
        return 0.0
    centered /= scale
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    try:
        _, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    normal = evecs[:, 0]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        return float("nan")
    normal /= nrm
    residuals = np.abs(centered @ normal) * scale
    return float(np.sqrt(np.mean(residuals**2)))


def unproject_pinhole(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: np.ndarray | None = None,
    depth_scale: float = 1.0,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard pinhole: depth = Z (distance along optical axis).
    Best matched to monocular networks (Depth Anything / CETI) in practice.
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    depth = normalize_depth_for_cloud(depth) * depth_scale

    ys = np.arange(0, h, subsample, dtype=np.float32)
    xs = np.arange(0, w, subsample, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth[::subsample, ::subsample]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask[::subsample, ::subsample].astype(bool)

    u = uu[valid]
    v = vv[valid]
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    return pts, valid


def unproject_fisheye_equidistant(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: np.ndarray | None = None,
    depth_scale: float = 1.0,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Equidistant fisheye unprojection.

    depth_semantics on intrinsics:
      - z_depth: network depth treated as Z (camera-axis distance); converts to range.
      - range_along_ray: depth is Euclidean range along the pixel ray (legacy).
    """
    h, w = depth.shape
    f = intrinsics.f_fisheye or intrinsics.fx
    cx, cy = intrinsics.cx, intrinsics.cy
    depth = normalize_depth_for_cloud(depth) * depth_scale
    semantics = getattr(intrinsics, "depth_semantics", "z_depth")

    ys = np.arange(0, h, subsample, dtype=np.float32)
    xs = np.arange(0, w, subsample, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    d = depth[::subsample, ::subsample]
    valid = np.isfinite(d) & (d > 0)
    if mask is not None:
        valid &= mask[::subsample, ::subsample].astype(bool)

    u = uu[valid]
    v = vv[valid]
    z_or_rng = d[valid]
    dx = u - cx
    dy = v - cy
    r = np.hypot(dx, dy)
    theta = np.where(r > 1e-6, r / f, 0.0)
    phi = np.arctan2(dy, dx)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    if semantics == "range_along_ray":
        rng = z_or_rng
    else:
        # z_depth: interpret network output as Z, recover range along ray.
        rng = z_or_rng / np.maximum(cos_t, 1e-3)

    pts = np.stack([sin_t * np.cos(phi) * rng, sin_t * np.sin(phi) * rng, cos_t * rng], axis=1)
    return pts, valid.astype(bool)


def _motor_seed_weights(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blue-motor seed weights for robust plane fitting (same hue gate as segmentation)."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([35, 28, 24], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma = np.abs(lab[:, :, 1].astype(np.int16) - 128) + np.abs(lab[:, :, 2].astype(np.int16) - 128)
    blue[chroma < 18] = 0
    return (blue > 0) & mask


def _depth_span(depth: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return 1.0
    lo, hi = np.percentile(depth[valid], [4, 96])
    return float(max(hi - lo, 1e-3))


def _payload_frame_ring(mask: np.ndarray, *, k: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Split mask into thin frame ring vs motor interior (never fill the void)."""
    import cv2

    eroded = cv2.erode(
        mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
        iterations=1,
    ).astype(bool)
    interior = eroded
    ring = mask & ~interior
    return ring, interior


def refine_payload_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    bgr: np.ndarray | None = None,
    *,
    shear_cap_frac: float = 0.12,
) -> np.ndarray:
    """
    Smooth payload depth with a tight frame-vs-motor cap.

    Keeps per-pixel variation (no voxel slabs) while preventing CETI depth ripples
    from becoming concentric shells or a detached floating frame in the viewer.
    """
    d = smooth_depth_bilateral(depth, diameter=7, sigma_space=5.0)
    if not mask.any():
        return d

    seed = _motor_seed_weights(bgr, mask) if bgr is not None else np.zeros_like(mask)
    ring, interior = _payload_frame_ring(mask)
    span = _depth_span(d, mask)

    motor_z = float(np.median(d[seed])) if seed.any() else float(np.median(d[interior])) if interior.any() else float(np.median(d[mask]))
    ring_z = float(np.median(d[ring])) if ring.any() else motor_z
    cap = span * shear_cap_frac
    ring_z = float(np.clip(ring_z, motor_z - cap, motor_z + cap))

    motor_body = (seed & mask) if seed.any() else interior
    if motor_body.any():
        blended = 0.88 * d[motor_body] + 0.12 * motor_z
        d[motor_body] = np.clip(blended, motor_z - cap * 0.5, motor_z + cap * 0.5)
    frame_only = ring & ~motor_body
    if frame_only.any():
        blended = 0.86 * d[frame_only] + 0.14 * ring_z
        d[frame_only] = np.clip(blended, motor_z - cap, motor_z + cap)
    return d


def refine_scene_depth(
    depth: np.ndarray,
    scene_mask: np.ndarray,
    payload_mask: np.ndarray,
    bgr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Tank scene depth: harmonized payload + lightly smoothed nearby tank surfaces.
    """
    d = smooth_depth_bilateral(depth, diameter=7, sigma_space=5.0)
    if payload_mask.any():
        payload_d = refine_payload_depth(depth, payload_mask, bgr)
        d[payload_mask] = payload_d[payload_mask]

    bg = scene_mask & ~payload_mask
    if not bg.any():
        return d

    rig_z = float(np.median(d[payload_mask])) if payload_mask.any() else float(np.median(d[bg]))
    span = _depth_span(d, payload_mask if payload_mask.any() else scene_mask)
    lo = rig_z - span * 0.18
    hi = rig_z + span * 0.22
    local = smooth_depth_bilateral(depth, diameter=5, sigma_space=4.0)[bg]
    d[bg] = np.clip(local, lo, hi)
    return d


def center_points(points: np.ndarray) -> np.ndarray:
    """Recenter cloud on its median — stable viewer framing without rotation."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts
    return pts - np.median(pts, axis=0)


def smooth_depth_bilateral(
    depth: np.ndarray,
    *,
    diameter: int = 7,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """Edge-preserving depth smooth — reduces ripple artifacts in scene clouds."""
    import cv2

    d = depth.astype(np.float32).copy()
    valid = np.isfinite(d)
    if not valid.any():
        return d
    lo, hi = np.percentile(d[valid], [1, 99])
    span = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    smooth_u8 = cv2.bilateralFilter(norm_u8, diameter, 25, sigma_space)
    out = smooth_u8.astype(np.float32) / 255.0 * span + lo
    out[~valid] = np.nan
    return out


def opencv_to_viewer_coords(points: np.ndarray) -> np.ndarray:
    """OpenCV camera (+Y down) → viewer (+Y up) for Three.js / matplotlib."""
    out = np.asarray(points, dtype=np.float32).copy()
    out[:, 1] *= -1.0
    return out


def statistical_outlier_filter(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    k: int = 16,
    std_ratio: float = 1.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove flying pixels / depth speckle (critical for scene contrast clouds)."""
    n = points.shape[0]
    if n < k + 2:
        return points, colors
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return points, colors

    tree = cKDTree(points.astype(np.float64))
    dists, _ = tree.query(points.astype(np.float64), k=k + 1)
    mean_dist = dists[:, 1:].mean(axis=1)
    med = float(np.median(mean_dist))
    std = float(np.std(mean_dist))
    thr = med + std_ratio * max(std, 1e-9)
    keep = mean_dist <= thr
    return points[keep], colors[keep]


def depth_rgb_to_point_cloud(
    depth: np.ndarray,
    bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    mask: np.ndarray | None = None,
    depth_scale: float = 1.0,
    subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (points N×3, colors N×3 RGB)."""
    if intrinsics.model == "fisheye_equidistant":
        pts, valid_idx = unproject_fisheye_equidistant(
            depth,
            intrinsics,
            mask=mask,
            depth_scale=depth_scale,
            subsample=subsample,
        )
    else:
        pts, valid_idx = unproject_pinhole(
            depth,
            intrinsics,
            mask=mask,
            depth_scale=depth_scale,
            subsample=subsample,
        )

    rgb = cv2_cvt(bgr)
    colors = rgb[::subsample, ::subsample][valid_idx]
    return pts, colors


def depth_rgb_to_scene_point_cloud(
    depth: np.ndarray,
    bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    scene_mask: np.ndarray,
    payload_mask: np.ndarray | None = None,
    *,
    depth_scale: float = 1.0,
    subsample: int = 3,
    smooth: bool = True,
    outlier_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tank scene cloud: payload rig + nearby tank interior (no window backdrop)."""
    payload_bool = (
        payload_mask.astype(bool)
        if payload_mask is not None
        else np.zeros(scene_mask.shape, dtype=bool)
    )
    if payload_bool.any() and smooth:
        d = refine_scene_depth(depth, scene_mask, payload_bool, bgr)
    elif smooth:
        d = smooth_depth_bilateral(depth)
    else:
        d = depth.astype(np.float32)

    h, w = depth.shape[:2]
    bg_mask = scene_mask & ~payload_bool
    bg_stride = min(max(int(subsample), 2), 3)
    payload_stride = 1
    payload_scene = scene_mask & payload_bool

    chunks_p: list[np.ndarray] = []
    chunks_c: list[np.ndarray] = []
    if payload_scene.any():
        pts_p, col_p = depth_rgb_to_point_cloud(
            d,
            bgr,
            intrinsics,
            mask=payload_scene,
            depth_scale=depth_scale,
            subsample=payload_stride,
        )
        if pts_p.shape[0] > 0:
            chunks_p.append(pts_p)
            chunks_c.append(col_p)
    if bg_mask.any():
        pts_b, col_b = depth_rgb_to_point_cloud(
            d,
            bgr,
            intrinsics,
            mask=bg_mask,
            depth_scale=depth_scale,
            subsample=bg_stride,
        )
        if pts_b.shape[0] > 0:
            chunks_p.append(pts_b)
            chunks_c.append(col_b)

    if not chunks_p:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    pts = np.vstack(chunks_p)
    colors = np.vstack(chunks_c)
    if outlier_filter and pts.shape[0] > 0:
        pts, colors = statistical_outlier_filter(pts, colors, k=12, std_ratio=1.6)
    if pts.shape[0] > 0:
        pts = center_points(pts)
        pts = opencv_to_viewer_coords(pts)
    return pts, colors


def adaptive_payload_subsample(h: int, w: int, mask_count: int, base: int = 1) -> int:
    """Prefer dense payload clouds — stride 1 unless the frame is enormous."""
    if (h * w) > 6_000_000:
        return max(base * 2, 2)
    return base


def depth_rgb_to_payload_point_cloud(
    depth: np.ndarray,
    bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    payload_mask: np.ndarray,
    *,
    depth_scale: float = 1.0,
    subsample: int | None = None,
    harmonize_depth: bool = True,
    outlier_filter: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Payload-only cloud: real depth structure, hollow frame preserved."""
    mask = payload_mask.astype(bool)
    h, w = depth.shape[:2]
    stride = subsample if subsample is not None else adaptive_payload_subsample(h, w, int(mask.sum()))
    d = refine_payload_depth(depth, mask, bgr) if harmonize_depth else depth.astype(np.float32)
    pts, colors = depth_rgb_to_point_cloud(
        d,
        bgr,
        intrinsics,
        mask=mask,
        depth_scale=depth_scale,
        subsample=stride,
    )
    if pts.shape[0] > 0:
        if outlier_filter:
            pts, colors = statistical_outlier_filter(pts, colors, k=10, std_ratio=1.6)
        pts = center_points(pts)
        pts = opencv_to_viewer_coords(pts)
    return pts, colors


def adaptive_scene_subsample(h: int, w: int, base: int = 1) -> int:
    """Scene backdrop stride — capped so window clouds stay dense enough for 3D."""
    megapix = (h * w) / 1_000_000.0
    if megapix > 4.0:
        return max(base * 3, 3)
    if megapix > 2.0:
        return max(base * 3, 3)
    if megapix > 1.0:
        return max(base * 2, 2)
    return max(base * 2, 2)


def cv2_cvt(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    binary: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    has_color = colors is not None and len(colors) == n

    if binary:
        return _write_ply_binary(path, points, colors if has_color else None)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header.append("end_header")
    lines = list(header)
    if has_color:
        for i in range(n):
            p = points[i]
            c = colors[i]
            lines.append(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}"
            )
    else:
        for i in range(n):
            p = points[i]
            lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray | None) -> Path:
    import struct

    n = points.shape[0]
    with path.open("wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write(
                b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
        f.write(b"end_header\n")
        if colors is not None:
            for i in range(n):
                f.write(
                    struct.pack(
                        "<fffBBB",
                        float(points[i, 0]),
                        float(points[i, 1]),
                        float(points[i, 2]),
                        int(colors[i, 0]),
                        int(colors[i, 1]),
                        int(colors[i, 2]),
                    )
                )
        else:
            for i in range(n):
                f.write(struct.pack("<fff", float(points[i, 0]), float(points[i, 1]), float(points[i, 2])))
    return path


def write_pointcloud_readme(
    path: Path,
    *,
    source: str,
    intrinsics: CameraIntrinsics,
    point_count: int,
    depth_scale: float,
    units_note: str,
    plane_rms: float | None = None,
) -> None:
    lines = [
        f"Source: {source}",
        "Output: 3D point cloud from CETI relative depth + DJI Osmo Action 4 intrinsics",
        "",
        f"Camera model: {intrinsics.model}",
        f"Aspect mode: {getattr(intrinsics, 'aspect_mode', 'unknown')}",
        f"Depth semantics: {getattr(intrinsics, 'depth_semantics', 'z_depth')}",
        f"Calibration: {getattr(intrinsics, 'calibration_source', 'approximate')}",
        f"Image size: {intrinsics.width} x {intrinsics.height}",
        f"fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f}",
        f"Points: {point_count}",
        f"Depth scale applied: {depth_scale}",
    ]
    if plane_rms is not None and np.isfinite(plane_rms):
        lines.append(f"Plane-fit RMS (shape diagnostic): {plane_rms:.4f} relative units")
    lines += [
        "",
        units_note,
        "",
        "ACCURACY: Relative monocular depth + spec-based intrinsics = qualitative geometry.",
        "Payload: motor-anchored depth (reduces shear). Scene: flat backdrop behind rig.",
        "Not metric survey data. See ceti/docs/POINTCLOUD_ACCURACY.md.",
        "",
        "Coordinate frame (viewer export): +X right, +Y up, +Z forward (Three.js).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
