"""
Accurate underwater depth in meters for professor uploads.

Combines:
  1. CETI fine-tuned relative depth (vitl, whale/marine) — good structure in UW scenes
  2. Metric depth prior (ZoeDepth outdoor) — approximate absolute scale in meters

Per frame we:
  - Flip relative depth if it is anticorrelated with the metric map (common with DPT heads)
  - Align relative → metric (robust scale + small shift)
  - Optionally rescale to a user-known center distance (CETI_DEPTH_CENTER_REF_M)

Outdoor metric depth is still weak on underwater imagery; use center reference calibration
when you know the distance to the subject (e.g. pool / tank tests).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ceti.depth.metric_infer import MetricDepthRunner, visualize_metric_panel, write_depth_readme


def _alignment_mask(
    depth_rel: np.ndarray,
    depth_metric: np.ndarray,
    *,
    min_metric_m: float = 0.3,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (
        np.isfinite(depth_metric)
        & np.isfinite(depth_rel)
        & (depth_metric > min_metric_m)
        & (depth_rel > 0)
    )
    if sample_mask is not None:
        mask &= sample_mask.astype(bool)
    return mask


def orient_relative_to_metric(
    depth_rel: np.ndarray,
    depth_metric: np.ndarray,
    *,
    min_metric_m: float = 0.3,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Ensure larger relative values mean farther (same trend as metric depth).

    Returns (oriented_rel, was_inverted).
    """
    mask = _alignment_mask(
        depth_rel, depth_metric, min_metric_m=min_metric_m, sample_mask=sample_mask
    )
    if mask.sum() < 64:
        return depth_rel, False

    p = depth_rel[mask].astype(np.float64).ravel()
    t = depth_metric[mask].astype(np.float64).ravel()
    corr = float(np.corrcoef(p, t)[0, 1])
    if corr < 0:
        return (depth_rel.max() - depth_rel).astype(np.float32), True
    return depth_rel, False


def align_relative_to_metric(
    depth_rel: np.ndarray,
    depth_metric: np.ndarray,
    *,
    min_metric_m: float = 0.3,
    align_mode: str | None = None,
    max_shift_m: float = 5.0,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, bool]:
    """
    Align oriented relative depth to metric prior.

    align_mode:
      - scale_median (default): scale = median(metric)/median(rel), shift = 0
      - affine: least-squares scale + shift (shift clamped to ±max_shift_m)

    Returns (aligned_depth_m, scale, shift, inverted_relative).
    """
    depth_rel, inverted = orient_relative_to_metric(
        depth_rel, depth_metric, min_metric_m=min_metric_m
    )
    mode = (align_mode or os.environ.get("CETI_DEPTH_ALIGN", "affine")).strip().lower()
    mask = _alignment_mask(
        depth_rel, depth_metric, min_metric_m=min_metric_m, sample_mask=sample_mask
    )

    if mask.sum() < 64:
        scale = float(np.median(depth_metric) / (np.median(depth_rel) + 1e-8))
        shift = 0.0
        aligned = (scale * depth_rel + shift).astype(np.float32)
        return _clip_depth(aligned), scale, shift, inverted

    p = depth_rel[mask].astype(np.float64)
    t = depth_metric[mask].astype(np.float64)

    if mode == "scale_median":
        scale = float(np.median(t) / (np.median(p) + 1e-8))
        shift = 0.0
    else:
        a_00 = (p * p).sum()
        a_01 = p.sum()
        a_11 = float(len(p))
        b_0 = (p * t).sum()
        b_1 = t.sum()
        det = a_00 * a_11 - a_01 * a_01
        if det <= 1e-8:
            scale = float(np.median(t) / (np.median(p) + 1e-8))
            shift = 0.0
        else:
            scale = float((a_11 * b_0 - a_01 * b_1) / det)
            shift = float((-a_01 * b_0 + a_00 * b_1) / det)
            if scale < 0:
                scale = abs(scale)
            shift = float(np.clip(shift, -max_shift_m, max_shift_m))

    aligned = (scale * depth_rel + shift).astype(np.float32)
    return _clip_depth(aligned), scale, shift, inverted


def _clip_depth(depth_m: np.ndarray) -> np.ndarray:
    max_m = float(os.environ.get("CETI_DEPTH_MAX_M", "80"))
    min_m = float(os.environ.get("CETI_DEPTH_MIN_M", "0.1"))
    return np.clip(depth_m, min_m, max_m).astype(np.float32)


def apply_center_reference_calibration(
    depth_m: np.ndarray,
    *,
    reference_center_m: float | None = None,
) -> tuple[np.ndarray, float | None]:
    """
    Rescale entire map so center pixel matches a known distance (pool/tank calibration).
    """
    ref = reference_center_m
    if ref is None:
        raw = os.environ.get("CETI_DEPTH_CENTER_REF_M", "").strip()
        if raw:
            ref = float(raw)
    if ref is None or ref <= 0:
        return depth_m, None

    cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
    center = float(depth_m[cy, cx])
    if center <= 1e-6:
        return depth_m, None
    factor = ref / center
    return _clip_depth(depth_m * factor), factor


def predict_accurate_depth_meters(
    bgr: np.ndarray,
    rel_model,
    rel_transform,
    device: str,
    metric_runner: MetricDepthRunner,
    *,
    underwater_preprocess: bool = True,
    preprocess_method: str = "combined",
    reference_center_m: float | None = None,
    sample_mask: np.ndarray | None = None,
    metric_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (depth_meters, bgr_preprocessed, info dict with scale/shift).
    """
    from ceti.depth.infer_robot import predict_depth_raw
    from ceti.preprocessing.underwater import preprocess_underwater

    proc = (
        preprocess_underwater(bgr, method=preprocess_method)
        if underwater_preprocess
        else bgr.copy()
    )
    depth_rel = predict_depth_raw(rel_model, rel_transform, proc, device)
    metric_src = metric_bgr if metric_bgr is not None else bgr
    depth_metric, _ = metric_runner.predict(
        metric_src,
        underwater_preprocess=underwater_preprocess,
        preprocess_method=preprocess_method,
    )
    aligned, scale, shift, inverted = align_relative_to_metric(
        depth_rel, depth_metric, sample_mask=sample_mask
    )
    aligned, cal_factor = apply_center_reference_calibration(
        aligned, reference_center_m=reference_center_m
    )
    from ceti.depth.depth_export import depth_summary

    info = {
        "scale": scale,
        "shift": shift,
        "inverted_relative": inverted,
        **depth_summary(aligned, sample_mask),
    }
    if cal_factor is not None:
        info["calibration_factor"] = cal_factor
        info["reference_center_m"] = float(
            reference_center_m
            or os.environ.get("CETI_DEPTH_CENTER_REF_M", "0")
        )
    return aligned, proc, info


def visualize_accurate_panel(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    info: dict,
    *,
    min_depth_m: float = 0.5,
    max_depth_m: float = 25.0,
) -> np.ndarray:
    panel = visualize_metric_panel(
        bgr, depth_m, min_depth_m=min_depth_m, max_depth_m=max_depth_m
    )
    inv = "inv" if info.get("inverted_relative") else "ok"
    extra = (
        f"CETI-aligned ({inv})  scale={info.get('scale', 0):.4f}  "
        f"shift={info.get('shift', 0):.3f} m"
    )
    if info.get("calibration_factor") is not None:
        extra += f"  cal×{info['calibration_factor']:.3f}"
    cv2.putText(
        panel,
        extra,
        (8, panel.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def write_accurate_readme(path: Path, *, depth_m: np.ndarray, source: str, info: dict) -> None:
    cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
    valid = depth_m > 0
    lines = [
        f"Source: {source}",
        "Mode: accurate (CETI fine-tuned structure + metric scale alignment)",
        "",
        "Depth is in METERS. Alignment: scale * CETI_relative + shift ≈ metric_prior.",
        f"  inverted_relative = {info.get('inverted_relative', False)}",
        f"  scale = {info.get('scale', 0):.6f}",
        f"  shift = {info.get('shift', 0):.4f} m",
    ]
    if info.get("calibration_factor") is not None:
        lines.append(f"  user center calibration factor = {info['calibration_factor']:.6f}")
        lines.append(f"  reference_center_m = {info.get('reference_center_m', '')}")
    lines.extend(
        [
            "",
            f"Center: {float(depth_m[cy, cx]):.3f} m",
            f"Min: {float(depth_m[valid].min()) if valid.any() else 0:.3f} m",
            f"Max: {float(depth_m.max()):.3f} m",
            f"Median: {float(np.median(depth_m[valid])) if valid.any() else 0:.3f} m",
            "",
            "Distances folder: .json / .csv / .npy",
            "",
            "Note: Outdoor metric prior is approximate underwater. For known subject distance,",
            "set CETI_DEPTH_CENTER_REF_M=<meters> or use portal 'Known center distance'.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
