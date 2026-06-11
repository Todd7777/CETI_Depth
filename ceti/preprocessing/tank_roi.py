"""
Region-of-interest (ROI) handling for aquarium / tank cameras.

Windows, tank walls, and fixtures at the frame edge skew monocular depth
(alignment uses global medians). Crop or mask to the water column / subject
before inference and when reporting distances.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Fractional margins: left, top, right, bottom (0–0.45 each)
DEFAULT_MARGINS = (0.10, 0.18, 0.10, 0.08)


@dataclass
class TankROI:
    """Pixel rectangle inside the full frame (inclusive origin)."""

    x: int
    y: int
    w: int
    h: int
    margins: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


def margins_from_env() -> tuple[float, float, float, float] | None:
    raw = os.environ.get("CETI_TANK_ROI", "").strip()
    if not raw:
        return None
    parts = [float(p.strip()) for p in raw.replace(";", ",").split(",")]
    if len(parts) == 1:
        m = max(0.0, min(0.45, parts[0]))
        return (m, m, m, m)
    if len(parts) != 4:
        raise ValueError(
            "CETI_TANK_ROI must be '0.1' (all sides) or 'left,top,right,bottom' fractions"
        )
    return tuple(max(0.0, min(0.45, p)) for p in parts)  # type: ignore[return-value]


def roi_enabled_by_env() -> bool:
    v = os.environ.get("CETI_TANK_ROI_ENABLE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def compute_roi(
    height: int,
    width: int,
    margins: tuple[float, float, float, float] = DEFAULT_MARGINS,
) -> TankROI:
    left, top, right, bottom = margins
    x0 = int(round(width * left))
    y0 = int(round(height * top))
    x1 = int(round(width * (1.0 - right)))
    y1 = int(round(height * (1.0 - bottom)))
    x0 = max(0, min(x0, width - 2))
    y0 = max(0, min(y0, height - 2))
    x1 = max(x0 + 2, min(x1, width))
    y1 = max(y0 + 2, min(y1, height))
    return TankROI(x=x0, y=y0, w=x1 - x0, h=y1 - y0, margins=margins)


def resolve_tank_roi(
    bgr: np.ndarray,
    *,
    enabled: bool | None = None,
    margins: tuple[float, float, float, float] | None = None,
) -> TankROI | None:
    if enabled is None:
        enabled = roi_enabled_by_env()
    if not enabled:
        return None
    if margins is None:
        margins = margins_from_env() or DEFAULT_MARGINS
    h, w = bgr.shape[:2]
    return compute_roi(h, w, margins)


def crop_to_roi(bgr: np.ndarray, roi: TankROI) -> np.ndarray:
    return bgr[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w].copy()


def paste_crop_to_canvas(
    crop: np.ndarray,
    roi: TankROI,
    full_shape: tuple[int, int],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Place crop into full H×W array (float); outside ROI = fill_value."""
    h, w = full_shape
    if np.isnan(fill_value):
        out = np.full((h, w), np.nan, dtype=np.float32)
    else:
        out = np.full((h, w), fill_value, dtype=np.float32)
    ch, cw = crop.shape[:2]
    out[roi.y : roi.y + ch, roi.x : roi.x + cw] = crop.astype(np.float32)
    return out


def _frac_rect(
    h: int,
    w: int,
    frac: list[float],
) -> tuple[int, int, int, int]:
    """frac = [x0, y0, x1, y1] normalized 0–1."""
    x0 = int(round(w * frac[0]))
    y0 = int(round(h * frac[1]))
    x1 = int(round(w * frac[2]))
    y1 = int(round(h * frac[3]))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def apply_exclude_zones(mask: np.ndarray, preset: dict) -> np.ndarray:
    h, w = mask.shape
    for zone in preset.get("exclude_zones", []):
        frac = zone.get("frac")
        if not frac or len(frac) != 4:
            continue
        x0, y0, x1, y1 = _frac_rect(h, w, frac)
        mask[y0:y1, x0:x1] = False
    return mask


def detect_blue_rig_mask(bgr: np.ndarray, roi: TankROI | None) -> np.ndarray | None:
    """
    Mask the blue cylindrical rig inside the black frame (largest blue blob in ROI).
    """
    h, w = bgr.shape[:2]
    work = bgr
    if roi is not None:
        work = crop_to_roi(bgr, roi)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    # Blue rig under green water cast
    lower = np.array([85, 40, 35], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    blue = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if n < 2:
        return None
    # Largest component excluding background
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0 or areas.max() < 80:
        return None
    idx = 1 + int(np.argmax(areas))
    blob = labels == idx
    full = np.zeros((h, w), dtype=bool)
    if roi is not None:
        bh, bw = blob.shape
        full[roi.y : roi.y + bh, roi.x : roi.x + bw] = blob
    else:
        full = blob
    # Slight dilation to include black frame neighborhood
    dil = cv2.dilate(full.astype(np.uint8), kernel, iterations=2).astype(bool)
    return dil


def build_subject_mask(
    bgr: np.ndarray,
    roi: TankROI | None,
    *,
    exclude_bright: bool = True,
    bright_l_threshold: int = 210,
    center_fraction: float = 0.72,
    preset: dict | None = None,
) -> np.ndarray:
    """
    Boolean mask (H×W) for depth statistics: inside ROI, optional center weighting,
    exclude very bright regions (windows / glare).
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if roi is not None:
        mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = True
    else:
        mask[:] = True

    if center_fraction < 1.0 and roi is not None:
        cy, cx = roi.y + roi.h // 2, roi.x + roi.w // 2
        rh = int(roi.h * center_fraction / 2)
        rw = int(roi.w * center_fraction / 2)
        inner = np.zeros_like(mask)
        y0, y1 = max(0, cy - rh), min(h, cy + rh)
        x0, x1 = max(0, cx - rw), min(w, cx + rw)
        inner[y0:y1, x0:x1] = True
        mask &= inner

    if preset:
        mask = apply_exclude_zones(mask, preset)
        subj = preset.get("subject_zone", {}).get("frac")
        if subj and len(subj) == 4:
            x0, y0, x1, y1 = _frac_rect(h, w, subj)
            band = np.zeros((h, w), dtype=bool)
            band[y0:y1, x0:x1] = True
            mask &= band

    if preset and preset.get("prefer_blue_subject"):
        blue = detect_blue_rig_mask(bgr, roi)
        if blue is not None and blue.sum() > 50:
            mask &= blue

    if exclude_bright:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        thr = int(preset.get("bright_l_threshold", bright_l_threshold) if preset else bright_l_threshold)
        mask &= l_chan < thr

    return mask


def _payload_backdrop_corridor(
    h: int,
    w: int,
    payload_mask: np.ndarray,
    window_mask: np.ndarray,
) -> np.ndarray:
    """
    Narrow window strip directly behind the payload rig (not the full tank window).

    Excludes ladder/tether on the left and green arm clutter on the right that
    share the window zone but sit outside the payload column.
    """
    payload = payload_mask.astype(bool)
    ys, xs = np.where(payload)
    if ys.size == 0:
        return np.zeros((h, w), dtype=bool)

    px0, px1 = int(xs.min()), int(xs.max())
    py0, py1 = int(ys.min()), int(ys.max())
    pw = max(px1 - px0 + 1, 1)
    ph = max(py1 - py0 + 1, 1)
    pad_x = max(int(pw * 0.45), 28)
    x0 = max(0, px0 - pad_x)
    x1 = min(w, px1 + pad_x)

    win_cols = window_mask.any(axis=0)
    if win_cols.any():
        y_top = int(np.where(window_mask.any(axis=1))[0].min())
    else:
        y_top = int(h * 0.30)
    y_bot = max(y_top + 8, py0 - max(int(ph * 0.06), 6))

    corridor = np.zeros((h, w), dtype=bool)
    corridor[y_top:y_bot, x0:x1] = True
    return corridor & window_mask


def _window_zone_mask(h: int, w: int, preset: dict | None) -> np.ndarray:
    """Tank window backdrop from preset exclude_zones (or a sensible default band)."""
    window = np.zeros((h, w), dtype=bool)
    if preset:
        for zone in preset.get("exclude_zones", []):
            name = str(zone.get("name", "")).lower()
            if "window" not in name:
                continue
            frac = zone.get("frac")
            if frac and len(frac) == 4:
                x0, y0, x1, y1 = _frac_rect(h, w, frac)
                window[y0:y1, x0:x1] = True
    if not window.any():
        window[int(h * 0.30) : int(h * 0.92), int(w * 0.20) : int(w * 0.80)] = True
    return window


def _tank_interior_mask(
    h: int,
    w: int,
    preset: dict | None,
    payload_mask: np.ndarray,
) -> np.ndarray:
    """
    Tight tank halo around the payload rig — sides and below only, never into the window.
    """
    payload = payload_mask.astype(bool)
    mask = np.zeros((h, w), dtype=bool)
    if not payload.any():
        return mask

    ys, xs = np.where(payload)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    ph = max(y1 - y0 + 1, 1)
    pw = max(x1 - x0 + 1, 1)
    pad_x = max(int(pw * 0.22), 18)
    pad_down = max(int(ph * 0.14), 10)

    # Do not extend above payload top — that band is the view-through window.
    mask[y0 : min(h, y1 + pad_down + 1), max(0, x0 - pad_x) : min(w, x1 + pad_x + 1)] = True
    mask[_window_zone_mask(h, w, preset)] = False
    mask |= payload
    return mask


def build_scene_contrast_mask(
    bgr: np.ndarray,
    depth: np.ndarray,
    roi: TankROI | None,
    *,
    preset: dict | None = None,
    payload_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Mask for tank scene point clouds: payload rig + nearby tank interior.

    The view-through window (ladder, far wall) is explicitly excluded so the scene
    explorer shows the rig in the tank, not a giant backdrop wall.
    """
    h, w = bgr.shape[:2]
    dz = depth.astype(np.float32)
    valid = np.isfinite(dz) & (dz > 0)
    payload_bool = payload_mask.astype(bool) if payload_mask is not None else np.zeros((h, w), dtype=bool)

    if payload_bool.any():
        mask = _tank_interior_mask(h, w, preset, payload_bool)
    else:
        mask = np.zeros((h, w), dtype=bool)
        if roi is not None:
            mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = True
        mask[_window_zone_mask(h, w, preset)] = False

    if roi is not None:
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = True
        mask &= roi_mask

    mask &= valid

    if preset:
        for zone in preset.get("exclude_zones", []):
            name = str(zone.get("name", "")).lower()
            if "window" in name:
                continue
            frac = zone.get("frac")
            if not frac or len(frac) != 4:
                continue
            x0, y0, x1, y1 = _frac_rect(h, w, frac)
            mask[y0:y1, x0:x1] = False

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    bright_thr = int(preset.get("bright_l_threshold", 230) if preset else 230)
    dark_enough = lab[:, :, 0] < bright_thr
    bg = mask & ~payload_bool
    mask[bg] &= dark_enough[bg]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green_arm = cv2.inRange(
        hsv, np.array([38, 70, 50], dtype=np.uint8), np.array([92, 255, 255], dtype=np.uint8)
    )
    right = np.zeros((h, w), dtype=bool)
    right[:, int(w * 0.62) :] = True
    mask[green_arm.astype(bool) & right] = False

    floor = np.zeros((h, w), dtype=bool)
    floor[int(h * 0.62) :, :] = True
    mask[floor] = False

    if payload_bool.any():
        mask |= payload_bool

    return mask


def build_pointcloud_mask(
    bgr: np.ndarray,
    roi: TankROI | None,
    *,
    preset: dict | None = None,
    depth: np.ndarray | None = None,
    exclude_bright: bool = True,
) -> np.ndarray:
    """
    Point-cloud mask. When depth is supplied, restricts to experiment payload only.
    Otherwise falls back to water-column ROI (legacy).
    """
    if depth is not None:
        from ceti.preprocessing.payload_segmentation import build_payload_mask

        mask, _mode = build_payload_mask(bgr, depth, preset=preset)
        if mask is not None:
            return mask

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if roi is not None:
        mask[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w] = True
    else:
        mask[:] = True

    if preset:
        mask = apply_exclude_zones(mask, preset)

    if exclude_bright:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        thr = int(preset.get("bright_l_threshold", 220) if preset else 220)
        mask &= lab[:, :, 0] < thr

    return mask


def draw_analysis_overlay(
    bgr: np.ndarray,
    roi: TankROI | None,
    subject_mask: np.ndarray,
    preset: dict | None = None,
) -> np.ndarray:
    """Debug view: ROI box, excluded zones (red), subject mask (green tint)."""
    out = draw_roi_overlay(bgr, roi)
    h, w = out.shape[:2]
    if preset:
        for zone in preset.get("exclude_zones", []):
            frac = zone.get("frac")
            if frac and len(frac) == 4:
                x0, y0, x1, y1 = _frac_rect(h, w, frac)
                cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
    tint = out.copy()
    tint[subject_mask] = (tint[subject_mask] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    return cv2.addWeighted(out, 0.65, tint, 0.35, 0)


def draw_roi_overlay(bgr: np.ndarray, roi: TankROI | None) -> np.ndarray:
    out = bgr.copy()
    if roi is None:
        return out
    cv2.rectangle(
        out,
        (roi.x, roi.y),
        (roi.x + roi.w - 1, roi.y + roi.h - 1),
        (0, 255, 120),
        2,
    )
    cv2.putText(
        out,
        "depth ROI",
        (roi.x + 4, max(roi.y - 6, 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 120),
        1,
        cv2.LINE_AA,
    )
    return out


def compose_tank_preview(
    full_bgr: np.ndarray,
    roi: TankROI | None,
    depth_panel: np.ndarray,
) -> np.ndarray:
    """Full frame with ROI box (left) | depth panel from crop (right)."""
    left = draw_roi_overlay(full_bgr, roi)
    h = left.shape[0]
    ph, pw = depth_panel.shape[:2]
    banner = min(56, ph // 4)
    depth_side = depth_panel[banner : banner + min(ph - banner, h), pw // 2 :]
    if depth_side.size == 0:
        depth_side = depth_panel[:, pw // 2 :]
    right = cv2.resize(depth_side, (left.shape[1], h))
    return np.concatenate([left, right], axis=1)


def load_roi_preset(path: Path | None = None) -> dict | None:
    """Optional JSON preset (margins, exclude_zones, subject_zone)."""
    default_name = os.environ.get("CETI_TANK_PRESET", "tank_roi_ceti_full")
    path = path or Path(
        os.environ.get(
            "CETI_TANK_ROI_PRESET",
            str(Path(__file__).resolve().parents[1] / "configs" / f"{default_name}.json"),
        )
    )
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "configs" / "tank_roi_default.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
