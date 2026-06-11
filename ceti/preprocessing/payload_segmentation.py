"""
Payload segmentation for CETI tank experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PayloadSegmentationResult:
    mask: np.ndarray
    mode: str
    zone_frac: list[float]
    pixel_count: int
    seed_found: bool = False
    coverage_frac: float = 0.0
    box_half_px: int = 0


def _cfg(preset: dict | None) -> dict:
    base = {
        "zone_frac": None,
        "min_area": 600,
        "bright_l_threshold": 155,
        "dark_l_max": 130,
        "dark_l_min": 12,
        "void_l_threshold": 120,
        "frame_scan_px": 30,
        "frame_min_margin_px": 11,
        "frame_max_above_motor_px": 24,
        "frame_max_below_motor_px": 20,
        "frame_pad_px": 2,
        "frame_bar_frac": 0.095,
        "frame_bar_min_px": 5,
        "frame_bar_max_px": 13,
        "edge_peak_thresh": 0.26,
        "prefer_blue_seed": True,
        "apply_hard_excludes": True,
    }
    if preset:
        base.update(preset.get("payload_segmentation") or {})
    return base


def _frac_rect(h: int, w: int, frac: list[float]) -> tuple[int, int, int, int]:
    x0 = int(round(w * frac[0]))
    y0 = int(round(h * frac[1]))
    x1 = int(round(w * frac[2]))
    y1 = int(round(h * frac[3]))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _blue_seed_mask(zone: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([35, 28, 24], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
    lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
    chroma = np.abs(lab[:, :, 1].astype(np.int16) - 128) + np.abs(lab[:, :, 2].astype(np.int16) - 128)
    blue[chroma < 18] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0 or areas.max() < 40:
        return None
    idx = 1 + int(np.argmax(areas))
    return labels == idx


def _motor_bounds(seed: np.ndarray) -> tuple[int, int, int, int, int, int]:
    ys, xs = np.where(seed)
    if ys.size == 0:
        return 0, 0, 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return cx, cy, x0, x1, y0, y1


def _apply_hard_excludes(mask: np.ndarray, preset: dict | None) -> np.ndarray:
    if not preset:
        return mask
    h, w = mask.shape
    for zone in preset.get("exclude_zones", []):
        name = str(zone.get("name", "")).lower()
        if "window" in name:
            continue
        frac = zone.get("frac")
        if not frac or len(frac) != 4:
            continue
        x0, y0, x1, y1 = _frac_rect(h, w, frac)
        mask[y0:y1, x0:x1] = False
    return mask


def _edge_map(zone: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.Canny(enhanced, 28, 95)


def _column_score(edges: np.ndarray, dark: np.ndarray, y0: int, y1: int, x: int) -> float:
    if y1 <= y0:
        return 0.0
    e = float(edges[y0:y1, x].max()) if edges[y0:y1, x].size else 0.0
    d = float(dark[y0:y1, x].mean()) if dark[y0:y1, x].size else 0.0
    return e + 0.4 * d


def _row_score(edges: np.ndarray, dark: np.ndarray, x0: int, x1: int, y: int) -> float:
    if x1 <= x0:
        return 0.0
    e = float(edges[y, x0:x1].max()) if edges[y, x0:x1].size else 0.0
    d = float(dark[y, x0:x1].mean()) if dark[y, x0:x1].size else 0.0
    return e + 0.4 * d


def _snap_frame_rect(
    edges: np.ndarray,
    dark: np.ndarray,
    mx0: int,
    mx1: int,
    my0: int,
    my1: int,
    zh: int,
    zw: int,
    cfg: dict,
) -> tuple[int, int, int, int]:
    """
    Snap payload outer rectangle by scanning outward from the motor bbox.
    Each side is independent (no forced square symmetry).
    """
    edges_f = cv2.dilate(
        edges.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    ).astype(np.float32)
    if edges_f.max() > 0:
        edges_f /= edges_f.max()

    v0 = my0 + max(1, int((my1 - my0) * 0.12))
    v1 = my1 - max(1, int((my1 - my0) * 0.12))
    h0 = mx0 + max(1, int((mx1 - mx0) * 0.12))
    h1 = mx1 - max(1, int((mx1 - mx0) * 0.12))

    scan = int(cfg.get("frame_scan_px", 30))
    thresh = float(cfg.get("edge_peak_thresh", 0.26))
    min_margin = int(cfg.get("frame_min_margin_px", 11))
    pad = int(cfg.get("frame_pad_px", 2))

    left = mx0
    for x in range(mx0, max(-1, mx0 - scan - 1), -1):
        if _column_score(edges_f, dark, v0, v1, x) >= thresh:
            left = x

    right = mx1
    for x in range(mx1, min(zw, mx1 + scan + 1)):
        if _column_score(edges_f, dark, v0, v1, x) >= thresh:
            right = x

    top = my0
    for y in range(my0, max(-1, my0 - scan - 1), -1):
        if _row_score(edges_f, dark, h0, h1, y) >= thresh:
            top = y

    bottom = my1
    for y in range(my1, min(zh, my1 + scan + 1)):
        if _row_score(edges_f, dark, h0, h1, y) >= thresh:
            bottom = y

    max_above = int(cfg.get("frame_max_above_motor_px", 24))
    max_below = int(cfg.get("frame_max_below_motor_px", 20))

    left = min(left, mx0 - min_margin)
    right = max(right, mx1 + min_margin)
    top = min(top, my0 - min_margin)
    bottom = max(bottom, my1 + min_margin)

    top = max(top, my0 - max_above)
    bottom = min(bottom, my1 + max_below)

    left = max(0, left - pad)
    right = min(zw - 1, right + pad)
    top = max(0, top - pad)
    bottom = min(zh - 1, bottom + pad)
    return left, right, top, bottom


def _frame_bar_thickness(rect_w: int, rect_h: int, cfg: dict) -> int:
    span = max(rect_w, rect_h)
    thick = int(round(span * float(cfg.get("frame_bar_frac", 0.095))))
    thick = max(int(cfg.get("frame_bar_min_px", 5)), thick)
    thick = min(int(cfg.get("frame_bar_max_px", 13)), thick)
    return thick


def _perimeter_band(zh: int, zw: int, left: int, right: int, top: int, bottom: int, thick: int) -> np.ndarray:
    band = np.zeros((zh, zw), dtype=bool)
    if right <= left or bottom <= top:
        return band
    band[top : top + thick, left : right + 1] = True
    band[bottom - thick + 1 : bottom + 1, left : right + 1] = True
    band[top : bottom + 1, left : left + thick] = True
    band[top : bottom + 1, right - thick + 1 : right + 1] = True
    return band


def _inner_cavity(left: int, right: int, top: int, bottom: int, thick: int, zh: int, zw: int) -> np.ndarray:
    cavity = np.zeros((zh, zw), dtype=bool)
    y0 = top + thick
    y1 = bottom - thick + 1
    x0 = left + thick
    x1 = right - thick + 1
    if y1 > y0 and x1 > x0:
        cavity[y0:y1, x0:x1] = True
    return cavity


def _void_mask(
    l_chan: np.ndarray,
    cavity: np.ndarray,
    seed: np.ndarray,
    blue: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Open center where the tank window shows through the frame."""
    void_l = int(cfg["void_l_threshold"])
    bright = l_chan >= void_l
    not_rig = l_chan >= void_l - 6
    void = cavity & bright & not_rig & ~seed & ~blue
    void &= l_chan > int(cfg["dark_l_max"]) - 4
    return void


def _frame_from_band(band: np.ndarray) -> np.ndarray:
    """Thin perimeter strips only — tight, connected frame rails."""
    return band.copy()


def segment_ceti_payload(
    bgr: np.ndarray,
    depth: np.ndarray,
    preset: dict | None = None,
) -> PayloadSegmentationResult | None:
    """
    Segment the payload box by snapping to real frame edges in the image.

    Frame bars come from dark/edge pixels on a tight perimeter band (not a drawn square).
    Interior is the blue motor plus dark rig surfaces; window void is cut from the center.
    """
    cfg = _cfg(preset)
    h, w = bgr.shape[:2]
    zone_frac = cfg.get("zone_frac") or (preset or {}).get("subject_zone", {}).get("frac")
    if not zone_frac or len(zone_frac) != 4:
        return None

    x0, y0, x1, y1 = _frac_rect(h, w, zone_frac)
    zone = bgr[y0:y1, x0:x1]
    if zone.size == 0:
        return None

    zh, zw = zone.shape[:2]
    lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    edges = _edge_map(zone)
    dark = (l_chan >= int(cfg["dark_l_min"])) & (l_chan <= int(cfg["dark_l_max"]))

    seed = _blue_seed_mask(zone) if cfg.get("prefer_blue_seed", True) else None
    if seed is None or not seed.any():
        seed = np.zeros((zh, zw), dtype=bool)
        seed[zh // 2 - 2 : zh // 2 + 3, zw // 2 - 2 : zw // 2 + 3] = True
        seed_found = False
    else:
        seed_found = True

    _, _, mx0, mx1, my0, my1 = _motor_bounds(seed)
    if mx1 <= mx0 or my1 <= my0:
        return None

    blue = seed.copy()
    left, right, top, bottom = _snap_frame_rect(edges, dark, mx0, mx1, my0, my1, zh, zw, cfg)
    rect_w = right - left + 1
    rect_h = bottom - top + 1
    thick = _frame_bar_thickness(rect_w, rect_h, cfg)

    band = _perimeter_band(zh, zw, left, right, top, bottom, thick)
    cavity = _inner_cavity(left, right, top, bottom, thick, zh, zw)
    void = _void_mask(l_chan, cavity, seed, blue, cfg)

    frame = _frame_from_band(band)
    motor_halo = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ).astype(bool)
    motor_dark = motor_halo & dark & cavity & ~void
    interior = (cavity & blue & ~void) | (seed & cavity) | motor_dark

    mask = frame | interior
    mask &= ~void

    envelope = np.zeros((zh, zw), dtype=bool)
    envelope[top : bottom + 1, left : right + 1] = True
    mask &= envelope

    tether_cut = np.zeros((zh, zw), dtype=bool)
    tether_cut[: max(0, top - 2), left : right + 1] = True
    mask &= ~tether_cut

    # Thin tether/cable above the motor column (visible through frame void).
    tether_col = np.zeros((zh, zw), dtype=bool)
    tether_col[:top, mx0:mx1 + 1] = True
    tether_line = tether_col & dark & ~blue & (l_chan <= int(cfg["void_l_threshold"]) + 35)
    mask &= ~tether_line

    below_cut = np.zeros((zh, zw), dtype=bool)
    below_cut[min(zh, bottom + 2) :, left : right + 1] = True
    mask &= ~below_cut

    full = np.zeros((h, w), dtype=bool)
    full[y0:y1, x0:x1] = mask

    if cfg.get("apply_hard_excludes", True):
        full = _apply_hard_excludes(full, preset)

    if int(full.sum()) < int(cfg["min_area"]):
        return None

    half = max(rect_w, rect_h) // 2
    zone_pixels = (y1 - y0) * (x1 - x0)
    return PayloadSegmentationResult(
        mask=full,
        mode="payload_box_image_fitted",
        zone_frac=list(zone_frac),
        pixel_count=int(full.sum()),
        seed_found=seed_found,
        coverage_frac=float(full.sum()) / max(zone_pixels, 1),
        box_half_px=half,
    )


def build_payload_mask(
    bgr: np.ndarray,
    depth: np.ndarray,
    *,
    preset: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    result = segment_ceti_payload(bgr, depth, preset)
    if result is None:
        return None, "payload_failed"
    return result.mask, result.mode


def render_payload_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    dim_strength: float = 0.72,
    highlight_bgr: tuple[int, int, int] = (72, 220, 95),
) -> np.ndarray:
    out = bgr.astype(np.float32)
    dim = out * (1.0 - dim_strength)
    tint = np.array(highlight_bgr, dtype=np.float32)
    out[mask] = out[mask] * 0.38 + tint * 0.62
    out[~mask] = dim[~mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_mask_border_overlay(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (80, 255, 120),
    thickness: int = 2,
) -> np.ndarray:
    out = bgr.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out
