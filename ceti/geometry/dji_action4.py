"""
DJI Osmo Action 4 camera intrinsics for depth unprojection.

IMPORTANT (scientific):
  These are SPEC-DERIVED APPROXIMATIONS, not checkerboard-calibrated parameters.
  For publication-grade metrology, run OpenCV fisheye calibration on tank footage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "ceti/configs/dji_action4.yaml"


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str  # pinhole | fisheye_equidistant
    f_fisheye: float | None = None
    aspect_mode: str = "auto"
    dfov_deg: float = 155.0
    distortion: list[float] | None = None  # OpenCV fisheye k1..k4 (None = not calibrated)
    depth_semantics: str = "z_depth"  # z_depth | range_along_ray
    calibration_source: str = "dji_specs_approximate"

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _load_config(path: Path | None = None) -> dict:
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def native_resolution(cfg: dict | None = None) -> tuple[int, int]:
    cfg = cfg or _load_config()
    r = cfg["native_resolution"]
    return int(r["width"]), int(r["height"])


def detect_aspect_mode(width: int, height: int, *, tolerance: float = 0.06) -> str:
    """Infer capture mode from pixel aspect ratio."""
    aspect = width / max(height, 1)
    if abs(aspect - 4.0 / 3.0) <= tolerance:
        return "4x3_photo"
    if abs(aspect - 16.0 / 9.0) <= tolerance:
        return "16x9_video"
    return "custom"


def physical_focal_mm(cfg: dict) -> float:
    """Convert 35 mm equivalent focal length to physical focal length on sensor."""
    lens = cfg["lens"]
    sensor = cfg["sensor_mm"]
    equiv = float(lens["focal_length_35mm_equiv_mm"])
    sw = float(sensor["width"])
    sh = float(sensor["height"])
    sensor_diag = float(np.hypot(sw, sh))
    return equiv * (sensor_diag / 43.27)


def intrinsics_from_fov_diagonal(
    width: int,
    height: int,
    dfov_deg: float = 155.0,
    *,
    aspect_mode: str = "custom",
    depth_semantics: str = "z_depth",
) -> CameraIntrinsics:
    """Equidistant fisheye focal from diagonal FOV at the actual image size."""
    diag = float(np.hypot(width, height))
    theta_max = np.deg2rad(dfov_deg) / 2.0
    f = (diag / 2.0) / theta_max
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=f,
        fy=f,
        cx=width / 2.0,
        cy=height / 2.0,
        model="fisheye_equidistant",
        f_fisheye=f,
        aspect_mode=aspect_mode,
        dfov_deg=dfov_deg,
        depth_semantics=depth_semantics,
    )


def intrinsics_pinhole_physical(
    width: int,
    height: int,
    cfg: dict,
    *,
    aspect_mode: str,
    depth_semantics: str = "z_depth",
) -> CameraIntrinsics:
    """
    Pinhole fx/fy from sensor geometry.

    16:9 video uses a vertical center crop of the 4:3 sensor (full width, reduced height).
    """
    nw, nh = native_resolution(cfg)
    sensor = cfg["sensor_mm"]
    sw = float(sensor["width"])
    sh = float(sensor["height"])
    f_mm = physical_focal_mm(cfg)
    dfov = float(cfg["lens"]["fov_diagonal_deg"])

    if aspect_mode == "16x9_video":
        crop_h_native = nw * height / width
        active_h_mm = sh * (crop_h_native / nh)
        fx = f_mm / sw * width
        fy = f_mm / active_h_mm * height
        cx = width / 2.0
        cy = height / 2.0
    elif aspect_mode == "4x3_photo":
        fx = f_mm / sw * width
        fy = f_mm / sh * height
        cx = width / 2.0
        cy = height / 2.0
    else:
        fx = f_mm / sw * width
        fy = f_mm / sh * height
        cx = width / 2.0
        cy = height / 2.0

    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        model="pinhole",
        aspect_mode=aspect_mode,
        dfov_deg=dfov,
        depth_semantics=depth_semantics,
    )


def dji_action4_intrinsics(
    width: int,
    height: int,
    *,
    model: str | None = None,
    config_path: Path | None = None,
    aspect_mode: str | None = None,
    depth_semantics: str | None = None,
) -> CameraIntrinsics:
    """
    Intrinsics for DJI Action 4 at the actual frame size.

    model:
      - pinhole (recommended default for monocular depth networks)
      - fisheye_equidistant (use with depth_semantics=z_depth or range_along_ray)
    """
    import os

    cfg = _load_config(config_path)
    dfov = float(cfg["lens"]["fov_diagonal_deg"])
    model = model or os.environ.get("CETI_UNPROJECT_MODEL", "pinhole")
    aspect_mode = aspect_mode or os.environ.get("CETI_ASPECT_MODE", "auto")
    if aspect_mode == "auto":
        aspect_mode = detect_aspect_mode(width, height)
    depth_semantics = depth_semantics or os.environ.get("CETI_DEPTH_SEMANTICS", "z_depth")

    if model == "pinhole":
        intr = intrinsics_pinhole_physical(
            width,
            height,
            cfg,
            aspect_mode=aspect_mode,
            depth_semantics=depth_semantics,
        )
    elif model == "fisheye_equidistant":
        intr = intrinsics_from_fov_diagonal(
            width,
            height,
            dfov,
            aspect_mode=aspect_mode,
            depth_semantics=depth_semantics,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return intr


def intrinsics_accuracy_notes(intr: CameraIntrinsics) -> dict:
    """Metadata bundled with exports — explicit limitations."""
    return {
        "calibration_source": intr.calibration_source,
        "aspect_mode": intr.aspect_mode,
        "depth_semantics": intr.depth_semantics,
        "limitations": [
            "Intrinsics are derived from DJI published specs (155 deg DFOV, 12.7 mm equiv., 1/1.3 in sensor).",
            "No checkerboard fisheye calibration was performed; focal length uncertainty is typically 3-8%.",
            "16:9 frames are modeled as a vertical center crop of the 4:3 sensor (standard Action cam behavior).",
            "CETI depth is relative/affine-invariant, not metric LiDAR range.",
            "Monocular depth on fisheye imagery can exhibit radial bias; geometry is qualitative.",
            "For metrology, calibrate with OpenCV fisheye (cv2.fisheye.calibrate) on tank chessboard footage.",
        ],
        "recommended_use": (
            "Shape comparison, segmentation-backed visualization, and noise-sensitivity studies — "
            "not survey-grade measurement without metric calibration."
        ),
    }


def write_intrinsics_json(path: Path, intrinsics: CameraIntrinsics, *, extra: dict | None = None) -> None:
    payload = {
        "camera": "DJI Osmo Action 4",
        **intrinsics.to_dict(),
        "accuracy": intrinsics_accuracy_notes(intrinsics),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
