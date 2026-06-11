"""
Metric depth inference in meters (ZoeDepth / Depth Anything Metric).

Separate from relative-depth whale fine-tuning (best.pt). Uses pretrained
metric heads — outdoor by default; optional CETI underwater fine-tune.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRIC_CKPT = REPO_ROOT / "checkpoints/depth_anything_metric_depth_outdoor.pt"

_metric_runner: "MetricDepthRunner | None" = None


def default_metric_resource(checkpoint: Path | None = None) -> str:
    ckpt = Path(checkpoint or DEFAULT_METRIC_CKPT)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Metric checkpoint missing: {ckpt}\n"
            "Run: bash ceti/scripts/download_checkpoints.sh"
        )
    return f"local::{ckpt.resolve()}"


class MetricDepthRunner:
    """Load ZoeDepth metric model once (requires metric_depth cwd + torchhub)."""

    def __init__(self, checkpoint: Path | str | None = None):
        import os

        from ceti.depth.ceti_metric import _infer_metric, load_metric_model, prepare_metric_depth_env

        prev_cwd = prepare_metric_depth_env()
        resource = default_metric_resource(Path(checkpoint) if checkpoint else None)
        self.model, self.device, self._config = load_metric_model(resource, dataset="nyu")
        self._infer = _infer_metric
        self.checkpoint = str(checkpoint or DEFAULT_METRIC_CKPT)
        os.chdir(prev_cwd)

    @torch.no_grad()
    def predict(
        self,
        bgr: np.ndarray,
        *,
        underwater_preprocess: bool = True,
        preprocess_method: str = "combined",
        min_depth_m: float = 0.5,
        max_depth_m: float = 25.0,
        input_size: int = 518,
    ) -> tuple[np.ndarray, np.ndarray]:
        from ceti.preprocessing.underwater import preprocess_underwater

        proc = (
            preprocess_underwater(bgr, method=preprocess_method)
            if underwater_preprocess
            else bgr.copy()
        )
        h, w = proc.shape[:2]
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = F.interpolate(
            tensor, (input_size, input_size), mode="bilinear", align_corners=False
        ).to(self.device)

        pred = self._infer(self.model, tensor)
        depth = pred.squeeze().float().cpu().numpy()
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = np.clip(depth.astype(np.float32), min_depth_m, max_depth_m * 2)
        return depth, proc


def get_metric_runner(checkpoint: Path | str | None = None) -> MetricDepthRunner:
    global _metric_runner
    ckpt = Path(checkpoint) if checkpoint else DEFAULT_METRIC_CKPT
    if _metric_runner is None or Path(_metric_runner.checkpoint).resolve() != ckpt.resolve():
        _metric_runner = MetricDepthRunner(ckpt)
    return _metric_runner


@torch.no_grad()
def predict_metric_depth_meters(
    bgr: np.ndarray,
    *,
    checkpoint: Path | str | None = None,
    underwater_preprocess: bool = True,
    preprocess_method: str = "combined",
    min_depth_m: float = 0.5,
    max_depth_m: float = 25.0,
    input_size: int = 518,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (depth_meters HxW float32, bgr_used_for_model)."""
    runner = get_metric_runner(checkpoint)
    return runner.predict(
        bgr,
        underwater_preprocess=underwater_preprocess,
        preprocess_method=preprocess_method,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        input_size=input_size,
    )


def visualize_metric_panel(
    bgr: np.ndarray,
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.5,
    max_depth_m: float = 25.0,
) -> np.ndarray:
    """RGB | colormap depth with meter scale + center-distance label."""
    h, w = bgr.shape[:2]
    clipped = np.clip(depth_m, min_depth_m, max_depth_m)
    norm = ((clipped - min_depth_m) / (max_depth_m - min_depth_m + 1e-8) * 255).astype(
        np.uint8
    )
    depth_vis = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    if depth_vis.shape[:2] != (h, w):
        depth_vis = cv2.resize(depth_vis, (w, h))

    cy, cx = h // 2, w // 2
    d_center = float(depth_m[cy, cx])
    d_med = float(np.median(depth_m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))

    banner_h = 56
    canvas = np.zeros((h + banner_h, w * 2, 3), dtype=np.uint8)
    canvas[banner_h : banner_h + h, :w] = bgr
    canvas[banner_h : banner_h + h, w:] = depth_vis

    lines = [
        f"Center distance: {d_center:.2f} m",
        f"Mid-field median: {d_med:.2f} m  (range {min_depth_m:.1f}-{max_depth_m:.1f} m colormap)",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Left: RGB  |  Right: metric depth (meters)",
        (8, banner_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_depth_readme(
    out_path: Path,
    *,
    depth_m: np.ndarray,
    source_name: str,
    checkpoint: str,
) -> None:
    cy, cx = depth_m.shape[0] // 2, depth_m.shape[1] // 2
    valid = depth_m > 0
    lines = [
        f"Source: {source_name}",
        f"Metric checkpoint: {checkpoint}",
        "",
        "Depth values are in METERS (ZoeDepth metric head).",
        "Accuracy underwater is approximate unless you fine-tune metric depth on UW RGB-D.",
        "",
        f"Center pixel: {float(depth_m[cy, cx]):.3f} m",
        f"Image min (valid): {float(depth_m[valid].min()) if valid.any() else 0:.3f} m",
        f"Image max: {float(depth_m.max()):.3f} m",
        f"Image median: {float(np.median(depth_m[valid])) if valid.any() else 0:.3f} m",
        "",
        "Raw array: same basename with _depth_meters.npy",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
