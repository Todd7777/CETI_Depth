#!/usr/bin/env bash
# Pre-crop files in ceti/inbox/uploads/ to the tank ROI (for manual QC before pipeline).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python3
UPLOAD="${REPO_ROOT}/ceti/inbox/uploads"
OUT="${UPLOAD}/cropped"
mkdir -p "$OUT"
"$PYTHON" - <<'PY'
from pathlib import Path
import cv2
from ceti.depth.infer_robot import predict_depth_raw
from ceti.depth.upload_pipeline import _load_models
from ceti.preprocessing.payload_segmentation import (
    build_payload_mask,
    render_payload_overlay,
)
from ceti.preprocessing.tank_roi import (
    crop_to_roi,
    load_roi_preset,
    resolve_tank_roi,
)
from ceti.preprocessing.underwater import preprocess_underwater
from ceti.utils.device import configure_compute, get_device

configure_compute()
device = str(get_device())
model, transform, _ = _load_models(Path("checkpoints/ceti_whale_depth/best.pt"), device)

upload = Path("ceti/inbox/uploads")
out = upload / "cropped"
for p in sorted(upload.iterdir()):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue
    if p.parent.name == "cropped":
        continue
    bgr = cv2.imread(str(p))
    if bgr is None:
        continue
    preset = load_roi_preset() or {}
    roi = resolve_tank_roi(bgr, enabled=True, margins=tuple(preset.get("margins", [])) or None)
    if roi is None:
        continue
    crop = crop_to_roi(bgr, roi)
    proc = preprocess_underwater(bgr, method="combined")
    depth = predict_depth_raw(model, transform, proc, device)
    mask, mode = build_payload_mask(bgr, depth, preset=preset)
    if mask is None:
        print(p.name, "payload segmentation failed")
        continue
    overlay = render_payload_overlay(bgr, mask)
    cv2.imwrite(str(out / f"{p.stem}_roi_crop{p.suffix}"), crop)
    cv2.imwrite(str(out / f"{p.stem}_mask_qc{p.suffix}"), overlay)
    print(p.name, "->", roi.w, "x", roi.h)
PY
echo "Crops written to ${OUT}/"
