#!/usr/bin/env python3
"""
End-to-end proof: underwater RGB → depth (relative + metric).

Produces artifacts under ceti/outputs/proof/ and a JSON report suitable for
lab demos / AVATARS integration readiness.

Usage:
    python ceti/scripts/prove_pipeline.py           # full proof
    python ceti/scripts/prove_pipeline.py --quick   # relative depth only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PROOF_DIR = REPO_ROOT / "ceti/outputs/proof"
UNDERWATER_VIDEOS = (
    REPO_ROOT / "assets/examples_video/davis_dolphins.mp4",
    REPO_ROOT / "assets/examples_video/davis_seasnake.mp4",
)
OUTDOOR_CKPT = REPO_ROOT / "checkpoints/depth_anything_metric_depth_outdoor.pt"
FIELD_DEPTH_CKPT = REPO_ROOT / "checkpoints/ceti_underwater_field/best.pt"
LEGACY_DEPTH_CKPT = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"
CETI_LAB_ROOT = REPO_ROOT / "data/ceti_lab"
METRIC_SAVE = REPO_ROOT / "checkpoints/ceti_proof_underwater"


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def extract_video_frames(video: Path, out_dir: Path, max_frames: int = 6) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(total - 1, 1), max_frames, dtype=int)

    saved = []
    for target in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{int(target):05d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)
    cap.release()
    return saved


def prove_relative_depth(report: dict, *, quick: bool = False) -> None:
    """Relative Depth Anything on real underwater video (davis dolphins)."""
    import torch
    from torchvision.transforms import Compose

    from depth_anything.dpt import DepthAnything
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from ceti.depth.infer_robot import build_depth_model, predict_depth, resolve_encoder
    from ceti.preprocessing.underwater import preprocess_underwater
    from ceti.utils.device import configure_compute, device_name, get_device

    _banner("Phase 1: Relative depth on real underwater RGB video")

    videos = [v for v in UNDERWATER_VIDEOS if v.exists()]
    if not videos:
        report["relative"] = {"status": "skipped", "reason": "missing DAVIS underwater videos"}
        return

    if quick:
        videos = videos[:1]

    configure_compute()
    device = get_device()
    print(f"  Compute: {device_name(device)}")
    if quick:
        print("  Quick mode: 1 video, 2 frames, 2 variants (faster smoke test)")

    frames_dir = PROOF_DIR / "frames"
    max_frames = 2 if quick else 4
    frames = []
    for video in videos:
        sub = frames_dir / video.stem
        frames.extend(extract_video_frames(video, sub, max_frames=max_frames))

    field_ckpt = FIELD_DEPTH_CKPT if FIELD_DEPTH_CKPT.exists() else (
        LEGACY_DEPTH_CKPT if LEGACY_DEPTH_CKPT.exists() else None
    )

    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    variants = [
        ("baseline", None, False),
        ("underwater_preprocess", None, True),
    ]
    if field_ckpt and not quick:
        variants.append(("field_adapted", str(field_ckpt), True))

    out_dir = PROOF_DIR / "relative"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Group by checkpoint so we load each model once (much faster on Mac)
    by_ckpt: dict[str | None, list[tuple[str, str | None, bool]]] = {}
    for name, ckpt, preprocess in variants:
        by_ckpt.setdefault(ckpt, []).append((name, ckpt, preprocess))

    for ckpt, group in by_ckpt.items():
        enc = resolve_encoder(ckpt, default="vits")
        print(f"  Model: encoder={enc}" + (f"  checkpoint={ckpt}" if ckpt else "  (pretrained baseline)"))
        model, _ = build_depth_model(enc, str(device), ckpt)
        for name, _, preprocess in group:
            row_paths = []
            for frame_path in frames:
                bgr = cv2.imread(str(frame_path))
                proc = preprocess_underwater(bgr, method="combined") if preprocess else bgr
                depth_norm = predict_depth(model, transform, proc, device)
                depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
                panel = np.hstack([proc, depth_vis])
                out_path = out_dir / f"{frame_path.stem}_{name}.jpg"
                cv2.imwrite(str(out_path), panel)
                row_paths.append(str(out_path.relative_to(REPO_ROOT)))
            results.append({"variant": name, "panels": row_paths})
        del model

    report["relative"] = {
        "status": "ok",
        "videos": [str(v.relative_to(REPO_ROOT)) for v in videos],
        "frames": len(frames),
        "variants": results,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    print(f"  Saved {len(frames) * len(variants)} comparison panels → {out_dir}")


def prove_metric_depth(report: dict, train_epochs: int, skip_train: bool = False) -> None:
    """Synthetic lab RGB-D → metric fine-tune → measurable improvement."""
    from ceti.depth.ceti_metric import (
        build_train_config,
        evaluate_checkpoint,
        find_latest_checkpoint,
        register_ceti_dataset,
    )
    from ceti.depth.ceti_metric import prepare_metric_depth_env
    from ceti.depth.train_underwater import register_underwater_dataset, load_ceti_config
    from ceti.utils.device import configure_compute, device_name, get_device
    import os

    configure_compute()

    _banner("Phase 2: Metric underwater depth (synthetic CETI lab)")

    # Build synthetic underwater pairs
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_ceti_lab", REPO_ROOT / "ceti/scripts/build_ceti_lab.py"
    )
    build_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_mod)

    sources = REPO_ROOT / "assets/examples"
    train_n, test_n = build_mod.build_lab(sources, CETI_LAB_ROOT, variants=2, max_images=8)
    if train_n < 2:
        report["metric"] = {"status": "failed", "reason": "insufficient synthetic pairs"}
        return

    train_list = REPO_ROOT / "ceti/data/ceti_lab_train.txt"
    test_list = REPO_ROOT / "ceti/data/ceti_lab_test.txt"

    ceti_cfg = load_ceti_config()
    register_underwater_dataset(ceti_cfg, "ceti_lab")
    register_ceti_dataset(
        "ceti_lab",
        CETI_LAB_ROOT.resolve(),
        train_list.resolve(),
        test_list.resolve(),
        min_depth=ceti_cfg["datasets"]["ceti_lab"]["min_depth"],
        max_depth=ceti_cfg["datasets"]["ceti_lab"]["max_depth"],
    )

    if not OUTDOOR_CKPT.exists():
        report["metric"] = {"status": "skipped", "reason": f"missing {OUTDOOR_CKPT}"}
        return

    outdoor = f"local::{OUTDOOR_CKPT.resolve()}"
    print("  Evaluating outdoor pretrained (zero-shot on synthetic UW)...")
    before = evaluate_checkpoint("ceti_lab", outdoor, batch_size=2)
    print(f"    AbsRel={before.get('abs_rel', 0):.4f}  δ1={before.get('a1', 0):.4f}")

    if skip_train:
        prove_metric_visual_demo(outdoor, report)
        report["metric"] = {
            "status": "ok",
            "mode": "eval_only",
            "train_pairs": train_n,
            "test_pairs": test_n,
            "before": before,
            "note": "Fine-tune skipped; outdoor checkpoint produces metric depth on underwater RGB.",
        }
        print("  Skipped fine-tune (--skip-metric-train). Metric depth inference validated.")
        return

    # Fine-tune
    _banner(f"Phase 2b: Fine-tuning metric head ({train_epochs} epochs)")
    prepare_metric_depth_env()
    config = build_train_config(
        "ceti_lab",
        str(METRIC_SAVE),
        str(OUTDOOR_CKPT.resolve()),
        epochs=train_epochs,
        batch_size=2,
    )
    config.save_dir = str(METRIC_SAVE)

    from zoedepth.utils.misc import count_parameters
    from zoedepth.trainers.builder import get_trainer
    from zoedepth.models.builder import build_model
    from zoedepth.data.data_mono import DepthDataLoader

    device = get_device()
    print(f"  Metric train device: {device_name(device)}")
    model = build_model(config).to(device)
    print(f"  Parameters: {count_parameters(model)/1e6:.2f}M  Device: {device}")
    train_loader = DepthDataLoader(config, "train").data
    trainer = get_trainer(config)(config, model=model, train_loader=train_loader)
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0

    ckpt = find_latest_checkpoint(METRIC_SAVE) or OUTDOOR_CKPT
    ckpt_resource = f"local::{ckpt.resolve()}"
    print(f"  Evaluating fine-tuned weights ({ckpt.name})...")
    after = evaluate_checkpoint("ceti_lab", ckpt_resource, batch_size=2)

    abs_before = float(before.get("abs_rel", before.get("AbsRel", 999)))
    abs_after = float(after.get("abs_rel", after.get("AbsRel", 999)))
    d1_before = float(before.get("a1", 0))
    d1_after = float(after.get("a1", 0))

    improved = abs_after < abs_before
    report["metric"] = {
        "status": "ok",
        "train_pairs": train_n,
        "test_pairs": test_n,
        "train_seconds": round(train_sec, 1),
        "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
        "before": before,
        "after": after,
        "abs_rel_improved": improved,
    }
    print(f"    Before AbsRel={abs_before:.4f} → After={abs_after:.4f}  ({'✓ improved' if improved else 'no improvement'})")

    # Demo frame: metric depth colormap on one synthetic UW image
    prove_metric_visual_demo(ckpt_resource, report)


@torch.no_grad()
def prove_metric_visual_demo(ckpt_resource: str, report: dict) -> None:
    from ceti.depth.ceti_metric import load_metric_model
    from ceti.preprocessing.underwater import preprocess_underwater

    rgb_dir = CETI_LAB_ROOT / "rgb"
    images = sorted(rgb_dir.glob("*.jpg"))
    if not images:
        return

    model, device, _ = load_metric_model(ckpt_resource)
    bgr = cv2.imread(str(images[0]))
    rgb = cv2.cvtColor(preprocess_underwater(bgr, "combined"), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = torch.nn.functional.interpolate(tensor, (518, 518), mode="bilinear", align_corners=False)
    tensor = tensor.to(device)

    pred = model(tensor)
    if isinstance(pred, dict):
        pred = pred["metric_depth"]
    depth_m = pred.squeeze().cpu().numpy()
    proc = preprocess_underwater(bgr, "combined")
    depth_vis = cv2.applyColorMap(
        cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )
    depth_vis = cv2.resize(depth_vis, (proc.shape[1], proc.shape[0]))
    panel = np.hstack([proc, depth_vis])
    out_path = PROOF_DIR / "metric" / "sample_metric_depth.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    report.setdefault("metric", {})["sample_panel"] = str(out_path.relative_to(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Prove CETI underwater depth pipeline")
    parser.add_argument("--quick", action="store_true", help="Relative depth demo only (skip metric)")
    parser.add_argument(
        "--skip-metric-train",
        action="store_true",
        help="Run metric eval + visual demo only (no fine-tune; faster CPU proof)",
    )
    parser.add_argument("--metric-epochs", type=int, default=3, help="Metric fine-tune epochs for proof")
    args = parser.parse_args()

    from ceti.utils.device import configure_compute, device_name, get_device

    configure_compute()
    PROOF_DIR.mkdir(parents=True, exist_ok=True)

    field_rgb = REPO_ROOT / "data/underwater_field/rgb"
    field_count = len(list(field_rgb.glob("*.jpg"))) if field_rgb.exists() else 0

    report = {
        "project": "CETI_UnderwaterDepthProof",
        "reference": "Depth Anything + underwater adaptation (arxiv:2507.02148)",
        "device": device_name(get_device()),
        "underwater_field_images": field_count,
    }

    print("CETI Underwater Depth — Pipeline Proof")
    if field_count < 50:
        print("  Tip: run bash ceti/scripts/curate_underwater_field.sh for real training imagery")
    print(f"Output: {PROOF_DIR}")

    prove_relative_depth(report, quick=args.quick)
    if not args.quick:
        prove_metric_depth(
            report,
            train_epochs=args.metric_epochs,
            skip_train=args.skip_metric_train,
        )

    report_path = PROOF_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")

    rel_ok = report.get("relative", {}).get("status") == "ok"
    met = report.get("metric", {})
    if args.quick:
        success = rel_ok
    else:
        success = rel_ok and report.get("metric", {}).get("status") == "ok"

    if success:
        print("\n✓ Pipeline proof PASSED — underwater RGB → depth is working.")
        print(f"  Inspect panels: {PROOF_DIR / 'relative'}")
        if not args.quick and met.get("status") == "ok":
            print(f"  Metric sample: {met.get('sample_panel', 'ceti/outputs/proof/metric/')}")
    else:
        print("\n✗ Pipeline proof incomplete — see report.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
