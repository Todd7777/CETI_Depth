#!/usr/bin/env python3
"""
CETI smoke test — verify installation and basic inference.

Runs without GPU if unavailable. Downloads vits model on first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_imports():
    print("[1/8] Testing imports...")
    import torch
    import cv2
    import numpy as np
    from depth_anything.dpt import DepthAnything
    from ceti.preprocessing.underwater import preprocess_underwater
    from ceti.robot.avatars_pipeline import AVATARSPipeline, WhaleDetection
    from ceti.utils.device import configure_compute, device_name, get_device

    configure_compute()
    print(f"  PyTorch {torch.__version__}, {device_name(get_device())}")
    print("  ✓ All imports OK")


def test_underwater_preprocess():
    print("[2/8] Testing underwater preprocessing...")
    import cv2
    import numpy as np
    from ceti.preprocessing.underwater import preprocess_underwater

    # Synthetic blue-green underwater image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 30   # low red
    img[:, :, 1] = 80   # moderate green
    img[:, :, 2] = 120  # high blue

    result = preprocess_underwater(img, method="combined")
    assert result.shape == img.shape
    assert result[:, :, 0].mean() > img[:, :, 0].mean(), "Red channel should be boosted"
    print("  ✓ Underwater preprocessing OK")


def test_depth_inference():
    print("[3/8] Testing depth inference (may download model)...")
    import cv2
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import Compose
    from depth_anything.dpt import DepthAnything
    from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
    from ceti.utils.device import configure_compute, get_device

    configure_compute()
    device = str(get_device())
    encoder = "vits"  # smallest for smoke test

    model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14").to(device).eval()

    transform = Compose([
        Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True,
               ensure_multiple_of=14, resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    import numpy as np

    # Use example image if available, else synthetic
    example = REPO_ROOT / "assets/examples/dog.jpg"
    if example.exists():
        raw = cv2.imread(str(example))
    else:
        raw = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    h, w = raw.shape[:2]
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) / 255.0
    tensor = transform({"image": rgb})["image"]
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(tensor)

    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
    assert depth.shape == (h, w)
    print(f"  ✓ Depth inference OK (output shape: {tuple(depth.shape)})")


def test_avatars_pipeline():
    print("[4/8] Testing AVATARS fusion pipeline...")
    from ceti.robot.avatars_pipeline import AVATARSPipeline, WhaleDetection, DiveState, TagAOA

    pipeline = AVATARSPipeline()
    detections = [
        WhaleDetection(bbox=[800, 200, 1100, 400], confidence=0.85, class_name="sperm_whale", range_estimate_m=200),
    ]
    dive = DiveState(phase="ascent", predicted_surface_time_s=120, predicted_surface_lat=15.42, predicted_surface_lon=-61.38)
    tag = TagAOA(azimuth_deg=15.0, signal_strength_dbm=-70, tag_id="SW-001")

    targets = pipeline.fuse(detections, tag_aoa=tag, dive_state=dive, drone_lat=15.41, drone_lon=-61.39, image_width=1920)
    assert len(targets) > 0

    route = pipeline.plan_rendezvous_route(targets, drone_lat=15.41, drone_lon=-61.39)
    assert len(route) >= 2
    print(f"  ✓ AVATARS fusion OK ({len(targets)} targets, {len(route)} waypoints)")


def test_whale_depth_pipeline():
    print("[5/8] Testing whale depth fine-tune pipeline (dry-run)...")
    import subprocess

    train_list = REPO_ROOT / "ceti/data/whale_depth_train.txt"
    config = REPO_ROOT / "ceti/configs/whale_depth_m5max_128gb.yaml"
    if not train_list.exists():
        print("  ⚠ Skipped (no whale_depth_train.txt)")
        return

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ceti/depth/train_whale_depth.py"),
            "--config",
            str(config),
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print("  ✓ Whale depth dry-run OK")


def test_underwater_train_config():
    print("[6/8] Testing underwater metric train config (dry-run)...")
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "ceti/depth/train_underwater.py"), "--dataset", "flsea", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    print("  ✓ Underwater train dry-run OK")


def test_prove_quick():
    print("[7/8] Testing pipeline proof (--quick)...")
    import os
    import subprocess

    env = os.environ.copy()
    env.setdefault("CETI_DEVICE", "mps")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "ceti/scripts/prove_pipeline.py"), "--quick"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        raise RuntimeError(tail)
    report = REPO_ROOT / "ceti/outputs/proof/report.json"
    if not report.exists():
        raise FileNotFoundError("proof report.json not written")
    print("  ✓ Pipeline proof (--quick) OK")


def test_whale_dataset():
    print("[8/8] Testing whale dataset utilities...")
    from ceti.whale.dataset import CLASS_NAMES, bbox_to_yolo_label, yolo_label_to_bbox

    label = bbox_to_yolo_label(1, [100, 50, 300, 250], 640, 480)
    cls_id, bbox = yolo_label_to_bbox(label, 640, 480)
    assert cls_id == 1
    assert abs(bbox[0] - 100) < 1
    print(f"  ✓ Whale dataset utils OK (classes: {list(CLASS_NAMES.values())})")


def main():
    print("=" * 60)
    print("CETI Smoke Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_underwater_preprocess,
        test_depth_inference,
        test_avatars_pipeline,
        test_whale_depth_pipeline,
        test_underwater_train_config,
        test_prove_quick,
        test_whale_dataset,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("=" * 60)
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
