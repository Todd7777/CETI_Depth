#!/usr/bin/env python3
"""
Unified robot inference: underwater depth + whale detection.

Supports image, video, webcam, and optional ROS2 publishing.
Designed for CETI lab robots and AVATARS aerial drones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from ceti.preprocessing.underwater import preprocess_underwater
from ceti.utils.device import configure_compute, device_name, get_device

VALID_ENCODERS = ("vits", "vitb", "vitl")


def resolve_encoder(
    checkpoint: str | Path | None,
    encoder: str | None = None,
    *,
    default: str = "vits",
) -> str:
    """
    Pick ViT variant for Depth Anything.

    CETI checkpoints store ``encoder`` (vitl for whale training). If a checkpoint
    path is given, its metadata wins unless ``encoder`` is set explicitly.
    """
    if encoder in VALID_ENCODERS:
        return encoder
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            enc = ckpt.get("encoder")
            if enc in VALID_ENCODERS:
                return enc
    return default


def build_depth_model(
    encoder: str | None,
    device: str,
    checkpoint: str | None = None,
) -> tuple[torch.nn.Module, Compose]:
    encoder = resolve_encoder(checkpoint, encoder)
    model = DepthAnything.from_pretrained(
        f"LiheYoung/depth_anything_{encoder}14"
    ).to(device)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_enc = ckpt.get("encoder")
            if ckpt_enc in VALID_ENCODERS and ckpt_enc != encoder:
                raise RuntimeError(
                    f"Checkpoint encoder={ckpt_enc} but model built as {encoder}. "
                    "Pass matching --encoder or omit it for auto-detect."
                )
            state = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(
                    f"WARNING: depth state_dict mismatch "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            tag = ckpt.get("type", "checkpoint")
            epoch = ckpt.get("epoch", "?")
            print(
                f"Loaded depth weights: {ckpt_path} ({tag}, encoder={encoder}, epoch={epoch})"
            )
        else:
            print(f"WARNING: depth checkpoint not found: {ckpt_path}")
    else:
        print(f"Using pretrained Depth Anything ({encoder}), no CETI checkpoint")

    model.eval()

    transform = Compose([
        Resize(
            width=518, height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return model, transform


def build_whale_detector(checkpoint: str | None):
    if checkpoint is None or not Path(checkpoint).exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(checkpoint)
    except ImportError:
        print("WARNING: ultralytics not installed; whale detection disabled")
        return None


@torch.no_grad()
def predict_depth_raw(
    model: torch.nn.Module,
    transform: Compose,
    bgr_image: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return raw relative depth (HxW float32, model units — not meters)."""
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) / 255.0
    tensor = transform({"image": rgb})["image"]
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

    depth = model(tensor)
    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
    return depth.float().cpu().numpy()


@torch.no_grad()
def predict_depth(
    model: torch.nn.Module,
    transform: Compose,
    bgr_image: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return normalized depth map (HxW, float32, 0-255)."""
    depth = predict_depth_raw(model, transform, bgr_image, device)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    return depth_norm.astype(np.float32)


def predict_whales(detector, bgr_image: np.ndarray, conf: float = 0.5) -> list[dict]:
    if detector is None:
        return []
    results = detector(bgr_image, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
            })
    return detections


def visualize(
    bgr_image: np.ndarray,
    depth_norm: np.ndarray,
    detections: list[dict],
) -> np.ndarray:
    depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    vis = bgr_image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Depth at bbox center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cy < depth_norm.shape[0] and 0 <= cx < depth_norm.shape[1]:
            rel_depth = depth_norm[cy, cx] / 255.0
            cv2.putText(vis, f"rel_depth={rel_depth:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    combined = np.hstack([vis, depth_vis])
    return combined


def process_frame(
    frame: np.ndarray,
    depth_model,
    transform,
    whale_detector,
    device: str,
    underwater_preprocess: bool,
    preprocess_method: str,
    whale_conf: float,
) -> tuple[np.ndarray, list[dict]]:
    if underwater_preprocess:
        frame = preprocess_underwater(frame, method=preprocess_method)

    depth_norm = predict_depth(depth_model, transform, frame, device)
    detections = predict_whales(whale_detector, frame, conf=whale_conf)
    vis = visualize(frame, depth_norm, detections)
    return vis, detections


def main():
    parser = argparse.ArgumentParser(description="CETI robot perception inference")
    parser.add_argument("--input", type=str, default="0", help="Image, video, directory, or camera index")
    parser.add_argument("--outdir", type=str, default="./ceti/outputs/inference")
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        choices=["vits", "vitb", "vitl"],
        help="ViT size (default: read from --depth-checkpoint, else vits)",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=str,
        default=None,
        help="CETI fine-tuned relative Depth Anything weights",
    )
    parser.add_argument(
        "--metric-checkpoint",
        type=str,
        default=None,
        help="CETI fine-tuned metric ZoeDepth weights (meters); uses outdoor head if unset",
    )
    parser.add_argument("--whale-checkpoint", type=str, default=None)
    parser.add_argument("--underwater-preprocess", action="store_true")
    parser.add_argument("--preprocess-method", type=str, default="combined")
    parser.add_argument("--whale-conf", type=float, default=0.5)
    parser.add_argument("--publish-ros", action="store_true", help="Publish to ROS2 topics (requires rclpy)")
    args = parser.parse_args()

    configure_compute()
    device = get_device()
    print(f"Device: {device_name(device)}")

    depth_enc = resolve_encoder(args.depth_checkpoint, args.encoder)
    depth_model, transform = build_depth_model(depth_enc, str(device), args.depth_checkpoint)
    whale_detector = build_whale_detector(args.whale_checkpoint)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = args.input

    # Camera
    if input_path.isdigit():
        cap = cv2.VideoCapture(int(input_path))
        if not cap.isOpened():
            print(f"Cannot open camera {input_path}")
            sys.exit(1)

        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, _ = process_frame(
                frame, depth_model, transform, whale_detector, device,
                args.underwater_preprocess, args.preprocess_method, args.whale_conf,
            )
            cv2.imshow("CETI Perception", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    path = Path(input_path)

    # Single image
    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        frame = cv2.imread(str(path))
        vis, dets = process_frame(
            frame, depth_model, transform, whale_detector, device,
            args.underwater_preprocess, args.preprocess_method, args.whale_conf,
        )
        out_path = outdir / f"{path.stem}_ceti.png"
        cv2.imwrite(str(out_path), vis)
        print(f"Saved {out_path} ({len(dets)} whale detections)")
        return

    # Video
    if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = outdir / f"{path.stem}_ceti.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))

        frame_idx = 0
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis, _ = process_frame(
                frame, depth_model, transform, whale_detector, device,
                args.underwater_preprocess, args.preprocess_method, args.whale_conf,
            )
            writer.write(vis)
            frame_idx += 1

        elapsed = time.time() - t0
        cap.release()
        writer.release()
        print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
        print(f"Saved {out_path}")
        return

    print(f"Unsupported input: {input_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
