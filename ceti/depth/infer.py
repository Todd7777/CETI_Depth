"""
CETI relative depth inference (Depth Anything backbone + fine-tuned weights).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

from ceti.bootstrap import REPO_ROOT, ensure_paths

ensure_paths()

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

VALID_ENCODERS = ("vits", "vitb", "vitl")


def resolve_encoder(
    checkpoint: str | Path | None,
    encoder: str | None = None,
    *,
    default: str = "vits",
) -> str:
    if encoder in VALID_ENCODERS:
        return encoder
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            enc = ckpt.get("encoder")
            if enc in VALID_ENCODERS:
                return enc
    return default


def _base_weights_path(encoder: str) -> Path:
    return REPO_ROOT / "checkpoints" / f"depth_anything_{encoder}14.pth"


def build_depth_model(
    encoder: str | None,
    device: str,
    checkpoint: str | None = None,
) -> tuple[torch.nn.Module, Compose]:
    encoder = resolve_encoder(checkpoint, encoder)
    model = DPT_DINOv2(encoder=encoder, localhub=True).to(device)

    base_ckpt = _base_weights_path(encoder)
    if base_ckpt.is_file():
        state = torch.load(base_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print(f"Loaded base Depth Anything weights: {base_ckpt.name}")
    else:
        print(f"WARNING: base weights missing ({base_ckpt}); run from fine-tune checkpoint only.")

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_enc = ckpt.get("encoder")
        if ckpt_enc in VALID_ENCODERS and ckpt_enc != encoder:
            raise RuntimeError(f"Checkpoint encoder={ckpt_enc} but model built as {encoder}.")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded CETI checkpoint: {ckpt_path} (encoder={encoder})")

    model.eval()
    transform = Compose([
        Resize(
            width=518,
            height=518,
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


@torch.no_grad()
def predict_depth_raw(
    model: torch.nn.Module,
    transform: Compose,
    bgr_image: np.ndarray,
    device: str,
) -> np.ndarray:
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) / 255.0
    tensor = transform({"image": rgb})["image"]
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
    depth = model(tensor)
    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[0, 0]
    return depth.float().cpu().numpy()
