#!/usr/bin/env python3
"""
Fine-tune Depth Anything on whale / marine imagery (domain not in pretraining).

Uses frozen teacher pseudo-depth + scale-shift invariant distillation so no
metric depth labels are required. Pair with Track B (YOLO whale detection) for
AVATARS range estimation.

Usage:
    bash ceti/scripts/prepare_whale_depth_data.sh
    python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth.yaml
    python ceti/depth/train_whale_depth.py --epochs 5 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ceti.depth.losses import GradientMatchingLoss, ScaleShiftInvariantLoss
from ceti.depth.whale_depth_dataset import WhaleDepthDataset, load_image_paths
from ceti.utils.device import (
    autocast_device_type,
    configure_compute,
    device_name,
    empty_cache,
    optimal_dataloader_workers,
    pin_memory_for_device,
    require_mps_for_training,
)


def load_config(path: Path | None) -> dict:
    cfg_path = path or REPO_ROOT / "ceti/configs/whale_depth.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def build_model(encoder: str, hub_id: str, device: str):
    from depth_anything.dpt import DepthAnything

    model = DepthAnything.from_pretrained(hub_id).to(device)
    return model


def param_groups(model, lr: float, encoder_lr: float, freeze_encoder: bool):
    if freeze_encoder:
        for p in model.pretrained.parameters():
            p.requires_grad = False
        return [{"params": model.depth_head.parameters(), "lr": lr}]

    return [
        {"params": model.pretrained.parameters(), "lr": encoder_lr},
        {"params": model.depth_head.parameters(), "lr": lr},
    ]


def predict_depth(model, images: torch.Tensor) -> torch.Tensor:
    return model(images)


def validate(
    student,
    teacher,
    loader: DataLoader,
    device: str,
    ss_loss: ScaleShiftInvariantLoss,
    grad_loss: GradientMatchingLoss,
    w_grad: float,
) -> float:
    student.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        with torch.no_grad():
            target = predict_depth(teacher, images)
        pred = predict_depth(student, images)
        loss = ss_loss(pred, target) + w_grad * grad_loss(pred, target)
        total += loss.item()
        n += 1
    student.train()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="CETI whale/marine Depth Anything fine-tuning")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--encoder", type=str, default=None, choices=["vits", "vitb", "vitl"])
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--train-list", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint .pt to resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    encoder = args.encoder or cfg["encoder"]
    hub_id = cfg.get("pretrained_model") or f"LiheYoung/depth_anything_{encoder}14"
    if "{encoder}" in hub_id:
        hub_id = hub_id.format(encoder=encoder)

    train_list = resolve_path(args.train_list or cfg["train_list"])
    val_list = resolve_path(cfg["val_list"])
    save_dir = resolve_path(args.save_dir or cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CETI Whale / Marine Depth Fine-Tuning")
    print("=" * 60)
    print(f"  Encoder:    {encoder}")
    print(f"  Teacher:    {hub_id} (frozen)")
    print(f"  Train list: {train_list}")
    print(f"  Save dir:   {save_dir}")

    if not train_list.exists():
        print(f"\nERROR: Training file list not found: {train_list}")
        print("Run: bash ceti/scripts/curate_underwater_field.sh")
        sys.exit(1)

    train_paths = load_image_paths(train_list, exist_only=True)
    if len(train_paths) == 0:
        print("\nERROR: No training image files on disk.")
        print("Run: bash ceti/scripts/ensure_training_data.sh")
        sys.exit(1)

    listed = len(load_image_paths(train_list))
    if listed > len(train_paths):
        print(f"  WARNING:    {listed - len(train_paths)} list entries missing locally")

    print(f"  Images:     {len(train_paths)} train (files on disk)")
    print("=" * 60)

    if args.dry_run:
        print("Dry run complete — config and data validated.")
        return

    n_threads = configure_compute()
    prefer = cfg.get("device") or None
    if prefer:
        import os
        os.environ.setdefault("CETI_DEVICE", str(prefer))
    device = require_mps_for_training()
    print(f"Device: {device_name(device)}  (threads={n_threads})")
    amp_dtype = autocast_device_type(device)

    teacher = build_model(encoder, hub_id, str(device)).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = build_model(encoder, hub_id, str(device))
    use_amp = cfg.get("use_amp", False) and device.type in ("cuda", "mps")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        student.load_state_dict(state, strict=False)
        print(f"Resumed from {args.resume}")

    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    image_size = cfg.get("image_size", 518)

    train_ds = WhaleDepthDataset(
        train_list,
        image_size=image_size,
        preprocess_method=cfg.get("preprocess_method", "none"),
        augment=cfg.get("marine_augment", True),
    )
    workers = optimal_dataloader_workers(cfg.get("workers"))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory_for_device(device),
        persistent_workers=workers > 0,
        drop_last=len(train_ds) >= batch_size,
    )

    val_loader = None
    if val_list.exists() and len(load_image_paths(val_list)) > 0:
        val_ds = WhaleDepthDataset(
            val_list,
            image_size=image_size,
            preprocess_method=cfg.get("preprocess_method", "none"),
            augment=False,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        param_groups(
            student,
            lr=cfg["lr"],
            encoder_lr=cfg.get("encoder_lr", cfg["lr"] * 0.1),
            freeze_encoder=cfg.get("freeze_encoder", False),
        ),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    ss_loss = ScaleShiftInvariantLoss()
    grad_loss = GradientMatchingLoss()
    w_grad = cfg.get("w_gradient", 0.4)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with torch.autocast(device_type=amp_dtype, enabled=use_amp):
                    target = predict_depth(teacher, images)

            with torch.autocast(device_type=amp_dtype, enabled=use_amp):
                pred = predict_depth(student, images)
                loss = cfg.get("w_scale_shift", 1.0) * ss_loss(pred, target)
                loss = loss + w_grad * grad_loss(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        empty_cache(device)

        avg_train = epoch_loss / len(train_loader)
        avg_val = None
        if val_loader is not None:
            avg_val = validate(student, teacher, val_loader, device, ss_loss, grad_loss, w_grad)
            print(f"Epoch {epoch}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                _save_checkpoint(save_dir / "best.pt", student, cfg, encoder, epoch, avg_val)
        else:
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}")

        _save_checkpoint(save_dir / "last.pt", student, cfg, encoder, epoch, avg_train)

    print(f"\nTraining complete. Checkpoints in {save_dir}")
    print("Inference:")
    print(f"  python ceti/depth/infer_robot.py --depth-checkpoint {save_dir}/best.pt --encoder {encoder}")


def _save_checkpoint(path: Path, model, cfg: dict, encoder: str, epoch: int, loss: float):
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": encoder,
            "epoch": epoch,
            "loss": loss,
            "config": cfg,
            "type": "ceti_underwater_field",
        },
        path,
    )


if __name__ == "__main__":
    main()
