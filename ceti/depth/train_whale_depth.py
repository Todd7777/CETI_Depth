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

import numpy as np
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
    configure_mps_for_training,
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


def _resolve_freeze_encoder(cfg: dict, is_resume: bool, args) -> bool:
    if args.no_freeze_encoder:
        return False
    if args.freeze_encoder:
        return True
    if is_resume and "resume_freeze_encoder" in cfg:
        return bool(cfg["resume_freeze_encoder"])
    return bool(cfg.get("freeze_encoder", False))


def _apply_resume_lr(cfg: dict, lr: float, encoder_lr: float, multiplier: float) -> tuple[float, float]:
    return lr * multiplier, encoder_lr * multiplier


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


def _batch_teacher_depth(batch: dict, device: torch.device) -> torch.Tensor | None:
    td = batch.get("teacher_depth")
    if td is None:
        return None
    if isinstance(td, torch.Tensor):
        t = td.to(device, dtype=torch.float32)
    else:
        t = torch.from_numpy(np.stack(td)).to(device, dtype=torch.float32)
    if t.dim() == 3:
        t = t.unsqueeze(1)
    return t


def _teacher_forward(
    teacher,
    images: torch.Tensor,
    device: torch.device,
    teacher_device: str,
    use_amp: bool,
) -> torch.Tensor:
    teacher_in = images if teacher_device == str(device) else images.to(teacher_device)
    with torch.autocast(
        device_type=autocast_device_type(torch.device(teacher_device)),
        enabled=use_amp and teacher_device != "cpu",
    ):
        target = predict_depth(teacher, teacher_in)
    if teacher_device != str(device):
        target = target.to(device)
    return target


def validate(
    student,
    teacher,
    loader: DataLoader,
    device: torch.device,
    ss_loss: ScaleShiftInvariantLoss,
    grad_loss: GradientMatchingLoss,
    w_grad: float,
    *,
    teacher_device: str = "mps",
    use_amp: bool = True,
) -> float:
    student.eval()
    total = 0.0
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        with torch.no_grad():
            target = _batch_teacher_depth(batch, device)
            if target is None:
                target = _teacher_forward(teacher, images, device, teacher_device, use_amp)
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
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=None,
        help="First epoch to run (default: checkpoint epoch + 1 when --resume)",
    )
    parser.add_argument(
        "--resume-lr-multiplier",
        type=float,
        default=None,
        help="Multiply lr/encoder_lr when resuming without optimizer state (config: resume_lr_multiplier)",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=None,
        help="Freeze ViT backbone (overrides config; default on safe resume)",
    )
    parser.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Train full encoder even when resuming",
    )
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

    listed_paths = load_image_paths(train_list)
    train_paths = [p for p in listed_paths if p.is_file()]

    if len(listed_paths) == 0:
        print("\nERROR: Training list is empty.")
        print("Run: bash ceti/scripts/ensure_training_data.sh")
        sys.exit(1)

    if args.dry_run:
        print(f"  List entries: {len(listed_paths)}")
        print(f"  On disk:      {len(train_paths)}")
        if len(train_paths) == 0:
            print(
                "  NOTE: No image files yet — OK for dry-run. "
                "Before training run: bash ceti/scripts/ensure_training_data.sh"
            )
        elif len(train_paths) < len(listed_paths):
            print(f"  WARNING:    {len(listed_paths) - len(train_paths)} list paths missing locally")
        print("=" * 60)
        print("Dry run complete — config and train list validated.")
        return

    if len(train_paths) == 0:
        print("\nERROR: No training image files on disk.")
        print("Run: bash ceti/scripts/ensure_training_data.sh")
        sys.exit(1)

    if len(train_paths) < len(listed_paths):
        print(f"  WARNING:    {len(listed_paths) - len(train_paths)} list entries missing locally")

    print(f"  Images:     {len(train_paths)} train (files on disk)")
    print("=" * 60)

    import os

    configure_mps_for_training()
    n_threads = configure_compute()
    prefer = cfg.get("device") or None
    if prefer:
        os.environ.setdefault("CETI_DEVICE", str(prefer))
    device = require_mps_for_training()
    print(f"Device: {device_name(device)}  (threads={n_threads})")
    amp_dtype = autocast_device_type(device)

    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    image_size = cfg.get("image_size", 518)
    val_every = int(cfg.get("val_every", 1))

    # Optional: CETI_TEACHER_ON_CPU=1 saves MPS memory (slower teacher forward)
    teacher_device = str(device)
    if os.environ.get("CETI_TEACHER_ON_CPU", "").strip() in ("1", "true", "yes"):
        teacher_device = "cpu"
        print("  Teacher on CPU (CETI_TEACHER_ON_CPU=1) — saves MPS memory")

    use_cache = cfg.get("cache_teacher", False) or os.environ.get("CETI_CACHE_TEACHER", "").strip() in (
        "1",
        "true",
        "yes",
    )
    cache_dir = resolve_path(cfg.get("teacher_cache_dir", f"./data/teacher_cache/{encoder}_{image_size}"))

    teacher = build_model(encoder, hub_id, teacher_device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    if use_cache:
        from ceti.depth.teacher_cache import count_cached, precompute_teacher_cache

        cache_batch = cfg.get("cache_batch_size", 14)
        cache_paths = list(train_paths)
        if val_list.exists():
            cache_paths.extend(p for p in load_image_paths(val_list) if p.is_file())
        have = count_cached(cache_dir, cache_paths)
        if have < len(cache_paths):
            precompute_teacher_cache(
                teacher,
                cache_paths,
                cache_dir,
                torch.device(teacher_device),
                image_size=image_size,
                preprocess_method=cfg.get("preprocess_method", "combined"),
                batch_size=cache_batch,
                use_amp=cfg.get("use_amp", True),
                hub_encoder=encoder,
            )
        del teacher
        teacher = None
        empty_cache(device)
        print(f"  Speed mode: cached teacher depth ({cache_dir})")

    student = build_model(encoder, hub_id, str(device))
    use_amp = cfg.get("use_amp", False) and device.type in ("cuda", "mps")

    if os.environ.get("CETI_TORCH_COMPILE", "").strip() in ("1", "true", "yes"):
        try:
            student = torch.compile(student)
            print("  torch.compile enabled on student")
        except Exception as e:
            print(f"  torch.compile skipped: {e}")
    start_epoch = 1
    best_val = float("inf")
    best_train = float("inf")
    resume_ckpt: dict | None = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state = resume_ckpt.get("model", resume_ckpt)
        student.load_state_dict(state, strict=False)
        done = int(resume_ckpt.get("epoch", 0))
        start_epoch = args.start_epoch if args.start_epoch is not None else done + 1
        resume_loss = float(resume_ckpt.get("loss", float("inf")))
        if resume_ckpt.get("metric") == "val":
            best_val = resume_loss
        else:
            best_train = resume_loss
        print(f"Resumed weights from {args.resume} (completed epoch {done})")
        print(f"  Training epochs {start_epoch}–{epochs}")

    train_augment = cfg.get("marine_augment", True) and not use_cache
    if use_cache and cfg.get("marine_augment", True):
        print("  Note: marine_augment disabled when cache_teacher=true (paired flip only)")

    train_ds = WhaleDepthDataset(
        train_list,
        image_size=image_size,
        preprocess_method=cfg.get("preprocess_method", "none"),
        augment=train_augment or use_cache,
        teacher_cache_dir=cache_dir if use_cache else None,
    )
    workers = optimal_dataloader_workers(cfg.get("workers"), device=device)
    print(f"  Batch size: {batch_size}  DataLoader workers: {workers}")
    if device.type == "mps" and batch_size > 12:
        print("  Tip: if training is 'Killed: 9', lower batch_size to 8 in config or --batch-size 8")

    loader_kw: dict = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory_for_device(device),
        persistent_workers=workers > 0,
        drop_last=len(train_ds) >= batch_size,
    )
    if workers > 0:
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **loader_kw)

    val_loader = None
    if val_list.exists() and len(load_image_paths(val_list)) > 0:
        val_ds = WhaleDepthDataset(
            val_list,
            image_size=image_size,
            preprocess_method=cfg.get("preprocess_method", "none"),
            augment=False,
            teacher_cache_dir=cache_dir if use_cache else None,
        )
        val_batch = int(cfg.get("val_batch_size", min(batch_size, 4)))
        val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=0)

    is_resume = resume_ckpt is not None
    freeze_encoder = _resolve_freeze_encoder(cfg, is_resume, args)
    lr = float(cfg["lr"])
    encoder_lr = float(cfg.get("encoder_lr", lr * 0.1))
    optimizer_restored = False

    if is_resume and resume_ckpt is not None:
        ckpt_freeze = resume_ckpt.get("freeze_encoder")
        has_optimizer = "optimizer" in resume_ckpt
        structures_match = has_optimizer and ckpt_freeze is not None and bool(ckpt_freeze) == freeze_encoder

        if structures_match:
            lr = float(resume_ckpt.get("lr", lr))
            encoder_lr = float(resume_ckpt.get("encoder_lr", encoder_lr))
        else:
            mult = args.resume_lr_multiplier
            if mult is None:
                mult = float(cfg.get("resume_lr_multiplier", 0.25))
            lr, encoder_lr = _apply_resume_lr(cfg, lr, encoder_lr, mult)
            if has_optimizer and not structures_match:
                print(
                    f"  Resume: optimizer state skipped (freeze_encoder {ckpt_freeze} → {freeze_encoder})"
                )
            elif not has_optimizer:
                print(
                    f"  Resume: no optimizer in checkpoint — lr×{mult}, freeze_encoder={freeze_encoder}"
                )

    optimizer = torch.optim.AdamW(
        param_groups(student, lr=lr, encoder_lr=encoder_lr, freeze_encoder=freeze_encoder),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    if is_resume and resume_ckpt is not None and "optimizer" in resume_ckpt:
        ckpt_freeze = resume_ckpt.get("freeze_encoder")
        if ckpt_freeze is not None and bool(ckpt_freeze) == freeze_encoder:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                optimizer_restored = True
                print("  Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"  WARNING: could not load optimizer state ({e})")

    print(
        f"  Optimizer: lr={lr:.2e} encoder_lr={encoder_lr:.2e} "
        f"freeze_encoder={freeze_encoder} restored={optimizer_restored}"
    )

    ss_loss = ScaleShiftInvariantLoss()
    grad_loss = GradientMatchingLoss()
    w_grad = cfg.get("w_gradient", 0.4)

    if teacher is None and not use_cache:
        raise RuntimeError("Teacher model required when cache_teacher is false")

    best_ckpt = save_dir / "best.pt"
    if best_ckpt.exists():
        try:
            b = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            b_loss = float(b.get("loss", float("inf")))
            if b.get("metric") == "val":
                best_val = min(best_val, b_loss)
            else:
                best_train = min(best_train, b_loss)
        except Exception:
            pass

    if start_epoch > epochs:
        print(f"Nothing to do: start_epoch={start_epoch} > epochs={epochs}")
        return

    max_train_loss = float(cfg.get("max_train_loss", 6.0))

    for epoch in range(start_epoch, epochs + 1):
        student.train()
        epoch_loss = 0.0
        n_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                target = _batch_teacher_depth(batch, device)
                if target is None:
                    target = _teacher_forward(teacher, images, device, teacher_device, use_amp)

            with torch.autocast(device_type=amp_dtype, enabled=use_amp):
                pred = predict_depth(student, images)
                loss = cfg.get("w_scale_shift", 1.0) * ss_loss(pred, target)
                loss = loss + w_grad * grad_loss(pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            step_loss = loss.item()
            if not np.isfinite(step_loss):
                print(f"\nERROR: non-finite loss at epoch {epoch} — stopping. Restore from best.pt")
                sys.exit(1)
            epoch_loss += step_loss
            n_steps += 1
            pbar.set_postfix(loss=f"{step_loss:.4f}")

        empty_cache(device)

        avg_train = epoch_loss / max(n_steps, 1)
        if avg_train > max_train_loss:
            print(
                f"\nERROR: epoch {epoch} train_loss={avg_train:.4f} > {max_train_loss} (diverged). "
                f"Stopping without overwriting checkpoints. Resume from best.pt:\n"
                f"  bash ceti/scripts/resume_mac_train.sh\n"
            )
            sys.exit(1)
        avg_val = None
        if val_loader is not None and (epoch % val_every == 0 or epoch == epochs):
            empty_cache(device)
            avg_val = validate(
                student,
                teacher,
                val_loader,
                device,
                ss_loss,
                grad_loss,
                w_grad,
                teacher_device=teacher_device,
                use_amp=use_amp,
            )
            print(f"Epoch {epoch}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
            if avg_val < best_val:
                best_val = avg_val
                _save_checkpoint(
                    save_dir / "best.pt",
                    student,
                    cfg,
                    encoder,
                    epoch,
                    avg_val,
                    metric="val",
                )
        else:
            print(f"Epoch {epoch}: train_loss={avg_train:.4f}")
            # Only update best.pt on train loss when we are not tracking val (avoids
            # overwriting a better val checkpoint on non-val epochs).
            if val_loader is None and avg_train < best_train:
                best_train = avg_train
                _save_checkpoint(
                    save_dir / "best.pt",
                    student,
                    cfg,
                    encoder,
                    epoch,
                    avg_train,
                    metric="train",
                )

        empty_cache(device)
        _save_checkpoint(
            save_dir / "last.pt",
            student,
            cfg,
            encoder,
            epoch,
            avg_train,
            metric="train",
            optimizer=optimizer,
            freeze_encoder=freeze_encoder,
            lr=lr,
            encoder_lr=encoder_lr,
            save_optimizer=True,
        )

    print(f"\nTraining complete. Checkpoints in {save_dir}")
    print("Inference:")
    print(f"  python ceti/depth/infer_robot.py --depth-checkpoint {save_dir}/best.pt --encoder {encoder}")


def _save_checkpoint(
    path: Path,
    model,
    cfg: dict,
    encoder: str,
    epoch: int,
    loss: float,
    *,
    metric: str = "train",
    optimizer: torch.optim.Optimizer | None = None,
    freeze_encoder: bool = False,
    lr: float | None = None,
    encoder_lr: float | None = None,
    save_optimizer: bool = False,
):
    payload: dict = {
        "model": model.state_dict(),
        "encoder": encoder,
        "epoch": epoch,
        "loss": loss,
        "metric": metric,
        "freeze_encoder": freeze_encoder,
        "config": cfg,
        "type": "ceti_underwater_field",
    }
    if lr is not None:
        payload["lr"] = lr
    if encoder_lr is not None:
        payload["encoder_lr"] = encoder_lr
    if save_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


if __name__ == "__main__":
    main()
