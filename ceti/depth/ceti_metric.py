"""
CETI helpers for ZoeDepth / metric depth (underwater fine-tuning and evaluation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
METRIC_DEPTH = REPO_ROOT / "metric_depth"


def _ensure_zoe_path() -> None:
    if str(METRIC_DEPTH) not in sys.path:
        sys.path.insert(0, str(METRIC_DEPTH))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def prepare_metric_depth_env() -> str:
    """
    ZoeDepth metric code expects cwd=metric_depth with ./checkpoints and ./torchhub.

    Returns:
        Previous working directory (caller should restore after loading metric model).
    """
    prev = os.getcwd()
    os.chdir(METRIC_DEPTH)
    for name, target in (
        ("checkpoints", REPO_ROOT / "checkpoints"),
        ("torchhub", REPO_ROOT / "torchhub"),
    ):
        link = METRIC_DEPTH / name
        if not target.exists():
            raise FileNotFoundError(f"Missing {name} dir: {target}")
        if link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)
    return prev


def register_ceti_dataset(
    name: str,
    data_path: Path,
    train_list: Path,
    test_list: Path,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
    input_height: int = 518,
    input_width: int = 518,
) -> None:
    """Register a CETI RGB-D dataset with ZoeDepth (depth PNG = uint16 millimeters)."""
    _ensure_zoe_path()
    from zoedepth.utils import config as zoe_config

    _orig = zoe_config.check_choices

    def _check(name_key, value, choices):
        if name_key == "Dataset" and value in zoe_config.DATASETS_CONFIG:
            return
        return _orig(name_key, value, choices)

    zoe_config.check_choices = _check

    zoe_config.DATASETS_CONFIG[name] = {
        "dataset": "nyu",
        "min_depth": min_depth,
        "max_depth": max_depth,
        "data_path": str(data_path),
        "gt_path": str(data_path),
        "filenames_file": str(train_list),
        "input_height": input_height,
        "input_width": input_width,
        "data_path_eval": str(data_path),
        "gt_path_eval": str(data_path),
        "filenames_file_eval": str(test_list),
        "min_depth_eval": min_depth,
        "max_depth_eval": max_depth,
        "do_random_rotate": True,
        "degree": 5.0,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": False,
        "avoid_boundary": False,
    }


def build_train_config(
    dataset: str,
    save_dir: str,
    pretrained_metric: str,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
) -> "edict":
    _ensure_zoe_path()
    from zoedepth.utils.config import get_config

    pretrained_resource = pretrained_metric
    if pretrained_resource and not pretrained_resource.startswith(("local::", "url::")):
        pretrained_resource = f"local::{pretrained_resource}"

    return get_config(
        "zoedepth",
        "train",
        dataset=dataset,
        config_version="zoedepth",
        save_dir=save_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        pretrained_resource=pretrained_resource,
        tags=f"ceti,{dataset}",
        notes=f"CETI metric depth on {dataset}",
        aug=True,
        random_crop=False,
    )


@torch.no_grad()
def _infer_metric(model, images: torch.Tensor) -> torch.Tensor:
    pred = model(images)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    return pred


def evaluate_checkpoint(
    dataset: str,
    pretrained_resource: str,
    batch_size: int = 4,
) -> dict[str, float]:
    """Return AbsRel, RMSE, delta1 on dataset test split."""
    _ensure_zoe_path()
    prepare_metric_depth_env()

    from zoedepth.data.data_mono import DepthDataLoader
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    from zoedepth.utils.misc import RunningAverageDict, compute_errors
    from tqdm import tqdm

    resource = pretrained_resource
    if resource and not resource.startswith(("local::", "url::")):
        resource = f"local::{resource}"

    config = get_config("zoedepth", "infer", dataset=dataset, pretrained_resource=resource)
    config.batch_size = batch_size

    from ceti.utils.device import get_device

    device = get_device()
    model = build_model(config).to(device).eval()

    loader = DepthDataLoader(config, "online_eval").data
    metrics = RunningAverageDict()

    for sample in tqdm(loader, desc=f"eval:{dataset}", leave=False):
        if "has_valid_depth" in sample and not sample["has_valid_depth"]:
            continue
        image = sample["image"].to(device)
        depth = sample["depth"].to(device).squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get("focal", torch.tensor([500.0], device=device))
        pred = _infer_metric(model, image)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = torch.nn.functional.interpolate(
                pred, depth.shape[-2:], mode="bilinear", align_corners=True
            )
        gt = depth.squeeze().cpu().numpy()
        pr = pred.squeeze().cpu().numpy()
        if pr.shape != gt.shape:
            pr = torch.nn.functional.interpolate(
                torch.from_numpy(pr).unsqueeze(0).unsqueeze(0),
                gt.shape,
                mode="bilinear",
                align_corners=True,
            ).squeeze().numpy()
        valid = (gt > config.min_depth_eval) & (gt < config.max_depth_eval)
        metrics.update(compute_errors(gt[valid], pr[valid]))

    return {k: float(v) for k, v in metrics.get_value().items()}


def load_metric_model(pretrained_resource: str, dataset: str = "ceti_lab"):
    _ensure_zoe_path()
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    resource = pretrained_resource
    if resource and not resource.startswith(("local::", "url::")):
        resource = f"local::{resource}"

    config = get_config("zoedepth", "infer", dataset=dataset, pretrained_resource=resource)
    from ceti.utils.device import get_device

    device = get_device()
    model = build_model(config).to(device).eval()
    return model, device, config


def find_latest_checkpoint(save_dir: Path, pattern: str = "best") -> Path | None:
    if not save_dir.exists():
        return None
    for glob_pat in (f"*{pattern}*.pt", "*latest*.pt", "*.pt"):
        candidates = sorted(save_dir.rglob(glob_pat))
        if candidates:
            return candidates[-1]
    return None
