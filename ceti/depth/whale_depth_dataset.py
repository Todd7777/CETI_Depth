"""
Whale / marine imagery dataset for Depth Anything domain adaptation.

Whale scenes are absent from Depth Anything pretraining. We fine-tune on
unlabeled (or weakly labeled) RGB images using frozen teacher pseudo-depth,
matching the relative-depth objective used at pretrain time.

Image list format (one path per line, # comments allowed):
    data/whale/yolo/train/images/demo1.jpg
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def collect_images_from_dirs(dirs: list[Path], extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> list[Path]:
    images: list[Path] = []
    for root in dirs:
        if not root.exists():
            continue
        for ext in extensions:
            images.extend(root.rglob(f"*{ext}"))
            images.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def build_whale_depth_file_list(
    sources: list[Path],
    output_path: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Scan source directories and write train/val image path lists."""
    images = collect_images_from_dirs(sources)
    val_path = output_path.parent / "whale_depth_val.txt"

    if not images:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# No images found — add whale imagery under data/whale/\n")
        val_path.write_text("# No validation images\n")
        return [], []

    rng = random.Random(seed)
    rng.shuffle(images)
    split = max(1, int(len(images) * (1 - val_ratio)))
    train_imgs = images[:split]
    val_imgs = images[split:] if split < len(images) else train_imgs[-1:]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_lines = [str(p.resolve()) for p in train_imgs]
    output_path.write_text("\n".join(train_lines) + "\n")
    val_lines = [str(p.resolve()) for p in val_imgs]
    val_path.write_text("\n".join(val_lines) + "\n")

    return train_lines, val_lines


def _repo_root_from_list_file(list_path: Path) -> Path:
    for parent in [list_path.parent, *list_path.resolve().parents]:
        if (parent / "ceti").is_dir() and (parent / "depth_anything").is_dir():
            return parent
    return list_path.resolve().parents[2]


def load_image_paths(
    filenames_file: str | Path,
    repo_root: Path | None = None,
    *,
    exist_only: bool = False,
) -> list[Path]:
    """Load paths from a list file; resolve relative paths against repo root."""
    list_path = Path(filenames_file)
    root = repo_root or _repo_root_from_list_file(list_path)
    paths: list[Path] = []
    with open(filenames_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = root / p
            p = p.resolve()
            if exist_only and not p.is_file():
                continue
            paths.append(p)
    return paths


class WhaleDepthDataset(Dataset):
    """RGB images for teacher-student Depth Anything fine-tuning on whale/marine scenes."""

    def __init__(
        self,
        filenames_file: str | Path,
        image_size: int = 518,
        preprocess_method: str = "none",
        augment: bool = True,
    ):
        all_paths = load_image_paths(filenames_file)
        self.paths = [p for p in all_paths if p.is_file()]
        missing = len(all_paths) - len(self.paths)
        if missing:
            print(
                f"WARNING: {missing} images in list are missing on disk "
                f"(using {len(self.paths)} found). "
                f"Run: bash ceti/scripts/ensure_training_data.sh"
            )
        if not self.paths:
            raise FileNotFoundError(
                f"No image files found for {filenames_file}. "
                "Run: bash ceti/scripts/ensure_training_data.sh"
            )
        self.image_size = image_size
        self.preprocess_method = preprocess_method
        self.augment = augment

        self.transform = Compose([
            Resize(
                width=image_size,
                height=image_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        if augment:
            from ceti.preprocessing.underwater import UnderwaterAugmentation
            self._aug = UnderwaterAugmentation(p=0.35)
        else:
            self._aug = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        from ceti.preprocessing.underwater import preprocess_underwater

        path = self.paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)

        if self.preprocess_method != "none":
            image = preprocess_underwater(image, method=self.preprocess_method)

        if self._aug is not None:
            image = self._aug(image)

        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sample = self.transform({"image": rgb})["image"]
        tensor = torch.from_numpy(sample).float()

        # Fixed spatial size for batched training (aspect-ratio resize varies width)
        h, w = tensor.shape[1], tensor.shape[2]
        if h != self.image_size or w != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return {"image": tensor, "path": str(path)}
