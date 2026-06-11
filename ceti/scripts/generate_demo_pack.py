#!/usr/bin/env python3
"""
Generate a large underwater demo pack: comparison stills + depth videos.

Outputs under ceti/outputs/demo_pack/:
  videos/           — fine-tuned depth on each clip
  videos_compare/   — RGB | baseline depth | fine-tuned depth (same frame)
  stills/           — val/train samples, 3-column compare panels
  stills_grid/      — contact sheets per source video

Usage:
  python ceti/scripts/generate_demo_pack.py
  python ceti/scripts/generate_demo_pack.py --max-stills 40 --frames-per-video 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "ceti/outputs/demo_pack"
DEFAULT_CKPT = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"
VAL_LIST = REPO_ROOT / "ceti/data/whale_depth_val.txt"

UNDERWATER_VIDEOS = [
    ("davis_dolphins", REPO_ROOT / "assets/examples_video/davis_dolphins.mp4"),
    ("davis_seasnake", REPO_ROOT / "assets/examples_video/davis_seasnake.mp4"),
    ("davis_rollercoaster", REPO_ROOT / "assets/examples_video/davis_rollercoaster.mp4"),
]


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def _depth_colormap(bgr: np.ndarray, depth_norm: np.ndarray) -> np.ndarray:
    vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
    if vis.shape[:2] != bgr.shape[:2]:
        vis = cv2.resize(vis, (bgr.shape[1], bgr.shape[0]))
    return vis


def _triple_panel(bgr: np.ndarray, depth_baseline: np.ndarray, depth_finetuned: np.ndarray) -> np.ndarray:
    """RGB | generic vits depth | CETI fine-tuned depth."""
    d0 = _depth_colormap(bgr, depth_baseline)
    d1 = _depth_colormap(bgr, depth_finetuned)
    h = bgr.shape[0]
    label_h = 28
    canvas = np.zeros((h + label_h, bgr.shape[1] * 3, 3), dtype=np.uint8)
    for i, (img, label) in enumerate(
        [(bgr, "RGB"), (d0, "Baseline (vits)"), (d1, "CETI fine-tuned (vitl)")]
    ):
        x0 = i * bgr.shape[1]
        canvas[label_h : label_h + h, x0 : x0 + bgr.shape[1]] = img
        cv2.putText(
            canvas,
            label,
            (x0 + 8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def sample_val_images(val_list: Path, n: int, seed: int = 0) -> list[Path]:
    from ceti.depth.whale_depth_dataset import load_image_paths

    paths = [p for p in load_image_paths(val_list, repo_root=REPO_ROOT) if p.is_file()]
    if not paths:
        return []
    if len(paths) <= n:
        return paths
    rng = np.random.default_rng(seed)
    idx = np.linspace(0, len(paths) - 1, n, dtype=int)
    # Add a few random picks for variety
    extra = list(rng.choice(len(paths), size=min(n // 3, 12), replace=False))
    chosen = sorted(set(idx.tolist() + extra))[:n]
    return [paths[i] for i in chosen]


def extract_frames(video: Path, out_dir: Path, n: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = np.linspace(0, total - 1, n, dtype=int)
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


def write_compare_video(
    video_path: Path,
    out_path: Path,
    model_base,
    model_ft,
    transform,
    device: str,
    *,
    preprocess: bool = True,
    max_frames: int | None = None,
) -> int:
    from ceti.depth.infer_robot import predict_depth
    from ceti.preprocessing.underwater import preprocess_underwater

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = w * 3
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_w, h + 28),
    )
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and n >= max_frames:
            break
        proc = preprocess_underwater(frame, method="combined") if preprocess else frame
        d0 = predict_depth(model_base, transform, proc, device)
        d1 = predict_depth(model_ft, transform, proc, device)
        panel = _triple_panel(proc, d0, d1)
        writer.write(panel)
        n += 1
        if n % 20 == 0:
            print(f"    {out_path.name}: {n} frames…")
    cap.release()
    writer.release()
    return n


def write_finetuned_video(
    video_path: Path,
    out_path: Path,
    model_ft,
    transform,
    device: str,
    *,
    preprocess: bool = True,
) -> int:
    from ceti.depth.infer_robot import process_frame

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, _ = process_frame(
            frame,
            model_ft,
            transform,
            None,
            device,
            preprocess,
            "combined",
            0.5,
        )
        writer.write(vis)
        n += 1
    cap.release()
    writer.release()
    return n


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 4) -> None:
    if not image_paths:
        return
    thumbs = []
    for p in image_paths[: cols * cols]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        thumbs.append(cv2.resize(im, (480, 270)))
    if not thumbs:
        return
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CETI underwater demo pack")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--max-stills", type=int, default=36)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument(
        "--compare-videos",
        action="store_true",
        default=True,
        help="Write 3-panel baseline vs fine-tuned videos (slower)",
    )
    parser.add_argument(
        "--no-compare-videos",
        action="store_true",
        help="Skip compare videos (faster; stills + finetuned-only videos)",
    )
    args = parser.parse_args()

    if args.no_compare_videos:
        args.compare_videos = False

    from ceti.depth.infer_robot import build_depth_model, predict_depth, resolve_encoder
    from ceti.preprocessing.underwater import preprocess_underwater
    from ceti.utils.device import configure_compute, device_name, get_device

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    configure_compute()
    device = str(get_device())
    _banner("CETI Demo Pack Generator")
    print(f"  Device: {device_name(get_device())}")
    print(f"  Checkpoint: {ckpt}")

    enc_ft = resolve_encoder(ckpt)
    model_ft, transform = build_depth_model(enc_ft, device, str(ckpt))
    model_base, _ = build_depth_model("vits", device, None)

    manifest: dict = {"checkpoint": str(ckpt.relative_to(REPO_ROOT)), "encoder": enc_ft, "videos": [], "stills": []}

    # --- Stills from validation set ---
    _banner(f"Stills: up to {args.max_stills} validation images (3-panel compare)")
    still_dir = OUT_ROOT / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    val_paths = sample_val_images(VAL_LIST, args.max_stills)
    for i, img_path in enumerate(val_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        proc = preprocess_underwater(bgr, method="combined")
        d0 = predict_depth(model_base, transform, proc, device)
        d1 = predict_depth(model_ft, transform, proc, device)
        panel = _triple_panel(proc, d0, d1)
        name = f"{i:03d}_{img_path.stem}_compare.jpg"
        out = still_dir / name
        cv2.imwrite(str(out), panel)
        manifest["stills"].append(str(out.relative_to(REPO_ROOT)))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(val_paths)} stills")
    print(f"  Saved {len(manifest['stills'])} stills → {still_dir}")

    # --- Video frames as stills + grids ---
    _banner("Video frame stills + contact sheets")
    frames_root = OUT_ROOT / "frames"
    grid_dir = OUT_ROOT / "stills_grid"
    for tag, video in UNDERWATER_VIDEOS:
        if not video.exists():
            print(f"  Skip missing: {video}")
            continue
        frames = extract_frames(video, frames_root / tag, args.frames_per_video)
        compare_paths = []
        for frame_path in frames:
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                continue
            proc = preprocess_underwater(bgr, method="combined")
            d0 = predict_depth(model_base, transform, proc, device)
            d1 = predict_depth(model_ft, transform, proc, device)
            panel = _triple_panel(proc, d0, d1)
            out = still_dir / f"{tag}_{frame_path.stem}_compare.jpg"
            cv2.imwrite(str(out), panel)
            compare_paths.append(out)
            manifest["stills"].append(str(out.relative_to(REPO_ROOT)))
        make_contact_sheet(compare_paths, grid_dir / f"{tag}_grid.jpg", cols=4)
        print(f"  {tag}: {len(compare_paths)} panels + grid")

    # --- Full videos (fine-tuned only, fast) ---
    _banner("Videos: fine-tuned RGB|depth (full clip)")
    vid_dir = OUT_ROOT / "videos"
    for tag, video in UNDERWATER_VIDEOS:
        if not video.exists():
            continue
        out = vid_dir / f"{tag}_finetuned.mp4"
        print(f"  Encoding {out.name}…")
        n = write_finetuned_video(video, out, model_ft, transform, device, preprocess=True)
        manifest["videos"].append(
            {"name": out.name, "frames": n, "type": "finetuned_rgb_depth", "source": str(video.relative_to(REPO_ROOT))}
        )
        print(f"    {n} frames → {out}")

    # --- Compare videos (3-panel, slower) ---
    if args.compare_videos:
        _banner("Videos: 3-panel baseline vs fine-tuned (underwater clips)")
        cmp_dir = OUT_ROOT / "videos_compare"
        for tag, video in UNDERWATER_VIDEOS:
            if not video.exists():
                continue
            if tag == "davis_rollercoaster":
                print(f"  Skip compare for {tag} (not underwater — stills only)")
                continue
            out = cmp_dir / f"{tag}_baseline_vs_finetuned.mp4"
            print(f"  Encoding {out.name}…")
            n = write_compare_video(
                video,
                out,
                model_base,
                model_ft,
                transform,
                device,
                preprocess=True,
            )
            manifest["videos"].append(
                {
                    "name": out.name,
                    "frames": n,
                    "type": "compare_3panel",
                    "source": str(video.relative_to(REPO_ROOT)),
                }
            )
            print(f"    {n} frames → {out}")

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _banner("Done")
    print(f"  Output root: {OUT_ROOT}")
    print(f"  Stills:       {still_dir}  ({len(manifest['stills'])} images)")
    print(f"  Videos:       {vid_dir}")
    if args.compare_videos:
        print(f"  Compare vids: {OUT_ROOT / 'videos_compare'}")
    print(f"  Manifest:     {manifest_path}")
    print("\n  open ceti/outputs/demo_pack/stills")
    print("  open ceti/outputs/demo_pack/videos")


if __name__ == "__main__":
    main()
