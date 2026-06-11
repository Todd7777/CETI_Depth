# CETI Point Cloud Portal

Web UI and batch CLI for tank image → 3D point cloud generation.

## Setup

```bash
bash ceti/scripts/setup.sh
bash ceti/scripts/download_checkpoint.sh
```

## Portal

```bash
bash ceti/scripts/launch_portal.sh
```

Open http://127.0.0.1:7860 — upload images, browse runs, 3D PLY explorer.

## Batch (CLI)

```bash
cp /path/to/tank/*.png ceti/inbox/uploads/
bash ceti/scripts/run_batch.sh
```

Output: `ceti/inbox/results/<timestamp>/` with `index.html`, `pointclouds/*.ply`, previews.

## Linux server

Same commands. `setup.sh` installs CUDA PyTorch when a GPU is detected; otherwise CPU.

Bind to all interfaces (optional):

```bash
# Edit ceti/scripts/ceti_depth_portal_web.py host=0.0.0.0 for remote access
CETI_PORTAL_PORT=7860 bash ceti/scripts/launch_portal.sh
```

## Environment

| Variable | Default |
|----------|---------|
| `CETI_DEVICE` | `cuda` (Linux) / `mps` (macOS) |
| `CETI_TANK_PRESET` | `tank_roi_ceti_full` |
| `CETI_DEPTH_CKPT` | `checkpoints/ceti_whale_depth/best.pt` |
| `CETI_INBOX` | `ceti/inbox/uploads` |
