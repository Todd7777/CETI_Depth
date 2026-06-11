# Research portal and upload pipeline

Upload JPEG or MP4 tank footage and produce CETI depth maps and 3D point clouds.

## Prerequisites

```bash
bash ceti/scripts/setup_mac_mps.sh
ls checkpoints/ceti_whale_depth/best.pt
```

## Web portal

```bash
bash ceti/scripts/launch_portal.sh
```

1. Open http://127.0.0.1:7860  
2. Upload media  
3. Download results or use **Explore payload 3D** / **Explore scene 3D**

## Drop folder

| Step | Action |
|------|--------|
| 1 | Copy `.jpg`, `.png`, or `.mp4` into `ceti/inbox/uploads/` |
| 2 | Run `bash ceti/scripts/run_upload_pipeline.sh` |
| 3 | Open `ceti/inbox/results/<timestamp>/index.html` |

## Outputs

```
results/<timestamp>/
  pointclouds/     *.ply, *_payload_mask.png, *_intrinsics.json
  previews/        showcase boards, segmentation overlays, 3D snapshots
  manifest.json
  web_api.json
  index.html
```

## Modes

| `CETI_DEPTH_MODE` | Output |
|-------------------|--------|
| `pointcloud` (default) | Payload + tank-scene PLY, segmentation, 3D explorer |
| `relative` | RGB \| depth previews only |
| `accurate` | Metric-aligned depth maps and distance JSON |

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| Missing checkpoint | Train or copy `checkpoints/ceti_whale_depth/best.pt` |
| Empty upload folder | Add files to `ceti/inbox/uploads/` |
| Slow video | ViT-L on MPS ~3–4 FPS; use shorter clips for demos |
| Flask missing | `pip install flask` (included in setup script) |

## Environment

```bash
export CETI_TANK_PRESET=tank_roi_ceti_full
export CETI_INBOX=~/path/to/uploads
export CETI_DEPTH_CKPT=/path/to/best.pt
```

## API integration

Structured paths for frontends: see `web_api.json` in each run directory.
