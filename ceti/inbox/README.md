# Inbox

Local upload and results directories. Generated content is not committed to git.

## Usage

```bash
cp images/*.png ceti/inbox/uploads/
bash ceti/scripts/run_upload_pipeline.sh
```

Or use the research portal:

```bash
bash ceti/scripts/launch_portal.sh
```

## Layout

| Path | Purpose |
|------|---------|
| `uploads/` | Drop JPEG, PNG, or MP4 files here |
| `results/<timestamp>/` | Pipeline output per run |

## Point cloud outputs

| Artifact | Description |
|----------|-------------|
| `pointclouds/*.ply` | Payload-only and tank-scene point clouds |
| `pointclouds/*_payload_mask.png` | Segmentation mask |
| `pointclouds/*_intrinsics.json` | Camera model used for unprojection |
| `previews/*_segmentation.jpg` | Mask overlay |
| `previews/*_showcase.jpg` | Four-panel summary board |

Camera: DJI Osmo Action 4 (`ceti/configs/dji_action4.yaml`).  
Segmentation preset: `CETI_TANK_PRESET=tank_roi_ceti_full`.

Accuracy limitations: [`ceti/docs/POINTCLOUD_ACCURACY.md`](../docs/POINTCLOUD_ACCURACY.md).

## Noise sensitivity study

```bash
python ceti/scripts/run_depth_noise_study.py
```

Results under `results/noise_study_*/`.
