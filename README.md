# CETI Depth

Monocular **3D point cloud** generation for CETI tank experiments (DJI Osmo Action 4). Fine-tuned Depth Anything → payload segmentation → colored `.ply` exports.

Harvard CETI lab.

## Quick start (Linux or macOS)

```bash
git clone https://github.com/Todd7777/CETI_Depth.git
cd CETI_Depth
bash ceti/scripts/setup.sh
bash ceti/scripts/download_checkpoint.sh   # ~1.2 GB from Hugging Face
```

### Web portal

```bash
bash ceti/scripts/launch_portal.sh
# → http://127.0.0.1:7860
```

Upload tank images, browse runs, open the interactive 3D PLY explorer.

### Batch CLI (all files in inbox)

```bash
cp /path/to/*.png ceti/inbox/uploads/
bash ceti/scripts/run_batch.sh
# Results: ceti/inbox/results/<timestamp>/
```

## Repository layout

```
ceti/                 CETI point cloud pipeline (our code)
├── depth/            Inference + upload pipeline
├── geometry/         Intrinsics, PLY export, 3D previews
├── preprocessing/    Tank ROI, payload segmentation
├── web/              Portal UI
├── scripts/          setup, portal, batch runner
├── configs/          DJI Action 4 intrinsics, tank ROI preset
└── inbox/            uploads/ + results/ (generated, gitignored)

3rd_party/            Vendored Depth Anything + legacy depth tools (see 3rd_party/README.md)
checkpoints/          Fine-tuned weights (gitignored — download separately)
archive/              Local legacy code (gitignored)
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `CETI_TANK_PRESET` | `tank_roi_ceti_full` | Tank ROI + segmentation preset |
| `CETI_DEVICE` | `cuda` (Linux) / `mps` (macOS) | PyTorch device |
| `CETI_UNPROJECT_MODEL` | `pinhole` | Camera model for unprojection |

Configs: [`ceti/configs/tank_roi_ceti_full.json`](ceti/configs/tank_roi_ceti_full.json), [`ceti/configs/dji_action4.yaml`](ceti/configs/dji_action4.yaml)

## Linux (CUDA server)

```bash
bash ceti/scripts/setup.sh          # detects CUDA, installs cu124 PyTorch if available
bash ceti/scripts/download_checkpoint.sh
bash ceti/scripts/launch_portal.sh
```

Full portal guide: [`ceti/docs/PORTAL.md`](ceti/docs/PORTAL.md)

## Accuracy note

Point clouds use **relative** monocular depth + approximate intrinsics — qualitative geometry, not survey-grade metrology. See [`ceti/docs/POINTCLOUD_ACCURACY.md`](ceti/docs/POINTCLOUD_ACCURACY.md).

## License

See [LICENSE](LICENSE). Third-party code in `3rd_party/` retains upstream licenses.

## References

- Yang et al., Depth Anything, CVPR 2024  
- [Project CETI](https://www.projectceti.org/)
