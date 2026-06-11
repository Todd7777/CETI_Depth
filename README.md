# CETI Depth

Monocular depth estimation and 3D point-cloud reconstruction for CETI tank experiments. Built on [Depth Anything](https://github.com/LiheYoung/Depth-Anything) with a fine-tuned underwater model, DJI Osmo Action 4 intrinsics, and payload segmentation for suspended-rig capture.

Harvard CETI / AVATARS lab stack.

## Capabilities

| Workflow | Description |
|----------|-------------|
| **Point clouds** | Payload-only and tank-scene colored `.ply` from stills or video |
| **Segmentation** | Image-fitted payload box (frame + motor); tank window and clutter excluded |
| **Research portal** | Upload UI, run browser, 3D PLY explorer, noise-sensitivity study |
| **Depth inference** | CETI fine-tuned relative depth (`checkpoints/ceti_whale_depth/best.pt`) |
| **Training** | Whale/marine depth fine-tune, metric depth (FLSea/SQUID), robot ROS2 node |

## Requirements

- macOS Apple Silicon (MPS) or Linux with CUDA  
- Python 3.11+  
- Fine-tuned checkpoint: `checkpoints/ceti_whale_depth/best.pt`  
- ~8 GB disk for venv; checkpoint supplied separately or produced via training  

## Setup

```bash
git clone https://github.com/Todd7777/CETI_Depth.git
cd CETI_Depth
bash ceti/scripts/setup_mac_mps.sh
```

Place `best.pt` under `checkpoints/ceti_whale_depth/` if not produced locally.

## Professor demo (recommended)

### Web portal

```bash
bash ceti/scripts/launch_professor_portal.sh
```

Open http://127.0.0.1:7860 — upload DJI Action 4 stills or clips, generate point clouds, browse runs, open the 3D explorer.

### Drop folder

```bash
cp /path/to/images/*.png ceti/inbox/uploads/
bash ceti/scripts/run_professor_pipeline.sh
```

Results: `ceti/inbox/results/<timestamp>/` (`index.html`, `pointclouds/*.ply`, previews, `manifest.json`).

Full instructions: [`ceti/docs/PROFESSOR_DEMO.md`](ceti/docs/PROFESSOR_DEMO.md)

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `CETI_TANK_PRESET` | `tank_roi_ceti_full` | ROI + payload segmentation preset |
| `CETI_DEPTH_MODE` | `pointcloud` | `pointcloud`, `relative`, or `accurate` |
| `CETI_UNPROJECT_MODEL` | `pinhole` | `pinhole` or `fisheye_equidistant` |
| `CETI_DEPTH_SEMANTICS` | `z_depth` | Depth unprojection semantics |

Tank ROI and segmentation: [`ceti/configs/tank_roi_ceti_full.json`](ceti/configs/tank_roi_ceti_full.json)  
Camera intrinsics: [`ceti/configs/dji_action4.yaml`](ceti/configs/dji_action4.yaml)

## Repository layout

```
ceti/
├── configs/           ROI, camera, training configs
├── depth/             Inference, training, upload pipeline
├── docs/              Professor demo, accuracy notes, setup
├── geometry/          Intrinsics, point-cloud export, 3D previews
├── inbox/             Upload drop zone and generated runs (gitignored)
├── preprocessing/     Tank ROI, payload segmentation, underwater prep
├── scripts/           Portal, pipeline, training, data scripts
└── web/               Research portal UI
checkpoints/           Model weights (not in git)
depth_anything/        Upstream Depth Anything encoder
metric_depth/          ZoeDepth metric head
```

## Point cloud accuracy

Relative monocular depth + approximate intrinsics → **qualitative 3D**, not survey-grade metrology. See [`ceti/docs/POINTCLOUD_ACCURACY.md`](ceti/docs/POINTCLOUD_ACCURACY.md).

## Training and research stack

Extended documentation: [`ceti/README.md`](ceti/README.md)  
Upstream Depth Anything paper/code: [`docs/DEPTH_ANYTHING_UPSTREAM.md`](docs/DEPTH_ANYTHING_UPSTREAM.md)

## License

Depth Anything components: see [`LICENSE`](LICENSE). CETI extensions follow the same license unless noted otherwise.

## References

- Yang et al., Depth Anything, CVPR 2024  
- Gil et al., AVATARS, Harvard SEAS  
- [Project CETI](https://www.projectceti.org/)
