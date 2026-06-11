# Third-party code

Vendored upstream dependencies and legacy depth-map tooling. **Not CETI-authored** — kept for reference and for Depth Anything backbone imports.

| Path | Source |
|------|--------|
| `depth_anything/` | [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything) — DPT relative depth |
| `metric_depth/` | ZoeDepth metric depth stack (legacy depth-map generation) |
| `torchhub/` | Vendored DINOv2 for ZoeDepth |
| `semseg/`, `controlnet/` | Upstream DA extensions |
| `assets/` | Upstream demo images and videos |
| `app.py`, `run.py`, `run_video.py` | Upstream Gradio/CLI demos |

## CETI point cloud pipeline

The active pipeline lives in `ceti/` and imports only:

```python
from depth_anything.dpt import DepthAnything  # via ceti/bootstrap.py → 3rd_party on sys.path
```

Fine-tuned weights: `checkpoints/ceti_whale_depth/best.pt` (gitignored, download via `bash ceti/scripts/download_checkpoint.sh`).
