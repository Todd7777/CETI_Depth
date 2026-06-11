# Inbox

Drop tank images here, then run:

```bash
bash ceti/scripts/run_batch.sh
```

Results appear under `results/<timestamp>/`:

| Path | Description |
|------|-------------|
| `index.html` | Browser gallery |
| `pointclouds/*.ply` | Payload + `*_scene.ply` tank view |
| `previews/*_showcase.jpg` | Presentation boards |
| `manifest.json` | Run metadata |

Generated content is gitignored; only `.gitkeep` files are tracked.
