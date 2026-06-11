#!/usr/bin/env python3
"""
Gradio portal: upload JPEGs / MP4s → CETI fine-tuned RGB|depth outputs.

Note: launch_portal.sh uses ceti_depth_portal_web.py (Flask) by default.
This Gradio variant is optional: python ceti/scripts/ceti_depth_portal.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT = REPO_ROOT / "checkpoints/ceti_whale_depth/best.pt"

try:
    import gradio as gr
except ImportError:
    print("Install Gradio: pip install gradio")
    sys.exit(1)


def _zip_outputs(run_dir: Path) -> str | None:
    outputs = run_dir / "outputs"
    if not outputs.is_dir():
        return None
    zpath = run_dir / "ceti_depth_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(outputs.iterdir()):
            if f.is_file():
                zf.write(f, arcname=f.name)
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            zf.write(manifest, arcname="manifest.json")
    return str(zpath)


def run_uploads(files, underwater_preprocess: bool) -> tuple[str, str | None, list]:
    from ceti.depth.upload_pipeline import IMAGE_EXTS, VIDEO_EXTS, run_pipeline

    if not files:
        return "Upload at least one image or video.", None, []

    if not DEFAULT_CKPT.is_file():
        return f"Missing checkpoint: {DEFAULT_CKPT}", None, []

    staging = Path(tempfile.mkdtemp(prefix="ceti_upload_"))
    paths: list[Path] = []
    try:
        for f in files:
            if isinstance(f, str):
                src = Path(f)
            elif isinstance(f, dict):
                src = Path(f.get("path") or f.get("name", ""))
            else:
                src = Path(getattr(f, "name", str(f)))
            if not src.is_file():
                continue
            if src.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
                continue
            dest = staging / src.name
            shutil.copy2(src, dest)
            paths.append(dest)

        if not paths:
            return "No supported files (.jpg, .png, .mp4, …).", None, []

        manifest = run_pipeline(
            paths,
            underwater_preprocess=underwater_preprocess,
            copy_inputs=False,
        )
        run_dir = REPO_ROOT / manifest["run_dir"]
        zpath = _zip_outputs(run_dir)
        index = run_dir / "index.html"
        gallery = []
        for item in manifest.get("outputs", []):
            p = run_dir / item["file"]
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                gallery.append(str(p))

        summary = (
            f"Run **{manifest['run_id']}** complete.\n\n"
            f"- Device: {manifest['device']}\n"
            f"- Encoder: {manifest['encoder']}\n"
            f"- Outputs: {len(manifest['outputs'])} file(s)\n"
            f"- Folder: `{manifest['run_dir']}`\n\n"
            "Download the ZIP or open the results folder. Videos are in `outputs/`."
        )
        return summary, zpath, gallery
    except Exception as e:
        return f"Error: {e}", None, []
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_demo() -> gr.Blocks:
    ckpt_line = (
        f"**Checkpoint:** `{DEFAULT_CKPT.relative_to(REPO_ROOT)}`"
        if DEFAULT_CKPT.is_file()
        else "⚠ Checkpoint missing — train or copy `best.pt` first."
    )

    with gr.Blocks(title="CETI Underwater Depth", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# CETI Underwater Depth

Upload **JPEG/PNG images** or **MP4 videos**. The fine-tuned **Depth Anything ViT-L** model
produces a side-by-side panel: **RGB (left) | relative depth (right)**.

Depth is **relative** (not meters). Underwater color correction is applied by default.
            """
        )
        gr.Markdown(ckpt_line)

        with gr.Row():
            uploads = gr.File(
                label="Upload images or videos",
                file_count="multiple",
                file_types=["image", "video"],
            )
        underwater = gr.Checkbox(
            value=True,
            label="Underwater preprocess (recommended for marine footage)",
        )
        run_btn = gr.Button("Generate RGB + depth", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            zip_out = gr.File(label="Download all outputs (ZIP)")
            gallery = gr.Gallery(label="Image previews", columns=2, height="auto")

        run_btn.click(
            fn=run_uploads,
            inputs=[uploads, underwater],
            outputs=[status, zip_out, gallery],
        )

        gr.Markdown(
            """
---
**Drop-folder alternative:** copy files into `ceti/inbox/uploads/` and run  
`bash ceti/scripts/run_upload_pipeline.sh`
            """
        )
    return demo


def main() -> None:
    import os

    demo = build_demo()
    port = int(os.environ.get("CETI_PORTAL_PORT", "7860"))
    share = os.environ.get("CETI_PORTAL_SHARE", "").strip() in ("1", "true", "yes")
    print(f"\nCETI Depth Portal → http://127.0.0.1:{port}\n")
    demo.launch(server_name="127.0.0.1", server_port=port, share=share)


if __name__ == "__main__":
    main()
