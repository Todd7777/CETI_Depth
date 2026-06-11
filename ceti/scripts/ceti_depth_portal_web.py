#!/usr/bin/env python3
"""
CETI Research Portal — point clouds + depth noise study (Flask).

  bash ceti/scripts/launch_portal.sh
  → http://127.0.0.1:7860
"""

from __future__ import annotations

import html
import json
import sys
import traceback
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
RESULTS_ROOT = REPO_ROOT / "ceti/inbox/results"

try:
    from flask import (
        Flask,
        abort,
        redirect,
        request,
        send_file,
        send_from_directory,
        url_for,
    )
except ImportError:
    print("Install Flask: pip install flask")
    sys.exit(1)

from ceti.depth.upload_pipeline import DEFAULT_CKPT, IMAGE_EXTS, VIDEO_EXTS, run_pipeline
from ceti.web.researcher_ui import (
    page_shell,
    portal_nav,
    render_browse_page,
    render_noise_study_detail,
    render_noise_study_index,
    render_pointcloud_explorer,
    render_pointcloud_results,
)

ALLOWED = IMAGE_EXTS | VIDEO_EXTS


def _esc(text: object) -> str:
    return html.escape(str(text))


def _safe_child(base: Path, *parts: str) -> Path:
    """Resolve path and ensure it stays inside base (no traversal)."""
    target = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    if not str(target).startswith(str(base_resolved)):
        abort(403)
    return target


def _list_runs(*, noise: bool = False) -> list[dict]:
    runs = []
    if not RESULTS_ROOT.is_dir():
        return runs
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        is_noise = d.name.startswith("noise_study")
        if noise != is_noise:
            continue
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if noise:
            runs.append({"run_id": d.name, "n_images": len(m.get("images", []))})
        else:
            runs.append(
                {
                    "run_id": d.name,
                    "n_outputs": len(m.get("outputs", [])),
                    "mode": m.get("depth_mode", ""),
                }
            )
    return runs


def _zip_run(run_dir: Path, zpath: Path) -> None:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in ("previews", "pointclouds", "distances", "outputs"):
            folder = run_dir / sub
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=str(Path(sub) / f.relative_to(folder)))
        for name in ("manifest.json", "web_api.json", "index.html"):
            mf = run_dir / name
            if mf.is_file():
                zf.write(mf, arcname=name)


def _zip_noise_study(study_dir: Path, zpath: Path) -> None:
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in study_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(study_dir)))


def _error_page(title: str, message: str, *, hint: str = "") -> str:
    body = f"""
<section class="hero">
  <h1>{_esc(title)}</h1>
  <p class="lead">{_esc(message)}</p>
  {f'<p class="meta">{_esc(hint)}</p>' if hint else ''}
  <p class="links"><a class="btn" href="/">← Back to home</a></p>
</section>
"""
    return page_shell(title=title, body=body, nav=portal_nav("home"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

    @app.route("/health")
    def health():
        return {"status": "ok", "checkpoint": DEFAULT_CKPT.is_file()}

    @app.errorhandler(404)
    def not_found(_e):
        return _error_page("Page not found", "That URL does not exist.", hint=request.path), 404

    @app.errorhandler(500)
    def server_error(_e):
        return _error_page("Server error", "Something went wrong. Check the terminal for details."), 500

    def _home_body(error: str = "") -> str:
        ckpt = DEFAULT_CKPT.relative_to(REPO_ROOT) if DEFAULT_CKPT.is_file() else "MISSING"
        err = f'<div class="notice">{_esc(error)}</div>' if error else ""
        return f"""
<section class="hero">
  <h1>CETI Underwater 3D Portal</h1>
  <p class="lead">Upload tank images from the DJI Osmo Action 4. Generate payload-only point clouds.</p>
  <p class="meta">Checkpoint: {_esc(str(ckpt))}</p>
</section>
{err}
<div class="grid-2">
  <article class="card upload-box">
    <h2>① Point cloud (main)</h2>
    <p class="meta">Payload segmentation → CETI depth → colored 3D PLY + showcase.</p>
    <form method="post" action="{url_for('process_pointcloud')}" enctype="multipart/form-data">
      <p><input type="file" name="files" multiple accept="image/*,video/*" required></p>
      <label class="chk"><input type="checkbox" name="underwater" value="1" checked> Underwater preprocess</label>
      <label class="chk"><input type="checkbox" name="tank_roi" value="1" checked> Payload masking</label>
      <p style="margin-top:1rem"><button class="btn" type="submit">Generate point clouds</button></p>
    </form>
  </article>
  <article class="card upload-box">
    <h2>② Noise study (extension)</h2>
    <p class="meta">Compare 0–40% depth noise with segmentation panels.</p>
    <p class="links"><a class="btn secondary" href="{url_for('noise_page')}">Open noise study →</a></p>
    <p class="links"><a href="{url_for('browse')}">Browse all past runs →</a></p>
  </article>
</div>
"""

    @app.route("/")
    def index():
        error = request.args.get("error", "")
        return page_shell(
            title="CETI Research Portal",
            body=_home_body(error),
            nav=portal_nav("home"),
        )

    @app.route("/noise")
    def noise_page():
        error = request.args.get("error", "")
        err = f'<div class="notice">{_esc(error)}</div>' if error else ""
        body = f"""
<section class="hero">
  <h1>Depth Noise Sensitivity Study</h1>
  <p class="lead">Upload tank images. Generates comparison boards: segmentation, clean vs noised depth, delta, 3D.</p>
</section>
{err}
<article class="card upload-box">
  <form method="post" action="{url_for('process_noise')}" enctype="multipart/form-data">
    <p><input type="file" name="files" multiple accept="image/*" required></p>
    <p style="margin-top:1rem"><button class="btn" type="submit">Run noise study</button></p>
  </form>
</article>
<p class="links"><a href="{url_for('browse')}">Browse past noise studies →</a></p>
"""
        return page_shell(title="Noise Study", body=body, nav=portal_nav("noise"))

    @app.route("/browse")
    def browse():
        return render_browse_page(_list_runs(noise=False), _list_runs(noise=True))

    @app.route("/process", methods=["POST"])
    @app.route("/process/pointcloud", methods=["POST"])
    def process_pointcloud():
        try:
            if not DEFAULT_CKPT.is_file():
                return _error_page("Missing checkpoint", str(DEFAULT_CKPT)), 500
            saved = _save_uploads()
            if not saved:
                return redirect(url_for("index", error="No supported image/video files uploaded."))
            manifest = run_pipeline(
                saved,
                underwater_preprocess=request.form.get("underwater") == "1",
                depth_mode="pointcloud",
                tank_roi=request.form.get("tank_roi") == "1",
            )
            return redirect(url_for("pointcloud_results", run_id=manifest["run_id"]))
        except Exception as exc:
            traceback.print_exc()
            return redirect(url_for("index", error=f"Pipeline failed: {exc}"))

    @app.route("/process/noise", methods=["POST"])
    def process_noise():
        try:
            if not DEFAULT_CKPT.is_file():
                return _error_page("Missing checkpoint", str(DEFAULT_CKPT)), 500
            saved = _save_uploads(images_only=True)
            if not saved:
                return redirect(url_for("noise_page", error="Upload at least one PNG/JPEG image."))
            from ceti.scripts.run_depth_noise_study import run_noise_study

            manifest = run_noise_study(saved)
            return redirect(url_for("noise_results", run_id=manifest["run_id"]))
        except Exception as exc:
            traceback.print_exc()
            return redirect(url_for("noise_page", error=f"Noise study failed: {exc}"))

    def _save_uploads(*, images_only: bool = False) -> list[Path]:
        files = request.files.getlist("files")
        staging = REPO_ROOT / "ceti/inbox/_web_staging"
        staging.mkdir(parents=True, exist_ok=True)
        for old in staging.iterdir():
            if old.is_file():
                old.unlink()
        saved: list[Path] = []
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if images_only and ext not in IMAGE_EXTS:
                continue
            if not images_only and ext not in ALLOWED:
                continue
            dest = staging / Path(f.filename).name
            f.save(dest)
            saved.append(dest)
        return saved

    def _pointcloud_asset_url(run_id: str, rel: str) -> str:
        parts = Path(rel).parts
        if len(parts) >= 2 and parts[0] == "previews":
            return url_for("serve_preview", run_id=run_id, filename="/".join(parts[1:]))
        if len(parts) >= 2 and parts[0] == "pointclouds":
            return url_for("serve_pointcloud", run_id=run_id, filename="/".join(parts[1:]))
        if len(parts) >= 2 and parts[0] == "distances":
            return url_for("serve_distances", run_id=run_id, filename="/".join(parts[1:]))
        return url_for("serve_run_file", run_id=run_id, filename=rel)

    @app.route("/results/<run_id>")
    def pointcloud_results(run_id: str):
        run_dir = RESULTS_ROOT / run_id
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            abort(404)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        html_out = render_pointcloud_results(
            manifest,
            run_id=run_id,
            zip_url=url_for("download_zip", run_id=run_id),
            api_url=url_for("web_api", run_id=run_id) if (run_dir / "web_api.json").is_file() else "",
            gallery_url=url_for("serve_gallery", run_id=run_id) if (run_dir / "index.html").is_file() else "",
            asset_url=_pointcloud_asset_url,
        )
        return html_out

    @app.route("/results/<run_id>/explorer")
    def pointcloud_explorer(run_id: str):
        ply_rel = request.args.get("ply", "").strip()
        if not ply_rel or not ply_rel.startswith("pointclouds/"):
            abort(400)
        ply_path = _safe_child(RESULTS_ROOT / run_id, ply_rel)
        if not ply_path.is_file():
            abort(404)
        mode = request.args.get("mode", "payload").strip().lower()
        if mode not in ("payload", "scene"):
            mode = "payload"

        manifest_path = RESULTS_ROOT / run_id / "manifest.json"
        point_count = None
        mask_url = ""
        contrast_url = ""
        source_name = Path(ply_rel).stem.replace("_scene", "")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest.get("outputs", []):
                match = (
                    item.get("pointcloud_ply") == ply_rel
                    if mode == "payload"
                    else item.get("pointcloud_ply_scene") == ply_rel
                )
                if not match:
                    continue
                point_count = (
                    item.get("point_count")
                    if mode == "payload"
                    else item.get("point_count_scene")
                )
                source_name = item.get("source", source_name)
                mask_rel = item.get("payload_mask")
                if mask_rel and mode == "payload":
                    mask_url = _pointcloud_asset_url(run_id, mask_rel)
                contrast_rel = item.get("pointcloud_3d_contrast")
                if contrast_rel:
                    contrast_url = _pointcloud_asset_url(run_id, contrast_rel)
                break
        ply_url = _pointcloud_asset_url(run_id, ply_rel)
        return render_pointcloud_explorer(
            title=source_name,
            ply_url=ply_url,
            back_url=url_for("pointcloud_results", run_id=run_id),
            point_count=point_count,
            mask_url=mask_url,
            source_name=source_name,
            mode=mode,
            contrast_url=contrast_url,
        )

    @app.route("/results/<run_id>/gallery")
    def serve_gallery(run_id: str):
        path = _safe_child(RESULTS_ROOT / run_id, "index.html")
        if not path.is_file():
            abort(404)
        return send_file(path)

    @app.route("/noise/results/<run_id>")
    def noise_results(run_id: str):
        study_dir = RESULTS_ROOT / run_id
        mf = study_dir / "manifest.json"
        if not mf.is_file():
            abort(404)
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        return render_noise_study_index(
            manifest,
            study_url=lambda rid, stem="": (
                url_for("noise_detail", run_id=rid, stem=stem) if stem else url_for("noise_results", run_id=rid)
            ),
            asset_url=lambda rid, rel: url_for("serve_noise_file", run_id=rid, filename=rel),
            zip_url=url_for("download_noise_zip", run_id=run_id),
        )

    @app.route("/noise/results/<run_id>/view/<stem>")
    def noise_detail(run_id: str, stem: str):
        study_dir = RESULTS_ROOT / run_id
        mf = study_dir / "manifest.json"
        if not mf.is_file():
            abort(404)
        manifest = json.loads(mf.read_text(encoding="utf-8"))
        item = next((i for i in manifest.get("images", []) if Path(i["source"]).stem == stem), None)
        if item is None:
            abort(404)
        return render_noise_study_detail(
            item,
            run_id=run_id,
            asset_url=lambda rid, rel: url_for("serve_noise_file", run_id=rid, filename=rel),
            index_url=url_for("noise_results", run_id=run_id),
        )

    @app.route("/results/<run_id>/previews/<path:filename>")
    def serve_preview(run_id: str, filename: str):
        directory = RESULTS_ROOT / run_id / "previews"
        path = _safe_child(directory, filename)
        if not path.is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/results/<run_id>/pointclouds/<path:filename>")
    def serve_pointcloud(run_id: str, filename: str):
        directory = RESULTS_ROOT / run_id / "pointclouds"
        path = _safe_child(directory, filename)
        if not path.is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/results/<run_id>/distances/<path:filename>")
    def serve_distances(run_id: str, filename: str):
        directory = RESULTS_ROOT / run_id / "distances"
        path = _safe_child(directory, filename)
        if not path.is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/results/<run_id>/file/<path:filename>")
    def serve_run_file(run_id: str, filename: str):
        directory = RESULTS_ROOT / run_id
        path = _safe_child(directory, filename)
        if not path.is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/noise/results/<run_id>/files/<path:filename>")
    def serve_noise_file(run_id: str, filename: str):
        directory = RESULTS_ROOT / run_id
        path = _safe_child(directory, filename)
        if not path.is_file():
            abort(404)
        return send_from_directory(directory, filename)

    @app.route("/results/<run_id>/web_api.json")
    def web_api(run_id: str):
        path = _safe_child(RESULTS_ROOT / run_id, "web_api.json")
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="application/json")

    @app.route("/results/<run_id>/download.zip")
    def download_zip(run_id: str):
        run_dir = RESULTS_ROOT / run_id
        if not run_dir.is_dir():
            abort(404)
        zpath = run_dir / "ceti_depth_results.zip"
        _zip_run(run_dir, zpath)
        return send_file(zpath, as_attachment=True, download_name=f"ceti_pointcloud_{run_id}.zip")

    @app.route("/noise/results/<run_id>/download.zip")
    def download_noise_zip(run_id: str):
        study_dir = RESULTS_ROOT / run_id
        if not study_dir.is_dir():
            abort(404)
        zpath = study_dir / "ceti_noise_study.zip"
        _zip_noise_study(study_dir, zpath)
        return send_file(zpath, as_attachment=True, download_name=f"ceti_noise_{run_id}.zip")

    return app


def main() -> None:
    import os

    port = int(os.environ.get("CETI_PORTAL_PORT", "7860"))
    app = create_app()
    print(f"\nCETI Research Portal → http://127.0.0.1:{port}")
    print(f"  Health: http://127.0.0.1:{port}/health\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
