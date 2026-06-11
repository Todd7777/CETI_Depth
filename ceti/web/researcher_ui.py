"""
Shared HTML/CSS for CETI researcher portal and static result galleries.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote


def _esc(text: str | int | float | None) -> str:
    if text is None:
        return ""
    return html.escape(str(text))


RESEARCHER_CSS = """
:root {
  --bg: #080d12;
  --panel: #121a24;
  --panel2: #182230;
  --text: #edf2f7;
  --muted: #93a4b8;
  --accent: #5fd38d;
  --accent2: #7eb8ff;
  --warn: #f0b429;
  --border: rgba(255,255,255,0.08);
  --shadow: 0 12px 40px rgba(0,0,0,0.35);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif;
  background:
    radial-gradient(900px 500px at 0% -5%, rgba(95,211,141,0.10), transparent 55%),
    radial-gradient(700px 400px at 100% 0%, rgba(126,184,255,0.08), transparent 50%),
    var(--bg);
  color: var(--text);
  line-height: 1.5;
}
a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 1.5rem 1.25rem 3rem; }
.topnav {
  display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
  margin-bottom: 1.25rem; padding: 0.65rem 0.85rem;
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
}
.topnav a, .topnav span.brand {
  padding: 0.45rem 0.8rem; border-radius: 8px; font-size: 0.92rem;
}
.topnav span.brand { color: var(--accent); font-weight: 700; margin-right: 0.25rem; }
.topnav a.active { background: rgba(95,211,141,0.14); color: #d8ffe7; }
.hero {
  background: linear-gradient(135deg, rgba(95,211,141,0.10), rgba(126,184,255,0.07));
  border: 1px solid var(--border); border-radius: 16px; padding: 1.35rem 1.5rem;
  margin-bottom: 1.25rem; box-shadow: var(--shadow);
}
.hero h1 { margin: 0 0 0.35rem; font-size: 1.65rem; letter-spacing: -0.02em; }
.lead { margin: 0; color: var(--muted); max-width: 72ch; }
.meta { color: var(--muted); font-size: 0.9rem; margin-top: 0.65rem; }
.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 1rem 1.1rem 1.15rem; margin: 1rem 0; box-shadow: var(--shadow);
}
.card h2, .card h3 { margin: 0 0 0.5rem; font-size: 1.05rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0; }
.chip {
  font-size: 0.76rem; color: #d8ffe7; background: rgba(95,211,141,0.12);
  border: 1px solid rgba(95,211,141,0.28); border-radius: 999px; padding: 0.18rem 0.6rem;
}
.chip.blue { color: #d8e8ff; background: rgba(126,184,255,0.12); border-color: rgba(126,184,255,0.28); }
.chip.warn { color: #ffe9b0; background: rgba(240,180,41,0.12); border-color: rgba(240,180,41,0.28); }
.links { margin-top: 0.65rem; color: var(--muted); font-size: 0.9rem; }
.links a { margin-right: 0.35rem; }
.hero-img { width: 100%; border-radius: 10px; margin-top: 0.75rem; display: block; }
.btn {
  display: inline-block; background: #1f6feb; color: #fff !important; border: 0;
  padding: 0.65rem 1.1rem; border-radius: 8px; font-size: 0.95rem; cursor: pointer;
  text-decoration: none !important;
}
.btn:hover { background: #1a5fd4; }
.btn.secondary { background: #2a3544; }
.btn.secondary:hover { background: #354357; }
.upload-box {
  border: 2px dashed rgba(126,184,255,0.35); border-radius: 14px;
  padding: 1.5rem; background: var(--panel2);
}
label.chk { display: block; margin: 0.45rem 0; color: var(--muted); }
table.metrics {
  width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.75rem;
}
table.metrics th, table.metrics td {
  border-bottom: 1px solid var(--border); padding: 0.55rem 0.45rem; text-align: left;
}
table.metrics th { color: var(--muted); font-weight: 600; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.thumb-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem;
}
.thumb {
  background: var(--panel2); border: 1px solid var(--border); border-radius: 10px;
  overflow: hidden;
}
.thumb img { width: 100%; display: block; }
.thumb .cap { padding: 0.45rem 0.55rem; font-size: 0.82rem; color: var(--muted); }
.tabs { display: flex; gap: 0.35rem; flex-wrap: wrap; margin-bottom: 1rem; }
.tab {
  padding: 0.45rem 0.85rem; border-radius: 8px; background: var(--panel2);
  border: 1px solid var(--border); color: var(--muted); font-size: 0.88rem;
}
.tab.active { color: #d8ffe7; border-color: rgba(95,211,141,0.35); background: rgba(95,211,141,0.10); }
.notice {
  border-left: 3px solid var(--warn); padding: 0.65rem 0.85rem;
  background: rgba(240,180,41,0.08); border-radius: 6px; color: #f5e6c8; font-size: 0.9rem;
}
"""


def page_shell(
    *,
    title: str,
    body: str,
    nav: list[tuple[str, str, bool]] | None = None,
) -> str:
    nav = nav or []
    nav_html = '<nav class="topnav"><span class="brand">CETI Research</span>'
    for label, href, active in nav:
        cls = "active" if active else ""
        nav_html += f'<a href="{_esc(href)}" class="{cls}">{_esc(label)}</a>'
    nav_html += "</nav>"
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{RESEARCHER_CSS}</style>
</head><body>
<div class="wrap">
{nav_html}
{body}
</div>
</body></html>"""


def portal_nav(active: str) -> list[tuple[str, str, bool]]:
    return [
        ("Upload", "/", active == "home"),
        ("Browse Runs", "/browse", active == "browse"),
    ]


def render_pointcloud_results(
    manifest: dict,
    *,
    run_id: str,
    zip_url: str,
    api_url: str = "",
    gallery_url: str = "",
    asset_url,
) -> str:
    rows = []
    for item in manifest.get("outputs", []):
        name = item.get("source", "")
        chips = []
        for key, label, cls in [
            ("point_count", lambda v: f"{v:,} pts", ""),
            ("mask_pixels", lambda v: f"{v:,} mask px", ""),
            ("mask_coverage", lambda v: f"{float(v):.0%} coverage", "blue"),
            ("mask_mode", str, ""),
            ("intrinsics_model", str, "blue"),
            ("depth_semantics", str, "blue"),
            ("aspect_mode", str, "blue"),
            ("plane_fit_rms", lambda v: f"RMS {float(v):.2f}", "warn"),
        ]:
            val = item.get(key)
            if val is not None and val != "":
                try:
                    text = label(val) if callable(label) else str(val)
                except (TypeError, ValueError):
                    text = str(val)
                chips.append(f'<span class="chip {cls}">{_esc(text)}</span>')
        chip_html = f'<div class="chips">{"".join(chips)}</div>' if chips else ""

        thumbs = []
        for key, cap in [
            ("showcase", "Showcase board"),
            ("segmentation_preview", "Segmentation"),
            ("pointcloud_3d_contrast", "3D contrast"),
            ("pointcloud_3d_payload", "3D payload only"),
            ("pointcloud_3d_scene", "3D tank scene"),
            ("contour_preview", "Contour"),
        ]:
            rel = item.get(key)
            if rel:
                thumbs.append(
                    f'<div class="thumb"><img src="{_esc(asset_url(run_id, rel))}" alt="{_esc(cap)}"/>'
                    f'<div class="cap">{_esc(cap)}</div></div>'
                )

        links = []
        ply_rel = item.get("pointcloud_ply")
        ply_scene = item.get("pointcloud_ply_scene")
        if ply_rel:
            explorer = f"/results/{run_id}/explorer?ply={quote(ply_rel, safe='')}&mode=payload"
            links.append(f'<a class="btn" href="{explorer}">Explore payload 3D ↗</a>')
        if ply_scene:
            explorer_scene = (
                f"/results/{run_id}/explorer?ply={quote(ply_scene, safe='')}&mode=scene"
            )
            links.append(f'<a class="btn secondary" href="{explorer_scene}">Explore scene 3D ↗</a>')
        for key, label in [
            ("pointcloud_ply", "PLY payload"),
            ("pointcloud_ply_scene", "PLY scene"),
            ("payload_mask", "Mask PNG"),
            ("pointcloud_intrinsics", "Intrinsics JSON"),
            ("distance_json", "Distances JSON"),
            ("distance_csv", "Distances CSV"),
        ]:
            rel = item.get(key)
            if rel:
                links.append(f'<a href="{_esc(asset_url(run_id, rel))}">{_esc(label)}</a>')

        hero = item.get("showcase") or item.get("preview") or item.get("file")
        hero_html = ""
        if hero:
            hero_html = (
                f'<a href="{_esc(asset_url(run_id, hero))}">'
                f'<img class="hero-img" src="{_esc(asset_url(run_id, hero))}" alt="{_esc(name)}"/></a>'
            )
        thumb_html = f'<div class="thumb-grid">{"".join(thumbs)}</div>' if thumbs else ""
        link_html = f'<p class="links">{" · ".join(links)}</p>' if links else ""

        rows.append(
            f'<article class="card"><header><h2>{_esc(name)}</h2>{chip_html}</header>'
            f"{hero_html}{thumb_html}{link_html}</article>"
        )

    mode = manifest.get("depth_mode", "")
    lead = (
        "Payload-only 3D reconstruction with DJI Action 4 intrinsics and CETI depth."
        if mode == "pointcloud"
        else f"Depth run (mode: {mode}). Legacy results may use distance JSON instead of PLY."
    )
    api_btn = f'<a class="btn secondary" href="{_esc(api_url)}">web_api.json</a>' if api_url else ""
    gallery_btn = (
        f'<a class="btn secondary" href="{_esc(gallery_url)}">Static gallery</a>' if gallery_url else ""
    )
    body = f"""
<section class="hero">
  <h1>Results — {_esc(run_id)}</h1>
  <p class="lead">{_esc(lead)}</p>
  <p class="meta">Device {_esc(manifest.get("device",""))} · Mode {_esc(mode)} · Model {_esc(manifest.get("unproject_model",""))}</p>
  <p class="links">
    <a class="btn" href="{_esc(zip_url)}">Download ZIP</a>
    {api_btn}
    {gallery_btn}
  </p>
</section>
{"".join(rows) if rows else '<article class="card"><p class="meta">No outputs in this run.</p></article>'}
<p class="meta"><a href="/">← Upload more</a> · <a href="/browse">Browse runs</a></p>
"""
    return page_shell(title=f"CETI Results — {run_id}", body=body, nav=portal_nav("home"))


def render_browse_page(pointcloud_runs: list[dict]) -> str:
    pc_items = "".join(
        f'<li><a href="/results/{_esc(r["run_id"])}">{_esc(r["run_id"])}</a> '
        f'<span class="meta">({r.get("n_outputs", 0)} outputs)</span></li>'
        for r in pointcloud_runs
    ) or "<li class='meta'>No runs yet — upload images from the home page.</li>"

    body = f"""
<section class="hero">
  <h1>Browse Results</h1>
  <p class="lead">All point-cloud pipeline runs.</p>
</section>
<article class="card"><ul>{pc_items}</ul></article>
<p class="meta"><a href="/">← Upload more</a></p>
"""
    return page_shell(title="Browse CETI Results", body=body, nav=portal_nav("browse"))


def render_pointcloud_explorer(
    *,
    title: str,
    ply_url: str,
    back_url: str,
    point_count: int | None = None,
    mask_url: str = "",
    source_name: str = "",
    mode: str = "payload",
    contrast_url: str = "",
) -> str:
    """Full-screen Three.js explorer (orbit / zoom / pan) for payload or scene PLY files."""
    mode_label = "Payload only" if mode == "payload" else "Tank scene (payload + interior)"
    pts_label = f"{point_count:,} points · {mode_label}" if point_count else mode_label
    mask_link = (
        f'<a href="{_esc(mask_url)}" target="_blank" rel="noopener">segmentation mask</a>'
        if mask_url
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)} — CETI Explorer</title>
<style>
  :root {{
    --bg: #060a0f;
    --panel: rgba(14, 20, 30, 0.92);
    --text: #edf2f7;
    --muted: #8fa3b8;
    --accent: #5fd38d;
    --accent2: #7eb8ff;
    --border: rgba(255,255,255,0.10);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: var(--bg); color: var(--text);
    font-family: "SF Pro Text", "Segoe UI", system-ui, sans-serif; }}
  #canvas-wrap {{ position: fixed; inset: 0; }}
  canvas {{ display: block; width: 100%; height: 100%; }}
  .hud {{
    position: fixed; top: 14px; left: 14px; right: 14px; z-index: 5;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; pointer-events: none;
  }}
  .panel {{
    pointer-events: auto; background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 14px; backdrop-filter: blur(8px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  }}
  .panel h1 {{ margin: 0 0 4px; font-size: 1rem; letter-spacing: -0.01em; }}
  .panel p {{ margin: 0; color: var(--muted); font-size: 0.86rem; }}
  .controls {{
    position: fixed; bottom: 14px; left: 14px; right: 14px; z-index: 5;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center; pointer-events: none;
  }}
  .controls .panel {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
  label {{ font-size: 0.84rem; color: var(--muted); display: flex; gap: 8px; align-items: center; }}
  input[type=range] {{ width: 120px; }}
  button, a.btn {{
    pointer-events: auto; border: 0; border-radius: 8px; padding: 0.5rem 0.85rem;
    background: #1f6feb; color: #fff; font-size: 0.86rem; cursor: pointer; text-decoration: none;
  }}
  button.secondary, a.btn.secondary {{ background: #2a3442; color: var(--text); }}
  #status {{ color: var(--accent); font-size: 0.84rem; }}
  .help {{ position: fixed; right: 14px; bottom: 14px; z-index: 4; color: var(--muted); font-size: 0.78rem; }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div class="hud">
  <div class="panel">
    <h1>{_esc(source_name or title)}</h1>
    <p>{_esc(pts_label)} · relative depth</p>
    <p id="status">Loading PLY…</p>
  </div>
  <div class="panel">
    <p><a class="btn secondary" href="{_esc(back_url)}">← Back to results</a></p>
    {f'<p style="margin-top:6px">{mask_link}</p>' if mask_link else ''}
    {f'<p style="margin-top:6px"><a class="btn secondary" href="{_esc(contrast_url)}">View 3D contrast board</a></p>' if contrast_url else ''}
  </div>
</div>
<div class="controls">
  <div class="panel">
    <label>Point size <input id="pt-size" type="range" min="0.1" max="4" step="0.05" value="{"0.35" if mode == "scene" else "0.55"}"/></label>
    <label><input id="show-grid" type="checkbox" checked/> Grid</label>
    <label><input id="show-axes" type="checkbox"/> Axes</label>
    <button id="reset-cam" class="secondary" type="button">Reset view</button>
    <button id="toggle-spin" class="secondary" type="button">Auto-rotate</button>
  </div>
</div>
<p class="help">Drag to orbit · scroll to zoom · right-drag to pan</p>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';

const plyUrl = {_json_esc(ply_url)};
const viewMode = {_json_esc(mode)};
const defaultPtSize = viewMode === 'scene' ? 0.35 : 0.55;
const wrap = document.getElementById('canvas-wrap');
const statusEl = document.getElementById('status');
const sizeInput = document.getElementById('pt-size');
const gridToggle = document.getElementById('show-grid');
const axesToggle = document.getElementById('show-axes');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060a0f);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
wrap.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

const grid = new THREE.GridHelper(200, 40, 0x2a3a4a, 0x182230);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
const axes = new THREE.AxesHelper(50);
axes.visible = false;
scene.add(axes);

let cloud = null;
let defaultCam = null;
let autoRotate = false;

function fitCameraToObject(obj) {{
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  const dist = maxDim * 2.2;
  // Angled perspective so depth (Z) is visible — not a top-down flat card.
  camera.position.set(center.x + dist * 0.62, center.y + dist * 0.38, center.z + dist * 0.58);
  controls.target.copy(center);
  controls.update();
  defaultCam = {{
    pos: camera.position.clone(),
    target: controls.target.clone(),
  }};
}}

function resize() {{
  const w = wrap.clientWidth, h = wrap.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}}
window.addEventListener('resize', resize);
resize();

const loader = new PLYLoader();
loader.load(
  plyUrl,
  (geometry) => {{
    geometry.computeVertexNormals();
    geometry.center();
    const hasColor = geometry.hasAttribute('color');
    const n = geometry.attributes.position.count;
    const autoSize = n > 50000 ? defaultPtSize * 0.22 : (n > 30000 ? defaultPtSize * 0.35 : (n > 12000 ? defaultPtSize * 0.55 : defaultPtSize));
    sizeInput.value = String(autoSize.toFixed(2));
    const material = new THREE.PointsMaterial({{
      size: parseFloat(sizeInput.value || autoSize),
      sizeAttenuation: true,
      vertexColors: hasColor,
      color: hasColor ? 0xffffff : 0x9fd8ff,
      depthWrite: viewMode !== 'scene',
      transparent: viewMode === 'scene',
      opacity: viewMode === 'scene' ? 0.88 : 1.0,
    }});
    cloud = new THREE.Points(geometry, material);
    scene.add(cloud);
    fitCameraToObject(cloud);
    statusEl.textContent = `Loaded ${{n.toLocaleString()}} vertices`;
  }},
  (xhr) => {{
    if (xhr.total) {{
      statusEl.textContent = `Loading… ${{Math.round(100 * xhr.loaded / xhr.total)}}%`;
    }}
  }},
  (err) => {{
    statusEl.textContent = 'Failed to load PLY';
    console.error(err);
  }},
);

sizeInput.addEventListener('input', () => {{
  if (cloud) cloud.material.size = parseFloat(sizeInput.value);
}});
gridToggle.addEventListener('change', () => {{ grid.visible = gridToggle.checked; }});
axesToggle.addEventListener('change', () => {{ axes.visible = axesToggle.checked; }});
document.getElementById('reset-cam').addEventListener('click', () => {{
  if (defaultCam) {{
    camera.position.copy(defaultCam.pos);
    controls.target.copy(defaultCam.target);
    controls.update();
  }} else if (cloud) {{
    fitCameraToObject(cloud);
  }}
}});
document.getElementById('toggle-spin').addEventListener('click', (e) => {{
  autoRotate = !autoRotate;
  e.target.textContent = autoRotate ? 'Stop rotate' : 'Auto-rotate';
}});

function animate() {{
  requestAnimationFrame(animate);
  if (autoRotate && cloud) cloud.rotation.y += 0.004;
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""


def _json_esc(value: str) -> str:
    return json.dumps(value)


def write_static_html(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
