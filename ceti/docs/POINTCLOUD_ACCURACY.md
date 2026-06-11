# Point Cloud Accuracy — Scientific Assessment

This document is an honest summary of what the CETI tank point-cloud pipeline can and cannot claim.

## What you are seeing

The showcase 3D preview can look **curved, bowl-like, or "melted"** even when segmentation is good. That is expected with the current stack, not necessarily a bug in visualization alone.

Three independent limitations combine:

### 1. Monocular relative depth (largest factor)

CETI fine-tunes Depth Anything, which predicts **affine-invariant relative depth**, not laser range.

- No true metric scale (meters) without external calibration.
- Depth can have **radial structure** on fisheye imagery (network was not trained specifically for 155° underwater fisheye).
- A physically flat surface can still receive a smooth center-to-edge depth gradient → curved 3D when lifted.

**Verdict:** Point clouds show **qualitative shape**, not metrology.

### 2. Intrinsics are spec-derived approximations

We model DJI Osmo Action 4 from published specs:

| Parameter | Source |
|-----------|--------|
| 155° diagonal FOV | DJI product page |
| 12.7 mm (35 mm equiv.) | DJI product page |
| 1/1.3″ sensor (~9.6 × 7.2 mm) | industry typical active area |
| 16:9 frames | modeled as **vertical center crop** of 4:3 sensor |

We do **not** have checkerboard fisheye calibration from your tank camera.

Typical focal-length error without calibration: **~3–8%**. Principal point and distortion coefficients (k₁…k₄) are not measured.

**Verdict:** Intrinsics are **reasonable for visualization**, not certification-grade.

### 3. Depth ↔ camera geometry coupling

Unprojection must assume what each depth pixel *means*:

| Mode | Assumption | When to use |
|------|------------|-------------|
| `pinhole` + `z_depth` (**default**) | Depth ≈ Z distance; pinhole K on distorted pixels | Best match to Depth Anything / CETI |
| `fisheye_equidistant` + `z_depth` | Depth ≈ Z; convert to ray range via cos(θ) | Wide FOV with Z-depth semantics |
| `fisheye_equidistant` + `range_along_ray` | Depth ≈ Euclidean range along ray | Legacy; tends to exaggerate curvature |

**Verdict:** Default changed to **pinhole + z_depth** because monocular networks align better with pinhole Z in practice. Residual curvature often remains due to (1), not only intrinsics.

## Intrinsics implementation

16:9 tank frames use center-crop sensor modeling:

- Aspect mode: `16x9_video`, `4x3_photo`, or `auto`
- Pinhole `fx`, `fy` from physical focal length and cropped sensor height
- Principal point at image center for video frames
- Fisheye focal length from diagonal FOV at output resolution
- Per-run `*_intrinsics.json` records model and limitation notes

## Payload and scene clouds

CETI depth assigns different relative Z to frame bars and motor. The export pipeline:

**Payload**
- Bilateral depth smooth with frame/motor shear cap
- Hollow frame preserved (no morphological fill of void)
- Pinhole unprojection with DJI Action 4 intrinsics
- Statistical outlier filter

**Tank scene**
- Payload plus tight tank halo; view-through window excluded
- Harmonized depth on payload; smoothed surround on nearby tank surfaces

## Interpretation

| Panel | Trust level |
|-------|-------------|
| Segmentation | Good for demo if mask covers payload |
| Depth heatmap | Shows relative structure, not meters |
| 3D preview | Illustrative; compare across conditions (e.g. noise study) |
| PLY in MeshLab | Same data as preview; scale is arbitrary |

**Plane-fit RMS** in metadata is a **shape diagnostic** (how planar the cloud looks), not ground-truth error.

## For publication-grade accuracy

1. **Fisheye calibration** — chessboard/charuco in tank, `cv2.fisheye.calibrate`.
2. **Metric scale** — known baseline in tank or rangefinder to scale depth.
3. **Undistort** — remap RGB + depth to rectified pinhole before unprojection.
4. **Stereo or RGB-D** — if true 3D accuracy is required.

## Environment variables

```bash
export CETI_UNPROJECT_MODEL=pinhole              # default
export CETI_DEPTH_SEMANTICS=z_depth              # default
export CETI_ASPECT_MODE=auto                     # or 16x9_video, 4x3_photo
export CETI_UNPROJECT_MODEL=fisheye_equidistant  # alternative
export CETI_DEPTH_SEMANTICS=range_along_ray      # legacy fisheye behavior
```

## Bottom line

**Segmentation** can be strong. **3D geometry** is scientifically **illustrative**: useful for comparing payloads, noise levels, and pipeline changes — not for stating absolute dimensions without calibration.
