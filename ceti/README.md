# CETI Depth & Whale Perception Stack

**Harvard SEAS / Project CETI** — Monocular underwater depth estimation and whale detection built on [Depth Anything](https://github.com/LiheYoung/Depth-Anything), extending it for lab robots and field deployment in support of Professor Stephanie Gil's AVATARS autonomy framework.

## Research Goals

| Track | Objective | Status |
|-------|-----------|--------|
| **A. Underwater Depth** | Metric depth on lab ROV/AUV cameras for navigation and 3D scene understanding | Pipeline ready; requires FLSea/SQUID download |
| **B. Whale Detection** | Visual detection of sperm whales (surface + partial submergence) for AVATARS rendezvous | Pipeline ready; **no CETI whale labels exist yet** — see Data Strategy |
| **B2. Whale Depth** | Relative depth adapted to marine/aerial whale scenes (not in DA pretrain) | **Fine-tune ready** — pseudo-depth distillation, no metric GT |
| **C. Robot Integration** | Real-time inference on lab robots via ROS2 | Node scaffold ready |

## Architecture

```
Camera (ROV / Aerial Drone)
        │
        ▼
┌───────────────────┐
│ Underwater Preproc │  ← color correction, dehazing (optional)
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌─────────┐ ┌──────────────┐
│ Depth   │ │ Whale        │
│ Anything│ │ Detector     │
│ (metric)│ │ (YOLO + DA   │
│         │ │  backbone)   │
└────┬────┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
   AVATARS Fusion (depth + bbox + tag AOA)
            ▼
   Rendezvous Planning / Conservation
```

## Quick Start (Mac M5 Max 128GB — recommended)

See **[README.md](../README.md)** for the CETI Depth professor demo and portal.

```bash
git clone https://github.com/Todd7777/CETI_Depth.git
cd CETI_Depth
bash ceti/scripts/setup_mac_mps.sh
bash ceti/scripts/download_all_online_data.sh
bash ceti/scripts/train_mac_full.sh
```

Professor Gil / AVATARS alignment: [docs/AVATARS_ALIGNMENT.md](docs/AVATARS_ALIGNMENT.md)

### Linux / CUDA (optional)

```bash
pip install -r requirements.txt && pip install -r ceti/requirements.txt
bash ceti/scripts/download_checkpoints.sh
python ceti/scripts/smoke_test.py
bash ceti/scripts/download_all_online_data.sh
CETI_DEVICE=cuda python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth_m5max_128gb.yaml
```

### Legacy quick path

```bash
bash ceti/scripts/curate_underwater_field.sh
bash ceti/scripts/train_mac_full.sh

# Or prove only (~1–2 min):
bash ceti/scripts/prove_pipeline.sh --skip-metric-train

# 6. Run depth on sample underwater video
python ceti/depth/infer_robot.py \
  --encoder vits \
  --input assets/examples_video/davis_dolphins.mp4 \
  --outdir ceti/outputs/depth_demo \
  --underwater-preprocess
```

Artifacts land in `ceti/outputs/proof/` (side-by-side panels + `report.json`).

### Professor / lab upload demo (JPEG + MP4 → RGB|depth)

**Drop folder:** copy files into `ceti/inbox/uploads/`, then:

```bash
bash ceti/scripts/run_professor_pipeline.sh
```

**Web portal:** `bash ceti/scripts/launch_professor_portal.sh` → http://127.0.0.1:7860

Full guide: [docs/PROFESSOR_DEMO.md](docs/PROFESSOR_DEMO.md) · inbox help: [inbox/README.md](inbox/README.md)

## Real Underwater Imagery (not synthetic)

CETI depth adaptation uses **real** open-water and in-situ marine RGB:

| Source | Content | Role |
|--------|---------|------|
| DAVIS `davis_dolphins` / `davis_seasnake` | Classic underwater video benchmarks | ROV-like color cast, motion |
| [Submerged3D](https://huggingface.co/datasets/theflash987/Submerged3D) | Deep-sea wreck RGB (BMVC 2025) | Turbidity, low light |
| [AQUA20](https://huggingface.co/datasets/taufiktrf/AQUA20) | Marine species in challenging visibility | Domain diversity |
| `data/ceti_field/` | Your CETI / Dominica / lab drops | Target domain |

```bash
bash ceti/scripts/curate_underwater_field.sh   # builds data/underwater_field/rgb/
python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth.yaml
```

Config **`ceti/configs/whale_depth_m5max_128gb.yaml`** — ViT-L, **batch 16**, 40 epochs, **`device: mps`**, AMP (128GB M5 Max).  
Fallback: `whale_depth.yaml` (batch 8).

Metric depth (meters) still requires FLSea/SQUID RGB-D; synthetic `build_ceti_lab.py` remains a fallback for pipeline smoke tests only.

## Pipeline Proof (Lab / AVATARS Readiness)

One command demonstrates that **underwater RGB produces depth maps** aligned with the CETI research stack ([Depth Anything](https://github.com/LiheYoung/Depth-Anything) + underwater adaptation per [arxiv:2507.02148](https://arxiv.org/abs/2507.02148)):

| Phase | What it proves |
|-------|----------------|
| **1. Relative depth** | Real underwater video (`davis_dolphins.mp4`) → depth; compares baseline, color correction, optional whale-adapted checkpoint |
| **2. Metric depth** | Synthetic underwater RGB-D lab set → ZoeDepth metric inference in **meters**; optional fine-tune shows measurable AbsRel gain |

```bash
# Fast CPU proof (~1 min): relative + metric inference
bash ceti/scripts/prove_pipeline.sh --skip-metric-train

# Full proof with metric fine-tune (use GPU)
bash ceti/scripts/prove_pipeline.sh --metric-epochs 3

# Relative depth only
bash ceti/scripts/prove_pipeline.sh --quick
```

Build the synthetic lab set manually (for custom scenes):

```bash
python ceti/scripts/build_ceti_lab.py --sources assets/examples --max-images 10
```

## Track A: Underwater Depth Fine-Tuning

Depth Anything performs poorly underwater out-of-the-box due to color attenuation, scattering, and turbidity (see [arxiv:2507.02148](https://arxiv.org/abs/2507.02148)). We fine-tune the **metric depth head** (ZoeDepth) on underwater datasets.

### Supported Datasets

| Dataset | Type | Depth Range | Download |
|---------|------|-------------|----------|
| [FLSea](https://github.com/xahidz/revisiting-dm) | Real metric | 0.5–20 m | Request from authors |
| [SQUID](https://csms.haifa.ac.il/profiles/tTreibitz/data-sets/underwater/17/) | Real metric | 1–10 m | Free registration |
| Synthetic underwater Hypersim | Synthetic | Configurable | Generated via `ceti/preprocessing/synthetic_underwater.py` |

### Prepare Data

```bash
bash ceti/scripts/prepare_underwater_data.sh --dataset flsea --root ./data/flsea
bash ceti/scripts/prepare_underwater_data.sh --dataset squid --root ./data/squid
```

### Train

```bash
cd metric_depth
python ../ceti/depth/train_underwater.py \
  --dataset flsea \
  --epochs 20 \
  --batch-size 8 \
  --pretrained-resource "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"
```

### Evaluate

```bash
cd metric_depth
python evaluate.py -m zoedepth \
  --pretrained_resource="local::./checkpoints/ceti_underwater_flsea.pt" \
  -d flsea
```

## Track B: Whale Detection — The Data Gap

**Critical:** Depth Anything and all standard depth datasets contain **zero sperm whale annotations**. CETI field footage is largely acoustic; visual whale data must be built deliberately.

### Three-Phase Data Strategy

#### Phase 1 — Bootstrap (Week 1–2)
Public datasets to pre-train a general *cetacean* detector:

| Source | Species | Modality | Labels | Script |
|--------|---------|----------|--------|--------|
| [Beluga-5k](https://github.com/parhamap/SSS_Beluga_Whales_Dataset) | Beluga | Surface photos | Bounding boxes | `data_curation/download_public.py --beluga` |
| [Whales from Space](https://doi.org/10.5285/C1AFE32C-493C-4DC7-AF9F-649593B97B2C) | Baleen whales | Satellite | Boxes + points | `--whales-from-space` |
| CETI aerial drone footage | Sperm whale | Aerial video | **Needs annotation** | `extract_frames.py` |

#### Phase 2 — CETI Domain Adaptation (Week 3–6)
1. Extract frames from lab robot cameras + Dominica field drones
2. Annotate with [CVAT](https://www.cvat.ai/) or [Label Studio](https://labelstud.io/) — YOLO format
3. Fine-tune on combined dataset

```bash
# Extract candidate frames (high motion / acoustic trigger sync)
python ceti/whale/data_curation/extract_frames.py \
  --video-dir ./data/ceti_field/ \
  --output ./data/whale/raw_frames/ \
  --sample-rate 2

# After manual annotation in CVAT, convert to YOLO format
python ceti/whale/data_curation/convert_annotations.py \
  --cvat-export ./data/whale/cvat_export/ \
  --output ./data/whale/yolo/

# Train
python ceti/whale/train_whale.py \
  --data ceti/configs/whale_detection.yaml \
  --epochs 100 \
  --encoder-backbone depth_anything_vits14
```

#### Phase 3 — Semi-Supervised Expansion (Week 7+)
When labeled data is scarce (<500 images), use teacher-student pseudo-labeling:

```bash
python ceti/whale/data_curation/semi_supervised_label.py \
  --teacher ceti/checkpoints/whale_v1.pt \
  --unlabeled ./data/ceti_field/unlabeled/ \
  --confidence 0.85 \
  --output ./data/whale/pseudo_labels/
```

Human review **required** before adding pseudo-labels to training set.

### Detection Model

We use **YOLOv8** with optional **Depth Anything ViT-S backbone** initialization (transfer from depth pretraining improves feature quality in low-contrast marine scenes). Depth maps from Track A provide auxiliary distance-to-whale estimates for AVATARS.

## Track B2: Whale / Marine Depth Fine-Tuning

**Problem:** Depth Anything was trained on general terrestrial video — sperm whales, open-ocean surface glare, and aerial drone geometry are **out of distribution**. Metric depth GT for whales essentially does not exist.

**Solution:** Domain-adapt the **relative** Depth Anything model on your whale/marine RGB using **frozen-teacher distillation** (scale-shift invariant loss). No depth labels required — only images.

### Prepare image lists

Collects frames from YOLO splits, bootstrap datasets, raw CETI field frames, etc.:

```bash
bash ceti/scripts/prepare_whale_depth_data.sh
```

### Train

```bash
python ceti/depth/train_whale_depth.py --config ceti/configs/whale_depth.yaml --epochs 30
# Quick validation (no GPU training):
python ceti/depth/train_whale_depth.py --dry-run
```

Checkpoints: `checkpoints/ceti_whale_depth/best.pt` and `last.pt`.

### Inference with fine-tuned depth

```bash
python ceti/depth/infer_robot.py \
  --encoder vits \
  --depth-checkpoint ./checkpoints/ceti_whale_depth/best.pt \
  --whale-checkpoint ./ceti/checkpoints/whale_detector/weights/best.pt \
  --input ./data/ceti_field/sample.mp4
```

**Recommended workflow:** (1) fine-tune underwater metric depth on FLSea/SQUID if ROV RGB-D available → (2) whale relative depth on aerial/surface frames → (3) YOLO whale detector → (4) fuse in AVATARS.

For ROV underwater whale video, set `preprocess_method: combined` in `ceti/configs/whale_depth.yaml`.

## Track C: Robot Deployment

### Standalone Inference

```bash
python ceti/depth/infer_robot.py \
  --encoder vits \
  --input 0 \
  --metric-checkpoint ./checkpoints/ceti_underwater_flsea.pt \
  --whale-checkpoint ./ceti/checkpoints/whale_v1.pt \
  --underwater-preprocess \
  --publish-ros  # optional ROS2 topics
```

### ROS2 Node

```bash
# Terminal 1: ROS2
source /opt/ros/humble/setup.bash
python ceti/robot/ros2_perception_node.py \
  --encoder vits \
  --image-topic /robot/camera/image_raw \
  --depth-topic /ceti/depth \
  --detection-topic /ceti/whale_detections
```

### AVATARS Integration

`ceti/robot/avatars_pipeline.py` fuses:
- Visual whale detections (bbox + confidence)
- Metric depth at bbox center (range estimate)
- VHF tag AOA (from existing AVATARS sensors)
- Sperm whale dive model predictions

Output: ranked rendezvous waypoints for autonomous drones.

## Directory Layout

```
ceti/
├── README.md                 ← this file
├── requirements.txt
├── configs/
│   ├── underwater_metric.yaml
│   ├── whale_detection.yaml
│   └── whale_depth.yaml
├── preprocessing/
│   ├── underwater.py         ← color correction, augmentation
│   └── synthetic_underwater.py ← physics-based underwater sim
├── depth/
│   ├── underwater_dataset.py ← FLSea/SQUID loaders
│   ├── train_underwater.py
│   ├── train_whale_depth.py   ← whale/marine relative depth fine-tune
│   ├── whale_depth_dataset.py
│   └── infer_robot.py
├── whale/
│   ├── dataset.py
│   ├── model.py
│   ├── train_whale.py
│   └── data_curation/
│       ├── download_public.py
│       ├── extract_frames.py
│       ├── convert_annotations.py
│       └── semi_supervised_label.py
├── robot/
│   ├── ros2_perception_node.py
│   └── avatars_pipeline.py
└── scripts/
    ├── setup.sh
    ├── download_checkpoints.sh
    ├── prepare_underwater_data.sh
    ├── prepare_whale_depth_data.sh
    ├── build_ceti_lab.py
    ├── prove_pipeline.py
    ├── prove_pipeline.sh
    └── smoke_test.py
```

## Evaluation Protocol

For publication-quality results, report on held-out splits:

**Depth (underwater):**
- AbsRel, RMSE, δ₁ on FLSea test + SQUID test
- Compare: zero-shot DA → fine-tuned outdoor → fine-tuned underwater

**Whale detection:**
- mAP@0.5, mAP@0.5:0.95 on CETI test set
- Per-condition breakdown: surface / partial / glare / choppy sea
- False positive rate on empty ocean frames (critical for AVATARS)

## References

- Yang et al., "Depth Anything", CVPR 2024
- Gil et al., "AVATARS: Autonomous Vehicles for whAle Tracking And Rendezvous by remote Sensing", Harvard SEAS 2024
- Project CETI: https://www.projectceti.org/
- Underwater metric depth benchmark: arxiv:2507.02148

## Citation

```bibtex
@inproceedings{depthanything,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR}, year={2024}
}
```
