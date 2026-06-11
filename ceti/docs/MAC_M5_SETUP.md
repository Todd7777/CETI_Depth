# M5 Max (128GB) — bulletproof training (final)

## One command

```bash
cd ~/Whale-Depth-Anything
git pull origin main
chmod +x ceti/scripts/*.sh

# optional: export HF_TOKEN='hf_...'

bash ceti/scripts/mac_train_bulletproof.sh
```

That is the **only** script you need after the first `setup_mac_mps.sh`.

---

## What it does (safe + fast)

| Step | Action |
|------|--------|
| 1 | Verify MPS |
| 2 | Ensure **5000+ images on disk** (downloads if missing) |
| 3 | **Cache teacher depth once** → `data/teacher_cache/vitl_518/` |
| 4 | Train **student only** on MPS (batch 14, workers 0) |
| 5 | Validate every **10** epochs + save `best.pt` |

**Why this is bulletproof**

- No `workers > 0` on Mac → avoids `Killed: 9`
- No dual ViT-L every step → ~2× faster, much less memory
- `batch_size: 14` only safe with cached teacher (one model on GPU)
- Train-loss checkpoint when val is skipped between val_every

---

## First time only

```bash
git clone https://github.com/Todd7777/Whale-Depth-Anything.git
cd Whale-Depth-Anything
bash ceti/scripts/setup_mac_mps.sh   # venv + MPS + smoke 8/8
bash ceti/scripts/mac_train_bulletproof.sh
```

---

## Config (do not change unless OOM)

`ceti/configs/whale_depth_m5max_128gb.yaml`

```yaml
cache_teacher: true
cache_batch_size: 16
batch_size: 14
workers: 0
val_every: 10
epochs: 40
```

---

## If `Killed: 9`

```bash
# Edit config: batch_size: 10 or 8
bash ceti/scripts/mac_train_bulletproof.sh
```

Or:

```bash
export CETI_TEACHER_ON_CPU=1
bash ceti/scripts/mac_train_bulletproof.sh
```

---

## Resume training

```bash
bash ceti/scripts/resume_mac_train.sh
```

The script **auto-picks `best.pt`** if `last.pt` diverged (train loss much higher than best). Safe resume uses `resume_lr_multiplier` and `resume_freeze_encoder` from config. `last.pt` stores **optimizer state** for seamless continuation; `best.pt` is inference-only (weights).

After `Killed: 9`, lower `batch_size` / `val_batch_size` if needed (default 8 / 4).

Teacher cache is reused if already built.

---

## Outputs

- `checkpoints/ceti_whale_depth/best.pt`
- `checkpoints/ceti_whale_depth/last.pt`
- `data/teacher_cache/vitl_518/` (reuse on re-run)

---

## Inference

```bash
source .venv/bin/activate
python ceti/depth/infer_robot.py \
  --encoder vitl \
  --depth-checkpoint checkpoints/ceti_whale_depth/best.pt \
  --input assets/examples_video/davis_dolphins.mp4 \
  --underwater-preprocess \
  --outdir ceti/outputs/depth_demo
```
