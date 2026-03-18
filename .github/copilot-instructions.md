# Copilot Instructions for ICLPose

## What This Project Does

**ICLPose** is a 6-DOF camera pose estimation system for 3D Gaussian Splatting (3DGS) feature fields. Given a query image with an initial pose estimate, it iteratively refines the pose through:
1. Differentiable rendering of multi-scale feature maps from a pre-trained 3DGS scene
2. Coarse-to-fine optical flow matching (7×10 → 15×20 → 35×46)
3. Image Jacobian-based weighted least-squares geometric pose solving via SE(3) Lie algebra

**Offline pre-processing** (done once per scene): extract Stable Diffusion + DINOv2 features for every training frame, and train 3DGS feature models per scale.

## Environment Setup

```bash
conda env create -f environment.key.yml
conda activate geo-aware            # NOTE: env is named geo-aware, not iclpose
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

Requires CUDA 11.6+ (RTX 4090 supported). See `docs/ENVIRONMENT_SETUP.md` for GPU-specific notes.

## Training

```bash
# Primary training script (current best)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp032_cosine_fiters8.yaml

# Resume from checkpoint (loads full state: model + optimizer + scheduler)
python scripts/train_ms_flow.py --config configs/exp032_cosine_fiters8.yaml \
    --resume output/exp032_cosine_fiters8/checkpoints/latest.pth

# Warmstart (load model weights only, reset optimizer/scheduler)
python scripts/train_ms_flow.py --config configs/exp033_new_exp.yaml \
    --warmstart output/exp032_cosine_fiters8/checkpoints/best.pth
```

**Monitor training:**
```bash
grep '\[Val E' output/exp032_train.log | tail -10
grep '★' output/exp032_train.log       # Best results
```

**Evaluation:**
```bash
python scripts/eval_iterative.py --config configs/exp032_cosine_fiters8.yaml \
    --checkpoint output/exp032_cosine_fiters8/checkpoints/best.pth
```

**Smoke test (single batch, no full dataset needed):**
```bash
python scripts/smoke_test_ms_flow.py
```

## Architecture

### Key Files
| File | Role |
|------|------|
| `scripts/train_ms_flow.py` | Training entry point + `MSFlowTrainer` class, loss functions |
| `ic_models/ms_flow_pose_net.py` | Core model `MSFlowPoseNet` (~676 lines, ~4.17M params) |
| `modules/geometry_solver.py` | Image Jacobian + weighted least-squares SE(3) solver |
| `modules/multiscale_renderer.py` | `MultiScaleRenderer` — gsplat-based 4-scale 3DGS rendering |
| `modules/lie_algebra.py` | `se3_exp`, `se3_log`, `so3_exp`, `compute_gt_flow` |
| `data/dataset_v4.py` | Current dataset loader |
| `configs/exp032_cosine_fiters8.yaml` | Best configuration (reference for new experiments) |

### Model Forward Pass (`MSFlowPoseNet`)
```
Query features (SD+DINO) + Rendered reference features (from 3DGS at current pose)
 ↓ ScaleDecoder (1×1 convs → 64-d, L2 normalized) per scale
 ↓ Stage 1 Coarse: global_correlation → FlowRefinementHead → flow_coarse  [7×10]
 ↓ Stage 2 Mid: guided_local_correlation → flow_mid  [15×20]
 ↓ Stage 3 Fine (8 GRU iters): FineDualDecoder (SD+DINO) → ConvGRU → flow_fine  [35×46]
 ↓ Stage 4 Geometry: compute_image_jacobian → diff_pose_solve → Δξ → se3_exp → ΔT
 → T_new = ΔT @ T_old   (outer loop repeats 3–5× per sample)
```

### Deprecated Training Scripts
- `train.py` — Original single-scale (ICPoseNet), has DDP support, **deprecated**
- `train_v2.py` — C2F + overlap detection (ICPoseNetV2), **deprecated**
- `train_v3.py` — Flow loss + curriculum (ICPoseNetV3), **deprecated**
- Use `scripts/train_ms_flow.py` for all current work

## Configuration System

Config files are plain YAML (no Hydra). Top-level keys: `exp_name`, `output_dir`, `model`, `renderer`, `data`, `training`.

**Creating a new experiment:**
```bash
cp configs/exp032_cosine_fiters8.yaml configs/exp033_your_change.yaml
# Edit exp_name, output_dir, and the parameters you're changing
```

**Key `training.loss` fields:**
- `rot_loss_type: cosine` — Use `1-cos(θ)`, **not** `acos` (avoids gradient explosion near 0°)
- `trans_weight: 10.0` — Translation gets higher weight than rotation (`rot_weight: 1.0`)
- `gamma: 0.8` — RAFT-style sequence loss decay over fine iterations

**Noise curriculum** (`training.noise_curriculum`): Starts at 2°/0.05m, ramps to 8°/0.25m over `warmup_epochs` (default 40). Fine-tuning experiments often use `warmstart` from exp032's best checkpoint.

## Data Layout

```
dataset/room_0/
  Sequence_1/traj_w_c.txt     # Poses as 4×4 camera-to-world matrices (one per line, space-separated)
  Sequence_1/depth/           # Depth PNGs at fine resolution (35×46)
  Sequence_2/                 # Val split

output/features_multiscale/room_0/
  coarse/rgb_0_coarse_1280x7x10.pt     # SD stage-5 features
  mid/rgb_0_mid_1280x15x20.pt          # SD stage-4 features
  fine_sd/rgb_0_fine_sd_640x35x46.pt   # SD stage-3 features
  fine_dino/rgb_0_fine_dino_768x35x46.pt  # DINOv2 features
```

Dataset V2 uses subdirectory names `sd_s5/sd_s4/sd_s3/dino` — both formats are auto-detected in `dataset_v4.py`.

**Pose convention**: Files store camera-to-world (c2w); internally the model uses world-to-camera (w2c). Dataset loader inverts automatically.

## Key Conventions

### Loss Functions (in `scripts/train_ms_flow.py`)
- `multiscale_flow_loss()`: Huber loss + RAFT gamma decay, masked by valid depth pixels
- `pose_loss()`: cosine rotation loss + L1 translation; computed in fp32 to avoid NaN
- `confidence_regularization_loss()`: penalizes confidence mean outside [0.3, 0.7]

### Tensor Conventions
- Images/features: `(B, C, H, W)` batch-first
- Poses: `(B, 4, 4)` homogeneous matrices
- Flow: `(B, 2, H, W)` in pixel units at the respective resolution scale
- All batch operations use `.to(device)` with the model's `.device` property

### Checkpoints
- Saved to `output/{exp_name}/checkpoints/{latest.pth, best.pth}`
- Contain: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `metrics`
- `--resume` restores everything; `--warmstart` restores model weights only

### Feature Extraction (offline, run once per scene)
```bash
python scripts/extract_multiscale_features.py --scene room_0
```
Uses `feature_extraction/fused_feature_extractor.py` which wraps `extractor_sd.py` (Stable Diffusion UNet internals) and `extractor_dino.py` (DINOv2 ViT-G/14).

### 3DGS Feature Model Training (offline, run once per scene per scale)
See `feature_3dgs/` — trains a separate Gaussian feature model for each scale (coarse/mid/fine_sd/fine_dino). Model paths are specified in the config under `renderer.scale_model_paths`.
