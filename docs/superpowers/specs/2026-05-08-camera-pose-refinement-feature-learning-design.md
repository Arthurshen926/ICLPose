# Task-Oriented Feature Learning for Camera Pose Refinement — Design Spec

## Context

ICLPose 项目目标：用 RADIO ViT-H/16 teacher 的特征空间作为起点，学习面向定位的 map/query 特征。当前系统有完整 pipeline（DCFF map field、query student、retrieval/init cache、pose refinement），但 real-init 下精度很差（top1 CLS ~4493mm），而 oracle-init 下 refinement 有效（~0.30°/166mm）。核心瓶颈：候选生成（retrieval）不可微，训练和评估之间存在断裂带。

## Strategy: Two-Phase Architecture

Paper 主线收敛到 **Camera Pose Refinement**（给定初始位姿，学习面向位姿精化的特征）。完整重定位（可微检索）作为 Phase 2 扩展。两个阶段共享同一组 coarse/fine 特征，不做架构改动。

## Coarse/Fine Definition by Task, Not Semantics

| 分支 | 职责 | 输入位姿误差 | 输出 |
|---|---|---|---|
| Coarse | large-basin pose correction | 0.5m–2m / 5°–30° | feature-metric energy for coarse alignment |
| Fine | high-precision correspondence | 0.05m–0.5m / 1°–10° | dense corr + WLS/PnP + featuremetric alignment |

训练时通过扰动分桶强制 coarse 和 fine 各司其职。

## Architecture

```
                    Shared Backbone
  DCFF Map Field                 Query Student
  (Gaussian latent               (CNN + WindowAttn +
   + hash grid +                  global context +
   spatial decoder)               projection heads)
         │                              │
    ┌────┴────┐                  ┌──────┴──────┐
    │ Coarse  │                  │   Coarse    │
    │ feature │                  │   feature   │
    │ map     │                  │   map       │
    └────┬────┘                  └──────┬──────┘
         │                              │
    ┌────┴────┐                  ┌──────┴──────┐
    │  Fine   │                  │    Fine     │
    │ feature │                  │   feature   │
    │  map    │                  │    map      │
    └─────────┘                  └─────────────┘
         │                              │
    ┌────┴──────────────────────────────┴────┐
    │  Phase 1: feature-metric refinement    │
    │  Phase 2: + differentiable retrieval   │
    └────────────────────────────────────────┘
```

### Key Components

**Query Student** (`RadioQueryStudent`): 4-stage CNN with optional windowed self-attention and global context injection. Outputs fine/coarse feature maps at different resolutions.

**DCFF Map Field** (`DeferredCascadedRenderer`): 2DGS geometry + hash grid + fine decoder + spatial refinement. Renders feature maps at any pose via alpha compositing.

**Pooling Head** `A_η` (pre-placed, Phase 2 only): Converts dense feature map [B,C,H,W] → global descriptor [B,D]. NetVLAD as primary choice (soft-assignment, differentiable, purpose-built for retrieval). D=256–512, L2-normalized.

### Feature Dimensions

Phase 1: fine=96d, coarse=32d (current, no change needed — FSM not required for CPR)
Phase 2: consider 64/64 for FSM compatibility (TBD based on Phase 1 results)

## Phase 1: Camera Pose Refinement (Paper Mainline)

### Training Protocol

**Fixed-init protocol**: GT pose + synthetic noise, bucketed by magnitude:

| Bucket | Trans | Rot | Purpose |
|---|---|---|---|
| 1 | 0.1m | 2° | Verify refinement ceiling |
| 2 | 0.25m | 5° | Normal basin |
| 3 | 0.5m | 10° | Extended basin |
| 4 | 1.0m–2.0m | 20°–30° | Coarse-only, large basin |

**Freeze/Train**:
- Freeze: DCFF geometry, hash grid, fine decoder (map field fixed)
- Train: query student (all or later stages + heads)

**Losses**:
- Coarse feature-metric energy on buckets 2–4 (large perturbations)
- Fine dense correlation + WLS/PnP on buckets 1–3
- Weak teacher anchor (low-weight RADIO distillation to prevent collapse)
- No retrieval loss, no candidate scorer, no pose_init_head

### Evaluation Metrics

- init-to-final error reduction per bucket
- Basin of convergence (max init error where refinement still helps)
- Recall@thresholds (0.1m/0.5°, 0.25m/2°)
- Ablation: with/without attention, with/without global context, fine-only vs coarse+fine

## Phase 2: Differentiable Retrieval (Extension)

### Additions on top of Phase 1

1. Enable `A_η` pooling head on coarse feature map → `z_q` and `z_i^db`
2. Sample DB poses uniformly in scene (from Gaussian XYZ extent, ~500–2000 poses)
3. Render DB descriptors from DCFF at sampled poses (cached, refreshed every N steps)
4. Multi-positive InfoNCE: positives = poses within δ of GT (e.g., 1m + 15°)
5. Gradients: L_ret → z_i^db → A_η → F_i^db → DCFF renderer → Gaussian latent
6. Freeze query student (or low-LR fine-tune), train DCFF latent + pooling head
7. Inference: cache DB descriptors → cosine top-K → feed to Phase 1 refiner

## What to Remove / Clean Up

- Remove `fine_loc_head` and `fine_loc` entirely (split-brain, already confirmed)
- Remove `pose_init_head` variants (AbsolutePoseInitHead, AnchorPoseInitHead, FeatureBankPoseInitHead) from training — not needed for CPR
- Remove candidate scorer training (CandidateScoreFusionHead) from CPR phase
- Clean up `feature_retrieval/` scripts — keep only evaluation, remove ensemble/grid-classifier/pose-regressor cruft
- All configs: set `fine_loc_head: false`, remove `pose_init_head` references

## Code Changes (Phase 1)

| Change | File | Effort |
|---|---|---|
| Remove fine_loc head creation/training | `radio_query_student.py`, `train_impl.py` | ~20 lines |
| Fixed-init noise bucket protocol | `train_impl.py` or new `perturb.py` | ~50 lines |
| Coarse feature-metric energy loss | `train_impl.py` | ~30 lines |
| Per-bucket metric logging | `train_impl.py` | ~20 lines |
| Remove pose_init/candidate scorer from joint training path | `train_impl.py` | ~30 lines |
| Clean configs (remove fine_loc, pose_init, scorer) | YAML configs | config only |
| Fixed-init evaluation script | new `pose_refine/evaluate_fixed_init.py` | ~100 lines |

Total new code: ~250 lines. Total removal: ~500+ lines.

## Validation

- Existing tests continue passing (`pytest tests/`)
- Phase 1: fixed-init refinement beats baseline (init-to-final gain > 0 across all buckets)
- Phase 1: coarse bucket 4 shows meaningful improvement (basin expansion)
- Phase 2: retrieval recall@K with trained DB descriptors > NetVLAD baseline
