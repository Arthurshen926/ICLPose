# Camera Pose Refinement Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip fine_loc_head, pose_init_head, and candidate scorer from the training path; add fixed-init noise bucket training/evaluation; establish clean CPR baseline.

**Architecture:** Single shared RadioQueryStudent → coarse/fine features. Coarse handles large-basin pose correction (feature-metric energy), fine handles high-precision correspondence (dense corr + WLS). No retrieval, no candidate selection, no pose regression head — pure refinement.

**Tech Stack:** PyTorch, 3DGS rasterizer, DCFF hash-grid renderer, CUDA

---

### Task 1: Remove fine_loc_head from RadioQueryStudent

**Files:**
- Modify: `feature_extract/students/radio_query_student.py:1058-1791`

- [ ] **Step 1: Remove fine_loc constructor parameters**

In `RadioQueryStudent.__init__`, remove the `fine_loc_head`, `fine_loc_mode`, `fine_loc_init`, `fine_loc_zero_init`, `fine_loc_detach_base`, `fine_loc_highres_source`, `fine_loc_highres_init`, `fine_loc_highres_zero_init`, `fine_loc_highres_detach` parameters (lines 1097-1105). Replace all 9 parameters with a single silent ignore:

```python
# Remove these lines (1097-1105):
# fine_loc_head=False,
# fine_loc_mode="residual",
# fine_loc_init=1.0,
# fine_loc_zero_init=True,
# fine_loc_detach_base=False,
# fine_loc_highres_source=None,
# fine_loc_highres_init=1.0,
# fine_loc_highres_zero_init=True,
# fine_loc_highres_detach=True,
```

No replacement parameters needed — the function will accept them via `**kwargs` silently for backwards compat during transition.

- [ ] **Step 2: Remove fine_loc attribute storage**

Remove lines 1200-1214 that store fine_loc attributes:

```python
# Remove these lines:
# self.fine_loc_highres_source = None
# ...
# self.fine_loc_highres_detach = bool(fine_loc_highres_detach)
```

- [ ] **Step 3: Remove fine_loc validation**

Remove lines 1206-1207 and 1327-1330 (fine_loc_mode and fine_loc_highres_source validation).

- [ ] **Step 4: Remove fine_loc module creation**

Replace lines 1389-1418 (the entire `if self.use_fine_loc_head:` block) with:

```python
self.fine_loc_head = None
self.fine_loc_scale = None
self.fine_loc_highres_fuse = None
self.fine_loc_highres_scale = None
```

- [ ] **Step 5: Remove fine_loc zero-init**

Remove lines 1596-1603 (fine_loc_zero_init and fine_loc_highres_zero_init init blocks).

- [ ] **Step 6: Remove fine_loc forward pass**

In `forward()`, replace lines 1722-1751 (fine_loc computation) with:

```python
fine_loc = None
```

- [ ] **Step 7: Verify model still builds**

```bash
python -c "
import torch
from feature_extract.students.radio_query_student import RadioQueryStudent
m = RadioQueryStudent(feature_dim=64, fine_feature_dim=96, coarse_feature_dim=32, fine_loc_head=False)
x = torch.randn(1, 3, 1088, 1920)
out = m(x)
assert 'fine' in out and 'coarse' in out
assert out.get('fine_loc') is None
print('OK: model builds and forward works without fine_loc')
"
```

- [ ] **Step 8: Commit**

```bash
git add feature_extract/students/radio_query_student.py
git commit -m "refactor: remove fine_loc_head from RadioQueryStudent"
```

---

### Task 2: Remove pose_init_head from RadioQueryStudent

**Files:**
- Modify: `feature_extract/students/radio_query_student.py:1058-1791`

- [ ] **Step 1: Remove pose_init constructor parameters**

Remove lines 1156-1167 (pose_init_head through pose_init_token_source parameters). These will be accepted silently via `**kwargs`.

- [ ] **Step 2: Remove pose_init attribute storage and validation**

Remove lines 1234-1257 (attribute storage + mode/feature_source/token_source validation).

- [ ] **Step 3: Remove pose_init module creation**

Replace lines 1473-1507 (pose_init head creation) with:

```python
self.pose_init_head = None
```

- [ ] **Step 4: Remove pose_init forward pass**

Replace lines 1773-1780 (pose_init output) with:

```python
# pose_init removed — no longer generated
```

- [ ] **Step 5: Commit**

```bash
git add feature_extract/students/radio_query_student.py
git commit -m "refactor: remove pose_init_head from RadioQueryStudent"
```

---

### Task 3: Strip candidate scorer construction from RadioQueryStudent

**Files:**
- Modify: `feature_extract/students/radio_query_student.py:1141-1147,1448-1472,1583-1589`

- [ ] **Step 1: Remove constructor parameters**

Remove lines 1141-1155 (all `candidate_score_fusion_*` and `candidate_score_map_fusion_*` parameters). Keep the `CandidateScoreFusionHead` and `CandidateScoreMapFusionHead` classes in the file (they don't hurt and tests may reference them).

- [ ] **Step 2: Remove attribute storage for candidate scorer**

Remove lines 1230-1232 (`self.candidate_score_map_fusion_head_enabled`, `self.candidate_score_fusion_head_enabled`).

- [ ] **Step 3: Remove module creation**

Replace lines 1448-1472 (candidate_score_fusion_head creation) with:

```python
self.candidate_score_fusion_head = None
```

- [ ] **Step 4: Remove zero-init for candidate scorer**

Remove lines 1583-1591 (candidate_score_fusion zero-init block).

- [ ] **Step 5: Verify model builds**

```bash
python -c "
import torch
from feature_extract.students.radio_query_student import RadioQueryStudent
m = RadioQueryStudent(feature_dim=64, fine_feature_dim=96, coarse_feature_dim=32)
assert m.pose_init_head is None
assert m.fine_loc_head is None
assert m.candidate_score_fusion_head is None
print('OK: stripped model builds')
"
```

- [ ] **Step 6: Commit**

```bash
git add feature_extract/students/radio_query_student.py
git commit -m "refactor: strip candidate scorer construction from RadioQueryStudent"
```

---

### Task 4: Strip fine_loc/pose_init/candidate_scorer from train_impl.py model config defaults and construction

**Files:**
- Modify: `feature_extract/train_impl.py`

- [ ] **Step 1: Remove model config defaults**

Remove or comment out the following default dict entries:

```python
# Lines 166-174: fine_loc defaults — remove all 9 entries
# Lines 210-224: candidate_score_fusion defaults — remove all 14 entries
# Lines 225-240: pose_init defaults — remove all 16 entries
```

These are in the `DEFAULT_MODEL_CFG` dict.

- [ ] **Step 2: Remove model constructor argument pass-through**

In the model construction call (around lines 4044-4052 for fine_loc, 4090-4114 for candidate scorer, 4116-4127 for pose_init), remove the named arguments. The parameters no longer exist on RadioQueryStudent so they'll cause errors if passed. Instead of passing them, remove the lines entirely.

The model construction should end up like:

```python
model = RadioQueryStudent(
    feature_dim=feature_dim,
    fine_feature_dim=fine_feature_dim,
    coarse_feature_dim=coarse_feature_dim,
    base_channels=...,
    stage_dims=...,
    output_hw=...,
    coarse_output_hw=...,
    input_hw=...,
    dropout=...,
    l2_normalize=...,
    predict_magnitude=...,
    fine_init_norm=...,
    coarse_init_norm=...,
    magnitude_min=...,
    retrieval_dim=...,
    retrieval_hidden_dim=...,
    retrieval_dropout=...,
    retrieval_l2_normalize=...,
    fine_low_level_skip=...,
    fine_low_level_init=...,
    fine_highres_skip=...,
    fine_highres_source=...,
    fine_highres_init=...,
    fine_highres_zero_init=...,
    global_context_enabled=...,
    global_context_zero_init=...,
    window_attention_layers=...,
    window_attention_heads=...,
    window_attention_size=...,
    window_attention_mlp_ratio=...,
    window_attention_dropout=...,
    window_attention_shift=...,
    window_attention_zero_init=...,
    teacher_fine_condition=...,
    teacher_fine_init=...,
    teacher_fine_zero_init=...,
    teacher_fine_detach=...,
    scene_coord_head=...,
    scene_coord_zero_init=...,
    scene_coord_detach_base=...,
    scene_coord_use_pixel_grid=...,
    scene_coord_global_context=...,
    local_matcher_enabled=...,
    local_matcher_radius=...,
    local_matcher_hidden_dim=...,
    local_matcher_zero_init=...,
    local_matcher_residual_scale=...,
    local_matcher_context_mode=...,
    local_flow_head_enabled=...,
    local_flow_head_radius=...,
    local_flow_head_hidden_dim=...,
    local_flow_head_zero_init=...,
    local_flow_head_max_flow=...,
    local_flow_head_base_flow_mode=...,
    local_flow_head_base_temperature=...,
    local_flow_head_context_mode=...,
    local_corr_projector_enabled=...,
    local_corr_projector_hidden_dim=...,
    local_corr_projector_output_dim=...,
    local_corr_projector_zero_init=...,
    local_corr_projector_l2_normalize=...,
    local_corr_projector_domain_adapter=...,
    local_corr_query_projector_zero_init=...,
    local_corr_render_projector_zero_init=...,
    query_channel_gate_enabled=...,
    query_channel_gate_hidden_dim=...,
    query_channel_gate_zero_init=...,
    apply_query_channel_gate=...,
)
```

- [ ] **Step 3: Remove pose_init_anchor metric computation**

Remove lines 1052-1078 (pose_init_anchor_* metric accumulation).

Remove lines 8849-8863 (pose_init_anchor_* metric logging).

- [ ] **Step 4: Remove pose_init anchor sampling and refresh logic**

Remove lines 9367-9414 (anchor sampling setup).

Remove lines 9524-9543 (feature bank refresh at epoch start).

Remove lines 9641-9655 (training log pose_init metrics).

- [ ] **Step 5: Remove pose_init validation**

Remove lines 723, 743, 796, 860 (pose_init validation in config parsing).

- [ ] **Step 6: Remove candidate_score_fusion_head from compute_map_supervision signatures and calls**

In `compute_map_supervision()` (line 6510), remove `candidate_score_fusion_head=None` parameter.
Remove the `candidate_score_fusion_head` pass-through in all call sites (lines 8806, 9597).

- [ ] **Step 7: Disable candidate-related loss branches in compute_map_supervision**

Set all candidate-related weight defaults in `DEFAULT_MAP_SUPERVISION_CFG` to 0.0 (lines 328-401):
- `coarse_pose_energy_weight: 0.0` (already default)
- `candidate_render_score_weight: 0.0` (already default)
- `candidate_score_fusion_weight: 0.0` (already default)
- `candidate_refined_pose_weight: 0.0` (already default)
- `candidate_two_stage_enabled: False` (already default)

No code removal needed here since the branches are gated by `weight > 0`.

- [ ] **Step 8: Remove pose_init from evaluate_impl model construction**

In `feature_extract/evaluate_impl.py` (lines 304-311 for fine_loc, 337-347 for pose_init), and `export_impl.py` (lines 299-306, 340-350), remove the config-to-constructor argument pass-through for removed parameters.

- [ ] **Step 9: Verify training import still works**

```bash
python -c "
from feature_extract.train_impl import DEFAULT_MODEL_CFG, DEFAULT_MAP_SUPERVISION_CFG
# Check no fine_loc/pose_init keys remain
for k in list(DEFAULT_MODEL_CFG.keys()):
    if 'fine_loc' in k or 'pose_init' in k or 'candidate_score_fusion' in k:
        print(f'ERROR: {k} still in DEFAULT_MODEL_CFG')
        break
else:
    print('OK: no leftover config keys')
"
```

- [ ] **Step 10: Commit**

```bash
git add feature_extract/train_impl.py feature_extract/evaluate_impl.py feature_extract/export_impl.py
git commit -m "refactor: strip fine_loc/pose_init/candidate_scorer from training config and construction"
```

---

### Task 5: Add noise bucket protocol to training

**Files:**
- Modify: `feature_extract/train_impl.py`
- Create: (no new files — add to existing dataset/noise infrastructure)

- [ ] **Step 1: Add noise bucket config to dataset defaults**

In `data/radio_loc_dataset.py`, add support for a `noise_buckets` parameter. If provided, `add_pose_noise()` is called with a randomly sampled bucket's parameters instead of the fixed `noise_rot_deg` / `noise_trans_m`.

Add to `RadioLocDataset.__init__` (after line 405):

```python
noise_buckets = None  # list of (trans_m, rot_deg) tuples
```

In `__getitem__` (around line 601), replace the fixed noise call:

```python
if self.noise_buckets is not None and self.split == 'train':
    bucket_idx = torch.randint(0, len(self.noise_buckets), (1,)).item()
    bucket_trans_m, bucket_rot_deg = self.noise_buckets[bucket_idx]
    pose_init = add_pose_noise(pose_gt.numpy(), bucket_rot_deg, bucket_trans_m)
    pose_init = torch.from_numpy(pose_init).float()
else:
    pose_init = add_pose_noise(pose_gt.numpy(), self.noise_rot_deg, self.noise_trans_m)
    pose_init = torch.from_numpy(pose_init).float()
```

Also store the bucket index for logging:

```python
result['noise_bucket'] = bucket_idx if self.noise_buckets is not None else -1
```

- [ ] **Step 2: Add config entry for noise buckets**

In `DEFAULT_CFG` in `feature_extract/train_impl.py`:

```python
"noise_buckets": None,  # list of [trans_m, rot_deg] pairs for bucketed noise training
```

Pass through to dataset construction (around line 9400):

```python
noise_buckets = cfg["dataset"].get("noise_buckets")
if noise_buckets is not None:
    noise_buckets = [tuple(b) for b in noise_buckets]
```

- [ ] **Step 3: Add bucket info to batch**

In `collate_fn` or the dataset, ensure `noise_bucket` is batched:

```python
batch['noise_bucket'] = torch.stack([item['noise_bucket'] for item in items])
```

- [ ] **Step 4: Verify dataset works with buckets**

```bash
python -c "
import torch
from data.radio_loc_dataset import RadioLocDataset, add_pose_noise
import numpy as np

pose = np.eye(4, dtype=np.float32)
# Test multi-bucket noise
buckets = [(0.1, 2.0), (0.5, 10.0), (2.0, 30.0)]
for trans, rot in buckets:
    noisy = add_pose_noise(pose, rot, trans)
    t_err = np.linalg.norm(noisy[:3, 3] - pose[:3, 3])
    print(f'bucket ({trans}m, {rot}deg): trans_err={t_err*1000:.1f}mm')
print('OK: noise buckets work')
"
```

- [ ] **Step 5: Commit**

```bash
git add data/radio_loc_dataset.py feature_extract/train_impl.py
git commit -m "feat: add noise bucket protocol for fixed-init training"
```

---

### Task 6: Add per-bucket metric logging

**Files:**
- Modify: `feature_extract/train_impl.py`

- [ ] **Step 1: Add per-bucket tracking to training loop**

In the main training loop (around line 9600), after computing `outputs["delta_pose_error_trans_mm"]` and similar metrics, bucketize them:

```python
if "noise_bucket" in batch:
    bucket_ids = batch["noise_bucket"]
    # Track per-bucket: init error, final error, gain
    for b_idx in bucket_ids.unique():
        b_mask = bucket_ids == b_idx
        b_trans_m, b_rot_deg = cfg["dataset"]["noise_buckets"][int(b_idx)]
        bucket_label = f"bucket_{b_trans_m:.2f}m_{b_rot_deg:.0f}deg"
        # Accumulate per-bucket metrics
        ...
```

- [ ] **Step 2: Log per-bucket metrics**

At the logging step (every `log_every` steps), compute and log per-bucket averages:

```python
for bucket_label, metrics in bucket_accumulators.items():
    writer.add_scalar(f"bucket/{bucket_label}/init_trans_mm", ...)
    writer.add_scalar(f"bucket/{bucket_label}/final_trans_mm", ...)
    writer.add_scalar(f"bucket/{bucket_label}/gain_mm", ...)
```

- [ ] **Step 3: Commit**

```bash
git add feature_extract/train_impl.py
git commit -m "feat: add per-bucket metric logging for fixed-init training"
```

---

### Task 7: Create fixed-init evaluation script

**Files:**
- Create: `pose_refine/evaluate_fixed_init.py`

- [ ] **Step 1: Create evaluation script**

```python
#!/usr/bin/env python3
"""Fixed-init evaluation: measure refinement per noise bucket.

Evaluates Camera Pose Refinement: given an initial pose (GT + controlled noise),
how much does the refiner improve the pose? Reports per-bucket and aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_dataset import RadioLocDataset, add_pose_noise, collate_fn
from pose_refine.evaluate_impl import (
    build_dcff,
    evaluate as evaluate_fixed,
    load_model,
)
from pose_refine.utils.geometry_solver import feature_metric_solve


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-init CPR evaluation")
    parser.add_argument("--config", required=True, help="Mainline YAML config")
    parser.add_argument("--checkpoint", required=True, help="Pose refiner checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    # Noise buckets: list of (trans_m, rot_deg) pairs
    parser.add_argument("--buckets", type=str,
                        default="0.1,2 0.25,5 0.5,10 1.0,20 2.0,30",
                        help="Space-separated trans_m,rot_deg pairs")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of noise seeds per bucket per image")
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--gru-iters", type=int, default=4)
    parser.add_argument("--direct-refine-iters", type=int, default=0)
    parser.add_argument("--render-h", type=int, default=68)
    parser.add_argument("--render-w", type=int, default=120)
    parser.add_argument("--solver", default="wls",
                        choices=["wls", "pnp", "hybrid"])
    parser.add_argument("--use-coarse", action="store_true", default=True)
    return parser.parse_args()


def parse_buckets(s: str) -> List[Tuple[float, float]]:
    buckets = []
    for pair in s.strip().split():
        parts = pair.split(",")
        if len(parts) == 2:
            buckets.append((float(parts[0]), float(parts[1])))
    return buckets


def evaluate_on_bucket(model, gaussians, dcff_renderer, feat_sharp,
                        dataset, bucket_trans_m, bucket_rot_deg,
                        n_seeds, device, args):
    """Evaluate refinement on a single noise bucket."""
    all_init_rot, all_init_trans = [], []
    all_final_rot, all_final_trans = [], []

    for idx in tqdm(range(len(dataset)), desc=f"bucket {bucket_trans_m:.2f}m/{bucket_rot_deg:.0f}deg"):
        sample = dataset[idx]
        query_fine = sample["query_fine"].unsqueeze(0).to(device)
        query_coarse = sample.get("query_coarse")
        if query_coarse is not None:
            query_coarse = query_coarse.unsqueeze(0).to(device)
        pose_gt = sample["pose_gt"].to(device)

        render_intr = model._scale_intrinsics(args.render_h, args.render_w)
        K = torch.tensor([
            [render_intr["fx"], 0, render_intr["cx"]],
            [0, render_intr["fy"], render_intr["cy"]],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        for seed in range(n_seeds):
            pose_init_np = add_pose_noise(
                pose_gt.cpu().numpy(), bucket_rot_deg, bucket_trans_m
            )
            pose_init = torch.from_numpy(pose_init_np).float().to(device)

            # Compute init error
            init_rot, init_trans = compute_pose_error(pose_init, pose_gt)
            all_init_rot.append(init_rot)
            all_init_trans.append(init_trans)

            # Run refinement
            from pose_refine.evaluate_impl import render_batch
            pose_cur = pose_init.unsqueeze(0)
            for outer_i in range(args.outer_iters):
                ref_fine, depth = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, args.render_h, args.render_w
                )
                pred = model(
                    query_fine, ref_fine, depth.unsqueeze(1), render_intr,
                    query_coarse=query_coarse,
                )
                from pose_refine.runtime import apply_pose_delta
                pose_cur = apply_pose_delta(pose_cur, pred["delta_xi"])

            # Optional direct feature-metric refinement
            for _ in range(args.direct_refine_iters):
                ref_fine, _ = render_batch(
                    gaussians, dcff_renderer, feat_sharp,
                    pose_cur, K, args.render_h, args.render_w
                )
                from feature_field.utils.geometry_solver import feature_metric_solve
                delta_xi, residual = feature_metric_solve(
                    query_fine, ref_fine, pose_cur, K, render_intr
                )
                from pose_refine.runtime import apply_pose_delta
                pose_cur = apply_pose_delta(pose_cur, delta_xi.unsqueeze(0))

            final_rot, final_trans = compute_pose_error(
                pose_cur.squeeze(0), pose_gt
            )
            all_final_rot.append(final_rot)
            all_final_trans.append(final_trans)

    return {
        "init_rot_deg_mean": float(np.mean(all_init_rot)),
        "init_trans_mm_mean": float(np.mean(all_init_trans)),
        "init_trans_mm_median": float(np.median(all_init_trans)),
        "final_rot_deg_mean": float(np.mean(all_final_rot)),
        "final_trans_mm_mean": float(np.mean(all_final_trans)),
        "final_trans_mm_median": float(np.median(all_final_trans)),
        "trans_gain_mm": float(np.mean(all_init_trans) - np.mean(all_final_trans)),
        "n_samples": len(all_init_trans),
    }
```

- [ ] **Step 2: Add pose error utility**

```python
def compute_pose_error(pose_pred, pose_gt):
    """Return (rot_err_deg, trans_err_mm) between two w2c poses."""
    R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
    R_gt, t_gt = pose_gt[:3, :3], pose_gt[:3, 3]

    # Rotation error (geodesic)
    R_rel = R_pred.T @ R_gt
    trace = torch.clamp(R_rel.trace(), -1.0, 3.0)
    rot_err = torch.acos((trace - 1.0) / 2.0).item() * 180.0 / math.pi

    # Translation error
    C_pred = -(R_pred.T @ t_pred)
    C_gt = -(R_gt.T @ t_gt)
    trans_err = (C_pred - C_gt).norm().item() * 1000.0  # mm

    return rot_err, trans_err
```

- [ ] **Step 3: Add main function**

```python
def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build DCFF
    gaussians, dcff_renderer, feat_sharp = build_dcff(config, device)

    # Load model
    model = load_model(config, args.checkpoint, device)
    model.eval()

    buckets = parse_buckets(args.buckets)

    # Build dataset with no noise (we apply noise per bucket)
    dataset = RadioLocDataset(
        feature_dir=config["dataset"]["feature_dir"],
        colmap_dir=config["dataset"]["colmap_dir"],
        split=args.split,
        split_file=config["dataset"].get(f"{args.split}_split"),
        noise_rot_deg=0.0,
        noise_trans_m=0.0,
        limit=args.limit,
    )

    results = {}
    for trans_m, rot_deg in buckets:
        bucket_key = f"{trans_m:.2f}m_{rot_deg:.0f}deg"
        results[bucket_key] = evaluate_on_bucket(
            model, gaussians, dcff_renderer, feat_sharp,
            dataset, trans_m, rot_deg, args.n_seeds, device, args
        )

    # Print summary
    print("\n=== Fixed-Init CPR Results ===")
    print(f"{'Bucket':<20s} {'Init(mm)':>10s} {'Final(mm)':>10s} {'Gain(mm)':>10s} {'Gain%':>8s}")
    print("-" * 58)
    for bucket_key, r in results.items():
        gain_pct = r["trans_gain_mm"] / max(r["init_trans_mm_mean"], 1e-6) * 100
        print(f"{bucket_key:<20s} {r['init_trans_mm_median']:>10.1f} "
              f"{r['final_trans_mm_median']:>10.1f} {r['trans_gain_mm']:>10.1f} "
              f"{gain_pct:>7.1f}%")

    # Save JSON
    output_path = args.output_json or os.path.join(
        os.path.dirname(args.checkpoint), "fixed_init_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Test script parses correctly**

```bash
python pose_refine/evaluate_fixed_init.py --help
```

- [ ] **Step 5: Commit**

```bash
git add pose_refine/evaluate_fixed_init.py
git commit -m "feat: add fixed-init CPR evaluation script with per-bucket reporting"
```

---

### Task 8: Clean up feature_retrieval/ — keep only evaluation entry point

**Files:**
- Modify: (none — move/remove files)

- [ ] **Step 1: Identify files to archive**

All ensemble, grid-classifier, pose-regressor variant files in `feature_retrieval/` that are not used by the main evaluation pipeline:

```
feature_retrieval/advanced_ensemble.py
feature_retrieval/analyze_grid_cells.py
feature_retrieval/build_best_ensemble_init.py
feature_retrieval/build_index.py
feature_retrieval/build_init_poses.py
feature_retrieval/build_learned_init_poses.py
feature_retrieval/cross_arch_ensemble.py
feature_retrieval/cross_arch_ensemble_v2.py
feature_retrieval/cross_ensemble_eval.py
feature_retrieval/decoupled_init_search.py
feature_retrieval/ensemble_eval.py
feature_retrieval/error_analysis.py
feature_retrieval/eval_checkpoint.py
feature_retrieval/eval_netvlad_oldhospital.py
feature_retrieval/fine_weight_search.py
feature_retrieval/full_eval_v2.py
feature_retrieval/hybrid_v6.py
feature_retrieval/knn_regressor.py
feature_retrieval/optimized_ensemble.py
feature_retrieval/patch_center_frustum_v1.py
feature_retrieval/patch_grid_classifier_v1.py
feature_retrieval/patch_memory_translation_v1.py
feature_retrieval/patch_regressor_v7.py
feature_retrieval/pose_regressor.py
feature_retrieval/pose_regressor_v2.py
feature_retrieval/pose_regressor_v3.py
feature_retrieval/pose_regressor_v4.py
feature_retrieval/retrieval_v5.py
feature_retrieval/run_init_eval.py
```

- [ ] **Step 2: Move to archive directory**

```bash
mkdir -p feature_retrieval/_archive
cd feature_retrieval
for f in advanced_ensemble.py analyze_grid_cells.py build_best_ensemble_init.py \
         build_index.py build_init_poses.py build_learned_init_poses.py \
         cross_arch_ensemble.py cross_arch_ensemble_v2.py cross_ensemble_eval.py \
         decoupled_init_search.py ensemble_eval.py error_analysis.py \
         eval_checkpoint.py eval_netvlad_oldhospital.py fine_weight_search.py \
         full_eval_v2.py hybrid_v6.py knn_regressor.py optimized_ensemble.py \
         patch_center_frustum_v1.py patch_grid_classifier_v1.py \
         patch_memory_translation_v1.py patch_regressor_v7.py \
         pose_regressor.py pose_regressor_v2.py pose_regressor_v3.py \
         pose_regressor_v4.py retrieval_v5.py run_init_eval.py; do
    [ -f "$f" ] && mv "$f" _archive/
done
```

- [ ] **Step 3: Verify evaluate_impl.py still imports**

```bash
python -c "from feature_retrieval.evaluate_impl import evaluate_real_init; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add feature_retrieval/
git commit -m "refactor: archive unused feature_retrieval ensemble/regressor scripts"
```

---

### Task 9: Create clean CPR baseline config

**Files:**
- Create: `feature_extract/configs/cpr_baseline_v1.yaml`

- [ ] **Step 1: Write clean config**

```yaml
# Camera Pose Refinement baseline — Phase 1
# No fine_loc, no pose_init, no candidate scorer, no retrieval.
# Coarse: feature-metric energy for large-basin correction.
# Fine: dense correlation + WLS for high-precision refinement.

base_config: null  # standalone, no parent
exp_name: cpr_baseline_v1
output_dir: /root/ICLPose/result/feature_extract

model:
  feature_dim: 64
  fine_feature_dim: 96
  coarse_feature_dim: 32
  base_channels: 32
  stage_dims: [32, 64, 96, 128]
  dropout: 0.0
  l2_normalize: true

  global_context_enabled: true
  global_context_zero_init: true
  window_attention_layers: 2
  window_attention_heads: 8
  window_attention_size: 16
  window_attention_mlp_ratio: 2.0
  window_attention_shift: true
  window_attention_zero_init: true

  local_matcher_enabled: true
  local_matcher_radius: 4
  local_matcher_hidden_dim: 64
  local_matcher_zero_init: true
  local_matcher_residual_scale: 1.0
  local_matcher_context_mode: basic

  local_corr_projector_enabled: true
  local_corr_projector_hidden_dim: 128
  local_corr_projector_output_dim: 64
  local_corr_projector_zero_init: false
  local_corr_projector_l2_normalize: true

  warmstart_strict: false
  # No fine_loc_head, no pose_init_head, no candidate scorer

dataset:
  input_hw: [1088, 1920]
  feature_hw: [68, 120]
  coarse_feature_hw: [17, 30]
  noise_buckets:
    - [0.1, 2.0]
    - [0.25, 5.0]
    - [0.5, 10.0]
    - [1.0, 20.0]
    - [2.0, 30.0]

training:
  seed: 973
  lr: 0.00005
  batch_size: 2
  max_steps: 600
  log_every: 10
  freeze_model_except_prefixes:
    - "global_context."
    - "window_attention."
    - "local_matcher."
    - "stage3."
    - "stage4."
    - "fine_fuse."
    - "fine_low_fuse."
    - "fine_low_scale"
    - "fine_highres_fuse."
    - "fine_highres_scale"
    - "fine_head."
    - "coarse_head."
    - "coarse_refine."
    - "local_corr_projector."
  model_lr_scales:
    global_context.: 0.25
    window_attention.: 0.25
    local_matcher.: 4.0
    local_corr_projector.: 8.0
    stage3.: 0.03
    stage4.: 0.08
    fine_head.: 0.35
    coarse_head.: 0.35

loss:
  fine_l1_weight: 0.3
  fine_cos_weight: 0.3
  coarse_l1_weight: 0.3
  coarse_cos_weight: 0.3

map_supervision:
  config_path: /root/ICLPose/configs/dcff_cambridge_oldhospital.yaml

  # Direct query-map alignment
  query_fine_key: fine
  query_local_fine_key: fine
  query_fine_weight: 0.10
  query_coarse_weight: 0.08
  query_fine_infonce_weight: 0.010

  # Coarse feature-metric energy (large-basin correction)
  coarse_feature_metric_energy_weight: 0.05
  coarse_feature_metric_energy_radius: 8

  # Fine correlation + WLS
  query_corr_radius: 8
  query_corr_ce_weight: 0.5
  query_corr_subpixel_weight: 0.8
  query_corr_peak_margin_weight: 5.0
  query_corr_ce_temperature: 0.050
  query_corr_wls_pose_weight: 0.10
  perturb_render_negatives: true
  perturb_trans_cm_choices: [2.0, 5.0, 10.0, 20.0]
  perturb_rot_deg: 0.5

  # Feature-metric pose loss (fine alignment)
  feature_metric_pose_weight: 0.05
  feature_metric_pose_damping: 0.05
  feature_metric_pose_update_scale: 0.25
  feature_metric_pose_rot_weight: 0.1
  feature_metric_pose_trans_weight: 30.0
  feature_metric_pose_normalize: true

  # Teacher anchor (prevent collapse, low weight)
  rendered_teacher_fine_weight: 0.01
  rendered_teacher_fine_weight_end: 0.002
  rendered_teacher_fine_weight_anneal_epochs: 4

  # Disabled
  candidate_render_score_weight: 0.0
  candidate_score_fusion_weight: 0.0
  candidate_refined_pose_weight: 0.0
  candidate_two_stage_enabled: false
  coarse_pose_energy_weight: 0.0

export:
  fine_key: fine
```

- [ ] **Step 2: Commit**

```bash
git add feature_extract/configs/cpr_baseline_v1.yaml
git commit -m "feat: add clean CPR baseline config v1"
```

---

### Task 10: Update tests for stripped model

**Files:**
- Modify: `tests/test_adaptive_joint_query_map.py`
- Modify: `tests/test_end_to_end_mainline.py`
- Modify: `tests/test_feature_visualization_pca.py`

- [ ] **Step 1: Update test_adaptive_joint_query_map.py**

Remove or skip tests that depend on removed functionality:

Tests to remove (fine_loc):
- `test_radio_query_student_fine_loc_head_starts_as_noop_and_receives_gradients` (~line 963)
- `test_radio_query_student_fine_loc_head_*` (all variants, lines 994-1052)
- Any test that sets `fine_loc_head: True` in model config
- Any test that sets `query_fine_key: "fine_loc"` in map_supervision

Tests to remove (pose_init):
- `test_pose_init_head_can_use_retrieval_descriptor_token` (~line 434)
- `test_feature_bank_pose_init_head_can_use_retrieval_descriptor_space` (~line 502)
- `test_pose_init_anchor_metrics_*` (lines 704-741)

Tests to keep but update (remove removed args):
- Any test that constructs `RadioQueryStudent(...)` with fine_loc/pose_init/candidate_scorer args — remove those args

- [ ] **Step 2: Update test_end_to_end_mainline.py**

Remove:
- `test_radio_query_student_pose_init_head_outputs_multihypothesis_pose` (~line 24)
- All anchor/feature_bank pose_init tests
- All fine_loc tests
- `test_refresh_pose_init_feature_bank_from_map_renderer_updates_feature_bank_buffers` (~line 473)
- Any `pose_init_anchor_*` metric assertions

- [ ] **Step 3: Update test_feature_visualization_pca.py**

Remove or update any test that passes `fine_loc_head: true` in config (around line 81-103).

- [ ] **Step 4: Run remaining tests**

```bash
pytest tests/ -x -q --timeout=120 2>&1 | tail -40
```

Expected: all non-removed tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: remove tests for stripped fine_loc/pose_init/candidate_scorer features"
```

---

### Task 11: Final integration validation

- [ ] **Step 1: Clean import check — all modules load**

```bash
python -c "
from feature_extract import RadioQueryStudent
from feature_extract.train_impl import compute_map_supervision
from feature_field.dcff import DeferredCascadedRenderer
from pose_refine.evaluate_impl import build_dcff, render_batch
print('All modules import OK')
"
```

- [ ] **Step 2: Model forward pass with all expected output keys**

```bash
python -c "
import torch
from feature_extract.students.radio_query_student import RadioQueryStudent
m = RadioQueryStudent(feature_dim=64, fine_feature_dim=96, coarse_feature_dim=32)
x = torch.randn(1, 3, 1088, 1920)
out = m(x)
expected = {'fine', 'coarse', 'global_pose_token'}
actual = set(out.keys())
assert expected.issubset(actual), f'Missing keys: {expected - actual}'
forbidden = {'fine_loc', 'pose_init'}
found = forbidden & actual
assert not found, f'Should not have: {found}'
print(f'Forward outputs: {sorted(actual)}')
print('OK')
"
```

- [ ] **Step 3: Full test suite (remaining tests)**

```bash
pytest tests/ -x -q --timeout=120 --ignore=tests/test_train_candidate_selector.py 2>&1 | tail -20
```

- [ ] **Step 4: Commit if all passes**

```bash
git add -A
git commit -m "chore: final integration validation for CPR Phase 1 cleanup"
```
