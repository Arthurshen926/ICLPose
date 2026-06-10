# Perturbation-Aware Geometric Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a scene-specific MATCHA-first geometric adapter that remains useful when render poses are perturbed or come from retrieval/reference initialization.

**Architecture:** Extend the existing MATCHA joint cache and training loop with pair-type metadata, perturbation-aware supervision, no-match/ignore labels, geometric repeatability targets, and mined hard false matches. Keep the current best `RADIO-dual + patch-correlation + pixel-shuffle` model intact and train perturb-aware variants by warm-starting from it.

**Tech Stack:** Python, NumPy, PyTorch, OpenCV/PnP, official 2DGS render path, Cambridge OldHospital data.

---

### Task 1: Pair Types And SE(3) Perturbation Cache

**Files:**
- Modify: `feature_extract/tools/vfm/build_matcha_joint_cache.py`
- Modify: `feature_extract/vfm/render_pose_protocol.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Add cache CLI controls for pair types `A_gt`, `B_trans025`, `C_trans050`, `D_reference`.
- [ ] Add deterministic SE(3) perturbation sampling with translation and yaw/pitch/roll bounds.
- [ ] Store `pair_type`, perturb translation/rotation metadata, and per-sample pair indices in the joint cache.
- [ ] Verify parser defaults preserve current GT-only behavior.

### Task 2: Dustbin / No-Match And Ignore Mask

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/vfm/matcha_joint_cache.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Add optional no-match labels and ignore masks to `MatchaJointTrainingSet`.
- [ ] Serialize/deserialize the new arrays in single NPZ and sharded manifest caches.
- [ ] Train heads with masked CE/BCE where ignored samples do not contribute.
- [ ] Verify legacy caches still load.

### Task 3: Detector Geometric Repeatability Target

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_cache.py`
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Generate repeatability maps from mutually valid coarse correspondences.
- [ ] Add repeatability loss as a detector reliability target.
- [ ] Keep ALIKE-style 65-bin detector labels as auxiliary supervision, not hard proposal filtering.

### Task 4: Hard False Match Mining

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/tools/vfm/train_matcha_joint_model.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Mine high-similarity descriptor candidates that disagree with GT coarse geometry.
- [ ] Add a weighted false-match margin loss.
- [ ] Record false-match mining metrics in training summaries.

### Task 5: q32 Warm-Start Perturb-Aware Training

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_model.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Add warm-start checkpoint loading for compatible joint models.
- [ ] Train perturb-aware q32 model from the current best `q16 dual+patchcorr+pixshuf` checkpoint or q32 if available.
- [ ] Use sharded manifest/lazy cache for q32 to avoid single huge NPZ bottlenecks.

### Task 6: Perturbation And Iterative Update Evaluation

**Files:**
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Create or modify: `feature_extract/tools/vfm/report_perturbation_adapter_eval.py`
- Test: `tests/test_render_rgb_pose_update_protocol.py`

- [ ] Evaluate GT, `+0.25m`, `+0.5m`, `reference_top1`, `reference_top5`, and iterative update.
- [ ] Report match GT@5/16/32, PnP-inlier GT@16/32, median t/r, S@10/S@25, solve rate.
- [ ] Compare against current best GT-render model and old q32 RGB-soft baseline.
