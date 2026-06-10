# MATCHA Q32 Sharded Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scale the current RADIO-dual MATCHA-first model to q32/full experiments without single huge NPZ files, add validation-guided checkpointing, and evaluate GT / 0.25m / 0.5m / reference init protocols.

**Architecture:** Use existing sharded manifest cache output as the canonical large-cache format. Add lazy shard loading for training so only one shard is resident at a time. Add periodic validation on a held-out joint cache and save best checkpoints by validation loss/metrics before running pose evaluation.

**Tech Stack:** Python, PyTorch, NumPy NPZ shards, pytest, existing `feature_extract.vfm.matcha_joint_training` and `feature_extract.tools.vfm` CLIs.

---

### Task 1: Lazy Manifest Training

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/tools/vfm/train_matcha_joint_model.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Write a failing test that monkeypatches shard loading and verifies manifest training does not eagerly load every shard during setup.
- [ ] Implement a lazy manifest training set that samples from shard metadata and loads shard tensors on demand with a small LRU cache.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_matcha_joint_training.py -k lazy`.

### Task 2: Validation-Guided Checkpointing

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/tools/vfm/train_matcha_joint_model.py`
- Test: `tests/test_matcha_joint_training.py`

- [ ] Write a failing test that trains with a validation set and verifies `best_joint_model.pt` plus validation metrics are recorded.
- [ ] Add CLI options for validation cache/manifest, validation interval, and best-checkpoint path.
- [ ] Save best checkpoint by validation loss, including validation summary fields in `summary.json`.
- [ ] Run `PYTHONPATH=. pytest -q tests/test_matcha_joint_training.py -k validation`.

### Task 3: Q32 Sharded Cache And Training

**Files:**
- Use: `feature_extract/tools/vfm/build_matcha_joint_cache.py`
- Use: `feature_extract/tools/vfm/train_matcha_joint_model.py`

- [ ] Build q32 sharded cache with `--output_manifest`.
- [ ] Train q32 RADIO-dual pixel-shuffle patch-correlation model with validation-guided checkpointing.
- [ ] Keep model summaries and remove transient caches only if they are not selected as current evidence.

### Task 4: Four-Protocol Pose Evaluation

**Files:**
- Use: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`

- [ ] Evaluate best q32 checkpoint on `render_pose_mode=gt`.
- [ ] Evaluate `gt_offset` with `render_pose_world_offset=0.25,0,0`.
- [ ] Evaluate `gt_offset` with `render_pose_world_offset=0.5,0,0`.
- [ ] Evaluate `reference_top1` using the available candidate bank if present.
- [ ] Report `median t/r`, `S@10`, `S@25`, `GT@16/32`, `PnP-inlier GT@16/32`, and solve rate.

### Task 5: Verification

**Files:**
- Test: relevant pytest suites

- [ ] Run targeted pytest and py_compile.
- [ ] Check no stray Python experiment process remains.
- [ ] Check disk usage and summarize retained large artifacts.
