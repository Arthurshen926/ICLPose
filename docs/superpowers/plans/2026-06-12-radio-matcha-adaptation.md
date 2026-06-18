# RADIO MATCHA Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RADIO-adapted MATCHA path train all heads from the same clean geometry correspondences.

**Architecture:** Keep RADIO as the feature source. Make `radio_dual_attention` the main MATCHA path with original coarse/fine fusion and full-map loss; keep the residual adapter for legacy row-only usage.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing 3DGS render/cache utilities.

---

### Task 1: Lock Full-Map MATCHA Training Behavior

**Files:**
- Modify: `tests/test_matcha_joint_training.py`
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`

- [x] **Step 1: Add failing tests**

Add tests that assert `radio_dual_attention` total loss does not call `forward_rows()` when full maps and geometry indices are present, and that `matcha_original` is the default attention fusion mode.

- [x] **Step 2: Run targeted tests**

Run: `pytest tests/test_matcha_joint_training.py::test_radio_dual_attention_total_loss_uses_full_map_geometry_path tests/test_matcha_streaming_manifest.py::test_streaming_training_cli_defaults_to_matcha_original_attention -q`

Expected before implementation: FAIL.

- [x] **Step 3: Implement routing**

Change `_total_loss()` so `radio_dual_attention` with full maps uses `_full_map_correspondence_loss()` as the descriptor/offset/fine geometry loss source and does not also call `_coarse_fine_loss()` for positive rows.

- [x] **Step 4: Run tests**

Run the same targeted tests.

Expected after implementation: PASS.

### Task 2: Make MATCHA Original Fusion the Default

**Files:**
- Modify: `feature_extract/vfm/matcha_joint_training.py`
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`

- [x] **Step 1: Update defaults**

Set `attention_fusion_mode="matcha_original"` in `MatchaJointTrainingConfig` and streaming CLI defaults.

- [x] **Step 2: Verify save/load compatibility**

Run: `pytest tests/test_matcha_joint_training.py::test_train_matcha_joint_model_can_use_radio_dual_attention_architecture -q`

Expected: PASS.

### Task 3: Tag Geometry Supervision Source

**Files:**
- Modify: `feature_extract/vfm/matcha_coarse_supervision.py`
- Modify: `feature_extract/vfm/matcha_joint_cache.py`
- Modify: `tests/test_matcha_coarse_supervision.py`

- [x] **Step 1: Add source metadata test**

Assert `build_matcha_coarse_supervision()` emits `source == "geometry_depth_pose"` and rejects matcher-derived source names.

- [x] **Step 2: Implement metadata**

Add a `source` field to `MatchaCoarseSupervision`, default it to `geometry_depth_pose`, and persist it into cache metadata.

- [x] **Step 3: Run tests**

Run: `pytest tests/test_matcha_coarse_supervision.py tests/test_matcha_joint_training.py::test_build_matcha_joint_index_training_set_preserves_robust_targets -q`

Expected: PASS.

### Task 4: Regression Sweep

**Files:**
- No production edits unless failures expose scoped regressions.

- [x] **Step 1: Run focused MATCHA tests**

Run: `pytest tests/test_matcha_joint_training.py tests/test_matcha_streaming_manifest.py tests/test_matcha_coarse_supervision.py -q`

Expected: PASS.

- [x] **Step 2: Report residual risks**

Document any skipped large-data/renderer tests and note that full 3DGS multiview anchor supervision is a follow-up after the pairwise depth/pose path is stable.
