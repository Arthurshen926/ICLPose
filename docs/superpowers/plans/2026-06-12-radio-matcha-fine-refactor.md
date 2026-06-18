# RADIO MATCHA Fine Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace collapsed center-bin fine supervision with geometry sub-cell seeds, make patch correlation the primary fine objective, and validate with three-tier localization metrics.

**Architecture:** The 3DGS multiview supervision builder remains the geometry gate. Streaming training gains a deterministic geometry sub-cell seed generator and entropy diagnostics. Evaluation uses local correlation/search presets as pose-backend refinements, not as supervision.

**Tech Stack:** Python, PyTorch, NumPy, pytest, RADIO extractor, existing 3DGS renderer and MATCHA-style training/eval scripts.

---

### Task 1: Fine Seed Tests

**Files:**
- Modify: `tests/test_matcha_streaming_manifest.py`
- Modify: `tests/test_matcha_multiview_supervision.py`

- [ ] Add a test that geometry sub-cell seed generation returns one point per render cell, stays inside image bounds, and produces non-center offset labels.
- [ ] Add a test that multiview supervision built from sub-cell seeds has non-collapsed render label entropy while keeping `source="geometry_3dgs_multiview"`.
- [ ] Run the targeted tests and verify they fail before implementation.

### Task 2: Geometry Sub-Cell Seed Implementation

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `feature_extract/vfm/matcha_multiview_supervision.py` only if the builder needs metadata, not for supervision semantics.

- [ ] Add deterministic render sub-cell seed generation driven by record seed and render grid/image geometry.
- [ ] Extend `--fine_supervision_source` with `render_subcell`.
- [ ] Preserve `cell_center` and `render_alike`.
- [ ] Add training-row diagnostics: query/render offset-label entropy, render center-bin fraction, query center-bin fraction.
- [ ] Run the targeted tests and verify they pass.

### Task 3: Patch Correlation as Primary Fine Objective

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `feature_extract/vfm/matcha_joint_training.py` if metrics need clearer names.
- Modify tests if parser/default behavior is covered.

- [ ] Change the new streaming defaults so `query_pair_fine_loss_weight=0.0` and `pair_fine_loss_weight=0.0`.
- [ ] Keep `patch_correlation_loss_weight` active by default.
- [ ] Ensure old explicit CLI values still work for ablations.
- [ ] Add or update a parser/default test.

### Task 4: RADIO-MATCHA Local Search Evaluation Preset

**Files:**
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Test: existing parser tests or a new focused eval parser test.

- [ ] Add an explicit preset that disables pair-MLP fine dependence and enables local correlation/render-side search settings.
- [ ] Ensure match table output cannot accidentally overwrite rows in the recommended command.
- [ ] Add a focused test for preset expansion.

### Task 5: Verification Runs

**Commands:**
- `PYTHONPATH=. pytest tests/test_matcha*.py -q`
- Streaming smoke train with `--fine_supervision_source render_subcell`, multiview support enabled, pair-MLP fine losses off.
- Three-tier eval from the smoke or full checkpoint:
  - `render_pose_mode=gt`
  - `render_pose_mode=gt_offset`
  - `render_pose_mode=reference_top5`

- [ ] Record whether label entropy is healthy.
- [ ] Record patch correlation accuracy.
- [ ] Record pose metrics for all three tiers.
- [ ] Keep only settings with real pose-metric evidence.
