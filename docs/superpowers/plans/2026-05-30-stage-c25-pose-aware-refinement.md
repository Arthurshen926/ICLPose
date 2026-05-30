# Stage C2.5 Pose-Aware Local Descriptor Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local pose-aware descriptor refinement without changing the sparse patch-to-3D pipeline.

**Architecture:** Extend the existing C2-S selector and patch matcher with local geometric supervision and conservative filtering. Keep candidate generation, sparse landmark matching, and PnP handoff fixed except for explicit sweepable thresholds.

**Tech Stack:** Python, NumPy, PyTorch, OpenCV PnP, pytest.

---

### Task 1: Reprojection-Margin Training Signal

**Files:**
- Modify: `feature_extract/vfm/patch_selector_training.py`
- Modify: `feature_extract/tools/vfm/train_stage_c2_safe_selector.py`
- Test: `tests/test_vfm_stage_c2_safe_selector.py`

- [x] **Step 1: Write failing tests**

Add tests proving sample caches round-trip reprojection distances and C2 training records `reprojection_margin_loss_weight`.

- [x] **Step 2: Implement data and loss**

Add optional positive/negative reprojection-distance arrays to `PatchSelectorTrainingSet`, persist them in NPZ caches, and add margin ranking to `_safe_selector_loss`.

- [x] **Step 3: Expose CLI**

Add `--reprojection_margin_loss_weight` and `--reprojection_margin_max`.

- [x] **Step 4: Verify**

Run `PYTHONPATH=. pytest -q tests/test_vfm_stage_c2_safe_selector.py`.

### Task 2: Resume C2.5 From C2-S

**Files:**
- Modify: `feature_extract/vfm/patch_selector_training.py`
- Modify: `feature_extract/tools/vfm/train_stage_c2_safe_selector.py`
- Test: `tests/test_vfm_stage_c2_safe_selector.py`

- [x] **Step 1: Write failing test**

Add a test that passes a previous `SafePatchSelectorTrainingRun` into `train_safe_patch_selector`.

- [x] **Step 2: Implement checkpoint initialization**

Add `init_run` support and `--init_safe_checkpoint`, validating descriptor dimensions and residual hidden size.

- [x] **Step 3: Verify**

Run the C2 safe selector tests.

### Task 3: Pairwise Filtering Without Reranking

**Files:**
- Modify: `feature_extract/vfm/patch_to_3d_matching.py`
- Modify: `feature_extract/tools/vfm/eval_patch_to_3d_vfm_matching.py`
- Test: `tests/test_vfm_patch_to_3d_matching.py`

- [x] **Step 1: Write failing test**

Add a matcher test where pairwise logits filter low-confidence MNN matches while preserving descriptor ranking.

- [x] **Step 2: Implement filters**

Add `pairwise_filter_keep_fraction`, `pairwise_filter_min_logit`, and `pairwise_filter_min_logprob`.

- [x] **Step 3: Verify**

Run patch-to-3D matcher tests.

### Task 4: Hard-Negative Cache Construction

**Files:**
- Modify: `feature_extract/vfm/patch_selector_training.py`
- Modify: `feature_extract/tools/vfm/train_stage_c1_patch_selector.py`
- Test: `tests/test_vfm_stage_c1_patch_selector.py`

- [x] **Step 1: Write failing test**

Add a sample-builder test where negatives are selected in an alternate descriptor space while raw features remain the training payload.

- [x] **Step 2: Implement alternate negative mining**

Allow `build_patch_selector_samples_for_query` to receive alternate query/landmark descriptors and write geometry distances.

- [x] **Step 3: Expose CLI**

Add `--negative_mining_safe_checkpoint` and `--negative_mining_device` to the C1 sample-cache builder path.

### Task 5: PnP Sweep Controls

**Files:**
- Modify: `feature_extract/vfm/query_to_3d_matching.py`
- Modify: `feature_extract/tools/vfm/eval_patch_to_3d_vfm_matching.py`
- Test: `tests/test_vfm_query_to_3d_matching.py`

- [x] **Step 1: Write failing test**

Add a PnP test where `min_inliers` rejects an otherwise solved pose.

- [x] **Step 2: Implement controls**

Add `min_inliers` to `estimate_pose_pnp_ransac` and `--pnp_min_inliers` to the evaluator.

### Task 6: C2.5 Experiments

**Files:**
- Output: `output/vfm/stage_c25_pose_refinement/`
- Modify: `docs/vfm/status_and_next_steps.md`

- [x] **Step 1: Build caches**

Build ShopFacade C2-S hard-negative cache and OldHospital raw-geometry cache.

- [x] **Step 2: Train local refinements**

Train `lambda=0.1` and `lambda=0.3` variants from C2-S checkpoints.

- [x] **Step 3: Evaluate**

Run pairwise-filter diagnostics and RANSAC threshold sweeps.

- [x] **Step 4: Summarize**

Write `output/vfm/stage_c25_pose_refinement/c25_summary.json` and update status docs.
