# Real RADIO Closed-Loop And Joint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-image selector + coarse + measurement closed-loop evaluation and a real-only joint training wrapper that cannot silently fall back to row-only or render-cache training.

**Architecture:** Put reusable pair loading, pipeline execution, row writing, and metric aggregation in `feature_extract.vfm.localization.pipeline`. Add one CLI for real-pair closed-loop proposal generation/evaluation and one CLI wrapping the existing `matcha_joint_training` backend with real-only validation and measurement loss enabled. Keep legacy render-cache scripts untouched.

**Tech Stack:** Python, NumPy, PyTorch, PIL, pytest, existing `JointFeatureMapper`, `MatchaTopKCoarseMatcher`, `RGBPatchMeasurementAdapter`, `matcha_joint_training`.

---

### Task 1: Closed-Loop Pipeline Tests

**Files:**
- Create: `tests/test_real_radio_closed_loop_pipeline.py`
- Create: `feature_extract/vfm/localization/pipeline.py`
- Create: `feature_extract/tools/vfm/eval_real_radio_closed_loop.py`

- [x] **Step 1: Write failing tests for feature/RGB loading, real-pair execution, GT metrics, and CLI parsing.**

- [x] **Step 2: Run tests and verify they fail because the module and CLI do not exist.**

- [x] **Step 3: Implement `pipeline.py` with CSV/NPY/NPZ/PIL IO, `SelectorCoarseMeasurementModel` execution, proposal rows, and GT metric aggregation.**

- [x] **Step 4: Implement `eval_real_radio_closed_loop.py` that loads a joint checkpoint and optional measurement checkpoint, builds `JointFeatureMapper + MatchaTopKCoarseMatcher + RGBPatchMeasurementAdapter`, and writes CSV/JSONL/summary outputs.**

- [x] **Step 5: Run pipeline tests and verify they pass.**

### Task 2: Real-Only Joint Training Wrapper Tests

**Files:**
- Create: `tests/test_train_real_radio_joint_localization.py`
- Create: `feature_extract/tools/vfm/train_real_radio_joint_localization.py`

- [x] **Step 1: Write failing tests proving the CLI has no `sample_cache` or render-cache arguments, defaults to `radio_dual_attention + matcha_original`, and enables measurement patch loss.**

- [x] **Step 2: Write a validation test proving row-only/sample-like joint sets are rejected because they cannot train full selector/coarse/measurement.**

- [x] **Step 3: Write a monkeypatched main test proving the wrapper builds `MatchaJointTrainingConfig` with measurement loss and calls the real joint training backend.**

- [x] **Step 4: Run tests and verify they fail for missing script/functions.**

### Task 3: Real-Only Joint Training Wrapper Implementation

**Files:**
- Create: `feature_extract/tools/vfm/train_real_radio_joint_localization.py`

- [x] **Step 1: Implement `_validate_joint_localization_training_set()` requiring full query/reference feature maps, cell indices, RGB images, and measurement-compatible RGB shapes.**

- [x] **Step 2: Implement `train_real_radio_joint_localization()` loading joint cache/manifest, optional validation cache/manifest, warm start, training, and saving both adapter and joint checkpoints.**

- [x] **Step 3: Implement CLI `parse_args()` and `main()` with real-only names and defaults.**

- [x] **Step 4: Run wrapper tests and verify they pass.**

### Task 4: Verification

**Files:**
- Modify tests from Tasks 1 and 2.

- [x] **Step 1: Run focused tests for new scripts.**

- [x] **Step 2: Run combined localization, MATCHA, measurement, and script tests.**

- [x] **Step 3: Run `compileall` for new/modified modules.**

- [x] **Step 4: Clean generated `__pycache__` and inspect `git status --short`.**
