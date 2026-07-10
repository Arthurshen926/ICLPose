# Real RADIO Landmark Hybrid Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an evaluation path for query-to-aggregated-landmark localization with optional real owner-view RGB measurement refinement.

**Architecture:** A new localization helper module owns landmark projection, owner-observation selection, measurement refinement, and eval artifact writing. A thin CLI wires existing manifests/checkpoints/data paths into that helper. Existing query-to-3D matching and PnP code remains the source of truth for descriptor matching and pose metrics.

**Tech Stack:** Python, NumPy, PyTorch, existing `TokenBankManifest`, `SelectedTrackFeatureBank`, `LandmarkMapIndex`, `JointFeatureMapper`, `RGBPatchMeasurementAdapter`, and PnP utilities.

---

### Task 1: Landmark Hybrid Helpers

**Files:**
- Create: `feature_extract/vfm/localization/landmark_hybrid.py`
- Test: `tests/test_real_radio_landmark_hybrid.py`

- [ ] **Step 1: Write failing tests**

Create tests for:

```python
def test_owner_observation_index_scales_xy_to_target_image_size():
    ...

def test_refine_matches_with_measurement_updates_query_xy_and_preserves_xyz():
    ...

def test_project_landmark_index_with_identity_mapper_preserves_metadata():
    ...
```

- [ ] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=. pytest tests/test_real_radio_landmark_hybrid.py -q`
Expected: import failure because `feature_extract.vfm.localization.landmark_hybrid` does not exist.

- [ ] **Step 3: Implement helper module**

Add:

```python
LandmarkOwnerObservation
LandmarkOwnerObservationIndex
project_landmark_index_features(...)
refine_landmark_matches_with_measurement(...)
write_jsonl(...)
run_real_radio_landmark_hybrid_eval(...)
```

Use `dataclasses.replace` for `QueryTo3DMatch` updates and group measurement proposals by owner reference image.

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=. pytest tests/test_real_radio_landmark_hybrid.py -q`
Expected: all tests pass.

### Task 2: CLI

**Files:**
- Create: `feature_extract/tools/vfm/eval_real_radio_landmark_hybrid.py`
- Test: `tests/test_real_radio_landmark_hybrid_cli.py`

- [ ] **Step 1: Write failing CLI parser test**

Create a parser smoke test that verifies defaults for `--measurement_mode none|owner_rgb`, matching args, and output paths.

- [ ] **Step 2: Run CLI test to verify failure**

Run: `PYTHONPATH=. pytest tests/test_real_radio_landmark_hybrid_cli.py -q`
Expected: import failure because the CLI does not exist.

- [ ] **Step 3: Implement CLI**

Load the query manifest, landmark bank, track observations, cameras, ground truth poses, joint checkpoint, and optional measurement branch. Call `run_real_radio_landmark_hybrid_eval`.

- [ ] **Step 4: Run CLI tests**

Run: `PYTHONPATH=. pytest tests/test_real_radio_landmark_hybrid_cli.py tests/test_real_radio_landmark_hybrid.py -q`
Expected: all tests pass.

### Task 3: Smoke Experiment

**Files:**
- No code changes expected.
- Output: `output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1/landmark_hybrid_*`

- [ ] **Step 1: Build or locate landmark bank**

If no full OldHospital `SelectedTrackFeatureBank` exists, run `build_raw_vfm_landmark_bank.py` from the balanced track JSONL and train RADIO token manifest.

- [ ] **Step 2: Run no-measurement smoke**

Evaluate 20 queries with joint-projected landmark features and no RGB measurement.

- [ ] **Step 3: Run owner-RGB measurement smoke**

Evaluate the same 20 queries with `--measurement_mode owner_rgb`.

- [ ] **Step 4: Compare with current support-cell result**

Compare association-free landmark match counts, PnP success, median pose error, and recall thresholds against `eval_pose_val_20pairs/summary.json`.

### Task 4: Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=. pytest tests/test_real_radio_landmark_hybrid.py tests/test_real_radio_landmark_hybrid_cli.py tests/test_real_radio_pose_eval.py tests/test_real_radio_pose_eval_cli.py -q
```

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- feature_extract/vfm/localization/landmark_hybrid.py feature_extract/tools/vfm/eval_real_radio_landmark_hybrid.py tests/test_real_radio_landmark_hybrid.py tests/test_real_radio_landmark_hybrid_cli.py`

- [ ] **Step 3: Report experiment metrics and method implications**

Summarize whether landmark hybrid improves match count, PnP stability, and pose recall, and explain whether 3D reference should be SfM landmarks, aggregated landmark features, or 2DGS geometry for the next training target.
