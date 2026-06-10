# Rendered RADIO Keypoint Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a MATCHA-style baseline that samples raw RADIO or selected descriptors at sparse RGB keypoints, matches query-to-render descriptors, backprojects rendered depth, and estimates pose with RANSAC/PnP.

**Architecture:** Add a focused library module for keypoint descriptor sampling, mutual-NN matching, depth backprojection, and PnP conversion. Add a CLI that consumes rendered RGB/feature/depth assets plus query token banks, then writes metrics and visualizations.

**Tech Stack:** NumPy, OpenCV, existing RADIO token NPZ files, existing VFM render/depth artifacts, optional SuperPoint/DISK inputs where available.

---

### Task 1: Core Keypoint Render Matching Utilities

**Files:**
- Create: `feature_extract/vfm/rendered_keypoint_matching.py`
- Test: `tests/test_vfm_rendered_keypoint_matching.py`

- [ ] Write tests for bilinear descriptor sampling, mutual-NN ratio matching, depth backprojection, and conversion to `QueryTo3DMatch`.
- [ ] Implement the tested utilities with no dependency on a specific keypoint detector.
- [ ] Verify with `PYTHONPATH=. pytest tests/test_vfm_rendered_keypoint_matching.py -q`.

### Task 2: Detector and Evaluator CLI

**Files:**
- Create: `feature_extract/tools/vfm/eval_rendered_feature_keypoint_pose.py`

- [ ] Load query RGB/token features and rendered RGB/feature/depth files.
- [ ] Support `detector=orb` as an always-available smoke backend and `detector=superpoint` when existing sidecar dependencies are available.
- [ ] Support descriptor modes `raw_radio` and `selector_npz` where selector features are precomputed or projected in a later pass.
- [ ] Write per-query rows, summary JSON, and match visualizations.

### Task 3: OldHospital q20 Validation

**Files:**
- Output only under `output/vfm/rendered_keypoint_baseline/`.

- [ ] Run raw RADIO q20 with rendered feature/depth assets already present in the workspace.
- [ ] If selector descriptors are available for the same rendered/query resolution, run selector mode; otherwise report selector mode as blocked by missing aligned rendered selector features.
- [ ] Report keypoint counts, match counts, depth-valid matches, PnP solve rate, pose metrics, and visualizations.
