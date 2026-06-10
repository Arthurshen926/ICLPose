# Stage R Render-RGB Full-Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a full-resolution render-RGB RADIO matching branch for local pose estimation.

**Architecture:** Render RGB and depth from the 3DGS at the query camera resolution, extract RADIO from the rendered RGB so query and render token grids are resolution-compatible, and evaluate keypoint matching plus PnP through the existing sparse matching backend. Adapter training is render-domain-specific and remains separate from the older sparse-landmark selector line.

**Tech Stack:** Python, NumPy, PyTorch/RADIO, OpenCV/SuperPoint, 3DGS render helpers, pytest.

---

### Task 1: Full-Resolution Protocol And Cache

**Files:**
- Modify: `feature_extract/tools/vfm/build_render_rgb_keypoint_adapter_samples.py`
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Test: `tests/test_render_rgb_keypoint_samples.py`

- [ ] Add `--render_width 0 --render_height 0` semantics: zero means use COLMAP camera width/height.
- [ ] Add RGB/depth/alpha cache helpers so repeated full-resolution sweeps do not rerender geometry.
- [ ] Report render token shape parity in summaries.
- [ ] Verify with pytest and q2 full-resolution smoke.

### Task 2: High-Resolution Adapter Data And Training

**Files:**
- Use: `feature_extract/tools/vfm/build_render_rgb_keypoint_adapter_samples.py`
- Use: `feature_extract/tools/vfm/train_matcha_rendered_keypoint_adapter.py`

- [ ] Build OldHospital q8/q16 full-resolution render-RGB keypoint sample caches.
- [ ] Train render-only 128D adapter without C2.5 initialization.
- [ ] Keep C2.5 initialization only as a diagnostic baseline.

### Task 3: MATCHA-Style Matching Evaluation

**Files:**
- Use: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`

- [ ] Evaluate raw RADIO, render-only adapter with MNN, render-only adapter with dual-softmax.
- [ ] Evaluate confidence-weighted multi-hypothesis pose rescoring.
- [ ] Report solve rate, median pose, success@25cm/10deg, inlier correctness, and per-query failure rows.

### Task 4: Pose-Safe Confidence And Uncertainty

**Files:**
- Modify: `feature_extract/vfm/rendered_pose_scoring.py`
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Test: existing rendered pose scoring tests or a new focused test.

- [ ] Add measurement/depth/render-alpha diagnostics to match and hypothesis rows.
- [ ] Add conservative confidence filtering that preserves spatial coverage.
- [ ] Keep this as a pose verifier layer, not as a replacement descriptor.

### Task 5: Deeper MATCHA Network Branch Decision

**Files:**
- Create only if Task 1-4 show a stable positive signal.

- [ ] Compare high-resolution adapter against raw and C2.5-initialized variants.
- [ ] If still bottlenecked by descriptor ranking, smoke-test a small MATCHA-style network.
- [ ] If bottlenecked by render/depth geometry or repeated structures, defer the network branch.
