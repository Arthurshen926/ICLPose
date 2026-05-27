# Raw VFM 3D Landmark Feature Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage raw high-dimensional VFM 3D landmark feature bank from COLMAP tracks using explicit feature aggregation methods.

**Architecture:** Add a focused aggregation module that consumes sampled `TrackObservation` records and outputs the existing explicit track feature bank format with extra metadata in reports. Keep selector, rendered-map scoring, pose refinement, and learned feature fields out of scope.

**Tech Stack:** Python, NumPy, existing `TrackObservation` / `SelectedTrackFeatureBank` dataclasses, JSON/NPZ artifacts, pytest.

---

### Task 1: Aggregation Methods

**Files:**
- Create: `feature_extract/vfm/landmark_feature_aggregation.py`
- Test: `tests/test_vfm_landmark_feature_aggregation.py`

- [ ] Write failing tests for mean, random observation, geometry-weighted, robust trimmed mean, and view-consistent aggregation.
- [ ] Run `python -m pytest tests/test_vfm_landmark_feature_aggregation.py -q` and confirm missing module failure.
- [ ] Implement `aggregate_landmark_features(observations, method, config)`.
- [ ] Verify tests pass.

### Task 2: CLI Artifact Builder

**Files:**
- Create: `feature_extract/tools/vfm/build_raw_vfm_landmark_bank.py`
- Test: `tests/test_vfm_landmark_feature_aggregation.py`

- [ ] Write a failing CLI smoke test using synthetic token maps and COLMAP observation JSONL.
- [ ] Implement CLI that samples raw VFM tokens, aggregates them, writes NPZ bank and JSON summary.
- [ ] Verify CLI smoke test passes.

### Task 3: Diagnostics

**Files:**
- Modify: `feature_extract/vfm/landmark_feature_aggregation.py`
- Test: `tests/test_vfm_landmark_feature_aggregation.py`
- Modify: `docs/vfm/status_and_next_steps.md`

- [ ] Write failing tests for aggregation stability and held-out observation retrieval diagnostics.
- [ ] Implement diagnostics with deterministic sampling.
- [ ] Document the first-stage protocol and expected metrics.
- [ ] Run `python -m pytest tests/test_vfm_landmark_feature_aggregation.py tests/test_vfm_mapability_pipeline.py tests/test_vfm_track_feature_sampling.py -q`.
