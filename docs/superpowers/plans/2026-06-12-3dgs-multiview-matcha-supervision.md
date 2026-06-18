# 3DGS Multiview MATCHA Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a geometry-only 3DGS multiview supervision builder for RADIO-adapted MATCHA training.

**Architecture:** Create a focused multiview supervision module that outputs `MatchaCoarseSupervision(source="geometry_3dgs_multiview")`. Keep the existing single-pair depth+pose builder intact and add optional streaming-trainer integration behind explicit CLI flags.

**Tech Stack:** Python, NumPy, PyTorch training cache utilities, pytest, existing COLMAP camera/pose and 3DGS render helpers.

---

### Task 1: Pure Multiview Builder

**Files:**
- Create: `feature_extract/vfm/matcha_multiview_supervision.py`
- Modify: `tests/test_matcha_multiview_supervision.py`

- [x] **Step 1: Write failing builder tests**

Add tests for identity-pose multiview aggregation, support-count filtering, and depth-inconsistent support rejection.

- [x] **Step 2: Run tests to verify red**

Run: `PYTHONPATH=. pytest tests/test_matcha_multiview_supervision.py -q`

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement builder**

Implement dataclasses for geometry views/config and a builder that validates render/query/support depth+alpha geometry without descriptor or matcher inputs.

- [x] **Step 4: Run tests to verify green**

Run: `PYTHONPATH=. pytest tests/test_matcha_multiview_supervision.py -q`

Expected: PASS.

### Task 2: Streaming Integration

**Files:**
- Modify: `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `tests/test_matcha_streaming_manifest.py`

- [x] **Step 1: Write failing CLI/default tests**

Assert the parser exposes multiview supervision knobs and defaults to disabled pairwise supervision.

- [x] **Step 2: Run targeted parser test**

Run: `PYTHONPATH=. pytest tests/test_matcha_streaming_manifest.py::test_streaming_training_cli_exposes_multiview_supervision_controls -q`

Expected: FAIL before parser changes.

- [x] **Step 3: Wire optional support views**

Add deterministic nearest-GT support-pose selection, render support depth/alpha only when enabled, and call the multiview builder.

- [x] **Step 4: Run targeted parser and MATCHA tests**

Run: `PYTHONPATH=. pytest tests/test_matcha_streaming_manifest.py tests/test_matcha_multiview_supervision.py -q`

Expected: PASS.

### Task 3: Regression Sweep

**Files:**
- No production edits unless failures expose scoped regressions.

- [x] **Step 1: Run focused tests**

Run: `PYTHONPATH=. pytest tests/test_matcha*.py -q`

Expected: PASS.

- [x] **Step 2: Report residual risks**

Note that full large-scene render throughput and support-view cache strategy require dataset-scale validation outside unit tests.
