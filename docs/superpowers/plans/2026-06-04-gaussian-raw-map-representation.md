# Gaussian Raw Map Representation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the no-selector Gaussian raw VFM map path by exposing 3DGS geometry attributes and adding diagnostics that reveal whether sampling and aggregation are localization-appropriate.

**Architecture:** Keep the existing Stage H2 raw-map pipeline. Extend `GaussianVFMSource` to preserve anisotropic scale, rotation quaternion, and a smallest-axis normal from 3DGS PLY files, then surface these geometry stats in raw-map summaries before changing matching or selector code.

**Tech Stack:** Python, NumPy, plyfile, pytest, existing VFM Stage H2 tools.

---

### Task 1: Preserve Gaussian Geometry Attributes

**Files:**
- Modify: `feature_extract/vfm/gaussian_vfm_field.py`
- Test: `tests/test_vfm_gaussian_vfm_field.py`

- [ ] Add a failing test that writes a tiny 3DGS-style PLY with anisotropic `scale_0..2` and `rot_0..3`, loads it with `load_gaussian_vfm_source_from_ply`, and asserts:
  - `source.scale_xyz` has shape `(N, 3)` and equals `exp(scale_i)`;
  - `source.rotation` has shape `(N, 4)` and is normalized;
  - `source.normal` has shape `(N, 3)` and follows the smallest-scale local axis.
- [ ] Run:
  `pytest tests/test_vfm_gaussian_vfm_field.py::test_load_gaussian_vfm_source_preserves_anisotropic_geometry -q`
  Expected: fail because `GaussianVFMSource` does not expose `scale_xyz`, `rotation`, or `normal`.
- [ ] Extend `GaussianVFMSource` with optional `scale_xyz`, `rotation`, and `normal` fields while keeping existing constructors valid.
- [ ] Update `load_gaussian_vfm_source_from_ply` to parse scales and rotations, compute the smallest-axis normal, and fall back cleanly when fields are absent.
- [ ] Re-run the single test and the existing Gaussian field tests.

### Task 2: Add Raw-Map Geometry Diagnostics

**Files:**
- Modify: `feature_extract/tools/vfm/build_stage_h2_raw_gaussian_anchor_map.py`
- Test: `tests/test_vfm_gaussian_raw_landmarks.py`

- [ ] Add a failing test for a helper that summarizes sampled Gaussian geometry:
  - opacity percentiles;
  - mean-scale percentiles;
  - anisotropy ratio percentiles when `scale_xyz` exists;
  - normal availability flag.
- [ ] Run:
  `pytest tests/test_vfm_gaussian_raw_landmarks.py::test_gaussian_source_geometry_summary_reports_anisotropy_and_normals -q`
  Expected: fail because the helper is not implemented.
- [ ] Implement `_source_geometry_stats(source, sampled_indices)` in the Stage H2 raw-map builder.
- [ ] Add `source_geometry_stats` and `anchor_geometry_stats` to the emitted summary JSON.
- [ ] Re-run the new test plus `tests/test_vfm_gaussian_raw_landmarks.py`.

### Task 3: Validate On Current OldHospital Map

**Files:**
- No production code changes expected.
- Output: `output/vfm/stage_h2_raw_gaussian_anchors/oldhospital/geometry_diagnostic_raw1280/`
- Report: `docs/vfm/stage_h2_gaussian_raw_map_representation.md`

- [ ] Run the raw-map builder on OldHospital q/diagnostic settings with raw RADIO features and no selector.
- [ ] Inspect summary JSON for anisotropy, normal availability, opacity, support, and variance stats.
- [ ] Run q32 no-selector patch-to-3D smoke on the generated map.
- [ ] Generate camera-view overlay diagnostics for selected anchors.
- [ ] Write a short report comparing this corrected diagnostic map with existing SfM raw and previous Gaussian raw results.

### Self-Review

- This plan does not touch selector training or learned descriptors.
- It adds source representation and diagnostics first, so later contribution-aware sampling can use geometry correctly.
- It preserves backward compatibility for tests that construct `GaussianVFMSource` with only `xyz`, `opacity`, `scale`, and `gaussian_indices`.
