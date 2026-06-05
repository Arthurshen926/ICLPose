# VFM-2DGS Anchor Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a no-localization VFM-2DGS mapping pipeline that builds surface-region anchors from VFM patch tokens and 2DGS surfel/disk supports.

**Architecture:** Add a standalone `feature_extract.vfm.vfm_2dgs_mapping` module. It treats loaded 2DGS Gaussians as surface elements, computes soft token-to-surface responsibilities from token footprints and projected disks, builds token-surface observations, associates observations by weighted surface support overlap, and saves a VFM-2DGS anchor map.

**Tech Stack:** Python, NumPy, SciPy KD-tree, existing token manifests / pose parsers / Gaussian source loader, pytest.

---

### Task 1: Core Data Structures And Soft Responsibility

**Files:**
- Create: `feature_extract/vfm/vfm_2dgs_mapping.py`
- Test: `tests/test_vfm_2dgs_mapping.py`

- [ ] Add tests for `build_surface_elements_from_2dgs_source`, `compute_token_surface_observations`, and `Vfm2DgsAnchorMap.save_npz/load_npz`.
- [ ] Implement surface elements from `GaussianVFMSource` with centers, normals, two tangent scales, opacity, adjacency, and area.
- [ ] Implement token soft footprint assignment over projected surface elements. Use patch token centers, footprint radius, opacity, depth epsilon, normal/view-angle confidence, component concentration, and normalized responsibilities.
- [ ] Implement anchor fusion by weighted surface IoU and feature/normal/center consistency.

### Task 2: CLI Builder

**Files:**
- Create: `feature_extract/tools/vfm/build_vfm_2dgs_anchor_map.py`

- [ ] Load 2DGS PLY, reference token manifest, poses, and camera intrinsics.
- [ ] Build surface elements and token observations for selected reference views.
- [ ] Fuse observations into anchors and save `anchor_map.npz` plus `summary.json`.

### Task 3: OldHospital Smoke

**Files/Outputs:**
- Output: `output/vfm/vfm_2dgs_anchor_maps/oldhospital/smoke_v32/`
- Report: `docs/vfm/vfm_2dgs_anchor_mapping_smoke.md`

- [ ] Build a v32 OldHospital raw RADIO VFM-2DGS anchor map without localization.
- [ ] Report observation count, anchor count, observation quality, support size, feature variance, and coverage diagnostics.
- [ ] Do not train selector and do not run pose localization in this stage.
