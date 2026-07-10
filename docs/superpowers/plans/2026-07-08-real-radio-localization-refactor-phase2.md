# Real RADIO Localization Refactor Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the real-image localization mainline beyond compatibility wrappers by exposing joint feature mapping, top-k coarse proposal pools, and RGB measurement through the new `feature_extract.vfm.localization` API.

**Architecture:** Keep legacy modules as implementation backends, but hide old `render` and `fine` naming from new public interfaces. The new pipeline remains real query/reference oriented and calls measurement only through a typed adapter.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing MATCHA joint models, existing RGB patch measurement branch.

---

## File Structure

- Modify `feature_extract/vfm/localization/feature_mapper.py`: add `JointFeatureMapper`.
- Modify `feature_extract/vfm/localization/coarse_matcher.py`: add `MatchaTopKCoarseMatcher`.
- Create `feature_extract/vfm/localization/measurement.py`: add `RGBPatchMeasurementAdapter` and proposal patch helpers.
- Modify `feature_extract/vfm/localization/model.py`: pass mapped context to measurement-capable adapters when supported.
- Modify `feature_extract/vfm/localization/__init__.py`: export the new public API.
- Modify `tests/test_localization_refactor.py`: add TDD coverage for each new API.

## Task 1: Joint Feature Mapper

- [x] Write a failing test proving `JointFeatureMapper` returns coarse descriptors separately from measurement context for `RadioDualAttentionFusionJointModel(attention_fusion_mode="matcha_original")`.
- [x] Implement `JointFeatureMapper.project()` using `forward_fuse_feature()` for matcha-original radio-dual models and `forward_feature_map()` otherwise.
- [x] Run `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_joint_feature_mapper_exposes_coarse_and_measurement_context_separately -q`.

## Task 2: Top-K Coarse Matcher

- [x] Write a failing test proving `MatchaTopKCoarseMatcher` returns multiple proposals per query with rank metadata and reference naming.
- [x] Implement it by wrapping `matcha_coarse_topk_matches()`.
- [x] Run `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_matcha_topk_coarse_matcher_returns_ranked_reference_proposals -q`.

## Task 3: RGB Measurement Adapter

- [x] Write a failing test with a deterministic fake branch proving measurement consumes query/reference RGB and coarse proposals.
- [x] Implement `RGBPatchMeasurementAdapter` as a real-image adapter with no render fields in its public API.
- [x] Run `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_rgb_patch_measurement_adapter_returns_measurements_for_reference_proposals -q`.

## Task 4: Pipeline Integration

- [x] Write a failing test proving `SelectorCoarseMeasurementModel` passes mapped query/reference context to measurement adapters that accept it.
- [x] Update `MeasurementBranch` protocol and `match_pair()` to call measurement with `mapped_query` and `mapped_reference` when the adapter supports those keyword arguments.
- [x] Run `PYTHONPATH=. pytest tests/test_localization_refactor.py tests/test_matcha_coarse_to_fine.py tests/test_matcha_joint_training.py -q`.

## Self-Review

- Scope stays in `feature_extract.vfm.localization`; no legacy deletion in this phase.
- The new API names are `query` and `reference`, not `render`.
- The selector means raw RADIO feature mapper, not heatmap/keypoint proposal selector.
