# Real RADIO Localization Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the localization stack into explicit selector, coarse matcher, measurement, and orchestration modules for a real-image-first pipeline.

**Architecture:** Add a new `feature_extract.vfm.localization` package with typed schemas and adapters around the existing MATCHA-style code. The first phase is compatibility-first: no legacy file deletion, no render dependency in the new public interfaces, and no behavior change unless covered by tests.

**Tech Stack:** Python, NumPy, PyTorch, pytest, existing `feature_extract.vfm.matcha_*` and `measurement_v1` modules.

---

## File Structure

- Create `feature_extract/vfm/localization/__init__.py`: public exports for the new package.
- Create `feature_extract/vfm/localization/schemas.py`: small dataclasses for feature maps, coarse proposals, measurements, and pipeline outputs.
- Create `feature_extract/vfm/localization/feature_mapper.py`: unified selector interface for raw RADIO feature maps and adapter-backed mappers.
- Create `feature_extract/vfm/localization/coarse_matcher.py`: wrapper around `matcha_coarse_to_fine` coarse matching, without measurement/fine naming.
- Create `feature_extract/vfm/localization/model.py`: container that composes feature mapper, coarse matcher, and optional measurement branch.
- Create `tests/test_localization_refactor.py`: focused tests for the new public API.
- Modify `feature_extract/vfm/__init__.py`: export the new package only after tests prove imports are stable.

## Task 1: Schemas

**Files:**
- Create: `feature_extract/vfm/localization/schemas.py`
- Create: `feature_extract/vfm/localization/__init__.py`
- Test: `tests/test_localization_refactor.py`

- [ ] **Step 1: Write the failing schema test**

```python
from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization import FeatureMapPair, CoarseProposal


def test_feature_map_pair_validates_real_query_and_reference_maps() -> None:
    query = np.zeros((4, 2, 3), dtype=np.float32)
    reference = np.zeros((4, 2, 3), dtype=np.float32)

    pair = FeatureMapPair(query=query, reference=reference, query_image_size=(30, 20), reference_image_size=(30, 20))

    assert pair.channels == 4
    assert pair.query_grid_hw == (2, 3)
    assert pair.reference_grid_hw == (2, 3)


def test_coarse_proposal_keeps_match_metadata_without_render_fields() -> None:
    proposal = CoarseProposal(
        query_index=1,
        reference_index=2,
        query_xy=np.asarray([10.0, 20.0], dtype=np.float32),
        reference_xy=np.asarray([30.0, 40.0], dtype=np.float32),
        score=0.7,
        confidence=0.6,
        rank=0,
    )

    assert proposal.query_index == 1
    assert proposal.reference_index == 2
    assert proposal.confidence == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_feature_map_pair_validates_real_query_and_reference_maps -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'feature_extract.vfm.localization'`.

- [ ] **Step 3: Implement schemas**

Create dataclasses that validate `(C, H, W)` float feature maps, image sizes, and proposal coordinates. Keep field names `reference`, not `render`.

- [ ] **Step 4: Run schema tests**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py -q`

Expected: PASS for schema tests.

## Task 2: Feature Mapper Interface

**Files:**
- Create: `feature_extract/vfm/localization/feature_mapper.py`
- Modify: `feature_extract/vfm/localization/__init__.py`
- Test: `tests/test_localization_refactor.py`

- [ ] **Step 1: Write the failing mapper test**

```python
import torch

from feature_extract.vfm.localization import AdapterFeatureMapper
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineAdapter


def test_adapter_feature_mapper_projects_raw_radio_maps_to_descriptor_maps() -> None:
    model = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    mapper = AdapterFeatureMapper(model, device="cpu")
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    output = mapper.project(raw)

    assert output.coarse_descriptors.shape == (4, 2, 2)
    assert output.measurement_context.shape == (4, 2, 2)
    assert output.offset_logits.shape == (65, 2, 2)
    assert torch.isfinite(torch.from_numpy(output.coarse_descriptors)).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_adapter_feature_mapper_projects_raw_radio_maps_to_descriptor_maps -q`

Expected: FAIL because `AdapterFeatureMapper` is not exported.

- [ ] **Step 3: Implement mapper**

Wrap `project_feature_map_with_matcha_adapter()` and return a `MappedFeatureMap` with `coarse_descriptors`, `measurement_context`, `offset_logits`, and optional `heatmap`.

- [ ] **Step 4: Run mapper tests**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py -q`

Expected: PASS.

## Task 3: Coarse Matcher Wrapper

**Files:**
- Create: `feature_extract/vfm/localization/coarse_matcher.py`
- Modify: `feature_extract/vfm/localization/__init__.py`
- Test: `tests/test_localization_refactor.py`

- [ ] **Step 1: Write the failing coarse matcher test**

```python
from feature_extract.vfm.localization import MatchaCoarseMatcher


def test_matcha_coarse_matcher_returns_reference_proposals_without_fine_measurement() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    reference = np.zeros((2, 1, 2), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    reference[:, 0, 0] = [1.0, 0.0]
    reference[:, 0, 1] = [0.0, 1.0]
    matcher = MatchaCoarseMatcher(logit_scale=12.0, mutual=True)

    proposals = matcher.match(query, reference, query_image_size=(20, 10), reference_image_size=(20, 10))

    assert [(item.query_index, item.reference_index) for item in proposals] == [(0, 0), (1, 1)]
    assert all(item.confidence is not None for item in proposals)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_matcha_coarse_matcher_returns_reference_proposals_without_fine_measurement -q`

Expected: FAIL because `MatchaCoarseMatcher` is not exported.

- [ ] **Step 3: Implement matcher**

Call `matcha_coarse_dual_softmax_matches()` and convert `KeypointFeatureMatch` records into `CoarseProposal` records. Do not call local fine, patch correlation, or measurement.

- [ ] **Step 4: Run coarse matcher tests**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py -q`

Expected: PASS.

## Task 4: Pipeline Container

**Files:**
- Create: `feature_extract/vfm/localization/model.py`
- Modify: `feature_extract/vfm/localization/__init__.py`
- Test: `tests/test_localization_refactor.py`

- [ ] **Step 1: Write the failing pipeline test**

```python
from feature_extract.vfm.localization import SelectorCoarseMeasurementModel


def test_selector_coarse_measurement_model_runs_selector_and_coarse_matcher() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    model = SelectorCoarseMeasurementModel(
        feature_mapper=AdapterFeatureMapper(adapter, device="cpu"),
        coarse_matcher=MatchaCoarseMatcher(logit_scale=12.0, mutual=True),
    )
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    result = model.match_pair(raw, raw.copy(), query_image_size=(20, 20), reference_image_size=(20, 20))

    assert result.mapped_query.coarse_descriptors.shape == (4, 2, 2)
    assert len(result.coarse_proposals) == 4
    assert result.measurements == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py::test_selector_coarse_measurement_model_runs_selector_and_coarse_matcher -q`

Expected: FAIL because `SelectorCoarseMeasurementModel` is not exported.

- [ ] **Step 3: Implement pipeline container**

Compose mapper and matcher. Leave measurement branch optional and return an empty measurement list when absent.

- [ ] **Step 4: Run package tests and existing MATCHA tests**

Run: `PYTHONPATH=. pytest tests/test_localization_refactor.py tests/test_matcha_coarse_to_fine.py tests/test_matcha_joint_training.py -q`

Expected: PASS.

## Self-Review

- Spec coverage: this plan establishes the new real-image-first module boundaries, distinguishes raw RADIO selector mapping from coarse matching, and avoids render fields in the new API.
- Placeholder scan: no placeholders remain; measurement is explicitly optional in this phase.
- Type consistency: `FeatureMapPair`, `MappedFeatureMap`, `CoarseProposal`, `MatchaCoarseMatcher`, `AdapterFeatureMapper`, and `SelectorCoarseMeasurementModel` are introduced before use.
