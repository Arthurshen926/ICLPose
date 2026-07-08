from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization import (
    AdapterFeatureMapper,
    CoarseProposal,
    FeatureMapPair,
    MatchaCoarseMatcher,
    SelectorCoarseMeasurementModel,
)
from feature_extract.vfm.matcha_coarse_fine_adapter import MatchaCoarseFineAdapter


def test_feature_map_pair_validates_real_query_and_reference_maps() -> None:
    query = np.zeros((4, 2, 3), dtype=np.float32)
    reference = np.zeros((4, 2, 3), dtype=np.float32)

    pair = FeatureMapPair(
        query=query,
        reference=reference,
        query_image_size=(30, 20),
        reference_image_size=(30, 20),
    )

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


def test_adapter_feature_mapper_projects_raw_radio_maps_to_descriptor_maps() -> None:
    model = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    mapper = AdapterFeatureMapper(model, device="cpu")
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    output = mapper.project(raw)

    assert output.coarse_descriptors.shape == (4, 2, 2)
    assert output.measurement_context.shape == (4, 2, 2)
    assert output.offset_logits is not None
    assert output.offset_logits.shape == (65, 2, 2)
    assert torch.isfinite(torch.from_numpy(output.coarse_descriptors)).all()


def test_matcha_coarse_matcher_returns_reference_proposals_without_fine_measurement() -> None:
    query = np.zeros((2, 1, 2), dtype=np.float32)
    reference = np.zeros((2, 1, 2), dtype=np.float32)
    query[:, 0, 0] = [1.0, 0.0]
    query[:, 0, 1] = [0.0, 1.0]
    reference[:, 0, 0] = [1.0, 0.0]
    reference[:, 0, 1] = [0.0, 1.0]
    matcher = MatchaCoarseMatcher(logit_scale=12.0, mutual=True)

    proposals = matcher.match(
        query,
        reference,
        query_image_size=(20, 10),
        reference_image_size=(20, 10),
    )

    assert [(item.query_index, item.reference_index) for item in proposals] == [(0, 0), (1, 1)]
    assert all(item.confidence is not None for item in proposals)


def test_selector_coarse_measurement_model_runs_selector_and_coarse_matcher() -> None:
    adapter = MatchaCoarseFineAdapter(input_dim=4, output_dim=4, residual_hidden_dim=8, group_size=2)
    model = SelectorCoarseMeasurementModel(
        feature_mapper=AdapterFeatureMapper(adapter, device="cpu"),
        coarse_matcher=MatchaCoarseMatcher(logit_scale=12.0, mutual=True),
    )
    raw = np.eye(4, dtype=np.float32).T.reshape(4, 2, 2)

    result = model.match_pair(
        raw,
        raw.copy(),
        query_image_size=(20, 20),
        reference_image_size=(20, 20),
    )

    assert result.mapped_query.coarse_descriptors.shape == (4, 2, 2)
    assert len(result.coarse_proposals) == 4
    assert result.measurements == []
