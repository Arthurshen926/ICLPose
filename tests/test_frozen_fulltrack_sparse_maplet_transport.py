from __future__ import annotations

import torch

from feature_extract.vfm.localization.frozen_fulltrack_sparse_maplet_transport import (
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES,
    pool_sparse_maplet_query_quadrants,
    sparse_maplet_quadrant_transport_features,
    sparse_maplet_support_usable,
    sparse_maplet_topology_control_features,
)


def test_partial_sparse_maplet_retains_visual_evidence_with_one_missing_quadrant() -> None:
    query = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]],
        dtype=torch.float32,
    )
    support = torch.tensor(
        [[[1.0, 0.0], [0.3, 0.7], [1.0, 1.0], [-1.0, 1.0]]],
        dtype=torch.float32,
    )
    query_valid = torch.ones((1, 4), dtype=torch.bool)
    support_valid = torch.tensor([[True, False, True, True]])
    output = sparse_maplet_quadrant_transport_features(
        query_quadrants=query,
        query_valid=query_valid,
        support_quadrants=support,
        support_valid=support_valid,
    )
    assert output.shape == (1, 27)
    assert torch.isfinite(output).all()
    changed = sparse_maplet_quadrant_transport_features(
        query_quadrants=query,
        query_valid=query_valid,
        support_quadrants=-support,
        support_valid=support_valid,
    )
    assert not torch.allclose(output, changed)


def test_topology_control_has_no_descriptor_dependency_and_tracks_partial_maplet() -> None:
    counts = torch.tensor([[4, 0, 1, 0], [1, 1, 1, 0]], dtype=torch.long)
    coverage = torch.tensor([[1.0, 0.5, 1.0, 0.25], [1.0, 1.0, 1.0, 1.0]])
    control = sparse_maplet_topology_control_features(
        query_original_coverage=coverage,
        support_neighbor_counts=counts,
    )
    assert control.shape == (2, 26)
    assert torch.isfinite(control).all()
    assert sparse_maplet_support_usable(counts).tolist() == [True, False]
    assert len(SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES) == 6 * 27
    assert len(SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES) == 6 * 26


def test_query_quadrants_exclude_the_anchor_cell_from_pooling() -> None:
    patches = torch.zeros((1, 5, 5, 2), dtype=torch.float32)
    patches[0, 2, 2] = torch.tensor([50.0, -50.0])
    values, valid, coverage = pool_sparse_maplet_query_quadrants(
        patches=patches,
        valid=torch.ones((1, 5, 5), dtype=torch.bool),
    )
    assert valid.tolist() == [[True, True, True, True]]
    assert torch.allclose(coverage, torch.ones_like(coverage))
    assert torch.allclose(values, torch.zeros_like(values))
