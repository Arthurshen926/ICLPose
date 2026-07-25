from __future__ import annotations

import pytest
import torch

from feature_extract.vfm.localization.candidate_edge_group_linear_probe import (
    CandidateEdgeGroupLinearBatch,
    CandidateEdgeGroupLinearProbe,
    fit_candidate_edge_group_linear_probe,
    group_softmin_pose_gaps,
)


def _batch(edge_features: torch.Tensor) -> CandidateEdgeGroupLinearBatch:
    return CandidateEdgeGroupLinearBatch(
        edge_features=edge_features,
        # Point 0 has two coherent-wrong candidate identities; point 1 has
        # two.  Both belong to the same coherent wrong-pose group.
        edge_to_point=torch.tensor([0, 0, 1, 1]),
        point_to_pose=torch.tensor([0, 0]),
    )


def test_group_softmin_tracks_the_strongest_wrong_candidate() -> None:
    batch = _batch(torch.ones((4, 1)))
    gaps = group_softmin_pose_gaps(
        edge_margins=torch.tensor([0.8, -0.2, 0.7, 0.9]),
        batch=batch,
        temperature=1e-4,
    )
    # Point 0 must keep -0.2 rather than average its two candidates.  Point
    # 1 keeps 0.7, then the coherent pose averages the two point margins.
    assert gaps == pytest.approx(torch.tensor([0.25]), abs=2e-4)


def test_group_softmin_is_invariant_to_duplicate_equivalent_candidates() -> None:
    original = CandidateEdgeGroupLinearBatch(
        edge_features=torch.ones((4, 1)),
        edge_to_point=torch.tensor([0, 0, 1, 1]),
        point_to_pose=torch.tensor([0, 0]),
    )
    duplicated = CandidateEdgeGroupLinearBatch(
        edge_features=torch.ones((8, 1)),
        edge_to_point=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        point_to_pose=torch.tensor([0, 0]),
    )
    first = group_softmin_pose_gaps(
        edge_margins=torch.tensor([0.3, 0.8, -0.1, 0.4]),
        batch=original,
        temperature=0.2,
    )
    second = group_softmin_pose_gaps(
        edge_margins=torch.tensor([0.3, 0.3, 0.8, 0.8, -0.1, -0.1, 0.4, 0.4]),
        batch=duplicated,
        temperature=0.2,
    )
    assert second == pytest.approx(first, abs=1e-6)


def test_group_aware_probe_learns_a_positive_coherent_margin_on_toy_data() -> None:
    batch = _batch(
        torch.tensor(
            [
                [1.0, 0.1],
                [0.7, -0.1],
                [1.1, 0.0],
                [0.8, 0.2],
            ],
            dtype=torch.float32,
        )
    )
    probe, stats = fit_candidate_edge_group_linear_probe(
        batch=batch,
        device="cpu",
        epochs=48,
        learning_rate=0.05,
        weight_decay=1e-3,
        softmin_temperature=0.1,
        seed=7,
    )
    assert isinstance(probe, CandidateEdgeGroupLinearProbe)
    assert probe.feature_scale.shape == (2,)
    assert probe.weights.shape == (2,)
    gaps = group_softmin_pose_gaps(
        edge_margins=probe.score(batch.edge_features),
        batch=batch,
        temperature=probe.softmin_temperature,
    )
    assert float(gaps.item()) > 0.0
    assert float(stats["final_group_softmin_mean_gap"]) > 0.0

