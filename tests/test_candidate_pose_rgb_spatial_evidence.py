from __future__ import annotations

import pytest
import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_evidence import (
    summarize_candidate_pose_rgb_spatial_evidence_layers,
    summarize_registered_observation_candidate_oracle,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialScore,
)


def _score(*, candidate: torch.Tensor, edge: torch.Tensor) -> CandidatePoseRGBSpatialScore:
    point = torch.logsumexp(candidate + torch.log(torch.tensor([[[0.25, 0.25]]])), dim=2)
    # The test uses a fixed explicit-null mixture: 0.25*exp(c0)+0.25*exp(c1)+0.5.
    point = torch.logsumexp(
        torch.cat(
            [
                candidate + torch.log(torch.tensor([[[0.25, 0.25]]])),
                torch.log(torch.tensor([[[0.5]]])).expand(candidate.shape[0], -1, -1),
            ],
            dim=2,
        ),
        dim=2,
    )
    return CandidatePoseRGBSpatialScore(
        pose_log_likelihood_ratios=point.mean(dim=1),
        point_log_likelihood_ratios=point,
        candidate_log_likelihood_ratios=candidate,
        edge_log_likelihood_ratios=edge,
        edge_usable=torch.ones_like(edge, dtype=torch.bool),
    )


def test_layer_audit_separates_direct_edge_signal_from_null_attenuation() -> None:
    correct = _score(
        candidate=torch.tensor([[[2.0, 0.0]]]),
        edge=torch.tensor([[[[2.0], [0.0]]]]),
    )
    wrong = _score(
        candidate=torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]]]),
        edge=torch.tensor([[[[0.0], [0.0]]], [[[1.0], [0.0]]]]),
    )
    metrics = summarize_candidate_pose_rgb_spatial_evidence_layers(
        correct=correct,
        wrong=wrong,
        correct_projection_offsets_xy=torch.zeros((1, 2, 2)),
        wrong_projection_offsets_xy=torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[2.0, 0.0], [2.0, 0.0]]],
            ]
        ),
        candidate_probabilities=torch.tensor([[0.25, 0.25]]),
        null_probabilities=torch.tensor([0.5]),
    )

    assert metrics["edge_count"] == 4.0
    assert metrics["edge_mean_correct_minus_wrong"] == pytest.approx(0.75)
    assert metrics["fixed_mixture_pose_mean_correct_minus_wrong"] > 0.0
    assert metrics["fixed_mixture_hardest_coherent_wrong_correct_minus_wrong"] > 0.0
    assert (
        metrics["counterfactual_prior_no_null_pose_mean_correct_minus_wrong"]
        > metrics["fixed_mixture_pose_mean_correct_minus_wrong"]
    )
    assert metrics["edge_displacement_1_to_2_count"] == 2.0
    assert metrics["edge_displacement_2_to_4_count"] == 2.0


def test_registered_observation_oracle_is_posthoc_and_requires_one_candidate_per_point() -> None:
    correct = _score(
        candidate=torch.tensor([[[2.0, 0.0]]]),
        edge=torch.tensor([[[[2.0], [0.0]]]]),
    )
    wrong = _score(
        candidate=torch.tensor([[[0.0, 0.0]]]),
        edge=torch.tensor([[[[0.0], [0.0]]]]),
    )
    metrics = summarize_registered_observation_candidate_oracle(
        correct=correct,
        wrong=wrong,
        observed_candidate_mask=torch.tensor([[True, False]]),
    )
    assert metrics["registered_observation_oracle_point_count"] == 1.0
    assert metrics["registered_observation_oracle_pose_mean_correct_minus_wrong"] == pytest.approx(2.0)
    assert (
        metrics["registered_observation_oracle_hardest_coherent_wrong_correct_minus_wrong"]
        == pytest.approx(2.0)
    )

    with pytest.raises(ValueError, match="oracle inputs"):
        summarize_registered_observation_candidate_oracle(
            correct=correct,
            wrong=wrong,
            observed_candidate_mask=torch.tensor([[True, True]]),
        )
