from __future__ import annotations

import pytest
import torch

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
    CandidateHighresRGBScalePrediction,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_pose_evidence_selector import (
    aggregate_static_point_log_likelihood_ratios,
    blend_selector_with_uniform_mass,
    combine_selector_confidences,
    phase_identity_confidence,
    rgb_spatial_mode_quality,
    runtime_visual_edge_availability,
    score_reweighted_selector_weights,
    spatial_diverse_topk_weights,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 0, 0]),
        query_xy=torch.tensor([[8.0, 8.0], [88.0, 8.0], [48.0, 40.0]]),
        support_image_indices=torch.tensor([[[1], [2]], [[1], [2]], [[1], [2]]]),
        support_xy=torch.full((3, 2, 1, 2), 12.0),
        support_view_valid=torch.ones((3, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((3, 2, 1)),
        candidate_probabilities=torch.full((3, 2), 0.45),
        null_probabilities=torch.full((3,), 0.10),
    )


def _phase() -> CandidateMultiscalePhaseIdentityPrediction:
    values = torch.tensor(
        [
            [[3.0], [-2.0]],
            [[0.0], [0.0]],
            [[1.5], [0.5]],
        ]
    )
    usable = torch.ones_like(values, dtype=torch.bool)
    return CandidateMultiscalePhaseIdentityPrediction(
        source_edge_log_likelihood_ratios={
            name: values.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        },
        source_edge_usable={
            name: usable.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
        },
        edge_log_likelihood_ratios=values,
        edge_usable=usable,
        source_weights={name: 1.0 for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES},
    )


def _rgb() -> CandidateHighresRGBMultiscalePrediction:
    offsets = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    logits = torch.tensor(
        [
            [[[5.0, 0.0, 0.0, 0.0]], [[5.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]],
            [[[1.5, 1.0, 0.5, 0.0]], [[1.5, 1.0, 0.5, 0.0]]],
        ],
        dtype=torch.float32,
    )
    dustbin = torch.tensor([[[0.0], [0.0]], [[5.0], [5.0]], [[0.0], [0.0]]])
    joint = normalized_spatial_log_probabilities_with_dustbin(
        logits.reshape(-1, 4), dustbin.reshape(-1)
    ).reshape(3, 2, 1, 5)
    scale = CandidateHighresRGBScalePrediction(
        spatial_logits=logits,
        non_dustbin_logits=dustbin,
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=torch.zeros((3, 2, 1)),
        edge_usable=torch.ones((3, 2, 1), dtype=torch.bool),
    )
    return CandidateHighresRGBMultiscalePrediction(sources={"fine": scale, "broad": scale})


def test_phase_identity_confidence_is_static_and_requires_visual_concentration() -> None:
    confidence = phase_identity_confidence(runtime=_runtime(), prediction=_phase())
    assert confidence.shape == (3,)
    assert confidence[0] > confidence[2] > confidence[1]
    unavailable = phase_identity_confidence(
        runtime=_runtime(),
        prediction=_phase(),
        edge_availability_override=torch.zeros((3, 2, 1), dtype=torch.bool),
    )
    torch.testing.assert_close(unavailable, torch.zeros_like(unavailable))


def test_rgb_spatial_mode_quality_prefers_peaked_non_dustbin_modes() -> None:
    quality = rgb_spatial_mode_quality(runtime=_runtime(), prediction=_rgb())
    assert quality.shape == (3,)
    assert quality[0] > quality[2] > quality[1]
    combined = combine_selector_confidences(
        phase_confidence=phase_identity_confidence(runtime=_runtime(), prediction=_phase()),
        rgb_quality=quality,
    )
    assert combined[0] > combined[1]


def test_runtime_visual_availability_has_no_counterfactual_input() -> None:
    availability = runtime_visual_edge_availability(
        runtime=_runtime(), phase_prediction=_phase(), rgb_prediction=_rgb()
    )
    assert availability.shape == (3, 2, 1)
    assert bool(availability.all())


def test_spatial_diverse_topk_keeps_one_token_per_occupied_cell() -> None:
    xy = torch.tensor([[5.0, 5.0], [15.0, 5.0], [65.0, 5.0], [75.0, 5.0]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    weights = spatial_diverse_topk_weights(
        xy=xy, scores=scores, image_size=(100, 50), top_k=2, grid_rows=1, grid_columns=2
    )
    assert weights.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_static_selector_aggregation_has_no_hypothesis_or_target_input() -> None:
    points = torch.tensor([[0.1, 1.0, -0.2], [0.2, -0.4, 0.6]])
    weights = torch.tensor([0.0, 1.0, 1.0])
    aggregated = aggregate_static_point_log_likelihood_ratios(
        point_log_likelihood_ratios=points, selector_weights=weights
    )
    torch.testing.assert_close(aggregated, torch.tensor([0.4, 0.1]))


def test_score_reweighting_uses_only_selected_target_free_scores_with_a_floor() -> None:
    weights = score_reweighted_selector_weights(
        selected_weights=torch.tensor([1.0, 1.0, 0.0]),
        scores=torch.tensor([0.5, 1.0, 100.0]),
        floor=0.20,
        power=1.0,
    )
    # The unselected score cannot influence the active-score normalization.
    torch.testing.assert_close(weights, torch.tensor([0.6, 1.0, 0.0]))
    assert bool(torch.all(weights[:2] >= 0.20))


def test_score_reweighting_rejects_an_argmax_collapse_or_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="reweighting"):
        score_reweighted_selector_weights(
            selected_weights=torch.tensor([0.0, 0.0]),
            scores=torch.tensor([0.5, 1.0]),
        )
    with pytest.raises(ValueError, match="reweighting"):
        score_reweighted_selector_weights(
            selected_weights=torch.tensor([1.0, 1.0]),
            scores=torch.tensor([0.5, 1.0]),
            floor=0.0,
        )


def test_uniform_mass_mixture_normalizes_each_component_before_blending() -> None:
    weights = blend_selector_with_uniform_mass(
        selector_weights=torch.tensor([0.0, 1.0, 1.0]), uniform_mass=0.50
    )
    torch.testing.assert_close(
        weights,
        torch.tensor([1.0 / 6.0, 5.0 / 12.0, 5.0 / 12.0]),
    )
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    with pytest.raises(ValueError, match="uniform selector mixture"):
        blend_selector_with_uniform_mass(
            selector_weights=torch.tensor([1.0, 1.0]), uniform_mass=1.1
        )
