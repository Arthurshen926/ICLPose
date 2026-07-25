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
from feature_extract.vfm.localization.candidate_phase_identity_spatial_likelihood import (
    permute_support_patches_with_phase_identity_point_blocks,
    phase_identity_spatial_edge_log_likelihood_ratios,
    resolve_phase_spatial_weights,
    score_candidate_phase_identity_spatial_batch,
    selected_candidate_phase_identity_spatial_log_likelihood_ratios,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[30.0, 30.0]]),
        support_image_indices=torch.tensor([[[1], [2]]]),
        support_xy=torch.tensor([[[[30.0, 30.0]], [[30.0, 30.0]]]]),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.6, 0.3]]),
        null_probabilities=torch.tensor([0.1]),
    )


def _phase(*, usable: torch.Tensor | None = None) -> CandidateMultiscalePhaseIdentityPrediction:
    values = torch.tensor([[[1.0], [-1.0]]])
    active = torch.ones_like(values, dtype=torch.bool) if usable is None else usable
    sources = {name: values.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES}
    masks = {name: active.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES}
    return CandidateMultiscalePhaseIdentityPrediction(
        source_edge_log_likelihood_ratios=sources,
        source_edge_usable=masks,
        edge_log_likelihood_ratios=torch.where(active, values, torch.zeros_like(values)),
        edge_usable=active,
        source_weights={name: 1.0 for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES},
    )


def _spatial() -> CandidateHighresRGBMultiscalePrediction:
    offsets = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    # Candidate zero peaks at the top-left mode; candidate one peaks at the
    # opposite mode.  Both are deliberately normalized with a dustbin class.
    logits = torch.tensor(
        [[[[4.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 4.0]]]], dtype=torch.float32
    )
    dustbin = torch.zeros((1, 2, 1), dtype=torch.float32)
    joint = normalized_spatial_log_probabilities_with_dustbin(
        logits.reshape(-1, 4), dustbin.reshape(-1)
    ).reshape(1, 2, 1, 5)
    fine = CandidateHighresRGBScalePrediction(
        spatial_logits=logits,
        non_dustbin_logits=dustbin,
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=torch.zeros((1, 2, 1)),
        edge_usable=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    broad = CandidateHighresRGBScalePrediction(
        spatial_logits=logits.clone(),
        non_dustbin_logits=dustbin.clone(),
        joint_log_probabilities=joint.clone(),
        offsets_xy=offsets.clone(),
        edge_log_likelihood_ratios=torch.zeros((1, 2, 1)),
        edge_usable=torch.ones((1, 2, 1), dtype=torch.bool),
    )
    return CandidateHighresRGBMultiscalePrediction(sources={"fine": fine, "broad": broad})


def test_hybrid_requires_the_rgb_projection_to_be_in_window_before_identity_can_help() -> None:
    runtime = _runtime()
    values, usable, _ = phase_identity_spatial_edge_log_likelihood_ratios(
        runtime=runtime,
        phase_prediction=_phase(),
        spatial_prediction=_spatial(),
        candidate_projection_offsets_xy=torch.tensor([[[[8.0, 8.0], [8.0, 8.0]]]]),
        candidate_projection_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        phase_source_name="radio_final",
    )
    assert not bool(usable.any())
    torch.testing.assert_close(values, torch.zeros_like(values))


def test_hybrid_requires_both_phase_and_rgb_source_availability() -> None:
    runtime = _runtime()
    phase_usable = torch.tensor([[[True], [False]]])
    values, usable, _ = phase_identity_spatial_edge_log_likelihood_ratios(
        runtime=runtime,
        phase_prediction=_phase(usable=phase_usable),
        spatial_prediction=_spatial(),
        candidate_projection_offsets_xy=torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]]),
        candidate_projection_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        phase_source_name="radio_final",
    )
    assert usable.tolist() == [[[[True], [False]]]]
    assert values[0, 0, 1, 0].item() == pytest.approx(0.0)


def test_hybrid_common_control_availability_cannot_create_or_remove_asymmetric_evidence() -> None:
    runtime = _runtime()
    values, usable, _ = phase_identity_spatial_edge_log_likelihood_ratios(
        runtime=runtime,
        phase_prediction=_phase(),
        spatial_prediction=_spatial(),
        candidate_projection_offsets_xy=torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]]),
        candidate_projection_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        phase_source_name="radio_final",
        edge_availability_override=torch.tensor([[[False], [True]]]),
    )
    assert usable.tolist() == [[[[False], [True]]]]
    assert values[0, 0, 0, 0].item() == pytest.approx(0.0)


def test_hybrid_keeps_fixed_topl_and_null_mixture_instead_of_argmax() -> None:
    runtime = _runtime()
    score = score_candidate_phase_identity_spatial_batch(
        runtime=runtime,
        phase_prediction=_phase(),
        spatial_prediction=_spatial(),
        candidate_projection_offsets_xy=torch.tensor([[[[-1.0, -1.0], [1.0, 1.0]]]]),
        candidate_projection_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        phase_source_name="radio_final",
    )
    assert score.candidate_log_likelihood_ratios.shape == (1, 1, 2)
    assert score.pose_log_likelihood_ratios.shape == (1,)
    assert score.edge_usable.all()
    assert score.candidate_log_likelihood_ratios[0, 0, 0] != score.candidate_log_likelihood_ratios[0, 0, 1]


def test_hybrid_weights_are_a_positive_convex_blend() -> None:
    assert resolve_phase_spatial_weights(identity_weight=2.0, spatial_weight=1.0) == pytest.approx(
        (2.0 / 3.0, 1.0 / 3.0)
    )
    with pytest.raises(ValueError, match="positive"):
        resolve_phase_spatial_weights(identity_weight=1.0, spatial_weight=0.0)


def test_selected_hybrid_score_and_phase_matched_rgb_derangement() -> None:
    runtime = _runtime()
    values, usable = selected_candidate_phase_identity_spatial_log_likelihood_ratios(
        runtime=runtime,
        phase_prediction=_phase(),
        spatial_prediction=_spatial(),
        point_indices=torch.tensor([0, 0]),
        candidate_indices=torch.tensor([0, 1]),
        offsets_xy=torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        phase_source_name="radio_final",
    )
    assert usable.tolist() == [True, True]
    assert values[0] > values[1]
    patches = torch.arange(2 * 2 * 1 * 3 * 3 * 3, dtype=torch.float32).reshape(2, 2, 1, 3, 3, 3)
    runtime_two = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 0]),
        query_xy=torch.tensor([[30.0, 30.0], [30.0, 30.0]]),
        support_image_indices=torch.tensor([[[1], [2]], [[1], [2]]]),
        support_xy=torch.full((2, 2, 1, 2), 30.0),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1)),
        candidate_probabilities=torch.full((2, 2), 0.45),
        null_probabilities=torch.full((2,), 0.10),
    )
    deranged = permute_support_patches_with_phase_identity_point_blocks(
        runtime=runtime_two, support_patches=patches, shift=1
    )
    torch.testing.assert_close(deranged, torch.roll(patches, shifts=1, dims=0))
