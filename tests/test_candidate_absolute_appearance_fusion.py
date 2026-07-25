from __future__ import annotations

import inspect

import pytest
import torch

from feature_extract.tools.vfm.train_candidate_absolute_appearance_fusion import (
    _phase_posterior_identity_audit,
)
from feature_extract.vfm.localization.candidate_absolute_appearance_fusion import (
    CandidateAbsoluteAppearanceFusion,
    phase_conditioned_candidate_posterior_margin_loss,
    phase_conditioned_candidate_posterior_nll,
)
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CandidateHighresRGBMultiscalePrediction,
    CandidateHighresRGBScalePrediction,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityPrediction,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    local_offset_grid,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[32.0, 32.0], [40.0, 40.0]]),
        support_image_indices=torch.tensor([[[2, 3], [4, 5]], [[2, 3], [4, 5]]]),
        support_xy=torch.tensor(
            [
                [[[31.0, 31.0], [33.0, 33.0]], [[50.0, 50.0], [52.0, 52.0]]],
                [[[41.0, 41.0], [39.0, 39.0]], [[60.0, 60.0], [62.0, 62.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.45, 0.35], [0.50, 0.30]]),
        null_probabilities=torch.tensor([0.20, 0.20]),
    )


def _phase_prediction(
    runtime: CandidatePoseRGBSpatialRuntime, *, zero: bool = False
) -> CandidateMultiscalePhaseIdentityPrediction:
    values = torch.zeros(runtime.support_view_valid.shape)
    if not zero:
        values[:, 0] = 0.8
        values[:, 1] = -0.4
    source_values = {
        name: values.clone() if name == "radio_final" else torch.zeros_like(values)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    source_usable = {
        name: runtime.support_view_valid.clone()
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    return CandidateMultiscalePhaseIdentityPrediction(
        source_edge_log_likelihood_ratios=source_values,
        source_edge_usable=source_usable,
        edge_log_likelihood_ratios=source_values["radio_final"],
        edge_usable=runtime.support_view_valid.clone(),
        source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
    )


def _rgb_prediction(
    runtime: CandidatePoseRGBSpatialRuntime, *, zero: bool = False
) -> CandidateHighresRGBMultiscalePrediction:
    offsets = local_offset_grid(search_radius_px=1.0, step_px=1.0)
    edge_shape = runtime.support_view_valid.shape
    spatial = torch.zeros((*edge_shape, len(offsets)))
    if not zero:
        center = int(len(offsets) // 2)
        spatial[..., center] = 2.0
    non_dustbin = torch.zeros(edge_shape)
    joint = normalized_spatial_log_probabilities_with_dustbin(
        spatial.reshape(-1, len(offsets)), non_dustbin.reshape(-1)
    ).reshape(*edge_shape, len(offsets) + 1)
    source = CandidateHighresRGBScalePrediction(
        spatial_logits=spatial,
        non_dustbin_logits=non_dustbin,
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=torch.zeros(edge_shape),
        edge_usable=runtime.support_view_valid.clone(),
    )
    return CandidateHighresRGBMultiscalePrediction(sources={"fine": source, "broad": source})


def _projections(
    runtime: CandidatePoseRGBSpatialRuntime, *, valid: bool = True, offset: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.full((1, runtime.point_count, runtime.candidate_count, 2), offset)
    return offsets, torch.full(offsets.shape[:-1], valid, dtype=torch.bool)


def _evidence(
    runtime: CandidatePoseRGBSpatialRuntime, *, phase_zero: bool = False, rgb_zero: bool = False
):
    fusion = CandidateAbsoluteAppearanceFusion()
    return fusion, fusion(
        runtime=runtime,
        phase_prediction=_phase_prediction(runtime, zero=phase_zero),
        rgb_prediction=_rgb_prediction(runtime, zero=rgb_zero),
    )


def test_visual_forward_boundary_accepts_no_pose_or_target() -> None:
    signature = inspect.signature(CandidateAbsoluteAppearanceFusion.forward)
    assert "pose" not in signature.parameters
    assert "target" not in signature.parameters
    runtime = _runtime()
    _, evidence = _evidence(runtime)
    assert evidence.phase_source_name == "radio_final"


def test_phase_reweights_candidate_prior_but_preserves_explicit_null() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    candidates, null, phase, usable, strength = fusion.phase_conditioned_candidate_probabilities(
        runtime=runtime,
        evidence=evidence,
        phase_prior_strength=1.0,
    )
    assert float(strength.item()) == 1.0
    assert bool(usable.all())
    assert torch.all(phase[:, 0] > phase[:, 1])
    assert torch.all(candidates[:, 0] > runtime.candidate_probabilities[:, 0])
    assert torch.all(candidates[:, 1] < runtime.candidate_probabilities[:, 1])
    torch.testing.assert_close(candidates.sum(dim=1) + null, torch.ones_like(null))


def test_phase_only_is_pose_invariant_and_rgb_only_is_pose_projected() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    correct_offsets, valid = _projections(runtime, offset=0.0)
    wrong_offsets, _ = _projections(runtime, offset=1.0)
    phase_correct = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=correct_offsets,
        candidate_projection_valid=valid,
        mode="phase_identity_only",
    )
    phase_wrong = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=wrong_offsets,
        candidate_projection_valid=valid,
        mode="phase_identity_only",
    )
    torch.testing.assert_close(
        phase_correct.pose_log_likelihood_ratios,
        torch.zeros_like(phase_correct.pose_log_likelihood_ratios),
    )
    torch.testing.assert_close(
        phase_correct.pose_log_likelihood_ratios,
        phase_wrong.pose_log_likelihood_ratios,
    )
    rgb_correct = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=correct_offsets,
        candidate_projection_valid=valid,
        mode="rgb_only",
    )
    rgb_wrong = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=wrong_offsets,
        candidate_projection_valid=valid,
        mode="rgb_only",
    )
    assert not torch.allclose(rgb_correct.pose_log_likelihood_ratios, rgb_wrong.pose_log_likelihood_ratios)


def test_zero_phase_returns_frozen_candidate_and_null_probabilities_exactly() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime, phase_zero=True)
    candidates, null, _, _, _ = fusion.phase_conditioned_candidate_probabilities(
        runtime=runtime,
        evidence=evidence,
    )
    torch.testing.assert_close(candidates, runtime.candidate_probabilities)
    torch.testing.assert_close(null, runtime.null_probabilities)


def test_out_of_window_rgb_projection_cannot_receive_phase_pose_reward() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    offsets, invalid = _projections(runtime, valid=False)
    score = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=invalid,
        mode="fused",
    )
    torch.testing.assert_close(
        score.pose_log_likelihood_ratios,
        torch.zeros_like(score.pose_log_likelihood_ratios),
    )
    assert not bool(score.edge_usable.any())


def test_zero_visual_evidence_is_exactly_neutral() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime, phase_zero=True, rgb_zero=True)
    offsets, valid = _projections(runtime)
    score = fusion.score(
        runtime=runtime,
        evidence=evidence,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=valid,
    )
    torch.testing.assert_close(
        score.pose_log_likelihood_ratios,
        torch.zeros_like(score.pose_log_likelihood_ratios),
    )


def test_phase_prior_strength_requires_a_nonzero_bounded_initialization() -> None:
    with pytest.raises(ValueError, match="phase prior strength"):
        CandidateAbsoluteAppearanceFusion(initial_phase_prior_strength=0.0)
    with pytest.raises(ValueError, match="phase prior strength"):
        CandidateAbsoluteAppearanceFusion(initial_phase_prior_strength=2.0, max_phase_prior_strength=2.0)


def test_posterior_identity_nll_trains_the_same_candidate_null_mixture_as_pose_scoring() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    loss, metrics = phase_conditioned_candidate_posterior_nll(
        fusion=fusion,
        runtime=runtime,
        evidence=evidence,
        observed_candidate_mask=torch.tensor([[True, False], [False, False]]),
        target_dustbin=torch.tensor([False, True]),
        target_supervised=torch.tensor([True, True]),
    )
    assert metrics["posterior_identity_active"] == 2.0
    loss.backward()
    assert fusion.phase_prior_logit.grad is not None
    assert float(fusion.phase_prior_logit.grad.abs().item()) > 0.0


def test_phase_posterior_audit_reports_target_joined_prior_lift_only_after_visual_forward() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    totals = _phase_posterior_identity_audit(
        fusion=fusion,
        runtime=runtime,
        evidence=evidence,
        observed=torch.tensor([[True, False], [True, False]]),
        dustbin=torch.tensor([False, False]),
        supervised=torch.tensor([True, True]),
    )
    assert totals.shape == (7,)
    assert float(totals[0].item()) == 2.0
    assert float(totals[6].item()) > float(totals[5].item())


def test_posterior_hard_repeat_margin_backpropagates_through_the_prior_conditioner() -> None:
    runtime = _runtime()
    fusion, evidence = _evidence(runtime)
    loss, metrics = phase_conditioned_candidate_posterior_margin_loss(
        fusion=fusion,
        runtime=runtime,
        evidence=evidence,
        point_indices=torch.tensor([0, 1]),
        positive_candidate_indices=torch.tensor([0, 0]),
        negative_candidate_indices=torch.tensor([1, 1]),
        margin=0.25,
    )
    assert metrics["posterior_hard_repeat_active"] == 2.0
    assert metrics["posterior_hard_repeat_gap"] > 0.0
    loss.backward()
    assert fusion.phase_prior_logit.grad is not None
    assert float(fusion.phase_prior_logit.grad.abs().item()) > 0.0
