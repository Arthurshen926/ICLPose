from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_multiscale_rgb_phase import (
    RGBPhaseDensity,
    RGBPhaseScale,
    extract_rgb_phase_density,
    resolve_rgb_phase_scales,
    selected_candidate_rgb_phase_interaction_log_likelihood_ratio,
    selected_candidate_rgb_phase_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialRuntime,
    selected_candidate_view_log_likelihood_ratio_at_offsets,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
)


def test_phase_scale_has_fixed_regular_patch_and_offset_geometry() -> None:
    scale = RGBPhaseScale("wide", search_radius_px=2.0, context_radius_px=4.0, step_px=1.0)
    assert scale.patch_side == 13
    assert scale.offset_count == 25
    offsets = scale.offsets_xy()
    assert offsets.shape == (25, 2)
    assert torch.allclose(offsets[0], torch.tensor([-2.0, -2.0]))
    assert torch.allclose(offsets[-1], torch.tensor([2.0, 2.0]))
    with pytest.raises(ValueError, match="sampling grid"):
        RGBPhaseScale("bad", search_radius_px=2.0, context_radius_px=3.0, step_px=2.0)


def test_phase_density_forward_is_target_free_and_normalized() -> None:
    torch.manual_seed(5)
    scale = RGBPhaseScale("tiny", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    encoder = TexturePatchEncoder(feature_dim=4, hidden_dim=4, input_mode="rgb_graygrad", encoder_arch="fpn").eval()
    query = torch.rand(2, 3, scale.patch_side, scale.patch_side)
    support = torch.rand(2, 2, 1, 3, scale.patch_side, scale.patch_side)
    usable = torch.ones((2, 2, 1), dtype=torch.bool)
    density = extract_rgb_phase_density(
        texture_encoder=encoder,
        scale=scale,
        query_patches=query,
        support_patches=support,
        edge_usable=usable,
        edge_chunk_size=2,
        temperature=5.0,
    )
    assert density.log_probabilities.shape == (2, 2, 1, 9)
    assert torch.allclose(
        torch.logsumexp(density.log_probabilities, dim=-1),
        torch.zeros((2, 2, 1)),
        atol=1e-5,
    )


def test_phase_selected_score_interpolates_probability_and_preserves_missing_view_mass() -> None:
    scale = RGBPhaseScale("tiny", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    # Uniform first view and a central peak in the second view.  The second
    # view is intentionally unavailable, so 75% of the immutable mixture mass
    # must remain neutral rather than being silently reassigned to view zero.
    uniform = torch.full((9,), -math.log(9.0))
    peak = torch.full((9,), -20.0)
    peak[4] = 0.0
    density = RGBPhaseDensity(
        scale=scale,
        log_probabilities=torch.stack([torch.stack([uniform, peak])]).reshape(1, 1, 2, 9),
        edge_usable=torch.tensor([[[True, False]]]),
    )
    score, usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density,
        candidate_view_weights=torch.tensor([[[0.25, 0.75]]]),
        point_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([0]),
        offsets_xy=torch.tensor([[0.0, 0.0]]),
    )
    assert bool(usable[0])
    # View 0 is uniform (LLR=0); unavailable view 1 is also fixed neutral.
    assert score == pytest.approx(torch.tensor([0.0]), abs=1e-6)


def test_phase_out_of_window_projection_is_fixed_neutral_not_a_dustbin_reward() -> None:
    scale = RGBPhaseScale("tiny", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    density = RGBPhaseDensity(
        scale=scale,
        log_probabilities=torch.full((1, 1, 1, 9), -math.log(9.0)),
        edge_usable=torch.ones((1, 1, 1), dtype=torch.bool),
    )
    score, usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density,
        candidate_view_weights=torch.ones((1, 1, 1)),
        point_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([0]),
        offsets_xy=torch.tensor([[2.0, 0.0]]),
    )
    assert not bool(usable[0])
    assert score == pytest.approx(torch.tensor([0.0]), abs=1e-6)


def test_default_scales_are_unique_and_keep_wide_physical_context() -> None:
    local, wide = resolve_rgb_phase_scales()
    assert local.name != wide.name
    assert wide.context_radius_px > local.context_radius_px
    assert wide.patch_side == local.patch_side


def test_local_phase_score_matches_fixed_dustbin_rgb_only_scorer() -> None:
    """The phase refactor must preserve the gate-checked local FPN LLR."""

    torch.manual_seed(17)
    scale = RGBPhaseScale("local", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    logits = torch.randn(1, 1, 2, scale.offset_count)
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[8.0, 8.0]]),
        support_image_indices=torch.tensor([[[1, 2]]]),
        support_xy=torch.tensor([[[[8.0, 8.0], [8.0, 8.0]]]]),
        support_view_valid=torch.tensor([[[True, True]]]),
        candidate_view_weights=torch.tensor([[[0.35, 0.65]]]),
        candidate_probabilities=torch.tensor([[0.8]]),
        null_probabilities=torch.tensor([0.2]),
    )
    prediction = CandidatePoseRGBSpatialEdgePrediction(
        spatial_logits=logits,
        non_dustbin_logits=torch.zeros((1, 1, 2)),
        joint_log_probabilities=normalized_spatial_log_probabilities_with_dustbin(
            logits.reshape(-1, scale.offset_count), torch.zeros(2)
        ).reshape(1, 1, 2, scale.offset_count + 1),
        offsets_xy=scale.offsets_xy(),
        context_log_likelihood_ratios=torch.zeros((1, 1, 2)),
        edge_usable=torch.ones((1, 1, 2), dtype=torch.bool),
        rgb_edge_usable=torch.ones((1, 1, 2), dtype=torch.bool),
        context_edge_usable=torch.zeros((1, 1, 2), dtype=torch.bool),
    )
    density = RGBPhaseDensity(
        scale=scale,
        log_probabilities=F.log_softmax(logits, dim=-1),
        edge_usable=torch.ones((1, 1, 2), dtype=torch.bool),
    )
    points = torch.tensor([0, 0])
    candidates = torch.tensor([0, 0])
    offsets = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    expected_score, expected_usable = selected_candidate_view_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=points,
        candidate_indices=candidates,
        offsets_xy=offsets,
        max_abs_log_likelihood_ratio=6.0,
    )
    actual_score, actual_usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density,
        candidate_view_weights=runtime.candidate_view_weights,
        point_indices=points,
        candidate_indices=candidates,
        offsets_xy=offsets,
        max_abs_log_likelihood_ratio=6.0,
    )
    assert torch.equal(actual_usable, expected_usable)
    assert torch.allclose(actual_score, expected_score, atol=1e-6, rtol=1e-6)


def test_phase_interaction_removes_single_source_cost_volume_modes() -> None:
    scale = RGBPhaseScale("tiny", search_radius_px=1.0, context_radius_px=1.0, step_px=1.0)
    uniform = torch.full((9,), -math.log(9.0))
    peak_logits = torch.full((9,), -20.0)
    peak_logits[4] = 0.0
    peak = peak_logits - torch.logsumexp(peak_logits, dim=0)

    def density(values: torch.Tensor) -> RGBPhaseDensity:
        return RGBPhaseDensity(
            scale=scale,
            log_probabilities=values.reshape(1, 1, 1, 9),
            edge_usable=torch.ones((1, 1, 1), dtype=torch.bool),
        )

    kwargs = {
        "candidate_view_weights": torch.ones((1, 1, 1)),
        "point_indices": torch.tensor([0]),
        "candidate_indices": torch.tensor([0]),
        "offsets_xy": torch.tensor([[0.0, 0.0]]),
    }
    interaction, usable = selected_candidate_rgb_phase_interaction_log_likelihood_ratio(
        pair_density=density(peak),
        query_zero_support_density=density(uniform),
        query_support_zero_density=density(uniform),
        zero_density=density(uniform),
        **kwargs,
    )
    direct, direct_usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density(peak), **kwargs
    )
    assert torch.equal(usable, direct_usable)
    assert torch.allclose(interaction, direct, atol=1e-6, rtol=1e-6)

    cancelled, cancelled_usable = selected_candidate_rgb_phase_interaction_log_likelihood_ratio(
        pair_density=density(peak),
        query_zero_support_density=density(peak),
        query_support_zero_density=density(peak),
        zero_density=density(peak),
        **kwargs,
    )
    assert bool(cancelled_usable[0])
    assert cancelled == pytest.approx(torch.tensor([0.0]), abs=1e-6)
