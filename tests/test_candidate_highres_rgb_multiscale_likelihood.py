from __future__ import annotations

import inspect

import pytest
import torch

from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_SOURCES,
    CandidateHighresRGBMultiscaleLikelihood,
    CandidateHighresRGBMultiscalePrediction,
    CandidateHighresRGBScalePrediction,
    _regular_grouped_template_cost_volume_logits,
    highres_rgb_candidate_identity_log_likelihood_ratios,
    highres_rgb_candidate_identity_plus_null_logits,
    highres_rgb_spatial_density_nll,
    highres_rgb_target_free_point_quality,
    score_candidate_highres_rgb_multiscale_batch,
    selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets,
    temper_candidate_prior_probabilities,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    template_search_cost_volume_logits,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_candidate_slots,
    permute_support_patch_appearance,
)


def _runtime(*, duplicate_view: bool = False) -> CandidatePoseRGBSpatialRuntime:
    view_count = 2 if duplicate_view else 2
    support_indices = torch.tensor(
        [
            [[2, 3], [4, 5]],
            [[2, 3], [4, 5]],
        ],
        dtype=torch.long,
    )
    support_xy = torch.tensor(
        [
            [[[44.0, 35.0], [49.0, 39.0]], [[58.0, 44.0], [61.0, 49.0]]],
            [[[47.0, 37.0], [51.0, 42.0]], [[57.0, 46.0], [63.0, 52.0]]],
        ],
        dtype=torch.float32,
    )
    if duplicate_view:
        support_indices[..., 1] = support_indices[..., 0]
        support_xy[..., 1, :] = support_xy[..., 0, :]
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[45.0, 35.0], [52.0, 42.0]]),
        support_image_indices=support_indices,
        support_xy=support_xy,
        support_view_valid=torch.ones((2, 2, view_count), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, view_count), 1.0 / float(view_count)),
        candidate_probabilities=torch.tensor([[0.45, 0.35], [0.50, 0.30]]),
        null_probabilities=torch.tensor([0.20, 0.20]),
    )


def _model() -> CandidateHighresRGBMultiscaleLikelihood:
    return CandidateHighresRGBMultiscaleLikelihood(
        image_sizes=torch.tensor([[96.0, 72.0]]).repeat(6, 1),
        fine_search_radius_px=2.0,
        fine_context_radius_px=2.0,
        fine_step_px=1.0,
        broad_search_radius_px=2.0,
        broad_context_radius_px=4.0,
        broad_feature_step_px=2.0,
        broad_output_step_px=2.0,
        texture_feature_dim=8,
        hidden_dim=8,
        edge_chunk_size=3,
        rgb_temperature=4.0,
    )


def _patches(
    model: CandidateHighresRGBMultiscaleLikelihood,
    runtime: CandidatePoseRGBSpatialRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(13)
    query = torch.rand((runtime.point_count, 3, model.patch_side, model.patch_side))
    support = torch.rand(
        (
            runtime.point_count,
            runtime.candidate_count,
            runtime.support_view_count,
            3,
            model.patch_side,
            model.patch_side,
        )
    )
    return query, support


def _candidate_gather(patches: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    index = order[:, :, None, None, None, None].expand(
        -1, -1, patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]
    )
    return patches.gather(1, index)


@pytest.mark.parametrize(
    ("side", "search_radius_px", "context_radius_px", "feature_step_px"),
    (
        (9, 2.0, 2.0, 1.0),
        (7, 2.0, 4.0, 2.0),
    ),
)
def test_regular_grouped_cost_volume_matches_generic_fp32(
    side: int,
    search_radius_px: float,
    context_radius_px: float,
    feature_step_px: float,
) -> None:
    """The AMP fast path must preserve the established regular-grid semantics."""

    torch.manual_seed(97)
    query = torch.randn((3, 8, side, side), dtype=torch.float32, requires_grad=True)
    support = torch.randn((3, 8, side, side), dtype=torch.float32, requires_grad=True)
    expected, _ = template_search_cost_volume_logits(
        query,
        support,
        search_radius_px=search_radius_px,
        context_radius_px=context_radius_px,
        step_px=feature_step_px,
        feature_step_px=feature_step_px,
        output_step_px=feature_step_px,
        temperature=4.0,
    )
    actual = _regular_grouped_template_cost_volume_logits(
        query_features=query,
        support_features=support,
        search_radius_px=search_radius_px,
        context_radius_px=context_radius_px,
        feature_step_px=feature_step_px,
        temperature=4.0,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert support.grad is not None and torch.isfinite(support.grad).all()


def test_forward_is_target_free_and_scale_densities_are_normalized() -> None:
    signature = inspect.signature(CandidateHighresRGBMultiscaleLikelihood.forward)
    assert "pose" not in signature.parameters
    assert "target" not in signature.parameters
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    prediction = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    assert set(prediction.sources) == set(CANDIDATE_HIGHRES_RGB_SOURCES)
    assert prediction.sources["fine"].spatial_logits.shape[-1] == 25
    assert prediction.sources["broad"].spatial_logits.shape[-1] == 9
    for source in prediction.sources.values():
        torch.testing.assert_close(
            source.joint_log_probabilities.exp().sum(dim=-1),
            torch.ones_like(source.non_dustbin_logits),
        )


def test_runtime_type_rejects_target_bearing_or_untyped_objects() -> None:
    model = _model().eval()
    with pytest.raises(ValueError, match="target-free runtime"):
        model(  # type: ignore[arg-type]
            runtime=object(),
            query_rgb_patches=torch.zeros((1, 3, model.patch_side, model.patch_side)),
            support_rgb_patches=torch.zeros((1, 1, 1, 3, model.patch_side, model.patch_side)),
        )


def test_zero_appearance_and_zero_source_are_exactly_neutral() -> None:
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    neutral = model(
        runtime=runtime,
        query_rgb_patches=query,
        support_rgb_patches=support,
        zero_appearance=True,
    )
    offsets = torch.zeros((1, runtime.point_count, runtime.candidate_count, 2))
    valid = torch.ones(offsets.shape[:-1], dtype=torch.bool)
    for source in neutral.sources.values():
        assert not bool(source.edge_usable.any())
        assert torch.equal(source.edge_log_likelihood_ratios, torch.zeros_like(source.edge_log_likelihood_ratios))
    score = score_candidate_highres_rgb_multiscale_batch(
        runtime=runtime,
        prediction=neutral,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=valid,
        source="combined",
    )
    zero = score_candidate_highres_rgb_multiscale_batch(
        runtime=runtime,
        prediction=model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support),
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=valid,
        source="zero",
    )
    torch.testing.assert_close(score.pose_log_likelihood_ratios, torch.zeros_like(score.pose_log_likelihood_ratios))
    torch.testing.assert_close(zero.pose_log_likelihood_ratios, torch.zeros_like(zero.pose_log_likelihood_ratios))


def test_candidate_identity_zero_source_exactly_replays_fixed_candidate_null_prior() -> None:
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    prediction = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    logits, usable, weights = highres_rgb_candidate_identity_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        source="zero",
    )
    expected = torch.cat(
        (runtime.candidate_probabilities, runtime.null_probabilities[:, None]), dim=1
    )
    torch.testing.assert_close(torch.softmax(logits, dim=1), expected)
    assert not bool(usable.any())
    assert weights == {"fine": 0.0, "broad": 0.0}


def test_candidate_identity_keeps_missing_view_mass_neutral() -> None:
    runtime = _runtime()
    offsets = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    joint = torch.full((2, 2, 2, 5), -torch.log(torch.tensor(5.0)))
    fine_llr = torch.zeros((2, 2, 2))
    fine_llr[0, 0, 0] = torch.log(torch.tensor(4.0))
    fine_usable = torch.zeros((2, 2, 2), dtype=torch.bool)
    fine_usable[0, 0, 0] = True
    fine = CandidateHighresRGBScalePrediction(
        spatial_logits=torch.zeros((2, 2, 2, 4)),
        non_dustbin_logits=torch.zeros((2, 2, 2)),
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=fine_llr,
        edge_usable=fine_usable,
    )
    broad = CandidateHighresRGBScalePrediction(
        spatial_logits=torch.zeros((2, 2, 2, 4)),
        non_dustbin_logits=torch.zeros((2, 2, 2)),
        joint_log_probabilities=joint,
        offsets_xy=offsets,
        edge_log_likelihood_ratios=torch.zeros((2, 2, 2)),
        edge_usable=torch.zeros((2, 2, 2), dtype=torch.bool),
    )
    prediction = CandidateHighresRGBMultiscalePrediction(sources={"fine": fine, "broad": broad})
    candidate_llr, usable, weights = highres_rgb_candidate_identity_log_likelihood_ratios(
        runtime=runtime,
        prediction=prediction,
        source="fine",
    )
    # The unavailable second support view remains at its original 0.5 mass:
    # log(0.5 * 4 + 0.5 * 1), rather than incorrectly renormalizing to log(4).
    torch.testing.assert_close(candidate_llr[0, 0], torch.log(torch.tensor(2.5)))
    torch.testing.assert_close(candidate_llr[0, 1], torch.tensor(0.0))
    assert bool(usable[0, 0])
    assert not bool(usable[0, 1])
    assert weights == {"fine": 1.0, "broad": 0.0}


def test_candidate_prior_temperature_preserves_null_mass_and_topl_support() -> None:
    candidates = torch.tensor([[0.45, 0.30, 0.15], [0.60, 0.10, 0.00]])
    null = torch.tensor([0.10, 0.30])
    identity = temper_candidate_prior_probabilities(
        candidate_probabilities=candidates, null_probabilities=null, temperature=1.0
    )
    sharp = temper_candidate_prior_probabilities(
        candidate_probabilities=candidates, null_probabilities=null, temperature=0.5
    )
    torch.testing.assert_close(identity, candidates)
    torch.testing.assert_close(sharp.sum(dim=1) + null, torch.ones_like(null))
    assert torch.equal(sharp > 0.0, candidates > 0.0)
    assert sharp[0, 0] > candidates[0, 0]
    with pytest.raises(ValueError, match="temperature"):
        temper_candidate_prior_probabilities(
            candidate_probabilities=candidates, null_probabilities=null, temperature=0.0
        )


def test_inactive_scale_is_structurally_neutral_without_changing_active_scale() -> None:
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    both = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    broad_only = model(
        runtime=runtime,
        query_rgb_patches=query,
        support_rgb_patches=support,
        active_sources=("broad",),
    )
    assert not bool(broad_only.sources["fine"].edge_usable.any())
    torch.testing.assert_close(
        broad_only.sources["broad"].spatial_logits,
        both.sources["broad"].spatial_logits,
    )


def test_candidate_slot_permutation_is_equivariant() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    order = torch.tensor([[1, 0], [1, 0]])
    original = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    permuted = model(
        runtime=permute_runtime_candidate_slots(runtime, permutations=order),
        query_rgb_patches=query,
        support_rgb_patches=_candidate_gather(support, order),
    )
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        expected = original.sources[name].spatial_logits.gather(
            1,
            order[:, :, None, None].expand(
                -1,
                -1,
                runtime.support_view_count,
                original.sources[name].spatial_logits.shape[-1],
            ),
        )
        torch.testing.assert_close(permuted.sources[name].spatial_logits, expected)


def test_support_permutation_changes_only_support_appearance_pairing() -> None:
    torch.manual_seed(6)
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    permuted_patches = permute_support_patch_appearance(
        runtime=runtime, support_patches=support, shift=1
    )
    normal = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    permuted = model(
        runtime=runtime,
        query_rgb_patches=query,
        support_rgb_patches=permuted_patches,
    )
    assert not torch.equal(support, permuted_patches)
    assert any(
        not torch.allclose(
            normal.sources[name].spatial_logits, permuted.sources[name].spatial_logits
        )
        for name in CANDIDATE_HIGHRES_RGB_SOURCES
    )
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        assert torch.equal(normal.sources[name].edge_usable, permuted.sources[name].edge_usable)


def test_out_of_window_projection_cannot_collect_learned_dustbin_evidence() -> None:
    model = _model().eval()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    prediction = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    offsets = torch.full((1, runtime.point_count, runtime.candidate_count, 2), 99.0)
    score = score_candidate_highres_rgb_multiscale_batch(
        runtime=runtime,
        prediction=prediction,
        candidate_projection_offsets_xy=offsets,
        candidate_projection_valid=torch.ones(offsets.shape[:-1], dtype=torch.bool),
    )
    torch.testing.assert_close(score.edge_log_likelihood_ratios, torch.zeros_like(score.edge_log_likelihood_ratios))
    assert not bool(score.edge_usable.any())
    torch.testing.assert_close(score.pose_log_likelihood_ratios, torch.zeros_like(score.pose_log_likelihood_ratios))


def test_density_uses_observed_and_dustbin_targets_and_backpropagates() -> None:
    model = _model().train()
    runtime = _runtime()
    query, support = _patches(model, runtime)
    prediction = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    targets = torch.zeros((runtime.point_count, runtime.candidate_count, 2))
    dustbin = torch.tensor([[False, True], [False, True]])
    loss, metrics = highres_rgb_spatial_density_nll(
        scale_prediction=prediction.sources["fine"],
        target_offsets_xy=targets,
        target_dustbin=dustbin,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["observed_edges"] > 0.0
    assert metrics["dustbin_edges"] > 0.0
    assert model.calibrators["fine"].network[-1].bias.grad is not None


def test_selected_candidate_score_interpolates_and_fixed_view_mixture_is_duplicate_stable() -> None:
    runtime = _runtime(duplicate_view=True)
    source_predictions: dict[str, CandidateHighresRGBScalePrediction] = {}
    for name, offsets in (
        ("fine", torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])),
        ("broad", torch.tensor([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])),
    ):
        probability = torch.tensor([0.10, 0.20, 0.30, 0.30, 0.10]).reshape(1, 1, 1, -1)
        joint = probability.expand(2, 2, 2, -1).clone().log()
        source_predictions[name] = CandidateHighresRGBScalePrediction(
            spatial_logits=torch.zeros((2, 2, 2, 4)),
            non_dustbin_logits=torch.zeros((2, 2, 2)),
            joint_log_probabilities=joint,
            offsets_xy=offsets,
            edge_log_likelihood_ratios=torch.zeros((2, 2, 2)),
            edge_usable=torch.ones((2, 2, 2), dtype=torch.bool),
        )
    prediction = CandidateHighresRGBMultiscalePrediction(sources=source_predictions)
    values, usable = selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=torch.tensor([0]),
        candidate_indices=torch.tensor([0]),
        offsets_xy=torch.tensor([[0.0, 0.0]]),
        source="fine",
        max_abs_log_likelihood_ratio=1e6,
    )
    assert bool(usable.item())
    # The fine scale has a 0.225 bilinear probability at zero, whereas its
    # neutral joint probability is 1 / (2 * 4) = 0.125.
    torch.testing.assert_close(values, torch.log(torch.tensor([0.225 / 0.125])), atol=1e-5, rtol=1e-5)
    quality = highres_rgb_target_free_point_quality(
        runtime=runtime, prediction=prediction, source="combined"
    )
    assert set(quality) == {"peakiness", "non_dustbin_probability", "edge_log_likelihood_ratio"}
    assert all(value.shape == (runtime.point_count,) for value in quality.values())
