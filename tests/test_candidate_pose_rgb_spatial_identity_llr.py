from __future__ import annotations

import inspect
import math

import pytest
import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CandidatePoseRGBSpatialIdentityLLR,
    CandidatePoseRGBSpatialIdentityLLREdgeRepresentation,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    bounded_log_likelihood_ratio,
    marginalize_candidate_pose_rgb_spatial_identity_llr,
    resolve_candidate_pose_rgb_spatial_identity_visual_source_scales,
    support_view_log_probabilities,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_candidate_slots,
)


def _sources() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    output: dict[str, torch.Tensor] = {}
    for name, dimension, side in (
        ("radio_final", 8, 5),
        ("radio_intermediate", 8, 5),
        ("alike", 6, 7),
    ):
        output[name] = torch.nn.functional.normalize(
            torch.randn(4, side, side, dimension), dim=-1
        )
    return output


def _runtime(*, border_support: bool = False) -> CandidatePoseRGBSpatialRuntime:
    support_xy = torch.tensor(
        [
            [
                [[58.0, 43.0], [62.0, 47.0]],
                [[70.0, 49.0], [74.0, 53.0]],
            ],
            [
                [[52.0, 42.0], [56.0, 46.0]],
                [[72.0, 50.0], [76.0, 54.0]],
            ],
        ],
        dtype=torch.float32,
    )
    if border_support:
        support_xy[0, 1, 0] = torch.tensor([2.0, 2.0])
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[60.0, 45.0], [56.0, 46.0]]),
        support_image_indices=torch.tensor(
            [
                [[2, 3], [2, 3]],
                [[2, 3], [2, 3]],
            ]
        ),
        support_xy=support_xy,
        support_view_valid=torch.ones((2, 2, 2), dtype=torch.bool),
        candidate_view_weights=torch.full((2, 2, 2), 0.5),
        candidate_probabilities=torch.tensor([[0.4, 0.4], [0.6, 0.3]]),
        null_probabilities=torch.tensor([0.2, 0.1]),
    )


def _model() -> CandidatePoseRGBSpatialIdentityLLR:
    return CandidatePoseRGBSpatialIdentityLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[128.0, 96.0]]).repeat(4, 1),
        rgb_context_radius_px=8.0,
        rgb_step_px=1.0,
        texture_feature_dim=8,
        hidden_dim=8,
        edge_chunk_size=3,
        context_windows={"radio_final": 3, "radio_intermediate": 3, "alike": 3},
    )


def _patches() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(9)
    return torch.rand(2, 3, 17, 17), torch.rand(2, 2, 2, 3, 17, 17)


def _randomize_expert_final_layers(model: CandidatePoseRGBSpatialIdentityLLR) -> None:
    with torch.no_grad():
        for name in ("rgb_identity_head", "context_coherence_head"):
            final = getattr(model, name)[-1]
            assert isinstance(final, torch.nn.Linear)
            final.weight.normal_(mean=0.0, std=0.1)


def test_identity_llr_keeps_context_evidence_when_an_rgb_crop_is_unavailable() -> None:
    model = _model().eval()
    context_final = model.context_coherence_head[-1]
    assert isinstance(context_final, torch.nn.Linear)
    with torch.no_grad():
        context_final.bias.fill_(0.5)
    query, support = _patches()
    prediction = model(
        runtime=_runtime(border_support=True),
        query_rgb_patches=query,
        support_rgb_patches=support,
    )
    assert prediction.edge_log_likelihood_ratios.shape == (2, 2, 2)
    assert prediction.rgb_identity_edge_log_likelihood_ratios.shape == (2, 2, 2)
    assert prediction.context_coherence_edge_log_likelihood_ratios.shape == (2, 2, 2)
    assert prediction.edge_usable.shape == (2, 2, 2)
    assert torch.max(torch.abs(prediction.edge_log_likelihood_ratios)) <= 4.0 + 1e-6
    # The support patch is outside the RGB window, but full-map RADIO/ALIKE
    # context is still valid. RGB must be neutral rather than suppressing the
    # independent context evidence for this candidate/view edge.
    assert bool(prediction.edge_usable[0, 1, 0])
    assert not bool(prediction.rgb_edge_usable[0, 1, 0])
    assert bool(prediction.context_edge_usable[0, 1, 0])
    assert float(prediction.rgb_identity_edge_log_likelihood_ratios[0, 1, 0]) == 0.0
    assert float(prediction.context_coherence_edge_log_likelihood_ratios[0, 1, 0]) > 0.0
    assert torch.allclose(
        prediction.edge_log_likelihood_ratios[0, 1, 0],
        prediction.context_coherence_edge_log_likelihood_ratios[0, 1, 0],
    )


def test_identity_llr_view_marginalization_preserves_fixed_weights() -> None:
    runtime = _runtime()
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=torch.tensor(
            [
                [[0.0, torch.log(torch.tensor(3.0))], [1.0, 1.0]],
                [[-1.0, -1.0], [2.0, 2.0]],
            ]
        ),
        edge_usable=torch.tensor(
            [
                [[True, True], [True, True]],
                [[True, False], [True, True]],
            ]
        ),
    )
    actual = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=prediction,
        runtime=runtime,
        missing_edge_log_likelihood_ratio=0.0,
    )
    expected_first = torch.log(torch.tensor(0.5 * 1.0 + 0.5 * 3.0))
    assert torch.allclose(actual[0, 0], expected_first)
    assert torch.allclose(
        actual[1, 0], torch.tensor(math.log(0.5 * math.exp(-1.0) + 0.5))
    )
    assert torch.isfinite(actual).all()


def test_identity_llr_learned_view_posterior_is_normalized_and_keeps_missing_mass_neutral() -> None:
    runtime = _runtime(border_support=True)
    llrs = torch.zeros((2, 2, 2), dtype=torch.float32)
    llrs[0, 0, 1] = math.log(3.0)
    logits = torch.zeros_like(llrs)
    logits[0, 0, 1] = math.log(3.0)
    # The unusable border view must not become a learned shortcut even when it
    # is assigned an extreme raw view logit.
    logits[0, 1, 0] = 100.0
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=llrs,
        edge_usable=torch.tensor(
            [
                [[True, True], [False, True]],
                [[True, True], [True, True]],
            ]
        ),
        support_view_logits=logits,
        point_null_log_likelihood_ratios=torch.tensor([0.4, -0.2]),
    )
    view_log_probabilities, available_mass, has_available = support_view_log_probabilities(
        prediction=prediction, runtime=runtime
    )
    assert bool(has_available.all())
    assert torch.allclose(available_mass[0], torch.tensor([1.0, 0.5]))
    assert torch.allclose(view_log_probabilities[0, 0].exp().sum(), torch.tensor(1.0))
    assert torch.isneginf(view_log_probabilities[0, 1, 0])
    values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=prediction, runtime=runtime
    )
    # Candidate 0 uses learned posterior [0.25, 0.75], hence 0.25*1+0.75*3.
    assert torch.allclose(values[0, 0], torch.tensor(math.log(2.5)), atol=1e-6)
    # Candidate 1 keeps the unavailable fixed mass neutral instead of moving
    # it onto its sole usable support view.
    assert torch.allclose(values[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(prediction.point_null_log_likelihood_ratios, torch.tensor([0.4, -0.2]))


def test_identity_llr_all_unavailable_support_views_have_finite_gradients() -> None:
    runtime = _runtime()
    edge_llrs = torch.zeros((2, 2, 2), dtype=torch.float32, requires_grad=True)
    view_logits = torch.zeros((2, 2, 2), dtype=torch.float32, requires_grad=True)
    prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=edge_llrs,
        edge_usable=torch.zeros((2, 2, 2), dtype=torch.bool),
        support_view_logits=view_logits,
    )
    # Every candidate falls back to its immutable missing-view mass.  This is
    # a normal padded/border condition, so no -inf intermediate is allowed to
    # poison the scalar-head backward pass.
    values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=prediction,
        runtime=runtime,
    )
    edge_grad, view_grad = torch.autograd.grad(values.sum(), (edge_llrs, view_logits))
    assert torch.isfinite(values).all()
    assert torch.isfinite(edge_grad).all()
    assert torch.isfinite(view_grad).all()
    assert torch.equal(edge_grad, torch.zeros_like(edge_grad))
    assert torch.equal(view_grad, torch.zeros_like(view_grad))


def test_identity_llr_support_appearance_changes_the_visual_edge_scores() -> None:
    model = _model().eval()
    _randomize_expert_final_layers(model)
    query, support = _patches()
    baseline = model(
        runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support
    ).edge_log_likelihood_ratios
    permuted = model(
        runtime=_runtime(),
        query_rgb_patches=query,
        support_rgb_patches=support.flip(2),
    ).edge_log_likelihood_ratios
    assert float(torch.max(torch.abs(baseline - permuted))) > 1e-6


def test_identity_llr_combines_only_the_two_expert_log_likelihood_factors() -> None:
    model = _model().eval()
    query, support = _patches()
    rgb_final = model.rgb_identity_head[-1]
    context_final = model.context_coherence_head[-1]
    assert isinstance(rgb_final, torch.nn.Linear)
    assert isinstance(context_final, torch.nn.Linear)
    with torch.no_grad():
        rgb_final.weight.normal_(mean=0.0, std=0.1)
        rgb_final.bias.normal_(mean=0.0, std=0.1)
        context_final.weight.zero_()
        context_final.bias.zero_()
    rgb_only = model(
        runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support
    )
    assert torch.allclose(
        rgb_only.edge_log_likelihood_ratios,
        rgb_only.rgb_identity_edge_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.equal(
        rgb_only.context_coherence_edge_log_likelihood_ratios,
        torch.zeros_like(rgb_only.edge_log_likelihood_ratios),
    )
    with torch.no_grad():
        rgb_final.weight.zero_()
        rgb_final.bias.zero_()
        context_final.weight.normal_(mean=0.0, std=0.1)
        context_final.bias.normal_(mean=0.0, std=0.1)
    context_only = model(
        runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support
    )
    assert torch.allclose(
        context_only.edge_log_likelihood_ratios,
        context_only.context_coherence_edge_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.equal(
        context_only.rgb_identity_edge_log_likelihood_ratios,
        torch.zeros_like(context_only.edge_log_likelihood_ratios),
    )


def test_identity_llr_is_equivariant_to_complete_candidate_slot_permutation() -> None:
    model = _model().eval()
    _randomize_expert_final_layers(model)
    with torch.no_grad():
        view_final = model.support_view_head[-1]
        assert isinstance(view_final, torch.nn.Linear)
        view_final.weight.normal_(mean=0.0, std=0.1)
        null_final = model.null_head[-1]
        assert isinstance(null_final, torch.nn.Linear)
        null_final.weight.normal_(mean=0.0, std=0.1)
    query, support = _patches()
    order = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    baseline = model(runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support)
    permuted_runtime = permute_runtime_candidate_slots(_runtime(), permutations=order)
    permuted_support = support.gather(
        1,
        order[:, :, None, None, None, None].expand_as(support),
    )
    permuted = model(
        runtime=permuted_runtime,
        query_rgb_patches=query,
        support_rgb_patches=permuted_support,
    )
    expected = baseline.edge_log_likelihood_ratios.gather(
        1, order.unsqueeze(-1).expand_as(baseline.edge_log_likelihood_ratios)
    )
    expected_usable = baseline.edge_usable.gather(
        1, order.unsqueeze(-1).expand_as(baseline.edge_usable)
    )
    assert torch.allclose(permuted.edge_log_likelihood_ratios, expected, atol=1e-6)
    for name in (
        "rgb_identity_edge_log_likelihood_ratios",
        "context_coherence_edge_log_likelihood_ratios",
    ):
        expected_component = getattr(baseline, name).gather(
            1, order.unsqueeze(-1).expand_as(getattr(baseline, name))
        )
        assert torch.allclose(getattr(permuted, name), expected_component, atol=1e-6)
    expected_views = baseline.support_view_logits.gather(
        1, order.unsqueeze(-1).expand_as(baseline.support_view_logits)
    )
    assert torch.allclose(permuted.support_view_logits, expected_views, atol=1e-6)
    assert torch.allclose(
        permuted.point_null_log_likelihood_ratios,
        baseline.point_null_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.equal(permuted.edge_usable, expected_usable)


def test_identity_llr_edge_representation_matches_the_scalar_forward_inputs() -> None:
    model = _model().eval()
    _randomize_expert_final_layers(model)
    query, support = _patches()
    representation = model.forward_edge_representation(
        runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support
    )
    prediction = model(runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support)
    assert isinstance(representation, CandidatePoseRGBSpatialIdentityLLREdgeRepresentation)
    assert set(representation.context_source_features) == {
        "radio_final",
        "radio_intermediate",
        "alike",
    }
    assert representation.edge_embeddings.shape == (2, 2, 2, 16)
    assert representation.relative_features.shape == (2, 2, 2, 64)
    assert representation.rgb_pair_features.shape[:3] == (2, 2, 2)
    assert representation.context_source_features["radio_final"].shape[:3] == (2, 2, 2)
    assert representation.context_source_features["radio_intermediate"].shape[:3] == (2, 2, 2)
    assert representation.context_source_features["alike"].shape[:3] == (2, 2, 2)
    rgb_raw = model._candidate_set_relative_edge_raw(
        relative_features=model._source_masked_relative_features(
            relative_features=representation.relative_features, source="rgb"
        ),
        edge_usable=representation.edge_usable,
        head=model.rgb_identity_head,
    )
    context_raw = model._candidate_set_relative_edge_raw(
        relative_features=model._source_masked_relative_features(
            relative_features=representation.relative_features, source="context"
        ),
        edge_usable=representation.edge_usable,
        head=model.context_coherence_head,
    )
    expected = bounded_log_likelihood_ratio(
        torch.where(representation.rgb_edge_usable, rgb_raw, torch.zeros_like(rgb_raw))
        + torch.where(
            representation.context_edge_usable, context_raw, torch.zeros_like(context_raw)
        ),
        max_abs_log_ratio=model.max_abs_log_ratio,
    )
    expected = torch.where(representation.edge_usable, expected, torch.zeros_like(expected))
    assert torch.allclose(prediction.edge_log_likelihood_ratios, expected, atol=1e-6)


def test_identity_llr_edge_representation_is_eval_only_and_keeps_missing_sources_neutral() -> None:
    query, support = _patches()
    model = _model()
    with pytest.raises(RuntimeError, match="model.eval"):
        model.forward_edge_representation(
            runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support
        )
    representation = model.eval().forward_edge_representation(
        runtime=_runtime(border_support=True), query_rgb_patches=query, support_rgb_patches=support
    )
    assert not bool(representation.rgb_edge_usable[0, 1, 0])
    assert bool(representation.context_edge_usable[0, 1, 0])
    assert torch.equal(
        representation.rgb_pair_features[0, 1, 0],
        torch.zeros_like(representation.rgb_pair_features[0, 1, 0]),
    )
    assert not torch.equal(
        representation.context_source_features["radio_final"][0, 1, 0],
        torch.zeros_like(representation.context_source_features["radio_final"][0, 1, 0]),
    )


def test_identity_llr_is_invariant_to_exact_support_view_duplication() -> None:
    """A fixed view mixture must not change when every view is duplicated.

    Broad observation pretraining provides one support observation per
    candidate, while the P1 runtime can provide multiple fixed support views.
    Duplicating every edge and splitting its fixed mixture mass is a semantic
    no-op; candidate-set normalization must preserve that fact.
    """

    model = _model().eval()
    _randomize_expert_final_layers(model)
    runtime = _runtime()
    query, support = _patches()
    baseline = model(runtime=runtime, query_rgb_patches=query, support_rgb_patches=support)
    duplicated_runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=runtime.support_image_indices.repeat_interleave(2, dim=2),
        support_xy=runtime.support_xy.repeat_interleave(2, dim=2),
        support_view_valid=runtime.support_view_valid.repeat_interleave(2, dim=2),
        candidate_view_weights=runtime.candidate_view_weights.repeat_interleave(2, dim=2)
        * 0.5,
        candidate_probabilities=runtime.candidate_probabilities,
        null_probabilities=runtime.null_probabilities,
    )
    duplicated = model(
        runtime=duplicated_runtime,
        query_rgb_patches=query,
        support_rgb_patches=support.repeat_interleave(2, dim=2),
    )
    baseline_values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=baseline, runtime=runtime
    )
    duplicated_values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=duplicated, runtime=duplicated_runtime
    )
    assert torch.allclose(duplicated_values, baseline_values, atol=1e-6)


def test_identity_llr_position_only_control_keeps_runtime_geometry_but_not_visual_values() -> None:
    model = _model().eval()
    _randomize_expert_final_layers(model)
    query, support = _patches()
    normal = model(runtime=_runtime(), query_rgb_patches=query, support_rgb_patches=support)
    position_only = model(
        runtime=_runtime(),
        query_rgb_patches=query,
        support_rgb_patches=support,
        visual_content_scale=0.0,
    )
    assert torch.equal(normal.edge_usable, position_only.edge_usable)
    assert torch.isfinite(position_only.edge_log_likelihood_ratios).all()
    assert float(
        torch.max(torch.abs(normal.edge_log_likelihood_ratios - position_only.edge_log_likelihood_ratios))
    ) > 1e-6


def test_identity_llr_visual_source_controls_preserve_the_all_zero_baseline() -> None:
    model = _model().eval()
    _randomize_expert_final_layers(model)
    query, support = _patches()
    legacy = model(
        runtime=_runtime(),
        query_rgb_patches=query,
        support_rgb_patches=support,
        visual_content_scale=0.0,
    )
    per_source = model(
        runtime=_runtime(),
        query_rgb_patches=query,
        support_rgb_patches=support,
        visual_source_scales={
            "radio_final": 0.0,
            "radio_intermediate": 0.0,
            "alike": 0.0,
            "rgb": 0.0,
        },
    )
    assert torch.equal(legacy.edge_usable, per_source.edge_usable)
    assert torch.allclose(
        legacy.edge_log_likelihood_ratios,
        per_source.edge_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.allclose(legacy.support_view_logits, per_source.support_view_logits, atol=1e-6)
    assert torch.allclose(
        legacy.point_null_log_likelihood_ratios,
        per_source.point_null_log_likelihood_ratios,
        atol=1e-6,
    )


def test_identity_llr_visual_source_controls_reject_unknown_or_out_of_range_scales() -> None:
    assert resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
        visual_content_scale=0.5,
        visual_source_scales={"radio_final": 0.4, "rgb": 0.0},
    ) == {
        "radio_final": 0.2,
        "radio_intermediate": 0.5,
        "alike": 0.5,
        "rgb": 0.0,
    }
    with pytest.raises(ValueError, match="source-scale set"):
        resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
            visual_source_scales={"track_id": 1.0}
        )
    with pytest.raises(ValueError, match="source scale"):
        resolve_candidate_pose_rgb_spatial_identity_visual_source_scales(
            visual_source_scales={"rgb": 1.1}
        )


def test_identity_llr_forward_does_not_accept_pose_or_target_inputs() -> None:
    parameter_names = set(inspect.signature(CandidatePoseRGBSpatialIdentityLLR.forward).parameters)
    forbidden = {"pose", "projection", "residual", "target", "track_id", "coarse_score"}
    assert not parameter_names & forbidden
