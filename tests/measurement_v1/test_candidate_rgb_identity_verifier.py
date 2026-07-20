from __future__ import annotations

import math

import pytest
import torch

from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    GT_POSE_SPATIAL_DENSITY_SEMANTICS,
    IndependentRGBCandidateVerifier,
    POSE_VIEW_MIXTURE_SEMANTICS,
    SUPPORT_VIEW_MIXTURE_CONTRACT,
    aggregate_view_log_likelihood_ratios,
    fuse_candidate_log_likelihood_ratios,
    measurement_mode_residual_and_success,
    normalize_pose_view_mixture_logits,
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_training import (
    coarse_prior_hard_negative_ranking_loss,
    _resolve_identity_context_config,
    _resolve_identity_context_layout_config,
    _resolve_spatial_support_config,
    _set_identity_context_residual_training_mode,
    _set_identity_context_layout_residual_training_mode,
    gt_pose_candidate_view_mixture_nll,
    gt_pose_spatial_density_nll,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_spatial_export import (
    MEASUREMENT_VALIDITY_SEMANTICS,
    NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS,
    _load_model,
    factorize_pose_view_ensemble,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    crop_rgb_window,
    crop_rgb_windows_by_owner,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import TensorImageLRUCache
from feature_extract.tools.vfm.train_independent_rgb_candidate_verifier import (
    _parse_prediction_splits,
)


def test_spatial_support_override_is_explicit_and_validated() -> None:
    initial = {
        "search_radius_px": 6.0,
        "context_radius_px": 12.0,
        "step_px": 0.5,
    }
    resolved, overrides = _resolve_spatial_support_config(
        initial,
        context_radius_px_override=48.0,
    )
    assert resolved == {
        "search_radius_px": 6.0,
        "context_radius_px": 48.0,
        "step_px": 0.5,
    }
    assert overrides == {
        "context_radius_px": {
            "checkpoint_value": 12.0,
            "resolved_value": 48.0,
        }
    }

    unchanged, no_overrides = _resolve_spatial_support_config(initial)
    assert unchanged == initial
    assert no_overrides == {}
    with pytest.raises(ValueError, match="context_radius_px_override must be non-negative"):
        _resolve_spatial_support_config(initial, context_radius_px_override=-1.0)


def test_identity_context_config_is_explicit_and_separate_from_spatial_support() -> None:
    initial = {
        "search_radius_px": 6.0,
        "context_radius_px": 12.0,
        "step_px": 0.5,
    }
    resolved, overrides = _resolve_identity_context_config(
        initial,
        identity_context_radius_px=54.0,
        identity_context_step_px=1.5,
    )
    assert resolved == {"enabled": True, "radius_px": 54.0, "step_px": 1.5}
    assert overrides == {
        "radius_px": {"checkpoint_value": None, "resolved_value": 54.0},
        "step_px": {"checkpoint_value": None, "resolved_value": 1.5},
    }
    with pytest.raises(ValueError, match="must be provided together"):
        _resolve_identity_context_config(
            initial,
            identity_context_radius_px=54.0,
        )


def test_identity_context_layout_config_is_opt_in_and_v6_compatible() -> None:
    v6_config = {
        "identity_context_enabled": True,
        "identity_context_radius_px": 54.0,
        "identity_context_step_px": 1.5,
        "identity_context_layout_enabled": False,
        "identity_context_layout_grid_size": None,
    }
    resolved, overrides = _resolve_identity_context_layout_config(
        v6_config,
        identity_context_layout_grid_size=3,
    )
    assert resolved == {"enabled": True, "grid_size": 3}
    assert overrides == {
        "grid_size": {"checkpoint_value": None, "resolved_value": 3}
    }
    unchanged, no_overrides = _resolve_identity_context_layout_config(v6_config)
    assert unchanged == {"enabled": False, "grid_size": None}
    assert no_overrides == {}
    with pytest.raises(ValueError, match=r"must be in \[2, 8\]"):
        _resolve_identity_context_layout_config(
            v6_config,
            identity_context_layout_grid_size=1,
        )


def test_coarse_prior_hard_negative_ranking_targets_confusable_candidates() -> None:
    logits = torch.zeros((1, 3), requires_grad=True)
    loss, active = coarse_prior_hard_negative_ranking_loss(
        logits,
        labels=torch.tensor([[True, False, False]]),
        supervision_valid=torch.tensor([[True, True, True]]),
        candidate_prior=torch.tensor([[0.1, 0.8, 0.1]]),
        margin=0.0,
        prior_power=1.0,
        evidence_weight=1.0,
    )
    assert torch.equal(active, torch.tensor([True]))
    assert torch.allclose(loss, torch.tensor(math.log(10.0)), atol=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1] > logits.grad[0, 2]


def test_coarse_prior_hard_negative_ranking_ignores_ambiguous_candidates() -> None:
    logits = torch.zeros((1, 3))
    loss, active = coarse_prior_hard_negative_ranking_loss(
        logits,
        labels=torch.tensor([[True, False, False]]),
        supervision_valid=torch.tensor([[True, True, False]]),
        candidate_prior=torch.tensor([[0.1, 0.1, 0.8]]),
        margin=0.0,
        prior_power=1.0,
        evidence_weight=1.0,
    )
    assert torch.equal(active, torch.tensor([True]))
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)


def test_coarse_prior_hard_negative_ranking_skips_groups_without_hard_pairs() -> None:
    loss, active = coarse_prior_hard_negative_ranking_loss(
        torch.zeros((2, 2)),
        labels=torch.tensor([[True, False], [False, False]]),
        supervision_valid=torch.tensor([[True, False], [True, True]]),
        candidate_prior=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        margin=0.0,
        prior_power=1.0,
        evidence_weight=1.0,
    )
    assert torch.equal(active, torch.tensor([False, False]))
    assert loss.item() == 0.0


def test_dual_scale_identity_context_starts_as_exact_local_fallback() -> None:
    torch.manual_seed(71)
    local = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    dual = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    ).eval()
    incompatible = dual.load_state_dict(local.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("identity_context_") for key in incompatible.missing_keys)
    dual.initialize_identity_context_from_local_encoder()

    local_inputs = {
        "query_patches_by_group": torch.rand((2, 3, 5, 5)),
        "support_patches": torch.rand((4, 3, 5, 5)),
        "pair_group_indices": torch.tensor([0, 0, 1, 1]),
        "pair_candidate_indices": torch.tensor([0, 1, 0, 1]),
        "pair_view_slots": torch.tensor([0, 0, 0, 0]),
        "pair_view_probabilities": torch.ones((4,)),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
    }
    context_inputs = {
        "identity_context_query_patches_by_group": torch.rand((2, 3, 7, 7)),
        "identity_context_support_patches": torch.rand((4, 3, 7, 7)),
    }
    with torch.no_grad():
        expected = local(**local_inputs)
        actual = dual(**local_inputs, **context_inputs)
    assert torch.equal(expected.view_identity_logits, actual.view_identity_logits)
    assert torch.equal(
        expected.candidate_log_likelihood_ratios,
        actual.candidate_log_likelihood_ratios,
    )
    assert torch.equal(expected.view_spatial_logits, actual.view_spatial_logits)
    assert torch.equal(
        expected.view_measurement_validity_logits,
        actual.view_measurement_validity_logits,
    )


def test_identity_context_can_change_identity_without_changing_local_measurement() -> None:
    torch.manual_seed(72)
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    ).eval()
    inputs = {
        "query_patches_by_group": torch.rand((1, 3, 5, 5)),
        "support_patches": torch.rand((2, 3, 5, 5)),
        "identity_context_query_patches_by_group": torch.rand((1, 3, 7, 7)),
        "identity_context_support_patches": torch.rand((2, 3, 7, 7)),
        "pair_group_indices": torch.tensor([0, 0]),
        "pair_candidate_indices": torch.tensor([0, 1]),
        "pair_view_slots": torch.tensor([0, 0]),
        "pair_view_probabilities": torch.ones((2,)),
        "candidate_valid": torch.ones((1, 2), dtype=torch.bool),
    }
    with torch.no_grad():
        before = model(**inputs)
        assert model.identity_context_residual is not None
        model.identity_context_residual[-1].bias.fill_(1.0)
        after = model(**inputs)
    assert not torch.allclose(before.view_identity_logits, after.view_identity_logits)
    assert torch.equal(before.view_spatial_logits, after.view_spatial_logits)
    assert torch.equal(
        before.view_measurement_validity_logits,
        after.view_measurement_validity_logits,
    )
    assert torch.equal(
        before.measured_set_availability_logit,
        after.measured_set_availability_logit,
    )
    assert torch.equal(
        before.measured_set_availability_probability,
        after.measured_set_availability_probability,
    )


def test_identity_context_zero_scale_is_exact_local_counterfactual() -> None:
    torch.manual_seed(73)
    local = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    dual = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    ).eval()
    dual.load_state_dict(local.state_dict(), strict=False)
    dual.initialize_identity_context_from_local_encoder()
    assert dual.identity_context_residual is not None
    with torch.no_grad():
        dual.identity_context_residual[-1].bias.fill_(1.0)
    inputs = {
        "query_patches_by_group": torch.rand((1, 3, 5, 5)),
        "support_patches": torch.rand((2, 3, 5, 5)),
        "pair_group_indices": torch.tensor([0, 0]),
        "pair_candidate_indices": torch.tensor([0, 1]),
        "pair_view_slots": torch.tensor([0, 0]),
        "pair_view_probabilities": torch.ones((2,)),
        "candidate_valid": torch.ones((1, 2), dtype=torch.bool),
    }
    with torch.no_grad():
        expected = local(**inputs)
        actual = dual(**inputs, identity_context_residual_scale=0.0)
    assert torch.equal(expected.view_identity_logits, actual.view_identity_logits)
    assert torch.equal(
        expected.candidate_log_likelihood_ratios,
        actual.candidate_log_likelihood_ratios,
    )
    assert torch.equal(expected.view_spatial_logits, actual.view_spatial_logits)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        dual(**inputs, identity_context_residual_scale=1.1)


def test_layout_context_starts_as_exact_v6_fallback_and_is_identity_only() -> None:
    torch.manual_seed(74)
    v6 = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    ).eval()
    layout = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
    ).eval()
    incompatible = layout.load_state_dict(v6.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("identity_context_layout_") for key in incompatible.missing_keys)
    inputs = {
        "query_patches_by_group": torch.rand((2, 3, 5, 5)),
        "support_patches": torch.rand((4, 3, 5, 5)),
        "identity_context_query_patches_by_group": torch.rand((2, 3, 7, 7)),
        "identity_context_support_patches": torch.rand((4, 3, 7, 7)),
        "pair_group_indices": torch.tensor([0, 0, 1, 1]),
        "pair_candidate_indices": torch.tensor([0, 1, 0, 1]),
        "pair_view_slots": torch.tensor([0, 0, 0, 0]),
        "pair_view_probabilities": torch.ones((4,)),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
    }
    with torch.no_grad():
        expected = v6(**inputs)
        before = layout(**inputs, identity_context_layout_residual_scale=1.0)
    assert torch.equal(expected.view_identity_logits, before.view_identity_logits)
    assert torch.equal(
        expected.candidate_log_likelihood_ratios,
        before.candidate_log_likelihood_ratios,
    )

    assert layout.identity_context_layout_residual is not None
    with torch.no_grad():
        layout.view_identity_head.weight.fill_(1.0)
        layout.view_identity_head.bias.zero_()
        layout.identity_context_layout_residual[-1].bias.fill_(1.0)
        baseline = layout(
            **inputs,
            identity_context_residual_scale=0.0,
            identity_context_layout_residual_scale=0.0,
        )
        after = layout(
            **inputs,
            identity_context_residual_scale=0.0,
            identity_context_layout_residual_scale=1.0,
        )
    assert not torch.allclose(baseline.view_identity_logits, after.view_identity_logits)
    assert not torch.allclose(
        baseline.candidate_log_likelihood_ratios,
        after.candidate_log_likelihood_ratios,
    )
    assert torch.equal(baseline.view_spatial_logits, after.view_spatial_logits)
    assert torch.equal(
        baseline.view_measurement_validity_logits,
        after.view_measurement_validity_logits,
    )
    assert torch.equal(
        baseline.view_pose_mixture_logits,
        after.view_pose_mixture_logits,
    )
    assert torch.equal(
        baseline.measured_set_availability_logit,
        after.measured_set_availability_logit,
    )
    assert torch.equal(
        baseline.measured_set_availability_probability,
        after.measured_set_availability_probability,
    )


def test_layout_encoder_is_initialized_from_frozen_v6_context_encoder() -> None:
    torch.manual_seed(75)
    v6 = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    )
    layout = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
    )
    layout.load_state_dict(v6.state_dict(), strict=False)
    layout.initialize_identity_context_layout_from_pooled_encoder()
    assert layout.identity_context_encoder is not None
    assert layout.identity_context_layout_encoder is not None
    pooled_state = layout.identity_context_encoder.state_dict()
    layout_state = layout.identity_context_layout_encoder.state_dict()
    assert pooled_state.keys() == layout_state.keys()
    assert all(torch.equal(pooled_state[key], layout_state[key]) for key in pooled_state)


def test_layout_context_training_keeps_local_and_pooled_paths_in_eval_mode() -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
    )
    _set_identity_context_layout_residual_training_mode(model)
    assert not model.encoder.training
    assert not model.view_evidence.training
    assert not model.identity_context_evidence.training
    assert not model.identity_context_residual.training
    assert model.identity_context_encoder is not None
    assert not model.identity_context_encoder.training
    assert model.identity_context_layout_encoder is not None
    assert model.identity_context_layout_fusion is not None
    assert model.identity_context_layout_evidence is not None
    assert model.identity_context_layout_residual is not None
    assert model.identity_context_layout_encoder.training
    assert model.identity_context_layout_fusion.training
    assert model.identity_context_layout_evidence.training
    assert model.identity_context_layout_residual.training


def test_context_only_training_keeps_local_modules_in_eval_mode() -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    )
    _set_identity_context_residual_training_mode(model)

    assert not model.encoder.training
    assert not model.view_evidence.training
    assert not model.view_identity_head.training
    assert not model.view_measurement_validity_head.training
    assert not model.view_pose_mixture_head.training
    assert not model.set_candidate_encoder.training
    assert not model.measured_set_availability_head.training
    assert model.identity_context_encoder is not None
    assert model.identity_context_evidence is not None
    assert model.identity_context_residual is not None
    assert model.identity_context_encoder.training
    assert model.identity_context_evidence.training
    assert model.identity_context_residual.training


def test_training_prediction_split_none_is_explicitly_supported() -> None:
    assert _parse_prediction_splits("none") == ()
    assert _parse_prediction_splits("") == ()
    assert _parse_prediction_splits("train, validation") == ("train", "validation")
    with pytest.raises(ValueError, match="may contain"):
        _parse_prediction_splits("validation,late")


def test_view_mixture_is_permutation_invariant_and_missing_views_are_neutral() -> None:
    logits = torch.tensor([1.2, -0.4, 0.7], dtype=torch.float32)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    groups = torch.tensor([0, 0, 0])
    candidates = torch.tensor([0, 0, 1])
    slots = torch.tensor([0, 2, 1])
    view_probability = torch.tensor([0.7, 0.2, 0.4])

    original = aggregate_view_log_likelihood_ratios(
        logits,
        hidden,
        pair_view_probabilities=view_probability,
        pair_group_indices=groups,
        pair_candidate_indices=candidates,
        pair_view_slots=slots,
        batch_size=1,
        candidate_count=3,
        max_views=4,
    )
    permuted = aggregate_view_log_likelihood_ratios(
        logits[[2, 0, 1]],
        hidden[[2, 0, 1]],
        pair_view_probabilities=view_probability[[2, 0, 1]],
        pair_group_indices=groups[[2, 0, 1]],
        pair_candidate_indices=candidates[[2, 0, 1]],
        pair_view_slots=torch.tensor([3, 1, 0]),
        batch_size=1,
        candidate_count=3,
        max_views=4,
    )

    assert torch.allclose(original[0], permuted[0])
    assert torch.allclose(original[1], permuted[1])
    assert torch.equal(original[2], permuted[2])
    assert torch.allclose(original[3], permuted[3])
    assert original[0][0, 2].item() == 0.0
    assert not bool(original[2][0, 2])
    expected_first = torch.log(
        0.1 + 0.7 * torch.exp(logits[0]) + 0.2 * torch.exp(logits[1])
    )
    assert torch.allclose(original[0][0, 0], expected_first)
    assert torch.allclose(original[3][0], torch.tensor([0.9, 0.4, 0.0]))


def test_rgb_fusion_preserves_availability_and_is_identity_for_missing_evidence() -> None:
    prior = torch.tensor([[0.20, 0.10, 0.05, 0.15]], dtype=torch.float32)
    unknown = torch.tensor([0.50], dtype=torch.float32)
    llr = torch.tensor([[4.0, -3.0, 2.0, -1.0]], dtype=torch.float32)
    measured = torch.zeros_like(prior, dtype=torch.bool)

    unchanged, unchanged_unknown = fuse_candidate_log_likelihood_ratios(
        prior,
        unknown,
        llr,
        measured_mask=measured,
        candidate_valid=torch.ones_like(measured),
    )
    assert torch.allclose(unchanged, prior, atol=1e-7)
    assert torch.equal(unchanged_unknown, unknown)

    measured[0, 1] = True
    fused, fused_unknown = fuse_candidate_log_likelihood_ratios(
        prior,
        unknown,
        llr,
        measured_mask=measured,
        candidate_valid=torch.ones_like(measured),
    )
    assert torch.allclose(torch.sum(fused, dim=1), torch.tensor([0.50]), atol=1e-7)
    assert torch.equal(fused_unknown, unknown)
    assert fused[0, 1] < prior[0, 1]


def test_candidate_and_view_order_are_explicit_not_array_shortcuts() -> None:
    torch.manual_seed(4)
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    query = torch.rand((2, 3, 5, 5))
    support = torch.rand((6, 3, 5, 5))
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    candidates = torch.tensor([0, 0, 2, 0, 1, 2])
    slots = torch.tensor([0, 1, 0, 0, 0, 0])
    view_probability = torch.tensor([0.6, 0.4, 0.8, 0.5, 0.5, 1.0])
    valid = torch.ones((2, 3), dtype=torch.bool)

    with torch.no_grad():
        original = model(
            query_patches_by_group=query,
            support_patches=support,
            pair_group_indices=groups,
            pair_candidate_indices=candidates,
            pair_view_slots=slots,
            pair_view_probabilities=view_probability,
            candidate_valid=valid,
        )
        candidate_permutation = torch.tensor([2, 0, 1])
        inverse = torch.empty_like(candidate_permutation)
        inverse[candidate_permutation] = torch.arange(3)
        permuted = model(
            query_patches_by_group=query,
            support_patches=support[[2, 0, 1, 5, 3, 4]],
            pair_group_indices=groups[[2, 0, 1, 5, 3, 4]],
            pair_candidate_indices=inverse[candidates[[2, 0, 1, 5, 3, 4]]],
            pair_view_slots=torch.tensor([1, 1, 0, 1, 1, 1]),
            pair_view_probabilities=view_probability[[2, 0, 1, 5, 3, 4]],
            candidate_valid=valid[:, candidate_permutation],
        )

    assert torch.allclose(
        original.candidate_log_likelihood_ratios[:, candidate_permutation],
        permuted.candidate_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.allclose(
        original.measured_set_availability_logit,
        permuted.measured_set_availability_logit,
        atol=1e-6,
    )


def test_pair_forward_microbatch_preserves_values_and_backpropagates() -> None:
    torch.manual_seed(23)
    base = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    chunked = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        pair_forward_batch_size=2,
        pair_forward_gradient_checkpointing=True,
    ).eval()
    chunked.load_state_dict(base.state_dict())
    query = torch.rand((2, 3, 5, 5))
    support = torch.rand((6, 3, 5, 5))
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    candidates = torch.tensor([0, 0, 2, 0, 1, 2])
    slots = torch.tensor([0, 1, 0, 0, 0, 0])
    view_probability = torch.tensor([0.6, 0.4, 0.8, 0.5, 0.5, 1.0])
    valid = torch.ones((2, 3), dtype=torch.bool)
    kwargs = {
        "query_patches_by_group": query,
        "support_patches": support,
        "pair_group_indices": groups,
        "pair_candidate_indices": candidates,
        "pair_view_slots": slots,
        "pair_view_probabilities": view_probability,
        "candidate_valid": valid,
    }
    with torch.no_grad():
        expected = base(**kwargs)
        actual = chunked(**kwargs)
    assert torch.allclose(
        expected.candidate_log_likelihood_ratios,
        actual.candidate_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.allclose(
        expected.view_spatial_logits,
        actual.view_spatial_logits,
        atol=1e-6,
    )
    assert torch.allclose(
        expected.view_spatial_offsets_xy,
        actual.view_spatial_offsets_xy,
        atol=1e-6,
    )
    assert torch.allclose(
        expected.measured_set_availability_logit,
        actual.measured_set_availability_logit,
        atol=1e-6,
    )

    chunked.train()
    train_prediction = chunked(**kwargs)
    loss = (
        train_prediction.view_identity_logits.square().mean()
        + train_prediction.view_spatial_logits.square().mean()
        + train_prediction.measured_set_availability_logit.square().mean()
    )
    loss.backward()
    encoder_gradients = [
        parameter.grad
        for parameter in chunked.encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_gradients
    assert all(torch.all(torch.isfinite(gradient)) for gradient in encoder_gradients)


def test_layout_context_support_encoder_streams_with_pair_microbatches() -> None:
    torch.manual_seed(24)
    base = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
    ).eval()
    chunked = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
        pair_forward_batch_size=2,
        pair_forward_gradient_checkpointing=True,
    ).eval()
    chunked.load_state_dict(base.state_dict())
    kwargs = {
        "query_patches_by_group": torch.rand((2, 3, 5, 5)),
        "support_patches": torch.rand((6, 3, 5, 5)),
        "identity_context_query_patches_by_group": torch.rand((2, 3, 7, 7)),
        "identity_context_support_patches": torch.rand((6, 3, 7, 7)),
        "pair_group_indices": torch.tensor([0, 0, 0, 1, 1, 1]),
        "pair_candidate_indices": torch.tensor([0, 0, 2, 0, 1, 2]),
        "pair_view_slots": torch.tensor([0, 1, 0, 0, 0, 0]),
        "pair_view_probabilities": torch.tensor([0.6, 0.4, 0.8, 0.5, 0.5, 1.0]),
        "candidate_valid": torch.ones((2, 3), dtype=torch.bool),
        "identity_context_residual_scale": 0.0,
        "identity_context_layout_residual_scale": 1.0,
    }
    with torch.no_grad():
        expected = base(**kwargs)
        actual = chunked(**kwargs)
    assert torch.allclose(
        expected.view_identity_logits, actual.view_identity_logits, atol=1e-6
    )
    assert torch.allclose(
        expected.candidate_log_likelihood_ratios,
        actual.candidate_log_likelihood_ratios,
        atol=1e-6,
    )
    assert torch.equal(expected.view_spatial_logits, actual.view_spatial_logits)
    assert torch.equal(
        expected.view_measurement_validity_logits,
        actual.view_measurement_validity_logits,
    )

    chunked.train()
    prediction = chunked(**kwargs)
    loss = prediction.candidate_log_likelihood_ratios.square().mean()
    loss.backward()
    assert chunked.identity_context_layout_encoder is not None
    gradients = [
        parameter.grad
        for parameter in chunked.identity_context_layout_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients)


def test_pose_view_mixture_is_permutation_equivariant_and_normalized() -> None:
    logits = torch.tensor([1.0, -0.5, 0.2, 0.8])
    groups = torch.tensor([0, 0, 0, 1])
    candidates = torch.tensor([0, 0, 1, 0])
    slots = torch.tensor([0, 2, 1, 3])
    pair_probability, padded = normalize_pose_view_mixture_logits(
        logits,
        pair_group_indices=groups,
        pair_candidate_indices=candidates,
        pair_view_slots=slots,
        batch_size=2,
        candidate_count=2,
        max_views=4,
    )
    order = torch.tensor([2, 0, 3, 1])
    permuted_pair_probability, permuted_padded = (
        normalize_pose_view_mixture_logits(
            logits[order],
            pair_group_indices=groups[order],
            pair_candidate_indices=candidates[order],
            pair_view_slots=slots[order],
            batch_size=2,
            candidate_count=2,
            max_views=4,
        )
    )

    assert torch.allclose(pair_probability[order], permuted_pair_probability)
    assert torch.allclose(padded, permuted_padded)
    measured = torch.sum(padded, dim=2) > 0.0
    assert torch.allclose(
        torch.sum(padded, dim=2)[measured], torch.ones((3,))
    )
    assert torch.count_nonzero(padded[~measured]) == 0


def test_candidate_pose_view_density_learns_to_weight_the_correct_view() -> None:
    pose_logits = torch.nn.Parameter(torch.zeros((2,), dtype=torch.float32))
    optimizer = torch.optim.SGD([pose_logits], lr=0.5)
    spatial_logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
    non_dustbin_logits = torch.tensor([8.0, 8.0])
    offsets = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    groups = torch.tensor([0, 0])
    candidates = torch.tensor([0, 0])
    slots = torch.tensor([0, 1])

    losses = []
    for _ in range(30):
        probabilities, _ = normalize_pose_view_mixture_logits(
            pose_logits,
            pair_group_indices=groups,
            pair_candidate_indices=candidates,
            pair_view_slots=slots,
            batch_size=1,
            candidate_count=1,
            max_views=2,
        )
        nll, _, evaluable = gt_pose_candidate_view_mixture_nll(
            spatial_logits,
            non_dustbin_logits,
            probabilities,
            offsets,
            torch.tensor([[[0.0, 0.0]]]),
            torch.tensor([[True]]),
            torch.tensor([[True]]),
            pair_group_indices=groups,
            pair_candidate_indices=candidates,
            pair_view_slots=slots,
            max_views=2,
            target_sigma_px=0.25,
        )
        loss = nll[evaluable].mean()
        losses.append(float(loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_probability, _ = normalize_pose_view_mixture_logits(
        pose_logits.detach(),
        pair_group_indices=groups,
        pair_candidate_indices=candidates,
        pair_view_slots=slots,
        batch_size=1,
        candidate_count=1,
        max_views=2,
    )
    assert losses[-1] < losses[0]
    assert final_probability[0] > 0.9
    assert final_probability[1] < 0.1


def test_pose_view_ensemble_factorization_is_exact() -> None:
    joint_probability = torch.tensor(
        [
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            [[0.2, 0.7, 0.1], [0.4, 0.2, 0.4]],
        ],
        dtype=torch.float32,
    )
    view_probability = torch.tensor(
        [[0.8, 0.2], [0.25, 0.75]], dtype=torch.float32
    )

    effective_log_probability, mean_view_probability = (
        factorize_pose_view_ensemble(
            torch.log(joint_probability), view_probability
        )
    )
    factorized = torch.sum(
        mean_view_probability[:, None]
        * torch.exp(effective_log_probability),
        dim=0,
    )
    direct = torch.mean(
        torch.sum(
            view_probability[..., None] * joint_probability, dim=1
        ),
        dim=0,
    )

    assert torch.allclose(factorized, direct, atol=1e-7, rtol=0.0)
    assert torch.allclose(torch.sum(factorized), torch.tensor(1.0))


def test_pose_view_head_does_not_change_identity_or_spatial_predictions() -> None:
    torch.manual_seed(91)
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        pose_view_mixture_enabled=True,
    ).eval()
    inputs = {
        "query_patches_by_group": torch.rand((1, 3, 5, 5)),
        "support_patches": torch.rand((2, 3, 5, 5)),
        "pair_group_indices": torch.tensor([0, 0]),
        "pair_candidate_indices": torch.tensor([0, 0]),
        "pair_view_slots": torch.tensor([0, 1]),
        "pair_view_probabilities": torch.tensor([0.7, 0.3]),
        "candidate_valid": torch.tensor([[True]]),
    }
    with torch.no_grad():
        before = model(**inputs)
        model.view_pose_mixture_head.weight.normal_()
        model.view_pose_mixture_head.bias.fill_(2.0)
        after = model(**inputs)

    assert torch.equal(before.view_identity_logits, after.view_identity_logits)
    assert torch.equal(
        before.candidate_log_likelihood_ratios,
        after.candidate_log_likelihood_ratios,
    )
    assert torch.equal(before.view_spatial_logits, after.view_spatial_logits)
    assert not torch.allclose(
        before.view_pose_mixture_probabilities,
        after.view_pose_mixture_probabilities,
    )


def test_fp16_image_cache_keeps_sampling_grid_float32() -> None:
    cache = TensorImageLRUCache(max_bytes=4096, storage_dtype=torch.float16)
    cache["image"] = torch.rand((3, 8, 8), dtype=torch.float32)

    patch, offsets = crop_rgb_window(
        cache["image"].unsqueeze(0),
        torch.tensor([[4.0, 4.0]]),
        radius_px=1.0,
        step_px=1.0,
        image_width=8,
        image_height=8,
    )

    assert cache["image"].dtype == torch.float16
    assert patch.dtype == torch.float32
    assert offsets.dtype == torch.float32


def test_multi_owner_batched_crop_matches_individual_crops() -> None:
    torch.manual_seed(12)
    images = torch.rand((3, 3, 11, 13), dtype=torch.float16)
    owners = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long)
    centers = torch.tensor(
        [[5.0, 4.0], [3.5, 5.0], [8.0, 6.0], [6.0, 3.0], [9.0, 7.0]],
        dtype=torch.float32,
    )
    actual, offsets = crop_rgb_windows_by_owner(
        images,
        owners,
        centers,
        radius_px=1.0,
        step_px=1.0,
        image_width=13,
        image_height=11,
    )
    expected = torch.cat(
        [
            crop_rgb_window(
                images[int(owner)].unsqueeze(0),
                center.unsqueeze(0),
                radius_px=1.0,
                step_px=1.0,
                image_width=13,
                image_height=11,
            )[0]
            for owner, center in zip(owners.tolist(), centers)
        ],
        dim=0,
    )
    assert offsets.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-6, rtol=0.0)


def test_natural_binary_prior_logit_is_the_neutral_view_likelihood() -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    prior = 0.2
    prior_logit = math.log(prior) - math.log1p(-prior)
    model.set_view_identity_prior(prior, initialize_head_bias=True)
    with torch.no_grad():
        model.view_identity_head.weight.zero_()
        prediction = model(
            query_patches_by_group=torch.rand((1, 3, 5, 5)),
            support_patches=torch.rand((1, 3, 5, 5)),
            pair_group_indices=torch.tensor([0]),
            pair_candidate_indices=torch.tensor([0]),
            pair_view_slots=torch.tensor([0]),
            pair_view_probabilities=torch.tensor([0.7]),
            candidate_valid=torch.tensor([[True, True]]),
        )

    assert torch.allclose(
        prediction.view_identity_logits, torch.tensor([prior_logit]), atol=1e-6
    )
    assert torch.allclose(
        prediction.view_log_likelihood_ratios, torch.zeros((1,)), atol=1e-6
    )
    assert torch.allclose(
        prediction.candidate_log_likelihood_ratios, torch.zeros((1, 2)), atol=1e-6
    )


def test_spatial_export_requires_explicit_view_mixture_contract(tmp_path) -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    )
    config = model.config()
    assert config["support_view_mixture"] == SUPPORT_VIEW_MIXTURE_CONTRACT
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v1",
            "config": config,
            "model": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="requires verifier v3"):
        _load_model(checkpoint, torch.device("cpu"))

    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v2",
            "config": config,
            "model": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        _load_model(checkpoint, torch.device("cpu"))
    legacy = _load_model(
        checkpoint,
        torch.device("cpu"),
        allow_legacy_identity_dustbin=True,
    )
    assert "DIAGNOSTIC_ONLY" in legacy._dustbin_probability_semantics

    config["support_view_mixture"] = "uniform_views"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v2",
            "config": config,
            "model": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="contract is missing or incompatible"):
        _load_model(
            checkpoint,
            torch.device("cpu"),
            allow_legacy_identity_dustbin=True,
        )

    config = model.config()
    config["measurement_validity_semantics"] = MEASUREMENT_VALIDITY_SEMANTICS
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v3",
            "config": config,
            "training": {"measurement_success_threshold_px": 2.0},
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )
    loaded = _load_model(checkpoint, torch.device("cpu"))
    assert loaded._measurement_success_threshold_px == 2.0


def test_measurement_mode_success_uses_gt_projection_not_identity_label() -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0], [5.0, 0.0, 0.0]])
    offsets = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    target = torch.tensor([[2.5, 0.0], [3.0, 0.0]])
    residual, success = measurement_mode_residual_and_success(
        logits,
        offsets,
        target,
        torch.tensor([True, True]),
        success_threshold_px=2.0,
    )

    assert torch.allclose(residual, torch.tensor([0.5, 3.0]))
    assert success.tolist() == [True, False]


def test_measurement_validity_head_is_distinct_from_identity_head() -> None:
    torch.manual_seed(14)
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
    ).eval()
    inputs = {
        "query_patches_by_group": torch.rand((1, 3, 5, 5)),
        "support_patches": torch.rand((1, 3, 5, 5)),
        "pair_group_indices": torch.tensor([0]),
        "pair_candidate_indices": torch.tensor([0]),
        "pair_view_slots": torch.tensor([0]),
        "pair_view_probabilities": torch.tensor([1.0]),
        "candidate_valid": torch.tensor([[True]]),
    }
    with torch.no_grad():
        before = model(**inputs)
        model.view_identity_head.weight.add_(10.0)
        model.view_identity_head.bias.add_(10.0)
        after = model(**inputs)

    assert not torch.allclose(before.view_identity_logits, after.view_identity_logits)
    assert torch.allclose(
        before.view_measurement_validity_logits,
        after.view_measurement_validity_logits,
    )


def test_normalized_spatial_density_preserves_k_plus_dustbin_mass() -> None:
    spatial_logits = torch.tensor(
        [[2.0, 0.0, -1.0], [-2.0, 1.0, 3.0]], dtype=torch.float32
    )
    non_dustbin_logits = torch.tensor([1.5, -0.7], dtype=torch.float32)
    log_probability = normalized_spatial_log_probabilities_with_dustbin(
        spatial_logits, non_dustbin_logits
    )
    probability = torch.exp(log_probability)

    assert log_probability.shape == (2, 4)
    assert torch.allclose(
        torch.sum(probability, dim=1), torch.ones((2,)), atol=1e-6
    )
    assert torch.allclose(
        torch.sum(probability[:, :-1], dim=1),
        torch.sigmoid(non_dustbin_logits),
        atol=1e-6,
    )
    assert torch.allclose(
        probability[:, -1], torch.sigmoid(-non_dustbin_logits), atol=1e-6
    )


def test_gt_pose_spatial_density_uses_in_support_offset_or_dustbin() -> None:
    logits = torch.zeros((3, 9), dtype=torch.float32, requires_grad=True)
    non_dustbin = torch.tensor(
        [3.0, 3.0, -3.0], dtype=torch.float32, requires_grad=True
    )
    offsets = torch.tensor(
        [
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    targets = torch.tensor([[0.2, -0.1], [4.0, 0.0], [float("nan"), 0.0]])
    nll, inside, evaluable = gt_pose_spatial_density_nll(
        logits,
        non_dustbin,
        offsets,
        targets,
        torch.tensor([True, True, True]),
        torch.tensor([True, True, False]),
        target_sigma_px=0.5,
    )

    assert inside.tolist() == [True, False, False]
    assert evaluable.tolist() == [True, True, True]
    assert torch.isfinite(nll).all()
    nll.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert non_dustbin.grad is not None and torch.isfinite(non_dustbin.grad).all()
    assert non_dustbin.grad[0] < 0.0
    assert non_dustbin.grad[1] > 0.0


def test_spatial_export_loads_normalized_v4_checkpoint(tmp_path) -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        measurement_validity_semantics=GT_POSE_SPATIAL_DENSITY_SEMANTICS,
    )
    checkpoint = tmp_path / "normalized.pt"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v4",
            "config": model.config(),
            "training": {"spatial_density_loss_weight": 1.0},
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="spatial target sigma"):
        _load_model(checkpoint, torch.device("cpu"))

    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v4",
            "config": model.config(),
            "training": {
                "spatial_density_loss_weight": 1.0,
                "spatial_target_sigma_px": 0.75,
            },
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded = _load_model(checkpoint, torch.device("cpu"))
    assert loaded.measurement_validity_semantics == (
        GT_POSE_SPATIAL_DENSITY_SEMANTICS
    )
    assert loaded._dustbin_probability_semantics == (
        NORMALIZED_SPATIAL_DUSTBIN_SEMANTICS
    )
    assert loaded._spatial_target_sigma_px == 0.75


def test_spatial_export_loads_learned_pose_view_v5_checkpoint(tmp_path) -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=4,
        measurement_validity_semantics=GT_POSE_SPATIAL_DENSITY_SEMANTICS,
        pose_view_mixture_enabled=True,
    )
    checkpoint = tmp_path / "pose_view_v5.pt"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v5",
            "config": model.config(),
            "training": {
                "spatial_target_sigma_px": 0.75,
                "pose_view_mixture_loss_weight": 1.0,
            },
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded = _load_model(checkpoint, torch.device("cpu"))

    assert loaded.pose_view_mixture_enabled
    assert loaded.config()["pose_view_mixture_semantics"] == (
        POSE_VIEW_MIXTURE_SEMANTICS
    )


def test_spatial_export_loads_dual_scale_v6_checkpoint(tmp_path) -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
    )
    checkpoint = tmp_path / "dual_scale.pt"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v6",
            "config": model.config(),
            "training": {"measurement_success_threshold_px": 2.0},
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded = _load_model(checkpoint, torch.device("cpu"))

    assert loaded.identity_context_enabled
    assert loaded.config()["identity_context_radius_px"] == 3.0


def test_spatial_export_loads_isolated_layout_v8_checkpoint(tmp_path) -> None:
    model = IndependentRGBCandidateVerifier(
        search_radius_px=1.0,
        context_radius_px=1.0,
        step_px=1.0,
        feature_dim=8,
        hidden_dim=16,
        input_mode="rgb",
        encoder_arch="simple",
        template_scale_factors=(1.0,),
        max_views=2,
        identity_context_radius_px=3.0,
        identity_context_step_px=1.0,
        identity_context_layout_grid_size=3,
    )
    checkpoint = tmp_path / "layout_v8.pt"
    torch.save(
        {
            "format": "independent_rgb_candidate_verifier_v8",
            "config": model.config(),
            "training": {"measurement_success_threshold_px": 2.0},
            "data_contract": {"version": 1},
            "model": model.state_dict(),
        },
        checkpoint,
    )

    loaded = _load_model(checkpoint, torch.device("cpu"))

    assert loaded.identity_context_enabled
    assert loaded.identity_context_layout_enabled
    assert loaded.config()["identity_context_layout_grid_size"] == 3
