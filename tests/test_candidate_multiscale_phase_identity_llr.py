from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityLLR,
    CandidateMultiscalePhaseIdentityPrediction,
    PhaseIdentitySourceConfig,
    candidate_phase_identity_log_likelihood_ratios,
    canonical_registered_identity_or_null_targets,
    current_hard_repeat_identity_margin_loss,
    exact_identity_or_null_cross_entropy,
    phase_identity_point_block_derangement_shift,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    phase_identity_static_hard_gate_decision,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[32.0, 32.0]]),
        support_image_indices=torch.tensor([[[1], [2]]]),
        support_xy=torch.tensor([[[[32.0, 32.0]], [[32.0, 32.0]]]]),
        support_view_valid=torch.ones((1, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((1, 2, 1)),
        candidate_probabilities=torch.tensor([[0.55, 0.35]]),
        null_probabilities=torch.tensor([0.10]),
    )


def _sources() -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    values = {
        name: F.normalize(torch.randn(3, 8, 8, 8), dim=-1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    # The two support images intentionally have distinct full 2-D layouts so
    # the support appearance permutation control cannot be a no-op.
    values["radio_final"][2] = F.normalize(values["radio_final"][2] + 0.4, dim=-1)
    return values


def _configs() -> dict[str, PhaseIdentitySourceConfig]:
    return {
        name: PhaseIdentitySourceConfig(name=name, window_size=3, shift_radius=1, region_bins=1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }


def _prediction(values: torch.Tensor, usable: torch.Tensor | None = None) -> CandidateMultiscalePhaseIdentityPrediction:
    edge_usable = torch.ones_like(values, dtype=torch.bool) if usable is None else usable
    source_llrs = {name: values.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES}
    source_usable = {name: edge_usable.clone() for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES}
    return CandidateMultiscalePhaseIdentityPrediction(
        source_edge_log_likelihood_ratios=source_llrs,
        source_edge_usable=source_usable,
        # An unavailable support view is neutral evidence.  Its fixed maplet
        # mass is preserved downstream by candidate_phase_identity_log_likelihood_ratios.
        edge_log_likelihood_ratios=torch.where(edge_usable, values, torch.zeros_like(values)),
        edge_usable=edge_usable,
        source_weights={name: 1.0 for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES},
    )


def test_phase_identity_forward_is_target_free_and_support_permutation_changes_only_appearance() -> None:
    model = CandidateMultiscalePhaseIdentityLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        source_configs=_configs(),
        hidden_dim=12,
        source_storage_dtype=torch.float32,
    ).eval()
    runtime = _runtime()
    normal = model(runtime=runtime)
    permuted = model(runtime=runtime, support_permutation_shift=1)
    assert normal.edge_log_likelihood_ratios.shape == (1, 2, 1)
    assert normal.edge_usable.shape == (1, 2, 1)
    assert set(normal.source_edge_log_likelihood_ratios) == set(
        CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    )
    assert torch.isfinite(normal.edge_log_likelihood_ratios).all()
    assert not torch.allclose(
        normal.edge_log_likelihood_ratios,
        permuted.edge_log_likelihood_ratios,
    )


def test_zero_appearance_is_exactly_neutral_and_has_no_hidden_geometry_score() -> None:
    model = CandidateMultiscalePhaseIdentityLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        source_configs=_configs(),
        hidden_dim=12,
        source_storage_dtype=torch.float32,
    ).eval()
    prediction = model(runtime=_runtime(), zero_appearance=True)
    assert torch.equal(prediction.edge_log_likelihood_ratios, torch.zeros((1, 2, 1)))
    for source in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        assert torch.equal(
            prediction.source_edge_log_likelihood_ratios[source],
            torch.zeros((1, 2, 1)),
        )


def test_zero_weight_phase_sources_are_explicit_neutral_without_field_evaluation() -> None:
    model = CandidateMultiscalePhaseIdentityLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        source_configs=_configs(),
        source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
        hidden_dim=12,
        source_storage_dtype=torch.float32,
    ).eval()
    prediction = model(runtime=_runtime())
    for source in ("radio_intermediate", "alike"):
        assert torch.equal(
            prediction.source_edge_log_likelihood_ratios[source],
            torch.zeros((1, 2, 1)),
        )
        assert not bool(prediction.source_edge_usable[source].any())
    assert torch.equal(
        prediction.edge_usable,
        prediction.source_edge_usable["radio_final"],
    )


def test_radio_final_only_model_does_not_require_inactive_descriptor_grids() -> None:
    model = CandidateMultiscalePhaseIdentityLLR(
        sources={"radio_final": _sources()["radio_final"]},
        image_sizes=torch.tensor([[64.0, 64.0]] * 3),
        source_configs=_configs(),
        source_weights={"radio_final": 1.0, "radio_intermediate": 0.0, "alike": 0.0},
        hidden_dim=12,
        source_storage_dtype=torch.float32,
    ).eval()
    prediction = model(runtime=_runtime())
    assert prediction.edge_log_likelihood_ratios.shape == (1, 2, 1)
    with pytest.raises(ValueError, match="configuration"):
        CandidateMultiscalePhaseIdentityLLR(
            sources={"radio_final": _sources()["radio_final"]},
            image_sizes=torch.tensor([[64.0, 64.0]] * 3),
            source_configs=_configs(),
            source_weights={"radio_final": 0.0, "radio_intermediate": 1.0, "alike": 0.0},
            hidden_dim=12,
            source_storage_dtype=torch.float32,
        )


def test_fixed_support_view_mass_is_not_reassigned_when_a_view_is_unavailable() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0]),
        query_xy=torch.tensor([[32.0, 32.0]]),
        support_image_indices=torch.tensor([[[1, 2]]]),
        support_xy=torch.tensor([[[[32.0, 32.0], [32.0, 32.0]]]]),
        support_view_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        candidate_view_weights=torch.tensor([[[0.25, 0.75]]]),
        candidate_probabilities=torch.tensor([[0.9]]),
        null_probabilities=torch.tensor([0.1]),
    )
    prediction = _prediction(
        torch.tensor([[[2.0, 5.0]]]),
        usable=torch.tensor([[[True, False]]]),
    )
    values, usable = candidate_phase_identity_log_likelihood_ratios(
        runtime=runtime, prediction=prediction
    )
    expected = torch.log(0.25 * torch.exp(torch.tensor(2.0)) + 0.75)
    assert bool(usable[0, 0])
    assert torch.allclose(values[0, 0], expected, atol=1e-6, rtol=1e-6)


def test_exact_identity_loss_rejects_geometric_multi_candidate_semantics() -> None:
    runtime = _runtime()
    prediction = _prediction(torch.zeros((1, 2, 1)))
    with pytest.raises(ValueError, match="exact target contract"):
        exact_identity_or_null_cross_entropy(
            runtime=runtime,
            prediction=prediction,
            observed_candidate_mask=torch.tensor([[True, True]]),
            target_dustbin=torch.tensor([False]),
            target_supervised=torch.tensor([True]),
        )


def test_exact_identity_loss_accepts_one_registered_candidate_or_explicit_null() -> None:
    runtime = _runtime()
    prediction = _prediction(torch.tensor([[[0.4], [-0.2]]]))
    loss, metrics = exact_identity_or_null_cross_entropy(
        runtime=runtime,
        prediction=prediction,
        observed_candidate_mask=torch.tensor([[True, False]]),
        target_dustbin=torch.tensor([False]),
        target_supervised=torch.tensor([True]),
    )
    assert torch.isfinite(loss)
    assert metrics["identity_active"] == 1.0


def test_registered_identity_target_canonicalization_rejects_partial_geometry_rows() -> None:
    observed, dustbin, supervised = canonical_registered_identity_or_null_targets(
        observed_candidate_mask=torch.tensor([[True, False], [False, False]]),
        candidate_dustbin_mask=torch.tensor([[False, True], [True, True]]),
        candidate_supervised_mask=torch.ones((2, 2), dtype=torch.bool),
    )
    assert observed.tolist() == [[True, False], [False, False]]
    assert dustbin.tolist() == [False, True]
    assert supervised.tolist() == [True, True]
    with pytest.raises(ValueError, match="complete candidate row"):
        canonical_registered_identity_or_null_targets(
            observed_candidate_mask=torch.tensor([[True, False]]),
            candidate_dustbin_mask=torch.tensor([[False, False]]),
            candidate_supervised_mask=torch.tensor([[True, False]]),
        )


def test_current_hard_repeat_margin_uses_identity_candidates_not_pose_offsets() -> None:
    runtime = _runtime()
    prediction = _prediction(torch.tensor([[[1.5], [-0.5]]]))
    loss, metrics = current_hard_repeat_identity_margin_loss(
        runtime=runtime,
        prediction=prediction,
        point_indices=torch.tensor([0]),
        positive_candidate_indices=torch.tensor([0]),
        negative_candidate_indices=torch.tensor([1]),
        margin=0.25,
    )
    assert torch.isfinite(loss)
    assert metrics["hard_repeat_active"] == 1.0
    assert metrics["hard_repeat_mean_gap"] == pytest.approx(2.0)
    assert metrics["hard_repeat_win_fraction"] == 1.0


def test_static_hard_gate_requires_visual_permutation_delta() -> None:
    common = {
        "eligible_query_fraction": 1.0,
        "normal_win_fraction": 0.70,
        "normal_mean_positive_minus_negative": 0.12,
        "permuted_mean_positive_minus_negative": 0.01,
    }
    accepted = phase_identity_static_hard_gate_decision(
        common,
        minimum_eligible_query_fraction=0.90,
        minimum_win_fraction=0.55,
        minimum_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert accepted["passed"] is True
    rejected = phase_identity_static_hard_gate_decision(
        {**common, "permuted_mean_positive_minus_negative": 0.10},
        minimum_eligible_query_fraction=0.90,
        minimum_win_fraction=0.55,
        minimum_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert rejected["passed"] is False


def test_point_block_derangement_uses_a_distant_nontrivial_shift() -> None:
    assert phase_identity_point_block_derangement_shift(point_count=8, shift=1) == 4
    assert phase_identity_point_block_derangement_shift(point_count=8, shift=2) == 5
    with pytest.raises(ValueError, match="two points"):
        phase_identity_point_block_derangement_shift(point_count=1, shift=1)
