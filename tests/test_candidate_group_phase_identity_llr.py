from __future__ import annotations

import inspect

import torch
from torch.nn import functional as F

from feature_extract.vfm.localization.candidate_group_phase_identity_llr import (
    CandidateGroupPhaseIdentityLLR,
    candidate_group_identity_plus_null_logits,
    candidate_group_identity_probabilities,
    current_group_hard_repeat_identity_margin_loss,
    exact_group_identity_or_null_cross_entropy,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    PhaseIdentitySourceConfig,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
)


def _runtime() -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.tensor([[24.0, 24.0], [28.0, 28.0]]),
        support_image_indices=torch.tensor(
            [
                [[2], [3], [4]],
                [[3], [4], [5]],
            ]
        ),
        support_xy=torch.tensor(
            [
                [[[25.0, 24.0]], [[26.0, 25.0]], [[27.0, 24.0]]],
                [[[29.0, 28.0]], [[28.0, 27.0]], [[30.0, 29.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 3, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 3, 1)),
        candidate_probabilities=torch.tensor([[0.45, 0.30, 0.15], [0.20, 0.25, 0.45]]),
        null_probabilities=torch.tensor([0.10, 0.10]),
    )


def _sources() -> dict[str, torch.Tensor]:
    torch.manual_seed(101)
    sources = {
        name: F.normalize(torch.randn(6, 8, 8, 12), dim=-1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    # Make every support image visibly distinct so a point-block support
    # derangement cannot accidentally be an identity-preserving no-op.
    for name, values in sources.items():
        offsets = torch.linspace(-0.25, 0.25, len(values)).view(-1, 1, 1, 1)
        sources[name] = F.normalize(values + offsets, dim=-1)
    return sources


def _configs() -> dict[str, PhaseIdentitySourceConfig]:
    return {
        name: PhaseIdentitySourceConfig(name=name, window_size=3, shift_radius=1, region_bins=1)
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }


def _model() -> CandidateGroupPhaseIdentityLLR:
    torch.manual_seed(102)
    return CandidateGroupPhaseIdentityLLR(
        sources=_sources(),
        image_sizes=torch.tensor([[64.0, 64.0]] * 6),
        source_configs=_configs(),
        source_embedding_dim=8,
        group_embedding_dim=12,
        source_storage_dtype=torch.float32,
    )


def _permuted_candidates(runtime: CandidatePoseRGBSpatialRuntime, permutation: torch.Tensor) -> CandidatePoseRGBSpatialRuntime:
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=runtime.query_image_indices,
        query_xy=runtime.query_xy,
        support_image_indices=runtime.support_image_indices.index_select(1, permutation),
        support_xy=runtime.support_xy.index_select(1, permutation),
        support_view_valid=runtime.support_view_valid.index_select(1, permutation),
        candidate_view_weights=runtime.candidate_view_weights.index_select(1, permutation),
        candidate_probabilities=runtime.candidate_probabilities.index_select(1, permutation),
        null_probabilities=runtime.null_probabilities,
    )


def test_visual_forward_has_no_pose_or_target_arguments() -> None:
    names = set(inspect.signature(CandidateGroupPhaseIdentityLLR.forward).parameters)
    assert {"runtime", "support_permutation_shift", "zero_appearance"}.issubset(names)
    assert not {"pose", "projection_offset", "residual", "track_id", "candidate_rank", "label"} & names


def test_candidate_order_permutation_is_exactly_equivariant() -> None:
    model = _model().eval()
    runtime = _runtime()
    permutation = torch.tensor([2, 0, 1])
    normal = model(runtime=runtime)
    permuted = model(runtime=_permuted_candidates(runtime, permutation))
    assert torch.allclose(
        permuted.candidate_log_likelihood_ratios,
        normal.candidate_log_likelihood_ratios.index_select(1, permutation),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(
        permuted.candidate_usable,
        normal.candidate_usable.index_select(1, permutation),
    )
    assert torch.allclose(permuted.null_log_likelihood_ratios, normal.null_log_likelihood_ratios)


def test_zero_appearance_is_an_exact_fixed_prior_noop() -> None:
    model = _model().eval()
    runtime = _runtime()
    prediction = model(runtime=runtime, zero_appearance=True)
    candidate, null = candidate_group_identity_probabilities(runtime=runtime, prediction=prediction)
    assert torch.allclose(candidate, runtime.candidate_probabilities, atol=1e-6, rtol=1e-6)
    assert torch.allclose(null, runtime.null_probabilities, atol=1e-6, rtol=1e-6)
    logits, _ = candidate_group_identity_plus_null_logits(runtime=runtime, prediction=prediction)
    assert torch.isfinite(logits).all()


def test_fresh_group_head_starts_as_an_exact_fixed_prior_noop() -> None:
    model = _model().eval()
    runtime = _runtime()
    prediction = model(runtime=runtime)
    candidate, null = candidate_group_identity_probabilities(runtime=runtime, prediction=prediction)
    assert torch.allclose(prediction.candidate_log_likelihood_ratios, torch.zeros_like(prediction.candidate_log_likelihood_ratios))
    assert torch.allclose(prediction.null_log_likelihood_ratios, torch.zeros_like(prediction.null_log_likelihood_ratios))
    assert torch.allclose(candidate, runtime.candidate_probabilities, atol=1e-6, rtol=1e-6)
    assert torch.allclose(null, runtime.null_probabilities, atol=1e-6, rtol=1e-6)
    assert not any(parameter.requires_grad for parameter in model.null_head.parameters())


def test_support_derangement_changes_appearance_without_changing_layout() -> None:
    model = _model().eval()
    # Fresh models intentionally emit an exact zero LLR.  Give the final
    # projection a deterministic nonzero trained-like value so this control
    # tests the visual path rather than initialization neutrality.
    with torch.no_grad():
        model.candidate_head[-1].weight.fill_(0.05)
    runtime = _runtime()
    normal = model(runtime=runtime)
    deranged = model(runtime=runtime, support_permutation_shift=1)
    assert normal.candidate_log_likelihood_ratios.shape == deranged.candidate_log_likelihood_ratios.shape
    assert torch.equal(normal.candidate_usable, deranged.candidate_usable)
    assert not torch.allclose(
        normal.candidate_log_likelihood_ratios, deranged.candidate_log_likelihood_ratios
    )


def test_exact_identity_and_hard_repeat_losses_backpropagate_after_visual_forward() -> None:
    model = _model().train()
    runtime = _runtime()
    prediction = model(runtime=runtime)
    observed = torch.tensor([[True, False, False], [False, False, True]])
    dustbin = ~observed
    supervised = torch.ones_like(observed, dtype=torch.bool)
    identity_loss, identity_metrics = exact_group_identity_or_null_cross_entropy(
        runtime=runtime,
        prediction=prediction,
        observed_candidate_mask=observed,
        candidate_dustbin_mask=dustbin,
        candidate_supervised_mask=supervised,
    )
    hard_loss, hard_metrics = current_group_hard_repeat_identity_margin_loss(
        runtime=runtime,
        prediction=prediction,
        point_indices=torch.tensor([0, 1]),
        positive_candidate_indices=torch.tensor([0, 2]),
        negative_candidate_indices=torch.tensor([1, 0]),
        margin=0.25,
    )
    (identity_loss + hard_loss).backward()
    assert identity_metrics["identity_active"] == 2.0
    assert hard_metrics["hard_repeat_active"] == 2.0
    assert model.candidate_head[-1].weight.grad is not None
    assert torch.isfinite(model.candidate_head[-1].weight.grad).all()
