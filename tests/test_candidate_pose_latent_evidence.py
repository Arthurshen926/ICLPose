from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization.candidate_pose_llr import CandidatePoseLLRRuntime
from feature_extract.vfm.localization.candidate_pose_latent_evidence import (
    CandidatePoseLatentEvidence,
    candidate_pose_point_log_mixture,
    direct_candidate_alignment_scores,
    identity_posterior_from_residual,
    same_track_alignment_margin_loss,
    weighted_pose_log_likelihood_ratio,
)


def test_identity_posterior_preserves_fixed_nonnull_mass() -> None:
    priors = torch.tensor([[0.40, 0.20], [0.30, 0.30]], dtype=torch.float32)
    null = torch.tensor([0.40, 0.40], dtype=torch.float32)
    residual = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    candidate, returned_null, conditional = identity_posterior_from_residual(
        candidate_residual=residual,
        candidate_probabilities=priors,
        null_probabilities=null,
    )

    assert torch.allclose(returned_null, null)
    assert torch.allclose(candidate.sum(dim=1) + returned_null, torch.ones(2))
    assert torch.allclose(conditional.sum(dim=1), torch.ones(2))
    assert conditional[0, 0] > conditional[0, 1]
    assert conditional[1, 1] > conditional[1, 0]


def test_weighted_pose_log_likelihood_uses_static_selector_weights() -> None:
    point_log_ratio = torch.tensor(
        [[1.0, -10.0], [3.0, 10.0]], dtype=torch.float32
    )
    selector = torch.tensor([1.0, 0.0], dtype=torch.float32)

    pose = weighted_pose_log_likelihood_ratio(
        point_log_likelihood_ratios=point_log_ratio,
        selector_weights=selector,
    )

    assert torch.allclose(pose, torch.tensor([1.0, 3.0]))


def test_pose_mixture_uses_soft_static_identity_posterior_without_argmax() -> None:
    alignment = torch.tensor([[[1.0, -1.0]]], dtype=torch.float32)
    coarse = torch.tensor([[0.45, 0.45]], dtype=torch.float32)
    identity = torch.tensor([[0.80, 0.10]], dtype=torch.float32)
    null = torch.tensor([0.10], dtype=torch.float32)

    coarse_score = candidate_pose_point_log_mixture(
        candidate_alignment=alignment,
        candidate_probabilities=coarse,
        null_probabilities=null,
    )
    identity_score = candidate_pose_point_log_mixture(
        candidate_alignment=alignment,
        candidate_probabilities=identity,
        null_probabilities=null,
    )

    assert coarse_score.shape == (1, 1)
    assert identity_score.shape == (1, 1)
    assert identity_score.item() > coarse_score.item()


def test_same_track_alignment_margin_prefers_correct_pose() -> None:
    correct = torch.tensor([1.0, 0.5], dtype=torch.float32)
    wrong = torch.tensor([[0.0, -0.5], [0.25, 0.0]], dtype=torch.float32)

    loss, metrics = same_track_alignment_margin_loss(
        correct_alignment=correct,
        coherent_wrong_alignment=wrong,
        margin=0.25,
    )

    assert loss.item() > 0.0
    assert metrics["pair_count"] == 2.0
    assert metrics["correct_win_fraction"] == 1.0


def test_direct_candidate_alignment_uses_only_exact_topl_track_edges() -> None:
    candidate_alignment = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 3.0], [8.0, 9.0], [10.0, 11.0]],
            [[5.0, 6.0], [8.0, 7.0], [12.0, 13.0], [14.0, 15.0]],
        ],
        dtype=torch.float32,
    )
    # Candidate count is two. Class two is the fixed-null target and -1 is
    # unlabelled, so only the first two positions may supervise alignment.
    target_classes = torch.tensor([1, 0, 2, -1], dtype=torch.long)

    scores, exact = direct_candidate_alignment_scores(
        candidate_alignment=candidate_alignment,
        target_classes=target_classes,
    )

    torch.testing.assert_close(exact, torch.tensor([True, True, False, False]))
    torch.testing.assert_close(scores, torch.tensor([3.0, 7.0]))


def _runtime() -> CandidatePoseLLRRuntime:
    return CandidatePoseLLRRuntime(
        query_image_indices=torch.tensor([0, 0]),
        support_image_indices=torch.tensor([[[1], [1]], [[1], [1]]]),
        support_xy=torch.tensor(
            [
                [[[12.0, 12.0]], [[20.0, 20.0]]],
                [[[12.0, 12.0]], [[20.0, 20.0]]],
            ]
        ),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1), dtype=torch.float32),
        candidate_probabilities=torch.tensor(
            [[0.45, 0.45], [0.45, 0.45]], dtype=torch.float32
        ),
        null_probabilities=torch.tensor([0.10, 0.10], dtype=torch.float32),
    )


def test_latent_evidence_uses_current_runtime_without_target_inputs() -> None:
    torch.manual_seed(7)
    sources = {
        name: torch.nn.functional.normalize(
            torch.randn((2, 16, 16, 4), dtype=torch.float32), dim=-1
        )
        for name in ("radio_final", "radio_intermediate", "alike")
    }
    model = CandidatePoseLatentEvidence(
        sources=sources,
        image_sizes=torch.tensor([[64.0, 64.0], [64.0, 64.0]]),
        hidden_dim=4,
        edge_chunk_size=8,
    )
    runtime = _runtime()
    observed_xy = torch.tensor([[12.0, 12.0], [20.0, 20.0]])
    projected_xy = torch.tensor(
        [
            [[[12.0, 12.0], [14.0, 12.0]], [[20.0, 20.0], [22.0, 20.0]]],
            [[[13.0, 12.0], [15.0, 12.0]], [[21.0, 20.0], [23.0, 20.0]]],
        ]
    )
    projected_valid = torch.ones((2, 2, 2), dtype=torch.bool)

    identity = model.identity_posterior(runtime=runtime, observed_xy=observed_xy)
    pose = model.pose_log_likelihood_ratios(
        runtime=runtime,
        candidate_query_xy=projected_xy,
        candidate_projection_valid=projected_valid,
        selector_weights=identity.selector_weights.detach(),
    )

    assert identity.candidate_probabilities.shape == (2, 2)
    assert torch.allclose(
        identity.candidate_probabilities.sum(dim=1) + identity.null_probabilities,
        torch.ones(2),
        atol=1e-5,
    )
    assert identity.selector_weights.shape == (2,)
    assert torch.isfinite(pose.pose_log_likelihood_ratios).all()
    assert pose.candidate_log_likelihood_ratios.shape == (2, 2, 2)


def test_latent_forward_detaches_identity_from_pose_alignment_gradient() -> None:
    torch.manual_seed(11)
    sources = {
        name: torch.nn.functional.normalize(
            torch.randn((2, 16, 16, 4), dtype=torch.float32), dim=-1
        )
        for name in ("radio_final", "radio_intermediate", "alike")
    }
    model = CandidatePoseLatentEvidence(
        sources=sources,
        image_sizes=torch.tensor([[64.0, 64.0], [64.0, 64.0]]),
        hidden_dim=4,
        edge_chunk_size=8,
    )
    observed_xy = torch.tensor([[12.0, 12.0], [20.0, 20.0]])
    projected_xy = observed_xy.reshape(1, 2, 1, 2).expand(2, -1, 2, -1).clone()
    projected_valid = torch.ones((2, 2, 2), dtype=torch.bool)

    identity, pose, has_alignment_tokens = model(
        runtime=_runtime(),
        observed_xy=observed_xy,
        candidate_query_xy=projected_xy,
        candidate_projection_valid=projected_valid,
        alignment_selector_mask=torch.tensor([True, False]),
    )
    pose.pose_log_likelihood_ratios.sum().backward()

    assert bool(has_alignment_tokens)
    assert identity.selector_weights.requires_grad
    assert pose.pose_log_likelihood_ratios.requires_grad
    assert all(parameter.grad is None for parameter in model.identity_head.parameters())
    assert any(parameter.grad is not None for parameter in model.alignment_head.parameters())
