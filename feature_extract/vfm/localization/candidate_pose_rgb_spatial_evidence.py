"""Post-encoder diagnostics for candidate-specific RGB spatial evidence.

The runtime scorer intentionally receives no pose target, track label, or
residual.  This module does not change that boundary: it only summarizes
already-emitted target-free scores after an audit has joined train-only pose
projections.  Keeping the accounting here makes it possible to distinguish a
weak local edge from attenuation in the support-view, candidate/null, or
cross-point mixtures before changing a model architecture.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialScore,
)


EVIDENCE_LAYER_AUDIT_FORMAT = "candidate_pose_rgb_spatial_evidence_layers_v1"


def _masked_summary(
    *,
    values: torch.Tensor,
    active: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """Summarize a finite tensor without silently counting inactive evidence."""

    data = torch.as_tensor(values, dtype=torch.float32)
    mask = torch.as_tensor(active, dtype=torch.bool, device=data.device)
    if data.shape != mask.shape or not torch.isfinite(data).all():
        raise ValueError("candidate RGB spatial evidence summary inputs are invalid")
    count = int(mask.sum().item())
    result = {
        f"{prefix}_count": float(count),
        f"{prefix}_fraction": float(mask.to(dtype=torch.float32).mean().item()),
        f"{prefix}_mean_correct_minus_wrong": 0.0,
        f"{prefix}_correct_win_fraction": 0.0,
    }
    if count:
        selected = data[mask]
        result[f"{prefix}_mean_correct_minus_wrong"] = float(selected.mean().item())
        result[f"{prefix}_correct_win_fraction"] = float(
            (selected > 0.0).to(dtype=torch.float32).mean().item()
        )
    return result


def _validate_score_pair(
    *,
    correct: CandidatePoseRGBSpatialScore,
    wrong: CandidatePoseRGBSpatialScore,
    correct_projection_offsets_xy: torch.Tensor,
    wrong_projection_offsets_xy: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate one correct pose against one or more coherent-wrong modes."""

    correct_edge = torch.as_tensor(correct.edge_log_likelihood_ratios, dtype=torch.float32)
    wrong_edge = torch.as_tensor(wrong.edge_log_likelihood_ratios, dtype=torch.float32)
    correct_candidate = torch.as_tensor(
        correct.candidate_log_likelihood_ratios, dtype=torch.float32
    )
    wrong_candidate = torch.as_tensor(
        wrong.candidate_log_likelihood_ratios, dtype=torch.float32
    )
    correct_point = torch.as_tensor(correct.point_log_likelihood_ratios, dtype=torch.float32)
    wrong_point = torch.as_tensor(wrong.point_log_likelihood_ratios, dtype=torch.float32)
    correct_pose = torch.as_tensor(correct.pose_log_likelihood_ratios, dtype=torch.float32)
    wrong_pose = torch.as_tensor(wrong.pose_log_likelihood_ratios, dtype=torch.float32)
    correct_usable = torch.as_tensor(correct.edge_usable, dtype=torch.bool)
    wrong_usable = torch.as_tensor(wrong.edge_usable, dtype=torch.bool)
    offsets = torch.as_tensor(correct_projection_offsets_xy, dtype=torch.float32)
    wrong_offsets = torch.as_tensor(wrong_projection_offsets_xy, dtype=torch.float32)
    priors = torch.as_tensor(candidate_probabilities, dtype=torch.float32)
    null = torch.as_tensor(null_probabilities, dtype=torch.float32)
    if correct_edge.ndim != 4 or correct_edge.shape[0] != 1:
        raise ValueError("correct candidate RGB spatial score must have one pose")
    wrong_count, point_count, candidate_count, view_count = wrong_edge.shape
    expected_edge = (1, point_count, candidate_count, view_count)
    if (
        wrong_count <= 0
        or correct_edge.shape != expected_edge
        or correct_usable.shape != expected_edge
        or wrong_usable.shape != wrong_edge.shape
        or correct_candidate.shape != (1, point_count, candidate_count)
        or wrong_candidate.shape != (wrong_count, point_count, candidate_count)
        or correct_point.shape != (1, point_count)
        or wrong_point.shape != (wrong_count, point_count)
        or correct_pose.shape != (1,)
        or wrong_pose.shape != (wrong_count,)
        or offsets.shape != (point_count, candidate_count, 2)
        or wrong_offsets.shape != (wrong_count, point_count, candidate_count, 2)
        or priors.shape != (point_count, candidate_count)
        or null.shape != (point_count,)
        or not torch.isfinite(correct_edge).all()
        or not torch.isfinite(wrong_edge).all()
        or not torch.isfinite(correct_candidate).all()
        or not torch.isfinite(wrong_candidate).all()
        or not torch.isfinite(correct_point).all()
        or not torch.isfinite(wrong_point).all()
        or not torch.isfinite(correct_pose).all()
        or not torch.isfinite(wrong_pose).all()
        or not torch.isfinite(offsets).all()
        or not torch.isfinite(wrong_offsets).all()
        or not torch.isfinite(priors).all()
        or not torch.isfinite(null).all()
        or torch.any(priors < 0.0)
        or torch.any(null < 0.0)
        or torch.any(torch.abs(priors.sum(dim=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("candidate RGB spatial evidence layers are incompatible")
    point_gap = correct_point.expand(wrong_count, -1) - wrong_point
    pose_gap = correct_pose.expand(wrong_count) - wrong_pose
    if not torch.allclose(pose_gap, point_gap.mean(dim=1), atol=2e-5, rtol=2e-5):
        raise ValueError("candidate RGB spatial pose score does not equal its point mixture")
    return wrong_count, point_count, candidate_count, view_count


def _candidate_mixture_without_null(
    *,
    candidate_log_likelihood_ratios: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    uniform: bool,
) -> torch.Tensor:
    """Counterfactual candidate mixture used only to diagnose attenuation."""

    candidate = torch.as_tensor(candidate_log_likelihood_ratios, dtype=torch.float32)
    priors = torch.as_tensor(candidate_probabilities, dtype=torch.float32, device=candidate.device)
    if candidate.ndim != 3 or priors.shape != candidate.shape[1:]:
        raise ValueError("candidate RGB spatial counterfactual mixture inputs are invalid")
    positive = priors > 0.0
    if not bool(torch.all(torch.any(positive, dim=1))):
        raise ValueError("candidate RGB spatial counterfactual has no candidate mass")
    if uniform:
        weights = positive.to(dtype=candidate.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
    else:
        weights = priors / priors.sum(dim=1, keepdim=True).clamp_min(torch.finfo(priors.dtype).tiny)
    safe_log_weights = torch.where(
        weights > 0.0,
        torch.log(weights),
        torch.full_like(weights, -torch.inf),
    )
    return torch.logsumexp(candidate + safe_log_weights.unsqueeze(0), dim=2)


def _pose_summary(*, correct_points: torch.Tensor, wrong_points: torch.Tensor, prefix: str) -> dict[str, float]:
    """Report correct-minus-wrong scores after one fixed point aggregation."""

    correct = torch.as_tensor(correct_points, dtype=torch.float32)
    wrong = torch.as_tensor(wrong_points, dtype=torch.float32, device=correct.device)
    if correct.ndim != 2 or correct.shape[0] != 1 or wrong.ndim != 2 or wrong.shape[1] != correct.shape[1]:
        raise ValueError("candidate RGB spatial pose counterfactual inputs are invalid")
    gaps = correct.expand_as(wrong).mean(dim=1) - wrong.mean(dim=1)
    return {
        f"{prefix}_pose_mean_correct_minus_wrong": float(gaps.mean().item()),
        f"{prefix}_pose_correct_win_fraction": float(
            (gaps > 0.0).to(dtype=torch.float32).mean().item()
        ),
        f"{prefix}_hardest_coherent_wrong_correct_minus_wrong": float(gaps.min().item()),
        f"{prefix}_hardest_coherent_wrong_correct_win": float(
            (gaps.min() > 0.0).to(dtype=torch.float32).item()
        ),
    }


def summarize_candidate_pose_rgb_spatial_evidence_layers(
    *,
    correct: CandidatePoseRGBSpatialScore,
    wrong: CandidatePoseRGBSpatialScore,
    correct_projection_offsets_xy: torch.Tensor,
    wrong_projection_offsets_xy: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
) -> dict[str, float]:
    """Decompose a correct-vs-wrong score without accessing target labels.

    Inputs are target-free prediction/scoring outputs plus the projection
    offsets supplied *after* a visual forward pass by a train-only audit.  The
    no-null, uniform, top-prior, and max-candidate values are counterfactual
    diagnostics only; none is a runtime scoring policy or a promotion result.
    """

    wrong_count, point_count, candidate_count, view_count = _validate_score_pair(
        correct=correct,
        wrong=wrong,
        correct_projection_offsets_xy=correct_projection_offsets_xy,
        wrong_projection_offsets_xy=wrong_projection_offsets_xy,
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
    )
    del null_probabilities  # Its normalization was checked above; current score already includes it.
    correct_edge = torch.as_tensor(correct.edge_log_likelihood_ratios, dtype=torch.float32)
    wrong_edge = torch.as_tensor(
        wrong.edge_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )
    correct_usable = torch.as_tensor(correct.edge_usable, dtype=torch.bool, device=correct_edge.device)
    wrong_usable = torch.as_tensor(wrong.edge_usable, dtype=torch.bool, device=correct_edge.device)
    correct_candidate = torch.as_tensor(
        correct.candidate_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )
    wrong_candidate = torch.as_tensor(
        wrong.candidate_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )
    correct_point = torch.as_tensor(
        correct.point_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )
    wrong_point = torch.as_tensor(
        wrong.point_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )
    correct_offsets = torch.as_tensor(
        correct_projection_offsets_xy, dtype=torch.float32, device=correct_edge.device
    )
    wrong_offsets = torch.as_tensor(
        wrong_projection_offsets_xy, dtype=torch.float32, device=correct_edge.device
    )
    priors = torch.as_tensor(candidate_probabilities, dtype=torch.float32, device=correct_edge.device)

    edge_gap = correct_edge.expand_as(wrong_edge) - wrong_edge
    edge_active = correct_usable.expand_as(wrong_usable) & wrong_usable
    candidate_gap = correct_candidate.expand_as(wrong_candidate) - wrong_candidate
    candidate_active = torch.any(edge_active, dim=3)
    point_gap = correct_point.expand_as(wrong_point) - wrong_point
    point_active = torch.any(candidate_active, dim=2)
    pose_gap = torch.as_tensor(
        correct.pose_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    ).expand(wrong_count) - torch.as_tensor(
        wrong.pose_log_likelihood_ratios, dtype=torch.float32, device=correct_edge.device
    )

    result: dict[str, float] = {
        "wrong_mode_count": float(wrong_count),
        "point_count": float(point_count),
        "candidate_count": float(candidate_count),
        "view_count": float(view_count),
        "fixed_mixture_pose_mean_correct_minus_wrong": float(pose_gap.mean().item()),
        "fixed_mixture_pose_correct_win_fraction": float(
            (pose_gap > 0.0).to(dtype=torch.float32).mean().item()
        ),
        "fixed_mixture_hardest_coherent_wrong_correct_minus_wrong": float(
            pose_gap.min().item()
        ),
        "fixed_mixture_hardest_coherent_wrong_correct_win": float(
            (pose_gap.min() > 0.0).to(dtype=torch.float32).item()
        ),
    }
    result.update(_masked_summary(values=edge_gap, active=edge_active, prefix="edge"))
    result.update(
        _masked_summary(values=candidate_gap, active=candidate_active, prefix="candidate_view_mixture")
    )
    result.update(_masked_summary(values=point_gap, active=point_active, prefix="point_fixed_mixture"))
    result.update(
        _masked_summary(
            values=point_gap,
            active=torch.ones_like(point_active),
            prefix="point_including_fixed_missing",
        )
    )

    displacement = torch.linalg.vector_norm(
        correct_offsets.unsqueeze(0) - wrong_offsets,
        dim=-1,
    )
    edge_displacement = displacement.unsqueeze(-1).expand_as(edge_gap)
    if bool(edge_active.any()):
        selected_displacement = edge_displacement[edge_active]
        result["edge_usable_projection_displacement_mean_px"] = float(
            selected_displacement.mean().item()
        )
        result["edge_usable_projection_displacement_median_px"] = float(
            selected_displacement.median().item()
        )
    else:
        result["edge_usable_projection_displacement_mean_px"] = 0.0
        result["edge_usable_projection_displacement_median_px"] = 0.0
    buckets = (
        (0.0, 0.5, "0_to_0p5"),
        (0.5, 1.0, "0p5_to_1"),
        (1.0, 2.0, "1_to_2"),
        (2.0, 4.0, "2_to_4"),
        (4.0, math.inf, "4_plus"),
    )
    for minimum, maximum, name in buckets:
        bucket_active = edge_active & (edge_displacement >= minimum) & (edge_displacement < maximum)
        result.update(
            _masked_summary(values=edge_gap, active=bucket_active, prefix=f"edge_displacement_{name}")
        )

    prior_no_null_correct = _candidate_mixture_without_null(
        candidate_log_likelihood_ratios=correct_candidate,
        candidate_probabilities=priors,
        uniform=False,
    )
    prior_no_null_wrong = _candidate_mixture_without_null(
        candidate_log_likelihood_ratios=wrong_candidate,
        candidate_probabilities=priors,
        uniform=False,
    )
    result.update(
        _pose_summary(
            correct_points=prior_no_null_correct,
            wrong_points=prior_no_null_wrong,
            prefix="counterfactual_prior_no_null",
        )
    )
    uniform_no_null_correct = _candidate_mixture_without_null(
        candidate_log_likelihood_ratios=correct_candidate,
        candidate_probabilities=priors,
        uniform=True,
    )
    uniform_no_null_wrong = _candidate_mixture_without_null(
        candidate_log_likelihood_ratios=wrong_candidate,
        candidate_probabilities=priors,
        uniform=True,
    )
    result.update(
        _pose_summary(
            correct_points=uniform_no_null_correct,
            wrong_points=uniform_no_null_wrong,
            prefix="counterfactual_uniform_no_null",
        )
    )
    top_prior_indices = torch.argmax(priors, dim=1)
    correct_top_prior = correct_candidate.gather(
        2, top_prior_indices.reshape(1, point_count, 1)
    ).squeeze(2)
    wrong_top_prior = wrong_candidate.gather(
        2,
        top_prior_indices.reshape(1, point_count, 1).expand(wrong_count, -1, -1),
    ).squeeze(2)
    result.update(
        _pose_summary(
            correct_points=correct_top_prior,
            wrong_points=wrong_top_prior,
            prefix="counterfactual_top_prior_candidate",
        )
    )
    positive = priors > 0.0
    masked_correct_candidate = correct_candidate.masked_fill(
        ~positive.unsqueeze(0), -torch.inf
    ).amax(dim=2)
    masked_wrong_candidate = wrong_candidate.masked_fill(
        ~positive.unsqueeze(0), -torch.inf
    ).amax(dim=2)
    result.update(
        _pose_summary(
            correct_points=masked_correct_candidate,
            wrong_points=masked_wrong_candidate,
            prefix="counterfactual_max_candidate",
        )
    )
    return result


def summarize_registered_observation_candidate_oracle(
    *,
    correct: CandidatePoseRGBSpatialScore,
    wrong: CandidatePoseRGBSpatialScore,
    observed_candidate_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize registered observed-track rows after an audit-only target join.

    This is an oracle diagnostic, not an inference policy.  It must only be
    called after the model has emitted target-free scores, and no value returned
    here may be fed back into a runtime scorer or checkpoint.
    """

    correct_candidate = torch.as_tensor(
        correct.candidate_log_likelihood_ratios, dtype=torch.float32
    )
    wrong_candidate = torch.as_tensor(
        wrong.candidate_log_likelihood_ratios,
        dtype=torch.float32,
        device=correct_candidate.device,
    )
    correct_usable = torch.as_tensor(correct.edge_usable, dtype=torch.bool, device=correct_candidate.device)
    wrong_usable = torch.as_tensor(wrong.edge_usable, dtype=torch.bool, device=correct_candidate.device)
    mask = torch.as_tensor(observed_candidate_mask, dtype=torch.bool, device=correct_candidate.device)
    if (
        correct_candidate.ndim != 3
        or correct_candidate.shape[0] != 1
        or wrong_candidate.ndim != 3
        or wrong_candidate.shape[1:] != correct_candidate.shape[1:]
        or correct_usable.shape[:3] != correct_candidate.shape
        or wrong_usable.shape[:3] != wrong_candidate.shape
        or mask.shape != correct_candidate.shape[1:]
        or torch.any(mask.sum(dim=1) > 1)
    ):
        raise ValueError("registered observation oracle inputs are invalid")
    wrong_count, point_count, _candidate_count = wrong_candidate.shape
    candidate_gap = correct_candidate.expand_as(wrong_candidate) - wrong_candidate
    candidate_active = torch.any(
        correct_usable.expand_as(wrong_usable) & wrong_usable,
        dim=3,
    )
    selected = mask.unsqueeze(0).expand(wrong_count, -1, -1)
    active = selected & candidate_active
    result = {
        "registered_observation_oracle_point_count": float(mask.any(dim=1).sum().item()),
    }
    result.update(
        _masked_summary(
            values=candidate_gap,
            active=active,
            prefix="registered_observation_oracle_candidate_view_mixture",
        )
    )
    selected_point_gap = (candidate_gap * mask.unsqueeze(0).to(dtype=candidate_gap.dtype)).sum(dim=2)
    selected_point_active = torch.any(active, dim=2)
    result.update(
        _masked_summary(
            values=selected_point_gap,
            active=selected_point_active,
            prefix="registered_observation_oracle_point",
        )
    )
    mode_scores: list[torch.Tensor] = []
    for mode in range(wrong_count):
        active_points = selected_point_active[mode]
        if bool(active_points.any()):
            mode_scores.append(selected_point_gap[mode, active_points].mean())
    if mode_scores:
        values = torch.stack(mode_scores)
        result["registered_observation_oracle_pose_mean_correct_minus_wrong"] = float(
            values.mean().item()
        )
        result["registered_observation_oracle_pose_correct_win_fraction"] = float(
            (values > 0.0).to(dtype=torch.float32).mean().item()
        )
        result[
            "registered_observation_oracle_hardest_coherent_wrong_correct_minus_wrong"
        ] = float(values.min().item())
        result[
            "registered_observation_oracle_hardest_coherent_wrong_correct_win"
        ] = float((values.min() > 0.0).to(dtype=torch.float32).item())
        result["registered_observation_oracle_pose_active_mode_count"] = float(len(values))
    else:
        result["registered_observation_oracle_pose_mean_correct_minus_wrong"] = 0.0
        result["registered_observation_oracle_pose_correct_win_fraction"] = 0.0
        result[
            "registered_observation_oracle_hardest_coherent_wrong_correct_minus_wrong"
        ] = 0.0
        result[
            "registered_observation_oracle_hardest_coherent_wrong_correct_win"
        ] = 0.0
        result["registered_observation_oracle_pose_active_mode_count"] = 0.0
    return result
