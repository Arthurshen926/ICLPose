"""Losses for localizable feature selection."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def track_consistency_loss(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """Mean squared distance for observations of the same 3D track."""

    if features_a.shape != features_b.shape:
        raise ValueError("features_a and features_b must have the same shape")
    return torch.mean((features_a - features_b) ** 2)


def hard_negative_contrastive_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE loss with one positive and multiple score-hard negatives."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if query.shape != positive.shape:
        raise ValueError("query and positive must have the same shape")
    if negatives.ndim != 3 or negatives.shape[0] != query.shape[0] or negatives.shape[2] != query.shape[1]:
        raise ValueError("negatives must have shape (B, N, D)")

    query_n = F.normalize(query, dim=-1, eps=1e-6)
    positive_n = F.normalize(positive, dim=-1, eps=1e-6)
    negative_n = F.normalize(negatives, dim=-1, eps=1e-6)
    pos_logits = torch.sum(query_n * positive_n, dim=-1, keepdim=True)
    neg_logits = torch.einsum("bd,bnd->bn", query_n, negative_n)
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, labels)


def listwise_pose_rank_loss(
    scores: torch.Tensor,
    costs: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """KL loss from scores to a soft target favoring low pose costs."""

    if scores.shape != costs.shape:
        raise ValueError("scores and costs must have the same shape")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    target = torch.softmax(-costs / temperature, dim=-1)
    log_probs = torch.log_softmax(scores, dim=-1)
    return F.kl_div(log_probs, target, reduction="batchmean")


def basin_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def group_sparsity_loss(gates: torch.Tensor) -> torch.Tensor:
    """L1 penalty for selector group gates."""

    return torch.mean(torch.abs(gates))


def uncertainty_calibration_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    uncertainty: torch.Tensor,
) -> torch.Tensor:
    """Encourage uncertainty to match absolute confidence error."""

    if probabilities.shape != labels.shape or probabilities.shape != uncertainty.shape:
        raise ValueError("probabilities, labels, and uncertainty must have the same shape")
    target_uncertainty = torch.abs(probabilities.detach() - labels.float())
    return F.mse_loss(uncertainty, target_uncertainty)
