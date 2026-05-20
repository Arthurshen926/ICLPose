"""Losses for POFD-FS pose-hypothesis ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_logits(logits: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return logits
    return logits.masked_fill(~valid_mask.bool(), torch.finfo(logits.dtype).min / 4.0)


def pose_distance_soft_rank_loss(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    temperature_m: float = 0.05,
    score_temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Listwise KL loss from continuous pose-distance soft labels."""
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    cost_logits = (-pose_cost_m.float() / max(float(temperature_m), 1.0e-6)).masked_fill(~valid, -1.0e9)
    target = torch.softmax(cost_logits, dim=1)
    pred_logp = torch.log_softmax(_masked_logits(scores.float() / max(float(score_temperature), 1.0e-6), valid), dim=1)
    loss_per = -(target * pred_logp).sum(dim=1)
    active = valid.any(dim=1)
    loss = loss_per[active].mean() if active.any() else scores.sum() * 0.0
    entropy = -(target * target.clamp_min(1.0e-9).log()).sum(dim=1)
    return loss, {
        "rank_target_entropy": entropy[active].mean() if active.any() else scores.new_zeros(()),
        "rank_active_frac": active.float().mean(),
    }


def basin_bce_loss(
    scores: torch.Tensor,
    basin_label: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """Binary candidate-basin classification loss from scorer logits."""
    if scores.shape != basin_label.shape:
        raise ValueError("scores and basin_label must have the same shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    if not valid.any():
        return scores.sum() * 0.0
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = scores.new_tensor(float(pos_weight))
    loss = F.binary_cross_entropy_with_logits(scores.float(), basin_label.float(), reduction="none", **kwargs)
    return loss[valid].mean()


def online_score_hard_negative_loss(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    cost_gap_m: float = 0.12,
    margin: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize the highest-scoring candidate that is geometrically worse."""
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    losses = []
    active_flags = []
    for row_scores, row_cost, row_valid in zip(scores.float(), pose_cost_m.float(), valid):
        if not row_valid.any():
            active_flags.append(row_scores.new_tensor(0.0))
            continue
        valid_cost = row_cost.masked_fill(~row_valid, float("inf"))
        pos_idx = int(valid_cost.argmin())
        hard_valid = row_valid & (row_cost > row_cost[pos_idx] + float(cost_gap_m))
        if not hard_valid.any():
            active_flags.append(row_scores.new_tensor(0.0))
            continue
        hard_scores = row_scores.masked_fill(~hard_valid, torch.finfo(row_scores.dtype).min / 4.0)
        neg_idx = int(hard_scores.argmax())
        losses.append(F.softplus(row_scores[neg_idx] - row_scores[pos_idx] + float(margin)))
        active_flags.append(row_scores.new_tensor(1.0))
    if losses:
        loss = torch.stack(losses).mean()
    else:
        loss = scores.sum() * 0.0
    active = torch.stack(active_flags).mean() if active_flags else scores.new_zeros(())
    return loss, {"online_hard_active": active}


def channel_sparsity_loss(channel_gate: torch.Tensor) -> torch.Tensor:
    return channel_gate.float().mean()


def spatial_utility_entropy_loss(utility: torch.Tensor) -> torch.Tensor:
    utility = utility.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    entropy = -(utility * utility.log() + (1.0 - utility) * (1.0 - utility).log())
    return entropy.mean()
