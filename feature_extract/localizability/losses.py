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


def spatial_utility_evidence_loss(
    score_maps: torch.Tensor,
    utility: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    score_channel: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align utility with pixels where the oracle candidate beats hard alternatives."""

    if score_maps.ndim == 5:
        channels = int(score_maps.shape[2])
        channel = int(score_channel)
        if channel < 0:
            channel = channels + channel
        if channel < 0 or channel >= channels:
            raise ValueError("score_channel outside score_maps channel range")
        score_maps = score_maps[:, :, channel]
    if score_maps.ndim != 4:
        raise ValueError("score_maps must have shape (B,K,H,W) or (B,K,C,H,W)")
    if pose_cost_m.ndim != 2 or pose_cost_m.shape != score_maps.shape[:2]:
        raise ValueError("pose_cost_m must have shape (B,K) matching score_maps")
    if utility.ndim != 4 or utility.shape[0] != score_maps.shape[0] or utility.shape[1] != 1:
        raise ValueError("utility must have shape (B,1,H,W)")
    if utility.shape[-2:] != score_maps.shape[-2:]:
        utility = F.interpolate(utility.float(), size=score_maps.shape[-2:], mode="bilinear", align_corners=False)

    valid = valid_mask.bool().to(device=score_maps.device) if valid_mask is not None else torch.ones_like(pose_cost_m, dtype=torch.bool)
    rows = []
    targets = []
    for bidx in range(score_maps.shape[0]):
        row_valid = valid[bidx]
        if not bool(row_valid.any()):
            continue
        row_cost = pose_cost_m[bidx].float().to(device=score_maps.device).masked_fill(~row_valid, float("inf"))
        oracle_idx = int(row_cost.argmin())
        non_oracle = row_valid.clone()
        non_oracle[oracle_idx] = False
        oracle_map = score_maps[bidx, oracle_idx].float()
        if bool(non_oracle.any()):
            competitor = score_maps[bidx, non_oracle].float().amax(dim=0)
            evidence = oracle_map - competitor
        else:
            evidence = oracle_map
        finite = torch.isfinite(evidence)
        if not bool(finite.any()):
            continue
        evidence = evidence.masked_fill(~finite, 0.0)
        ev_min = evidence.amin()
        ev_max = evidence.amax()
        target = (evidence - ev_min) / (ev_max - ev_min).clamp_min(1.0e-6)
        rows.append(utility[bidx, 0].float().clamp(1.0e-6, 1.0 - 1.0e-6))
        targets.append(target.detach())
    if not rows:
        return utility.sum() * 0.0, {"utility_evidence_active": utility.new_zeros(())}
    pred = torch.stack(rows, dim=0)
    target = torch.stack(targets, dim=0)
    loss = F.binary_cross_entropy(pred, target)
    return loss, {"utility_evidence_active": utility.new_tensor(float(len(rows)) / float(score_maps.shape[0]))}
