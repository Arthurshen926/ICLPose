"""Counterfactual interpretability diagnostics for POFD-FS."""

from __future__ import annotations

import torch


def _selected_cost(scores: torch.Tensor, costs: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or costs.ndim != 2:
        raise ValueError("scores and costs must have shape (B,K)")
    if scores.shape != costs.shape:
        raise ValueError("scores and costs must have the same shape")
    masked_scores = scores.float()
    if valid_mask is not None:
        masked_scores = masked_scores.masked_fill(~valid_mask.to(device=scores.device).bool(), torch.finfo(masked_scores.dtype).min)
    selected = masked_scores.argmax(dim=1)
    batch = torch.arange(scores.shape[0], device=scores.device)
    return selected, costs.to(device=scores.device).float()[batch, selected]


def channel_group_counterfactual_drop(
    base_scores: torch.Tensor,
    ablated_group_scores: torch.Tensor,
    costs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Report ranking damage when each channel group is removed.

    ``ablated_group_scores`` is expected to be shaped ``(G,B,K)``, where each
    slice contains candidate scores recomputed after removing one channel group.
    Positive ``group_pred_cost_drop_m`` means the group was useful: removing it
    increased selected pose cost.
    """
    if ablated_group_scores.ndim != 3:
        raise ValueError("ablated_group_scores must have shape (G,B,K)")
    if ablated_group_scores.shape[1:] != base_scores.shape:
        raise ValueError("ablated_group_scores shape must match base_scores after the group dimension")
    base_selected, base_cost = _selected_cost(base_scores, costs, valid_mask=valid_mask)
    group_costs = []
    group_selected = []
    for group_idx in range(ablated_group_scores.shape[0]):
        selected, selected_cost = _selected_cost(ablated_group_scores[group_idx], costs, valid_mask=valid_mask)
        group_selected.append(selected)
        group_costs.append(selected_cost.mean())
    group_pred_cost = torch.stack(group_costs)
    group_pred_cost_drop = group_pred_cost - base_cost.mean()
    return {
        "base_selected_idx": base_selected,
        "base_pred_cost_m": base_cost.mean(),
        "group_selected_idx": torch.stack(group_selected),
        "group_pred_cost_m": group_pred_cost,
        "group_pred_cost_drop_m": group_pred_cost_drop,
        "worst_group_idx": group_pred_cost_drop.argmax(),
        "least_important_group_idx": group_pred_cost_drop.argmin(),
    }


def spatial_utility_counterfactual_drop(
    score_maps: torch.Tensor,
    utility: torch.Tensor,
    costs: torch.Tensor,
    *,
    drop_fraction: float = 0.2,
    base_weight: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare ranking after masking highest-utility vs lowest-utility pixels."""
    if score_maps.ndim != 4:
        raise ValueError("score_maps must have shape (B,K,H,W)")
    if utility.ndim != 4 or utility.shape[1] != 1:
        raise ValueError("utility must have shape (B,1,H,W)")
    if score_maps.shape[0] != utility.shape[0] or score_maps.shape[-2:] != utility.shape[-2:]:
        raise ValueError("score_maps and utility must share batch and spatial dimensions")
    bsz, _num_candidates, height, width = score_maps.shape
    num_pixels = height * width
    drop_count = max(1, min(num_pixels - 1, int(round(float(drop_fraction) * num_pixels))))
    util_flat = utility[:, 0].float().reshape(bsz, num_pixels)
    high_idx = util_flat.topk(drop_count, dim=1, largest=True).indices
    low_idx = util_flat.topk(drop_count, dim=1, largest=False).indices

    if base_weight is None:
        base_weight_flat = torch.ones((bsz, num_pixels), device=score_maps.device, dtype=score_maps.dtype)
    else:
        if base_weight.ndim != 4 or base_weight.shape[0] != bsz or base_weight.shape[1] != 1:
            raise ValueError("base_weight must have shape (B,1,H,W)")
        if base_weight.shape[-2:] != (height, width):
            raise ValueError("base_weight must share spatial dimensions with score_maps")
        base_weight_flat = base_weight[:, 0].to(device=score_maps.device, dtype=score_maps.dtype).reshape(bsz, num_pixels)
        base_weight_flat = base_weight_flat.clamp_min(0.0)
    high_weight = base_weight_flat.clone()
    low_weight = base_weight_flat.clone()
    high_weight.scatter_(1, high_idx.to(device=score_maps.device), 0.0)
    low_weight.scatter_(1, low_idx.to(device=score_maps.device), 0.0)

    flat_maps = score_maps.float().reshape(bsz, score_maps.shape[1], num_pixels)

    def weighted_scores(weight: torch.Tensor) -> torch.Tensor:
        denom = weight.sum(dim=1).clamp_min(1.0)
        return (flat_maps * weight[:, None]).sum(dim=-1) / denom[:, None]

    base_scores = weighted_scores(base_weight_flat)
    high_scores = weighted_scores(high_weight)
    low_scores = weighted_scores(low_weight)
    base_selected, base_cost = _selected_cost(base_scores, costs, valid_mask=valid_mask)
    high_selected, high_cost = _selected_cost(high_scores, costs, valid_mask=valid_mask)
    low_selected, low_cost = _selected_cost(low_scores, costs, valid_mask=valid_mask)
    return {
        "base_scores": base_scores,
        "drop_high_scores": high_scores,
        "drop_low_scores": low_scores,
        "base_selected_idx": base_selected,
        "drop_high_selected_idx": high_selected,
        "drop_low_selected_idx": low_selected,
        "base_pred_cost_m": base_cost.mean(),
        "drop_high_pred_cost_m": high_cost.mean(),
        "drop_low_pred_cost_m": low_cost.mean(),
        "drop_high_cost_delta_m": high_cost.mean() - base_cost.mean(),
        "drop_low_cost_delta_m": low_cost.mean() - base_cost.mean(),
    }
