"""Hard-case subset builders for localization hypothesis verification."""

from __future__ import annotations

import torch


def _valid(valid_mask: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    return valid_mask.to(device=like.device).bool() if valid_mask is not None else torch.ones_like(like, dtype=torch.bool)


def _selected_values(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices[:, None]).squeeze(1)


def build_hard_case_masks(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    basin_label: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    pnp_inliers: torch.Tensor | None = None,
    delta_trans_m: torch.Tensor | None = None,
    delta_rot_deg: torch.Tensor | None = None,
    retrieval_topk: int = 10,
    near_identity_trans_m: float = 0.05,
    near_identity_rot_deg: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return per-query hard-case masks for false-accept analysis."""

    if scores.shape != pose_cost_m.shape or basin_label.shape != scores.shape:
        raise ValueError("scores, pose_cost_m, and basin_label must have shape (B,K)")
    valid = _valid(valid_mask, scores)
    basin = basin_label.to(device=scores.device).bool() & valid
    masked_scores = scores.float().masked_fill(~valid, float("-inf"))
    pred_idx = masked_scores.argmax(dim=1)
    selected_basin = _selected_values(basin, pred_idx)
    any_basin = basin.any(dim=1)
    score_top1_false_accept = (~selected_basin) & any_basin

    topk = max(1, min(int(retrieval_topk), scores.shape[1]))
    retrieval_top1_valid = valid[:, 0]
    retrieval_top1_basin = basin[:, 0] & retrieval_top1_valid
    retrieval_topk_has_basin = basin[:, :topk].any(dim=1)
    retrieval_top1_wrong_but_topk_basin = retrieval_top1_valid & (~retrieval_top1_basin) & retrieval_topk_has_basin

    if delta_trans_m is not None and delta_rot_deg is not None:
        if delta_trans_m.shape != scores.shape or delta_rot_deg.shape != scores.shape:
            raise ValueError("delta_trans_m and delta_rot_deg must match scores shape")
        selected_delta_trans = _selected_values(delta_trans_m.to(device=scores.device).float(), pred_idx)
        selected_delta_rot = _selected_values(delta_rot_deg.to(device=scores.device).float(), pred_idx)
        near_identity = (selected_delta_trans <= float(near_identity_trans_m)) & (
            selected_delta_rot <= float(near_identity_rot_deg)
        )
    else:
        near_identity = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)

    if pnp_inliers is not None:
        if pnp_inliers.shape != scores.shape:
            raise ValueError("pnp_inliers must match scores shape")
        pnp_idx = pnp_inliers.to(device=scores.device).float().masked_fill(~valid, float("-inf")).argmax(dim=1)
        pnp_basin = _selected_values(basin, pnp_idx)
        pnp_high_score_wrong = (~pnp_basin) & any_basin
    else:
        pnp_high_score_wrong = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)

    return {
        "score_top1_false_accept": score_top1_false_accept,
        "retrieval_top1_wrong_but_topk_basin": retrieval_top1_wrong_but_topk_basin,
        "near_identity_false_positive": score_top1_false_accept & near_identity,
        "pnp_high_score_wrong": pnp_high_score_wrong,
        "has_basin_candidate": any_basin,
    }


def summarize_hard_case_masks(masks: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Summarize boolean hard-case masks as counts and fractions."""

    if not masks:
        return {}
    first = next(iter(masks.values()))
    denom = max(1, int(first.numel()))
    summary: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        count = int(mask.bool().sum().detach().cpu())
        summary[name] = {"count": count, "fraction": float(count) / float(denom)}
    return summary


__all__ = ["build_hard_case_masks", "summarize_hard_case_masks"]
