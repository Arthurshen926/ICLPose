"""Negative controls and counterfactual masks for POFD-FS scoring audits."""

from __future__ import annotations

from typing import Mapping

import torch


def _as_tensor(fields: Mapping[str, torch.Tensor], *names: str) -> torch.Tensor:
    for name in names:
        value = fields.get(name)
        if value is not None:
            return value.float()
    raise KeyError(f"None of these metadata fields are available: {names}")


def _mask_invalid(scores: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return scores
    valid = valid_mask.to(device=scores.device).bool()
    if valid.shape != scores.shape:
        raise ValueError("valid_mask must match metadata score shape")
    return scores.masked_fill(~valid, float("-inf"))


def metadata_baseline_scores(
    fields: Mapping[str, torch.Tensor],
    *,
    mode: str,
    valid_mask: torch.Tensor | None = None,
    rot_weight: float = 0.01,
) -> torch.Tensor:
    """Build a no-feature baseline score table from candidate metadata only."""

    if mode == "candidate_rank":
        scores = -_as_tensor(fields, "candidate_rank", "retrieval_rank", "candidate_idx")
    elif mode == "retrieval_score":
        scores = _as_tensor(fields, "retrieval_score", "retrieval_scores", "retrieval_scores_candidates")
    elif mode == "pnp_inliers":
        scores = _as_tensor(fields, "pnp_inliers", "retrieval_pnp_num_inliers_candidates", "num_inliers")
    elif mode == "reproj_median":
        scores = -_as_tensor(fields, "reproj_median_px", "retrieval_pnp_reproj_median_candidates", "reproj_median")
    elif mode == "delta_pose":
        trans = _as_tensor(fields, "delta_trans_m", "pose_delta_trans_m")
        rot = _as_tensor(fields, "delta_rot_deg", "pose_delta_rot_deg")
        scores = -(trans + float(rot_weight) * rot)
    elif mode == "score_margin":
        scores = _as_tensor(fields, "score_margin", "retrieval_score_margin", "metadata_score_margin")
    else:
        raise ValueError(f"Unsupported metadata baseline mode: {mode}")
    return _mask_invalid(scores.float(), valid_mask)


def feature_batch_shuffle_control(
    feature: torch.Tensor,
    *,
    permutation: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffle feature batches across queries while preserving tensor shape."""

    if feature.ndim < 1:
        raise ValueError("feature must have a batch dimension")
    if permutation is None:
        permutation = torch.randperm(feature.shape[0], device=feature.device, generator=generator)
    permutation = permutation.to(device=feature.device).long()
    if permutation.shape != (feature.shape[0],):
        raise ValueError("permutation must have shape (B,)")
    return feature.index_select(0, permutation), permutation


def wrong_scene_feature_control(feature: torch.Tensor, wrong_scene_feature: torch.Tensor) -> torch.Tensor:
    """Replace a feature tensor with wrong-scene evidence for leakage checks."""

    if wrong_scene_feature.shape != feature.shape:
        raise ValueError("wrong_scene_feature must match feature shape")
    return wrong_scene_feature.to(device=feature.device, dtype=feature.dtype)


def _num_to_mask(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return max(1, min(total, int(torch.ceil(torch.tensor(float(fraction) * float(total))).item())))


def counterfactual_channel_mask(
    feature: torch.Tensor,
    utility: torch.Tensor,
    *,
    mode: str,
    fraction: float,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Remove high- or low-utility channels from a dense feature map."""

    if feature.ndim != 4:
        raise ValueError("feature must have shape (B,C,H,W)")
    if mode not in {"remove_high", "remove_low"}:
        raise ValueError("mode must be 'remove_high' or 'remove_low'")
    util = utility.to(device=feature.device).float()
    if util.ndim == 1:
        util = util.view(1, -1).expand(feature.shape[0], -1)
    elif util.ndim == 4:
        util = util.flatten(2).mean(dim=2)
    elif util.ndim != 2:
        raise ValueError("utility must have shape (C,), (B,C), or (B,C,H,W)")
    if util.shape != feature.shape[:2]:
        raise ValueError("channel utility must match feature batch/channel dimensions")
    k = _num_to_mask(feature.shape[1], fraction)
    order = torch.argsort(util, dim=1, descending=(mode == "remove_high"))
    out = feature.clone()
    for batch_idx in range(feature.shape[0]):
        out[batch_idx, order[batch_idx, :k]] = float(fill_value)
    return out


def counterfactual_spatial_mask(
    feature: torch.Tensor,
    utility_map: torch.Tensor,
    *,
    mode: str,
    fraction: float,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Remove high- or low-utility spatial regions from a dense feature map."""

    if feature.ndim != 4:
        raise ValueError("feature must have shape (B,C,H,W)")
    if mode not in {"remove_high", "remove_low"}:
        raise ValueError("mode must be 'remove_high' or 'remove_low'")
    util = utility_map.to(device=feature.device).float()
    if util.ndim == 3:
        util = util[:, None]
    if util.ndim != 4 or util.shape[0] != feature.shape[0] or util.shape[-2:] != feature.shape[-2:]:
        raise ValueError("utility_map must have shape (B,1,H,W) or (B,H,W)")
    flat = util.flatten(2).mean(dim=1)
    k = _num_to_mask(flat.shape[1], fraction)
    order = torch.argsort(flat, dim=1, descending=(mode == "remove_high"))
    out = feature.clone().flatten(2)
    for batch_idx in range(feature.shape[0]):
        out[batch_idx, :, order[batch_idx, :k]] = float(fill_value)
    return out.reshape_as(feature)


__all__ = [
    "counterfactual_channel_mask",
    "counterfactual_spatial_mask",
    "feature_batch_shuffle_control",
    "metadata_baseline_scores",
    "wrong_scene_feature_control",
]
