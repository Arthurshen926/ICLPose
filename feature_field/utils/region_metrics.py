from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


REGION_NAMES = (
    'foreground',
    'near',
    'mid',
    'far',
    'edge',
    'non_edge',
    'far_edge',
)


def empty_region_score_dict() -> Dict[str, list[float]]:
    return {name: [] for name in REGION_NAMES}


def extend_region_scores(target: Dict[str, list[float]], source: Dict[str, list[float]]) -> None:
    for name in REGION_NAMES:
        target.setdefault(name, []).extend(source.get(name, []))


def compute_depth_edge_strength(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    height, width = depth.shape[-2:]
    depth_up = F.interpolate(depth, (height * 2, width * 2), mode='bilinear', align_corners=False)
    high_freq = torch.abs(depth_up[:, :, ::2, ::2] - depth)
    local_mean = F.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    local_var = torch.abs(depth - local_mean)
    return F.avg_pool2d(high_freq + local_var, kernel_size=3, stride=1, padding=1)


def compute_region_masks(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    alpha_threshold: float = 0.5,
    near_quantile: float = 1.0 / 3.0,
    far_quantile: float = 2.0 / 3.0,
    edge_quantile: float = 0.85,
    min_valid_pixels: int = 32,
    min_depth: float = 1e-4,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    if depth.shape[-2:] != alpha.shape[-2:]:
        depth = F.interpolate(depth, alpha.shape[-2:], mode='bilinear', align_corners=False)

    valid = (alpha > alpha_threshold) & (depth > min_depth)
    edge_strength = compute_depth_edge_strength(depth)
    masks = {name: torch.zeros_like(valid, dtype=torch.bool) for name in REGION_NAMES}
    masks['foreground'] = valid.bool()

    for batch_idx in range(depth.shape[0]):
        valid_mask = valid[batch_idx, 0]
        if not valid_mask.any():
            continue

        depth_map = depth[batch_idx, 0]
        depth_vals = depth_map[valid_mask]
        if depth_vals.numel() >= min_valid_pixels:
            near_cut = torch.quantile(depth_vals, near_quantile)
            far_cut = torch.quantile(depth_vals, far_quantile)
        else:
            near_cut = depth_vals.median()
            far_cut = depth_vals.median()

        near_mask = valid_mask & (depth_map <= near_cut)
        far_mask = valid_mask & (depth_map > far_cut)
        mid_mask = valid_mask & ~(near_mask | far_mask)

        edge_vals = edge_strength[batch_idx, 0][valid_mask]
        if edge_vals.numel() >= min_valid_pixels:
            edge_cut = torch.quantile(edge_vals, edge_quantile)
        elif edge_vals.numel() > 0:
            edge_cut = edge_vals.mean()
        else:
            edge_cut = depth_map.new_tensor(0.0)

        if float(edge_cut.item()) > 0.0:
            edge_mask = valid_mask & (edge_strength[batch_idx, 0] >= edge_cut)
        else:
            edge_mask = valid_mask & (edge_strength[batch_idx, 0] > 0.0)
        non_edge_mask = valid_mask & ~edge_mask

        masks['near'][batch_idx, 0] = near_mask
        masks['mid'][batch_idx, 0] = mid_mask
        masks['far'][batch_idx, 0] = far_mask
        masks['edge'][batch_idx, 0] = edge_mask
        masks['non_edge'][batch_idx, 0] = non_edge_mask
        masks['far_edge'][batch_idx, 0] = far_mask & edge_mask

    return masks, edge_strength


def score_regions(score_map: torch.Tensor, region_masks: Dict[str, torch.Tensor]) -> Dict[str, list[float]]:
    if score_map.ndim == 2:
        score_map = score_map.unsqueeze(0)
    scores = empty_region_score_dict()
    for name, mask in region_masks.items():
        mask_hw = mask.squeeze(1).bool() if mask.ndim == 4 else mask.bool()
        values = []
        for batch_idx in range(score_map.shape[0]):
            valid = mask_hw[batch_idx]
            if valid.any():
                values.append(float(score_map[batch_idx][valid].mean().item()))
            else:
                values.append(float('nan'))
        scores[name] = values
    return scores


def summarize_region_scores(region_scores: Dict[str, list[float]]) -> Dict[str, dict[str, float | int]]:
    summary: Dict[str, dict[str, float | int]] = {}
    for name, values in region_scores.items():
        valid = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
        if valid.size == 0:
            summary[name] = {
                'mean': float('nan'),
                'std': float('nan'),
                'count': 0,
            }
            continue
        summary[name] = {
            'mean': float(valid.mean()),
            'std': float(valid.std()),
            'count': int(valid.size),
        }
    return summary
