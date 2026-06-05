"""Diagnostics for VFM-token to 2DGS surface mapping."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from feature_extract.vfm.vfm_2dgs_mapping import Vfm2DgsContributionBuffer


def token_purity_diagnostic_grid(
    contribution_buffer: Vfm2DgsContributionBuffer,
    grid_shape: tuple[int, int],
    component_threshold: float = 0.6,
    entropy_threshold: float = 0.7,
    support_threshold: int = 8,
) -> Mapping[str, object]:
    """Project per-token mapping purity diagnostics onto the VFM token grid.

    A token is considered cross-surface-suspect when its dominant connected
    component is weak, its renderer alpha entropy is high, or its mapped
    surface support is unusually broad.
    """
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("grid_shape must contain positive dimensions")
    token_indices = np.asarray(contribution_buffer.token_indices, dtype=np.int64).reshape(-1)
    valid = (token_indices >= 0) & (token_indices < int(height * width))
    token_indices = token_indices[valid]
    rows = token_indices // int(width)
    cols = token_indices % int(width)

    purity_grid = np.full((height, width), np.nan, dtype=np.float32)
    component_grid = np.full((height, width), np.nan, dtype=np.float32)
    entropy_grid = np.full((height, width), np.nan, dtype=np.float32)
    top_alpha_grid = np.full((height, width), np.nan, dtype=np.float32)
    support_count_grid = np.zeros((height, width), dtype=np.int32)
    suspect_grid = np.zeros((height, width), dtype=bool)

    support_counts_all = np.diff(np.asarray(contribution_buffer.support_offsets, dtype=np.int64))
    support_counts = support_counts_all[valid] if support_counts_all.shape[0] == valid.shape[0] else np.zeros_like(token_indices)
    purity = np.asarray(contribution_buffer.purity_scores, dtype=np.float32).reshape(-1)[valid]
    component = np.asarray(contribution_buffer.component_concentrations, dtype=np.float32).reshape(-1)[valid]
    entropy = np.asarray(contribution_buffer.alpha_entropy, dtype=np.float32).reshape(-1)[valid]
    top_alpha = np.asarray(contribution_buffer.top_alpha, dtype=np.float32).reshape(-1)[valid]
    component_suspect = component < float(component_threshold)
    entropy_suspect = entropy > float(entropy_threshold)
    support_suspect = support_counts > int(support_threshold)
    suspect = component_suspect | entropy_suspect | support_suspect

    purity_grid[rows, cols] = purity
    component_grid[rows, cols] = component
    entropy_grid[rows, cols] = entropy
    top_alpha_grid[rows, cols] = top_alpha
    support_count_grid[rows, cols] = support_counts.astype(np.int32, copy=False)
    suspect_grid[rows, cols] = suspect

    token_count = int(token_indices.size)
    suspect_count = int(np.sum(suspect))
    return {
        "token_count": token_count,
        "cross_surface_suspect_count": suspect_count,
        "cross_surface_suspect_fraction": float(suspect_count / max(token_count, 1)),
        "low_component_count": int(np.sum(component_suspect)),
        "low_component_fraction": float(np.sum(component_suspect) / max(token_count, 1)),
        "high_entropy_count": int(np.sum(entropy_suspect)),
        "high_entropy_fraction": float(np.sum(entropy_suspect) / max(token_count, 1)),
        "large_support_count": int(np.sum(support_suspect)),
        "large_support_fraction": float(np.sum(support_suspect) / max(token_count, 1)),
        "mean_purity": 0.0 if token_count == 0 else float(np.mean(purity)),
        "mean_component_concentration": 0.0 if token_count == 0 else float(np.mean(component)),
        "mean_alpha_entropy": 0.0 if token_count == 0 else float(np.mean(entropy)),
        "mean_support_count": 0.0 if token_count == 0 else float(np.mean(support_counts)),
        "purity_grid": purity_grid,
        "component_concentration_grid": component_grid,
        "alpha_entropy_grid": entropy_grid,
        "top_alpha_grid": top_alpha_grid,
        "support_count_grid": support_count_grid,
        "cross_surface_suspect_grid": suspect_grid,
    }
