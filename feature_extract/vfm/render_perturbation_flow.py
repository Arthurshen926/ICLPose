from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _threshold_key(threshold: float) -> str:
    value = float(threshold)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _cell_indices(xy: np.ndarray, image_width: int, image_height: int, grid_width: int, grid_height: int) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    col = np.floor(coords[:, 0] / max(cell_w, 1e-12)).astype(np.int64)
    row = np.floor(coords[:, 1] / max(cell_h, 1e-12)).astype(np.int64)
    col = np.clip(col, 0, int(grid_width) - 1)
    row = np.clip(row, 0, int(grid_height) - 1)
    return np.stack([col, row], axis=1)


def flow_capture_stats(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    thresholds_px: Sequence[float] = (8.0, 16.0, 32.0),
    cell_radius_thresholds: Sequence[int] = (0, 1, 2),
) -> dict[str, float | int | None]:
    """Summarize whether pose-induced optical flow stays within local matcher capture range."""

    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError("source_xy and target_xy must have the same shape")
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(grid_width) <= 0 or int(grid_height) <= 0:
        raise ValueError("grid dimensions must be positive")
    valid = (
        np.isfinite(source).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & (source[:, 0] >= 0.0)
        & (source[:, 0] <= float(image_width - 1))
        & (source[:, 1] >= 0.0)
        & (source[:, 1] <= float(image_height - 1))
        & (target[:, 0] >= 0.0)
        & (target[:, 0] <= float(image_width - 1))
        & (target[:, 1] >= 0.0)
        & (target[:, 1] <= float(image_height - 1))
    )
    source_valid = source[valid]
    target_valid = target[valid]
    flow = np.linalg.norm(target_valid - source_valid, axis=1)
    stats: dict[str, float | int | None] = {
        "flow_count": int(source.shape[0]),
        "flow_valid_count": int(flow.size),
        "flow_visible_fraction": float(flow.size / source.shape[0]) if source.shape[0] else 0.0,
        "flow_median_px": None if flow.size == 0 else float(np.median(flow)),
        "flow_p90_px": None if flow.size == 0 else float(np.percentile(flow, 90.0)),
        "flow_p95_px": None if flow.size == 0 else float(np.percentile(flow, 95.0)),
    }
    for threshold in thresholds_px:
        key = f"flow_within_{_threshold_key(float(threshold))}px"
        stats[key] = None if flow.size == 0 else float(np.mean(flow <= float(threshold)))
    if flow.size == 0:
        stats["flow_within_same_cell"] = None
        for radius in cell_radius_thresholds:
            stats[f"flow_within_cell_radius_{int(radius)}"] = None
        stats["flow_cell_radius_median"] = None
        stats["flow_cell_radius_p90"] = None
        stats["flow_cell_radius_p95"] = None
        stats["flow_cell_radius_max"] = None
    else:
        source_cell = _cell_indices(source_valid, image_width, image_height, grid_width, grid_height)
        target_cell = _cell_indices(target_valid, image_width, image_height, grid_width, grid_height)
        cell_delta = np.abs(target_cell - source_cell)
        cell_radius = np.max(cell_delta, axis=1)
        stats["flow_within_same_cell"] = float(np.mean(cell_radius <= 0))
        for radius in cell_radius_thresholds:
            stats[f"flow_within_cell_radius_{int(radius)}"] = float(np.mean(cell_radius <= int(radius)))
        stats["flow_cell_radius_median"] = float(np.median(cell_radius))
        stats["flow_cell_radius_p90"] = float(np.percentile(cell_radius, 90.0))
        stats["flow_cell_radius_p95"] = float(np.percentile(cell_radius, 95.0))
        stats["flow_cell_radius_max"] = int(np.max(cell_radius))
    return stats
