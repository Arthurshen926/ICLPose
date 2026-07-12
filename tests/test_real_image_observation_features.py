from __future__ import annotations

import numpy as np
import torch

from feature_extract.vfm.localization.real_image_observation_features import (
    sample_dense_feature_points,
    spatially_diverse_detection_indices,
)


def test_sample_dense_feature_points_uses_pixel_endpoints() -> None:
    feature_map = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
    descriptors, _scores = sample_dense_feature_points(
        feature_map,
        np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        image_width=2,
        image_height=2,
    )
    np.testing.assert_allclose(descriptors, [[1.0, 0.0], [1.0, 0.0]], atol=1e-6)


def test_spatial_detection_selection_applies_nms_and_grid_coverage() -> None:
    xy = np.asarray([[1, 1], [2, 1], [80, 1], [1, 80], [80, 80]], dtype=np.float32)
    scores = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    selected = spatially_diverse_detection_indices(
        xy,
        scores,
        top_k=4,
        nms_radius_px=2.0,
        image_width=100,
        image_height=100,
        grid_rows=2,
        grid_cols=2,
    )
    np.testing.assert_array_equal(selected, [0, 2, 3, 4])


def _naive_spatial_detection_indices(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    nms_radius_px: float,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
    min_score: float | None = None,
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    valid &= (points[:, 0] >= 0.0) & (points[:, 0] <= float(image_width - 1))
    valid &= (points[:, 1] >= 0.0) & (points[:, 1] <= float(image_height - 1))
    if min_score is not None:
        valid &= values >= float(min_score)
    candidates = np.flatnonzero(valid)
    order = np.lexsort((candidates, -values[candidates]))
    ranked = candidates[order]
    limit = min(int(top_k), int(ranked.size))
    quota = max(1, int(np.ceil(float(limit) / float(grid_rows * grid_cols))))
    selected: list[int] = []
    selected_set: set[int] = set()
    cell_counts: dict[tuple[int, int], int] = {}
    radius2 = max(float(nms_radius_px), 0.0) ** 2

    def cell(index: int) -> tuple[int, int]:
        col = int(np.clip(np.floor(points[index, 0] / float(image_width) * grid_cols), 0, grid_cols - 1))
        row = int(np.clip(np.floor(points[index, 1] / float(image_height) * grid_rows), 0, grid_rows - 1))
        return row, col

    for use_quota in (True, False):
        for index in ranked.tolist():
            if len(selected) >= limit:
                break
            if int(index) in selected_set:
                continue
            if radius2 > 0.0 and selected:
                delta = points[np.asarray(selected, dtype=np.int64)] - points[int(index)]
                if np.any(np.sum(delta * delta, axis=1) <= radius2):
                    continue
            key = cell(int(index))
            if use_quota and cell_counts.get(key, 0) >= quota:
                continue
            selected.append(int(index))
            selected_set.add(int(index))
            cell_counts[key] = cell_counts.get(key, 0) + 1
    return np.asarray(selected, dtype=np.int64)


def test_spatial_hash_selection_is_exactly_equal_to_naive_selection() -> None:
    for seed in range(10):
        rng = np.random.default_rng(seed)
        xy = rng.uniform([-4.0, -4.0], [131.0, 99.0], size=(750, 2)).astype(np.float32)
        scores = np.round(rng.random(750), decimals=2).astype(np.float32)
        xy[seed] = np.nan
        scores[seed + 10] = np.nan
        for radius in (0.0, 1.0, 4.0, 11.5):
            kwargs = {
                "top_k": 256,
                "nms_radius_px": radius,
                "image_width": 128,
                "image_height": 96,
                "grid_rows": 4,
                "grid_cols": 5,
                "min_score": 0.1,
            }
            expected = _naive_spatial_detection_indices(xy, scores, **kwargs)
            actual = spatially_diverse_detection_indices(xy, scores, **kwargs)
            np.testing.assert_array_equal(actual, expected)
