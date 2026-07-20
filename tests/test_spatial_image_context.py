from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
    load_spatial_image_context_cache,
    save_spatial_image_context_cache,
)


def _grid(count: int, size: int) -> np.ndarray:
    angles = np.linspace(0.1, 2.8, count * size * size, dtype=np.float32)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).reshape(
        count, size * size, 2
    )


def _cache() -> SpatialImageContextCache:
    return SpatialImageContextCache(
        image_ids=np.asarray(["first.png", "second.png"]),
        image_sizes=np.asarray([[80, 40], [160, 80]], dtype=np.int64),
        grids={4: _grid(2, 4), 8: _grid(2, 8)},
        metadata={
            "format": "test_spatial_image_context_v1",
            "pose_or_ground_truth_used": False,
            "spatial_grid_sizes": [4, 8],
        },
    )


def test_spatial_image_context_returns_the_requested_image_and_grid() -> None:
    cache = _cache()
    grid, size = cache.image_grid_descriptors("second.png", grid_size=8)

    assert grid.shape == (8, 8, 2)
    np.testing.assert_allclose(grid.reshape(64, 2), cache.grids[8][1])
    np.testing.assert_array_equal(size, [160, 80])
    with pytest.raises(ValueError, match="no grid16"):
        cache.grid_descriptors(16)


def test_spatial_image_context_round_trip_and_rejects_stale_metadata(tmp_path) -> None:
    path = tmp_path / "cache.npz"
    save_spatial_image_context_cache(_cache(), path)

    loaded = load_spatial_image_context_cache(
        path,
        expected_format="test_spatial_image_context_v1",
        expected_metadata={"spatial_grid_sizes": [4, 8]},
    )
    np.testing.assert_allclose(loaded.grids[4], _cache().grids[4], atol=5e-4)
    with pytest.raises(ValueError, match="stale spatial image context cache"):
        load_spatial_image_context_cache(
            path,
            expected_format="test_spatial_image_context_v1",
            expected_metadata={"spatial_grid_sizes": [16]},
        )


def test_spatial_image_context_rejects_untracked_grids() -> None:
    with pytest.raises(ValueError, match="grid metadata is stale"):
        SpatialImageContextCache(
            image_ids=np.asarray(["image.png"]),
            image_sizes=np.asarray([[80, 40]], dtype=np.int64),
            grids={4: _grid(1, 4)},
            metadata={
                "format": "test_spatial_image_context_v1",
                "pose_or_ground_truth_used": False,
                "spatial_grid_sizes": [8],
            },
        )


def test_spatial_image_context_rejects_unsafe_extra_arrays(tmp_path) -> None:
    with pytest.raises(ValueError, match="object dtype"):
        save_spatial_image_context_cache(
            _cache(), tmp_path / "object.npz", extra_arrays={"bad": np.asarray([object()])}
        )
    with pytest.raises(ValueError, match="non-finite"):
        save_spatial_image_context_cache(
            _cache(), tmp_path / "nan.npz", extra_arrays={"bad": np.asarray([np.nan])}
        )
