from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.localization.radio_final_context import (
    RADIO_FINAL_CONTEXT_PCA_FORMAT,
    RadioFinalContextPcaCache,
    load_radio_final_context_pca_cache,
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _cache(
    *, pose_used: bool = False, include_grid8: bool = False, include_grid16: bool = False
) -> RadioFinalContextPcaCache:
    summary = _unit_rows(np.asarray([[1.0, 1.0]], dtype=np.float32))
    global_descriptors = _unit_rows(np.asarray([[1.0, -1.0]], dtype=np.float32))
    angles = np.linspace(0.1, 1.6, 16, dtype=np.float32)
    grid = np.stack([np.cos(angles), np.sin(angles)], axis=1)[None]
    grid8 = None
    if include_grid8:
        angles8 = np.linspace(0.1, 3.0, 64, dtype=np.float32)
        grid8 = np.stack([np.cos(angles8), np.sin(angles8)], axis=1)[None]
    grid16 = None
    if include_grid16:
        angles16 = np.linspace(0.1, 6.0, 256, dtype=np.float32)
        grid16 = np.stack([np.cos(angles16), np.sin(angles16)], axis=1)[None]
    return RadioFinalContextPcaCache(
        image_ids=np.asarray(["image.png"]),
        image_sizes=np.asarray([[400, 200]], dtype=np.int64),
        summary_descriptors=summary,
        global_descriptors=global_descriptors,
        grid4_descriptors=grid,
        metadata={
            "format": RADIO_FINAL_CONTEXT_PCA_FORMAT,
            "pose_or_ground_truth_used": pose_used,
        },
        grid8_descriptors=grid8,
        grid16_descriptors=grid16,
    )


def test_radio_final_context_uses_candidate_specific_grid_cell() -> None:
    cache = _cache()
    xy = np.asarray(
        [[0.0, 0.0], [99.9, 49.9], [100.0, 50.0], [399.0, 199.0]],
        dtype=np.float32,
    )
    descriptors = cache.node_descriptors("image.png", xy)

    assert descriptors.shape == (4, 6)
    np.testing.assert_allclose(descriptors[0, :4], descriptors[3, :4])
    np.testing.assert_allclose(descriptors[0, 4:], cache.grid4_descriptors[0, 0])
    np.testing.assert_allclose(descriptors[1, 4:], cache.grid4_descriptors[0, 0])
    np.testing.assert_allclose(descriptors[2, 4:], cache.grid4_descriptors[0, 5])
    np.testing.assert_allclose(descriptors[3, 4:], cache.grid4_descriptors[0, 15])
    np.testing.assert_allclose(
        cache.node_grid4_descriptors("image.png", xy), descriptors[:, 4:]
    )


def test_radio_final_context_rejects_pose_supervision() -> None:
    with pytest.raises(ValueError, match="pose/GT free"):
        _cache(pose_used=True)


def test_radio_final_context_exposes_only_spatial_grid8_when_present() -> None:
    cache = _cache(include_grid8=True)
    grid, size = cache.image_grid_descriptors("image.png", grid_size=8)

    assert grid.shape == (8, 8, 2)
    np.testing.assert_array_equal(size, [400, 200])
    np.testing.assert_allclose(grid.reshape(64, 2), cache.grid8_descriptors[0])
    with pytest.raises(ValueError, match="no grid8"):
        _cache().image_grid_descriptors("image.png", grid_size=8)


def test_radio_final_context_exposes_grid16_when_present() -> None:
    cache = _cache(include_grid16=True)
    grid, size = cache.image_grid_descriptors("image.png", grid_size=16)

    assert grid.shape == (16, 16, 2)
    np.testing.assert_array_equal(size, [400, 200])
    np.testing.assert_allclose(grid.reshape(256, 2), cache.grid16_descriptors[0])
    with pytest.raises(ValueError, match="no grid16"):
        _cache().image_grid_descriptors("image.png", grid_size=16)


def test_radio_final_context_selects_the_requested_image_grid() -> None:
    first = _cache(include_grid16=True)
    second_grid = np.asarray(first.grid16_descriptors, dtype=np.float32).copy()
    second_grid[0] = second_grid[0, ::-1]
    cache = RadioFinalContextPcaCache(
        image_ids=np.asarray(["first.png", "second.png"]),
        image_sizes=np.asarray([[400, 200], [400, 200]], dtype=np.int64),
        summary_descriptors=np.repeat(first.summary_descriptors, 2, axis=0),
        global_descriptors=np.repeat(first.global_descriptors, 2, axis=0),
        grid4_descriptors=np.repeat(first.grid4_descriptors, 2, axis=0),
        grid16_descriptors=np.concatenate(
            [first.grid16_descriptors, second_grid], axis=0
        ),
        metadata=first.metadata,
    )

    grid, _size = cache.image_grid_descriptors("second.png", grid_size=16)
    np.testing.assert_allclose(grid.reshape(256, 2), second_grid[0])


def test_radio_final_context_loader_rejects_stale_metadata(tmp_path: Path) -> None:
    cache = _cache()
    path = tmp_path / "context.npz"
    np.savez_compressed(
        path,
        image_ids=cache.image_ids,
        image_sizes=cache.image_sizes,
        summary_descriptors=cache.summary_descriptors.astype(np.float16),
        global_descriptors=cache.global_descriptors.astype(np.float16),
        grid4_descriptors=cache.grid4_descriptors.astype(np.float16),
        metadata_json=np.asarray(json.dumps(cache.metadata, sort_keys=True)),
    )

    loaded = load_radio_final_context_pca_cache(
        path, expected_metadata={"format": RADIO_FINAL_CONTEXT_PCA_FORMAT}
    )
    assert loaded.node_feature_dim == 6
    with pytest.raises(ValueError, match="stale RADIO final context"):
        load_radio_final_context_pca_cache(
            path, expected_metadata={"radio_checkpoint_sha256": "different"}
        )
