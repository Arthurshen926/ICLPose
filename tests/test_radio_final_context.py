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


def _cache(*, pose_used: bool = False) -> RadioFinalContextPcaCache:
    summary = _unit_rows(np.asarray([[1.0, 1.0]], dtype=np.float32))
    global_descriptors = _unit_rows(np.asarray([[1.0, -1.0]], dtype=np.float32))
    angles = np.linspace(0.1, 1.6, 16, dtype=np.float32)
    grid = np.stack([np.cos(angles), np.sin(angles)], axis=1)[None]
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


def test_radio_final_context_rejects_pose_supervision() -> None:
    with pytest.raises(ValueError, match="pose/GT free"):
        _cache(pose_used=True)


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
