from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.pfir import ContributorLabels
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    remap_pinhole_contributors_to_raw_grid,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    _remap_ideal_hits_to_raw_tokens,
)


def test_sparse_online_hit_remap_matches_dense_contributor_coordinate_contract():
    height, width = 6, 8
    camera = ColmapCamera(0, 2, width, height, (6.5, 4.0, 3.0, 0.08))
    source_ids = np.arange(height * width, dtype=np.int64).reshape(height, width, 1)
    source_weight = np.linspace(0.1, 0.9, height * width, dtype=np.float32).reshape(
        height, width, 1
    )
    dense, _ = remap_pinhole_contributors_to_raw_grid(
        ContributorLabels(source_ids, source_weight, np.eye(4)),
        camera_model_id=camera.model_id,
        camera_width=camera.width,
        camera_height=camera.height,
        camera_params=np.asarray(camera.params),
    )
    pixel = np.arange(height * width, dtype=np.int64)
    remapped_pixel, remapped_row, remapped_weight = _remap_ideal_hits_to_raw_tokens(
        pixel,
        source_ids.reshape(-1),
        source_weight.reshape(-1),
        camera,
        token_width=width,
        token_height=height,
        supersample_factor=1,
    )
    actual_ids = np.full((height * width,), -1, dtype=np.int64)
    actual_weight = np.zeros((height * width,), dtype=np.float32)
    actual_ids[remapped_pixel] = remapped_row
    actual_weight[remapped_pixel] = remapped_weight
    np.testing.assert_array_equal(actual_ids.reshape(height, width), dense.topk_primitive_ids[..., 0])
    np.testing.assert_array_equal(actual_weight.reshape(height, width), dense.topk_weights[..., 0])


def test_coordinate_supersampling_averages_raw_samples_into_tokens():
    camera = ColmapCamera(0, 0, 4, 4, (4.0, 2.0, 2.0))
    pixel = np.arange(16, dtype=np.int64)
    token, row, weight = _remap_ideal_hits_to_raw_tokens(
        pixel, pixel, np.ones(16, dtype=np.float32), camera,
        token_width=2, token_height=2, supersample_factor=2,
    )
    np.testing.assert_array_equal(np.bincount(token, minlength=4), 4)
    np.testing.assert_allclose(np.bincount(token, weights=weight, minlength=4), 1.0)
    np.testing.assert_array_equal(row, pixel)
