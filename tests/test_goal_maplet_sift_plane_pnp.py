from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_sift_plane_pnp import (
    _grid_indices,
    _homography_inliers,
    _mapping_world_point,
    _mutual_ratio_matches,
)


def test_grid_indices_preserve_area_resize_pixel_centres() -> None:
    x, y = _grid_indices(np.asarray([[1.5, 1.5], [1021.5, 573.5]]), 1024, 576)
    np.testing.assert_array_equal(x, [0, 255])
    np.testing.assert_array_equal(y, [0, 143])


def test_mutual_ratio_matches_rejects_ambiguous_descriptor() -> None:
    query = np.asarray([[0, 0], [10, 0], [5, 5]], np.float32)
    source = np.asarray([[0, 0], [10, 0], [5.1, 5], [4.9, 5]], np.float32)
    qi, si, _ = _mutual_ratio_matches(query, source, ratio=0.8)
    np.testing.assert_array_equal(qi, [0, 1])
    np.testing.assert_array_equal(si, [0, 1])


def test_homography_keeps_exact_high_resolution_matches() -> None:
    source = np.asarray([[0, 0], [100, 0], [0, 100], [100, 100], [50, 20], [20, 50]], float)
    query = source * np.asarray([1.2, 0.8]) + np.asarray([7, 11])
    assert _homography_inliers(query, source).all()


def test_mapping_contributor_cache_is_numerically_transparent(tmp_path) -> None:
    contributor = tmp_path / "source.npz"
    np.savez(
        contributor,
        dominant_depth=np.full((144, 256), 2.0, np.float32),
        pose_w2c=np.eye(4, dtype=np.float64),
        camera_model_id=np.asarray(0),
        camera_width=np.asarray(1024),
        camera_height=np.asarray(576),
        camera_params=np.asarray([400.0, 511.5, 287.5]),
    )
    labels = np.full((144, 256), 3, np.int32)
    point = np.asarray([511.5, 287.5])
    cache = {}
    first = _mapping_world_point(point, labels, 3, contributor, cache)
    contributor.unlink()
    second = _mapping_world_point(point, labels, 3, contributor, cache)
    np.testing.assert_array_equal(first, second)
    # The 3x3 median support is centred on integer low-resolution pixel 128,
    # while the resized principal point is at half-pixel 127.5.
    np.testing.assert_allclose(first, [0.01, 0.01, 2.0], atol=1e-12)
