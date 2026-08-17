from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.resident_surface_renderer import (
    FrozenSoftSurfaceSceneGPU,
)
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    _RAW_TO_IDEAL_TOKEN_WARP_CACHE,
    _raw_to_ideal_token_warp,
    _remap_ideal_hits_to_raw_tokens,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(0, 2, 8, 8, (6.0, 4.0, 4.0, 0.04))


def test_raw_to_ideal_warp_is_camera_bound_and_cached():
    _RAW_TO_IDEAL_TOKEN_WARP_CACHE.clear()
    camera = _camera()
    first = _raw_to_ideal_token_warp(
        camera, token_width=2, token_height=2, supersample_factor=2
    )
    second = _raw_to_ideal_token_warp(
        camera, token_width=2, token_height=2, supersample_factor=2
    )
    assert first is second
    assert first.source_ideal_pixel_ids.flags.writeable is False
    assert first.destination_token_pixel_ids.flags.writeable is False
    changed = ColmapCamera(0, 2, 8, 8, (6.0, 4.0, 4.0, 0.05))
    third = _raw_to_ideal_token_warp(
        changed, token_width=2, token_height=2, supersample_factor=2
    )
    assert third is not first
    assert len(_RAW_TO_IDEAL_TOKEN_WARP_CACHE) == 2


def test_raw_warp_accepts_all_declared_pinhole_camera_models():
    simple = ColmapCamera(0, 0, 8, 8, (6.0, 4.0, 4.0))
    pinhole = ColmapCamera(0, 1, 8, 8, (6.0, 5.5, 4.0, 4.0))
    for camera in (simple, pinhole, _camera()):
        warp = _raw_to_ideal_token_warp(
            camera, token_width=2, token_height=2, supersample_factor=2
        )
        assert warp.source_ideal_pixel_ids.size > 0
        assert warp.source_ideal_pixel_ids.shape == warp.destination_token_pixel_ids.shape


def test_batched_raw_token_remap_matches_independent_scalar_remap():
    camera = _camera()
    # Two primitive hits at every ideal pixel.  The arrays are already in the
    # global-pixel order emitted by the resident compositor.
    local_pixels = np.repeat(np.arange(16, dtype=np.int64), 2)
    local_rows = np.tile(np.asarray([3, 7], dtype=np.int64), 16)
    local_weights = np.tile(np.asarray([0.6, 0.2], dtype=np.float32), 16)
    global_pixels = np.concatenate([local_pixels, local_pixels + 16])
    rows = np.concatenate([local_rows, local_rows])
    weights = np.concatenate([local_weights, local_weights])
    token, batch_rows, batch_weights = FrozenSoftSurfaceSceneGPU._batch_token_remap(
        global_pixels, rows, weights, camera, batch_size=2,
        token_width=2, token_height=2, supersample_factor=2,
    )
    for batch in range(2):
        scalar_token, scalar_rows, scalar_weights = _remap_ideal_hits_to_raw_tokens(
            local_pixels, local_rows, local_weights, camera,
            token_width=2, token_height=2, supersample_factor=2,
        )
        mask = (token >= batch * 4) & (token < (batch + 1) * 4)
        np.testing.assert_array_equal(token[mask] - batch * 4, scalar_token)
        np.testing.assert_array_equal(batch_rows[mask], scalar_rows)
        np.testing.assert_array_equal(batch_weights[mask], scalar_weights)


def test_batched_raw_token_remap_rejects_unsorted_hits():
    with np.testing.assert_raises_regex(ValueError, "globally sorted"):
        FrozenSoftSurfaceSceneGPU._batch_token_remap(
            np.asarray([2, 1], dtype=np.int64),
            np.asarray([3, 4], dtype=np.int64),
            np.asarray([0.2, 0.1], dtype=np.float32),
            _camera(), batch_size=1, token_width=2, token_height=2,
            supersample_factor=2,
        )
