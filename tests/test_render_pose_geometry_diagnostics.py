from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.render_pose_diagnostics import (
    render_depth_roundtrip_stats,
    synthetic_render_lock_stats,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))


def _pose_w2c_from_center(center: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64).reshape(3)
    return pose


def test_render_depth_roundtrip_reprojects_back_to_input_pixels() -> None:
    camera = _camera()
    pose_w2c = _pose_w2c_from_center(np.asarray([0.3, -0.2, 0.1], dtype=np.float64))
    xy = np.asarray(
        [
            [320.0, 240.0],
            [360.0, 240.0],
            [300.0, 260.0],
            [410.0, 180.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    stats = render_depth_roundtrip_stats(xy, depth, camera, pose_w2c)

    assert stats["count"] == 4
    assert stats["valid_fraction"] == 1.0
    assert stats["median_roundtrip_error_px"] < 1e-6
    assert stats["p95_roundtrip_error_px"] < 1e-6


def test_synthetic_render_lock_solves_back_to_render_pose() -> None:
    camera = _camera()
    render_pose_w2c = _pose_w2c_from_center(np.asarray([0.25, 0.0, 0.0], dtype=np.float64))
    xy = np.asarray(
        [
            [260.0, 190.0],
            [380.0, 190.0],
            [260.0, 290.0],
            [380.0, 290.0],
            [320.0, 160.0],
            [420.0, 260.0],
            [220.0, 260.0],
            [340.0, 330.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5], dtype=np.float64)

    stats = synthetic_render_lock_stats(xy, depth, camera, render_pose_w2c)

    assert stats["pnp_success"] is True
    assert stats["pnp_render_translation_delta_m"] < 1e-4
    assert stats["pnp_render_rotation_delta_deg"] < 1e-4
