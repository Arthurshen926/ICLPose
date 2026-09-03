from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_plane_pnp_observation_bank import (
    _plane_token_world_statistics,
)


def test_plane_token_geometry_excludes_foreground_pixels(tmp_path) -> None:
    depth = np.full((144, 256), 10.0, np.float32)
    mask = np.zeros((144, 256), bool); mask[:2, :4] = True; depth[mask] = 2.0
    contributor = tmp_path / "view.npz"
    np.savez(
        contributor, dominant_depth=depth, pose_w2c=np.eye(4),
        camera_model_id=np.asarray(0, np.int32), camera_width=np.asarray(256, np.int32),
        camera_height=np.asarray(144, np.int32),
        camera_params=np.asarray([100.0, 127.5, 71.5]),
    )
    point, covariance, purity, dispersion, kept = _plane_token_world_statistics(
        contributor, np.asarray([0]), mask, token_grid=(36, 64),
    )
    np.testing.assert_array_equal(kept, [0])
    np.testing.assert_allclose(point[:, 2], [2.0])
    np.testing.assert_allclose(purity, [0.5])
    np.testing.assert_allclose(dispersion, [0.0])
    assert covariance.shape == (1, 3, 3)
