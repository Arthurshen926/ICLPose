from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.anchors import build_surface_anchors
from feature_extract.vfm.measurement_v1.types import QueryMeasurement, SurfaceAnchor


def _camera(width: int = 8, height: int = 8) -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=width, height=height, params=(4.0, 4.0, width / 2, height / 2))


def test_surface_anchor_world_xyz_is_immutable_from_query_measurement() -> None:
    anchor = SurfaceAnchor(
        anchor_id=7,
        token_index=3,
        subanchor_index=0,
        render_xy_px=np.asarray([4.0, 4.0], dtype=np.float64),
        world_xyz=np.asarray([0.0, 0.0, 5.0], dtype=np.float64),
        cov_world_3x3=np.eye(3, dtype=np.float64) * 0.01,
        depth_m=5.0,
        alpha=1.0,
        quality=0.9,
        normal_world=None,
        surface_id=None,
    )
    before = anchor.world_xyz.copy()
    measurement = QueryMeasurement(
        anchor_id=7,
        query_xy_mean_px=np.asarray([5.25, 3.75], dtype=np.float64),
        cov_query_2x2=np.eye(2, dtype=np.float64),
        p_visible=1.0,
        p_assignment=0.8,
        local_log_likelihood=None,
        mode_probability=0.7,
        diagnostics={"offset_x": 1.25},
    )

    assert measurement.anchor_id == anchor.anchor_id
    assert np.allclose(anchor.world_xyz, before, atol=0.0)
    assert anchor.world_xyz.flags.writeable is False


def test_build_surface_anchors_filters_depth_edges_and_backprojects_once() -> None:
    depth = np.full((8, 8), 4.0, dtype=np.float32)
    depth[:, 4:] = 9.0
    alpha = np.ones((8, 8), dtype=np.float32)

    anchors, rows = build_surface_anchors(
        depth,
        alpha,
        _camera(),
        np.eye(4, dtype=np.float64),
        token_grid_width=2,
        token_grid_height=2,
        subanchors_per_token=1,
        max_anchors=4,
        min_alpha=0.5,
        max_depth_gradient=1.0,
    )

    assert anchors
    assert all(anchor.world_xyz.flags.writeable is False for anchor in anchors)
    assert all(abs(float(anchor.render_xy_px[0]) - 4.0) > 0.5 for anchor in anchors)
    assert any(row["rejection_reason"] == "depth_gradient" for row in rows)
