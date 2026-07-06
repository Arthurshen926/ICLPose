from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.probabilistic_pnp import estimate_pose_from_measurements
from feature_extract.vfm.measurement_v1.types import QueryMeasurement, SurfaceAnchor
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _anchor(anchor_id: int, xy: tuple[float, float], z: float) -> SurfaceAnchor:
    x = (float(xy[0]) - 50.0) / 80.0 * z
    y = (float(xy[1]) - 50.0) / 80.0 * z
    return SurfaceAnchor(
        anchor_id=anchor_id,
        token_index=anchor_id,
        subanchor_index=0,
        render_xy_px=np.asarray(xy, dtype=np.float64),
        world_xyz=np.asarray([x, y, z], dtype=np.float64),
        cov_world_3x3=np.eye(3, dtype=np.float64) * 1e-8,
        depth_m=z,
        alpha=1.0,
        quality=1.0,
        normal_world=None,
        surface_id=None,
    )


def test_covariance_pnp_recovers_noiseless_pose() -> None:
    xy_values = [(30.0, 30.0), (70.0, 30.0), (30.0, 70.0), (70.0, 70.0), (50.0, 50.0), (60.0, 45.0)]
    anchors = [_anchor(idx, xy, 4.0 + idx * 0.2) for idx, xy in enumerate(xy_values)]
    measurements = [
        QueryMeasurement(
            anchor_id=anchor.anchor_id,
            query_xy_mean_px=anchor.render_xy_px,
            cov_query_2x2=np.eye(2, dtype=np.float64) * 0.25,
            p_visible=1.0,
            p_assignment=1.0,
            local_log_likelihood=None,
            mode_probability=1.0,
            diagnostics={},
        )
        for anchor in anchors
    ]

    pose = estimate_pose_from_measurements(anchors, measurements, _camera(), use_covariance=True)
    error = pnp_pose_error(pose.pose_w2c, np.eye(4, dtype=np.float64))

    assert pose.success
    assert error.translation_m < 1e-5
    assert error.rotation_deg < 1e-4
    assert pose.pose_cov_6x6 is not None


def test_covariance_pnp_downweights_anisotropic_bad_measurement() -> None:
    xy_values = [(25.0, 25.0), (75.0, 25.0), (25.0, 75.0), (75.0, 75.0), (50.0, 25.0), (50.0, 75.0)]
    anchors = [_anchor(idx, xy, 5.0 + idx * 0.3) for idx, xy in enumerate(xy_values)]
    measurements = []
    for idx, anchor in enumerate(anchors):
        xy = anchor.render_xy_px.copy()
        cov = np.eye(2, dtype=np.float64) * 0.25
        if idx == 0:
            xy += np.asarray([12.0, 0.0], dtype=np.float64)
            cov = np.asarray([[400.0, 0.0], [0.0, 0.25]], dtype=np.float64)
        measurements.append(
            QueryMeasurement(
                anchor_id=anchor.anchor_id,
                query_xy_mean_px=xy,
                cov_query_2x2=cov,
                p_visible=1.0,
                p_assignment=1.0,
                local_log_likelihood=None,
                mode_probability=1.0,
                diagnostics={},
            )
        )

    uniform = estimate_pose_from_measurements(anchors, measurements, _camera(), use_covariance=False)
    weighted = estimate_pose_from_measurements(anchors, measurements, _camera(), use_covariance=True)
    uniform_error = pnp_pose_error(uniform.pose_w2c, np.eye(4, dtype=np.float64))
    weighted_error = pnp_pose_error(weighted.pose_w2c, np.eye(4, dtype=np.float64))

    assert weighted.success and uniform.success
    assert weighted_error.translation_m < uniform_error.translation_m
