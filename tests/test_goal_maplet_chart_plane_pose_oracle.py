import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_chart_plane_pose_oracle import (
    _camera_points,
    _robust_point_pose,
    _umeyama,
)


def test_umeyama_recovers_similarity_without_target_pose_input() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(128, 3))
    rotation = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    scale = 1.37
    translation = np.asarray([2.0, -3.0, 0.5])
    target = scale * (source @ rotation.T) + translation
    estimate_rotation, estimate_translation, estimate_scale = _umeyama(
        source, target, estimate_scale=True,
    )
    np.testing.assert_allclose(estimate_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(estimate_translation, translation, atol=1e-12)
    assert abs(estimate_scale - scale) < 1e-12


def test_robust_similarity_rejects_sparse_bad_planar_support() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(256, 3))
    rotation = Rotation.from_rotvec([-0.12, 0.08, 0.04]).as_matrix()
    scale = 0.87
    translation = np.asarray([-1.0, 2.5, 3.0])
    target = scale * (source @ rotation.T) + translation
    target[:24] += 5.0
    estimate_rotation, estimate_translation, estimate_scale, inlier, _ = _robust_point_pose(
        source, target, estimate_scale=True,
    )
    np.testing.assert_allclose(estimate_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(estimate_translation, translation, atol=1e-10)
    assert abs(estimate_scale - scale) < 1e-10
    assert not inlier[:24].any()


def test_camera_points_use_explicit_pixel_center_principal_point() -> None:
    depth = np.full((3, 5), 2.0)
    points = _camera_points(
        depth,
        np.asarray([4.0, 4.0]),
        np.asarray([2.0, 1.0]),
    )
    np.testing.assert_allclose(points[1, 2], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(points[1, 4], [1.0, 0.0, 2.0])
