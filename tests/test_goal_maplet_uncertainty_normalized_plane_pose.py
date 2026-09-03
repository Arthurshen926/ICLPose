import numpy as np

from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _projection_variance_px2,
    _select,
    _uncertainty_normalized_token_likelihood,
)


def test_projection_covariance_and_density_penalize_uncertainty():
    xyz = np.asarray([[0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    assert _projection_variance_px2(xyz, np.zeros((1, 3, 3)), np.eye(3), K)[0] == 0.0
    pose = np.eye(4)
    token = np.asarray([0])
    clean, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz, token, np.zeros((1, 3, 3)), np.ones(1), K, 0.0,
    )
    uncertain, _ = _uncertainty_normalized_token_likelihood(
        pose, xyz, token, np.eye(3)[None], np.ones(1), K, 0.0,
    )
    assert clean == 1.0
    assert uncertain < clean


def test_purity_and_per_token_marginalization():
    pose = np.eye(4)
    xyz = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    score, token = _uncertainty_normalized_token_likelihood(
        pose, xyz, np.asarray([0, 0]), np.zeros((2, 3, 3)), np.asarray([0.5, 0.8]), K, 0.0,
    )
    assert token.tolist() == [0.8]
    assert score == 0.8


def test_selection_is_stable():
    score = np.asarray([[[0.5, 0.5], [0.6, 0.6]], [[0.5, 0.5], [0.5, 0.5]]])
    assert _select(score).tolist() == [1, 0]
