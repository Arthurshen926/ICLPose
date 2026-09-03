import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_continuous_coordinate_pose_oracle import (
    _inside_token,
    _oracle_candidate_rows,
    _pose_error,
)


def test_candidate_oracle_selects_one_nearest_hypothesis_per_token():
    pose = np.eye(4)
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    world = np.asarray([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0], [2.0, 0.0, -1.0]])
    token = np.asarray([0, 0, 1])
    pixel = np.asarray([[1.5, 1.5], [1.5, 1.5], [5.5, 1.5]])
    rows, projected, error = _oracle_candidate_rows(pose, world, token, pixel, K, 0.0)
    assert rows.tolist() == [0]
    assert error[0] < error[1]
    assert projected.shape == (3, 2)


def test_inside_token_rejects_neighboring_token_projection():
    token = np.asarray([0, 1, 64])
    point = np.asarray([[1.5, 1.5], [3.9, 1.5], [1.5, 4.0]])
    assert _inside_token(point, token).tolist() == [True, False, True]


def test_pose_error_is_zero_for_identical_pose():
    assert _pose_error(np.eye(4), np.eye(4)) == (0.0, 0.0)
    assert _pose_error(None, np.eye(4)) == (float("inf"), float("inf"))
