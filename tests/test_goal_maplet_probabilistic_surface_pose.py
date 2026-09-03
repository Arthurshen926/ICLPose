import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_probabilistic_surface_pose import (
    _candidate_pose,
    _responsibilities,
)


def test_responsibilities_normalize_per_token_including_null():
    token = np.asarray([0, 0, 1])
    responsibility, null, _ = _responsibilities(token, np.ones(3))
    assert abs(float(np.sum(responsibility[:2])) + float(null[0]) - 1.0) < 1e-12
    assert abs(float(responsibility[2]) + float(null[1]) - 1.0) < 1e-12
    np.testing.assert_allclose(responsibility[0], responsibility[1])


def test_extra_identical_hypothesis_does_not_increase_total_match_prior():
    one, null_one, _ = _responsibilities(np.asarray([0]), np.asarray([1.0]))
    two, null_two, _ = _responsibilities(np.asarray([0, 0]), np.asarray([1.0, 1.0]))
    np.testing.assert_allclose(one.sum(), two.sum())
    np.testing.assert_allclose(null_one, null_two)


def test_candidate_pose_is_left_se3_update():
    initial = np.eye(4); initial[:3, 3] = [1.0, 0.0, 0.0]
    parameter = np.asarray([0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    output = _candidate_pose(initial, parameter)
    np.testing.assert_allclose(output[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)
