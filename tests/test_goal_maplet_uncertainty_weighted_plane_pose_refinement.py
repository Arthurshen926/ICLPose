import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    _fixed_reprojection_sigma_px,
    _fixed_token_hypotheses,
    _pose_step,
    _refine_pose,
)


def test_fixed_hypotheses_use_one_best_row_per_token():
    pose = np.eye(4)
    world = np.asarray([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [1.0, 0.0, -1.0]])
    token = np.asarray([0, 0, 1])
    pixel = np.asarray([[1.5, 1.5], [1.5, 1.5], [5.5, 1.5]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    rows, error = _fixed_token_hypotheses(pose, world, token, pixel, K, 0.0)
    assert rows.tolist() == [0]
    assert error[0] < error[1]


def test_uncertainty_sigma_uses_covariance_dispersion_and_purity():
    pose = np.eye(4)
    world = np.asarray([[0.0, 0.0, 2.0]])
    K = np.asarray([[4.0, 0.0, 1.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]])
    clean = _fixed_reprojection_sigma_px(
        pose, world, np.zeros((1, 3, 3)), np.ones(1), np.zeros(1), K,
    )
    covariance = _fixed_reprojection_sigma_px(
        pose, world, np.eye(3)[None], np.ones(1), np.zeros(1), K,
    )
    dispersion = _fixed_reprojection_sigma_px(
        pose, world, np.zeros((1, 3, 3)), np.ones(1), np.ones(1), K,
    )
    impure = _fixed_reprojection_sigma_px(
        pose, world, np.zeros((1, 3, 3)), np.asarray([0.25]), np.zeros(1), K,
    )
    assert covariance[0] > clean[0]
    assert dispersion[0] > clean[0]
    assert impure[0] == 2.0 * clean[0]


def test_refinement_is_bounded_and_improves_weighted_reprojection():
    # Non-coplanar synthetic map with a small, known camera translation error.
    rng = np.random.default_rng(260903)
    world = rng.uniform([-1.0, -0.8, 3.0], [1.0, 0.8, 6.0], size=(24, 3))
    K = np.asarray([[300.0, 0.0, 127.5], [0.0, 300.0, 71.5], [0.0, 0.0, 1.0]])
    true_pose = np.eye(4)
    projected = np.c_[
        K[0, 0] * world[:, 0] / world[:, 2] + K[0, 2],
        K[1, 1] * world[:, 1] / world[:, 2] + K[1, 2],
    ]
    initial = np.eye(4); initial[0, 3] = 0.02
    output, accepted, detail = _refine_pose(
        initial, world, projected, np.arange(24) % 2, np.ones(24), K, 0.0,
    )
    rotation_step, center_step = _pose_step(initial, output)
    assert accepted
    assert detail["final_weighted_huber_objective"] < detail["initial_weighted_huber_objective"]
    assert rotation_step <= 5.0 and center_step <= 0.5
    assert np.linalg.norm(output - true_pose) < np.linalg.norm(initial - true_pose)


def test_refinement_fails_closed_for_one_plane():
    world = np.c_[np.linspace(-1.0, 1.0, 12), np.zeros(12), np.full(12, 4.0)]
    K = np.asarray([[100.0, 0.0, 1.5], [0.0, 100.0, 1.5], [0.0, 0.0, 1.0]])
    pixel = np.c_[100.0 * world[:, 0] / 4.0 + 1.5, np.full(12, 1.5)]
    output, accepted, detail = _refine_pose(
        np.eye(4), world, pixel, np.zeros(12, np.int64), np.ones(12), K, 0.0,
    )
    assert not accepted and not detail["solver_attempted"]
    np.testing.assert_array_equal(output, np.eye(4))


def test_pose_step_uses_camera_center_and_rotation_magnitude():
    initial = np.eye(4)
    final = np.eye(4)
    angle = np.deg2rad(3.0)
    final[:3, :3] = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    final[:3, 3] = -final[:3, :3] @ np.asarray([0.2, 0.0, 0.0])
    rotation, translation = _pose_step(initial, final)
    assert abs(rotation - 3.0) < 1e-10
    assert abs(translation - 0.2) < 1e-10
