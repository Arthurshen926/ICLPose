from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization_goal_maplet.multibasin_pattern_search import (
    _probe_basin,
    batched_multibasin_pattern_search,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    left_retract_pose_w2c,
)


def _pose(center):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64)
    return pose


def test_batched_multibasin_search_converges_without_collapsing_modes():
    calls = []

    def score(poses):
        calls.append(int(poses.shape[0]))
        centers = -np.swapaxes(poses[:, :3, :3], 1, 2) @ poses[:, :3, 3, None]
        centers = centers[..., 0]
        targets = np.asarray([[1.0, -0.5, 0.25], [8.0, 0.5, -0.25]])
        return -np.min(
            np.sum((centers[:, None, :] - targets[None, :, :]) ** 2, axis=2),
            axis=1,
        )

    result = batched_multibasin_pattern_search(
        np.asarray([_pose([0.0, 0.0, 0.0]), _pose([7.0, 0.0, 0.0])]),
        score,
        translation_radius_m=1.0,
        rotation_radius_deg=5.0,
        minimum_translation_radius_m=0.125,
        minimum_rotation_radius_deg=0.625,
        maximum_sweeps=12,
    )
    assert len(result.basins) == 2
    assert result.evaluator_calls == len(calls)
    assert all(count == 26 for count in calls[1:])
    centers = np.asarray([
        -state.pose_w2c[:3, :3].T @ state.pose_w2c[:3, 3]
        for state in sorted(result.basins, key=lambda state: state.source_basin_index)
    ])
    np.testing.assert_allclose(
        centers, [[1.0, -0.5, 0.25], [8.0, 0.5, -0.25]], atol=0.13
    )
    assert all(state.accepted_updates > 0 for state in result.basins)


def test_pattern_search_rejects_nonfinite_scores():
    try:
        batched_multibasin_pattern_search(
            np.asarray([np.eye(4)]),
            lambda poses: np.full((poses.shape[0],), np.nan),
        )
    except ValueError as error:
        assert "scorer output" in str(error)
    else:
        raise AssertionError("nonfinite score must fail closed")


def test_pattern_search_probes_use_the_shared_left_se3_retraction():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray([
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    ])
    pose[:3, 3] = [0.3, -0.2, 1.1]
    probes = _probe_basin(pose, 0.75, 11.0)
    assert probes.shape == (13, 4, 4)
    np.testing.assert_array_equal(probes[0], pose)
    eye = np.eye(6, dtype=np.float64)
    expected = [pose]
    for axis in eye:
        for sign in (1.0, -1.0):
            expected.append(left_retract_pose_w2c(
                pose, sign * axis,
                translation_step_m=0.75,
                rotation_step_degrees=11.0,
            ))
    np.testing.assert_allclose(probes, np.asarray(expected), atol=0.0, rtol=0.0)
