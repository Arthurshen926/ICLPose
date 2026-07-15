from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.mine_pose_conditioned_system_hard_negatives import (
    PoseConditionedHardNegativeConfig,
    mine_query_system_error_modes,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera


def _repeated_shift_fixture():
    camera = ColmapCamera(
        camera_id=1,
        model_id=1,
        width=1000,
        height=800,
        params=(1000.0, 1000.0, 500.0, 400.0),
    )
    correct_xyz = np.asarray(
        [
            [-3.0, -2.0, 10.0],
            [-1.0, -2.0, 10.0],
            [1.0, -2.0, 10.0],
            [-3.0, 2.0, 10.0],
            [-1.0, 2.0, 10.0],
            [1.0, 2.0, 10.0],
        ],
        dtype=np.float64,
    )
    query_xy = np.stack(
        [
            1000.0 * correct_xyz[:, 0] / correct_xyz[:, 2] + 500.0,
            1000.0 * correct_xyz[:, 1] / correct_xyz[:, 2] + 400.0,
        ],
        axis=1,
    )
    wrong_xyz = correct_xyz.copy()
    wrong_xyz[:, 0] -= 1.0
    candidate_xyz = np.stack([correct_xyz, wrong_xyz], axis=1)
    valid = np.ones((len(query_xy), 2), dtype=bool)
    positive = np.zeros_like(valid)
    positive[:, 0] = True
    scores = np.tile(np.asarray([0.85, 0.90]), (len(query_xy), 1))
    gt_pose = np.eye(4, dtype=np.float64)
    bad_pose = np.eye(4, dtype=np.float64)
    bad_pose[0, 3] = 1.0
    return camera, query_xy, candidate_xyz, valid, positive, scores, gt_pose, bad_pose


def test_pose_conditioned_mining_recovers_coherent_repeated_shift() -> None:
    camera, xy, xyz, valid, positive, scores, gt_pose, bad_pose = (
        _repeated_shift_fixture()
    )
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=np.stack([gt_pose, bad_pose]),
        hypothesis_scores=np.asarray([-0.4, -0.2]),
        hypothesis_translation_errors_m=np.asarray([0.0, 1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0, 0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_translation_m=0.25,
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            min_consistent_groups=6,
            min_consistent_grid_cells=3,
        ),
    )

    expected = np.zeros_like(valid)
    expected[:, 1] = True
    np.testing.assert_array_equal(result["hard_negative_mask"], expected)
    np.testing.assert_array_equal(result["group_hard_mask"], np.ones(6, dtype=bool))
    assert len(result["selected_modes"]) == 1
    assert result["selected_modes"][0]["consistent_group_count"] == 6
    assert result["selected_mode_hard_masks"].shape == (1, 6, 2)
    np.testing.assert_array_equal(
        result["selected_mode_hard_masks"][0], expected
    )
    assert np.max(np.sum(result["hard_negative_mask"], axis=1)) == 1


def test_pose_conditioned_mining_requires_score_plausible_wrong_identity() -> None:
    camera, xy, xyz, valid, positive, scores, _, bad_pose = (
        _repeated_shift_fixture()
    )
    scores[:, 1] = 0.2
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=bad_pose[None],
        hypothesis_scores=np.asarray([-0.2]),
        hypothesis_translation_errors_m=np.asarray([1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            hard_score_log_margin=0.7,
            min_consistent_groups=6,
            min_consistent_grid_cells=3,
        ),
    )

    assert not np.any(result["hard_negative_mask"])
    assert result["selected_modes"] == []
    assert result["selected_mode_hard_masks"].shape == (0, 6, 2)


def test_pose_conditioned_mining_never_labels_no_positive_group() -> None:
    camera, xy, xyz, valid, positive, scores, _, bad_pose = (
        _repeated_shift_fixture()
    )
    positive[0] = False
    result = mine_query_system_error_modes(
        query_xy=xy,
        candidate_xyz=xyz,
        candidate_scores=scores,
        valid_mask=valid,
        positive_mask=positive,
        hypothesis_poses_w2c=bad_pose[None],
        hypothesis_scores=np.asarray([-0.2]),
        hypothesis_translation_errors_m=np.asarray([1.0]),
        hypothesis_rotation_errors_deg=np.asarray([0.0]),
        camera=camera,
        config=PoseConditionedHardNegativeConfig(
            min_bad_rotation_deg=0.0,
            bad_pose_consistency_px=1.0,
            min_consistent_groups=5,
            min_consistent_grid_cells=3,
        ),
    )

    assert not np.any(result["hard_negative_mask"][0])
    assert not result["group_hard_mask"][0]
    assert np.all(result["hard_negative_mask"][1:, 1])
