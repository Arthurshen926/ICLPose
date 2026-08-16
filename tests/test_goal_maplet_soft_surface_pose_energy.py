from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    rotate_camera_local,
    score_soft_surface_pose_energy,
    translate_camera_world,
)
from test_goal_maplet_pure_retrieval import _metadata
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    PureRadioPhysicalRetrieval,
    all_radio_token_coordinates,
)


def _retrieval() -> PureRadioPhysicalRetrieval:
    xy = all_radio_token_coordinates(2, 2)
    return PureRadioPhysicalRetrieval(
        image_id="q", token_xy=xy,
        token_parent_ids=np.zeros((4, 1), dtype=np.int64),
        token_parent_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        token_out_of_map_probabilities=np.full(4, 0.1, dtype=np.float32),
        token_in_map_tail_probabilities=np.full(4, 0.1, dtype=np.float32),
        token_child_rows=np.asarray([[0], [1], [0], [1]], dtype=np.int64),
        token_child_probabilities=np.full((4, 1), 0.8, dtype=np.float32),
        scene_parent_ids=np.asarray([0]), scene_parent_scores=np.asarray([1.0]),
        scene_child_rows=np.asarray([0, 1]), scene_child_scores=np.asarray([1.0, 1.0]),
        physical_map_sha256="p", metadata=_metadata(),
    )


def test_soft_pose_energy_marginalizes_child_and_missing_cannot_improve():
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[0] = 1.0
    correct = SimpleNamespace(
        feature=query.copy(), child_id=np.asarray([[0, 1], [0, 1]]),
        visibility=np.ones((2, 2), dtype=bool), mask=np.ones((2, 2), dtype=bool),
    )
    wrong = SimpleNamespace(
        feature=np.stack([np.zeros((2, 2)), np.ones((2, 2))]),
        child_id=np.asarray([[1, 0], [1, 0]]),
        visibility=np.ones((2, 2), dtype=bool), mask=np.ones((2, 2), dtype=bool),
    )
    missing = SimpleNamespace(
        feature=np.zeros_like(query), child_id=np.full((2, 2), -1),
        visibility=np.zeros((2, 2), dtype=bool), mask=np.zeros((2, 2), dtype=bool),
    )
    correct_score = score_soft_surface_pose_energy(query, _retrieval(), correct)
    wrong_score = score_soft_surface_pose_energy(query, _retrieval(), wrong)
    missing_score = score_soft_surface_pose_energy(query, _retrieval(), missing)
    assert correct_score.combined_score == 1.0
    assert wrong_score.combined_score == -0.5
    assert missing_score.combined_score == -1.0


def test_pose_perturbations_preserve_camera_and_se3():
    pose = np.eye(4, dtype=np.float64)
    moved = translate_camera_world(pose, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(-moved[:3, :3].T @ moved[:3, 3], [1.0, 2.0, 3.0])
    rotated = rotate_camera_local(moved, np.asarray([0.0, 1.0, 0.0]), 10.0)
    np.testing.assert_allclose(rotated[:3, :3] @ rotated[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(-rotated[:3, :3].T @ rotated[:3, 3], [1.0, 2.0, 3.0])
