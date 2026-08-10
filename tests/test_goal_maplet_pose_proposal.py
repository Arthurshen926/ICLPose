from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.child_retrieval import ChildTilePosterior
from feature_extract.vfm.localization_goal_maplet.pose_proposal import (
    CoarsePoseModes,
    _score_pose,
    _select_child_for_parent,
)


def test_pose_mode_contract_accepts_empty_modes():
    result = CoarsePoseModes(
        poses_w2c=np.zeros((0, 4, 4), dtype=np.float64),
        scores=np.zeros((0,), dtype=np.float64),
        supporting_region_count=np.zeros((0,), dtype=np.int64),
    )
    assert result.poses_w2c.shape == (0, 4, 4)


def test_pose_mode_contract_rejects_misaligned_values():
    try:
        CoarsePoseModes(np.zeros((1, 4, 4)), np.zeros((2,)), np.zeros((1,)))
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned pose modes must fail")


def test_parent_conditioned_child_enumeration_does_not_depend_on_score_weight():
    posterior = ChildTilePosterior(
        candidate_child_rows=np.asarray([[1, 0]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.8, 0.2]], dtype=np.float32),
        null_probabilities=np.asarray([0.0], dtype=np.float32),
        conditional_parent_ids=np.asarray([[10, 20]], dtype=np.int64),
        conditional_parent_log_evidence=np.asarray([[3.0, -2.0]], dtype=np.float32),
        best_child_rows_by_parent=np.asarray([[0, 1]], dtype=np.int64),
        best_child_probabilities_by_parent=np.asarray([[0.75, 0.6]], dtype=np.float32),
    )
    physical = SimpleNamespace(child_parent_rows=np.asarray([0, 1], dtype=np.int64))
    child, probability = _select_child_for_parent(
        posterior,
        physical,
        support=0,
        parent_slot=0,
        parent_row=0,
        parent_probability=0.4,
    )
    assert child == 0
    assert np.isclose(probability, 0.3)


def test_pose_scoring_returns_child_rows_aligned_to_all_query_supports():
    count = 100
    posterior = ChildTilePosterior(
        candidate_child_rows=np.zeros((count, 1), dtype=np.int64),
        candidate_probabilities=np.ones((count, 1), dtype=np.float32),
        null_probabilities=np.zeros((count,), dtype=np.float32),
    )
    physical = SimpleNamespace(
        child_centers=np.asarray([[0.0, 0.0, 5.0]]),
        child_normals=np.asarray([[0.0, 0.0, -1.0]]),
        child_parent_rows=np.asarray([0], dtype=np.int64),
        maplet_sidedness=np.asarray([2], dtype=np.uint8),
    )
    camera = ColmapCamera(
        camera_id=0,
        model_id=0,
        width=640,
        height=480,
        params=(500.0, 320.0, 240.0),
    )
    _score, _support, chosen = _score_pose(
        np.eye(4),
        np.tile(np.asarray([[320.0, 240.0]]), (count, 1)),
        np.full((count, 2), 20.0),
        posterior,
        physical,
        camera,
        maximum_regions=96,
    )
    assert chosen.shape == (count,)
    assert np.all(chosen[:96] == 0)
    assert np.all(chosen[96:] == -1)
