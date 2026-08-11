from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.child_retrieval import ChildTilePosterior
from feature_extract.vfm.localization_goal_maplet.geometry_guided_pose_proposal import (
    scale_invariant_configuration_geometry,
)
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


def test_pose_mode_contract_preserves_configuration_provenance():
    result = CoarsePoseModes(
        np.repeat(np.eye(4)[None], 2, axis=0),
        np.asarray([2.0, 1.0]),
        np.asarray([8, 7]),
        np.asarray([[1, 2, -1], [3, 4, 5]]),
        np.asarray([[10, 20, -1], [30, 40, 50]]),
        np.asarray([[1, 2], [3, -1]]),
        np.asarray([[5, 9], [2, -1]]),
    )
    assert result.configuration_parent_rows.shape == (2, 3)
    assert result.configuration_child_rows[1, 2] == 50
    assert result.proposal_seed_parent_rows[0].tolist() == [1, 2]
    assert result.proposal_seed_support_rows[0].tolist() == [5, 9]


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


def test_pose_scoring_never_refines_an_unexplained_zero_likelihood_row():
    posterior = ChildTilePosterior(
        candidate_child_rows=np.asarray([[0]], dtype=np.int64),
        candidate_probabilities=np.asarray([[1.0]], dtype=np.float32),
        null_probabilities=np.asarray([0.0], dtype=np.float32),
    )
    physical = SimpleNamespace(
        child_centers=np.asarray([[0.0, 0.0, -5.0]]),
        child_normals=np.asarray([[0.0, 0.0, 1.0]]),
        child_parent_rows=np.asarray([0], dtype=np.int64),
        maplet_sidedness=np.asarray([2], dtype=np.uint8),
    )
    camera = ColmapCamera(
        camera_id=0, model_id=0, width=640, height=480,
        params=(500.0, 320.0, 240.0),
    )
    _score, support, chosen = _score_pose(
        np.eye(4), np.asarray([[320.0, 240.0]]), np.asarray([[20.0, 20.0]]),
        posterior, physical, camera,
    )
    assert support == 0
    assert chosen.tolist() == [-1]


def test_configuration_geometry_is_scale_invariant_and_rejects_shape_distortion():
    centers = np.asarray([
        [-1.0, -0.5, 4.0],
        [1.0, -0.5, 5.0],
        [-0.5, 1.0, 6.0],
        [1.2, 0.8, 7.0],
    ])
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (4, 1))
    confidence = np.ones((4,), dtype=np.float64)
    assigned = np.arange(4, dtype=np.int64)
    reference = scale_invariant_configuration_geometry(
        np.eye(4), 2.7 * centers, normals, confidence, assigned, centers, normals,
    )
    distorted = 2.7 * centers
    distorted[3, 2] += 6.0
    mismatch = scale_invariant_configuration_geometry(
        np.eye(4), distorted, normals, confidence, assigned, centers, normals,
    )
    assert np.isclose(reference["scale"], 2.7)
    assert reference["score"] > 0.99
    assert mismatch["score"] < reference["score"] - 0.1


def test_configuration_geometry_requires_three_physical_children():
    result = scale_invariant_configuration_geometry(
        np.eye(4),
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.ones((2,)),
        np.asarray([0, 1]),
        np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert result["score"] is None
