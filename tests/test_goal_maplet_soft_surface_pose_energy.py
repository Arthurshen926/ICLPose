from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.soft_surface_pose_energy import (
    rotate_camera_local,
    score_hierarchical_spatial_soft_surface_pose_energy,
    score_bidirectional_soft_surface_pose_energy,
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
    axis = np.asarray([0.0, 2.0, 0.0])
    axis_before = axis.copy()
    rotated = rotate_camera_local(moved, axis, 10.0)
    np.testing.assert_array_equal(axis, axis_before)
    np.testing.assert_allclose(rotated[:3, :3] @ rotated[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(-rotated[:3, :3].T @ rotated[:3, 3], [1.0, 2.0, 3.0])


def _soft_render(*, coupled: bool, missing: bool = False):
    rows = np.full((2, 2, 2), -1, dtype=np.int64)
    weights = np.zeros((2, 2, 2), dtype=np.float32)
    features = np.zeros((2, 2, 2, 2), dtype=np.float32)
    valid = np.zeros((2, 2, 2), dtype=bool)
    expected = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
    if not missing:
        # The correct child is deliberately secondary: dominant-only identity
        # would reject it, whereas the bilateral mixture must retain it.
        rows[..., 0] = 1 - expected
        rows[..., 1] = expected
        weights[..., 0] = 0.6
        weights[..., 1] = 0.4
        valid[:] = True
        features[..., 0, 1 if coupled else 0] = 1.0
        features[..., 1, 0 if coupled else 1] = 1.0
    return SimpleNamespace(
        child_rows=rows,
        child_weights=weights,
        child_features=features,
        child_feature_valid=valid,
        child_tail_weight=np.zeros((2, 2), dtype=np.float32),
        field_missing_weight=np.zeros((2, 2), dtype=np.float32),
        background_weight=np.ones((2, 2), dtype=np.float32) if missing else np.zeros((2, 2), dtype=np.float32),
        payload_missing_weight=np.zeros((2, 2), dtype=np.float32),
        null_weight=np.ones((2, 2), dtype=np.float32) if missing else np.zeros((2, 2), dtype=np.float32),
    )


def test_bidirectional_energy_retains_secondary_child_and_couples_feature():
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[0] = 1.0
    coupled = score_bidirectional_soft_surface_pose_energy(
        query, _retrieval(), _soft_render(coupled=True)
    )
    decoupled = score_bidirectional_soft_surface_pose_energy(
        query, _retrieval(), _soft_render(coupled=False)
    )
    missing = score_bidirectional_soft_surface_pose_energy(
        query, _retrieval(), _soft_render(coupled=True, missing=True)
    )
    assert coupled.mean_child_overlap > 0.0
    assert coupled.child_coupled_radio_score > decoupled.child_coupled_radio_score
    assert coupled.combined_score > missing.combined_score


def test_bidirectional_energy_rejects_token_order_and_typed_mass_drift():
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[0] = 1.0
    retrieval = _retrieval()
    permuted = SimpleNamespace(**retrieval.__dict__)
    permuted.token_xy = retrieval.token_xy[[1, 0, 2, 3]]
    with np.testing.assert_raises_regex(ValueError, "token order"):
        score_bidirectional_soft_surface_pose_energy(
            query, permuted, _soft_render(coupled=True)
        )
    rendered = _soft_render(coupled=True)
    rendered.background_weight[0, 0] = 0.1
    rendered.null_weight[0, 0] = 0.1
    with np.testing.assert_raises_regex(ValueError, "not conserved"):
        score_bidirectional_soft_surface_pose_energy(query, retrieval, rendered)


def test_hierarchical_spatial_score_separates_parent_support_and_child_precision():
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[0] = 1.0
    rendered = _soft_render(coupled=True)
    child_to_parent = np.asarray([0, 0], dtype=np.int64)
    score = score_hierarchical_spatial_soft_surface_pose_energy(
        query, _retrieval(), rendered, child_to_parent_ids=child_to_parent,
    )
    assert score.mean_parent_overlap > score.mean_child_overlap
    assert score.parent_support_score > score.child_precision_score
    assert score.spatial_kernel.startswith("fixed_center_half")


def test_hierarchical_spatial_missing_and_offgrid_support_never_improve():
    query = np.zeros((2, 2, 2), dtype=np.float32)
    query[0] = 1.0
    child_to_parent = np.asarray([0, 0], dtype=np.int64)
    present = score_hierarchical_spatial_soft_surface_pose_energy(
        query, _retrieval(), _soft_render(coupled=True),
        child_to_parent_ids=child_to_parent,
    )
    missing = score_hierarchical_spatial_soft_surface_pose_energy(
        query, _retrieval(), _soft_render(coupled=True, missing=True),
        child_to_parent_ids=child_to_parent,
    )
    assert missing.combined_score == -1.0
    assert present.combined_score > missing.combined_score
