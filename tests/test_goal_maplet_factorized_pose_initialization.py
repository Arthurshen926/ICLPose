from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.vfm.localization_goal_maplet.factorized_pose_initialization import (
    FACTORIZED_HIERARCHICAL_INITIALIZATION_SEMANTICS,
    factorized_hierarchical_pose_initialization,
)


def _pose(center=(0.0, 0.0, 0.0), angle_deg=0.0):
    angle = np.deg2rad(float(angle_deg))
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ np.asarray(center)
    return result


def test_factorized_hierarchy_reaches_joint_pose_without_dense_product():
    target = _pose(center=(6.3, -3.4, 1.2), angle_deg=32.0)

    def position_score(poses):
        translation, _ = _pose_errors(poses, target)
        return -translation

    def orientation_score(poses):
        translation, rotation = _pose_errors(poses, target)
        return -np.maximum(translation / 0.5, rotation / 5.0)

    result = factorized_hierarchical_pose_initialization(
        _pose()[None], position_score, orientation_score,
        position_beam_width_per_seed=2, survivors_per_seed=2,
    )
    translation, rotation = _pose_errors(result.poses_w2c, target)
    assert np.any((translation <= 0.5) & (rotation <= 5.0))
    assert result.evaluated_pose_count < 13_125
    assert result.semantics == FACTORIZED_HIERARCHICAL_INITIALIZATION_SEMANTICS
    assert result.position_stage_pose_counts[0] == 125
    assert result.orientation_stage_pose_count == 210


def test_factorized_hierarchy_keeps_equal_budget_per_seed_and_is_deterministic():
    seeds = np.stack([_pose(), _pose(center=(20.0, 0.0, 0.0))])

    def score(poses):
        centers = (-np.swapaxes(poses[:, :3, :3], 1, 2) @ poses[:, :3, 3, None])[:, :, 0]
        return -np.linalg.norm(centers, axis=1)

    first = factorized_hierarchical_pose_initialization(seeds, score, score)
    second = factorized_hierarchical_pose_initialization(seeds, score, score)
    np.testing.assert_array_equal(first.poses_w2c, second.poses_w2c)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.source_seed_indices, [0, 0, 1, 1])


def test_factorized_hierarchy_rejects_invalid_budget_contract():
    try:
        factorized_hierarchical_pose_initialization(
            _pose()[None], lambda poses: np.zeros(poses.shape[0]),
            lambda poses: np.zeros(poses.shape[0]),
            position_beam_width_per_seed=0,
        )
    except ValueError as error:
        assert "initialization contract" in str(error)
    else:
        raise AssertionError("invalid factorized budget was accepted")


def test_factorized_hierarchy_adapts_to_medium_2m_20deg_domain():
    target = _pose(center=(1.6, -1.3, 0.9), angle_deg=18.0)

    def position_score(poses):
        translation, _ = _pose_errors(poses, target)
        return -translation

    def joint_score(poses):
        translation, rotation = _pose_errors(poses, target)
        return -np.maximum(translation / 0.5, rotation / 5.0)

    result = factorized_hierarchical_pose_initialization(
        _pose()[None], position_score, joint_score,
        translation_half_extent_m=2.0,
        coarse_translation_step_m=1.0,
        refinement_translation_steps_m=(0.5, 0.25, 0.125),
        rotation_radius_deg=20.0,
        position_beam_width_per_seed=2,
        survivors_per_seed=2,
    )
    translation, rotation = _pose_errors(result.poses_w2c, target)
    assert np.any((translation <= 0.5) & (rotation <= 5.0))
    assert result.position_stage_pose_counts[0] == 125
    # Identity plus 26 signed cubic axes at 10 and 20 degrees, at two centres.
    assert result.orientation_stage_pose_count == 106
