from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_fulltoken_ranker_pattern_search import (
    _evaluated_pose_trace_diagnostics,
    _factorized_trace_diagnostics,
    _global_joint_seed_poses,
    _inside_seed_domain_union,
    _protected_hypothesis_union,
    _validate_natural_transfer_model_pair,
)
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    build_local_pose_supervision_candidates,
    deterministic_global_joint_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import left_retract_pose_w2c
from feature_extract.vfm.localization_goal_maplet.multibasin_pattern_search import (
    PoseBasinState,
)


def _pose(center=(0.0, 0.0, 0.0), angle_deg=0.0):
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = -rotation @ np.asarray(center)
    return value


def test_seed_domain_union_uses_closed_world_box_and_rotation_ball():
    seeds = np.stack([_pose(), _pose(center=(20.0, 0.0, 0.0))])
    candidates = np.stack([
        _pose(center=(8.0, -8.0, 8.0), angle_deg=45.0),
        _pose(center=(28.0, 0.0, 0.0), angle_deg=0.0),
        _pose(center=(8.01, -8.0, 8.0), angle_deg=45.0),
        _pose(center=(0.0, 0.0, 0.0), angle_deg=45.01),
    ])
    np.testing.assert_array_equal(
        _inside_seed_domain_union(
            candidates, seeds,
            translation_half_extent_m=8.0, rotation_radius_deg=45.0,
        ),
        [True, True, False, False],
    )


def test_global_joint_search_replays_exact_supervision_scale_and_order():
    seed = np.eye(4, dtype=np.float64)
    search, source = _global_joint_seed_poses(seed[None])
    supervised, _, _ = build_local_pose_supervision_candidates(seed)
    # Supervision rows 187:315 are the same 128 global probes. Search adds
    # the domain seed first and must then replay those probes exactly.
    np.testing.assert_array_equal(search[0], seed)
    np.testing.assert_allclose(search[1:], supervised[187:], atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(source, np.zeros(129, dtype=np.int64))


def test_global_joint_search_supports_explicit_medium_domain_scale():
    seed = np.eye(4, dtype=np.float64)
    search, source = _global_joint_seed_poses(
        seed[None], translation_step_m=2.0, rotation_step_degrees=20.0,
    )
    coordinates = deterministic_global_joint_coordinates()
    expected = np.asarray([
        left_retract_pose_w2c(
            seed, coordinate, translation_step_m=2.0,
            rotation_step_degrees=20.0,
        )
        for coordinate in coordinates
    ])
    np.testing.assert_array_equal(search[0], seed)
    np.testing.assert_allclose(search[1:], expected, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(source, np.zeros(129, dtype=np.int64))


def test_protected_union_cannot_erase_retrieval_or_initializer_pose():
    seed = _pose(center=(3.0, 0.0, 0.0))[None]
    initializer = _pose(center=(2.0, 0.0, 0.0))[None]
    refined = _pose(center=(8.0, 0.0, 0.0))
    states = (PoseBasinState(
        pose_w2c=refined,
        score=3.0,
        translation_radii_m=np.ones(3),
        rotation_radii_deg=np.ones(3),
        source_basin_index=0,
        accepted_updates=2,
    ),)
    poses, scores, updates, kinds = _protected_hypothesis_union(
        seed, np.asarray([1.0]), initializer, np.asarray([2.0]), states,
    )
    np.testing.assert_array_equal(poses, np.concatenate((seed, initializer, refined[None])))
    np.testing.assert_array_equal(scores, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(updates, [0, 0, 2])
    np.testing.assert_array_equal(
        kinds, ["retrieval_seed", "search_initializer", "refined_hypothesis"],
    )


def test_factorized_trace_separates_missing_probe_from_score_pruning():
    target = _pose(center=(1.0, 0.0, 0.0), angle_deg=5.0)
    poses = np.stack([
        target,
        _pose(center=(8.0, 0.0, 0.0), angle_deg=30.0),
    ])
    rows = _factorized_trace_diagnostics([
        ("location", poses, np.asarray([0.1, 0.9])),
        ("joint", poses[1:], np.asarray([0.9])),
    ], target)
    assert rows[0]["loose_pose_count"] == 1
    assert rows[0]["oracle_score_rank"] == 2
    assert np.isclose(rows[0]["topscore_translation_m"], 7.0)
    assert rows[1]["loose_pose_count"] == 0
    assert rows[1]["oracle_score_rank"] == 1


def test_natural_transfer_separates_frozen_coarse_and_trajectory_local_roles():
    coarse = {
        "model_content_sha256": "a" * 64,
        "dataset_content_sha256": "b" * 64,
        "local_supervision": True,
        "multiscale_supervision": True,
    }
    local = {
        "initial_model_content_sha256": "a" * 64,
        "local_pose_supervision_semantics": (
            "frozen_natural_seed_to_gt_center_linear_so3_shortest_arc_training_only_v1"
        ),
        "local_supervision": True,
        "multiscale_supervision": True,
    }
    _validate_natural_transfer_model_pair(
        coarse, local, evaluation_dataset_content_sha256="c" * 64,
    )

    bad = dict(local, initial_model_content_sha256="d" * 64)
    with np.testing.assert_raises_regex(ValueError, "direct trajectory/local refinement"):
        _validate_natural_transfer_model_pair(
            coarse, bad, evaluation_dataset_content_sha256="c" * 64,
        )


def test_evaluated_pose_trace_separates_generation_from_score_selection():
    target = _pose(center=(1.0, 0.0, 0.0), angle_deg=5.0)
    poses = np.stack([
        _pose(center=(8.0, 0.0, 0.0), angle_deg=30.0),
        target,
        target,
    ])
    result = _evaluated_pose_trace_diagnostics(
        poses, np.asarray([0.9, 0.1, 0.1]), target,
    )
    assert result["unique_rendered_pose_count"] == 2
    assert result["duplicate_rendered_pose_count"] == 1
    assert result["duplicate_score_drift_count"] == 0
    assert result["any_strict_0_5m_5deg"] is True
    assert result["any_loose_1m_10deg"] is True
    assert result["oracle_pose_score_rank"] == 2
    assert result["best_translation_m"] < 1.0e-8
    assert result["best_rotation_deg"] < 2.0e-6
