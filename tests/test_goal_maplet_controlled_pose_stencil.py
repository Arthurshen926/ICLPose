from __future__ import annotations

import json

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.tools.vfm.build_goal_maplet_sparse_pose_transport_dataset import (
    _CANDIDATE_OBSERVATION_ARRAYS,
    _dataset_budget_audit,
)
from feature_extract.vfm.localization_goal_maplet.controlled_pose_stencil import (
    QUADRATIC_COMPLETE_6DOF_STENCIL_SEMANTICS,
    TWIST_ORDER,
    build_medium_quadratic_complete_6dof_stencil,
    controlled_pose_stencil_audit,
    stencil_candidate_poses,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp


def test_medium_stencil_covers_every_axis_pair_and_spans_symmetric_6dof():
    stencil = build_medium_quadratic_complete_6dof_stencil()
    audit = controlled_pose_stencil_audit(stencil)
    assert stencil.semantics == QUADRATIC_COMPLETE_6DOF_STENCIL_SEMANTICS
    assert stencil.candidate_count == 85
    assert stencil.direction_count == 21
    assert stencil.radial_path_count == 42
    assert stencil.twists_left_camera.shape == (85, 6)
    assert stencil.radial_paths.shape == (42, 3)
    assert stencil.normalized_directions.shape == (21, 6)
    assert tuple(TWIST_ORDER) == ("r_x", "r_y", "r_z", "t_x", "t_y", "t_z")
    assert audit["covers_all_six_tangent_axes"] is True
    assert audit["covers_all_fifteen_axis_pairs"] is True
    assert audit["symmetric_quadratic_design_rank"] == 21
    assert audit["local_quadratic_6dof_identifiable"] is True
    assert audit["adjacent_monotonic_pair_count"] == 84
    assert audit["basin_capture_claim_eligible"] is False
    # The audit is an artifact payload, not a collection of NumPy scalars.
    json.dumps(audit, sort_keys=True)


def test_medium_stencil_paths_are_signed_two_radius_rays():
    stencil = build_medium_quadratic_complete_6dof_stencil()
    np.testing.assert_array_equal(stencil.twists_left_camera[0], np.zeros(6))
    assert stencil.candidate_direction_ids[0] == -1
    assert stencil.candidate_signs[0] == 0
    assert stencil.candidate_radius_fractions[0] == 0.0
    seen = set()
    for anchor, inner, outer in stencil.radial_paths.tolist():
        assert anchor == 0
        direction_id = int(stencil.candidate_direction_ids[inner])
        sign = int(stencil.candidate_signs[inner])
        assert stencil.candidate_direction_ids[outer] == direction_id
        assert stencil.candidate_signs[outer] == sign
        assert stencil.candidate_radius_fractions[inner] == 0.5
        assert stencil.candidate_radius_fractions[outer] == 1.0
        np.testing.assert_allclose(
            stencil.twists_left_camera[outer],
            2.0 * stencil.twists_left_camera[inner],
            atol=1.0e-12,
        )
        seen.add((direction_id, sign))
    assert seen == {
        (direction, sign)
        for direction in range(stencil.direction_count)
        for sign in (-1, 1)
    }


def test_medium_stencil_pose_errors_increase_on_every_radial_path():
    stencil = build_medium_quadratic_complete_6dof_stencil()
    target = se3_exp(
        np.asarray([0.07, -0.11, 0.03, 4.0, -1.5, 7.0], dtype=np.float64)
    )
    candidates = stencil_candidate_poses(target, stencil)
    translation_m, rotation_deg = _pose_errors(candidates, target)
    joint = np.maximum(translation_m / 1.0, rotation_deg / 15.0)
    assert float(joint[0]) <= 1.0e-6
    for path in stencil.radial_paths:
        values = joint[np.asarray(path, dtype=np.int64)]
        assert values[0] < values[1] < values[2]
        np.testing.assert_allclose(values[1:], [0.5, 1.0], atol=2.0e-4)


def test_medium_stencil_application_is_left_multiplicative_and_deterministic():
    first = build_medium_quadratic_complete_6dof_stencil()
    second = build_medium_quadratic_complete_6dof_stencil()
    assert first.content_sha256 == second.content_sha256
    target = np.eye(4, dtype=np.float64)
    target[:3, 3] = [1.0, 2.0, 3.0]
    candidates = stencil_candidate_poses(target, first)
    np.testing.assert_allclose(candidates[0], target, atol=1.0e-12)
    for index in (1, 4, 24, 57, 84):
        np.testing.assert_allclose(
            candidates[index],
            se3_exp(first.twists_left_camera[index]) @ target,
            atol=1.0e-12,
        )


def test_medium_stencil_rejects_invalid_target_pose():
    stencil = build_medium_quadratic_complete_6dof_stencil()
    bad = np.eye(4, dtype=np.float64)
    bad[3, 3] = 0.0
    with np.testing.assert_raises_regex(ValueError, "homogeneous"):
        stencil_candidate_poses(bad, stencil)


def test_dataset_budget_reports_exact_candidate_storage_and_batches():
    arrays = {
        name: np.zeros((3, 85, 2), dtype=np.float16)
        for name in _CANDIDATE_OBSERVATION_ARRAYS
    }
    arrays["query_only"] = np.zeros((3, 7), dtype=np.float32)
    audit = _dataset_budget_audit(
        arrays,
        query_count=3,
        candidate_count=85,
        render_batch_size=4,
        total_render_seconds=10.2,
    )
    expected_candidate_bytes = 9 * 3 * 85 * 2 * np.dtype(np.float16).itemsize
    assert audit["rendered_pose_count"] == 255
    assert audit["render_batch_count"] == 66
    assert audit["uncompressed_candidate_observation_bytes"] == expected_candidate_bytes
    assert audit["uncompressed_candidate_observation_bytes_per_query"] == (
        expected_candidate_bytes // 3
    )
    assert audit["uncompressed_all_array_bytes"] == (
        expected_candidate_bytes + 3 * 7 * np.dtype(np.float32).itemsize
    )
    json.dumps(audit, sort_keys=True)
