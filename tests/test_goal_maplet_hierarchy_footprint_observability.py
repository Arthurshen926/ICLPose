import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_hierarchy_footprint_observability import (
    _bounded_conjunction,
    _deployable_hierarchy_footprint_scores,
    _deployable_spatial_hierarchy_footprint_scores,
    _dice_from_histograms,
    _gt_visible_oracle_scores,
    _flatten_42_signed_ray_metrics,
    _metric_log_depth,
    _oracle_profile_agreement,
    _payload_content_sha256,
    _signed_direction_per_query_audit,
    _stage_child_group_ids,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
)


def _hierarchy() -> PoseTransportHierarchy:
    return PoseTransportHierarchy(
        child_parent_ids=np.asarray([0, 0, 1], dtype=np.int64),
        child_support_ids=np.asarray([0, 1, 1], dtype=np.int64),
        adjacency_offsets=np.asarray([0, 0, 0, 0], dtype=np.int64),
        adjacency_child_rows=np.empty((0,), dtype=np.int64),
        content_sha256="0" * 64,
    )


def test_stage_identity_scale_is_parent_support_child() -> None:
    hierarchy = _hierarchy()
    np.testing.assert_array_equal(
        _stage_child_group_ids(hierarchy, "coarse"), [0, 0, 1],
    )
    np.testing.assert_array_equal(
        _stage_child_group_ids(hierarchy, "medium"), [0, 1, 1],
    )
    np.testing.assert_array_equal(
        _stage_child_group_ids(hierarchy, "fine"), [0, 1, 2],
    )


def test_mass_dice_is_bounded_and_penalizes_unmatched_surplus() -> None:
    assert _dice_from_histograms(np.asarray([1.0]), np.asarray([1.0])) == 1.0
    assert _dice_from_histograms(np.asarray([1.0]), np.asarray([1.0, 1.0])) == 2.0 / 3.0
    assert _dice_from_histograms(np.asarray([1.0]), np.asarray([0.0, 1.0])) == 0.0


def test_deployable_footprint_uses_reliable_mass_without_candidate_normalization() -> None:
    token_count = 36 * 64
    source_rows = np.full((token_count, 1), -1, dtype=np.int64)
    source_mass = np.zeros((token_count, 1), dtype=np.float64)
    source_rows[0, 0] = 0; source_mass[0, 0] = 1.0
    target_rows = np.full((2, token_count, 2), -1, dtype=np.int64)
    target_mass = np.zeros((2, token_count, 2), dtype=np.float64)
    target_rows[:, 0, 0] = 0; target_mass[:, 0, 0] = 1.0
    target_rows[1, 0, 1] = 1; target_mass[1, 0, 1] = 1.0
    score, audit = _deployable_hierarchy_footprint_scores(
        source_rows, source_mass, np.ones(token_count), target_rows, target_mass,
        np.asarray([True, True]), _hierarchy(), stage="fine",
    )
    np.testing.assert_allclose(score, [1.0, 1.0 / 3.0], atol=1.0e-7)
    assert audit["stage_identity"] == "exact_surface_child"


def test_deployable_spatial_footprint_preserves_radio_token_identity() -> None:
    token_count = 36 * 64
    source_rows = np.full((token_count, 1), -1, dtype=np.int64)
    source_mass = np.zeros((token_count, 1), dtype=np.float64)
    source_rows[0, 0] = 0; source_mass[0, 0] = 1.0
    target_rows = np.full((2, token_count, 1), -1, dtype=np.int64)
    target_mass = np.zeros((2, token_count, 1), dtype=np.float64)
    target_rows[0, 0, 0] = 0; target_mass[0, 0, 0] = 1.0
    target_rows[1, 1, 0] = 0; target_mass[1, 1, 0] = 1.0
    score, audit = _deployable_spatial_hierarchy_footprint_scores(
        source_rows, source_mass, np.ones(token_count), target_rows, target_mass,
        np.asarray([True, True]), _hierarchy(), stage="fine",
    )
    np.testing.assert_allclose(score, [1.0, -1.0], atol=1.0e-7)
    assert audit["spatial_identity"] == "exact_RADIO_token_x_stage_physical_identity"
    assert audit["uses_GT_query_observation"] is False


def test_metric_log_depth_uses_absolute_camera_depth() -> None:
    centers = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    rows = np.asarray([[0]], dtype=np.int64)
    identity = np.eye(4, dtype=np.float64)
    shifted = identity.copy(); shifted[2, 3] = 2.0
    depth0, valid0 = _metric_log_depth(rows, identity, centers)
    depth1, valid1 = _metric_log_depth(rows, shifted, centers)
    assert valid0.item() and valid1.item()
    np.testing.assert_allclose(depth0, np.log(2.0))
    np.testing.assert_allclose(depth1, np.log(4.0))


def test_oracle_depth_reduces_agreement_without_changing_projected_footprint() -> None:
    reference = (
        np.asarray([7]), np.asarray([1.0]), np.asarray([1.0]), np.asarray([0.0]),
    )
    shifted = (
        np.asarray([7]), np.asarray([1.0]), np.asarray([1.0]), np.asarray([0.5]),
    )
    footprint, depth = _oracle_profile_agreement(reference, shifted, depth_scale=0.25)
    assert footprint == 1.0
    np.testing.assert_allclose(depth, np.exp(-2.0))


def test_gt_visible_candidate_zero_is_exact_and_metric_depth_is_discriminative() -> None:
    token_count = 36 * 64
    rows = np.full((2, token_count, 1), -1, dtype=np.int64)
    mass = np.zeros((2, token_count, 1), dtype=np.float64)
    rows[:, 0, 0] = 0; mass[:, 0, 0] = 1.0
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    poses[1, 2, 3] = 2.0
    footprint, depth, audit = _gt_visible_oracle_scores(
        rows, mass, poses, np.asarray([True, True]), np.ones(token_count),
        _hierarchy(), np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [1.0, 0.0, 3.0]]),
        stage="fine",
    )
    np.testing.assert_allclose(footprint, [1.0, 1.0])
    assert depth[0] == 1.0 and depth[1] < 1.0
    assert audit["reference_candidate_index"] == 0


def test_bounded_conjunction_has_no_weight_and_stays_bounded() -> None:
    left = np.asarray([1.0, 0.0, -1.0], dtype=np.float32)
    right = np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        _bounded_conjunction(left, right, "product"), [0.0, 0.0, -1.0],
    )
    np.testing.assert_allclose(
        _bounded_conjunction(left, right, "minimum"), [0.0, 0.0, -1.0],
    )


def test_signed_direction_audit_exposes_each_query_and_strict_margin() -> None:
    scores = np.asarray([
        [3.0, 2.0, 1.0, 2.5, 2.0],
        [3.0, 3.5, 1.0, 2.5, 2.6],
    ])
    result = _signed_direction_per_query_audit(
        scores, np.asarray(["q0", "q1"]),
        np.asarray([[0, 1, 2], [0, 3, 4]]),
        np.asarray([-1, 5, 5, 5, 5]),
        np.asarray([0, -1, -1, 1, 1]), direction_id=5,
    )
    assert result["negative"][0]["complete_path"] is True
    assert result["negative"][1]["complete_path"] is False
    assert result["positive"][0]["complete_path"] is True
    assert result["positive"][1]["complete_path"] is False


def test_flattened_direction_report_has_exactly_42_explicit_signed_rays() -> None:
    directions = []
    for direction in range(21):
        summary = {
            "radial_pair_correct": 12,
            "radial_pair_count": 12,
            "radial_pair_accuracy": 1.0,
            "complete_path_count": 6,
            "signed_path_count": 6,
            "complete_path_rate": 1.0,
        }
        directions.append({
            "direction_id": direction,
            "kind": "coordinate_axis" if direction < 6 else "pair_coupling",
            "label": f"d{direction}",
            "negative_sign": summary,
            "positive_sign": summary,
        })
    result = _flatten_42_signed_ray_metrics({
        "full_6dof_directional_capture": {"per_direction": directions},
    })
    assert len(result) == 42
    assert all(row["strictly_monotonic_for_every_query"] for row in result)


def test_payload_hash_is_canonical_and_non_self_referential() -> None:
    assert _payload_content_sha256({"a": 1, "b": 2}) == _payload_content_sha256(
        {"b": 2, "a": 1},
    )
    try:
        _payload_content_sha256({"report_payload_content_sha256": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("self-referential report hash must be rejected")
