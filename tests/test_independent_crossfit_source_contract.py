import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.run_independent_crossfit_pose_alignment import (
    SELECTED_POSE_ARTIFACT_FORMAT,
    _canonical_hash,
    _crossfit_rank_shortlist,
    _crossfit_role_partitions,
    _filtered_sharded_group_keys,
    _hypothesis_group,
    _resolve_immutable_source_pose_artifact,
    _validate_local_descriptor_verification_bank,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.independent_landmark_pose_likelihood import (
    IndependentVerificationPoints,
)


def _write_selected_pose_artifact(path: Path) -> None:
    source_manifest = {
        "inputs": {
            "colmap_cameras_bin_sha256": "camera-hash",
            "colmap_images_bin_sha256": "image-hash",
        }
    }
    metadata = {
        "format": SELECTED_POSE_ARTIFACT_FORMAT,
        "row_count": 2,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_reestimated_during_replay": False,
        "source_manifest_sha256": _canonical_hash(source_manifest),
        "source_manifest": source_manifest,
    }
    poses = np.stack(
        [np.eye(4, dtype=np.float64), np.full((4, 4), np.nan, dtype=np.float64)]
    )
    np.savez_compressed(
        path,
        query_ids=np.asarray(["query-success", "query-failure"]),
        split_names=np.asarray(["validation", "test"]),
        evaluation_labels=np.asarray(["frozen-policy", "frozen-policy"]),
        success=np.asarray([True, False]),
        poses_w2c=poses,
        match_counts=np.asarray([32, 0], dtype=np.int64),
        inlier_counts=np.asarray([20, 0], dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def test_immutable_source_is_resolved_from_hypothesis_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "selected_pose.npz"
    _write_selected_pose_artifact(artifact)
    metadata = {
        "inputs": {
            "immutable_baseline_pose_artifact": str(artifact),
            "immutable_baseline_pose_artifact_sha256": file_sha256_short(artifact),
            "immutable_baseline_pose_evaluation_label": "frozen-policy",
        }
    }

    source = _resolve_immutable_source_pose_artifact(
        metadata,
        requested_path="",
        requested_evaluation_label="",
        expected_colmap_cameras_sha256="camera-hash",
        expected_colmap_images_sha256="image-hash",
    )

    records = source["records"]
    assert source["evaluation_label"] == "frozen-policy"
    assert records[("validation", "query-success")]["success"] is True
    np.testing.assert_array_equal(
        records[("validation", "query-success")]["pose_w2c"], np.eye(4)
    )
    assert records[("test", "query-failure")]["success"] is False
    assert records[("test", "query-failure")]["pose_w2c"] is None


def test_immutable_source_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "selected_pose.npz"
    _write_selected_pose_artifact(artifact)
    metadata = {
        "inputs": {
            "immutable_baseline_pose_artifact": str(artifact),
            "immutable_baseline_pose_artifact_sha256": "wrong-hash",
            "immutable_baseline_pose_evaluation_label": "frozen-policy",
        }
    }

    with pytest.raises(ValueError, match="hash differs"):
        _resolve_immutable_source_pose_artifact(
            metadata,
            requested_path="",
            requested_evaluation_label="",
            expected_colmap_cameras_sha256="camera-hash",
            expected_colmap_images_sha256="image-hash",
        )


def test_local_descriptor_bank_requires_exact_map_and_alike_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_bank.npz"
    track_ids = np.asarray([7, 8], dtype=np.int64)
    xyz = np.asarray([[0.0, 0.0, 5.0], [1.0, 0.0, 6.0]], dtype=np.float64)
    np.savez_compressed(source, track_ids=track_ids, xyz=xyz)
    bank = SimpleNamespace(
        track_ids=track_ids,
        xyz=xyz,
        features=np.zeros((2, 64), dtype=np.float32),
    )
    metadata = {
        "descriptor_space_manifest": {
            "alike_checkpoint_sha256": "alike-hash",
            "descriptor_dimension": 64,
            "projection_source": "real_image_alike_observation_full_map",
            "coordinate_source": "sfm_observation_xy",
        }
    }
    detector = {
        "alike_checkpoint_sha256": "alike-hash",
        "local_descriptor_dimension": 64,
    }

    contract = _validate_local_descriptor_verification_bank(
        source, bank, metadata, detector
    )
    assert contract["physical_track_geometry_exact"] is True

    bad_metadata = json.loads(json.dumps(metadata))
    bad_metadata["descriptor_space_manifest"]["alike_checkpoint_sha256"] = "stale"
    with pytest.raises(ValueError, match="descriptor contract differs"):
        _validate_local_descriptor_verification_bank(
            source, bank, bad_metadata, detector
        )


def test_hypothesis_scope_forces_artifact_chosen_pose() -> None:
    group = _hypothesis_group(
        np.asarray([0, 1, 2], dtype=np.int64),
        shortlisted=np.asarray([False, True, False]),
        artifact_chosen=np.asarray([False, False, True]),
        preliminary_scores=np.asarray([0.1, 0.2, 0.3]),
        scope="shortlisted",
        limit=1,
    )

    np.testing.assert_array_equal(group, np.asarray([1, 2], dtype=np.int64))


def test_split_filter_is_applied_before_query_sharding() -> None:
    keys = [
        ("test", "policy", "test-0"),
        ("validation", "policy", "val-0"),
        ("validation", "policy", "val-1"),
        ("validation", "policy", "val-2"),
    ]

    selected = _filtered_sharded_group_keys(
        keys,
        split_filter="validation",
        shard_count=2,
        shard_index=1,
    )

    assert selected == [("validation", "policy", "val-1")]


def test_crossfit_rank_shortlist_forces_source_and_is_deterministic() -> None:
    selected = _crossfit_rank_shortlist(
        np.asarray([0.5, 0.9, 0.8, 0.8]),
        np.asarray([20, 10, 12, 12]),
        np.asarray([3, 2, 5, 4]),
        size=3,
        forced_indices=[0],
    )

    np.testing.assert_array_equal(selected, np.asarray([0, 1, 3]))
    source_only = _crossfit_rank_shortlist(
        np.asarray([0.5, 0.9]),
        np.asarray([20, 10]),
        np.asarray([3, 2]),
        size=1,
        forced_indices=[0],
    )
    np.testing.assert_array_equal(source_only, np.asarray([0]))


def test_two_role_crossfit_uses_all_points_without_rank_audit_overlap() -> None:
    points = IndependentVerificationPoints(
        xy=np.arange(12, dtype=np.float64).reshape(6, 2),
        descriptors=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (6, 1)),
        descriptor_reference_scores=np.zeros((6,), dtype=np.float32),
        source_row_indices=np.arange(100, 106, dtype=np.int64),
    )
    point_folds = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    base = np.ones((4,), dtype=bool)
    landmark_folds = np.asarray([0, 1, 0, 1], dtype=np.int64)

    role_points, role_landmarks, active = _crossfit_role_partitions(
        points,
        point_folds,
        base,
        landmark_folds,
        role_count=2,
    )

    assert active == (1, 2)
    assert role_points[0] is role_points[1]
    assert len(role_points[1]) + len(role_points[2]) == len(points)
    assert not np.intersect1d(
        role_points[1].source_row_indices, role_points[2].source_row_indices
    ).size
    assert not np.any(role_landmarks[1] & role_landmarks[2])
