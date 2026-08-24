from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_natural_sparse_pose_transport_scores import (
    _camera_without_pose,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.natural_pose_transport_bridge import (
    bind_scores_to_direct_labels,
    camera_intrinsics_content_sha256,
    ranked_natural_candidate_metrics,
)


def _write_contributor(path: Path, *, pose_payload: object) -> str:
    np.savez(
        path,
        camera_model_id=np.asarray(1, dtype=np.int32),
        camera_width=np.asarray(1024, dtype=np.int32),
        camera_height=np.asarray(576, dtype=np.int32),
        camera_params=np.asarray([700.0, 701.0, 512.0, 288.0]),
        # Object dtype is a trap: allow_pickle=False raises if any Phase-1
        # code accidentally opens this GT member.
        pose_w2c=np.asarray([pose_payload], dtype=object),
    )
    return file_sha256(path)


def test_pose_free_camera_binding_never_opens_or_hashes_gt_member(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_contributor(first, pose_payload={"secret_gt": "first"})
    _write_contributor(second, pose_payload={"secret_gt": "different"})
    camera_a, binding_a = _camera_without_pose(
        first, expected_image_id="seq14/frame00001.png",
    )
    camera_b, binding_b = _camera_without_pose(
        second, expected_image_id="seq14/frame00001.png",
    )
    assert camera_a.params == camera_b.params
    assert binding_a == binding_b == camera_intrinsics_content_sha256(
        "seq14/frame00001.png", 1, 1024, 576, [700.0, 701.0, 512.0, 288.0]
    )
    with np.load(first, allow_pickle=False) as data:
        with pytest.raises(ValueError, match="Object arrays"):
            np.asarray(data["pose_w2c"])


def test_ranked_metrics_report_raw_prefix_and_qpose_survival_without_anchor() -> None:
    poses = np.tile(np.eye(4), (2, 4, 1, 1)).astype(np.float64)
    # Camera centres are -t for identity rotation; make all four basins distinct.
    poses[:, :, 0, 3] = np.asarray([0.0, 1.0, 2.0, 3.0])[None]
    score = np.asarray([[0.1, 0.2, 0.9, 0.3], [0.9, 0.8, 0.7, 0.6]])
    translation = np.asarray([[3.0, 2.0, 0.2, 1.5], [3.0, 2.5, 2.2, 2.1]])
    rotation = np.asarray([[30.0, 20.0, 2.0, 8.0], [20.0, 20.0, 20.0, 20.0]])
    valid = np.ones((2, 4), dtype=bool)
    result = ranked_natural_candidate_metrics(
        score, poses, translation, rotation, valid, ranks=(1, 2, 4),
    )
    assert result["candidate_zero_diagnostic_gt_anchor_absent"] is True
    assert result["raw_pose_free_prefix_acquisition_upper_bound"][
        "strict_0_5m_5deg"
    ]["at_1"]["hits"] == 0
    assert result["raw_pose_free_prefix_acquisition_upper_bound"][
        "strict_0_5m_5deg"
    ]["at_4"]["hits"] == 1
    assert result["q_pose_ranked_full_pool_basin_survival"][
        "strict_0_5m_5deg"
    ]["at_1"]["hits"] == 1


def test_phase2_binding_strips_gt_anchor_and_checks_pose_free_camera_hash(
    tmp_path: Path,
) -> None:
    image_id = "seq14/frame00001.png"
    poses = np.tile(np.eye(4), (2, 1, 1)).astype(np.float64)
    poses[0, 0, 3] = 1.0
    poses[1, 0, 3] = 2.0
    contributor = tmp_path / "contributor.npz"
    np.savez(
        contributor,
        camera_model_id=np.asarray(1, dtype=np.int32),
        camera_width=np.asarray(1024, dtype=np.int32),
        camera_height=np.asarray(576, dtype=np.int32),
        camera_params=np.asarray([700.0, 701.0, 512.0, 288.0]),
        pose_w2c=np.eye(4, dtype=np.float64),
    )
    contributor_hash = file_sha256(contributor)
    pool: dict[str, object] = {
        "artifact_type": "goal_maplet_pose_free_visibility_candidate_pool_v1",
        "query_count": 1,
        "query_route": "seq14",
        "candidate_semantics": "progressive_hierarchical_location_orientation_v3",
        "maximum_modes": 2,
        "candidate_prefix_stable_across_budgets": True,
        "scores_are_pose_free_and_not_consumed_by_transport_builder": True,
        "route_disjoint_atlas_audit": {
            "route_allowlist_enforced": True,
            "query_route_excluded_from_atlas": True,
            "allowed_trajectories": ["seq1"],
            "coordinate_correct": True,
            "coordinate_contract": (
                "raw_simple_radial_equal_area_samples_inverse_warped_to_ideal_pinhole_"
                "contributor_grid_nearest_center_v1"
            ),
        },
        "uses_alike": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "uses_query_ground_truth": False,
        "uses_query_pose": False,
        "rows": [{
            "image_id": image_id,
            "retrieval_artifact": str(tmp_path / "retrieval.npz"),
            "retrieval_content_sha256": "a" * 64,
            "mode_details": {"actual_parent_actual_child": [
                {"rank": 1, "pose_w2c": poses[0].tolist()},
                {"rank": 2, "pose_w2c": poses[1].tolist()},
            ]},
        }],
    }
    pool["content_sha256"] = canonical_json_sha256(pool)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    direct_arrays = {
        "image_ids": np.asarray([image_id]),
        "radio_token_paths": np.asarray(["/tmp/radio.npz"]),
        "radio_file_sha256": np.asarray(["b" * 64]),
        "contributor_paths": np.asarray([str(contributor)]),
        "contributor_file_sha256": np.asarray([contributor_hash]),
        "candidate_poses_w2c": np.concatenate([
            np.eye(4)[None, None], poses[None],
        ], axis=1),
        "translation_m": np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
        "rotation_deg": np.asarray([[0.0, 10.0, 20.0]], dtype=np.float32),
        "candidate_valid": np.ones((1, 3), dtype=bool),
    }
    direct_metadata = {
        "artifact_type": "goal_maplet_direct_pose_candidate_dataset_v1",
        "content_sha256": arrays_sha256(direct_arrays),
        "candidate_pool_content_sha256": pool["content_sha256"],
        "candidate_pool_file_sha256": file_sha256(pool_path),
        "candidate_zero_is_diagnostic_gt_anchor": True,
        "candidate_pool_frozen_before_target_pose_opened": True,
        "pose_errors_computed_only_after_candidate_freeze": True,
        "nonanchor_candidates_preserve_pose_free_pool_exact_order": True,
        "gt_anchor_does_not_change_nonanchor_candidate_membership": True,
        "pose_free_pool_internal_duplicates_rejected_before_gt_join": True,
        "canonical_map_excludes_query_route": True,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
    }
    direct_path = tmp_path / "direct.npz"
    np.savez_compressed(
        direct_path, **direct_arrays,
        metadata_json=np.asarray(json.dumps(direct_metadata, sort_keys=True)),
    )
    camera_binding = camera_intrinsics_content_sha256(
        image_id, 1, 1024, 576, [700.0, 701.0, 512.0, 288.0]
    )
    score_arrays = {
        "image_ids": np.asarray([image_id]),
        "candidate_poses_w2c": poses[None],
        "candidate_valid": np.ones((1, 2), dtype=bool),
        "radio_file_sha256": np.asarray(["b" * 64]),
        "camera_intrinsics_content_sha256": np.asarray([camera_binding]),
    }
    score_metadata = {
        "candidate_pool_content_sha256": pool["content_sha256"],
        "candidate_pool_file_sha256": file_sha256(pool_path),
    }
    labels = bind_scores_to_direct_labels(
        score_arrays, score_metadata, direct_path, pool_path,
    )
    assert labels["translation_m"].tolist() == [[1.0, 2.0]]
    # Labels are recomputed from the Phase-2 GT pose, not trusted from the
    # direct artifact's potentially shifted/stale non-anchor rows.
    assert labels["rotation_deg"].tolist() == [[0.0, 0.0]]
