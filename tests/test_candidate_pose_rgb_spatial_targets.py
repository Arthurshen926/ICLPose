from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
    project_simple_radial_offsets,
    save_candidate_pose_rgb_spatial_training_targets,
)


def _targets(*, metadata: dict[str, object] | None = None) -> CandidatePoseRGBSpatialTrainingTargets:
    target_metadata = {
        "format": CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "rgb_spatial_layout_sha256": "layout-hash",
        "train_pairs_sha256": "pairs-hash",
        "support_geometry_index_sha256": "geometry-hash",
        "projected_landmark_bank_sha256": "bank-hash",
        "projection_space_id": "projection-space",
        "descriptor_space_id": "descriptor-space",
        "spatial_search_radius_px": 8.0,
    }
    if metadata is not None:
        target_metadata.update(metadata)
    return CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray([11, 13], dtype=np.int64),
        query_ids=np.asarray(["train/a.png", "train/a.png"]),
        spatial_target_offsets_xy=np.asarray(
            [[[1.0, -2.0], [20.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        spatial_target_observed=np.asarray([[True, True], [False, False]]),
        spatial_target_dustbin=np.asarray([[False, True], [True, True]]),
        pair_query_ids=np.asarray(["train/a.png"]),
        pair_ids=np.asarray([7], dtype=np.int64),
        pair_point_offsets=np.asarray([0, 2], dtype=np.int64),
        pair_source_point_ids=np.asarray([11, 13], dtype=np.int64),
        correct_projection_offsets_xy=np.asarray(
            [[[1.0, -2.0], [18.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        correct_projection_valid=np.asarray([[True, True], [True, True]]),
        coherent_wrong_projection_offsets_xy=np.asarray(
            [[[4.0, -2.0], [24.0, 0.0]], [[3.0, 1.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        coherent_wrong_projection_valid=np.asarray([[True, True], [True, False]]),
        metadata=target_metadata,
    )


def test_rgb_spatial_targets_round_trip_and_keep_train_only_lineage(tmp_path) -> None:
    expected = _targets()
    path = tmp_path / "targets.npz"

    save_candidate_pose_rgb_spatial_training_targets(expected, path)
    actual = load_candidate_pose_rgb_spatial_training_targets(path)

    np.testing.assert_array_equal(actual.source_point_ids, expected.source_point_ids)
    np.testing.assert_array_equal(actual.pair_source_point_ids, expected.pair_source_point_ids)
    np.testing.assert_allclose(
        actual.coherent_wrong_projection_offsets_xy,
        expected.coherent_wrong_projection_offsets_xy,
    )
    np.testing.assert_array_equal(
        actual.spatial_target_supervised, np.ones((2, 2), dtype=bool)
    )
    assert actual.metadata["training_only_target_artifact"] is True


def test_identity_target_keeps_unsupervised_anchors_out_of_dustbin_loss(tmp_path) -> None:
    metadata = {
        **_targets().metadata,
        "format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
        "spatial_supervision_mode": "registered_exact_identity",
        "spatial_target_semantics": (
            "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
        ),
        "spatial_class_balance": "per_batch_observed_dustbin_mean_v1",
        "registered_identity_radius_px": 2.0,
        "colmap_images_sha256": "images-hash",
    }
    targets = CandidatePoseRGBSpatialTrainingTargets(
        **{
            **_targets().__dict__,
            "spatial_target_observed": np.asarray([[True, False], [False, False]]),
            "spatial_target_dustbin": np.asarray([[False, True], [False, False]]),
            "spatial_target_supervised": np.asarray([[True, True], [False, False]]),
            "metadata": metadata,
        }
    )
    path = tmp_path / "identity_targets.npz"
    save_candidate_pose_rgb_spatial_training_targets(targets, path)
    loaded = load_candidate_pose_rgb_spatial_training_targets(path)
    np.testing.assert_array_equal(loaded.spatial_target_supervised, targets.spatial_target_supervised)
    np.testing.assert_array_equal(loaded.spatial_target_dustbin, targets.spatial_target_dustbin)


def test_rgb_spatial_targets_reject_validation_or_test_target_metadata() -> None:
    with pytest.raises(ValueError, match="train-only"):
        _targets(metadata={"contains_validation_or_test_targets": True})


def test_rgb_spatial_targets_reject_pair_points_from_another_query() -> None:
    with pytest.raises(ValueError, match="query"):
        CandidatePoseRGBSpatialTrainingTargets(
            **{
                **_targets().__dict__,
                "pair_query_ids": np.asarray(["train/b.png"]),
            }
        )


def test_rgb_spatial_targets_reject_tampered_metadata_on_load(tmp_path) -> None:
    path = tmp_path / "targets.npz"
    save_candidate_pose_rgb_spatial_training_targets(_targets(), path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["runtime_layout_is_target_free"] = False
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ValueError, match="train-only"):
        load_candidate_pose_rgb_spatial_training_targets(path)


def test_simple_radial_projection_offsets_marks_depth_and_image_failures() -> None:
    offsets, valid = project_simple_radial_offsets(
        xyz=np.asarray([[0.0, 0.0, 2.0], [10.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        poses_w2c=np.broadcast_to(np.eye(4), (1, 4, 4)).copy(),
        query_xy=np.asarray([[50.0, 40.0]], dtype=np.float32),
        focal_length=100.0,
        principal_x=50.0,
        principal_y=40.0,
        radial_k=0.0,
        image_width=100,
        image_height=80,
    )

    np.testing.assert_allclose(offsets[0, 0], [0.0, 0.0])
    assert valid.tolist() == [[True, False, False]]


def test_simple_radial_projection_supports_a_different_candidate_bank_per_point() -> None:
    offsets, valid = project_simple_radial_offsets(
        xyz=np.asarray(
            [
                [[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]],
                [[0.0, 0.0, 4.0], [-0.2, 0.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        poses_w2c=np.broadcast_to(np.eye(4), (2, 4, 4)).copy(),
        query_xy=np.asarray([[50.0, 40.0], [50.0, 40.0]], dtype=np.float32),
        focal_length=100.0,
        principal_x=50.0,
        principal_y=40.0,
        radial_k=0.0,
        image_width=100,
        image_height=80,
    )

    np.testing.assert_allclose(offsets[:, 0], [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(offsets[0, 1], [5.0, 0.0])
    np.testing.assert_allclose(offsets[1, 1], [-10.0, 0.0])
    assert valid.tolist() == [[True, True], [True, True]]
