from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_hard_pose_identity_pairs import (
    build_candidate_pose_rgb_spatial_hard_pose_identity_pairs,
)
from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_hard_pose_identity_targets import (
    build_candidate_pose_rgb_spatial_hard_pose_identity_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT,
    CandidatePoseRGBSpatialHardPosePairs,
    save_candidate_pose_rgb_spatial_hard_pose_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT,
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
    save_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_identity import (
    load_candidate_pose_rgb_spatial_hard_pose_identity_targets,
)


def _observation_metadata() -> dict[str, object]:
    return {
        "format": CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_scorer_must_not_load_this_artifact": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "candidate_set": "fixed_positive_same_track_plus_radio_pca_global_landmark_hard_negatives",
        "query_split": "train_only_inner_partition_v1",
        "hard_negative_semantics": "radio_intermediate_pca_global_landmark_ann_distinct_track_v1",
        "negative_count": 2,
        "inner_validation_fold_count": 5,
        "inner_validation_fold_index": 1,
        "train_query_layout_sha256": "layout",
        "colmap_images_bin_sha256": "images",
        "support_observation_index_sha256": "support",
        "hard_negative_context_cache_sha256": "context",
        "hard_negative_landmark_bank_sha256": "bank",
        "train_query_image_list_sha256": "queries",
    }


def _source_pairs() -> CandidatePoseRGBSpatialObservationPairs:
    return CandidatePoseRGBSpatialObservationPairs(
        anchor_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_image_ids=np.asarray(["q/train.png", "q/train.png", "q/val.png", "q/val.png"]),
        query_xy=np.asarray([[20.0, 20.0], [30.0, 20.0], [20.0, 30.0], [30.0, 30.0]]),
        positive_support_image_ids=np.asarray(["m/a.png", "m/b.png", "m/c.png", "m/d.png"]),
        positive_support_xy=np.full((4, 2), 24.0, dtype=np.float32),
        positive_track_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        negative_support_image_ids=np.asarray(
            [["m/n1.png", "m/n2.png"]] * 4
        ),
        negative_support_xy=np.full((4, 2, 2), 28.0, dtype=np.float32),
        negative_track_ids=np.asarray([[20, 21], [22, 23], [24, 25], [26, 27]], dtype=np.int64),
        negative_sources=np.asarray([["ann", "ann"]] * 4),
        split_names=np.asarray(["inner_train", "inner_train", "inner_validation", "inner_validation"]),
        metadata=_observation_metadata(),
    )


def _hard_pairs(source_sha256: str) -> CandidatePoseRGBSpatialHardPosePairs:
    return CandidatePoseRGBSpatialHardPosePairs(
        row_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        pose_pair_ids=np.asarray([5, 5, 6, 6], dtype=np.int64),
        query_image_ids=np.asarray(["q/train.png", "q/train.png", "q/val.png", "q/val.png"]),
        query_xy=np.asarray([[21.0, 20.0], [29.0, 20.0], [21.0, 30.0], [29.0, 30.0]]),
        candidate_track_ids=np.asarray(
            [[10, 20, 21], [11, 22, 23], [12, 24, 25], [13, 26, 27]], dtype=np.int64
        ),
        support_image_ids=np.asarray(
            [
                ["m/a.png", "m/n1.png", "m/n2.png"],
                ["m/b.png", "m/n1.png", "m/n2.png"],
                ["m/c.png", "m/n1.png", "m/n2.png"],
                ["m/d.png", "m/n1.png", "m/n2.png"],
            ]
        ),
        support_xy=np.full((4, 3, 2), 24.0, dtype=np.float32),
        correct_projection_offsets_xy=np.zeros((4, 3, 2), dtype=np.float32),
        correct_projection_valid=np.ones((4, 3), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.ones((4, 3, 2), dtype=np.float32),
        coherent_wrong_projection_valid=np.ones((4, 3), dtype=bool),
        spatial_target_observed=np.asarray([[True, False, False]] * 4),
        spatial_target_dustbin=np.asarray([[False, True, True]] * 4),
        split_names=np.asarray(["inner_train", "inner_train", "inner_validation", "inner_validation"]),
        metadata={
            "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT,
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_scorer_must_not_load_this_artifact": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "candidate_set": "same_track_mapping_support_plus_radio_pca_global_landmark_ann_hard_negatives",
            "pose_target_semantics": "correct_vs_coherent_wrong_simple_radial_candidate_projection_offsets_v1",
            "candidate_count": 3,
            "anchors_per_pose_pair": 2,
            "spatial_search_radius_px": 8.0,
            "source_observation_pairs_sha256": source_sha256,
            "train_pairs_sha256": "train-pairs",
            "colmap_images_sha256": "images",
            "colmap_points3d_sha256": "points",
            "hard_pose_source": "train_only_post_inference_hard_modes_v2",
            "anchor_jitter_radius_px": 4,
            "min_wrong_positive_offset_delta_px": 1.0,
        },
    )


def test_converter_preserves_only_fixed_identity_inputs(tmp_path) -> None:
    source_path = tmp_path / "source.npz"
    save_candidate_pose_rgb_spatial_observation_pairs(_source_pairs(), source_path)
    hard_path = tmp_path / "hard.npz"
    save_candidate_pose_rgb_spatial_hard_pose_pairs(
        _hard_pairs(file_sha256_short(source_path)), hard_path
    )
    output = tmp_path / "identity.npz"
    summary = build_candidate_pose_rgb_spatial_hard_pose_identity_pairs(
        hard_pose_pairs=hard_path,
        observation_pairs=source_path,
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )
    artifact = load_candidate_pose_rgb_spatial_observation_pairs(output)
    assert summary["row_count"] == 4
    assert artifact.negative_count == 2
    np.testing.assert_array_equal(artifact.anchor_ids, [10, 11, 12, 13])
    np.testing.assert_array_equal(artifact.positive_track_ids, [10, 11, 12, 13])
    assert artifact.metadata["hard_pose_projection_targets_serialized"] is False
    with np.load(output, allow_pickle=False) as payload:
        assert "correct_projection_offsets_xy" not in payload.files
        assert "coherent_wrong_projection_offsets_xy" not in payload.files
    target_path = tmp_path / "hard_identity_targets.npz"
    target_summary = build_candidate_pose_rgb_spatial_hard_pose_identity_targets(
        hard_pose_pairs=hard_path,
        hard_pose_identity_pairs=output,
        coherent_wrong_local_radius_px=8.0,
        output=target_path,
        summary_json=tmp_path / "hard_identity_targets.json",
        force=False,
    )
    targets = load_candidate_pose_rgb_spatial_hard_pose_identity_targets(target_path)
    assert target_summary["target_count"] == 4
    assert not np.any(targets.hard_negative_candidate_mask[:, 0])
    assert np.all(np.any(targets.hard_negative_candidate_mask[:, 1:], axis=1))
    with np.load(target_path, allow_pickle=False) as payload:
        assert "coherent_wrong_projection_offsets_xy" not in payload.files


def test_converter_rejects_hard_pose_from_another_observation_source(tmp_path) -> None:
    source_path = tmp_path / "source.npz"
    save_candidate_pose_rgb_spatial_observation_pairs(_source_pairs(), source_path)
    hard_path = tmp_path / "hard.npz"
    save_candidate_pose_rgb_spatial_hard_pose_pairs(_hard_pairs("stale-source"), hard_path)
    with pytest.raises(ValueError, match="do not derive"):
        build_candidate_pose_rgb_spatial_hard_pose_identity_pairs(
            hard_pose_pairs=hard_path,
            observation_pairs=source_path,
            output=tmp_path / "identity.npz",
            summary_json=tmp_path / "summary.json",
            force=False,
        )
