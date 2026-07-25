from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_hard_repeat_targets import (
    build_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    save_candidate_pose_rgb_spatial_training_targets,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([1, 2], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/a.png"]),
        split_names=np.asarray(["train", "train"]),
        xy=np.asarray([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32),
        point_sources=np.asarray(["alike", "alike"]),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        candidate_coarse_similarities=np.full((2, 2), 0.5, dtype=np.float32),
        candidate_prior_probabilities=np.asarray(
            [[0.6, 0.2], [0.6, 0.2]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.2, 0.2], dtype=np.float32),
        support_image_ids=np.asarray(
            [[["m/a.png"], ["m/b.png"]], [["m/c.png"], ["m/d.png"]]]
        ),
        support_xy=np.zeros((2, 2, 1, 2), dtype=np.float32),
        support_view_valid=np.ones((2, 2, 1), dtype=bool),
        support_view_weights=np.ones((2, 2, 1), dtype=np.float32),
        support_coverage_counts=np.ones((2, 2, 1), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
        },
    )


def test_builder_materializes_only_distinct_correct_and_coherent_wrong_candidates(tmp_path) -> None:
    layout_path = tmp_path / "layout.npz"
    save_candidate_pose_rgb_spatial_layout(_layout(), layout_path)
    target_path = tmp_path / "targets.npz"
    targets = CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray([1, 2], dtype=np.int64),
        query_ids=np.asarray(["q/a.png", "q/a.png"]),
        spatial_target_offsets_xy=np.zeros((2, 2, 2), dtype=np.float32),
        spatial_target_observed=np.ones((2, 2), dtype=bool),
        spatial_target_dustbin=np.zeros((2, 2), dtype=bool),
        pair_query_ids=np.asarray(["q/a.png"]),
        pair_ids=np.asarray([7], dtype=np.int64),
        pair_point_offsets=np.asarray([0, 2], dtype=np.int64),
        pair_source_point_ids=np.asarray([1, 2], dtype=np.int64),
        correct_projection_offsets_xy=np.asarray(
            [[[0.0, 0.0], [7.0, 0.0]], [[0.0, 0.0], [7.0, 0.0]]],
            dtype=np.float32,
        ),
        correct_projection_valid=np.ones((2, 2), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.asarray(
            [[[7.0, 0.0], [0.0, 0.0]], [[7.0, 0.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        coherent_wrong_projection_valid=np.ones((2, 2), dtype=bool),
        metadata={
            "format": "candidate_pose_rgb_spatial_targets_v1",
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": file_sha256_short(layout_path),
            "train_pairs_sha256": "pairs",
            "support_geometry_index_sha256": "geometry",
            "projected_landmark_bank_sha256": "bank",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "spatial_search_radius_px": 12.0,
        },
    )
    save_candidate_pose_rgb_spatial_training_targets(targets, target_path)
    output = tmp_path / "hard_repeat.npz"
    summary = build_candidate_pose_rgb_spatial_hard_repeat_targets(
        rgb_spatial_layout=layout_path,
        rgb_spatial_targets=target_path,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        output=output,
        summary_json=tmp_path / "summary.json",
        force=False,
    )
    result = load_candidate_pose_rgb_spatial_hard_repeat_targets(output)
    assert summary["target_count"] == 2
    np.testing.assert_array_equal(result.positive_candidate_indices, [0, 0])
    np.testing.assert_array_equal(result.negative_candidate_indices, [1, 1])


def test_builder_materializes_all_distinct_wrong_candidates_in_v2(tmp_path) -> None:
    layout_path = tmp_path / "layout_multi.npz"
    layout = CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([1], dtype=np.int64),
        query_ids=np.asarray(["q/a.png"]),
        split_names=np.asarray(["train"]),
        xy=np.asarray([[10.0, 10.0]], dtype=np.float32),
        point_sources=np.asarray(["alike"]),
        candidate_track_ids=np.asarray([[10, 11, 12]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[0, 1, 2]], dtype=np.int64),
        candidate_coarse_similarities=np.full((1, 3), 0.5, dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.5, 0.2, 0.1]], dtype=np.float32),
        null_probabilities=np.asarray([0.2], dtype=np.float32),
        support_image_ids=np.asarray([[["m/a.png"], ["m/b.png"], ["m/c.png"]]]),
        support_xy=np.zeros((1, 3, 1, 2), dtype=np.float32),
        support_view_valid=np.ones((1, 3, 1), dtype=bool),
        support_view_weights=np.ones((1, 3, 1), dtype=np.float32),
        support_coverage_counts=np.ones((1, 3, 1), dtype=np.int32),
        metadata=_layout().metadata,
    )
    save_candidate_pose_rgb_spatial_layout(layout, layout_path)
    target_path = tmp_path / "targets_multi.npz"
    targets = CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray([1], dtype=np.int64),
        query_ids=np.asarray(["q/a.png"]),
        spatial_target_offsets_xy=np.asarray([[[0.0, 0.0], [8.0, 0.0], [8.0, 0.0]]]),
        spatial_target_observed=np.asarray([[True, False, False]]),
        spatial_target_dustbin=np.asarray([[False, True, True]]),
        pair_query_ids=np.asarray(["q/a.png"]),
        pair_ids=np.asarray([7], dtype=np.int64),
        pair_point_offsets=np.asarray([0, 1], dtype=np.int64),
        pair_source_point_ids=np.asarray([1], dtype=np.int64),
        correct_projection_offsets_xy=np.asarray([[[0.0, 0.0], [8.0, 0.0], [8.0, 0.0]]]),
        correct_projection_valid=np.ones((1, 3), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.asarray(
            [[[9.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
        ),
        coherent_wrong_projection_valid=np.ones((1, 3), dtype=bool),
        metadata={
            "format": "candidate_pose_rgb_spatial_targets_v1",
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "rgb_spatial_layout_sha256": file_sha256_short(layout_path),
            "train_pairs_sha256": "pairs",
            "support_geometry_index_sha256": "geometry",
            "projected_landmark_bank_sha256": "bank",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
            "spatial_search_radius_px": 12.0,
            "spatial_supervision_mode": "registered_exact_identity",
        },
    )
    save_candidate_pose_rgb_spatial_training_targets(targets, target_path)
    output = tmp_path / "hard_repeat_multi.npz"
    summary = build_candidate_pose_rgb_spatial_hard_repeat_targets(
        rgb_spatial_layout=layout_path,
        rgb_spatial_targets=target_path,
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        max_negatives_per_source_pair=0,
        output=output,
        summary_json=tmp_path / "summary_multi.json",
        force=False,
    )
    result = load_candidate_pose_rgb_spatial_hard_repeat_targets(output)
    assert summary["target_count"] == 2
    assert result.metadata["format"] == CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT
    np.testing.assert_array_equal(result.positive_candidate_indices, [0, 0])
    np.testing.assert_array_equal(result.negative_candidate_indices, [1, 2])
