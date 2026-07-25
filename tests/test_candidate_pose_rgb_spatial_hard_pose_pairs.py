from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT,
    CandidatePoseRGBSpatialHardPosePairs,
    load_candidate_pose_rgb_spatial_hard_pose_pairs,
    save_candidate_pose_rgb_spatial_hard_pose_pairs,
)


def _metadata() -> dict[str, object]:
    return {
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
        "source_observation_pairs_sha256": "source",
        "train_pairs_sha256": "train",
        "colmap_images_sha256": "images",
        "colmap_points3d_sha256": "points",
    }


def _pairs() -> CandidatePoseRGBSpatialHardPosePairs:
    count = 4
    candidate_count = 3
    observed = np.zeros((count, candidate_count), dtype=bool)
    observed[:, 0] = True
    return CandidatePoseRGBSpatialHardPosePairs(
        row_ids=np.arange(count, dtype=np.int64),
        pose_pair_ids=np.asarray([3, 3, 7, 7], dtype=np.int64),
        query_image_ids=np.asarray(["q/a.png", "q/a.png", "q/b.png", "q/b.png"]),
        query_xy=np.asarray([[20.0, 20.0], [30.0, 30.0], [40.0, 40.0], [50.0, 50.0]]),
        candidate_track_ids=np.asarray(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.int64
        ),
        support_image_ids=np.asarray(
            [
                ["m/a.png", "m/b.png", "m/c.png"],
                ["m/a.png", "m/b.png", "m/c.png"],
                ["m/d.png", "m/e.png", "m/f.png"],
                ["m/d.png", "m/e.png", "m/f.png"],
            ]
        ),
        support_xy=np.full((count, candidate_count, 2), 32.0, dtype=np.float32),
        correct_projection_offsets_xy=np.asarray(
            [
                [[1.0, -1.0], [3.0, 0.0], [-2.0, 2.0]],
                [[2.0, 0.0], [4.0, -1.0], [-1.0, 3.0]],
                [[-2.0, 1.0], [1.0, 4.0], [2.0, -3.0]],
                [[0.5, 2.0], [-3.0, 1.0], [3.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        correct_projection_valid=np.ones((count, candidate_count), dtype=bool),
        coherent_wrong_projection_offsets_xy=np.full(
            (count, candidate_count, 2), 4.0, dtype=np.float32
        ),
        coherent_wrong_projection_valid=np.ones((count, candidate_count), dtype=bool),
        spatial_target_observed=observed,
        spatial_target_dustbin=~observed,
        split_names=np.asarray(
            ["inner_train", "inner_train", "inner_validation", "inner_validation"]
        ),
        metadata=_metadata(),
    )


def test_hard_pose_pairs_round_trip_and_keep_targets_train_only(tmp_path) -> None:
    pairs = _pairs()
    path = tmp_path / "pairs.npz"
    save_candidate_pose_rgb_spatial_hard_pose_pairs(pairs, path)
    loaded = load_candidate_pose_rgb_spatial_hard_pose_pairs(path)
    assert loaded.group_count == 2
    assert loaded.candidate_count == 3
    np.testing.assert_array_equal(loaded.coherent_wrong_projection_valid, pairs.coherent_wrong_projection_valid)
    assert loaded.metadata["runtime_scorer_must_not_load_this_artifact"] is True


def test_hard_pose_pairs_reject_cross_query_group_and_nonpositive_slot() -> None:
    pairs = _pairs()
    with pytest.raises(ValueError, match="crosses"):
        CandidatePoseRGBSpatialHardPosePairs(
            **{
                **pairs.__dict__,
                "query_image_ids": np.asarray(["q/a.png", "q/z.png", "q/b.png", "q/b.png"]),
            }
        )
    observed = np.asarray(pairs.spatial_target_observed, dtype=bool).copy()
    observed[0, 1] = True
    with pytest.raises(ValueError, match="candidate/target"):
        CandidatePoseRGBSpatialHardPosePairs(
            **{
                **pairs.__dict__,
                "spatial_target_observed": observed,
                "spatial_target_dustbin": ~observed,
            }
        )
