from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT,
    CandidatePoseRGBSpatialHardRepeatTargets,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
    save_candidate_pose_rgb_spatial_hard_repeat_targets,
    select_coherent_hard_repeat_candidate_edges,
    select_coherent_hard_repeat_candidates,
)


def _metadata() -> dict[str, object]:
    return {
        "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
        "training_only_target_artifact": True,
        "contains_ground_truth": True,
        "contains_validation_or_test_targets": False,
        "runtime_layout_is_target_free": True,
        "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "rgb_spatial_layout_sha256": "layout",
        "rgb_spatial_targets_sha256": "targets",
        "projection_space_id": "projection",
        "descriptor_space_id": "descriptor",
        "candidate_count": 3,
        "positive_radius_px": 4.0,
        "negative_radius_px": 4.0,
    }


def test_coherent_hard_repeat_selection_prefers_nearby_distinct_candidate() -> None:
    positive, negative = select_coherent_hard_repeat_candidates(
        correct_offsets_xy=np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0], [8.0, 0.0]],
                [[8.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
            ],
            dtype=np.float32,
        ),
        correct_valid=np.ones((2, 3), dtype=bool),
        wrong_offsets_xy=np.asarray(
            [
                [[9.0, 0.0], [0.5, 0.5], [1.5, 0.0]],
                [[0.5, 0.5], [9.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        wrong_valid=np.ones((2, 3), dtype=bool),
        candidate_prior_probabilities=np.asarray(
            [[0.2, 0.7, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32
        ),
        positive_radius_px=4.0,
        negative_radius_px=4.0,
    )
    # Row 0 uses candidate 0 under correct pose; candidate 1 is excluded
    # because it is also correct-local, leaving candidate 2 as the coherent
    # wrong edge.  Row 1's best correct candidate is 2, so candidate 0 is the
    # distinct coherent-wrong edge.
    np.testing.assert_array_equal(positive, [0, 2])
    np.testing.assert_array_equal(negative, [2, 0])


def test_exact_identity_hard_repeat_keeps_other_correct_local_track_as_negative() -> None:
    positive, negative = select_coherent_hard_repeat_candidates(
        correct_offsets_xy=np.asarray([[[1.0, 0.0], [0.5, 0.0], [6.0, 0.0]]]),
        correct_valid=np.ones((1, 3), dtype=bool),
        wrong_offsets_xy=np.asarray([[[8.0, 0.0], [1.0, 0.0], [8.0, 0.0]]]),
        wrong_valid=np.ones((1, 3), dtype=bool),
        candidate_prior_probabilities=np.asarray([[0.2, 0.7, 0.1]], dtype=np.float32),
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        positive_candidate_mask=np.asarray([[True, False, False]]),
    )
    # Candidate 1 is correct-pose local but is not the registered identity,
    # so it remains the intended coherent-repeat negative in exact mode.
    np.testing.assert_array_equal(positive, [0])
    np.testing.assert_array_equal(negative, [1])


def test_multi_negative_selection_keeps_all_distinct_coherent_wrong_candidates() -> None:
    rows, positive, negative = select_coherent_hard_repeat_candidate_edges(
        correct_offsets_xy=np.asarray([[[0.0, 0.0], [1.0, 0.0], [6.0, 0.0], [6.0, 0.0]]]),
        correct_valid=np.ones((1, 4), dtype=bool),
        wrong_offsets_xy=np.asarray([[[9.0, 0.0], [0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        wrong_valid=np.ones((1, 4), dtype=bool),
        candidate_prior_probabilities=np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32),
        positive_radius_px=4.0,
        negative_radius_px=4.0,
        positive_candidate_mask=np.asarray([[True, False, False, False]]),
        max_negatives_per_source_pair=0,
    )
    np.testing.assert_array_equal(rows, [0, 0, 0])
    np.testing.assert_array_equal(positive, [0, 0, 0])
    # Ranking is train-only: nearest projection, then frozen prior, then slot.
    np.testing.assert_array_equal(negative, [1, 2, 3])


def test_multi_negative_artifact_requires_one_positive_and_distinct_edges() -> None:
    metadata = {
        **_metadata(),
        "format": CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT,
        "negative_selection": "all_or_capped_distinct_coherent_wrong_local_candidates_v1",
        "max_negatives_per_source_pair": 0,
    }
    artifact = CandidatePoseRGBSpatialHardRepeatTargets(
        source_point_ids=np.asarray([11, 11], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png"]),
        pair_ids=np.asarray([5, 5], dtype=np.int64),
        positive_candidate_indices=np.asarray([0, 0], dtype=np.int64),
        negative_candidate_indices=np.asarray([1, 2], dtype=np.int64),
        positive_offsets_xy=np.zeros((2, 2), dtype=np.float32),
        negative_offsets_xy=np.ones((2, 2), dtype=np.float32),
        metadata=metadata,
    )
    assert artifact.count == 2
    with pytest.raises(ValueError, match="positive identity"):
        CandidatePoseRGBSpatialHardRepeatTargets(
            **{
                **artifact.__dict__,
                "positive_candidate_indices": np.asarray([0, 1], dtype=np.int64),
            }
        )
    with pytest.raises(ValueError, match="invalid"):
        CandidatePoseRGBSpatialHardRepeatTargets(
            **{
                **artifact.__dict__,
                "negative_candidate_indices": np.asarray([1, 1], dtype=np.int64),
            }
        )


def test_hard_repeat_artifact_round_trips_and_rejects_same_identity(tmp_path) -> None:
    artifact = CandidatePoseRGBSpatialHardRepeatTargets(
        source_point_ids=np.asarray([11, 13], dtype=np.int64),
        query_ids=np.asarray(["query/a.png", "query/a.png"]),
        pair_ids=np.asarray([5, 5], dtype=np.int64),
        positive_candidate_indices=np.asarray([0, 1], dtype=np.int64),
        negative_candidate_indices=np.asarray([2, 2], dtype=np.int64),
        positive_offsets_xy=np.zeros((2, 2), dtype=np.float32),
        negative_offsets_xy=np.ones((2, 2), dtype=np.float32),
        metadata=_metadata(),
    )
    path = tmp_path / "hard_repeat.npz"
    save_candidate_pose_rgb_spatial_hard_repeat_targets(artifact, path)
    loaded = load_candidate_pose_rgb_spatial_hard_repeat_targets(path)
    np.testing.assert_array_equal(loaded.negative_candidate_indices, [2, 2])
    with pytest.raises(ValueError, match="invalid"):
        CandidatePoseRGBSpatialHardRepeatTargets(
            **{**artifact.__dict__, "negative_candidate_indices": np.asarray([0, 1])}
        )
