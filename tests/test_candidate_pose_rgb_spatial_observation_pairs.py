from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT,
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
    save_candidate_pose_rgb_spatial_observation_pairs,
)


def _metadata() -> dict[str, object]:
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
        "inner_validation_fold_index": 0,
        "train_query_layout_sha256": "layout",
        "colmap_images_bin_sha256": "images",
        "support_observation_index_sha256": "support",
        "hard_negative_context_cache_sha256": "context",
        "hard_negative_landmark_bank_sha256": "bank",
        "train_query_image_list_sha256": "queries",
    }


def _pairs(*, metadata: dict[str, object] | None = None) -> CandidatePoseRGBSpatialObservationPairs:
    return CandidatePoseRGBSpatialObservationPairs(
        anchor_ids=np.asarray([4, 9], dtype=np.int64),
        query_image_ids=np.asarray(["query/a.png", "query/b.png"]),
        query_xy=np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        positive_support_image_ids=np.asarray(["map/a.png", "map/b.png"]),
        positive_support_xy=np.asarray([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32),
        positive_track_ids=np.asarray([101, 102], dtype=np.int64),
        negative_support_image_ids=np.asarray(
            [["map/c.png", "map/d.png"], ["map/e.png", "map/f.png"]]
        ),
        negative_support_xy=np.asarray(
            [
                [[12.0, 22.0], [13.0, 23.0]],
                [[32.0, 42.0], [33.0, 43.0]],
            ],
            dtype=np.float32,
        ),
        negative_track_ids=np.asarray([[201, 202], [203, 204]], dtype=np.int64),
        negative_sources=np.asarray(
            [["radio_ann", "radio_ann"], ["radio_ann", "radio_ann"]]
        ),
        split_names=np.asarray(["inner_train", "inner_validation"]),
        metadata=_metadata() if metadata is None else metadata,
    )


def test_observation_pair_artifact_round_trip_preserves_train_only_contract(tmp_path) -> None:
    path = tmp_path / "pairs.npz"
    expected = _pairs()
    save_candidate_pose_rgb_spatial_observation_pairs(expected, path)
    actual = load_candidate_pose_rgb_spatial_observation_pairs(path)

    np.testing.assert_array_equal(actual.anchor_ids, expected.anchor_ids)
    np.testing.assert_array_equal(actual.negative_track_ids, expected.negative_track_ids)
    assert actual.metadata["runtime_scorer_must_not_load_this_artifact"] is True


def test_observation_pairs_reject_same_track_as_a_hard_negative() -> None:
    with pytest.raises(ValueError, match="invalid"):
        CandidatePoseRGBSpatialObservationPairs(
            **{
                **_pairs().__dict__,
                "negative_track_ids": np.asarray([[101, 202], [203, 204]], dtype=np.int64),
            }
        )


def test_observation_pairs_reject_validation_or_test_target_metadata() -> None:
    metadata = _metadata()
    metadata["contains_validation_or_test_targets"] = True
    with pytest.raises(ValueError, match="train-only"):
        _pairs(metadata=metadata)


def test_observation_pairs_reject_tampered_metadata_on_load(tmp_path) -> None:
    path = tmp_path / "pairs.npz"
    save_candidate_pose_rgb_spatial_observation_pairs(_pairs(), path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["runtime_scorer_must_not_load_this_artifact"] = False
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ValueError, match="train-only"):
        load_candidate_pose_rgb_spatial_observation_pairs(path)
