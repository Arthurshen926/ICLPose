from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial_full_pool_scores import (
    CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
    CandidatePoseRGBSpatialTrainFullPoolScores,
    load_candidate_pose_rgb_spatial_train_full_pool_scores,
    save_candidate_pose_rgb_spatial_train_full_pool_scores,
)


def _metadata() -> dict[str, object]:
    return {
        "format": CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "runtime_layout_is_target_free": True,
        "projection_after_network_only": True,
        "train_rows_only": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "complete_train_coverage": True,
        "hypothesis_artifacts": [{"path": "frozen.npz", "sha256": "frozen"}],
        "hypothesis_semantic_hash": "semantic",
        "rgb_spatial_layout_sha256": "layout",
        "checkpoint": {"path": "checkpoint.pt", "sha256": "checkpoint"},
    }


def _scores(metadata: dict[str, object] | None = None) -> CandidatePoseRGBSpatialTrainFullPoolScores:
    return CandidatePoseRGBSpatialTrainFullPoolScores(
        source_artifact_indices=np.asarray([0, 0], dtype=np.int64),
        source_row_indices=np.asarray([3, 9], dtype=np.int64),
        query_ids=np.asarray(["seq1/train.png", "seq1/train.png"]),
        split_names=np.asarray(["train", "train"]),
        evaluation_labels=np.asarray(["frozen", "frozen"]),
        hypothesis_indices=np.asarray([3, 9], dtype=np.int64),
        pose_log_likelihood_ratios=np.asarray([0.3, -0.2], dtype=np.float32),
        metadata=_metadata() if metadata is None else metadata,
    )


def test_train_full_pool_scores_round_trip_without_pose_or_target_fields(tmp_path) -> None:
    path = tmp_path / "scores.npz"
    save_candidate_pose_rgb_spatial_train_full_pool_scores(_scores(), path)
    loaded = load_candidate_pose_rgb_spatial_train_full_pool_scores(path)

    assert loaded.row_count == 2
    np.testing.assert_array_equal(loaded.source_row_indices, [3, 9])
    assert loaded.metadata["contains_target_fields"] is False
    with np.load(path, allow_pickle=False) as payload:
        assert not any("pose" in name for name in payload.files if name != "pose_log_likelihood_ratios")
        assert not any(token in name for name in payload.files for token in ("target", "residual"))


def test_train_full_pool_scores_reject_target_bearing_metadata() -> None:
    metadata = _metadata()
    metadata["contains_target_fields"] = True
    with pytest.raises(ValueError, match="target-free contract"):
        _scores(metadata)


def test_train_full_pool_scores_require_an_explicit_coverage_declaration() -> None:
    metadata = _metadata()
    del metadata["complete_train_coverage"]
    with pytest.raises(ValueError, match="coverage declaration"):
        _scores(metadata)


def test_train_full_pool_scores_reject_duplicate_frozen_source_rows() -> None:
    with pytest.raises(ValueError, match="not unique"):
        CandidatePoseRGBSpatialTrainFullPoolScores(
            source_artifact_indices=np.asarray([0, 0], dtype=np.int64),
            source_row_indices=np.asarray([3, 3], dtype=np.int64),
            query_ids=np.asarray(["q", "q"]),
            split_names=np.asarray(["train", "train"]),
            evaluation_labels=np.asarray(["frozen", "frozen"]),
            hypothesis_indices=np.asarray([3, 4], dtype=np.int64),
            pose_log_likelihood_ratios=np.asarray([0.0, 0.1], dtype=np.float32),
            metadata=_metadata(),
        )
