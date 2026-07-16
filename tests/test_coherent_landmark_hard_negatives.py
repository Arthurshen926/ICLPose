from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.coherent_landmark_hard_negatives import (
    CoherentLandmarkHardNegativeIndex,
    attach_coherent_landmark_hard_negatives,
)


def _artifacts(tmp_path):
    proposals = tmp_path / "proposals.npz"
    np.savez_compressed(
        proposals,
        query_ids=np.asarray(["train/q.png", "train/q.png"]),
        xy=np.asarray([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[100, 200, 300], [400, 500, 600]]),
    )
    metadata = {
        "format": "pose_conditioned_system_hard_modes_v2",
        "training_only_target_artifact": True,
        "pose_or_ground_truth_used_for_hypothesis_generation": False,
        "split_names": ["train"],
        "inputs": {"proposals_sha256": file_sha256_short(proposals)},
    }
    artifact = tmp_path / "hard_modes.npz"
    np.savez_compressed(
        artifact,
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        selected_columns=np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64),
        hard_negative_mask_TARGET_ONLY=np.asarray(
            [[False, True, True], [True, False, False]], dtype=bool
        ),
        hard_mode_ids_TARGET_ONLY=np.asarray([[0, 1], [2, -1]], dtype=np.int64),
        hard_mode_candidate_mask_TARGET_ONLY=np.asarray(
            [
                [[False, True, False], [False, False, True]],
                [[True, False, False], [False, False, False]],
            ],
            dtype=bool,
        ),
        query_ids=np.asarray(["train/q.png", "train/q.png"]),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return artifact, proposals


def test_coherent_index_recovers_explicit_wrong_track_ids_with_lineage(tmp_path) -> None:
    artifact, proposals = _artifacts(tmp_path)
    index = CoherentLandmarkHardNegativeIndex.from_pose_mode_artifact(
        artifact, proposals
    )

    tracks, modes, distance = index.nearest_tracks(
        "train/q.png", np.asarray([11.0, 10.0]), max_distance_px=3.0
    )

    np.testing.assert_array_equal(tracks, np.asarray([200, 300]))
    np.testing.assert_array_equal(modes, np.asarray([0, 1]))
    assert distance == pytest.approx(1.0)


def test_coherent_index_rejects_stale_proposals(tmp_path) -> None:
    artifact, proposals = _artifacts(tmp_path)
    replacement = tmp_path / "stale.npz"
    np.savez_compressed(
        replacement,
        query_ids=np.asarray(["train/q.png"]),
        xy=np.asarray([[10.0, 10.0]], dtype=np.float32),
        candidate_track_ids=np.asarray([[100, 200, 300]]),
    )
    with pytest.raises(ValueError, match="lineage mismatch"):
        CoherentLandmarkHardNegativeIndex.from_pose_mode_artifact(
            artifact, replacement
        )


@dataclass(frozen=True)
class _Samples:
    landmark_track_ids: np.ndarray
    pair_query_ids: np.ndarray
    landmark_sample_pair_indices: np.ndarray
    landmark_query_xy: np.ndarray
    landmark_known_positive_offsets: np.ndarray | None = None
    landmark_known_positive_track_ids: np.ndarray | None = None
    landmark_strict_positive_offsets: np.ndarray | None = None
    landmark_strict_positive_track_ids: np.ndarray | None = None
    landmark_coherent_hard_negative_offsets: np.ndarray | None = None
    landmark_coherent_hard_negative_track_ids: np.ndarray | None = None
    landmark_coherent_hard_negative_mode_ids: np.ndarray | None = None


def test_attach_coherent_tracks_excludes_target_and_ambiguity(tmp_path) -> None:
    artifact, proposals = _artifacts(tmp_path)
    index = CoherentLandmarkHardNegativeIndex.from_pose_mode_artifact(
        artifact, proposals
    )
    samples = _Samples(
        landmark_track_ids=np.asarray([200, 700]),
        pair_query_ids=np.asarray(["train/q.png"]),
        landmark_sample_pair_indices=np.asarray([0, 0]),
        landmark_query_xy=np.asarray([[10.0, 10.0], [80.0, 80.0]]),
        landmark_known_positive_offsets=np.asarray([0, 1, 1]),
        landmark_known_positive_track_ids=np.asarray([300]),
    )

    attached, audit = attach_coherent_landmark_hard_negatives(
        samples, index, max_distance_px=4.0
    )

    np.testing.assert_array_equal(
        attached.landmark_coherent_hard_negative_offsets, np.asarray([0, 0, 0])
    )
    assert attached.landmark_coherent_hard_negative_track_ids.size == 0
    assert audit["matched_landmark_row_count"] == 0


def test_attach_preserves_one_track_in_multiple_pose_modes() -> None:
    index = CoherentLandmarkHardNegativeIndex(
        query_xy_by_id={"train/q.png": np.asarray([[10.0, 10.0]], dtype=np.float32)},
        track_ids_by_id={"train/q.png": (np.asarray([200, 200, 300]),)},
        mode_ids_by_id={"train/q.png": (np.asarray([4, 7, 7]),)},
        metadata={"format": "coherent_landmark_hard_negative_index_v1"},
    )
    samples = _Samples(
        landmark_track_ids=np.asarray([100]),
        pair_query_ids=np.asarray(["train/q.png"]),
        landmark_sample_pair_indices=np.asarray([0]),
        landmark_query_xy=np.asarray([[10.0, 10.0]]),
    )

    attached, audit = attach_coherent_landmark_hard_negatives(
        samples, index, max_distance_px=1.0
    )

    np.testing.assert_array_equal(
        attached.landmark_coherent_hard_negative_track_ids,
        np.asarray([200, 200, 300]),
    )
    np.testing.assert_array_equal(
        attached.landmark_coherent_hard_negative_mode_ids,
        np.asarray([4, 7, 7]),
    )
    assert audit["coherent_track_incidence_count"] == 3
