from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.score_v5_dynamic_absolute_context_pose_evidence import (
    _formal_p1_mixed_evidence_for_query,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    POINT_SOURCE_ALIKE,
    POINT_SOURCE_RADIO_FINAL,
    POINT_SOURCE_RADIO_INTERMEDIATE,
)


def _write_formal_points(path, *, candidate_sha: str, detector_sha: str, bank_sha: str) -> None:
    point_sources = np.asarray(
        [POINT_SOURCE_ALIKE] * 64
        + [POINT_SOURCE_RADIO_INTERMEDIATE] * 64
        + [POINT_SOURCE_RADIO_FINAL] * 64,
        dtype=np.str_,
    )
    count = len(point_sources)
    tracks = np.full((count, 20), -1, dtype=np.int64)
    tracks[:, 0] = 100
    bank_rows = np.full_like(tracks, -1)
    bank_rows[:, 0] = 0
    prototypes = np.full_like(tracks, -1)
    prototypes[:, 0] = 0
    similarities = np.zeros((count, 20), dtype=np.float32)
    similarities[:, 0] = 0.8
    probabilities = np.zeros((count, 20), dtype=np.float32)
    probabilities[:, 0] = 0.9
    metadata = {
        "format": MIXED_VERIFICATION_POINTS_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "candidate_reselection": False,
        "candidate_set": "fixed_full_global_faiss_top_l_unique_tracks",
        "global_landmark_ann_scope": "full_projected_landmark_bank_only",
        "candidate_top_k": 20,
        "candidate_fit_artifact_sha256": candidate_sha,
        "detector_query_cache_sha256": detector_sha,
        "projected_landmark_bank_sha256": bank_sha,
        "exported_splits": ["train", "validation"],
        "test_source_points_materialized": False,
    }
    np.savez_compressed(
        path,
        source_point_ids=np.arange(count, dtype=np.int64),
        query_ids=np.full((count,), "q.png", dtype="<U16"),
        split_names=np.full((count,), "validation", dtype="<U16"),
        xy=np.stack(
            [np.arange(count, dtype=np.float32), np.arange(count, dtype=np.float32)], axis=1
        ),
        point_sources=point_sources,
        source_detector_rows=np.concatenate(
            [np.arange(1, 65, dtype=np.int64), np.full((128,), -1, dtype=np.int64)]
        ),
        descriptors=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (count, 1)),
        candidate_bank_rows=bank_rows,
        candidate_track_ids=tracks,
        candidate_prototype_ids=prototypes,
        candidate_coarse_similarities=similarities,
        candidate_prior_probabilities=probabilities,
        null_probabilities=np.full((count,), 0.1, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _inputs(tmp_path):
    detector_path = tmp_path / "detector.bin"
    candidate_path = tmp_path / "candidate.bin"
    bank_path = tmp_path / "bank.bin"
    for path, payload in (
        (detector_path, b"detector"),
        (candidate_path, b"candidate"),
        (bank_path, b"bank"),
    ):
        path.write_bytes(payload)
    points_path = tmp_path / "mixed.npz"
    _write_formal_points(
        points_path,
        candidate_sha=file_sha256_short(candidate_path),
        detector_sha=file_sha256_short(detector_path),
        bank_sha=file_sha256_short(bank_path),
    )
    detector = {
        "image_ids": np.asarray(["q.png"]),
        "offsets": np.asarray([0, 65], dtype=np.int64),
    }
    proposals = {"query_ids": np.full((65,), "q.png", dtype="<U16")}
    return points_path, detector_path, candidate_path, bank_path, detector, proposals


def test_formal_p1_mixed_evidence_enforces_source_and_fit_disjointness(tmp_path) -> None:
    points_path, detector_path, candidate_path, bank_path, detector, proposals = _inputs(tmp_path)

    arrays, selection, metadata = _formal_p1_mixed_evidence_for_query(
        points_path=points_path,
        query_id="q.png",
        query_split="validation",
        detector_path=detector_path,
        candidate_path=candidate_path,
        landmark_bank_path=bank_path,
        detector=detector,
        proposals=proposals,
        selected_rows=np.asarray([0], dtype=np.int64),
        point_count=192,
    )

    assert arrays["candidate_track_ids"].shape == (192, 20)
    assert selection["point_sources"] == {
        POINT_SOURCE_ALIKE: 64,
        POINT_SOURCE_RADIO_INTERMEDIATE: 64,
        POINT_SOURCE_RADIO_FINAL: 64,
    }
    assert selection["candidate_fit_rows_excluded_from_alike"] is True
    assert selection["dense_point_detector_rows_are_sentinel_minus_one"] is True
    assert selection["development_test_source_points_excluded"] is True
    assert metadata["candidate_top_k"] == 20


def test_formal_p1_mixed_evidence_rejects_fit_row_reuse(tmp_path) -> None:
    points_path, detector_path, candidate_path, bank_path, detector, proposals = _inputs(tmp_path)
    with np.load(points_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    arrays["source_detector_rows"][0] = 0
    np.savez_compressed(points_path, **arrays)

    with pytest.raises(ValueError, match="reuses a frozen-hypothesis fit row"):
        _formal_p1_mixed_evidence_for_query(
            points_path=points_path,
            query_id="q.png",
            query_split="validation",
            detector_path=detector_path,
            candidate_path=candidate_path,
            landmark_bank_path=bank_path,
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray([0], dtype=np.int64),
            point_count=192,
        )


def test_formal_p1_validation_rejects_materialized_test_sources(tmp_path) -> None:
    points_path, detector_path, candidate_path, bank_path, detector, proposals = _inputs(tmp_path)
    with np.load(points_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["exported_splits"] = ["train", "validation", "test"]
    metadata["test_source_points_materialized"] = True
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(points_path, **arrays)

    with pytest.raises(ValueError, match="no materialized test sources"):
        _formal_p1_mixed_evidence_for_query(
            points_path=points_path,
            query_id="q.png",
            query_split="validation",
            detector_path=detector_path,
            candidate_path=candidate_path,
            landmark_bank_path=bank_path,
            detector=detector,
            proposals=proposals,
            selected_rows=np.asarray([0], dtype=np.int64),
            point_count=192,
        )
