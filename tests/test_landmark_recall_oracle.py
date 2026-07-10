from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.landmark_recall_oracle import (
    LandmarkRecallRecord,
    summarize_landmark_recall_records,
    rank_correct_landmark,
)
from feature_extract.vfm.query_to_3d_matching import normalize_rows


def test_rank_correct_landmark_reports_rank_and_gap() -> None:
    query = np.asarray([0.9, 0.1], dtype=np.float32)
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    track_ids = np.asarray([11, 22, 33], dtype=np.int64)

    record = rank_correct_landmark(
        query_descriptor=query,
        landmark_features=features,
        landmark_track_ids=track_ids,
        correct_track_id=33,
    )

    assert record.correct_rank == 2
    assert record.rank1_track_id == 11
    assert record.score_gap_to_rank1 > 0.0


def test_rank_correct_landmark_accepts_pre_normalized_landmarks() -> None:
    query = np.asarray([0.9, 0.1], dtype=np.float32)
    features = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    track_ids = np.asarray([11, 22, 33], dtype=np.int64)
    normalized, valid = normalize_rows(features)

    record = rank_correct_landmark(
        query_descriptor=query,
        landmark_features=normalized,
        landmark_track_ids=track_ids,
        correct_track_id=33,
        landmark_features_are_normalized=True,
        valid_landmark_mask=valid,
    )

    assert record.correct_rank == 2
    assert record.rank1_track_id == 11


def test_summarize_landmark_recall_records_computes_recall_at_k() -> None:
    records = [
        LandmarkRecallRecord("q0", 1, 1, 0.9, 1, 1.0, 0.0),
        LandmarkRecallRecord("q0", 2, 5, 0.7, 1, 0.9, 0.2),
        LandmarkRecallRecord("q1", 3, None, None, 4, 0.8, None),
    ]

    summary = summarize_landmark_recall_records(records, top_ks=(1, 5, 10))

    assert summary["sample_count"] == 3
    assert summary["found_count"] == 2
    assert summary["recall_at_1"] == pytest.approx(1 / 3)
    assert summary["recall_at_5"] == pytest.approx(2 / 3)
    assert summary["median_correct_rank"] == pytest.approx(3.0)
