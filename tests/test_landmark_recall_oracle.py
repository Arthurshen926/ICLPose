from __future__ import annotations

import csv

import numpy as np
import pytest

from feature_extract.tools.vfm.diagnose_landmark_recall_oracle import _write_csv
from feature_extract.vfm.localization.landmark_recall_oracle import (
    LandmarkRecallRecord,
    rank_correct_landmarks_batch,
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


def test_rank_correct_landmark_collapses_multiple_prototypes_per_track() -> None:
    record = rank_correct_landmark(
        query_descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
        landmark_features=np.asarray(
            [
                [0.99, 0.01],
                [0.98, 0.02],
                [0.97, 0.03],
                [0.96, 0.04],
            ],
            dtype=np.float32,
        ),
        landmark_track_ids=np.asarray([10, 10, 20, 30], dtype=np.int64),
        correct_track_id=20,
    )

    assert record.rank1_track_id == 10
    assert record.correct_rank == 2


@pytest.mark.parametrize("multi_prototype", [False, True])
def test_batched_landmark_rank_matches_scalar_track_rank(multi_prototype: bool) -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [0.0, 1.0]],
        dtype=np.float32,
    )
    track_ids = np.asarray([10, 10, 20, 30] if multi_prototype else [10, 11, 20, 30], dtype=np.int64)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    correct_ids = [20, 30]

    expected = [
        rank_correct_landmark(
            query_descriptor=query,
            landmark_features=features,
            landmark_track_ids=track_ids,
            correct_track_id=correct_track,
            query_id=f"q{index}",
        )
        for index, (query, correct_track) in enumerate(zip(queries, correct_ids))
    ]
    actual = rank_correct_landmarks_batch(
        query_descriptors=queries,
        landmark_features=features,
        landmark_track_ids=track_ids,
        correct_track_ids=correct_ids,
        query_ids=["q0", "q1"],
        device="cpu",
        batch_size=2,
    )

    assert [record.correct_rank for record in actual] == [record.correct_rank for record in expected]
    assert [record.rank1_track_id for record in actual] == [record.rank1_track_id for record in expected]
    np.testing.assert_allclose(
        [record.correct_score for record in actual],
        [record.correct_score for record in expected],
        atol=1e-6,
    )


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
    assert summary["mean_reciprocal_rank"] == pytest.approx(0.4)
    assert summary["macro_query_recall_at_1"] == pytest.approx(0.25)


def test_recall_csv_schema_includes_oracle_geometry(tmp_path) -> None:
    output = tmp_path / "rows.csv"
    _write_csv(
        output,
        [
            LandmarkRecallRecord(
                "q0",
                1,
                2,
                0.8,
                3,
                0.9,
                0.1,
                query_x=10.0,
                query_y=20.0,
                landmark_x=1.0,
                landmark_y=2.0,
                landmark_z=3.0,
            )
        ],
    )

    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["query_x"] == "10.0"
    assert row["landmark_z"] == "3.0"
