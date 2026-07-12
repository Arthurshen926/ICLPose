from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.measurement_pose_evidence import (
    FrozenMeasurementPoseEvidence,
    FrozenMeasurementUpdateEvidence,
)


def test_missing_measurement_assignment_remains_unknown() -> None:
    evidence = FrozenMeasurementPoseEvidence(
        feature_set="measurement_plus_support",
        verification_threshold=0.64,
        probability_by_assignment={("query", 10, 100): 0.9},
        manifest={},
    )

    matrix = evidence.candidate_probability_matrix(
        query_id="query",
        token_indices=(10, 11),
        measured_track_ids=(100, 200),
        candidate_track_ids=np.asarray([[101, 100], [200, 201]], dtype=np.int64),
    )

    assert np.isclose(matrix[0, 1], 0.9)
    assert np.isnan(matrix[0, 0])
    assert np.all(np.isnan(matrix[1]))


def test_measured_track_must_exist_in_frozen_candidate_pool() -> None:
    evidence = FrozenMeasurementPoseEvidence(
        feature_set="measurement_plus_support",
        verification_threshold=0.64,
        probability_by_assignment={("query", 10, 100): 0.9},
        manifest={},
    )

    with pytest.raises(ValueError, match="absent from its frozen top-L pool"):
        evidence.candidate_probability_matrix(
            query_id="query",
            token_indices=(10,),
            measured_track_ids=(100,),
            candidate_track_ids=np.asarray([[101, 102]], dtype=np.int64),
        )


def test_update_evidence_keeps_missing_rows_unknown() -> None:
    evidence = FrozenMeasurementUpdateEvidence(
        update_threshold=0.7,
        update_by_assignment={("query", 10, 100): (0.8, (12.5, 15.0))},
        manifest={},
    )

    probabilities, xy = evidence.assignment_updates(
        query_id="query",
        token_indices=(10, 11),
        measured_track_ids=(100, 200),
    )

    assert np.isclose(probabilities[0], 0.8)
    assert np.allclose(xy[0], (12.5, 15.0))
    assert np.isnan(probabilities[1])
    assert np.all(np.isnan(xy[1]))
