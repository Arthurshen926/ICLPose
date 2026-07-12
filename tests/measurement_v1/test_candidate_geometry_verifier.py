from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.candidate_geometry_verifier import (
    CANDIDATE_GEOMETRY_FEATURE_NAMES,
    CandidateGeometryVerifier,
    candidate_selection_metrics,
)


def test_candidate_geometry_model_round_trip_and_probability() -> None:
    count = len(CANDIDATE_GEOMETRY_FEATURE_NAMES)
    model = CandidateGeometryVerifier(
        feature_names=CANDIDATE_GEOMETRY_FEATURE_NAMES,
        feature_mean=tuple([0.0] * count),
        feature_scale=tuple([1.0] * count),
        coefficients=tuple([1.0] + [0.0] * (count - 1)),
        intercept=0.0,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        geometry_threshold_px=5.0,
        verification_threshold=0.7,
        promotion_probability_min=0.7,
        promotion_margin_min=0.1,
        measurement_checkpoint_sha256="0123456789abcdef",
    )
    restored = CandidateGeometryVerifier.from_dict(model.to_dict())
    probabilities = restored.predict(np.asarray([[0.0] * count, [1.0] + [0.0] * (count - 1)]))
    assert np.allclose(probabilities, [0.5, 1.0 / (1.0 + np.exp(-1.0))])


def test_candidate_selection_metrics_counts_rescue_and_harm() -> None:
    examples = [
        {"source_query_row": 1, "candidate_measurement_rank": 1, "label_2px": False, "label_5px": False},
        {"source_query_row": 1, "candidate_measurement_rank": 2, "label_2px": True, "label_5px": True},
        {"source_query_row": 2, "candidate_measurement_rank": 1, "label_2px": True, "label_5px": True},
        {"source_query_row": 2, "candidate_measurement_rank": 2, "label_2px": False, "label_5px": False},
    ]
    metrics = candidate_selection_metrics(examples, np.asarray([0.1, 0.9, 0.8, 0.2]))
    block = metrics["geometry_correct_5px"]
    assert block["rescued_count"] == 1
    assert block["harmed_count"] == 0
    assert block["net_correct_change"] == 1
