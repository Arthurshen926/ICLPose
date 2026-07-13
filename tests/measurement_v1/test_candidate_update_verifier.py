from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.candidate_update_verifier import (
    CANDIDATE_UPDATE_FEATURE_NAMES,
    CandidateUpdateVerifier,
)


def test_candidate_update_model_round_trip_and_probability() -> None:
    count = len(CANDIDATE_UPDATE_FEATURE_NAMES)
    model = CandidateUpdateVerifier(
        feature_names=CANDIDATE_UPDATE_FEATURE_NAMES,
        feature_mean=tuple([0.0] * count),
        feature_scale=tuple([1.0] * count),
        coefficients=tuple([1.0] + [0.0] * (count - 1)),
        intercept=0.0,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        update_threshold=0.8,
        minimum_baseline_residual_px=1.0,
        maximum_baseline_residual_px=5.0,
        minimum_improvement_px=0.1,
        measurement_checkpoint_sha256="0123456789abcdef",
        candidate_geometry_verifier_sha256="fedcba9876543210",
    )

    restored = CandidateUpdateVerifier.from_dict(model.to_dict())
    probabilities = restored.predict(
        np.asarray([[0.0] * count, [1.0] + [0.0] * (count - 1)])
    )

    assert np.allclose(
        probabilities, [0.5, 1.0 / (1.0 + np.exp(-1.0))]
    )
