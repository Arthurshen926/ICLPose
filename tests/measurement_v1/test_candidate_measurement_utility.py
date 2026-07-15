from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.candidate_measurement_utility import (
    UTILITY_FEATURE_NAMES,
    CandidateMeasurementUtilityGate,
    aggregate_candidate_measurement_views,
)
from feature_extract.vfm.localization.candidate_pose_evidence import (
    measurement_utility_action_gate,
)


def test_measurement_utility_model_round_trip() -> None:
    count = len(UTILITY_FEATURE_NAMES)
    model = CandidateMeasurementUtilityGate(
        feature_names=UTILITY_FEATURE_NAMES,
        feature_mean=tuple([0.0] * count),
        feature_scale=tuple([1.0] * count),
        coefficients=tuple([1.0] + [0.0] * (count - 1)),
        intercept=0.0,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        update_threshold=0.8,
        minimum_baseline_residual_px=1.0,
        minimum_improvement_px=0.1,
        measurement_checkpoint_sha256="0123456789abcdef",
        candidate_evidence_sha256="1123456789abcdef",
        candidate_inference_evidence_sha256="2123456789abcdef",
        coordinate_space_id="3123456789abcdef",
        offsets_sha256="4123456789abcdef",
    )

    restored = CandidateMeasurementUtilityGate.from_dict(model.to_dict())
    probabilities = restored.predict(
        np.asarray([[0.0] * count, [1.0] + [0.0] * (count - 1)])
    )

    assert np.allclose(probabilities, [0.5, 1.0 / (1.0 + np.exp(-1.0))])


def test_measurement_utility_aggregates_multiview_distribution_without_identity() -> None:
    offsets = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    probability = np.asarray([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    rows = [
        {
            "track_length": "8",
            "support_reprojection_error": "0.5",
            "support_frame_gap": "2",
        },
        {
            "track_length": "8",
            "support_reprojection_error": "1.0",
            "support_frame_gap": "4",
        },
    ]

    features, proposed_offset = aggregate_candidate_measurement_views(
        offsets_xy=offsets,
        local_log_probabilities=np.log(probability),
        dustbin_probabilities=np.asarray([0.1, 0.5]),
        support_view_probabilities=np.asarray([0.75, 0.25]),
        likelihood_entropy=-np.sum(probability * np.log(probability), axis=1),
        likelihood_covariance_trace_px2=np.asarray([0.4, 0.5]),
        observable_rows=rows,
    )

    accepted = np.asarray([0.75 * 0.9, 0.25 * 0.5])
    mixture = np.sum(accepted[:, None] * probability, axis=0)
    mixture /= np.sum(mixture)
    assert np.allclose(proposed_offset, mixture @ offsets)
    assert features.shape == (len(UTILITY_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert not any(
        "identity" in name or "retrieval" in name or "rank" in name
        for name in UTILITY_FEATURE_NAMES
    )


def test_measurement_utility_rejects_invalid_support_mass() -> None:
    with np.testing.assert_raises(ValueError):
        aggregate_candidate_measurement_views(
            offsets_xy=np.asarray([[0.0, 0.0]]),
            local_log_probabilities=np.asarray([[0.0]]),
            dustbin_probabilities=np.asarray([0.5]),
            support_view_probabilities=np.asarray([1.1]),
            likelihood_entropy=np.asarray([0.0]),
            likelihood_covariance_trace_px2=np.asarray([0.0]),
            observable_rows=[
                {
                    "track_length": "2",
                    "support_reprojection_error": "0",
                    "support_frame_gap": "0",
                }
            ],
        )


def test_measurement_utility_is_a_thresholded_coordinate_action_only() -> None:
    assert measurement_utility_action_gate(0.99, 0.8, 0.0) == 1.0
    assert measurement_utility_action_gate(0.79, 0.8, 1.0) == 0.0
    assert measurement_utility_action_gate(0.80, 0.8, 1.0) == 1.0
    assert measurement_utility_action_gate(0.79, 0.8, 0.25) == 0.75
