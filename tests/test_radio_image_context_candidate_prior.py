import numpy as np

from feature_extract.tools.vfm.probe_radio_image_context_candidate_prior import (
    _identity_metrics,
    _posterior,
    _support_view_score,
)


def test_context_posterior_preserves_explicit_null_probability_mass() -> None:
    probabilities, null = _posterior(
        np.asarray([[0.2, 0.3, 0.0]], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.asarray([[1.0, 2.0, -1.0]], dtype=np.float32),
        np.asarray([[True, True, False]]),
        0.5,
    )

    np.testing.assert_allclose(probabilities[0, 2], 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1) + null, 1.0, atol=1e-6)


def test_identity_metrics_respects_configured_positive_threshold() -> None:
    probabilities = np.asarray([[0.8, 0.2]], dtype=np.float32)
    residuals = np.asarray([[3.0, 1.0]], dtype=np.float32)
    valid = np.asarray([[True, True]])

    strict = _identity_metrics(
        probabilities,
        residuals,
        valid,
        positive_threshold_px=2.0,
    )
    loose = _identity_metrics(
        probabilities,
        residuals,
        valid,
        positive_threshold_px=4.0,
    )

    assert strict["top1_positive_rate_all"] == 0.0
    assert loose["top1_positive_rate_all"] == 1.0


def test_top2_support_aggregation_limits_track_length_bias() -> None:
    scores = np.asarray([0.1, 0.4, 0.9], dtype=np.float32)

    assert np.isclose(_support_view_score(scores, "top1"), 0.9)
    assert np.isclose(_support_view_score(scores, "top2_mean"), 0.65)
