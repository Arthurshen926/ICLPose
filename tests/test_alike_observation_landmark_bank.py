from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_alike_observation_landmark_bank import (
    aggregate_alike_track_descriptors,
)


def test_alike_track_aggregation_normalizes_and_preserves_track_order() -> None:
    tracks, features, counts, variances = aggregate_alike_track_descriptors(
        np.asarray([2, 1, 1], dtype=np.int64),
        np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.float32),
        np.asarray([1.0, 0.1, 1.0], dtype=np.float32),
        method="normalized_mean",
    )
    np.testing.assert_array_equal(tracks, [1, 2])
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-6)
    assert variances[1] == 0.0


def test_detector_weighted_aggregation_moves_toward_high_score_view() -> None:
    args = (
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        np.asarray([1.0, 0.1], dtype=np.float32),
    )
    _tracks, mean, _counts, _variances = aggregate_alike_track_descriptors(
        *args, method="normalized_mean"
    )
    _tracks, weighted, _counts, _variances = aggregate_alike_track_descriptors(
        *args, method="detector_weighted_mean"
    )
    assert weighted[0, 0] > mean[0, 0]
