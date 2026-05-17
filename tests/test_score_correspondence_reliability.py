from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_retrieval.tools.score_correspondence_reliability import (
    average_precision_from_scores,
    brier_score,
    build_reliability_features,
    train_logistic_reliability,
)


def test_build_reliability_features_uses_confidence_and_query_xy_without_map_leakage():
    payload = {
        "query_xy": np.array([[0.0, 0.0], [9.0, 4.0]], dtype=np.float32),
        "map_xy": np.array([[8.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        "confidence": np.array([0.2, 0.8], dtype=np.float32),
        "pnp_inlier_mask": np.array([False, True]),
        "query_hw": np.array([5, 10], dtype=np.int32),
        "map_hw": np.array([5, 10], dtype=np.int32),
    }

    features, labels, names = build_reliability_features(payload, feature_set="confidence_query_xy")

    assert names == ["confidence", "query_x_norm", "query_y_norm"]
    assert labels.tolist() == [False, True]
    assert np.allclose(features, [[0.2, 0.0, 0.0], [0.8, 1.0, 1.0]])


def test_average_precision_and_brier_score_are_well_defined():
    labels = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.3, 0.1], dtype=np.float32)

    assert average_precision_from_scores(labels, scores) == 0.8333333333333333
    assert brier_score(labels, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)) == 0.0


def test_train_logistic_reliability_learns_separable_confidence_signal():
    features = np.array([[0.05], [0.1], [0.9], [0.95]], dtype=np.float32)
    labels = np.array([False, False, True, True])

    model = train_logistic_reliability(features, labels, epochs=300, lr=0.5, seed=7)
    pred = model.predict_proba(features)

    assert pred[labels].mean() > 0.8
    assert pred[~labels].mean() < 0.2
