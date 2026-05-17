from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_retrieval.tools.sweep_correspondence_pnp_filters import (
    model_from_reliability_json_payload,
    select_top_fraction_mask,
    summarize_pose_errors,
)


def test_select_top_fraction_mask_keeps_high_scores_with_minimum_count():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7], dtype=np.float32)

    mask = select_top_fraction_mask(scores, keep_frac=0.4, min_points=1)

    assert mask.tolist() == [False, True, False, True, False]

    min_mask = select_top_fraction_mask(scores, keep_frac=0.2, min_points=3)

    assert min_mask.tolist() == [False, True, False, True, True]


def test_summarize_pose_errors_counts_failures_as_unsuccessful_thresholds():
    summary = summarize_pose_errors(
        rot_deg=np.array([0.5, 2.0], dtype=np.float32),
        trans_m=np.array([0.04, 0.20], dtype=np.float32),
        total=3,
    )

    assert summary["num_success"] == 2
    assert summary["num_total"] == 3
    assert summary["trans_median"] == 0.12000000104308128
    assert summary["joint_1deg_50mm"] == 1.0 / 3.0
    assert summary["joint_5deg_250mm"] == 2.0 / 3.0


def test_model_from_reliability_json_payload_restores_probability_model():
    payload = {
        "model": {
            "feature_names": ["confidence"],
            "weights": [2.0],
            "bias": 0.0,
            "feature_mean": [0.5],
            "feature_std": [0.5],
        }
    }

    model, names = model_from_reliability_json_payload(payload)
    scores = model.predict_proba(np.array([[0.0], [1.0]], dtype=np.float32))

    assert names == ["confidence"]
    assert scores[1] > scores[0]
