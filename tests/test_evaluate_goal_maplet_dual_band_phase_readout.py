import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_dual_band_phase_readout import (
    _apply_pairwise,
    _fit_pairwise,
    _policy_metrics,
)


def _row(offset=0.0):
    return {
        "image_id": "seq1/frame.png",
        "trajectory_id": "seq1",
        "feature": np.asarray([
            [0.9 + offset, 0.8, 0.7, 0.7, 0.65, 0.65, 0.6, 0.6, 0.8],
            [0.5 + offset, 0.7, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.9],
            [0.2 + offset, 0.1, -0.1, -0.1, -0.15, -0.15, -0.2, -0.2, 0.4],
        ]),
        "translation_m": np.asarray([0.2, 0.8, 6.0]),
        "rotation_deg": np.asarray([2.0, 3.0, 40.0]),
    }


def test_pairwise_dual_band_ranker_prefers_usable_target():
    rows = [_row(), _row(0.01)]
    scaler, model, report = _fit_pairwise(rows)
    metrics, scores = _apply_pairwise(rows, scaler, model)
    assert report["usable_query_count"] == 2
    assert metrics["strict_0.5m_5deg"] == 1.0
    assert all(value[0] > value[1] > value[2] for value in scores)


def test_policy_report_keeps_fixed_dual_band_coordinate_free():
    metrics = _policy_metrics([_row()])
    assert metrics["mapper_cosine"]["strict_0.5m_5deg"] == 1.0
    assert metrics["dual_band_fixed"]["selected_original_indices"] == [0]
