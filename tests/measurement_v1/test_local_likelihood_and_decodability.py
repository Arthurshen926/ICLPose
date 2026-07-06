from __future__ import annotations

import numpy as np

from feature_extract.vfm.measurement_v1.decodability import decodability_metrics
from feature_extract.vfm.measurement_v1.local_likelihood import compute_local_likelihood


def _one_hot_feature_map(width: int = 9, height: int = 9) -> np.ndarray:
    fmap = np.zeros((2, height, width), dtype=np.float32)
    fmap[0, :, :] = 1.0
    fmap[1, 4, 6] = 10.0
    return fmap


def test_local_likelihood_recovers_query_side_peak_and_covariance() -> None:
    result = compute_local_likelihood(
        anchor_descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
        query_feature_map=_one_hot_feature_map(),
        center_xy_px=np.asarray([4.0, 4.0], dtype=np.float64),
        image_width=9,
        image_height=9,
        search_radius_px=3.0,
        step_px=1.0,
        temperature=0.25,
    )

    assert np.allclose(result.mode_xy_px, [6.0, 4.0], atol=1e-6)
    assert np.linalg.norm(result.mean_xy_px - np.asarray([6.0, 4.0])) < 0.1
    assert result.cov_query_2x2.shape == (2, 2)
    assert result.dustbin_probability == 0.0


def test_local_likelihood_marks_window_out_target_as_dustbin() -> None:
    result = compute_local_likelihood(
        anchor_descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
        query_feature_map=_one_hot_feature_map(),
        center_xy_px=np.asarray([0.0, 0.0], dtype=np.float64),
        image_width=9,
        image_height=9,
        search_radius_px=1.0,
        step_px=1.0,
        temperature=1.0,
        gt_xy_px=np.asarray([6.0, 4.0], dtype=np.float64),
    )

    assert result.gt_in_window is False
    assert result.target_is_dustbin is True


def test_decodability_metrics_report_rank_recall_nll_and_entropy() -> None:
    result = compute_local_likelihood(
        anchor_descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
        query_feature_map=_one_hot_feature_map(),
        center_xy_px=np.asarray([4.0, 4.0], dtype=np.float64),
        image_width=9,
        image_height=9,
        search_radius_px=3.0,
        step_px=1.0,
        temperature=0.25,
    )

    metrics = decodability_metrics(result, gt_xy_px=np.asarray([6.0, 4.0], dtype=np.float64))

    assert metrics["gt_rank"] == 1
    assert metrics["recall_1px"] == 1.0
    assert metrics["epe_px"] < 0.1
    assert metrics["nll"] < 0.01
    assert metrics["entropy"] >= 0.0
