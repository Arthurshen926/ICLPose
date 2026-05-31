from __future__ import annotations

import numpy as np

from feature_extract.vfm.patch_offset_feasibility import (
    calibration_summary,
    offset_error_bucket_summary,
    pose_metric_summary,
)


def test_calibration_summary_reports_error_monotonicity_by_confidence() -> None:
    summary = calibration_summary(
        offset_errors_px=np.asarray([10.0, 8.0, 2.0, 1.0], dtype=np.float32),
        confidences=np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        good_threshold_px=4.0,
        bin_count=2,
    )

    assert summary["count"] == 4
    assert summary["precision_at_top10_percent"] == 1.0
    assert summary["bins"][0]["mean_error_px"] > summary["bins"][1]["mean_error_px"]
    assert summary["ece"] < 0.5


def test_offset_error_bucket_summary_splits_positive_inlier_and_margin_groups() -> None:
    rows = [
        {"patch_positive_label": True, "pnp_inlier": True, "similarity_margin": 0.4, "positive_count": 1},
        {"patch_positive_label": False, "pnp_inlier": True, "similarity_margin": 0.3, "positive_count": 3},
        {"patch_positive_label": True, "pnp_inlier": False, "similarity_margin": 0.05, "positive_count": 1},
        {"patch_positive_label": False, "pnp_inlier": False, "similarity_margin": 0.01, "positive_count": 0},
    ]
    errors = np.asarray([1.0, 5.0, 2.0, 9.0], dtype=np.float32)
    confidences = np.asarray([0.9, 0.5, 0.8, 0.1], dtype=np.float32)

    summary = offset_error_bucket_summary(rows, errors, confidences)

    assert summary["patch_positive"]["count"] == 2
    assert summary["patch_negative"]["median_error_px"] == 7.0
    assert summary["first_pass_inlier"]["count"] == 2
    assert summary["single_positive_patch"]["mean_error_px"] == 1.5
    assert summary["high_margin"]["mean_confidence"] > summary["low_margin"]["mean_confidence"]


def test_pose_metric_summary_tracks_success_rates_and_medians() -> None:
    rows = [
        {"success": True, "translation_error_m": 0.10, "rotation_error_deg": 2.0, "inlier_count": 10},
        {"success": True, "translation_error_m": 0.40, "rotation_error_deg": 4.0, "inlier_count": 8},
        {"success": False, "translation_error_m": None, "rotation_error_deg": None, "inlier_count": 0},
    ]

    summary = pose_metric_summary(rows)

    assert summary["solve_rate"] == 2 / 3
    assert summary["success_25cm_10deg"] == 1 / 3
    assert summary["success_50cm_10deg"] == 2 / 3
    assert summary["median_translation_error_m"] == 0.25
    assert summary["mean_inlier_count"] == 6.0
