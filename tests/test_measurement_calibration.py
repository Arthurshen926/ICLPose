from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.fit_measurement_calibration import parse_args
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.measurement_calibration import (
    MeasurementCalibrationSample,
    fit_geometry_probability_model,
    calibration_rows_from_matches,
    fit_confidence_temperature_bias,
    fit_uncertainty_scale_floor,
)


def test_calibration_rows_project_gt_landmark_and_compute_residual() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=80, params=(10.0, 10.0, 50.0, 40.0))
    pose = np.eye(4, dtype=np.float64)
    rows = [
        {
            "query_id": "q.png",
            "x": "52.0",
            "y": "41.0",
            "xyz": "[0.0, 0.0, 10.0]",
            "patch_offset_confidence": "0.8",
            "measurement_sigma_px": "2.0",
            "local_consistency_score": "0.75",
            "patch_offset_consistency_before_px": "3.0",
        }
    ]

    samples = calibration_rows_from_matches(rows, cameras_by_query={"q.png": camera}, pose_w2c_by_query={"q.png": pose})

    assert len(samples) == 1
    assert samples[0].gt_x == pytest.approx(50.0)
    assert samples[0].gt_y == pytest.approx(40.0)
    assert samples[0].residual_px == pytest.approx(np.sqrt(5.0))
    assert samples[0].confidence == pytest.approx(0.8)
    assert samples[0].uncertainty_px == pytest.approx(2.0)
    assert samples[0].local_consistency_score == pytest.approx(0.75)
    assert samples[0].coarse_pnp_residual_px == pytest.approx(3.0)


def test_fit_confidence_temperature_bias_improves_binary_cross_entropy() -> None:
    samples = [
        MeasurementCalibrationSample("a", 0.0, 0.0, 0.0, 0.0, 0.95, 2.0, 12.0),
        MeasurementCalibrationSample("b", 0.0, 0.0, 0.0, 0.0, 0.90, 2.0, 10.0),
        MeasurementCalibrationSample("c", 0.0, 0.0, 0.0, 0.0, 0.60, 2.0, 0.5),
        MeasurementCalibrationSample("d", 0.0, 0.0, 0.0, 0.0, 0.55, 2.0, 0.8),
    ]

    fit = fit_confidence_temperature_bias(samples, inlier_threshold_px=2.0, temperatures=(1.0, 2.0, 4.0), biases=(-4.0, 0.0, 4.0))

    assert fit["sample_count"] == 4
    assert fit["bce_after"] < fit["bce_before"]
    assert (fit["confidence_temperature"], fit["confidence_bias"]) != (1.0, 0.0)


def test_fit_geometry_probability_model_learns_residual_labels() -> None:
    samples = [
        MeasurementCalibrationSample("g0", 0.0, 0.0, 0.0, 0.0, 0.85, 1.0, 0.8, similarity=0.90, query_heatmap_score=0.9, observation_count=12, landmark_reprojection_error=0.1, local_consistency_score=0.9, coarse_pnp_residual_px=1.0),
        MeasurementCalibrationSample("g1", 0.0, 0.0, 0.0, 0.0, 0.80, 1.5, 1.5, similarity=0.86, query_heatmap_score=0.8, observation_count=10, landmark_reprojection_error=0.2, local_consistency_score=0.8, coarse_pnp_residual_px=2.0),
        MeasurementCalibrationSample("b0", 0.0, 0.0, 0.0, 0.0, 0.75, 8.0, 18.0, similarity=0.40, query_heatmap_score=0.2, observation_count=2, landmark_reprojection_error=3.0, local_consistency_score=0.1, coarse_pnp_residual_px=20.0),
        MeasurementCalibrationSample("b1", 0.0, 0.0, 0.0, 0.0, 0.70, 10.0, 22.0, similarity=0.35, query_heatmap_score=0.1, observation_count=1, landmark_reprojection_error=4.0, local_consistency_score=0.05, coarse_pnp_residual_px=30.0),
    ]

    fit = fit_geometry_probability_model(samples, inlier_threshold_px=4.0, max_iter=400)
    model = fit["model"]
    good_probability = model.predict_sample(samples[0])
    bad_probability = model.predict_sample(samples[-1])

    assert fit["sample_count"] == 4
    assert fit["bce_after"] < fit["bce_before"]
    assert good_probability > 0.75
    assert bad_probability < 0.25


def test_fit_uncertainty_scale_floor_matches_residual_median() -> None:
    samples = [
        MeasurementCalibrationSample("a", 0.0, 0.0, 0.0, 0.0, 0.9, 2.0, 4.0),
        MeasurementCalibrationSample("b", 0.0, 0.0, 0.0, 0.0, 0.9, 3.0, 6.0),
        MeasurementCalibrationSample("c", 0.0, 0.0, 0.0, 0.0, 0.9, 4.0, 8.0),
        MeasurementCalibrationSample("bad", 0.0, 0.0, 0.0, 0.0, 0.1, 2.0, 500.0),
    ]

    fit = fit_uncertainty_scale_floor(samples, floors=(0.0, 1.0), max_residual_px=20.0)

    assert fit["sample_count"] == 3
    assert fit["excluded_sample_count"] == 1
    assert fit["uncertainty_scale"] == pytest.approx(2.0)
    assert fit["uncertainty_floor_px"] == pytest.approx(0.0)


def test_fit_measurement_calibration_cli_parse_args(tmp_path) -> None:
    args = parse_args(
        [
            "--matches_csv",
            "matches.csv",
            "--colmap_model_dir",
            "sparse/0",
            "--image_root",
            "images",
            "--query_pose_file",
            "poses.txt",
            "--output_json",
            str(tmp_path / "calibration.json"),
            "--output_samples_csv",
            str(tmp_path / "samples.csv"),
            "--inlier_threshold_px",
            "3.0",
            "--uncertainty_max_residual_px",
            "12.0",
            "--geometry_inlier_threshold_px",
            "4.0",
            "--source_measurement_confidence_temperature",
            "1.0",
        ]
    )

    assert args.matches_csv == "matches.csv"
    assert args.inlier_threshold_px == 3.0
    assert args.uncertainty_max_residual_px == 12.0
    assert args.geometry_inlier_threshold_px == 4.0
    assert args.source_measurement_confidence_temperature == 1.0
