from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.render_pose_diagnostics import (
    ideal_depth_correspondences,
    match_validity_calibration_stats,
    run_oracle_correspondence_pnp_diagnostics,
    run_pnp_solver_ablation,
    render_depth_roundtrip_stats,
    synthetic_render_lock_stats,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))


def _pose_w2c_from_center(center: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64).reshape(3)
    return pose


def test_render_depth_roundtrip_reprojects_back_to_input_pixels() -> None:
    camera = _camera()
    pose_w2c = _pose_w2c_from_center(np.asarray([0.3, -0.2, 0.1], dtype=np.float64))
    xy = np.asarray(
        [
            [320.0, 240.0],
            [360.0, 240.0],
            [300.0, 260.0],
            [410.0, 180.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    stats = render_depth_roundtrip_stats(xy, depth, camera, pose_w2c)

    assert stats["count"] == 4
    assert stats["valid_fraction"] == 1.0
    assert stats["median_roundtrip_error_px"] < 1e-6
    assert stats["p95_roundtrip_error_px"] < 1e-6


def test_synthetic_render_lock_solves_back_to_render_pose() -> None:
    camera = _camera()
    render_pose_w2c = _pose_w2c_from_center(np.asarray([0.25, 0.0, 0.0], dtype=np.float64))
    xy = np.asarray(
        [
            [260.0, 190.0],
            [380.0, 190.0],
            [260.0, 290.0],
            [380.0, 290.0],
            [320.0, 160.0],
            [420.0, 260.0],
            [220.0, 260.0],
            [340.0, 330.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5], dtype=np.float64)

    stats = synthetic_render_lock_stats(xy, depth, camera, render_pose_w2c)

    assert stats["pnp_success"] is True
    assert stats["pnp_render_translation_delta_m"] < 1e-4
    assert stats["pnp_render_rotation_delta_deg"] < 1e-4


def test_ideal_depth_correspondences_support_solver_ablation() -> None:
    camera = _camera()
    pose_w2c = _pose_w2c_from_center(np.asarray([0.1, -0.05, 0.2], dtype=np.float64))
    xy = np.asarray(
        [
            [220.0, 160.0],
            [420.0, 160.0],
            [220.0, 320.0],
            [420.0, 320.0],
            [300.0, 210.0],
            [360.0, 260.0],
            [260.0, 285.0],
            [390.0, 220.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([4.0, 4.8, 5.5, 6.2, 7.0, 7.8, 8.6, 9.4], dtype=np.float64)

    matches, match_summary = ideal_depth_correspondences(
        xy,
        depth,
        camera,
        pose_w2c,
        source="unit_test_ideal_depth",
    )
    ablation = run_pnp_solver_ablation(
        matches,
        camera,
        gt_pose_w2c=pose_w2c,
        solvers=("plain", "ransac", "weighted", "oracle_uncertainty"),
        reprojection_error_px=1.0,
    )

    assert match_summary["valid_match_count"] == 8
    assert set(ablation) == {"plain", "ransac", "weighted", "oracle_uncertainty"}
    for row in ablation.values():
        assert row["success"] is True
        assert row["translation_error_m"] < 1e-4
        assert row["rotation_error_deg"] < 1e-4
        assert row["residual_median_px"] < 1e-4


def test_match_validity_calibration_stats_reports_threshold_quality() -> None:
    camera = _camera()
    pose_w2c = np.eye(4, dtype=np.float64)
    xy = np.asarray(
        [
            [300.0, 220.0],
            [340.0, 220.0],
            [300.0, 260.0],
            [340.0, 260.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    matches, _summary = ideal_depth_correspondences(xy, depth, camera, pose_w2c)
    scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    noisy_matches = list(matches)
    # Deliberately corrupt the last two observations so confidence calibration
    # has both positives and negatives at the 5px threshold.
    noisy_matches[2] = noisy_matches[2].__class__(
        **{**noisy_matches[2].__dict__, "xy": noisy_matches[2].xy + np.asarray([20.0, 0.0])}
    )
    noisy_matches[3] = noisy_matches[3].__class__(
        **{**noisy_matches[3].__dict__, "xy": noisy_matches[3].xy + np.asarray([30.0, 0.0])}
    )

    stats = match_validity_calibration_stats(
        noisy_matches,
        pose_w2c,
        camera,
        scores=scores,
        thresholds_px=(5.0, 10.0),
        bin_count=5,
    )

    assert stats["validity_5px_positive_rate"] == 0.5
    assert stats["validity_5px_brier"] < 0.05
    assert stats["validity_5px_ece"] < 0.25
    assert "validity_10px_positive_rate" in stats


def test_oracle_correspondence_pnp_diagnostics_separate_fine_and_match_limits() -> None:
    camera = _camera()
    pose_w2c = np.eye(4, dtype=np.float64)
    xy = np.asarray(
        [
            [300.0, 220.0],
            [340.0, 220.0],
            [300.0, 260.0],
            [340.0, 260.0],
            [280.0, 240.0],
            [360.0, 240.0],
        ],
        dtype=np.float64,
    )
    depth = np.asarray([5.0, 6.0, 7.0, 8.0, 6.5, 7.5], dtype=np.float64)
    matches, _summary = ideal_depth_correspondences(xy, depth, camera, pose_w2c)
    noisy = list(matches)
    for idx in (3, 4, 5):
        noisy[idx] = noisy[idx].__class__(
            **{**noisy[idx].__dict__, "xy": noisy[idx].xy + np.asarray([35.0, 0.0])}
        )

    diagnostics = run_oracle_correspondence_pnp_diagnostics(
        noisy,
        camera,
        gt_pose_w2c=pose_w2c,
        thresholds_px=(5.0,),
        solvers=("plain",),
    )

    assert diagnostics["oracle_match_5px_plain"]["match_count"] == 3
    assert diagnostics["oracle_fine_all_plain"]["match_count"] == 6
    assert diagnostics["oracle_fine_all_plain"]["translation_error_m"] < 1e-4
    assert diagnostics["oracle_fine_all_plain"]["rotation_error_deg"] < 1e-4
