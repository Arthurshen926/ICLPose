from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.dense_depth_measurement_fusion import (
    DENSE_DEPTH_FUSION_FIELDNAMES,
    augment_rows_with_query_gt_projection,
    dense_depth_matches_from_rows,
    dense_depth_measurement_summary,
    dense_depth_pose_ablation_from_rows,
    dense_depth_rows_from_rows,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))


def test_dense_depth_fusion_backprojects_render_depth_and_only_refines_query_pixel() -> None:
    camera = _camera()
    render_pose = np.eye(4, dtype=np.float64)
    rows = [
        {
            "query_id": "q0.png",
            "match_index": "7",
            "query_x": "322.0",
            "query_y": "239.0",
            "render_x": "320.0",
            "render_y": "240.0",
            "render_depth": "5.0",
            "query_refined_x": "321.25",
            "query_refined_y": "240.50",
            "measurement_cov_xx": "0.25",
            "measurement_cov_xy": "0.0",
            "measurement_cov_yy": "0.25",
            "measurement_valid_prob": "0.9",
            "similarity": "0.75",
            "local_cost_entropy": "0.2",
        }
    ]

    matches, summary = dense_depth_matches_from_rows(rows, camera=camera, render_pose_w2c=render_pose)
    fusion_rows = dense_depth_rows_from_rows(rows, camera=camera, render_pose_w2c=render_pose)

    assert summary["input_count"] == 1
    assert summary["valid_match_count"] == 1
    assert summary["depth_valid_count"] == 1
    assert summary["depth_valid_fraction"] == 1.0
    assert summary["valid_match_fraction"] == 1.0
    assert len(matches) == 1
    assert np.allclose(matches[0].xy, np.asarray([321.25, 240.50]))
    assert np.allclose(project_world_to_image(matches[0].xyz.reshape(1, 3), render_pose, camera)[0], np.asarray([320.0, 240.0]))
    assert matches[0].render_depth == 5.0
    assert matches[0].measurement_sigma_px == 0.5
    assert matches[0].pnp_soft_score == 0.9
    assert len(fusion_rows) == 1
    assert set(DENSE_DEPTH_FUSION_FIELDNAMES).issubset(fusion_rows[0].keys())
    assert fusion_rows[0]["query_center_x"] == 322.0
    assert fusion_rows[0]["query_refined_x"] == 321.25
    assert fusion_rows[0]["measurement_dx"] == -0.75
    assert fusion_rows[0]["render_x"] == 320.0
    assert fusion_rows[0]["render_depth"] == 5.0


def test_dense_depth_fusion_filters_invalid_depth_without_dropping_denominator() -> None:
    camera = _camera()
    render_pose = np.eye(4, dtype=np.float64)
    rows = [
        {"query_id": "q0.png", "match_index": "0", "query_x": "1", "query_y": "2", "render_x": "3", "render_y": "4", "render_depth": "-1"},
        {"query_id": "q0.png", "match_index": "1", "query_x": "5", "query_y": "6", "render_x": "7", "render_y": "8", "render_depth": ""},
    ]

    matches, summary = dense_depth_matches_from_rows(rows, camera=camera, render_pose_w2c=render_pose)

    assert matches == []
    assert summary["input_count"] == 2
    assert summary["valid_match_count"] == 0
    assert summary["invalid_depth_count"] == 2
    assert summary["depth_valid_count"] == 0
    assert summary["depth_valid_fraction"] == 0.0
    assert summary["valid_match_fraction"] == 0.0


def test_dense_depth_fusion_can_use_existing_dense_depth_world_xyz_without_render_pose() -> None:
    camera = _camera()
    row = {
        "query_id": "q0.png",
        "match_index": "3",
        "query_center_x": "10.0",
        "query_center_y": "20.0",
        "query_refined_x": "11.0",
        "query_refined_y": "20.0",
        "render_x": "100.0",
        "render_y": "120.0",
        "render_depth": "4.0",
        "world_x": "0.1",
        "world_y": "0.2",
        "world_z": "4.0",
    }

    matches, summary = dense_depth_matches_from_rows([row], camera=camera, render_pose_w2c=None)
    dense_rows = dense_depth_rows_from_rows([row], camera=camera, render_pose_w2c=None)

    assert summary["valid_match_count"] == 1
    assert summary["missing_geometry_count"] == 0
    assert summary["valid_match_fraction"] == 1.0
    assert np.allclose(matches[0].xyz, np.asarray([0.1, 0.2, 4.0]))
    assert dense_rows[0]["world_x"] == 0.1


def test_augment_rows_with_query_gt_projection_projects_existing_world_xyz() -> None:
    camera = _camera()
    pose = np.eye(4, dtype=np.float64)
    rows = [
        {
            "query_id": "q0.png",
            "world_x": "0.0",
            "world_y": "0.0",
            "world_z": "5.0",
        },
        {
            "query_id": "missing.png",
            "world_x": "0.0",
            "world_y": "0.0",
            "world_z": "5.0",
        },
    ]

    augmented, summary = augment_rows_with_query_gt_projection(
        rows,
        query_pose_w2c_by_id={"q0.png": pose},
        camera=camera,
    )

    assert summary["projected_query_gt_count"] == 1
    assert summary["missing_query_pose_count"] == 1
    assert augmented[0]["query_gt_x"] == 320.0
    assert augmented[0]["query_gt_y"] == 240.0
    assert "query_gt_x" not in augmented[1]


def test_dense_depth_pose_ablation_compares_center_measurement_and_oracle() -> None:
    camera = _camera()
    render_pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [
            [-0.8, -0.5, 4.5],
            [0.8, -0.5, 5.0],
            [-0.8, 0.5, 5.5],
            [0.8, 0.5, 6.0],
            [-0.2, -0.8, 4.8],
            [0.4, 0.7, 5.8],
            [1.0, 0.0, 6.4],
            [-1.0, 0.1, 5.2],
        ],
        dtype=np.float64,
    )
    xy = project_world_to_image(points, render_pose, camera)
    rows = []
    for index, ((x, y), depth) in enumerate(zip(xy, points[:, 2])):
        rows.append(
            {
                "query_id": "q0.png",
                "match_index": str(index),
                "render_x": str(x),
                "render_y": str(y),
                "render_depth": str(depth),
                "query_center_x": str(x + 3.0),
                "query_center_y": str(y),
                "query_refined_x": str(x),
                "query_refined_y": str(y),
                "query_gt_x": str(x),
                "query_gt_y": str(y),
                "measurement_valid_prob": "1.0",
            }
        )

    report = dense_depth_pose_ablation_from_rows(
        rows,
        camera=camera,
        render_pose_w2c=render_pose,
        gt_pose_w2c=render_pose,
        solvers=("plain",),
    )

    assert set(report["variants"]) == {"center", "measurement", "oracle"}
    center_error = report["variants"]["center"]["plain"]["translation_error_m"]
    measurement_error = report["variants"]["measurement"]["plain"]["translation_error_m"]
    oracle_error = report["variants"]["oracle"]["plain"]["translation_error_m"]
    assert center_error > 1e-3
    assert measurement_error < center_error
    assert oracle_error < 1e-4
    assert report["measurement_summary"]["measurement_epe_median_px"] == 0.0
    assert report["measurement_summary"]["measurement_improve_ratio"] == 1.0


def test_dense_depth_measurement_summary_reports_center_baseline_improvement() -> None:
    rows = [
        {
            "query_center_x": 12.0,
            "query_center_y": 10.0,
            "query_refined_x": 10.5,
            "query_refined_y": 10.0,
            "query_gt_x": 10.0,
            "query_gt_y": 10.0,
            "depth_valid": True,
            "measurement_valid_prob": 0.75,
            "local_cost_entropy": 0.2,
        },
        {
            "query_center_x": 0.0,
            "query_center_y": 0.0,
            "query_refined_x": 2.0,
            "query_refined_y": 0.0,
            "query_gt_x": 1.0,
            "query_gt_y": 0.0,
            "depth_valid": "False",
            "measurement_valid_prob": 0.25,
        },
    ]

    summary = dense_depth_measurement_summary(rows)

    assert summary["row_count"] == 2
    assert summary["depth_valid_rate"] == 0.5
    assert summary["center_epe_median_px"] == 1.5
    assert summary["measurement_epe_median_px"] == 0.75
    assert summary["measurement_improve_ratio"] == 0.5
    assert summary["measurement_valid_prob_mean"] == 0.5
