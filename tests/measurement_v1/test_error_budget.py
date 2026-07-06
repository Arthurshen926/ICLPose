from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.error_budget import ErrorBudgetConfig, run_error_budget_for_rows
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


def test_error_budget_reports_noise_and_quantization_pose_headroom() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=160, height=120, params=(100.0, 100.0, 80.0, 60.0))
    pose = np.eye(4, dtype=np.float64)
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
    xy = project_world_to_image(points, pose, camera)
    rows = [
        {
            "query_id": "q0",
            "world_x": float(x),
            "world_y": float(y),
            "world_z": float(z),
            "query_x": float(u),
            "query_y": float(v),
            "cell_center_x": float(round(u / 8.0) * 8.0),
            "cell_center_y": float(round(v / 8.0) * 8.0),
        }
        for (x, y, z), (u, v) in zip(points, xy)
    ]

    report = run_error_budget_for_rows(
        rows=rows,
        gt_pose_by_query={"q0": pose},
        camera=camera,
        config=ErrorBudgetConfig(noise_sigmas_px=(0.0, 1.0), quantization_strides_px=(0.0, 8.0), trials=2),
    )

    assert report["variants"]["continuous_noise_0px"]["median_translation_error_m"] < 1e-4
    assert "continuous_noise_1px" in report["variants"]
    assert "continuous_quant_stride8px" in report["variants"]
    assert "cell_center_noise_0px" in report["variants"]
    assert report["variants"]["continuous_noise_0px"]["success_3cm_1deg"] == 1.0


def test_error_budget_success_rates_count_pnp_failures_as_failures() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=160, height=120, params=(100.0, 100.0, 80.0, 60.0))
    pose = np.eye(4, dtype=np.float64)
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
    xy = project_world_to_image(points, pose, camera)
    rows = []
    for query_id, count in (("q_ok", 8), ("q_fail", 3)):
        for (x, y, z), (u, v) in zip(points[:count], xy[:count]):
            rows.append(
                {
                    "query_id": query_id,
                    "world_x": float(x),
                    "world_y": float(y),
                    "world_z": float(z),
                    "query_x": float(u),
                    "query_y": float(v),
                    "cell_center_x": float(u),
                    "cell_center_y": float(v),
                }
            )

    report = run_error_budget_for_rows(
        rows=rows,
        gt_pose_by_query={"q_ok": pose, "q_fail": pose},
        camera=camera,
        config=ErrorBudgetConfig(
            noise_sigmas_px=(0.0,),
            quantization_strides_px=(0.0,),
            trials=1,
            pnp_min_inliers=6,
        ),
    )

    metrics = report["variants"]["continuous_noise_0px"]
    assert metrics["query_trial_count"] == 2
    assert metrics["solve_rate"] == 0.5
    assert metrics["success_3cm_1deg"] == 0.5
