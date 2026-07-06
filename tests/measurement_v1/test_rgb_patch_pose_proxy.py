from __future__ import annotations

import csv
import json
import struct

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, ColmapImageObservation
from feature_extract.vfm.measurement_v1.rgb_patch_pose_proxy import (
    deduplicate_prediction_rows,
    evaluate_pose_proxy_from_prediction_rows,
    scaled_colmap_camera,
)


def _write_pose_proxy_cameras_bin(path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 100, 100))
        handle.write(struct.pack("<dddd", 80.0, 80.0, 50.0, 50.0))


def _write_pose_proxy_images_bin(path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 1))
        handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
        handle.write(struct.pack("<ddd", 0.0, 0.0, 0.0))
        handle.write(struct.pack("<i", 1))
        handle.write(b"q.png\x00")
        handle.write(struct.pack("<Q", 0))


def _write_pose_proxy_points3d_bin(path, points: dict[int, np.ndarray]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point_id, xyz in points.items():
            handle.write(
                struct.pack(
                    "<QdddBBBd",
                    int(point_id),
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    255,
                    0,
                    0,
                    0.1,
                )
            )
            handle.write(struct.pack("<Q", 0))


def _write_prediction_rows(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_deduplicate_prediction_rows_keeps_lowest_dustbin_per_query_track_and_filters() -> None:
    rows = [
        {"query_id": "q.png", "track_id": "7", "query_pred_x": "1.0", "query_pred_y": "2.0", "dustbin_probability": "0.8"},
        {"query_id": "q.png", "track_id": "7", "query_pred_x": "3.0", "query_pred_y": "4.0", "dustbin_probability": "0.2"},
        {"query_id": "q.png", "track_id": "8", "query_pred_x": "5.0", "query_pred_y": "6.0", "dustbin_probability": "0.7"},
    ]

    kept = deduplicate_prediction_rows(rows, dustbin_threshold=0.5)

    assert len(kept) == 1
    assert kept[0]["track_id"] == "7"
    assert kept[0]["query_pred_x"] == "3.0"


def test_scaled_colmap_camera_scales_focal_and_principal_point() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=50, params=(80.0, 90.0, 50.0, 25.0))

    scaled = scaled_colmap_camera(camera, image_width=200, image_height=100)

    assert scaled.width == 200
    assert scaled.height == 100
    assert scaled.params == (160.0, 180.0, 100.0, 50.0)


def test_pose_proxy_recovers_identity_pose_from_prediction_rows() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    image = ColmapImageObservation(
        image_id=1,
        image_name="q.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.zeros(3, dtype=np.float64),
        xys=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
        5: np.asarray([0.0, 0.0, 4.2], dtype=np.float64),
        6: np.asarray([0.5, 0.7, 5.2], dtype=np.float64),
    }
    rows = []
    for track_id, xyz in points.items():
        x = 80.0 * xyz[0] / xyz[2] + 50.0
        y = 80.0 * xyz[1] / xyz[2] + 50.0
        rows.append(
            {
                "query_id": "q.png",
                "track_id": str(track_id),
                "query_pred_x": str(x),
                "query_pred_y": str(y),
                "dustbin_probability": "0.1",
            }
        )

    summary = evaluate_pose_proxy_from_prediction_rows(
        rows,
        cameras={1: camera},
        images_by_name={"q.png": image},
        xyz_by_track=points,
        image_width=100,
        image_height=100,
        dustbin_threshold=0.5,
        reprojection_error_px=2.0,
    )

    assert summary["query_count"] == 1
    assert summary["success_count"] == 1
    assert summary["median_translation_error_m"] < 1e-5
    assert summary["median_rotation_error_deg"] < 1e-4
    assert summary["translation_error_p90_m"] < 1e-5
    assert summary["rotation_error_p90_deg"] < 1e-4
    assert summary["rate_3cm_1deg"] == 1.0
    assert summary["rate_5cm_2deg"] == 1.0
    assert summary["rate_10cm_5deg"] == 1.0
    assert summary["median_match_count"] == 6.0
    assert summary["kept_measurement_count"] == 6
    assert summary["matched_measurement_count"] == 6
    assert summary["missing_xyz_count"] == 0


def test_pose_proxy_reports_measurements_missing_3d_tracks() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    image = ColmapImageObservation(
        image_id=1,
        image_name="q.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.zeros(3, dtype=np.float64),
        xys=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
    }
    rows = []
    for track_id, xyz in points.items():
        rows.append(
            {
                "query_id": "q.png",
                "track_id": str(track_id),
                "query_pred_x": str(80.0 * xyz[0] / xyz[2] + 50.0),
                "query_pred_y": str(80.0 * xyz[1] / xyz[2] + 50.0),
                "dustbin_probability": "0.1",
            }
        )
    rows.append(
        {
            "query_id": "q.png",
            "track_id": "999",
            "query_pred_x": "50.0",
            "query_pred_y": "50.0",
            "dustbin_probability": "0.1",
        }
    )

    summary = evaluate_pose_proxy_from_prediction_rows(
        rows,
        cameras={1: camera},
        images_by_name={"q.png": image},
        xyz_by_track=points,
        image_width=100,
        image_height=100,
        dustbin_threshold=0.5,
        reprojection_error_px=2.0,
    )

    assert summary["kept_measurement_count"] == 5
    assert summary["matched_measurement_count"] == 4
    assert summary["missing_xyz_count"] == 1


def test_pose_proxy_counts_queries_missing_from_model_as_failed() -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))
    image = ColmapImageObservation(
        image_id=1,
        image_name="q.png",
        camera_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        tvec=np.zeros(3, dtype=np.float64),
        xys=np.zeros((0, 2), dtype=np.float64),
        point3d_ids=np.zeros((0,), dtype=np.int64),
    )
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
    }
    rows = []
    for query_id in ("q.png", "missing.png"):
        for track_id, xyz in points.items():
            rows.append(
                {
                    "query_id": query_id,
                    "track_id": str(track_id),
                    "query_pred_x": str(80.0 * xyz[0] / xyz[2] + 50.0),
                    "query_pred_y": str(80.0 * xyz[1] / xyz[2] + 50.0),
                    "dustbin_probability": "0.1",
                }
            )

    summary = evaluate_pose_proxy_from_prediction_rows(
        rows,
        cameras={1: camera},
        images_by_name={"q.png": image},
        xyz_by_track=points,
        image_width=100,
        image_height=100,
        dustbin_threshold=0.5,
        reprojection_error_px=2.0,
    )

    assert summary["query_count"] == 2
    assert summary["missing_query_count"] == 1
    assert summary["success_count"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["rate_3cm_1deg"] == 0.5
    failed_rows = [row for row in summary["pose_rows"] if row["query_id"] == "missing.png"]
    assert failed_rows == [
        {
            "query_id": "missing.png",
            "match_count": 0,
            "success": False,
            "inlier_count": 0,
            "translation_error_m": float("inf"),
            "rotation_error_deg": float("inf"),
            "failure_reason": "missing_query_pose",
        }
    ]


def test_eval_rgb_patch_pose_proxy_cli_writes_pose_rows_and_can_select_prediction_columns(tmp_path) -> None:
    from feature_extract.tools.vfm.eval_rgb_patch_pose_proxy import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
        5: np.asarray([0.0, 0.0, 4.2], dtype=np.float64),
        6: np.asarray([0.5, 0.7, 5.2], dtype=np.float64),
    }
    _write_pose_proxy_cameras_bin(model_dir / "cameras.bin")
    _write_pose_proxy_images_bin(model_dir / "images.bin")
    _write_pose_proxy_points3d_bin(model_dir / "points3D.bin", points)
    rows = []
    for track_id, xyz in points.items():
        x = 80.0 * xyz[0] / xyz[2] + 50.0
        y = 80.0 * xyz[1] / xyz[2] + 50.0
        rows.append(
            {
                "query_id": "q.png",
                "track_id": str(track_id),
                "query_pred_x": str(x + 20.0),
                "query_pred_y": str(y + 20.0),
                "center_x": str(x),
                "center_y": str(y),
                "dustbin_probability": "0.1",
            }
        )
    prediction_rows = tmp_path / "diagnostic_rows.csv"
    _write_prediction_rows(prediction_rows, rows)
    output_dir = tmp_path / "pose_proxy"

    main(
        [
            "--prediction_rows_csv",
            str(prediction_rows),
            "--model_dir",
            str(model_dir),
            "--output_dir",
            str(output_dir),
            "--image_width",
            "100",
            "--image_height",
            "100",
            "--prediction_x_key",
            "center_x",
            "--prediction_y_key",
            "center_y",
            "--reprojection_error_px",
            "2.0",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    pose_rows = list(csv.DictReader((output_dir / "pose_rows.csv").open()))
    assert summary["query_count"] == 1
    assert summary["success_count"] == 1
    assert summary["median_translation_error_m"] < 1e-5
    assert summary["rate_3cm_1deg"] == 1.0
    assert len(pose_rows) == 1
    assert pose_rows[0]["query_id"] == "q.png"


def test_eval_rgb_patch_pose_proxy_cli_can_filter_by_baseline_epe(tmp_path) -> None:
    from feature_extract.tools.vfm.eval_rgb_patch_pose_proxy import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
    }
    _write_pose_proxy_cameras_bin(model_dir / "cameras.bin")
    _write_pose_proxy_images_bin(model_dir / "images.bin")
    _write_pose_proxy_points3d_bin(model_dir / "points3D.bin", points)
    rows = []
    for track_id, xyz in points.items():
        x = 80.0 * xyz[0] / xyz[2] + 50.0
        y = 80.0 * xyz[1] / xyz[2] + 50.0
        rows.append(
            {
                "query_id": "q.png",
                "track_id": str(track_id),
                "query_pred_x": str(x + 30.0),
                "query_pred_y": str(y + 30.0),
                "center_x": str(x),
                "center_y": str(y),
                "baseline_epe_px": "3.0",
                "dustbin_probability": "0.1",
            }
        )
        rows.append(
            {
                "query_id": "q.png",
                "track_id": str(track_id),
                "query_pred_x": str(x),
                "query_pred_y": str(y),
                "center_x": str(x),
                "center_y": str(y),
                "baseline_epe_px": "0.5",
                "dustbin_probability": "0.1",
            }
        )
    prediction_rows = tmp_path / "diagnostic_rows.csv"
    _write_prediction_rows(prediction_rows, rows)
    output_dir = tmp_path / "pose_proxy_filtered"

    main(
        [
            "--prediction_rows_csv",
            str(prediction_rows),
            "--model_dir",
            str(model_dir),
            "--output_dir",
            str(output_dir),
            "--image_width",
            "100",
            "--image_height",
            "100",
            "--max_baseline_epe_px",
            "0.6",
            "--reprojection_error_px",
            "2.0",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["input_row_count"] == 8
    assert summary["filtered_row_count"] == 4
    assert summary["success_count"] == 1
    assert summary["median_translation_error_m"] < 1e-5


def test_run_rgb_patch_pose_proxy_matrix_writes_method_and_bin_summary(tmp_path) -> None:
    from feature_extract.tools.vfm.run_rgb_patch_pose_proxy_matrix import main

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    points = {
        1: np.asarray([-1.0, -1.0, 4.0], dtype=np.float64),
        2: np.asarray([1.0, -1.0, 4.5], dtype=np.float64),
        3: np.asarray([-1.0, 1.0, 5.0], dtype=np.float64),
        4: np.asarray([1.0, 1.0, 5.5], dtype=np.float64),
    }
    _write_pose_proxy_cameras_bin(model_dir / "cameras.bin")
    _write_pose_proxy_images_bin(model_dir / "images.bin")
    _write_pose_proxy_points3d_bin(model_dir / "points3D.bin", points)
    rows = []
    for bin_value in (0.5, 2.0):
        for track_id, xyz in points.items():
            x = 80.0 * xyz[0] / xyz[2] + 50.0
            y = 80.0 * xyz[1] / xyz[2] + 50.0
            rows.append(
                {
                    "query_id": "q.png",
                    "track_id": str(track_id),
                    "query_pred_x": str(x),
                    "query_pred_y": str(y),
                    "center_x": str(x),
                    "center_y": str(y),
                    "query_gt_x": str(x),
                    "query_gt_y": str(y),
                    "baseline_epe_px": str(bin_value),
                    "dustbin_probability": "0.1",
                }
            )
    prediction_rows = tmp_path / "diagnostic_rows.csv"
    _write_prediction_rows(prediction_rows, rows)
    output_dir = tmp_path / "pose_proxy_matrix"

    main(
        [
            "--prediction_rows_csv",
            str(prediction_rows),
            "--model_dir",
            str(model_dir),
            "--output_dir",
            str(output_dir),
            "--image_width",
            "100",
            "--image_height",
            "100",
            "--baseline_bins_px",
            "0.5",
            "2.0",
            "--bin_tolerance_px",
            "0.01",
            "--reprojection_error_px",
            "2.0",
        ]
    )

    summary = json.loads((output_dir / "matrix_summary.json").read_text())
    tsv_rows = list(csv.DictReader((output_dir / "matrix_summary.tsv").open(), delimiter="\t"))
    comparison_rows = list(csv.DictReader((output_dir / "matrix_comparison.tsv").open(), delimiter="\t"))
    assert summary["stage"] == "measurement_v1_rgb_patch_pose_proxy_matrix"
    assert summary["row_count"] == 9
    assert summary["comparison_row_count"] == 6
    assert len(tsv_rows) == 9
    assert len(comparison_rows) == 6
    assert {row["method"] for row in tsv_rows} == {"measurement", "center", "oracle_gt"}
    assert {row["bin_label"] for row in tsv_rows} == {"all", "bin_0p500", "bin_2p000"}
    assert all(row["missing_query_count"] == "0" for row in tsv_rows)
    assert all(row["missing_camera_count"] == "0" for row in tsv_rows)
    assert all(float(row["rate_3cm_1deg"]) == 1.0 for row in tsv_rows)
    assert (output_dir / "matrix_comparison.json").exists()
    assert {row["comparison"] for row in comparison_rows} == {"measurement_vs_center", "measurement_vs_oracle_gt"}
    assert all(float(row["translation_median_improvement_m"]) == 0.0 for row in comparison_rows)
