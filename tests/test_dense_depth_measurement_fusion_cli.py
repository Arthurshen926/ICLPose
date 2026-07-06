from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.eval_dense_depth_measurement_fusion import main
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_eval_dense_depth_measurement_fusion_cli_writes_tables_and_summary(tmp_path: Path) -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [
            [-0.8, -0.5, 4.5],
            [0.8, -0.5, 5.0],
            [-0.8, 0.5, 5.5],
            [0.8, 0.5, 6.0],
            [-0.2, -0.8, 4.8],
            [0.4, 0.7, 5.8],
        ],
        dtype=np.float64,
    )
    xy = project_world_to_image(points, pose, camera)
    rows = []
    for index, ((x, y), depth) in enumerate(zip(xy, points[:, 2])):
        rows.append(
            {
                "query_id": "q0.png",
                "match_index": index,
                "render_x": x,
                "render_y": y,
                "render_depth": depth,
                "world_x": points[index, 0],
                "world_y": points[index, 1],
                "world_z": points[index, 2],
                "query_center_x": x + 2.0,
                "query_center_y": y,
                "query_refined_x": x,
                "query_refined_y": y,
                "query_gt_x": x,
                "query_gt_y": y,
                "measurement_valid_prob": 1.0,
            }
        )
    match_table = tmp_path / "match_table.csv"
    output_dir = tmp_path / "fusion"
    _write_csv(match_table, rows)

    main(
        [
            "--match_table_csv",
            str(match_table),
            "--output_dir",
            str(output_dir),
            "--camera_width",
            "640",
            "--camera_height",
            "480",
            "--camera_model_id",
            "1",
            "--camera_params",
            "500",
            "500",
            "320",
            "240",
            "--gt_pose_w2c_json",
            json.dumps(pose.tolist()),
            "--solvers",
            "plain",
            "--geometry_source",
            "prefer_world_xyz",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    dense_rows = list(csv.DictReader((output_dir / "match_table.csv").open()))
    pose_rows = list(csv.DictReader((output_dir / "pose_rows.csv").open()))
    ablation_rows = list(csv.DictReader((output_dir / "ablation_summary.tsv").open(), delimiter="\t"))
    matrix_rows = list(csv.DictReader((output_dir / "matrix_summary.tsv").open(), delimiter="\t"))
    jsonl_rows = [json.loads(line) for line in (output_dir / "match_table.jsonl").read_text().splitlines()]
    assert summary["stage"] == "radio_matcha_dense_depth_measurement_fusion_cli"
    assert summary["input_count"] == 6
    assert summary["query_count"] == 1
    assert summary["dense_depth_summary"]["measurement_epe_median_px"] == 0.0
    assert summary["dense_depth_summary"]["measurement_improve_ratio"] == 1.0
    assert summary["outputs"]["match_table_csv"].endswith("match_table.csv")
    assert summary["outputs"]["pose_rows_csv"].endswith("pose_rows.csv")
    assert summary["outputs"]["matrix_summary_tsv"].endswith("matrix_summary.tsv")
    assert len(dense_rows) == 6
    assert len(pose_rows) == len(ablation_rows)
    assert len(jsonl_rows) == 6
    assert len(matrix_rows) == 3
    assert {row["variant"] for row in matrix_rows} == {"center", "measurement", "oracle"}
    assert all(float(row["measurement_improve_ratio"]) == 1.0 for row in matrix_rows)
    assert {row["variant"] for row in ablation_rows} == {"center", "measurement", "oracle"}
    measurement = next(row for row in ablation_rows if row["variant"] == "measurement")
    assert float(measurement["translation_error_m"]) < 1e-4


def test_eval_dense_depth_measurement_fusion_cli_can_project_query_gt_from_pose_file(tmp_path: Path) -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))
    pose = np.eye(4, dtype=np.float64)
    points = np.asarray(
        [
            [-0.8, -0.5, 4.5],
            [0.8, -0.5, 5.0],
            [-0.8, 0.5, 5.5],
            [0.8, 0.5, 6.0],
            [-0.2, -0.8, 4.8],
            [0.4, 0.7, 5.8],
        ],
        dtype=np.float64,
    )
    xy = project_world_to_image(points, pose, camera)
    rows = []
    for index, ((x, y), depth) in enumerate(zip(xy, points[:, 2])):
        rows.append(
            {
                "query_id": "q0.png",
                "match_index": index,
                "render_x": x,
                "render_y": y,
                "render_depth": depth,
                "world_x": points[index, 0],
                "world_y": points[index, 1],
                "world_z": points[index, 2],
                "query_center_x": x + 2.0,
                "query_center_y": y,
                "query_refined_x": x,
                "query_refined_y": y,
                "measurement_valid_prob": 1.0,
            }
        )
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
    )
    match_table = tmp_path / "match_table.csv"
    output_dir = tmp_path / "fusion"
    _write_csv(match_table, rows)

    main(
        [
            "--match_table_csv",
            str(match_table),
            "--output_dir",
            str(output_dir),
            "--camera_width",
            "640",
            "--camera_height",
            "480",
            "--camera_model_id",
            "1",
            "--camera_params",
            "500",
            "500",
            "320",
            "240",
            "--query_pose_file",
            str(pose_file),
            "--solvers",
            "plain",
            "--geometry_source",
            "prefer_world_xyz",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    dense_rows = list(csv.DictReader((output_dir / "match_table.csv").open()))
    assert summary["query_gt_projection_summary"]["projected_query_gt_count"] == 6
    assert summary["dense_depth_summary"]["measurement_improve_ratio"] == 1.0
    assert dense_rows[0]["query_gt_x"] != ""


def test_eval_dense_depth_measurement_fusion_cli_groups_by_candidate_id(tmp_path: Path) -> None:
    camera = ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))
    points = np.asarray(
        [
            [-0.8, -0.5, 4.5],
            [0.8, -0.5, 5.0],
            [-0.8, 0.5, 5.5],
            [0.8, 0.5, 6.0],
            [-0.2, -0.8, 4.8],
            [0.4, 0.7, 5.8],
        ],
        dtype=np.float64,
    )
    xy = project_world_to_image(points, np.eye(4, dtype=np.float64), camera)
    rows = []
    for candidate_id, shift in [("c0", 0.0), ("c1", 3.0)]:
        for index, (point, (x, y)) in enumerate(zip(points, xy)):
            rows.append(
                {
                    "query_id": "q0.png",
                    "candidate_id": candidate_id,
                    "match_index": f"{candidate_id}_{index}",
                    "world_x": point[0],
                    "world_y": point[1],
                    "world_z": point[2],
                    "query_center_x": x + shift,
                    "query_center_y": y,
                    "query_refined_x": x + shift,
                    "query_refined_y": y,
                    "query_gt_x": x,
                    "query_gt_y": y,
                }
            )
    match_table = tmp_path / "match_table.csv"
    output_dir = tmp_path / "fusion"
    _write_csv(match_table, rows)

    main(
        [
            "--match_table_csv",
            str(match_table),
            "--output_dir",
            str(output_dir),
            "--camera_width",
            "640",
            "--camera_height",
            "480",
            "--camera_model_id",
            "1",
            "--camera_params",
            "500",
            "500",
            "320",
            "240",
            "--solvers",
            "plain",
            "--geometry_source",
            "prefer_world_xyz",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    pose_rows = list(csv.DictReader((output_dir / "pose_rows.csv").open()))
    assert summary["query_count"] == 1
    assert summary["pose_group_count"] == 2
    assert summary["candidate_group_summary"]["max_candidates_per_query"] == 2
    assert {row["candidate_id"] for row in pose_rows} == {"c0", "c1"}


def test_eval_dense_depth_measurement_fusion_cli_rejects_single_render_pose_for_multiple_groups(tmp_path: Path) -> None:
    rows = [
        {
            "query_id": "q0.png",
            "match_index": "0",
            "render_x": "320.0",
            "render_y": "240.0",
            "render_depth": "4.0",
            "query_center_x": "320.0",
            "query_center_y": "240.0",
            "query_refined_x": "320.0",
            "query_refined_y": "240.0",
        },
        {
            "query_id": "q1.png",
            "match_index": "0",
            "render_x": "320.0",
            "render_y": "240.0",
            "render_depth": "4.0",
            "query_center_x": "320.0",
            "query_center_y": "240.0",
            "query_refined_x": "320.0",
            "query_refined_y": "240.0",
        },
    ]
    match_table = tmp_path / "match_table.csv"
    _write_csv(match_table, rows)

    with np.testing.assert_raises(ValueError):
        main(
            [
                "--match_table_csv",
                str(match_table),
                "--output_dir",
                str(tmp_path / "fusion"),
                "--camera_width",
                "640",
                "--camera_height",
                "480",
                "--camera_model_id",
                "1",
                "--camera_params",
                "500",
                "500",
                "320",
                "240",
                "--render_pose_w2c_json",
                json.dumps(np.eye(4, dtype=np.float64).tolist()),
                "--solvers",
                "plain",
            ]
        )
