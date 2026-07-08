from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

from feature_extract.tools.vfm.build_rgb_patch_measurement_rows_from_match_table import (
    build_rgb_patch_measurement_rows_from_match_table,
)


def _write_cameras_bin(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 100, 80))
        handle.write(struct.pack("<dddd", 50.0, 50.0, 50.0, 40.0))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_rows_preserves_actual_matcha_coarse_center_and_bins_residual(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin")
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
    )
    match_table = tmp_path / "match_table.csv"
    _write_csv(
        match_table,
        [
            {
                "query_id": "q0.png",
                "candidate_id": "cand0",
                "render_pose_id": "gt",
                "match_index": 7,
                "query_x": 55.0,
                "query_y": 44.0,
                "render_x": 50.0,
                "render_y": 40.0,
                "render_depth": 4.0,
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 4.0,
                "coarse_score": 0.75,
                "coarse_rank": 2,
                "similarity": 0.8,
            }
        ],
    )

    summary = build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=match_table,
        query_pose_file=pose_file,
        camera_model_dir=model_dir,
        output_dir=tmp_path / "rows",
        measurement_search_radius_px=4.0,
        requested_residual_bin_px=1.0,
        residual_bin_edges_px=(1.0, 2.0, 4.0, 8.0),
        proposal_source="unit_actual_coarse",
    )

    rows = list(csv.DictReader((tmp_path / "rows" / "measurement_rows.csv").open()))
    row = rows[0]
    assert summary["row_count"] == 1
    assert row["candidate_id"] == "gt"
    assert row["match_candidate_id"] == "cand0"
    assert row["proposal_source"] == "unit_actual_coarse"
    assert row["measurement_policy"] == "actual_matcha_coarse_to_projected_query_gt"
    assert row["center_source"] == "query_x,query_y"
    assert float(row["query_gt_x"]) == 50.0
    assert float(row["query_gt_y"]) == 40.0
    assert float(row["query_center_x"]) == 55.0
    assert float(row["query_center_y"]) == 44.0
    assert row["coarse_residual_bin"] == "4_8px"
    assert row["within_measurement_window"] == "False"
    assert row["target_is_dustbin"] == "True"
    assert row["coarse_score"] == "0.75"
    assert row["coarse_rank"] == "2"
    assert json.loads((tmp_path / "rows" / "summary.json").read_text())["proposal_source"] == "unit_actual_coarse"


def test_build_rows_scales_match_table_query_center_to_measurement_canvas(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _write_cameras_bin(model_dir / "cameras.bin")
    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        "q0.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0\n"
    )
    match_table = tmp_path / "match_table.csv"
    _write_csv(
        match_table,
        [
            {
                "query_id": "q0.png",
                "query_x": 25.0,
                "query_y": 20.0,
                "render_x": 50.0,
                "render_y": 40.0,
                "render_depth": 4.0,
                "world_x": 0.0,
                "world_y": 0.0,
                "world_z": 4.0,
            }
        ],
    )

    build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=match_table,
        query_pose_file=pose_file,
        camera_model_dir=model_dir,
        output_dir=tmp_path / "scaled_rows",
        match_table_query_image_width=50,
        match_table_query_image_height=40,
        measurement_query_image_width=100,
        measurement_query_image_height=80,
    )

    row = next(csv.DictReader((tmp_path / "scaled_rows" / "measurement_rows.csv").open()))
    assert float(row["query_center_x"]) == 50.0
    assert float(row["query_center_y"]) == 40.0
    assert float(row["query_gt_x"]) == 50.0
    assert float(row["query_gt_y"]) == 40.0
    assert float(row["center_residual_px"]) == 0.0
    summary = json.loads((tmp_path / "scaled_rows" / "summary.json").read_text())
    assert summary["center_scale_x"] == 2.0
    assert summary["center_scale_y"] == 2.0
