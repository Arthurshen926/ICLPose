from __future__ import annotations

import csv
from pathlib import Path

import pytest

from feature_extract.tools.vfm.build_rgb_patch_fine_rows_from_coarse_measurements import (
    build_rgb_patch_fine_rows_from_coarse_measurements,
)


def test_build_fine_rows_recenters_on_coarse_prediction_and_marks_dustbin(tmp_path: Path) -> None:
    coarse_rows = tmp_path / "coarse.csv"
    with coarse_rows.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query_refined_x",
                "query_refined_y",
                "query_gt_x",
                "query_gt_y",
                "render_x",
                "render_y",
                "render_depth",
                "world_x",
                "world_y",
                "world_z",
                "measurement_valid_prob",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "near.png",
                "query_refined_x": "10.0",
                "query_refined_y": "10.0",
                "query_gt_x": "11.0",
                "query_gt_y": "10.0",
                "render_x": "5.0",
                "render_y": "6.0",
                "render_depth": "3.0",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
                "measurement_valid_prob": "0.9",
            }
        )
        writer.writerow(
            {
                "query_id": "far.png",
                "query_refined_x": "20.0",
                "query_refined_y": "20.0",
                "query_gt_x": "30.0",
                "query_gt_y": "20.0",
                "render_x": "7.0",
                "render_y": "8.0",
                "render_depth": "4.0",
                "world_x": "4.0",
                "world_y": "5.0",
                "world_z": "6.0",
                "measurement_valid_prob": "0.2",
            }
        )

    summary = build_rgb_patch_fine_rows_from_coarse_measurements(
        coarse_rows_csv=coarse_rows,
        output_dir=tmp_path / "out",
        dustbin_residual_px=4.0,
    )

    with Path(summary["outputs"]["rows_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["row_count"] == 2
    assert summary["dustbin_count"] == 1
    assert rows[0]["center_x"] == "10.0"
    assert rows[0]["center_y"] == "10.0"
    assert rows[0]["target_is_dustbin"] == "False"
    assert float(rows[0]["center_residual_px"]) == pytest.approx(1.0)
    assert rows[1]["target_is_dustbin"] == "True"
    assert float(rows[1]["center_residual_px"]) == pytest.approx(10.0)
