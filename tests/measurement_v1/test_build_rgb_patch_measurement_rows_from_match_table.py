from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.tools.vfm import build_rgb_patch_measurement_rows_from_match_table as tool


def test_build_rgb_patch_measurement_rows_from_match_table_projects_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    match_table = tmp_path / "match_table.csv"
    with match_table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "match_index",
                "query_x",
                "query_y",
                "render_x",
                "render_y",
                "render_depth",
                "world_x",
                "world_y",
                "world_z",
                "similarity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "q0.png",
                "match_index": "7",
                "query_x": "10.0",
                "query_y": "20.0",
                "render_x": "5.0",
                "render_y": "6.0",
                "render_depth": "3.0",
                "world_x": "1.0",
                "world_y": "2.0",
                "world_z": "3.0",
                "similarity": "0.9",
            }
        )
    monkeypatch.setattr(
        tool,
        "_pose_lookup",
        lambda _path: {"q0.png": np.eye(4, dtype=np.float64)},
    )
    monkeypatch.setattr(
        tool,
        "_load_camera_from_model_dir",
        lambda _path: ColmapCamera(camera_id=1, model_id=1, width=32, height=32, params=(1.0, 1.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(
        tool,
        "project_world_to_image",
        lambda _xyz, _pose, _camera: np.asarray([[12.0, 23.0]], dtype=np.float64),
    )

    summary = tool.build_rgb_patch_measurement_rows_from_match_table(
        match_table_csv=match_table,
        query_pose_file=tmp_path / "poses.txt",
        camera_model_dir=tmp_path,
        output_dir=tmp_path / "out",
        requested_residual_bin_px=1.0,
    )

    assert summary["row_count"] == 1
    assert summary["center_residual_median_px"] == pytest.approx(13.0**0.5)
    with Path(summary["outputs"]["rows_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["center_x"] == "10.0"
    assert rows[0]["center_y"] == "20.0"
    assert rows[0]["query_gt_x"] == "12.0"
    assert rows[0]["query_gt_y"] == "23.0"
    assert float(rows[0]["center_residual_px"]) == pytest.approx(13.0**0.5)
    assert rows[0]["requested_residual_px"] == "4.0"
    assert rows[0]["render_x"] == "5.0"
    assert rows[0]["render_y"] == "6.0"
    assert rows[0]["radio_match_score"] == "0.9"
