from __future__ import annotations

import csv
from pathlib import Path

from feature_extract.vfm.measurement_v1.rgb_patch_training_rows import build_rgb_patch_measurement_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_rgb_patch_measurement_rows_keeps_target_fixed_and_offsets_center(tmp_path: Path) -> None:
    anchor_rows = tmp_path / "anchor_rows.csv"
    _write_csv(
        anchor_rows,
        [
            {"query_id": "q0.png", "anchor_id": 7, "render_x": 8.0, "render_y": 9.0, "quality": 0.9},
            {"query_id": "q1.png", "anchor_id": 8, "render_x": 1.0, "render_y": 1.0, "quality": 0.9},
        ],
    )
    output_rows = tmp_path / "rgb_patch_rows.csv"

    summary = build_rgb_patch_measurement_rows(
        anchor_rows_csv=anchor_rows,
        output_rows_csv=output_rows,
        image_width=20,
        image_height=20,
        search_radius_px=2.0,
        context_radius_px=3.0,
        offsets_per_anchor=3,
        max_rows=None,
        seed=3,
    )

    assert summary["source_anchor_count"] == 2
    assert summary["usable_anchor_count"] == 1
    assert summary["row_count"] == 3
    rows = list(csv.DictReader(output_rows.open()))
    assert len(rows) == 3
    for row in rows:
        assert row["query_id"] == "q0.png"
        assert row["anchor_id"] == "7"
        assert float(row["query_gt_x"]) == 8.0
        assert float(row["query_gt_y"]) == 9.0
        dx = float(row["query_gt_x"]) - float(row["center_x"])
        dy = float(row["query_gt_y"]) - float(row["center_y"])
        assert abs(dx) <= 2.0
        assert abs(dy) <= 2.0


def test_build_rgb_patch_measurement_rows_can_exclude_identity_offsets(tmp_path: Path) -> None:
    anchor_rows = tmp_path / "anchor_rows.csv"
    _write_csv(anchor_rows, [{"query_id": "q0.png", "anchor_id": 7, "render_x": 8.0, "render_y": 9.0, "quality": 0.9}])
    output_rows = tmp_path / "rgb_patch_rows.csv"

    summary = build_rgb_patch_measurement_rows(
        anchor_rows_csv=anchor_rows,
        output_rows_csv=output_rows,
        image_width=20,
        image_height=20,
        search_radius_px=2.0,
        context_radius_px=3.0,
        offsets_per_anchor=3,
        max_rows=None,
        seed=3,
        include_identity=False,
    )

    rows = list(csv.DictReader(output_rows.open()))
    assert summary["row_count"] == 3
    assert all(int(row["offset_index"]) >= 1 for row in rows)
    assert all(abs(float(row["delta_x"])) > 1e-8 or abs(float(row["delta_y"])) > 1e-8 for row in rows)
