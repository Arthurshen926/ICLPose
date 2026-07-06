from __future__ import annotations

import csv
from pathlib import Path

from feature_extract.vfm.measurement_v1.observability_training_rows import export_observability_dustbin_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_export_observability_dustbin_rows_marks_unobservable_rows_as_dustbin(tmp_path: Path) -> None:
    rows_csv = tmp_path / "rows.csv"
    _write_csv(
        rows_csv,
        [
            {"query_id": "a.png", "observability_class": "observable_subpixel", "target_is_dustbin": "False", "center_x": "1"},
            {"query_id": "b.png", "observability_class": "observable_coarse_only", "target_is_dustbin": "False", "center_x": "2"},
            {"query_id": "c.png", "observability_class": "ambiguous", "target_is_dustbin": "False", "center_x": "3"},
            {"query_id": "d.png", "observability_class": "window_out", "target_is_dustbin": "False", "center_x": "4"},
        ],
    )

    summary = export_observability_dustbin_rows(rows_csv=rows_csv, output_csv=tmp_path / "out.csv")

    assert summary["row_count"] == 4
    assert summary["valid_rows"] == 2
    assert summary["dustbin_rows"] == 2
    rows = list(csv.DictReader(Path(summary["output_csv"]).open()))
    assert [row["target_is_dustbin"] for row in rows] == ["False", "False", "True", "True"]
    assert [row["observability_positive"] for row in rows] == ["1", "1", "0", "0"]
    assert all(row["observability_dustbin_policy"] == "positive_classes_valid_else_dustbin" for row in rows)
