from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.eval_measurement_v1 import main
from feature_extract.vfm.measurement_v1.gt_measurement_export import export_gt_measurement_from_legacy_eval


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_eval_dir(eval_dir: Path) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        eval_dir / "rows.csv",
        [
            {"query_id": "q0.png", "translation_error_m": 0.1},
            {"query_id": "q1.png", "translation_error_m": 0.2},
            {"query_id": "q_missing.png", "translation_error_m": ""},
        ],
    )
    _write_csv(
        eval_dir / "match_table.csv",
        [
            {
                "query_id": "q0.png",
                "candidate_id": "gt",
                "render_index": 4,
                "render_x": 8.0,
                "render_y": 9.0,
                "query_x": 8.2,
                "query_y": 9.1,
                "world_x": 1.0,
                "world_y": 2.0,
                "world_z": 3.0,
                "render_depth": 5.0,
                "render_alpha": 0.9,
                "baseline_reproj_residual_px": 1.2,
                "gt_reproj_error_px": 0.25,
                "confidence": 0.8,
            },
            {
                "query_id": "q1.png",
                "candidate_id": "gt",
                "render_index": 5,
                "render_x": 4.0,
                "render_y": 5.0,
                "query_x": 4.3,
                "query_y": 5.4,
                "world_x": 2.0,
                "world_y": 3.0,
                "world_z": 4.0,
                "render_depth": 6.0,
                "render_alpha": 0.7,
                "baseline_reproj_residual_px": 0.3,
                "gt_reproj_error_px": 0.8,
                "confidence": 0.2,
            },
        ],
    )


def test_export_gt_measurement_from_legacy_eval_writes_tables_and_gate(tmp_path: Path) -> None:
    eval_dir = tmp_path / "gt"
    _write_eval_dir(eval_dir)
    output_dir = tmp_path / "measurement"

    summary = export_gt_measurement_from_legacy_eval(eval_dir=eval_dir, output_dir=output_dir)

    assert (output_dir / "anchor_rows.csv").exists()
    assert (output_dir / "measurement_rows.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert summary["stage"] == "measurement_v1_gt_render_from_legacy_eval"
    assert summary["metrics"]["source_query_count"] == 3
    assert summary["metrics"]["match_table_query_count"] == 2
    assert summary["metrics"]["missing_match_table_query_count"] == 1
    assert summary["metrics"]["measurement_count"] == 2
    assert summary["metrics"]["fine_after_better_ratio"] == 0.5
    assert summary["gates"]["G3_measurement_effective"] is False
    rows = list(csv.DictReader((output_dir / "measurement_rows.csv").open()))
    assert rows[0]["query_pred_x"] == "8.2"
    assert rows[0]["residual_after_px"] == "0.25"


def test_export_gt_measurement_improve_ratio_does_not_pair_residuals_across_rows(tmp_path: Path) -> None:
    eval_dir = tmp_path / "gt_missing_residuals"
    eval_dir.mkdir(parents=True)
    _write_csv(
        eval_dir / "rows.csv",
        [
            {"query_id": "q0.png"},
            {"query_id": "q1.png"},
        ],
    )
    _write_csv(
        eval_dir / "match_table.csv",
        [
            {
                "query_id": "q0.png",
                "candidate_id": "gt",
                "render_index": 4,
                "render_x": 8.0,
                "render_y": 9.0,
                "query_x": 8.2,
                "query_y": 9.1,
                "world_x": 1.0,
                "world_y": 2.0,
                "world_z": 3.0,
                "render_depth": 5.0,
                "render_alpha": 0.9,
                "baseline_reproj_residual_px": 1.2,
                "gt_reproj_error_px": "",
                "confidence": 0.8,
            },
            {
                "query_id": "q1.png",
                "candidate_id": "gt",
                "render_index": 5,
                "render_x": 4.0,
                "render_y": 5.0,
                "query_x": 4.3,
                "query_y": 5.4,
                "world_x": 2.0,
                "world_y": 3.0,
                "world_z": 4.0,
                "render_depth": 6.0,
                "render_alpha": 0.7,
                "baseline_reproj_residual_px": "",
                "gt_reproj_error_px": 0.25,
                "confidence": 0.2,
            },
        ],
    )

    summary = export_gt_measurement_from_legacy_eval(eval_dir=eval_dir, output_dir=tmp_path / "measurement")

    assert summary["metrics"]["paired_residual_count"] == 0
    assert summary["metrics"]["fine_after_better_ratio"] is None


def test_eval_measurement_v1_cli_exports_gt_measurement_tables(tmp_path: Path) -> None:
    eval_dir = tmp_path / "gt"
    _write_eval_dir(eval_dir)
    output_dir = tmp_path / "cli_measurement"

    main(["--gt_measurement_eval_dir", str(eval_dir), "--output_dir", str(output_dir)])

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["metrics"]["measurement_count"] == 2
    assert (output_dir / "measurement_rows.csv").exists()
