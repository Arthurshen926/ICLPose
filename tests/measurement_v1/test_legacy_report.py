from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.eval_measurement_v1 import main
from feature_extract.vfm.measurement_v1.legacy_report import summarize_legacy_measurement_eval


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_legacy_measurement_eval_reports_fine_and_oracle_gaps(tmp_path: Path) -> None:
    eval_dir = tmp_path / "gt"
    (eval_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "query_count": 2,
                    "median_translation_error_m": 0.02,
                    "success_10cm_5deg": 0.75,
                    "oracle_pnp_oracle_fine_all_ransac_median_translation_error_m": 0.000001,
                    "oracle_pnp_all_ransac_median_translation_error_m": 0.02,
                    "match_validity_ece_5px": 0.4,
                }
            }
        )
        + "\n"
    )
    _write_csv(
        eval_dir / "match_table.csv",
        [
            {
                "query_id": "q0",
                "baseline_reproj_residual_px": 0.5,
                "gt_reproj_error_px": 0.8,
                "gt_correct_5px": "True",
                "pnp_inlier": "True",
                "confidence": 0.2,
            },
            {
                "query_id": "q0",
                "baseline_reproj_residual_px": 3.0,
                "gt_reproj_error_px": 2.0,
                "gt_correct_5px": "True",
                "pnp_inlier": "False",
                "confidence": 0.1,
            },
        ],
    )

    report = summarize_legacy_measurement_eval(eval_dir)

    assert report["metrics"]["query_count"] == 2
    assert report["measurement"]["fine_after_better_ratio"] == 0.5
    assert report["measurement"]["after_recall_5px"] == 1.0
    assert report["oracle"]["oracle_fine_gap_m"] < 0.0201
    assert report["assessment"]["fine_measurement_gate_passed"] is False
    assert report["assessment"]["claim_level"] == "partial_gt_render_signal_only"


def test_eval_measurement_v1_legacy_report_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    eval_dir = tmp_path / "gt"
    eval_dir.mkdir()
    (eval_dir / "summary.json").write_text(json.dumps({"metrics": {"query_count": 1}}) + "\n")
    _write_csv(
        eval_dir / "match_table.csv",
        [
            {
                "query_id": "q0",
                "baseline_reproj_residual_px": 1.0,
                "gt_reproj_error_px": 0.5,
                "gt_correct_5px": "True",
                "pnp_inlier": "True",
                "confidence": 0.8,
            }
        ],
    )
    output_dir = tmp_path / "report"

    main(["--legacy_eval_dir", str(eval_dir), "--output_dir", str(output_dir)])

    summary = json.loads((output_dir / "measurement_v1_legacy_report.json").read_text())
    assert summary["stage"] == "measurement_v1_legacy_report"
    assert (output_dir / "measurement_v1_legacy_report.md").exists()
