from __future__ import annotations

import csv
from pathlib import Path

from feature_extract.vfm.measurement_v1.protocol_audit import audit_zero_perturbation_consistency


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_zero_perturbation_audit_is_query_level_not_aggregate_only(tmp_path: Path) -> None:
    p1 = tmp_path / "p1.csv"
    p3 = tmp_path / "p3.csv"
    _write_csv(
        p1,
        [
            {
                "query_id": "q0",
                "status": "ok",
                "translation_error_m": 0.01,
                "rotation_error_deg": 0.2,
                "pnp_inlier_count": 12,
                "match_count": 100,
            },
            {
                "query_id": "q1",
                "status": "ok",
                "translation_error_m": 0.02,
                "rotation_error_deg": 0.3,
                "pnp_inlier_count": 13,
                "match_count": 100,
            },
        ],
    )
    _write_csv(
        p3,
        [
            {
                "query_id": "q0",
                "status": "ok",
                "translation_error_m": 0.01,
                "rotation_error_deg": 0.2,
                "pnp_inlier_count": 12,
                "match_count": 100,
            },
            {
                "query_id": "q1",
                "status": "ok",
                "translation_error_m": 0.021,
                "rotation_error_deg": 0.3,
                "pnp_inlier_count": 13,
                "match_count": 100,
            },
        ],
    )

    report = audit_zero_perturbation_consistency(reference_rows_csv=p1, zero_rows_csv=p3, output_dir=tmp_path / "audit")

    assert report["query_count_common"] == 2
    assert report["pass"] is False
    assert report["field_failures"]["translation_error_m"]["failure_count"] == 1
    assert (tmp_path / "audit" / "zero_consistency_report.json").exists()
