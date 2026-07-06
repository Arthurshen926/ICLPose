from __future__ import annotations

import csv
import json
from pathlib import Path

from feature_extract.tools.vfm.eval_measurement_v1 import main


def test_eval_measurement_v1_synthetic_smoke_writes_standard_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / "measurement_v1_smoke"

    main(["--synthetic_smoke", "--output_dir", str(output_dir)])

    for name in ("anchor_rows.csv", "measurement_rows.csv", "pose_rows.csv", "summary.json"):
        assert (output_dir / name).exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stage"] == "measurement_v1_synthetic_smoke"
    assert summary["anchor_count"] >= 4
    assert summary["pose"]["success"] is True
    assert summary["pose"]["translation_error_m"] < 1e-5
    assert summary["pose"]["rotation_error_deg"] < 1e-4

    measurement_rows = list(csv.DictReader((output_dir / "measurement_rows.csv").open()))
    assert measurement_rows
    assert {"query_pred_x", "query_pred_y", "cov_xx", "cov_xy", "cov_yy", "p_visible", "p_assignment"}.issubset(
        measurement_rows[0].keys()
    )
