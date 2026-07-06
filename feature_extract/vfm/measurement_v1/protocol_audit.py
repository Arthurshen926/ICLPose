from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_NUMERIC_FIELDS = (
    "translation_error_m",
    "rotation_error_deg",
    "render_translation_error_m",
    "render_rotation_error_deg",
    "realized_initial_translation_error_m",
    "realized_initial_rotation_error_deg",
    "pnp_inlier_count",
    "match_count",
    "depth_valid_match_count",
    "pose_candidate_match_count",
    "coarse_same_cell_fraction",
    "coarse_mean_cell_delta_x",
    "coarse_mean_cell_delta_y",
)
DEFAULT_EXACT_FIELDS = ("status", "pnp_success")


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            query_id = str(row.get("query_id", "")).strip()
            if query_id:
                out[query_id] = dict(row)
        return out


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _compare_numeric(
    reference_rows: Mapping[str, Mapping[str, object]],
    zero_rows: Mapping[str, Mapping[str, object]],
    *,
    field: str,
    tolerance: float,
) -> dict[str, Any]:
    failures = []
    compared = 0
    max_abs_delta = 0.0
    for query_id in sorted(set(reference_rows) & set(zero_rows)):
        ref = _float_or_none(reference_rows[query_id].get(field))
        cur = _float_or_none(zero_rows[query_id].get(field))
        if ref is None and cur is None:
            continue
        compared += 1
        if ref is None or cur is None:
            failures.append({"query_id": query_id, "reference": ref, "zero": cur, "delta": None})
            continue
        delta = abs(float(cur) - float(ref))
        max_abs_delta = max(max_abs_delta, delta)
        if delta > float(tolerance):
            failures.append({"query_id": query_id, "reference": ref, "zero": cur, "delta": delta})
    return {
        "compared_count": int(compared),
        "failure_count": int(len(failures)),
        "max_abs_delta": float(max_abs_delta),
        "examples": failures[:10],
    }


def _compare_exact(
    reference_rows: Mapping[str, Mapping[str, object]],
    zero_rows: Mapping[str, Mapping[str, object]],
    *,
    field: str,
) -> dict[str, Any]:
    failures = []
    compared = 0
    for query_id in sorted(set(reference_rows) & set(zero_rows)):
        ref = reference_rows[query_id].get(field)
        cur = zero_rows[query_id].get(field)
        if (ref is None or ref == "") and (cur is None or cur == ""):
            continue
        compared += 1
        if str(ref) != str(cur):
            failures.append({"query_id": query_id, "reference": ref, "zero": cur})
    return {
        "compared_count": int(compared),
        "failure_count": int(len(failures)),
        "examples": failures[:10],
    }


def audit_zero_perturbation_consistency(
    *,
    reference_rows_csv: Path,
    zero_rows_csv: Path,
    output_dir: Path | None = None,
    numeric_fields: Sequence[str] = DEFAULT_NUMERIC_FIELDS,
    exact_fields: Sequence[str] = DEFAULT_EXACT_FIELDS,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compare P1 GT-render rows and P3 zero-perturbation rows query-by-query."""

    reference_rows = _read_rows(Path(reference_rows_csv))
    zero_rows = _read_rows(Path(zero_rows_csv))
    common = sorted(set(reference_rows) & set(zero_rows))
    missing_in_zero = sorted(set(reference_rows) - set(zero_rows))
    missing_in_reference = sorted(set(zero_rows) - set(reference_rows))
    field_failures: dict[str, Any] = {}
    for field in numeric_fields:
        report = _compare_numeric(reference_rows, zero_rows, field=field, tolerance=float(tolerance))
        if report["compared_count"]:
            field_failures[field] = report
    exact_failures: dict[str, Any] = {}
    for field in exact_fields:
        report = _compare_exact(reference_rows, zero_rows, field=field)
        if report["compared_count"]:
            exact_failures[field] = report
    failed_numeric = sum(int(report["failure_count"]) for report in field_failures.values())
    failed_exact = sum(int(report["failure_count"]) for report in exact_failures.values())
    out = {
        "stage": "measurement_v1_zero_perturbation_protocol_audit",
        "reference_rows": str(reference_rows_csv),
        "zero_rows": str(zero_rows_csv),
        "query_count_reference": int(len(reference_rows)),
        "query_count_zero": int(len(zero_rows)),
        "query_count_common": int(len(common)),
        "missing_in_zero_count": int(len(missing_in_zero)),
        "missing_in_reference_count": int(len(missing_in_reference)),
        "missing_in_zero_examples": missing_in_zero[:10],
        "missing_in_reference_examples": missing_in_reference[:10],
        "field_failures": field_failures,
        "exact_failures": exact_failures,
        "pass": bool(not missing_in_zero and not missing_in_reference and failed_numeric == 0 and failed_exact == 0),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "zero_consistency_report.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out
