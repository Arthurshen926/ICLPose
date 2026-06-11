"""Report guarded handoff between baseline and anti-lock render-pose candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _float_value(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        item = float(value)
    except (TypeError, ValueError):
        return None
    return item if np.isfinite(item) else None


def _success_25cm_10deg(row: Mapping[str, object]) -> bool:
    translation = _float_value(row, "translation_error_m")
    rotation = _float_value(row, "rotation_error_deg")
    return translation is not None and rotation is not None and translation <= 0.25 and rotation <= 10.0


def _read_rows(path: Path) -> list[dict[str, object]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def summarize_guarded_handoff(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    min_pnp_render_delta_m: float,
    max_pnp_render_delta_m: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    baseline_by_query = {str(row.get("query_id")): dict(row) for row in baseline_rows}
    candidate_by_query = {str(row.get("query_id")): dict(row) for row in candidate_rows}
    query_ids = sorted(set(baseline_by_query).intersection(candidate_by_query))
    selected_rows: list[dict[str, object]] = []
    selected_candidate_count = 0
    rescued = 0
    broken = 0
    for query_id in query_ids:
        baseline = baseline_by_query[query_id]
        candidate = candidate_by_query[query_id]
        delta = _float_value(candidate, "pnp_render_translation_delta_m")
        use_candidate = (
            delta is not None
            and delta >= float(min_pnp_render_delta_m)
            and delta <= float(max_pnp_render_delta_m)
        )
        selected = candidate if use_candidate else baseline
        if use_candidate:
            selected_candidate_count += 1
        baseline_success = _success_25cm_10deg(baseline)
        selected_success = _success_25cm_10deg(selected)
        if not baseline_success and selected_success:
            rescued += 1
        if baseline_success and not selected_success:
            broken += 1
        row = dict(selected)
        row.update(
            {
                "selected_source": "candidate" if use_candidate else "baseline",
                "baseline_translation_error_m": _float_value(baseline, "translation_error_m"),
                "candidate_translation_error_m": _float_value(candidate, "translation_error_m"),
                "candidate_pnp_render_translation_delta_m": delta,
                "baseline_success_25cm_10deg": baseline_success,
                "selected_success_25cm_10deg": selected_success,
            }
        )
        selected_rows.append(row)

    translations = [
        _float_value(row, "translation_error_m")
        for row in selected_rows
        if _float_value(row, "translation_error_m") is not None
    ]
    rotations = [
        _float_value(row, "rotation_error_deg")
        for row in selected_rows
        if _float_value(row, "rotation_error_deg") is not None
    ]
    success_values = [_success_25cm_10deg(row) for row in selected_rows]
    summary = {
        "query_count": int(len(selected_rows)),
        "selected_candidate_count": int(selected_candidate_count),
        "selected_candidate_rate": None
        if not selected_rows
        else float(selected_candidate_count) / float(len(selected_rows)),
        "min_pnp_render_delta_m": float(min_pnp_render_delta_m),
        "max_pnp_render_delta_m": float(max_pnp_render_delta_m),
        "median_translation_error_m": None if not translations else float(np.median(translations)),
        "median_rotation_error_deg": None if not rotations else float(np.median(rotations)),
        "success_25cm_10deg": None if not success_values else float(np.mean(success_values)),
        "rescued_success_25cm_10deg_count": int(rescued),
        "broken_success_25cm_10deg_count": int(broken),
        "rescue_break_ratio": None if broken == 0 else float(rescued) / float(broken),
    }
    return summary, selected_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_rows", required=True)
    parser.add_argument("--candidate_rows", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--min_pnp_render_delta_m", type=float, default=0.20)
    parser.add_argument("--max_pnp_render_delta_m", type=float, default=0.40)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary, rows = summarize_guarded_handoff(
        _read_rows(Path(args.baseline_rows)),
        _read_rows(Path(args.candidate_rows)),
        min_pnp_render_delta_m=float(args.min_pnp_render_delta_m),
        max_pnp_render_delta_m=float(args.max_pnp_render_delta_m),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_rows(Path(args.output_rows), rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
