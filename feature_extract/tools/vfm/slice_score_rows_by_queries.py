"""Slice score-row tables by a query-id set and re-evaluate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_fusion import (
    rows_from_json_payload,
    rows_to_json_payload,
    slice_score_rows_by_query_ids,
)
from feature_extract.vfm.score_table import evaluate_score_table


def _load_rows(path: Path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"score rows file must be a non-empty JSON list: {path}")
    return rows_from_json_payload(payload)


def _load_query_ids(path: Path, key: str) -> set[str]:
    payload = json.loads(Path(path).read_text())
    if key not in payload:
        raise ValueError(f"query set key {key!r} not found in {path}")
    values = payload[key]
    if not isinstance(values, list):
        raise ValueError(f"query set {key!r} must be a JSON list")
    return {str(value) for value in values}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Slice VFM score rows by query id set")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--query_set_json", required=True)
    parser.add_argument("--query_set_key", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    rows_path = Path(args.rows)
    query_set_path = Path(args.query_set_json)
    query_ids = _load_query_ids(query_set_path, args.query_set_key)
    rows = slice_score_rows_by_query_ids(_load_rows(rows_path), sorted(query_ids))

    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(rows_to_json_payload(rows), indent=2, sort_keys=True) + "\n")

    report = evaluate_score_table(rows).to_dict()
    report["inputs"] = {
        "rows": {
            "path": str(rows_path),
            "sha256": file_sha256_short(rows_path),
        },
        "query_set_json": {
            "path": str(query_set_path),
            "sha256": file_sha256_short(query_set_path),
        },
        "query_set_key": args.query_set_key,
        "query_set_count": len(query_ids),
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table(f"Score Rows: {args.query_set_key}", [report]) + "\n")


if __name__ == "__main__":
    main()
