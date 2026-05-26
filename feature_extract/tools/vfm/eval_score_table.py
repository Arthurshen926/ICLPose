"""Evaluate a VFM hypothesis-verification score table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_table import ScoreRow, evaluate_score_table


def _load_rows(path: Path) -> list[ScoreRow]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("rows file must contain a non-empty JSON list")
    return [ScoreRow(**dict(item)) for item in payload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a VFM score table")
    parser.add_argument("--rows", required=True, help="JSON list of score rows")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default=None)
    args = parser.parse_args()

    report = evaluate_score_table(_load_rows(Path(args.rows)))
    report_dict = report.to_dict()

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report_dict, indent=2, sort_keys=True) + "\n")

    if args.output_md is not None:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Hypothesis Verification", [report_dict]) + "\n")


if __name__ == "__main__":
    main()
