#!/usr/bin/env python3
"""Report risk-coverage and false-accept diagnostics for a candidate table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.localizability.risk_report import (  # noqa: E402
    load_candidate_table_jsonl,
    summarize_candidate_table_risk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--topk", default="1,5")
    parser.add_argument("--confidence-mode", choices=("top1_margin", "top1_score"), default="top1_margin")
    parser.add_argument("--trans-basin-m", type=float, default=None)
    parser.add_argument("--rot-basin-deg", type=float, default=None)
    return parser.parse_args()


def _parse_topk(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def main() -> None:
    args = parse_args()
    rows = load_candidate_table_jsonl(args.candidate_table)
    summary = summarize_candidate_table_risk(
        rows,
        topk=_parse_topk(args.topk),
        confidence_mode=args.confidence_mode,
        trans_basin_m=args.trans_basin_m,
        rot_basin_deg=args.rot_basin_deg,
    )
    summary["candidate_table"] = str(args.candidate_table)
    summary["topk"] = list(_parse_topk(args.topk))
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
