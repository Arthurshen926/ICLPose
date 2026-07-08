"""Summarize coarse proposal-pool valid residual counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.coarse_proposal_pool_diagnostics import (
    compare_pool_summaries,
    read_csv_rows,
    summarize_pool_rows,
    write_summary_json,
)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    out = tuple(float(item) for item in str(value).split(",") if item.strip())
    if not out:
        raise ValueError("at least one threshold is required")
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--thresholds_px", default="2,5")
    parser.add_argument("--label", default="")
    parser.add_argument("--baseline_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = _parse_thresholds(str(args.thresholds_px))
    rows = read_csv_rows(args.match_table)
    summary = summarize_pool_rows(rows, thresholds_px=thresholds)
    summary["label"] = str(args.label)
    summary["match_table"] = str(args.match_table)
    if str(args.baseline_json):
        baseline = json.loads(Path(args.baseline_json).read_text())
        summary["comparison_to_baseline"] = compare_pool_summaries(baseline, summary)
    write_summary_json(args.output_json, summary)


if __name__ == "__main__":
    main()
