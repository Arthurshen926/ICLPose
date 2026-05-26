"""Summarize repeated-seed VFM score reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.seed_report_summary import (
    DEFAULT_SEED_REPORT_METRICS,
    seed_report_summary_markdown,
    summarize_seed_reports,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated-seed VFM report JSON files")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_SEED_REPORT_METRICS))
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    summary = summarize_seed_reports(
        [Path(path) for path in args.reports],
        label=args.label,
        metrics=tuple(args.metrics),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(seed_report_summary_markdown(summary))


if __name__ == "__main__":
    main()
