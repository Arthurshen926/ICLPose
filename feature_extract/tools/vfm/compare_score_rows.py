"""Compare two VFM score-row tables with paired query-level statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.paired_score_comparison import (
    DEFAULT_PAIRED_METRICS,
    attach_comparison_inputs,
    compare_score_rows,
    load_score_rows_json,
    paired_score_comparison_markdown,
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare method score rows against a paired baseline")
    parser.add_argument("--method_rows", required=True)
    parser.add_argument("--baseline_rows", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_PAIRED_METRICS))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    method_path = Path(args.method_rows)
    baseline_path = Path(args.baseline_rows)
    summary = compare_score_rows(
        load_score_rows_json(method_path),
        load_score_rows_json(baseline_path),
        label=args.label,
        metrics=tuple(args.metrics),
        resamples=args.resamples,
        seed=args.seed,
    )
    payload = attach_comparison_inputs(summary, method_path, baseline_path)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(paired_score_comparison_markdown(payload))


if __name__ == "__main__":
    main()
