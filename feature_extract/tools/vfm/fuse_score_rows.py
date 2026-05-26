"""Fuse aligned VFM score-row tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.reporting import build_gate_table
from feature_extract.vfm.score_fusion import (
    fuse_two_score_rows,
    fuse_score_rows,
    rows_from_json_payload,
    rows_to_json_payload,
)
from feature_extract.vfm.score_table import evaluate_score_table


def _load_rows(path: Path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"score rows file must be a non-empty JSON list: {path}")
    return rows_from_json_payload(payload)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Fuse aligned candidate score-row tables")
    parser.add_argument("--score_rows", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--primary_rows", default="")
    parser.add_argument("--secondary_rows", default="")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alignment", default="candidate_id", choices=("candidate_id", "query_rank", "init_lattice_id"))
    parser.add_argument(
        "--normalization",
        default="zscore",
        choices=("none", "zscore", "minmax", "rank_percentile"),
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output_rows", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--output_md", default="")
    args = parser.parse_args(argv)

    if args.primary_rows or args.secondary_rows or args.alpha is not None:
        if not args.primary_rows or not args.secondary_rows or args.alpha is None:
            raise ValueError("--primary_rows, --secondary_rows, and --alpha must be provided together")
        if not 0.0 <= float(args.alpha) <= 1.0:
            raise ValueError("--alpha must be in [0, 1]")
        paths = [Path(args.primary_rows), Path(args.secondary_rows)]
        weights = [1.0 - float(args.alpha), float(args.alpha)]
    else:
        if not args.score_rows or not args.weights:
            raise ValueError("provide either --score_rows/--weights or --primary_rows/--secondary_rows/--alpha")
        paths = [Path(path) for path in args.score_rows]
        weights = [float(weight) for weight in args.weights]

    rows = [_load_rows(path) for path in paths]
    if len(rows) == 2 and args.alpha is not None:
        fused = fuse_two_score_rows(
            rows[0],
            rows[1],
            alpha=float(args.alpha),
            method=args.method,
            normalization=args.normalization,
            alignment=args.alignment,
        )
    else:
        fused = fuse_score_rows(
            rows,
            weights=weights,
            method=args.method,
            normalization=args.normalization,
            alignment=args.alignment,
        )
    output_rows = Path(args.output_rows)
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    output_rows.write_text(json.dumps(rows_to_json_payload(fused), indent=2, sort_keys=True) + "\n")

    report = evaluate_score_table(fused).to_dict()
    report["inputs"] = {
        "alignment": args.alignment,
        "normalization": args.normalization,
        "weights": weights,
        "score_rows": [
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
            }
            for path in paths
        ],
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_gate_table("Fused Score Rows", [report]) + "\n")


if __name__ == "__main__":
    main()
