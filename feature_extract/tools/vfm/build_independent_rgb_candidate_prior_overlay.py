"""Fuse target-free RGB candidate LLR predictions into a pose-prior overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_rgb_identity_overlay import (
    build_independent_rgb_candidate_prior_overlay,
)


def _path_list(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise argparse.ArgumentTypeError("at least one comma-separated path is required")
    return paths


def _float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("LLR weights must be comma-separated floats") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one LLR weight is required")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_overlay", required=True)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument(
        "--availability_evidence",
        default="",
        help="defaults to --candidate_evidence when both contracts are identical",
    )
    parser.add_argument("--prediction_artifacts", required=True, type=_path_list)
    parser.add_argument("--llr_weights", default="1", type=_float_list)
    parser.add_argument(
        "--llr_aggregation",
        choices=("sum", "mean"),
        default="sum",
        help=(
            "sum preserves historical per-artifact weights; mean averages the "
            "weighted seed LLRs so a common weight remains the ensemble weight"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    availability = (
        Path(args.candidate_evidence)
        if not str(args.availability_evidence).strip()
        else Path(args.availability_evidence)
    )
    result = build_independent_rgb_candidate_prior_overlay(
        base_overlay_path=Path(args.base_overlay),
        candidate_evidence_path=Path(args.candidate_evidence),
        availability_evidence_path=availability,
        prediction_paths=tuple(args.prediction_artifacts),
        llr_weights=tuple(args.llr_weights),
        llr_aggregation=str(args.llr_aggregation),
        output_path=Path(args.output),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
