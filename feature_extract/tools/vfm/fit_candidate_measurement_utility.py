#!/usr/bin/env python3
"""Fit a train-only candidate RGB coordinate-update utility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_measurement_utility import (
    fit_candidate_measurement_utility,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_spatial_likelihood",
        action="append",
        required=True,
        help="Repeat for deterministic train query shards.",
    )
    parser.add_argument("--train_supervision_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--c_value", type=float, default=0.1)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--target_precision", type=float, default=0.8)
    parser.add_argument("--minimum_selected_fraction", type=float, default=0.01)
    parser.add_argument("--minimum_baseline_residual_px", type=float, default=1.0)
    parser.add_argument("--minimum_improvement_px", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_candidate_measurement_utility(
        train_spatial_paths=[Path(path) for path in args.train_spatial_likelihood],
        train_supervision_rows_csv=Path(args.train_supervision_rows_csv),
        output_dir=Path(args.output_dir),
        c_value=float(args.c_value),
        fold_count=int(args.fold_count),
        target_precision=float(args.target_precision),
        minimum_selected_fraction=float(args.minimum_selected_fraction),
        minimum_baseline_residual_px=float(args.minimum_baseline_residual_px),
        minimum_improvement_px=float(args.minimum_improvement_px),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
