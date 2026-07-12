"""Fit a pose-free RGB measurement verifier from true GT pose residual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.pose_free_geometry_verifier import (
    fit_pose_free_geometry_verifier,
)


def _float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--train_diagnostics_csv", required=True)
    parser.add_argument("--train_support_selector_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--validation_diagnostics_csv", required=True)
    parser.add_argument("--validation_support_selector_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--c_values", type=_float_list, default=(0.01, 0.03, 0.1, 0.3, 1.0))
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--minimum_verification_precision", type=float, default=0.8)
    parser.add_argument("--minimum_selected_fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_pose_free_geometry_verifier(
        train_rows_csv=Path(args.train_rows_csv),
        train_diagnostics_csv=Path(args.train_diagnostics_csv),
        train_support_selector_rows_csv=Path(args.train_support_selector_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        validation_diagnostics_csv=Path(args.validation_diagnostics_csv),
        validation_support_selector_rows_csv=Path(
            args.validation_support_selector_rows_csv
        ),
        output_dir=Path(args.output_dir),
        c_values=args.c_values,
        fold_count=int(args.fold_count),
        minimum_verification_precision=float(args.minimum_verification_precision),
        minimum_selected_fraction=float(args.minimum_selected_fraction),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
