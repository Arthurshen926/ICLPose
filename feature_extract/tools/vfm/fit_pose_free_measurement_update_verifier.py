"""Fit a pose-free probability that an RGB measurement offset is beneficial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.pose_free_update_verifier import (
    fit_pose_free_update_verifier,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train_rows_csv",
        "train_diagnostics_csv",
        "train_support_selector_rows_csv",
        "validation_rows_csv",
        "validation_diagnostics_csv",
        "validation_support_selector_rows_csv",
        "output_dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--minimum_update_precision", type=float, default=0.8)
    parser.add_argument("--minimum_selected_fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_pose_free_update_verifier(
        train_rows_csv=Path(args.train_rows_csv),
        train_diagnostics_csv=Path(args.train_diagnostics_csv),
        train_support_selector_rows_csv=Path(args.train_support_selector_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        validation_diagnostics_csv=Path(args.validation_diagnostics_csv),
        validation_support_selector_rows_csv=Path(
            args.validation_support_selector_rows_csv
        ),
        output_dir=Path(args.output_dir),
        fold_count=int(args.fold_count),
        minimum_update_precision=float(args.minimum_update_precision),
        minimum_selected_fraction=float(args.minimum_selected_fraction),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
