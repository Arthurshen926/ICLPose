"""Fit a pose-free verifier for safe candidate-specific RGB coordinate updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_update_verifier import (
    fit_candidate_update_verifier,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train_diagnostic_rows_csv",
        "validation_diagnostic_rows_csv",
        "train_oof_geometry_probabilities_csv",
        "candidate_geometry_verifier_json",
        "measurement_checkpoint",
        "output_dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--c_value", type=float, default=0.1)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--target_precision", type=float, default=0.8)
    parser.add_argument("--minimum_selected_fraction", type=float, default=0.005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_candidate_update_verifier(
        train_diagnostic_rows_csv=Path(args.train_diagnostic_rows_csv),
        validation_diagnostic_rows_csv=Path(args.validation_diagnostic_rows_csv),
        train_oof_geometry_probabilities_csv=Path(
            args.train_oof_geometry_probabilities_csv
        ),
        candidate_geometry_verifier_json=Path(
            args.candidate_geometry_verifier_json
        ),
        measurement_checkpoint=Path(args.measurement_checkpoint),
        output_dir=Path(args.output_dir),
        c_value=float(args.c_value),
        fold_count=int(args.fold_count),
        target_precision=float(args.target_precision),
        minimum_selected_fraction=float(args.minimum_selected_fraction),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
