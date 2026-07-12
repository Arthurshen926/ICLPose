"""Fit selected-only KEEP/UPDATE/DROP calibration from GT pose residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.action_calibration import (
    fit_measurement_action_calibrator,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--train_diagnostics_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--validation_diagnostics_csv", required=True)
    parser.add_argument("--train_coarse_pose_context_csv", default="")
    parser.add_argument("--validation_coarse_pose_context_csv", default="")
    parser.add_argument("--train_measurement_support_selector_rows_csv", default="")
    parser.add_argument("--validation_measurement_support_selector_rows_csv", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--minimum_update_gain_px", type=float, default=0.1)
    parser.add_argument("--maximum_safe_worsening_px", type=float, default=0.1)
    parser.add_argument("--c_value", type=float, default=0.25)
    parser.add_argument(
        "--maximum_low_residual_worsen_ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--minimum_mid_residual_improvement_fraction", type=float, default=0.30
    )
    parser.add_argument(
        "--minimum_mid_residual_improve_ratio", type=float, default=0.60
    )
    parser.add_argument("--minimum_drop_precision", type=float, default=0.90)
    parser.add_argument("--require_diagnostic_manifests", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = fit_measurement_action_calibrator(
        train_rows_csv=Path(args.train_rows_csv),
        train_diagnostics_csv=Path(args.train_diagnostics_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        validation_diagnostics_csv=Path(args.validation_diagnostics_csv),
        train_coarse_pose_context_csv=(
            Path(args.train_coarse_pose_context_csv)
            if str(args.train_coarse_pose_context_csv)
            else None
        ),
        validation_coarse_pose_context_csv=(
            Path(args.validation_coarse_pose_context_csv)
            if str(args.validation_coarse_pose_context_csv)
            else None
        ),
        train_measurement_support_selector_rows_csv=(
            Path(args.train_measurement_support_selector_rows_csv)
            if str(args.train_measurement_support_selector_rows_csv)
            else None
        ),
        validation_measurement_support_selector_rows_csv=(
            Path(args.validation_measurement_support_selector_rows_csv)
            if str(args.validation_measurement_support_selector_rows_csv)
            else None
        ),
        output_dir=Path(args.output_dir),
        minimum_update_gain_px=float(args.minimum_update_gain_px),
        maximum_safe_worsening_px=float(args.maximum_safe_worsening_px),
        c_value=float(args.c_value),
        maximum_low_residual_worsen_ratio=float(
            args.maximum_low_residual_worsen_ratio
        ),
        minimum_mid_residual_improvement_fraction=float(
            args.minimum_mid_residual_improvement_fraction
        ),
        minimum_mid_residual_improve_ratio=float(
            args.minimum_mid_residual_improve_ratio
        ),
        minimum_drop_precision=float(args.minimum_drop_precision),
        require_diagnostic_manifests=bool(args.require_diagnostic_manifests),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
