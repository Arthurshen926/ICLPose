"""Apply a frozen selected-measurement action calibrator without threshold search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.action_calibration import (
    apply_measurement_action_calibrator,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--diagnostics_csv", required=True)
    parser.add_argument("--calibrator_json", required=True)
    parser.add_argument("--coarse_pose_context_csv", default="")
    parser.add_argument("--measurement_support_selector_rows_csv", default="")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_measurement_action_calibrator(
        rows_csv=Path(args.rows_csv),
        diagnostics_csv=Path(args.diagnostics_csv),
        calibrator_json=Path(args.calibrator_json),
        coarse_pose_context_csv=(
            Path(args.coarse_pose_context_csv)
            if str(args.coarse_pose_context_csv)
            else None
        ),
        measurement_support_selector_rows_csv=(
            Path(args.measurement_support_selector_rows_csv)
            if str(args.measurement_support_selector_rows_csv)
            else None
        ),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
