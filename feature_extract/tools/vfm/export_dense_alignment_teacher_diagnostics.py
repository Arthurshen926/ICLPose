"""Export measurement-v1 dense LK alignment teacher diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.dense_alignment_teacher import (
    export_dense_alignment_teacher_diagnostics,
)


def _optional_float(value: float) -> float | None:
    return None if float(value) < 0.0 else float(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--win_size_px", type=int, default=21)
    parser.add_argument("--max_level", type=int, default=2)
    parser.add_argument("--criteria_count", type=int, default=30)
    parser.add_argument("--criteria_eps", type=float, default=0.01)
    parser.add_argument("--min_eig_threshold", type=float, default=1e-4)
    parser.add_argument("--max_lk_error", type=float, default=40.0, help="negative disables this gate")
    parser.add_argument("--max_flow_from_center_px", type=float, default=8.0, help="negative disables this gate")
    parser.add_argument("--fb_max_error_px", type=float, default=1.0, help="negative disables forward-backward gate")
    parser.add_argument(
        "--center_preserve_below_px",
        type=float,
        default=-1.0,
        help="non-negative keeps center baseline for rows with requested_residual_px <= this value",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_dense_alignment_teacher_diagnostics(
        rows_csv=Path(args.rows_csv),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        win_size_px=int(args.win_size_px),
        max_level=int(args.max_level),
        criteria_count=int(args.criteria_count),
        criteria_eps=float(args.criteria_eps),
        min_eig_threshold=float(args.min_eig_threshold),
        max_lk_error=_optional_float(float(args.max_lk_error)),
        max_flow_from_center_px=_optional_float(float(args.max_flow_from_center_px)),
        fb_max_error_px=_optional_float(float(args.fb_max_error_px)),
        center_preserve_below_px=_optional_float(float(args.center_preserve_below_px)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
