"""Build measurement-v1 policy target rows from dense teacher diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.policy_rows import build_measurement_policy_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--dense_teacher_rows_csv", required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--center_preserve_below_px", type=float, default=0.5)
    parser.add_argument("--teacher_valid_max_epe_px", type=float, default=1.0)
    parser.add_argument("--center_loss_weight", type=float, default=1.0)
    parser.add_argument("--teacher_loss_weight", type=float, default=1.0)
    parser.add_argument("--fallback_loss_weight", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_measurement_policy_rows(
        rows_csv=Path(args.rows_csv),
        dense_teacher_rows_csv=Path(args.dense_teacher_rows_csv),
        output_rows_csv=Path(args.output_rows_csv),
        center_preserve_below_px=float(args.center_preserve_below_px),
        teacher_valid_max_epe_px=float(args.teacher_valid_max_epe_px),
        center_loss_weight=float(args.center_loss_weight),
        teacher_loss_weight=float(args.teacher_loss_weight),
        fallback_loss_weight=float(args.fallback_loss_weight),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
