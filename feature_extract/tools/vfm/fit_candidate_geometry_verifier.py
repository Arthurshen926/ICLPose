"""Fit pose-free candidate geometry probability from true residual targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.measurement_v1.candidate_geometry_verifier import (
    fit_candidate_geometry_verifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_diagnostic_rows_csv", required=True)
    parser.add_argument("--validation_diagnostic_rows_csv", required=True)
    parser.add_argument("--measurement_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geometry_threshold_px", type=float, default=5.0)
    parser.add_argument("--c_value", type=float, default=1.0)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--target_precision", type=float, default=0.8)
    parser.add_argument("--allow_legacy_missing_checkpoint_hash", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            fit_candidate_geometry_verifier(
                train_diagnostic_rows_csv=Path(args.train_diagnostic_rows_csv),
                validation_diagnostic_rows_csv=Path(args.validation_diagnostic_rows_csv),
                measurement_checkpoint=Path(args.measurement_checkpoint),
                output_dir=Path(args.output_dir),
                geometry_threshold_px=float(args.geometry_threshold_px),
                c_value=float(args.c_value),
                fold_count=int(args.fold_count),
                target_precision=float(args.target_precision),
                allow_legacy_missing_checkpoint_hash=bool(
                    args.allow_legacy_missing_checkpoint_hash
                ),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
