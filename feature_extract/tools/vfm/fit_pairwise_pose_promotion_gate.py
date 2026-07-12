"""Fit an abstaining pairwise gate for optional pose hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization.pairwise_pose_promotion import (
    fit_pairwise_pose_promotion_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_pose_dir", required=True)
    parser.add_argument("--validation_pose_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_precision", type=float, default=0.8)
    parser.add_argument("--c_value", type=float, default=1.0)
    parser.add_argument("--fold_count", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            fit_pairwise_pose_promotion_gate(
                train_pose_dir=Path(args.train_pose_dir),
                validation_pose_dir=Path(args.validation_pose_dir),
                output_dir=Path(args.output_dir),
                target_precision=float(args.target_precision),
                c_value=float(args.c_value),
                fold_count=int(args.fold_count),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
