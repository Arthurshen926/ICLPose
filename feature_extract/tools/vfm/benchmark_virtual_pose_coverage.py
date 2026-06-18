"""Benchmark oracle coverage for virtual reference pose databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.virtual_pose_coverage import (
    parse_float_list,
    parse_thresholds,
    summarize_pose_file_oracle_coverage,
    summarize_virtual_grid_oracle_coverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--mode", choices=("pose_file", "grid"), default="pose_file")
    parser.add_argument("--thresholds", default="0.25,5")
    parser.add_argument("--grid_step_m", type=float, default=0.25)
    parser.add_argument("--grid_margin_m", type=float, default=0.0)
    parser.add_argument("--grid_height_mode", choices=("nearest", "idw"), default="nearest")
    parser.add_argument("--grid_height_knn", type=int, default=4)
    parser.add_argument("--grid_height_offsets_m", default="0")
    parser.add_argument("--grid_orientation_knn", type=int, default=1)
    parser.add_argument("--yaw_offsets_deg", default="0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    thresholds = parse_thresholds(str(args.thresholds))
    if args.mode == "grid":
        summary = summarize_virtual_grid_oracle_coverage(
            reference_pose_file=Path(args.reference_pose_file),
            query_pose_file=Path(args.query_pose_file),
            grid_step_m=float(args.grid_step_m),
            yaw_offsets_deg=parse_float_list(str(args.yaw_offsets_deg)),
            height_mode=str(args.grid_height_mode),
            height_knn=int(args.grid_height_knn),
            height_offsets_m=parse_float_list(str(args.grid_height_offsets_m)),
            orientation_knn=int(args.grid_orientation_knn),
            grid_margin_m=float(args.grid_margin_m),
            thresholds=thresholds,
        )
    else:
        summary = summarize_pose_file_oracle_coverage(
            query_pose_file=Path(args.query_pose_file),
            reference_pose_file=Path(args.reference_pose_file),
            thresholds=thresholds,
        )
    summary["inputs"] = {
        "query_pose_file": str(args.query_pose_file),
        "reference_pose_file": str(args.reference_pose_file),
        "mode": str(args.mode),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
