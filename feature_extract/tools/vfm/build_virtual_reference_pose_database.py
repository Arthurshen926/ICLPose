"""Build a synthetic/virtual reference pose database from Cambridge poses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.cambridge_pose_lattice import (
    build_virtual_reference_pose_grid_records,
    build_virtual_reference_pose_records,
    parse_world_offsets,
    write_cambridge_pose_file,
)


def _parse_yaw_offsets(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("at least one yaw offset is required")
    return values


def _parse_float_list(text: str, *, label: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError(f"at least one {label} is required")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--mode", default="offset", choices=["offset", "grid"])
    parser.add_argument("--offsets", default="0,0,0")
    parser.add_argument("--grid_step_m", type=float, default=0.5)
    parser.add_argument("--grid_margin_m", type=float, default=0.0)
    parser.add_argument("--grid_height_mode", default="nearest", choices=["nearest", "idw"])
    parser.add_argument("--grid_height_knn", type=int, default=4)
    parser.add_argument("--grid_height_offsets_m", default="0")
    parser.add_argument("--grid_orientation_knn", type=int, default=1)
    parser.add_argument("--yaw_offsets_deg", default="0")
    parser.add_argument("--image_prefix", default="virtual_reference")
    parser.add_argument("--output_pose_file", required=True)
    args = parser.parse_args(argv)

    if args.mode == "grid":
        records = build_virtual_reference_pose_grid_records(
            reference_pose_file=Path(args.reference_pose_file),
            grid_step_m=float(args.grid_step_m),
            yaw_offsets_deg=_parse_yaw_offsets(str(args.yaw_offsets_deg)),
            image_prefix=str(args.image_prefix),
            margin_m=float(args.grid_margin_m),
            height_mode=str(args.grid_height_mode),
            height_knn=int(args.grid_height_knn),
            height_offsets_m=_parse_float_list(str(args.grid_height_offsets_m), label="height offset"),
            orientation_knn=int(args.grid_orientation_knn),
        )
    else:
        records = build_virtual_reference_pose_records(
            reference_pose_file=Path(args.reference_pose_file),
            offsets=parse_world_offsets(str(args.offsets)),
            yaw_offsets_deg=_parse_yaw_offsets(str(args.yaw_offsets_deg)),
            image_prefix=str(args.image_prefix),
        )
    write_cambridge_pose_file(records, Path(args.output_pose_file))


if __name__ == "__main__":
    main()
