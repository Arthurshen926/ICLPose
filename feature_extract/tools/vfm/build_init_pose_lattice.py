"""Build init-centered rendered-pose lattice candidate banks."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.cambridge_pose_lattice import (
    build_init_pose_lattice_bank,
    fine_lattice_world_offsets,
    parse_world_offsets,
    q_level_world_offsets,
)
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an init-centered rendered-pose lattice bank")
    parser.add_argument("--init_bank", required=True)
    parser.add_argument("--gt_pose_file", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--q_level", choices=("q10", "q25", "q50"), default=None)
    parser.add_argument("--offsets", default=None)
    parser.add_argument("--fine_radius_m", type=float, default=None)
    parser.add_argument("--fine_step_m", type=float, default=0.1)
    parser.add_argument("--fine_height_offsets_m", default="0")
    parser.add_argument("--yaw_offsets_deg", default="0")
    parser.add_argument("--pitch_offsets_deg", default="0")
    parser.add_argument("--roll_offsets_deg", default="0")
    parser.add_argument("--max_inits_per_query", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.fine_radius_m is not None:
        height_offsets = tuple(float(item.strip()) for item in str(args.fine_height_offsets_m).split(",") if item.strip())
        offsets = fine_lattice_world_offsets(
            radius_m=float(args.fine_radius_m),
            step_m=float(args.fine_step_m),
            height_offsets_m=height_offsets,
        )
    elif args.offsets:
        offsets = parse_world_offsets(args.offsets)
    elif args.q_level:
        offsets = q_level_world_offsets(args.q_level)
    else:
        offsets = parse_world_offsets("0,0,0")
    bank = build_init_pose_lattice_bank(
        init_bank=CandidateHypothesisBank.from_jsonl(Path(args.init_bank)),
        gt_pose_file=Path(args.gt_pose_file),
        protocol_name=args.protocol_name,
        offsets=offsets,
        max_inits_per_query=args.max_inits_per_query,
        yaw_offsets_deg=tuple(float(item.strip()) for item in str(args.yaw_offsets_deg).split(",") if item.strip()),
        pitch_offsets_deg=tuple(float(item.strip()) for item in str(args.pitch_offsets_deg).split(",") if item.strip()),
        roll_offsets_deg=tuple(float(item.strip()) for item in str(args.roll_offsets_deg).split(",") if item.strip()),
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
