"""Build a controlled Cambridge rendered-pose lattice candidate bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.cambridge_pose_lattice import (
    build_cambridge_pose_lattice_bank,
    parse_world_offsets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GT-centered Cambridge pose lattice bank")
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--offsets", default="0,0,0;0.25,0,0;-0.25,0,0;0,0.25,0;0,-0.25,0;0,0,0.25;0,0,-0.25")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = build_cambridge_pose_lattice_bank(
        pose_file=Path(args.pose_file),
        protocol_name=args.protocol_name,
        offsets=parse_world_offsets(args.offsets),
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
