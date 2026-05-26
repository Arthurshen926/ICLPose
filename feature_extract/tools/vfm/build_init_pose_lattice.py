"""Build init-centered rendered-pose lattice candidate banks."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.cambridge_pose_lattice import (
    build_init_pose_lattice_bank,
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
    parser.add_argument("--max_inits_per_query", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.offsets:
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
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
