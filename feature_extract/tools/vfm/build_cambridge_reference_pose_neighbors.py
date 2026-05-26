"""Build Cambridge reference-pose nearest-neighbor candidate banks."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.cambridge_pose_lattice import build_cambridge_reference_pose_neighbor_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Cambridge reference-pose neighbor bank")
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--rot_cost_weight", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = build_cambridge_reference_pose_neighbor_bank(
        query_pose_file=Path(args.query_pose_file),
        reference_pose_file=Path(args.reference_pose_file),
        protocol_name=args.protocol_name,
        top_k=args.top_k,
        exclude_same_image=args.exclude_same_image,
        rot_cost_weight=args.rot_cost_weight,
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
