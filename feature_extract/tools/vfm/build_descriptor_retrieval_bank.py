"""Build descriptor-retrieval candidate banks with Cambridge pose labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from feature_extract.vfm.descriptor_retrieval_candidates import (
    build_descriptor_retrieval_reference_pose_bank,
)
from feature_extract.vfm.token_descriptor_bank import TokenDescriptorBank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a descriptor-retrieval reference-pose bank")
    parser.add_argument("--query_descriptors", required=True)
    parser.add_argument("--map_descriptors", required=True)
    parser.add_argument("--query_pose_file", required=True)
    parser.add_argument("--reference_pose_file", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--exclude_same_image", action="store_true")
    parser.add_argument("--rot_cost_weight", type=float, default=0.05)
    parser.add_argument("--max_abs_pose_center", type=float, default=None)
    parser.add_argument("--score_block_size", type=int, default=65536)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bank = build_descriptor_retrieval_reference_pose_bank(
        query_descriptors=TokenDescriptorBank.from_npz(Path(args.query_descriptors)),
        map_descriptors=TokenDescriptorBank.from_npz(Path(args.map_descriptors)),
        query_pose_file=Path(args.query_pose_file),
        reference_pose_file=Path(args.reference_pose_file),
        protocol_name=args.protocol_name,
        top_k=args.top_k,
        exclude_same_image=args.exclude_same_image,
        rot_cost_weight=args.rot_cost_weight,
        max_abs_pose_center=args.max_abs_pose_center,
        score_block_size=int(args.score_block_size),
    )
    bank.to_jsonl(Path(args.output))


if __name__ == "__main__":
    main()
