#!/usr/bin/env python3
"""Canonical FeatureRetrieval init-pose generation entrypoint."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.radio_loc_retrieval_dataset import (
    build_retrieval_init_entries,
    save_retrieval_init_entries,
)


def main():
    parser = argparse.ArgumentParser(description="Build retrieval init poses")
    parser.add_argument("--colmap_dir", required=True, help="COLMAP sparse dir")
    parser.add_argument("--train_split", required=True, help="Train split file")
    parser.add_argument("--query_split", required=True, help="Query/test split file")
    parser.add_argument(
        "--retrieval_feature_dir",
        default=None,
        help="Root containing cls/ retrieval features",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "cls", "nearest_train_pose_gt"],
        help="Init source",
    )
    parser.add_argument("--topk", type=int, default=1, help="Number of retrieval candidates to save per query")
    parser.add_argument(
        "--fallback_mode",
        default="nearest_train_pose_gt",
        choices=["nearest_train_pose_gt", "synthetic_noise"],
        help="Fallback when retrieval is unavailable",
    )
    parser.add_argument("--fallback_noise_deg", type=float, default=3.0)
    parser.add_argument("--fallback_noise_m", type=float, default=0.10)
    parser.add_argument("--save_path", required=True, help="Output .npz path")
    args = parser.parse_args()

    entries, stats = build_retrieval_init_entries(
        colmap_dir=args.colmap_dir,
        train_split_file=args.train_split,
        query_split_file=args.query_split,
        retrieval_feature_dir=args.retrieval_feature_dir,
        method=args.method,
        topk=args.topk,
        fallback_mode=args.fallback_mode,
        fallback_noise_rot_deg=args.fallback_noise_deg,
        fallback_noise_trans_m=args.fallback_noise_m,
    )
    save_retrieval_init_entries(entries, stats, args.save_path)

    print(f"Saved {len(entries)} init poses to {args.save_path}")
    print(f"Requested method: {stats.get('method_requested')}")
    print(f"Used method: {stats.get('method_used')}")
    print(f"Top-k: {stats.get('retrieval_topk_requested', 1)}")
    print(f"Counts by source: {stats.get('counts_by_source')}")


if __name__ == "__main__":
    main()
