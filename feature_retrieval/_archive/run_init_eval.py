#!/usr/bin/env python3
"""Thin wrapper to run full pose-refine evaluation with a learned init cache."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run full pipeline eval with learned init poses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--init_poses_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--pose_fusion", default="consensus_centroid", choices=["none", "centroid", "consensus", "consensus_centroid", "rgb_select"])
    parser.add_argument("--consensus_radius_m", type=float, default=1.0)
    parser.add_argument("--consensus_min_size", type=int, default=2)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--qual_limit", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--solver", default="flow+direct5")
    parser.add_argument("--outer_iters", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        "python",
        "feature_retrieval/evaluate_impl.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--gpu", str(args.gpu),
        "--init_poses_path", args.init_poses_path,
        "--retrieval_topk", str(args.retrieval_topk),
        "--pose_fusion", args.pose_fusion,
        "--consensus_radius_m", str(args.consensus_radius_m),
        "--consensus_min_size", str(args.consensus_min_size),
        "--qual_limit", str(args.qual_limit),
        "--solver", args.solver,
    ]
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    if args.max_samples > 0:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.outer_iters is not None:
        cmd.extend(["--outer_iters", str(args.outer_iters)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
