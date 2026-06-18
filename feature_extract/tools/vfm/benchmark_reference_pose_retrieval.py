"""Benchmark a reference-pose retrieval candidate bank by pose-cost labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.retrieval_benchmark import summarize_reference_pose_retrieval


def _parse_top_ks(value: str) -> tuple[int, ...]:
    top_ks = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not top_ks:
        raise argparse.ArgumentTypeError("--top_ks must contain at least one integer")
    if min(top_ks) <= 0:
        raise argparse.ArgumentTypeError("--top_ks values must be positive")
    return top_ks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_bank", required=True)
    parser.add_argument("--top_ks", type=_parse_top_ks, default=(1, 5, 10))
    parser.add_argument(
        "--target_preset",
        default="custom",
        choices=["custom", "decimeter_coarse", "half_meter_coarse"],
    )
    parser.add_argument("--translation_threshold_m", type=float, default=0.25)
    parser.add_argument("--rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--rot_cost_weight", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    translation_threshold_m = float(args.translation_threshold_m)
    rotation_threshold_deg = float(args.rotation_threshold_deg)
    if args.target_preset == "decimeter_coarse":
        translation_threshold_m = 0.25
        rotation_threshold_deg = 5.0
    elif args.target_preset == "half_meter_coarse":
        translation_threshold_m = 0.5
        rotation_threshold_deg = 10.0

    summary = summarize_reference_pose_retrieval(
        CandidateHypothesisBank.from_jsonl(Path(args.candidate_bank)),
        top_ks=args.top_ks,
        translation_threshold_m=translation_threshold_m,
        rotation_threshold_deg=rotation_threshold_deg,
        rot_cost_weight=float(args.rot_cost_weight),
    )
    summary["inputs"] = {"candidate_bank": str(args.candidate_bank), "target_preset": str(args.target_preset)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
