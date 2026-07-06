"""Export measurement-v1 SuperPoint teacher diagnostics for query-dense rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.superpoint_teacher_diagnostics import (
    export_superpoint_teacher_diagnostics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keypoint_cache_dir", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate_radius_px", type=float, default=8.0)
    parser.add_argument("--support_radius_px", type=float, default=8.0)
    parser.add_argument("--min_query_score", type=float, default=0.005)
    parser.add_argument("--min_support_score", type=float, default=0.005)
    parser.add_argument("--descriptor_weight", type=float, default=1.0)
    parser.add_argument("--query_score_weight", type=float, default=0.25)
    parser.add_argument("--center_penalty_weight", type=float, default=0.15)
    parser.add_argument("--support_distance_penalty_weight", type=float, default=0.10)
    parser.add_argument("--score_threshold", type=float, default=0.2)
    parser.add_argument("--support_selection_strategy", default="nearest", choices=("nearest", "highest_score"))
    parser.add_argument("--superpoint_nms_radius", type=int, default=4)
    parser.add_argument("--superpoint_keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--superpoint_max_keypoints", type=int, default=-1)
    parser.add_argument("--superpoint_remove_borders", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_superpoint_teacher_diagnostics(
        rows_csv=Path(args.rows_csv),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        keypoint_cache_dir=Path(args.keypoint_cache_dir) if str(args.keypoint_cache_dir) else None,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        device=str(args.device),
        candidate_radius_px=float(args.candidate_radius_px),
        support_radius_px=float(args.support_radius_px),
        min_query_score=float(args.min_query_score),
        min_support_score=float(args.min_support_score),
        descriptor_weight=float(args.descriptor_weight),
        query_score_weight=float(args.query_score_weight),
        center_penalty_weight=float(args.center_penalty_weight),
        support_distance_penalty_weight=float(args.support_distance_penalty_weight),
        score_threshold=float(args.score_threshold),
        support_selection_strategy=str(args.support_selection_strategy),
        superpoint_nms_radius=int(args.superpoint_nms_radius),
        superpoint_keypoint_threshold=float(args.superpoint_keypoint_threshold),
        superpoint_max_keypoints=int(args.superpoint_max_keypoints),
        superpoint_remove_borders=int(args.superpoint_remove_borders),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
