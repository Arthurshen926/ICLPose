#!/usr/bin/env python3
"""Report geometry-only diagnostics for pose-candidate caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples  # noqa: E402
from feature_extract.localizability.pose_cache_report import (  # noqa: E402
    pose_candidate_cache_diagnostics_to_markdown,
    query_names_from_candidate_table,
    query_names_from_cache,
    summarize_pose_candidate_cache_geometry,
)


def _parse_cache_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--cache must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"--cache must be LABEL=PATH, got: {value}")
    return label, path


def _parse_topk(value: str) -> tuple[int, ...]:
    topk = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not topk:
        raise ValueError("--topk must contain at least one positive integer")
    if any(k <= 0 for k in topk):
        raise ValueError("--topk values must be positive")
    return topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True, help="Cache spec in LABEL=PATH form")
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--reference-cache", default=None)
    parser.add_argument("--query-table", default=None)
    parser.add_argument("--protocol", default="pose_candidate_cache_geometry_diagnostics")
    parser.add_argument("--rot-cost-weight", type=float, default=0.1)
    parser.add_argument("--basin-trans-m", type=float, default=0.25)
    parser.add_argument("--basin-rot-deg", type=float, default=10.0)
    parser.add_argument("--topk", default="1,5,10,20")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_cache and args.query_table:
        raise ValueError("--reference-cache and --query-table are mutually exclusive")
    samples = list_colmap_split_samples(args.colmap_dir, args.query_split)
    gt_poses_by_name = {
        str(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32)
        for sample in samples
    }
    if args.query_table:
        query_names = query_names_from_candidate_table(args.query_table)
    else:
        query_names = query_names_from_cache(args.reference_cache) if args.reference_cache else None
    topk = _parse_topk(str(args.topk))
    rows = [
        summarize_pose_candidate_cache_geometry(
            cache_path=cache_path,
            label=label,
            gt_poses_by_name=gt_poses_by_name,
            query_names=query_names,
            rot_cost_weight=float(args.rot_cost_weight),
            basin_trans_m=float(args.basin_trans_m),
            basin_rot_deg=float(args.basin_rot_deg),
            topk=topk,
        )
        for label, cache_path in [_parse_cache_spec(spec) for spec in args.cache]
    ]
    report = {
        "protocol": str(args.protocol),
        "colmap_dir": str(args.colmap_dir),
        "query_split": str(args.query_split),
        "query_subset_size": None if query_names is None else int(len(query_names)),
        "reference_cache": str(args.reference_cache) if args.reference_cache else None,
        "query_table": str(args.query_table) if args.query_table else None,
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(pose_candidate_cache_diagnostics_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
