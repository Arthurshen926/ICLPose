#!/usr/bin/env python3
"""Compare POFD-FS solver handoff caches with standard localization caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.radio_loc_retrieval_dataset import list_colmap_split_samples  # noqa: E402
from feature_extract.localizability.pose_cache_report import (  # noqa: E402
    build_pose_cache_comparison,
    pose_cache_comparison_to_markdown,
    query_names_from_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Cache spec in LABEL=PATH form. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--reference-cache",
        default=None,
        help="Optional cache whose query order defines the comparison subset.",
    )
    parser.add_argument("--colmap-dir", required=True)
    parser.add_argument("--query-split", required=True)
    parser.add_argument("--protocol", default="solver_handoff_localization_comparison")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _parse_cache_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"--cache must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"--cache must be LABEL=PATH, got: {value}")
    return label, path


def main() -> None:
    args = parse_args()
    samples = list_colmap_split_samples(args.colmap_dir, args.query_split)
    gt_poses_by_name = {
        str(sample["image_name"]): np.asarray(sample["pose_w2c"], dtype=np.float32)
        for sample in samples
    }
    query_names = query_names_from_cache(args.reference_cache) if args.reference_cache else None
    caches = [_parse_cache_spec(spec) for spec in args.cache]
    report = build_pose_cache_comparison(
        caches=caches,
        gt_poses_by_name=gt_poses_by_name,
        query_names=query_names,
        protocol=args.protocol,
    )
    report["colmap_dir"] = str(args.colmap_dir)
    report["query_split"] = str(args.query_split)
    report["reference_cache"] = str(args.reference_cache) if args.reference_cache else None

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(pose_cache_comparison_to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
