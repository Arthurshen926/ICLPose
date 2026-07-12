"""Build local track maplets and support-view choices for a landmark bank."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.local_maplet_matching import (
    build_hybrid_maplet_support_index,
    local_maplet_support_index_stats,
    save_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark_index", required=True)
    parser.add_argument("--output_index", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--maplet_k", type=int, default=32)
    parser.add_argument("--candidate_k", type=int, default=128)
    parser.add_argument("--max_support_views", type=int, default=8)
    parser.add_argument("--radius_m", type=float, default=0.0)
    parser.add_argument("--context_pool", default="quality_mean", choices=("mean", "quality_mean", "topk"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.landmark_index)
    landmark_index, source_metadata = load_landmark_index_npz(source_path)
    descriptor_space_id = str(source_metadata.get("descriptor_space_id", ""))
    if not descriptor_space_id:
        raise ValueError("landmark index is missing descriptor_space_id")
    start = time.time()
    maplets = build_hybrid_maplet_support_index(
        landmark_index,
        maplet_k=int(args.maplet_k),
        candidate_k=int(args.candidate_k),
        max_support_views=int(args.max_support_views),
        radius=None if float(args.radius_m) <= 0.0 else float(args.radius_m),
        context_pool=str(args.context_pool),
    )
    elapsed = float(time.time() - start)
    metadata = {
        "source_landmark_index": str(source_path),
        "source_landmark_index_sha256": file_sha256_short(source_path),
        "source_descriptor_space_id": descriptor_space_id,
        "source_unique_track_count": int(len(landmark_index)),
        "maplet_k": int(args.maplet_k),
        "candidate_k": int(args.candidate_k),
        "max_support_views": int(args.max_support_views),
        "radius_m": None if float(args.radius_m) <= 0.0 else float(args.radius_m),
        "context_pool": str(args.context_pool),
        "build_seconds": elapsed,
    }
    output_path = Path(args.output_index)
    save_local_maplet_support_index_npz(maplets, output_path, metadata=metadata)
    summary = {
        "stage": "landmark_local_maplet_support_index",
        "metadata": metadata,
        "stats": local_maplet_support_index_stats(maplets),
        "outputs": {
            "maplet_support_index": str(output_path),
            "summary": str(args.summary_json),
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
