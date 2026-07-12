"""Build candidate-specific real RGB measurement rows from a frozen top-M pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_measurement_rows import (
    build_candidate_measurement_rows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_artifact", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_name", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--search_radius_px", type=float, default=6.0)
    parser.add_argument("--context_radius_px", type=float, default=12.0)
    parser.add_argument("--support_views_per_candidate", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_measurement_rows(
        selection_artifact=Path(args.selection_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        maplet_support_index=Path(args.maplet_support_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        split_name=str(args.split_name),
        output_rows_csv=Path(args.output_rows_csv),
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        support_views_per_candidate=int(args.support_views_per_candidate),
        max_candidates=int(args.max_candidates) if int(args.max_candidates) > 0 else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
