"""Build real RGB measurement rows from a frozen S4 assignment policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.selected_policy_rows import (
    build_selected_policy_measurement_rows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_artifact", required=True)
    parser.add_argument("--support_geometry_index", required=True)
    parser.add_argument("--maplet_support_index", required=True)
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--split_name", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--search_radius_px", type=float, default=6.0)
    parser.add_argument("--context_radius_px", type=float, default=12.0)
    parser.add_argument("--minimum_geometry_p05", type=float, default=0.0)
    parser.add_argument("--support_views_per_row", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_selected_policy_measurement_rows(
        policy_artifact=Path(args.policy_artifact),
        support_geometry_index=Path(args.support_geometry_index),
        maplet_support_index=Path(args.maplet_support_index),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        split_json=Path(args.split_json),
        split_name=str(args.split_name),
        output_rows_csv=Path(args.output_rows_csv),
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        minimum_geometry_p05=float(args.minimum_geometry_p05),
        support_views_per_row=int(args.support_views_per_row),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
