"""Augment measurement-v1 rows with local SfM-track geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.local_affine_rows import (
    augment_measurement_rows_with_local_affine_from_colmap_model,
    augment_measurement_rows_with_local_affine_from_jsonl,
    augment_measurement_rows_with_local_homography_from_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_model", choices=("affine", "homography"), default="affine")
    parser.add_argument("--rows_csv", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--track_observations_jsonl", default="")
    source.add_argument("--model_dir", default="")
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--image_width", type=int, default=0)
    parser.add_argument("--image_height", type=int, default=0)
    parser.add_argument("--local_radius_px", type=float, default=32.0)
    parser.add_argument("--min_points", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=64)
    parser.add_argument("--max_rmse_px", type=float, default=0.0)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    augment = (
        augment_measurement_rows_with_local_homography_from_jsonl
        if str(args.geometry_model) == "homography"
        else augment_measurement_rows_with_local_affine_from_jsonl
    )
    common = {
        "rows_csv": Path(args.rows_csv),
        "output_rows_csv": Path(args.output_rows_csv),
        "image_width": int(args.image_width) if int(args.image_width) > 0 else None,
        "image_height": int(args.image_height) if int(args.image_height) > 0 else None,
        "local_radius_px": float(args.local_radius_px),
        "min_points": int(args.min_points),
        "max_points": int(args.max_points),
        "max_rmse_px": float(args.max_rmse_px) if float(args.max_rmse_px) > 0.0 else None,
        "max_rows": int(args.max_rows) if int(args.max_rows) > 0 else None,
    }
    if str(args.model_dir):
        if str(args.geometry_model) != "affine":
            raise SystemExit("--model_dir currently supports diagnostic affine only")
        summary = augment_measurement_rows_with_local_affine_from_colmap_model(
            model_dir=Path(args.model_dir),
            **common,
        )
    else:
        summary = augment(
            track_observations_jsonl=Path(args.track_observations_jsonl),
            **common,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
