"""Materialize local feature patch caches for measurement-v1 training rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.feature_patch_cache import (
    materialize_feature_patch_cache_rows,
    write_feature_patch_cache_summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--output_patch_dir", required=True)
    parser.add_argument("--feature_name", required=True)
    parser.add_argument("--feature_key", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--crop_radius_px", type=float, required=True)
    parser.add_argument("--step_px", type=float, required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--output_dtype", default="float32", choices=("float32", "float16", "fp32", "fp16"))
    parser.add_argument("--source_feature_cache_capacity", type=int, default=0)
    parser.add_argument("--max_patch_bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = materialize_feature_patch_cache_rows(
        rows_csv=Path(args.rows_csv),
        output_rows_csv=Path(args.output_rows_csv),
        output_patch_dir=Path(args.output_patch_dir),
        feature_name=str(args.feature_name),
        feature_key=str(args.feature_key),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        crop_radius_px=float(args.crop_radius_px),
        step_px=float(args.step_px),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        output_dtype=str(args.output_dtype),
        skip_existing=not bool(args.overwrite),
        source_feature_cache_capacity=int(args.source_feature_cache_capacity) if int(args.source_feature_cache_capacity) > 0 else None,
        max_patch_bytes=int(args.max_patch_bytes) if int(args.max_patch_bytes) > 0 else None,
    )
    if str(args.summary_json):
        write_feature_patch_cache_summary(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
