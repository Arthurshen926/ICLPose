"""Materialize original-resolution RGB patch caches for measurement-v1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_native_patch_cache import (
    materialize_rgb_native_patch_cache_rows,
    write_rgb_native_patch_cache_summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_rows_csv", required=True)
    parser.add_argument("--output_patch_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--render_cache_manifest_csv", default="")
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--crop_radius_px", type=float, required=True)
    parser.add_argument("--step_px", type=float, required=True)
    parser.add_argument("--query_source", default="real", choices=("real", "real_pair", "render"))
    parser.add_argument("--feature_name", default="rgb_native")
    parser.add_argument("--feature_key", default="rgb_native")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--output_dtype", default="float32", choices=("float32", "float16", "fp32", "fp16"))
    parser.add_argument("--image_cache_capacity", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = materialize_rgb_native_patch_cache_rows(
        rows_csv=Path(args.rows_csv),
        output_rows_csv=Path(args.output_rows_csv),
        output_patch_dir=Path(args.output_patch_dir),
        image_root=Path(args.image_root),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv) if str(args.render_cache_manifest_csv) else None,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        crop_radius_px=float(args.crop_radius_px),
        step_px=float(args.step_px),
        query_source=str(args.query_source),
        feature_name=str(args.feature_name),
        feature_key=str(args.feature_key),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        output_dtype=str(args.output_dtype),
        image_cache_capacity=int(args.image_cache_capacity) if int(args.image_cache_capacity) > 0 else None,
        skip_existing=not bool(args.overwrite),
    )
    if str(args.summary_json):
        write_rgb_native_patch_cache_summary(Path(args.summary_json), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
