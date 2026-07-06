"""Run D0 feature residual-flow decodability probe from feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.d0_probe import parse_feature_specs, run_d0_probe


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_specs", required=True, help="name:key or name:query_key:render_key, comma-separated")
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--search_radius_px", type=float, default=32.0)
    parser.add_argument("--step_px", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--query_feature_cache_dir", default="")
    parser.add_argument("--render_feature_cache_dir", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--feature_cache_capacity", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_d0_probe(
        rows_csv=Path(args.rows_csv),
        output_dir=Path(args.output_dir),
        feature_specs=parse_feature_specs(str(args.feature_specs)),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        search_radius_px=float(args.search_radius_px),
        step_px=float(args.step_px),
        temperature=float(args.temperature),
        query_feature_cache_dir=Path(args.query_feature_cache_dir) if str(args.query_feature_cache_dir) else None,
        render_feature_cache_dir=Path(args.render_feature_cache_dir) if str(args.render_feature_cache_dir) else None,
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        feature_cache_capacity=int(args.feature_cache_capacity) if int(args.feature_cache_capacity) > 0 else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
