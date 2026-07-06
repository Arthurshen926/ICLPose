"""Apply RGB template-to-search measurements to a RADIO/MATCHA match table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_match_table_fusion import (
    apply_rgb_patch_measurements_to_match_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, default=0, help="Legacy shortcut used for both query and render dimensions.")
    parser.add_argument("--image_height", type=int, default=0, help="Legacy shortcut used for both query and render dimensions.")
    parser.add_argument("--query_image_width", type=int, default=0)
    parser.add_argument("--query_image_height", type=int, default=0)
    parser.add_argument("--render_image_width", type=int, default=0)
    parser.add_argument("--render_image_height", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument(
        "--prediction_head",
        default="center",
        choices=("center", "noop", "likelihood_mode", "likelihood_mean", "likelihood", "mode", "direct"),
    )
    parser.add_argument("--prior_scale_key", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_rgb_patch_measurements_to_match_table(
        match_table_csv=Path(args.match_table_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv),
        image_root=Path(args.image_root),
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width) if int(args.image_width) > 0 else None,
        image_height=int(args.image_height) if int(args.image_height) > 0 else None,
        query_image_width=int(args.query_image_width) if int(args.query_image_width) > 0 else None,
        query_image_height=int(args.query_image_height) if int(args.query_image_height) > 0 else None,
        render_image_width=int(args.render_image_width) if int(args.render_image_width) > 0 else None,
        render_image_height=int(args.render_image_height) if int(args.render_image_height) > 0 else None,
        batch_size=max(1, int(args.batch_size)),
        device=str(args.device),
        base_dir=Path(args.base_dir),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        prediction_head=str(args.prediction_head),
        prior_scale_key=str(args.prior_scale_key),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
