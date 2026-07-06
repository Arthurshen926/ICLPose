"""Apply two-stage coarse-to-fine RGB measurements to a RADIO/MATCHA match table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_coarse_to_fine_fusion import (
    apply_coarse_to_fine_rgb_patch_measurements_to_match_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match_table_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--coarse_checkpoint", required=True)
    parser.add_argument("--fine_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--query_image_width", type=int, required=True)
    parser.add_argument("--query_image_height", type=int, required=True)
    parser.add_argument("--render_image_width", type=int, required=True)
    parser.add_argument("--render_image_height", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--coarse_recenter_head", default="mode", choices=("mode", "likelihood"))
    parser.add_argument("--coarse_prior_scale_key", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = apply_coarse_to_fine_rgb_patch_measurements_to_match_table(
        match_table_csv=Path(args.match_table_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv),
        image_root=Path(args.image_root),
        coarse_checkpoint=Path(args.coarse_checkpoint),
        fine_checkpoint=Path(args.fine_checkpoint),
        output_dir=Path(args.output_dir),
        query_image_width=int(args.query_image_width),
        query_image_height=int(args.query_image_height),
        render_image_width=int(args.render_image_width),
        render_image_height=int(args.render_image_height),
        batch_size=max(1, int(args.batch_size)),
        device=str(args.device),
        base_dir=Path(args.base_dir),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        coarse_recenter_head=str(args.coarse_recenter_head),
        coarse_prior_scale_key=str(args.coarse_prior_scale_key),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
