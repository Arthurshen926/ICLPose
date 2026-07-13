"""Export measurement-v1 RGB patch diagnostic rows and heatmap visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_diagnostics import export_rgb_patch_diagnostics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", default="")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--query_source", default="real", choices=("real", "render", "render_augmented", "real_pair"))
    parser.add_argument("--support_patch_warp", default="none", choices=("none", "local_affine", "local_homography"))
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--visualize_limit", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_images_on_device", action="store_true")
    parser.add_argument(
        "--image_cache_max_gb",
        type=float,
        default=0.0,
        help="Optional shared query/support image-cache byte budget in GiB.",
    )
    parser.add_argument("--prior_scale_key", default="")
    parser.add_argument(
        "--target_dustbin_filter",
        default="all",
        choices=("all", "valid", "dustbin"),
    )
    parser.add_argument("--export_full_likelihood", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_rgb_patch_diagnostics(
        rows_csv=Path(args.rows_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv) if str(args.render_cache_manifest_csv) else None,
        image_root=Path(args.image_root),
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        query_source=str(args.query_source),
        support_patch_warp=str(args.support_patch_warp),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        visualize_limit=int(args.visualize_limit),
        batch_size=max(1, int(args.batch_size)),
        cache_images_on_device=bool(args.cache_images_on_device),
        image_cache_max_gb=float(args.image_cache_max_gb) if float(args.image_cache_max_gb) > 0.0 else None,
        prior_scale_key=str(args.prior_scale_key),
        target_dustbin_filter=str(args.target_dustbin_filter),
        export_full_likelihood=bool(args.export_full_likelihood),
        device=str(args.device),
        base_dir=Path(args.base_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
