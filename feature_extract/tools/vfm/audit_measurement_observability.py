"""Audit whether render-query local patches contain usable measurement evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.observability_audit import export_measurement_observability_audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--render_cache_manifest_csv", default="")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--query_image_width", type=int, default=0)
    parser.add_argument("--query_image_height", type=int, default=0)
    parser.add_argument("--render_image_width", type=int, default=0)
    parser.add_argument("--render_image_height", type=int, default=0)
    parser.add_argument("--search_radius_px", type=float, default=32.0)
    parser.add_argument("--context_radius_px", type=float, default=16.0)
    parser.add_argument("--step_px", type=float, default=2.0)
    parser.add_argument("--input_mode", default="norm_graygrad", choices=("rgb", "graygrad", "norm_graygrad", "rgb_graygrad"))
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--template_scale_factors", nargs="+", type=float, default=[0.75, 1.0, 1.25])
    parser.add_argument("--query_source", default="real", choices=("real", "render", "render_augmented", "real_pair"))
    parser.add_argument("--support_patch_warp", default="none", choices=("none", "local_affine", "local_homography"))
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--cache_images_on_device", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = export_measurement_observability_audit(
        rows_csv=Path(args.rows_csv),
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv) if str(args.render_cache_manifest_csv) else None,
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        query_image_width=int(args.query_image_width) if int(args.query_image_width) > 0 else None,
        query_image_height=int(args.query_image_height) if int(args.query_image_height) > 0 else None,
        render_image_width=int(args.render_image_width) if int(args.render_image_width) > 0 else None,
        render_image_height=int(args.render_image_height) if int(args.render_image_height) > 0 else None,
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        step_px=float(args.step_px),
        input_mode=str(args.input_mode),
        temperature=float(args.temperature),
        template_scale_factors=[float(value) for value in args.template_scale_factors],
        query_source=str(args.query_source),
        support_patch_warp=str(args.support_patch_warp),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        batch_size=max(1, int(args.batch_size)),
        device=str(args.device),
        base_dir=Path(args.base_dir),
        cache_images_on_device=bool(args.cache_images_on_device),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
