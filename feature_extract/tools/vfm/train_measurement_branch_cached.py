"""Train a measurement-v1 cache-backed correlation branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.measurement_training import train_cached_measurement_branch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_name", default="radio_dual")
    parser.add_argument("--feature_key", default="radio_dual")
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--search_radius_px", type=float, default=32.0)
    parser.add_argument("--context_radius_px", type=float, default=0.0)
    parser.add_argument("--step_px", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--output_dim", type=int, default=64)
    parser.add_argument(
        "--projection_type",
        default="linear1x1",
        choices=("linear1x1", "conv3", "texture_rgb", "texture_rgb_graygrad", "texture_norm_graygrad"),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_group_key", default="")
    parser.add_argument("--max_eval_rows", type=int, default=512)
    parser.add_argument("--feature_cache_capacity", type=int, default=0)
    parser.add_argument("--project_full_feature_map", action="store_true")
    parser.add_argument("--target_x_key", default="query_gt_x")
    parser.add_argument("--target_y_key", default="query_gt_y")
    parser.add_argument("--sample_weight_key", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_cached_measurement_branch(
        rows_csv=Path(args.rows_csv),
        output_dir=Path(args.output_dir),
        feature_name=str(args.feature_name),
        feature_key=str(args.feature_key),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        search_radius_px=float(args.search_radius_px),
        context_radius_px=float(args.context_radius_px),
        step_px=float(args.step_px),
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        output_dim=int(args.output_dim),
        projection_type=str(args.projection_type),
        lr=float(args.lr),
        temperature=float(args.temperature),
        device=str(args.device),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
        val_group_key=str(args.val_group_key),
        max_eval_rows=int(args.max_eval_rows) if int(args.max_eval_rows) > 0 else None,
        feature_cache_capacity=int(args.feature_cache_capacity) if int(args.feature_cache_capacity) > 0 else None,
        crop_before_projection=not bool(args.project_full_feature_map),
        target_x_key=str(args.target_x_key),
        target_y_key=str(args.target_y_key),
        sample_weight_key=str(args.sample_weight_key),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
