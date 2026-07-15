"""Train pose-free real-RGB likelihood evidence for frozen top-L candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.candidate_rgb_identity_training import (
    train_independent_rgb_candidate_verifier,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_evidence", required=True)
    parser.add_argument("--availability_evidence", required=True)
    parser.add_argument("--train_rows_csv", required=True)
    parser.add_argument("--validation_rows_csv", required=True)
    parser.add_argument("--test_rows_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--init_measurement_checkpoint", required=True)
    parser.add_argument("--init_independent_checkpoint", default="")
    parser.add_argument("--freeze_for_measurement_validity", action="store_true")
    parser.add_argument("--freeze_for_spatial_density", action="store_true")
    parser.add_argument("--freeze_for_pose_view_mixture", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--group_batch_size", type=int, default=32)
    parser.add_argument("--query_images_per_batch", type=int, default=8)
    parser.add_argument(
        "--appearance_positive_group_fraction", type=float, default=0.5
    )
    parser.add_argument("--eval_group_batch_size", type=int, default=32)
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--head_learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--identity_threshold_px", type=float, default=2.0)
    parser.add_argument("--identity_negative_threshold_px", type=float, default=5.0)
    parser.add_argument("--identity_loss_weight", type=float, default=1.0)
    parser.add_argument("--availability_loss_weight", type=float, default=0.5)
    parser.add_argument("--pair_loss_weight", type=float, default=0.25)
    parser.add_argument("--spatial_loss_weight", type=float, default=0.25)
    parser.add_argument("--spatial_target_sigma_px", type=float, default=0.75)
    parser.add_argument("--measurement_validity_loss_weight", type=float, default=0.5)
    parser.add_argument("--measurement_success_threshold_px", type=float, default=2.0)
    parser.add_argument("--spatial_density_loss_weight", type=float, default=0.0)
    parser.add_argument("--pose_view_mixture_loss_weight", type=float, default=0.0)
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_cache_max_gb", type=float, default=12.0)
    parser.add_argument(
        "--gpu_non_cache_reserve_gb",
        type=float,
        default=14.0,
        help="GPU memory reserved for model parameters, activations, and allocator headroom",
    )
    parser.add_argument(
        "--image_cache_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--prediction_splits", default="validation,test")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prediction_splits = tuple(
        value.strip() for value in str(args.prediction_splits).split(",") if value.strip()
    )
    invalid_splits = set(prediction_splits) - {"train", "validation", "test"}
    if invalid_splits or not prediction_splits:
        raise ValueError(
            f"prediction_splits must contain train/validation/test: {sorted(invalid_splits)}"
        )
    summary = train_independent_rgb_candidate_verifier(
        candidate_evidence=Path(args.candidate_evidence),
        availability_evidence=Path(args.availability_evidence),
        train_rows_csv=Path(args.train_rows_csv),
        validation_rows_csv=Path(args.validation_rows_csv),
        test_rows_csv=Path(args.test_rows_csv),
        image_root=Path(args.image_root),
        init_measurement_checkpoint=Path(args.init_measurement_checkpoint),
        init_independent_checkpoint=(
            None
            if not str(args.init_independent_checkpoint).strip()
            else Path(args.init_independent_checkpoint)
        ),
        freeze_for_measurement_validity=bool(
            args.freeze_for_measurement_validity
        ),
        freeze_for_spatial_density=bool(args.freeze_for_spatial_density),
        freeze_for_pose_view_mixture=bool(
            args.freeze_for_pose_view_mixture
        ),
        output_dir=Path(args.output_dir),
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        steps=int(args.steps),
        group_batch_size=int(args.group_batch_size),
        query_images_per_batch=int(args.query_images_per_batch),
        appearance_positive_group_fraction=float(
            args.appearance_positive_group_fraction
        ),
        eval_group_batch_size=int(args.eval_group_batch_size),
        encoder_learning_rate=float(args.encoder_learning_rate),
        head_learning_rate=float(args.head_learning_rate),
        weight_decay=float(args.weight_decay),
        identity_threshold_px=float(args.identity_threshold_px),
        identity_negative_threshold_px=float(args.identity_negative_threshold_px),
        identity_loss_weight=float(args.identity_loss_weight),
        availability_loss_weight=float(args.availability_loss_weight),
        pair_loss_weight=float(args.pair_loss_weight),
        spatial_loss_weight=float(args.spatial_loss_weight),
        spatial_target_sigma_px=float(args.spatial_target_sigma_px),
        measurement_validity_loss_weight=float(
            args.measurement_validity_loss_weight
        ),
        measurement_success_threshold_px=float(
            args.measurement_success_threshold_px
        ),
        spatial_density_loss_weight=float(args.spatial_density_loss_weight),
        pose_view_mixture_loss_weight=float(
            args.pose_view_mixture_loss_weight
        ),
        max_views=int(args.max_views),
        seed=int(args.seed),
        device=str(args.device),
        image_cache_max_gb=float(args.image_cache_max_gb),
        gpu_non_cache_reserve_gb=float(args.gpu_non_cache_reserve_gb),
        image_cache_dtype=str(args.image_cache_dtype),
        use_amp=not bool(args.no_amp),
        log_every=int(args.log_every),
        prediction_splits=prediction_splits,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
