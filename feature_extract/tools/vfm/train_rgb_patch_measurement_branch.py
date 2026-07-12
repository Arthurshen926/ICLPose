"""Train measurement-v1 high-resolution RGB patch branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feature_extract.vfm.measurement_v1.rgb_patch_training import train_rgb_patch_measurement_branch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--val_rows_csv", default="")
    parser.add_argument("--render_cache_manifest_csv", default="")
    parser.add_argument("--val_render_cache_manifest_csv", default="")
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--query_image_width", type=int, default=0)
    parser.add_argument("--query_image_height", type=int, default=0)
    parser.add_argument("--render_image_width", type=int, default=0)
    parser.add_argument("--render_image_height", type=int, default=0)
    parser.add_argument("--search_radius_px", type=float, default=2.0)
    parser.add_argument("--context_radius_px", type=float, default=8.0)
    parser.add_argument("--step_px", type=float, default=0.25)
    parser.add_argument("--coarse_search_radius_px", type=float, default=-1.0)
    parser.add_argument("--coarse_step_px", type=float, default=-1.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--input_mode", default="rgb", choices=("rgb", "rgb_graygrad", "norm_graygrad"))
    parser.add_argument("--encoder_arch", default="simple", choices=("simple", "fpn"))
    parser.add_argument("--template_scale_factors", nargs="+", type=float, default=[1.0])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epe_weight", type=float, default=0.25)
    parser.add_argument("--delta_loss_weight", type=float, default=1.0)
    parser.add_argument("--gated_delta_loss_weight", type=float, default=0.0)
    parser.add_argument("--gate_supervision_loss_weight", type=float, default=0.0)
    parser.add_argument("--gate_center_radius_px", type=float, default=0.5)
    parser.add_argument("--gate_full_radius_px", type=float, default=2.0)
    parser.add_argument(
        "--gate_target_mode",
        default="residual",
        choices=("residual", "utility", "binary_utility"),
    )
    parser.add_argument("--gate_utility_temperature_px", type=float, default=0.25)
    parser.add_argument("--gate_minimum_update_gain_px", type=float, default=0.1)
    parser.add_argument("--gate_positive_weight", type=float, default=1.0)
    parser.add_argument("--gate_low_residual_threshold_px", type=float, default=1.0)
    parser.add_argument("--gate_low_residual_negative_weight", type=float, default=1.0)
    parser.add_argument("--likelihood_loss_weight", type=float, default=0.1)
    parser.add_argument("--coarse_likelihood_loss_weight", type=float, default=0.0)
    parser.add_argument("--dustbin_bce_weight", type=float, default=0.0)
    parser.add_argument("--dustbin_positive_weight", type=float, default=1.0)
    parser.add_argument("--target_heatmap_sigma_px", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_group_key", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_eval_rows", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_images_on_device", action="store_true")
    parser.add_argument(
        "--image_cache_max_gb",
        type=float,
        default=0.0,
        help="Optional shared query/support image-cache byte budget in GiB.",
    )
    parser.add_argument("--base_dir", default=".")
    parser.add_argument("--query_source", default="real", choices=("real", "render", "render_augmented", "real_pair"))
    parser.add_argument("--support_patch_warp", default="none", choices=("none", "local_affine", "local_homography"))
    parser.add_argument("--render_patch_augmentation", default="none", choices=("none", "realistic"))
    parser.add_argument("--hard_negative_fraction", type=float, default=0.0)
    parser.add_argument("--train_dustbin_head_only", action="store_true")
    parser.add_argument("--train_measurement_gate_head_only", action="store_true")
    parser.add_argument("--condition_on_prior_scale", action="store_true")
    parser.add_argument("--prior_scale_key", default="")
    parser.add_argument("--prior_scale_expert_centers_px", nargs="*", type=float, default=[])
    parser.add_argument("--prior_scale_expert_projection", action="store_true")
    parser.add_argument("--prior_scale_expert_gate", default="soft", choices=("soft", "hard"))
    parser.add_argument("--target_x_key", default="query_gt_x")
    parser.add_argument("--target_y_key", default="query_gt_y")
    parser.add_argument("--loss_weight_key", default="")
    parser.add_argument(
        "--min_loss_weight",
        type=float,
        default=-1.0,
        help="Optionally discard rows below this loss weight before sampling and image loading.",
    )
    parser.add_argument("--target_dustbin_filter", default="all", choices=("all", "valid", "dustbin"))
    parser.add_argument("--baseline_epe_min_px", type=float, default=-1.0)
    parser.add_argument("--baseline_epe_max_px", type=float, default=-1.0)
    parser.add_argument("--residual_balanced_sampling", action="store_true")
    parser.add_argument(
        "--residual_sampling_bins_px",
        nargs="*",
        type=float,
        default=[],
        help="Residual-bin lower edges used by balanced batch sampling.",
    )
    parser.add_argument("--data_parallel_device_ids", nargs="*", type=int, default=[])
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--train_stage", default="")
    parser.add_argument("--gate_median_epe_px", type=float, default=0.5)
    parser.add_argument("--gate_improve_ratio", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_rgb_patch_measurement_branch(
        rows_csv=Path(args.rows_csv),
        val_rows_csv=Path(args.val_rows_csv) if str(args.val_rows_csv) else None,
        render_cache_manifest_csv=Path(args.render_cache_manifest_csv) if str(args.render_cache_manifest_csv) else None,
        val_render_cache_manifest_csv=Path(args.val_render_cache_manifest_csv) if str(args.val_render_cache_manifest_csv) else None,
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
        coarse_search_radius_px=float(args.coarse_search_radius_px) if float(args.coarse_search_radius_px) >= 0.0 else None,
        coarse_step_px=float(args.coarse_step_px) if float(args.coarse_step_px) > 0.0 else None,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim) if int(args.hidden_dim) > 0 else None,
        input_mode=str(args.input_mode),
        encoder_arch=str(args.encoder_arch),
        template_scale_factors=[float(value) for value in args.template_scale_factors],
        lr=float(args.lr),
        epe_weight=float(args.epe_weight),
        delta_loss_weight=float(args.delta_loss_weight),
        gated_delta_loss_weight=float(args.gated_delta_loss_weight),
        gate_supervision_loss_weight=float(args.gate_supervision_loss_weight),
        gate_center_radius_px=float(args.gate_center_radius_px),
        gate_full_radius_px=float(args.gate_full_radius_px),
        gate_target_mode=str(args.gate_target_mode),
        gate_utility_temperature_px=float(args.gate_utility_temperature_px),
        gate_minimum_update_gain_px=float(args.gate_minimum_update_gain_px),
        gate_positive_weight=float(args.gate_positive_weight),
        gate_low_residual_threshold_px=float(args.gate_low_residual_threshold_px),
        gate_low_residual_negative_weight=float(args.gate_low_residual_negative_weight),
        likelihood_loss_weight=float(args.likelihood_loss_weight),
        coarse_likelihood_loss_weight=float(args.coarse_likelihood_loss_weight),
        dustbin_bce_weight=float(args.dustbin_bce_weight),
        dustbin_positive_weight=float(args.dustbin_positive_weight),
        target_heatmap_sigma_px=float(args.target_heatmap_sigma_px),
        val_fraction=float(args.val_fraction),
        val_group_key=str(args.val_group_key),
        max_rows=int(args.max_rows) if int(args.max_rows) > 0 else None,
        max_eval_rows=int(args.max_eval_rows) if int(args.max_eval_rows) > 0 else None,
        eval_batch_size=int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else None,
        seed=int(args.seed),
        device=str(args.device),
        cache_images_on_device=bool(args.cache_images_on_device),
        base_dir=Path(args.base_dir),
        query_source=str(args.query_source),
        support_patch_warp=str(args.support_patch_warp),
        render_patch_augmentation=str(args.render_patch_augmentation),
        hard_negative_fraction=float(args.hard_negative_fraction),
        train_dustbin_head_only=bool(args.train_dustbin_head_only),
        train_measurement_gate_head_only=bool(args.train_measurement_gate_head_only),
        condition_on_prior_scale=bool(args.condition_on_prior_scale),
        prior_scale_key=str(args.prior_scale_key),
        prior_scale_expert_centers_px=[float(value) for value in args.prior_scale_expert_centers_px],
        prior_scale_expert_projection=bool(args.prior_scale_expert_projection),
        prior_scale_expert_gate=str(args.prior_scale_expert_gate),
        target_x_key=str(args.target_x_key),
        target_y_key=str(args.target_y_key),
        loss_weight_key=str(args.loss_weight_key),
        min_loss_weight=float(args.min_loss_weight) if float(args.min_loss_weight) >= 0.0 else None,
        target_dustbin_filter=str(args.target_dustbin_filter),
        baseline_epe_min_px=float(args.baseline_epe_min_px) if float(args.baseline_epe_min_px) >= 0.0 else None,
        baseline_epe_max_px=float(args.baseline_epe_max_px) if float(args.baseline_epe_max_px) >= 0.0 else None,
        residual_balanced_sampling=bool(args.residual_balanced_sampling),
        residual_sampling_bins_px=[float(value) for value in args.residual_sampling_bins_px],
        data_parallel_device_ids=[int(value) for value in args.data_parallel_device_ids],
        image_cache_max_gb=float(args.image_cache_max_gb) if float(args.image_cache_max_gb) > 0.0 else None,
        init_checkpoint=Path(args.init_checkpoint) if str(args.init_checkpoint) else None,
        train_stage=str(args.train_stage),
        gate_median_epe_px=float(args.gate_median_epe_px),
        gate_improve_ratio=float(args.gate_improve_ratio),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
