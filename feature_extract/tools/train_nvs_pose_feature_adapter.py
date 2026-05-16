#!/usr/bin/env python3
"""Train NVS-supervised pose-conditioned localization feature adapters.

This is the clean Stage-1 entry point for CPR feature adaptation.  It freezes
the query student and 3DGS/DCFF geometry, then trains lightweight query/render
feature adapters with dense GT-pose alignment and candidate-level pose ranking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from feature_extract.students.pose_energy_net import (  # noqa: E402
    PairConditionedLocalMatcher,
    PoseEnergyNet,
    PoseFeatureDomainAdapter,
    pose_costs_and_residual_targets,
    pose_energy_factorized_selection,
    pose_energy_losses,
    pose_energy_selection_scores,
)
from feature_extract.pose_observability import feature_pose_fisher_stats  # noqa: E402
from feature_extract.tools.eval_cpr_buckets import build_model_and_data, map_pose_gt_for_batch  # noqa: E402
from feature_extract.tools.train_pose_energy import (  # noqa: E402
    _good_bad_auc_rows,
    _set_trainable,
    _spearman_rows,
    apply_config_defaults as apply_pose_energy_defaults,
    apply_pose_feature_adapter,
    build_pose_feature_adapter,
    candidate_center_pose,
    maybe_replace_with_synthetic_queries,
)
from feature_extract.train_impl import (  # noqa: E402
    FINE_CANDIDATE_SELECTOR_CENTER_DELTA_VECTOR_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_UNCERTAINTY_FEATURE_NAMES,
    FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES,
    build_local_pose_lattice_candidates,
    camera_centers_from_w2c,
    candidate_quality_features_from_batch,
    fine_candidate_selector_features,
    load_config,
    move_batch_to_device,
    pose_error_tensors,
    sample_feature_at_xy,
    set_seed,
    sparse_teacher_correspondence_loss,
    sparse_teacher_local_patch_loss,
    _scale_xy_tensor,
)
from pose_refine import apply_pose_delta  # noqa: E402
from pose_refine.utils.lie_algebra import pose_inverse, se3_log  # noqa: E402


def apply_pose_energy_residual_update(
    pose_w2c: torch.Tensor,
    residual_delta: torch.Tensor,
    *,
    update_scale: float = 1.0,
) -> torch.Tensor:
    return apply_pose_delta(pose_w2c.float(), residual_delta.float() * float(update_scale))


def pose_energy_factorized_selection_metrics(
    outputs: Dict[str, torch.Tensor],
    candidate_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    rot_cost_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Measure the composed pose selected by translation and rotation heads."""
    selected_pose, trans_idx, rot_idx = pose_energy_factorized_selection(
        outputs,
        candidate_pose,
        valid_mask=valid_mask,
    )
    device = selected_pose.device
    selected_cost, _selected_residual, selected_trans, selected_rot = pose_costs_and_residual_targets(
        selected_pose.float()[:, None],
        pose_gt.float().to(device),
        valid_mask=torch.ones((selected_pose.shape[0], 1), device=device, dtype=torch.bool),
        rot_cost_weight=float(rot_cost_weight),
    )
    pose_cost, _residual, trans_err, rot_err = pose_costs_and_residual_targets(
        candidate_pose.float().to(device),
        pose_gt.float().to(device),
        valid_mask=valid_mask.to(device).bool() if valid_mask is not None else None,
        rot_cost_weight=float(rot_cost_weight),
    )
    trans_target = trans_err.argmin(dim=1)
    rot_target = rot_err.argmin(dim=1)
    joint_oracle = pose_cost.min(dim=1).values
    return {
        "factorized_pred_cost_m": selected_cost[:, 0],
        "factorized_trans_err_m": selected_trans[:, 0],
        "factorized_rot_err_rad": selected_rot[:, 0],
        "factorized_oracle_cost_m": joint_oracle,
        "factorized_oracle_gap_m": selected_cost[:, 0] - joint_oracle,
        "factorized_translation_idx": trans_idx,
        "factorized_rotation_idx": rot_idx,
        "factorized_translation_target_idx": trans_target,
        "factorized_rotation_target_idx": rot_target,
        "factorized_translation_top1_acc": (trans_idx == trans_target).float().mean(),
        "factorized_rotation_top1_acc": (rot_idx == rot_target).float().mean(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--map-checkpoint", default=None)
    parser.add_argument("--resume-adapter", default=None)
    parser.add_argument("--resume-adapter-strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-split", choices=("train", "val"), default="train")
    parser.add_argument("--eval-split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--candidate-render-batch-size", type=int, default=None)
    parser.add_argument("--pose-observability-diagnostic-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--best-metric", default=None)
    parser.add_argument("--best-metric-mode", choices=("min", "max"), default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-skipped-optimizer-steps", type=int, default=None)
    parser.add_argument("--train-projector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-model-prefixes", default=None)
    parser.add_argument("--train-pose-feature-adapter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-pose-feature-adapter-domains",
        choices=("all", "query", "render", "none"),
        default=None,
    )
    parser.add_argument("--train-pose-energy-net", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-pair-matcher", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-hidden-dim", type=int, default=None)
    parser.add_argument("--pose-feature-adapter-residual-scale", type=float, default=None)
    parser.add_argument("--pose-feature-adapter-zero-init", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-l2-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-uncertainty-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-rgb-context-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-rgb-context-channels", type=int, default=None)
    parser.add_argument("--pose-feature-adapter-texture-branch-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-texture-branch-hidden-dim", type=int, default=None)
    parser.add_argument("--pose-feature-adapter-texture-branch-scale", type=float, default=None)
    parser.add_argument("--pose-feature-adapter-texture-branch-zero-init", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-feature-adapter-texture-fusion-mode", choices=("residual", "replace"), default=None)
    parser.add_argument("--pose-feature-adapter-base-anchor-weight", type=float, default=None)
    parser.add_argument("--pair-matcher-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair-matcher-weight", type=float, default=None)
    parser.add_argument("--pair-matcher-hidden-dim", type=int, default=None)
    parser.add_argument("--pair-matcher-radius", type=int, default=None)
    parser.add_argument("--pair-matcher-temperature", type=float, default=None)
    parser.add_argument("--pair-matcher-min-confidence", type=float, default=None)
    parser.add_argument("--pair-matcher-min-points", type=int, default=None)
    parser.add_argument("--pair-matcher-positive-weight", type=float, default=None)
    parser.add_argument("--pair-matcher-margin-weight", type=float, default=None)
    parser.add_argument("--pair-matcher-margin", type=float, default=None)
    parser.add_argument("--pair-matcher-base-dot-weight", type=float, default=None)
    parser.add_argument("--pair-matcher-zero-init-residual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pair-matcher-score-stride", type=int, default=None)
    parser.add_argument("--pair-matcher-score-chunk-points", type=int, default=None)
    parser.add_argument("--pair-matcher-score-candidate-chunk-size", type=int, default=None)
    parser.add_argument("--pair-matcher-score-offset-chunk-size", type=int, default=None)
    parser.add_argument(
        "--pair-matcher-candidate-score-mode",
        choices=("center_logprob_margin", "center_margin"),
        default=None,
    )
    parser.add_argument("--query-fine-key", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--lattice-trans-cm", default=None)
    parser.add_argument("--lattice-rot-deg", default=None)
    parser.add_argument("--lattice-direction-mode", choices=("axis", "cube"), default=None)
    parser.add_argument("--combine-trans-rot", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--limit-strategy", default=None)
    parser.add_argument(
        "--candidate-bank-mode",
        choices=(
            "lattice",
            "cache",
            "balanced",
            "direction_balanced",
            "adaptive",
            "adaptive_balanced",
            "bucket_adaptive",
            "adaptive_direction_balanced",
        ),
        default=None,
    )
    parser.add_argument("--direction-fractions", default=None)
    parser.add_argument("--candidate-center-mode", choices=("gt", "noisy_init", "batch"), default=None)
    parser.add_argument("--candidate-center-noise-mode", choices=("uniform", "fixed"), default=None)
    parser.add_argument("--candidate-center-trans-cm", type=float, default=None)
    parser.add_argument("--candidate-center-rot-deg", type=float, default=None)
    parser.add_argument("--candidate-center-buckets", default=None)
    parser.add_argument("--include-identity-candidate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-append-gt-candidate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-append-gt-candidate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rot-cost-weight", type=float, default=None)
    parser.add_argument("--rank-temperature-m", type=float, default=None)
    parser.add_argument("--rank-ce-weight", type=float, default=None)
    parser.add_argument("--rank-pairwise-weight", type=float, default=None)
    parser.add_argument("--rank-pairwise-min-gap-m", type=float, default=None)
    parser.add_argument("--rank-pairwise-logit-margin", type=float, default=None)
    parser.add_argument("--score-correction-cosine-weight", type=float, default=None)
    parser.add_argument("--score-correction-cosine-temperature", type=float, default=None)
    parser.add_argument("--score-correction-cosine-min-cos", type=float, default=None)
    parser.add_argument("--score-pose-improvement-weight", type=float, default=None)
    parser.add_argument("--score-pose-improvement-temperature-m", type=float, default=None)
    parser.add_argument("--score-pose-improvement-min-improvement-m", type=float, default=None)
    parser.add_argument("--score-anti-identity-weight", type=float, default=None)
    parser.add_argument("--score-anti-identity-min-gap-m", type=float, default=None)
    parser.add_argument("--score-anti-identity-logit-margin", type=float, default=None)
    parser.add_argument("--score-anti-identity-index", type=int, default=None)
    parser.add_argument("--observability-contrast-weight", type=float, default=None)
    parser.add_argument("--observability-contrast-margin", type=float, default=None)
    parser.add_argument("--observability-negative-min-cost-m", type=float, default=None)
    parser.add_argument("--candidate-observability-weight", type=float, default=None)
    parser.add_argument("--candidate-observability-margin", type=float, default=None)
    parser.add_argument("--candidate-observability-min-cost-gap-m", type=float, default=None)
    parser.add_argument(
        "--observability-contrast-score-source",
        choices=("dense", "selection_score"),
        default=None,
    )
    parser.add_argument("--candidate-teacher-quality-weight", type=float, default=None)
    parser.add_argument("--candidate-teacher-quality-target-mode", default=None)
    parser.add_argument("--candidate-teacher-quality-temperature", type=float, default=None)
    parser.add_argument("--candidate-teacher-quality-pairwise-weight", type=float, default=None)
    parser.add_argument("--candidate-teacher-quality-pairwise-min-gap", type=float, default=None)
    parser.add_argument("--candidate-teacher-quality-logit-margin", type=float, default=None)
    parser.add_argument("--candidate-teacher-quality-start-step", type=int, default=None)
    parser.add_argument("--candidate-teacher-quality-warmup-steps", type=int, default=None)
    parser.add_argument("--align-weight", type=float, default=None)
    parser.add_argument("--align-margin", type=float, default=None)
    parser.add_argument("--warp-align-weight", type=float, default=None)
    parser.add_argument("--warp-depth-tolerance-m", type=float, default=None)
    parser.add_argument("--warp-depth-tolerance-rel", type=float, default=None)
    parser.add_argument("--drift-weight", type=float, default=None)
    parser.add_argument("--variance-weight", type=float, default=None)
    parser.add_argument("--variance-min-std", type=float, default=None)
    parser.add_argument("--score-temperature", type=float, default=None)
    parser.add_argument("--score-use-uncertainty", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--score-feature-hw", default=None)
    parser.add_argument(
        "--score-mode",
        choices=(
            "same_pixel",
            "local_zero_offset",
            "local_zero_only",
            "local_neg_expected_offset",
            "local_neg_peak_offset",
            "pair_matcher_local",
        ),
        default=None,
    )
    parser.add_argument("--local-corr-radius", type=int, default=None)
    parser.add_argument("--local-corr-temperature", type=float, default=None)
    parser.add_argument("--local-zero-peak-gap-weight", type=float, default=None)
    parser.add_argument("--local-zero-offset-weight", type=float, default=None)
    parser.add_argument("--local-flow-nce-weight", type=float, default=None)
    parser.add_argument(
        "--local-flow-nce-candidate-mode",
        choices=("all", "best", "best_only", "best_and_hard_negative", "hard_negative", "hard"),
        default=None,
    )
    parser.add_argument("--local-flow-nce-hard-negative-min-cost-gap-m", type=float, default=None)
    parser.add_argument("--teacher-corr-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-temperature", type=float, default=None)
    parser.add_argument("--teacher-corr-min-confidence", type=float, default=None)
    parser.add_argument("--teacher-corr-min-points", type=int, default=None)
    parser.add_argument("--teacher-corr-negative-exclusion-px", type=float, default=None)
    parser.add_argument("--teacher-corr-positive-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-margin-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-margin", type=float, default=None)
    parser.add_argument("--teacher-corr-local-patch-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-local-patch-radius", type=int, default=None)
    parser.add_argument("--teacher-corr-local-patch-temperature", type=float, default=None)
    parser.add_argument("--teacher-corr-local-patch-min-points", type=int, default=None)
    parser.add_argument("--teacher-corr-local-patch-positive-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-local-patch-margin-weight", type=float, default=None)
    parser.add_argument("--teacher-corr-local-patch-margin", type=float, default=None)
    parser.add_argument("--pose-energy-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-weight", type=float, default=None)
    parser.add_argument("--pose-energy-hidden-dim", type=int, default=None)
    parser.add_argument("--pose-energy-map-channels", type=int, default=None)
    parser.add_argument("--pose-energy-grid-size", type=int, default=None)
    parser.add_argument("--pose-energy-context-layers", type=int, default=None)
    parser.add_argument("--pose-energy-context-heads", type=int, default=None)
    parser.add_argument("--pose-energy-factorized-heads", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-translation-weight", type=float, default=None)
    parser.add_argument("--pose-energy-rotation-weight", type=float, default=None)
    parser.add_argument("--pose-energy-joint-weight", type=float, default=None)
    parser.add_argument("--pose-energy-confidence-weight", type=float, default=None)
    parser.add_argument("--pose-energy-confidence-temperature-m", type=float, default=None)
    parser.add_argument("--pose-energy-selection-confidence-weight", type=float, default=None)
    parser.add_argument("--pose-energy-selection-residual-norm-weight", type=float, default=None)
    parser.add_argument("--pose-energy-base-score-prior-weight", type=float, default=None)
    parser.add_argument(
        "--pose-energy-base-score-prior-mode",
        choices=("raw", "centered", "zscore"),
        default=None,
    )
    parser.add_argument("--pose-energy-zero-init-heads", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-zero-init-residual-head", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-score-preprocess", default=None)
    parser.add_argument("--pose-energy-score-highpass-kernel", type=int, default=None)
    parser.add_argument("--pose-energy-score-map-mode", default=None)
    parser.add_argument(
        "--pose-energy-score-source",
        choices=("local_corr", "pair_matcher_heatmap"),
        default=None,
    )
    parser.add_argument("--pose-energy-use-candidate-delta", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-use-delta-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-use-center-delta-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-use-rgb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-use-uncertainty", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-target-temperature-m", type=float, default=None)
    parser.add_argument("--pose-energy-hard-ce-weight", type=float, default=None)
    parser.add_argument("--pose-energy-component-hard-ce-weight", type=float, default=None)
    parser.add_argument("--pose-energy-residual-weight", type=float, default=None)
    parser.add_argument("--pose-energy-residual-target-mode", default=None)
    parser.add_argument("--pose-energy-residual-soft-topk-temperature-m", type=float, default=None)
    parser.add_argument("--pose-energy-improve-weight", type=float, default=None)
    parser.add_argument("--pose-energy-improve-margin-m", type=float, default=None)
    parser.add_argument("--pose-energy-monotonicity-weight", type=float, default=None)
    parser.add_argument("--pose-energy-monotonicity-margin", type=float, default=None)
    parser.add_argument("--pose-energy-monotonicity-improved-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose-energy-update-scale", type=float, default=None)
    parser.add_argument("--pose-energy-residual-trans-scale-m", type=float, default=None)
    parser.add_argument("--pose-energy-residual-rot-scale-deg", type=float, default=None)
    parser.add_argument("--pose-energy-anti-identity-weight", type=float, default=None)
    parser.add_argument("--pose-energy-anti-identity-margin", type=float, default=None)
    parser.add_argument("--pose-energy-anti-identity-min-gap-m", type=float, default=None)
    parser.add_argument("--pose-energy-identity-index", type=int, default=None)
    parser.add_argument("--pose-energy-pairwise-rank-weight", type=float, default=None)
    parser.add_argument("--pose-energy-pairwise-rank-min-gap-m", type=float, default=None)
    parser.add_argument("--pose-energy-pairwise-rank-logit-margin", type=float, default=None)
    parser.add_argument("--pose-energy-direction-weight", type=float, default=None)
    parser.add_argument("--pose-energy-direction-min-cos-gap", type=float, default=None)
    parser.add_argument("--pose-energy-direction-logit-margin", type=float, default=None)
    parser.add_argument("--pose-energy-correction-cosine-weight", type=float, default=None)
    parser.add_argument("--pose-energy-correction-cosine-temperature", type=float, default=None)
    parser.add_argument("--pose-energy-correction-cosine-min-cos", type=float, default=None)
    parser.add_argument("--pose-energy-select-for-metric", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--synthetic-ratio", type=float, default=None)
    parser.add_argument("--eval-synthetic-ratio", type=float, default=None)
    parser.add_argument("--synthetic-trans-cm", type=float, default=None)
    parser.add_argument("--synthetic-rot-deg", type=float, default=None)
    parser.add_argument("--auc-good-m", type=float, default=None)
    parser.add_argument("--auc-bad-m", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260512)
    return parser.parse_args()


def apply_config_defaults(args: argparse.Namespace, cfg: Dict) -> argparse.Namespace:
    cli_values = dict(vars(args))
    args = apply_pose_energy_defaults(args, cfg)
    train_cfg = dict(cfg.get("training", {}) or {})
    training_defaults = {
        "batch_size": 8,
        "num_workers": 0,
        "eval_max_samples": 64,
        "max_steps": 500,
        "eval_every": 50,
        "save_every": 250,
        "lr": 2.0e-4,
        "weight_decay": 1.0e-5,
        "grad_clip": 1.0,
        "max_skipped_optimizer_steps": 5,
    }
    for key, fallback in training_defaults.items():
        if cli_values.get(key, None) is not None:
            continue
        if getattr(args, key, None) is None:
            setattr(args, key, train_cfg.get(key, fallback))
    nvs_cfg = dict(cfg.get("nvs_pose_feature_adapter", {}) or {})
    defaults = {
        "rank_temperature_m": 0.05,
        "rank_ce_weight": 1.0,
        "rank_pairwise_weight": 0.5,
        "rank_pairwise_min_gap_m": 0.03,
        "rank_pairwise_logit_margin": 0.1,
        "score_correction_cosine_weight": 0.0,
        "score_correction_cosine_temperature": 0.1,
        "score_correction_cosine_min_cos": 0.0,
        "score_pose_improvement_weight": 0.0,
        "score_pose_improvement_temperature_m": 0.05,
        "score_pose_improvement_min_improvement_m": 0.0,
        "score_anti_identity_weight": 0.0,
        "score_anti_identity_min_gap_m": 0.03,
        "score_anti_identity_logit_margin": 0.5,
        "score_anti_identity_index": 0,
        "observability_contrast_weight": 0.0,
        "observability_contrast_margin": 0.05,
        "observability_negative_min_cost_m": 0.05,
        "candidate_observability_weight": 0.0,
        "candidate_observability_margin": 0.05,
        "candidate_observability_min_cost_gap_m": 0.05,
        "observability_contrast_score_source": "dense",
        "candidate_teacher_quality_weight": 0.0,
        "candidate_teacher_quality_target_mode": "pnp_composite",
        "candidate_teacher_quality_temperature": 0.5,
        "candidate_teacher_quality_pairwise_weight": 0.1,
        "candidate_teacher_quality_pairwise_min_gap": 0.25,
        "candidate_teacher_quality_logit_margin": 0.5,
        "candidate_teacher_quality_start_step": 0,
        "candidate_teacher_quality_warmup_steps": 0,
        "candidate_bank_mode": "lattice",
        "topk": 64,
        "lattice_trans_cm": "0,2,5,10,25,50",
        "lattice_rot_deg": "0,0.5,1,2,5,10",
        "lattice_direction_mode": "cube",
        "combine_trans_rot": False,
        "limit_strategy": "uniform",
        "candidate_center_buckets": None,
        "direction_fractions": "0.75,0.5,0.25",
        "include_identity_candidate": False,
        "train_append_gt_candidate": True,
        "eval_append_gt_candidate": False,
        "align_weight": 2.0,
        "align_margin": 0.0,
        "warp_align_weight": 1.0,
        "warp_depth_tolerance_m": 0.08,
        "warp_depth_tolerance_rel": 0.08,
        "drift_weight": 0.05,
        "variance_weight": 0.02,
        "variance_min_std": 0.03,
        "score_temperature": 0.1,
        "score_use_uncertainty": False,
        "score_mode": "same_pixel",
        "score_feature_hw": None,
        "local_corr_radius": 3,
        "local_corr_temperature": 0.07,
        "local_zero_peak_gap_weight": 0.5,
        "local_zero_offset_weight": 0.05,
        "local_flow_nce_weight": 0.0,
        "local_flow_nce_candidate_mode": "all",
        "local_flow_nce_hard_negative_min_cost_gap_m": 0.10,
        "teacher_corr_weight": 0.0,
        "teacher_corr_temperature": 0.07,
        "teacher_corr_min_confidence": 0.0,
        "teacher_corr_min_points": 4,
        "teacher_corr_negative_exclusion_px": 0.0,
        "teacher_corr_positive_weight": 0.0,
        "teacher_corr_margin_weight": 0.0,
        "teacher_corr_margin": 0.1,
        "teacher_corr_local_patch_weight": 0.0,
        "teacher_corr_local_patch_radius": 2,
        "teacher_corr_local_patch_temperature": 0.05,
        "teacher_corr_local_patch_min_points": 4,
        "teacher_corr_local_patch_positive_weight": 0.0,
        "teacher_corr_local_patch_margin_weight": 0.0,
        "teacher_corr_local_patch_margin": 0.05,
        "pose_energy_enabled": False,
        "pose_energy_weight": 1.0,
        "pose_energy_hidden_dim": 128,
        "pose_energy_map_channels": 16,
        "pose_energy_grid_size": 4,
        "pose_energy_context_layers": 1,
        "pose_energy_context_heads": 1,
        "pose_energy_factorized_heads": False,
        "pose_energy_translation_weight": 0.0,
        "pose_energy_rotation_weight": 0.0,
        "pose_energy_joint_weight": 0.0,
        "pose_energy_confidence_weight": 0.0,
        "pose_energy_confidence_temperature_m": None,
        "pose_energy_selection_confidence_weight": 0.0,
        "pose_energy_selection_residual_norm_weight": 0.0,
        "pose_energy_base_score_prior_weight": 0.0,
        "pose_energy_base_score_prior_mode": "zscore",
        "pose_energy_zero_init_heads": False,
        "pose_energy_zero_init_residual_head": True,
        "pose_energy_score_preprocess": "spatial_center",
        "pose_energy_score_highpass_kernel": 5,
        "pose_energy_score_map_mode": "peak_offset",
        "pose_energy_score_source": "local_corr",
        "pose_energy_use_candidate_delta": True,
        "pose_energy_use_delta_vector": False,
        "pose_energy_use_center_delta_vector": False,
        "pose_energy_use_rgb": False,
        "pose_energy_use_uncertainty": False,
        "pose_energy_target_temperature_m": 0.05,
        "pose_energy_hard_ce_weight": 0.0,
        "pose_energy_component_hard_ce_weight": 0.0,
        "pose_energy_residual_weight": 0.5,
        "pose_energy_residual_target_mode": "all",
        "pose_energy_residual_soft_topk_temperature_m": 0.05,
        "pose_energy_improve_weight": 0.2,
        "pose_energy_improve_margin_m": 0.0,
        "pose_energy_monotonicity_weight": 0.0,
        "pose_energy_monotonicity_margin": 0.05,
        "pose_energy_monotonicity_improved_only": True,
        "pose_energy_update_scale": 1.0,
        "pose_energy_residual_trans_scale_m": 0.25,
        "pose_energy_residual_rot_scale_deg": 5.0,
        "pose_energy_anti_identity_weight": 0.5,
        "pose_energy_anti_identity_margin": 0.5,
        "pose_energy_anti_identity_min_gap_m": 0.03,
        "pose_energy_identity_index": 0,
        "pose_energy_pairwise_rank_weight": 0.1,
        "pose_energy_pairwise_rank_min_gap_m": 0.03,
        "pose_energy_pairwise_rank_logit_margin": 0.5,
        "pose_energy_direction_weight": 0.0,
        "pose_energy_direction_min_cos_gap": 0.25,
        "pose_energy_direction_logit_margin": 0.5,
        "pose_energy_correction_cosine_weight": 0.0,
        "pose_energy_correction_cosine_temperature": 0.1,
        "pose_energy_correction_cosine_min_cos": 0.0,
        "pose_energy_select_for_metric": False,
        "train_projector": False,
        "train_model_prefixes": "",
        "train_pose_feature_adapter_domains": "all",
        "train_pose_energy_net": True,
        "train_pair_matcher": True,
        "resume_adapter_strict": True,
        "pose_feature_adapter_enabled": True,
        "pose_feature_adapter_hidden_dim": 96,
        "pose_feature_adapter_residual_scale": 0.2,
        "pose_feature_adapter_zero_init": True,
        "pose_feature_adapter_l2_normalize": True,
        "pose_feature_adapter_uncertainty_enabled": False,
        "pose_feature_adapter_rgb_context_enabled": False,
        "pose_feature_adapter_rgb_context_channels": 8,
        "pose_feature_adapter_texture_branch_enabled": False,
        "pose_feature_adapter_texture_branch_hidden_dim": 32,
        "pose_feature_adapter_texture_branch_scale": 0.25,
        "pose_feature_adapter_texture_branch_zero_init": True,
        "pose_feature_adapter_texture_fusion_mode": "residual",
        "pose_feature_adapter_base_anchor_weight": 1.0,
        "pose_observability_diagnostic_enabled": False,
        "pair_matcher_enabled": False,
        "pair_matcher_weight": 0.0,
        "pair_matcher_hidden_dim": 64,
        "pair_matcher_radius": 3,
        "pair_matcher_temperature": 0.05,
        "pair_matcher_min_confidence": 0.0,
        "pair_matcher_min_points": 4,
        "pair_matcher_positive_weight": 0.0,
        "pair_matcher_margin_weight": 0.0,
        "pair_matcher_margin": 0.05,
        "pair_matcher_base_dot_weight": 1.0,
        "pair_matcher_zero_init_residual": True,
        "pair_matcher_score_stride": 8,
        "pair_matcher_score_chunk_points": 65536,
        "pair_matcher_score_candidate_chunk_size": 0,
        "pair_matcher_score_offset_chunk_size": 0,
        "pair_matcher_candidate_score_mode": "center_logprob_margin",
        "synthetic_ratio": 0.5,
        "synthetic_trans_cm": 50.0,
        "synthetic_rot_deg": 10.0,
        "auc_good_m": 0.05,
        "auc_bad_m": 0.25,
    }
    for key, fallback in defaults.items():
        if cli_values.get(key, None) is not None:
            continue
        if key in nvs_cfg:
            setattr(args, key, nvs_cfg[key])
        elif getattr(args, key, None) is None:
            setattr(args, key, fallback)
    if getattr(args, "best_metric", None) is None:
        default_best = "pose_energy_pred_cost_m" if bool(args.pose_energy_enabled) else "pred_cost_m"
        args.best_metric = nvs_cfg.get("best_metric", default_best)
    if getattr(args, "best_metric_mode", None) is None:
        args.best_metric_mode = nvs_cfg.get("best_metric_mode", "min")
    return args


def _csv_or_list(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(part) for part in value)
    return str(value)


def parse_float_csv(value) -> List[float]:
    return [float(part) for part in _csv_or_list(value).split(",") if part.strip()]


def parse_score_feature_hw(value) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [int(part) for part in value]
    else:
        raw = str(value).strip()
        if not raw:
            return None
        parts = [int(part) for part in raw.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("score_feature_hw must be formatted as H,W")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError("score_feature_hw values must be positive")
    return height, width


def parse_str_csv(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def candidate_correction_cosines(
    candidate_pose: torch.Tensor,
    init_pose: torch.Tensor,
    pose_gt: torch.Tensor,
) -> torch.Tensor:
    """Return camera-center correction cosine for each candidate relative to init->GT."""
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    if init_pose.ndim != 3 or init_pose.shape[-2:] != (4, 4):
        raise ValueError("init_pose must have shape (B,4,4)")
    if pose_gt.ndim != 3 or pose_gt.shape[-2:] != (4, 4):
        raise ValueError("pose_gt must have shape (B,4,4)")
    bsz, num_candidates = candidate_pose.shape[:2]
    if init_pose.shape[0] != bsz or pose_gt.shape[0] != bsz:
        raise ValueError("batch size must match")
    device = candidate_pose.device
    init_centers = camera_centers_from_w2c(init_pose.to(device=device).float())
    gt_centers = camera_centers_from_w2c(pose_gt.to(device=device).float())
    cand_centers = camera_centers_from_w2c(candidate_pose.float().reshape(-1, 4, 4)).reshape(
        bsz,
        num_candidates,
        3,
    )
    target = gt_centers - init_centers
    correction = cand_centers - init_centers[:, None]
    target_norm = torch.linalg.norm(target, dim=-1, keepdim=True)
    correction_norm = torch.linalg.norm(correction, dim=-1)
    denom = (correction_norm * target_norm).clamp(min=1.0e-8)
    cos = (correction * target[:, None]).sum(dim=-1) / denom
    valid = (correction_norm > 1.0e-6) & (target_norm > 1.0e-6)
    return torch.where(valid, cos, torch.zeros_like(cos))


def candidate_identity_mask(
    candidate_pose: torch.Tensor,
    init_pose: torch.Tensor,
    *,
    trans_eps_m: float = 1.0e-5,
    rot_eps_rad: float = 1.0e-4,
) -> torch.Tensor:
    """Detect candidates that reproduce the initialization pose."""
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    if init_pose.ndim != 3 or init_pose.shape[-2:] != (4, 4):
        raise ValueError("init_pose must have shape (B,4,4)")
    bsz, num_candidates = candidate_pose.shape[:2]
    if init_pose.shape[0] != bsz:
        raise ValueError("batch size must match")
    device = candidate_pose.device
    init = init_pose.to(device=device).float()
    cand = candidate_pose.float()
    init_centers = camera_centers_from_w2c(init)
    cand_centers = camera_centers_from_w2c(cand.reshape(-1, 4, 4)).reshape(bsz, num_candidates, 3)
    trans_delta = torch.linalg.norm(cand_centers - init_centers[:, None], dim=-1)
    rel_rot = torch.matmul(cand[:, :, :3, :3], init[:, None, :3, :3].transpose(-1, -2))
    trace = rel_rot[:, :, 0, 0] + rel_rot[:, :, 1, 1] + rel_rot[:, :, 2, 2]
    rot_delta = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    return (trans_delta <= float(trans_eps_m)) & (rot_delta <= float(rot_eps_rad))


def adaptive_lattice_spec_for_error(trans_m: float, rot_rad: float) -> tuple[List[float], List[float]]:
    """Return bucket-appropriate local lattice magnitudes.

    This mirrors the CPR plan: small basins should not expose 50cm/100cm
    actions, while medium and large basins must keep enough range to recover.
    """
    trans_cm = abs(float(trans_m)) * 100.0
    rot_deg = abs(float(rot_rad)) * 180.0 / math.pi
    if trans_cm <= 12.5 and rot_deg <= 2.5:
        return [0.0, 2.0, 5.0, 10.0, 25.0], [0.0, 0.5, 1.0, 2.0, 5.0]
    if trans_cm <= 30.0 and rot_deg <= 6.0:
        return [0.0, 5.0, 10.0, 25.0], [0.0, 1.0, 2.0, 5.0]
    if trans_cm <= 60.0 and rot_deg <= 12.0:
        return [0.0, 10.0, 25.0, 50.0], [0.0, 2.0, 5.0, 10.0]
    return [0.0, 25.0, 50.0, 100.0], [0.0, 5.0, 10.0, 20.0]


def nvs_pose_energy_vector_dim(args: argparse.Namespace) -> int:
    dim = len(FINE_CANDIDATE_SELECTOR_VECTOR_FEATURE_NAMES) - len(FINE_CANDIDATE_SELECTOR_UNCERTAINTY_FEATURE_NAMES)
    if bool(getattr(args, "pose_energy_use_rgb", False)):
        dim += 3
    if bool(getattr(args, "pose_energy_use_uncertainty", False)):
        dim += len(FINE_CANDIDATE_SELECTOR_UNCERTAINTY_FEATURE_NAMES)
    if bool(getattr(args, "pose_energy_use_delta_vector", False)):
        dim += len(FINE_CANDIDATE_SELECTOR_DELTA_VECTOR_FEATURE_NAMES)
    if bool(getattr(args, "pose_energy_use_center_delta_vector", False)):
        dim += len(FINE_CANDIDATE_SELECTOR_CENTER_DELTA_VECTOR_FEATURE_NAMES)
    return dim


def project_render_bank_with_uncertainty(
    adapter: PoseFeatureDomainAdapter,
    render_feature: torch.Tensor,
    *,
    render_rgb: torch.Tensor | None = None,
    render_chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if render_feature.ndim != 5:
        raise ValueError("render_feature must have shape (B,K,C,H,W)")
    bsz, num_candidates, channels, height, width = render_feature.shape
    render_flat = render_feature.reshape(bsz * num_candidates, channels, height, width)
    rgb_flat = None
    if render_rgb is not None:
        if render_rgb.ndim != 5 or render_rgb.shape[:2] != (bsz, num_candidates) or render_rgb.shape[2] != 3:
            raise ValueError("render_rgb must have shape (B,K,3,H,W)")
        rgb_flat = render_rgb.reshape(bsz * num_candidates, 3, render_rgb.shape[-2], render_rgb.shape[-1])
    chunk_size = int(render_chunk_size or 0)
    if chunk_size > 0 and render_flat.shape[0] > chunk_size:
        loc_pieces = []
        unc_pieces = []
        for start in range(0, render_flat.shape[0], chunk_size):
            rgb_chunk = rgb_flat[start : start + chunk_size] if rgb_flat is not None else None
            loc, unc = adapter.project_render_with_uncertainty(render_flat[start : start + chunk_size], rgb=rgb_chunk)
            loc_pieces.append(loc)
            unc_pieces.append(unc)
        render_loc = torch.cat(loc_pieces, dim=0)
        render_uncertainty = torch.cat(unc_pieces, dim=0)
    else:
        render_loc, render_uncertainty = adapter.project_render_with_uncertainty(render_flat, rgb=rgb_flat)
    return (
        render_loc.reshape(bsz, num_candidates, channels, height, width),
        render_uncertainty.reshape(bsz, num_candidates, 1, height, width),
    )


def project_render_bank(
    adapter: PoseFeatureDomainAdapter,
    render_feature: torch.Tensor,
    *,
    render_rgb: torch.Tensor | None = None,
    render_chunk_size: int = 0,
) -> torch.Tensor:
    if render_feature.ndim != 5:
        raise ValueError("render_feature must have shape (B,K,C,H,W)")
    bsz, num_candidates, channels, height, width = render_feature.shape
    render_flat = render_feature.reshape(bsz * num_candidates, channels, height, width)
    rgb_flat = None
    if render_rgb is not None:
        if render_rgb.ndim != 5 or render_rgb.shape[:2] != (bsz, num_candidates) or render_rgb.shape[2] != 3:
            raise ValueError("render_rgb must have shape (B,K,3,H,W)")
        rgb_flat = render_rgb.reshape(bsz * num_candidates, 3, render_rgb.shape[-2], render_rgb.shape[-1])
    chunk_size = int(render_chunk_size or 0)
    if chunk_size > 0 and render_flat.shape[0] > chunk_size:
        pieces = []
        for start in range(0, render_flat.shape[0], chunk_size):
            rgb_chunk = rgb_flat[start : start + chunk_size] if rgb_flat is not None else None
            pieces.append(adapter.project_render(render_flat[start : start + chunk_size], rgb=rgb_chunk))
        render_loc = torch.cat(pieces, dim=0)
    else:
        render_loc = adapter.project_render(render_flat, rgb=rgb_flat)
    return render_loc.reshape(bsz, num_candidates, channels, height, width)


def _first_candidate_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 5:
        return tensor[:, 0]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    return tensor


def pose_observability_diagnostic_metrics(
    query_feature: torch.Tensor,
    map_feature: torch.Tensor,
    depth: torch.Tensor | None,
    intrinsics: torch.Tensor | None,
    *,
    mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    zero = query_feature.new_zeros(())
    metrics = {
        "query_pose_obs_logdet": zero,
        "query_pose_obs_trace": zero,
        "query_pose_obs_trace_inv": zero,
        "query_pose_obs_condition": zero,
        "query_pose_obs_valid_frac": zero,
        "map_pose_obs_logdet": zero,
        "map_pose_obs_trace": zero,
        "map_pose_obs_trace_inv": zero,
        "map_pose_obs_condition": zero,
        "map_pose_obs_valid_frac": zero,
        "query_pose_obs_unit_logdet": zero,
        "query_pose_obs_unit_trace": zero,
        "query_pose_obs_unit_trace_inv": zero,
        "query_pose_obs_unit_condition": zero,
        "map_pose_obs_unit_logdet": zero,
        "map_pose_obs_unit_trace": zero,
        "map_pose_obs_unit_trace_inv": zero,
        "map_pose_obs_unit_condition": zero,
        "pose_obs_diagnostic_missing": query_feature.new_ones(()),
    }
    if depth is None or intrinsics is None:
        return metrics
    map_feature = _first_candidate_tensor(map_feature)
    depth = _first_candidate_tensor(depth)
    mask = _first_candidate_tensor(mask)
    query_stats = feature_pose_fisher_stats(query_feature.detach().float(), depth.detach().float(), intrinsics, mask=mask)
    map_stats = feature_pose_fisher_stats(map_feature.detach().float(), depth.detach().float(), intrinsics, mask=mask)
    query_unit_stats = feature_pose_fisher_stats(
        query_feature.detach().float(),
        depth.detach().float(),
        intrinsics,
        mask=mask,
        normalize_channels=True,
    )
    map_unit_stats = feature_pose_fisher_stats(
        map_feature.detach().float(),
        depth.detach().float(),
        intrinsics,
        mask=mask,
        normalize_channels=True,
    )
    for key, value in query_stats.items():
        metrics[f"query_pose_obs_{key}"] = value.to(device=query_feature.device, dtype=query_feature.dtype)
    for key, value in map_stats.items():
        metrics[f"map_pose_obs_{key}"] = value.to(device=query_feature.device, dtype=query_feature.dtype)
    for key, value in query_unit_stats.items():
        if key == "valid_frac":
            continue
        metrics[f"query_pose_obs_unit_{key}"] = value.to(device=query_feature.device, dtype=query_feature.dtype)
    for key, value in map_unit_stats.items():
        if key == "valid_frac":
            continue
        metrics[f"map_pose_obs_unit_{key}"] = value.to(device=query_feature.device, dtype=query_feature.dtype)
    metrics["pose_obs_diagnostic_missing"] = zero
    return metrics


def _direction_fractions(args: argparse.Namespace) -> List[float]:
    raw = getattr(args, "direction_fractions", None)
    if raw is None:
        return [0.75, 0.5, 0.25]
    fractions = [value for value in parse_float_csv(raw) if 0.0 < value < 1.0]
    return fractions or [0.75, 0.5, 0.25]


def _direction_balanced_candidates(
    init_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    target_delta = se3_log(pose_gt.float() @ pose_inverse(init_pose.float())).to(
        device=init_pose.device,
        dtype=init_pose.dtype,
    )
    delta_rows = []
    if bool(args.include_identity_candidate):
        delta_rows.append(torch.zeros_like(target_delta))
    for fraction in _direction_fractions(args):
        frac = float(fraction)
        joint = target_delta * frac
        trans_only = torch.zeros_like(target_delta)
        trans_only[:, :3] = target_delta[:, :3] * frac
        rot_only = torch.zeros_like(target_delta)
        rot_only[:, 3:] = target_delta[:, 3:] * frac
        for delta in (joint, -joint, trans_only, -trans_only, rot_only, -rot_only):
            delta_rows.append(delta)
    directed_delta = torch.stack(delta_rows, dim=1)
    bsz, directed_count = directed_delta.shape[:2]
    return apply_pose_delta(
        init_pose[:, None].expand(-1, directed_count, -1, -1).reshape(bsz * directed_count, 4, 4),
        directed_delta.reshape(bsz * directed_count, 6),
    ).reshape(bsz, directed_count, 4, 4).to(device=init_pose.device, dtype=init_pose.dtype)


def _adaptive_candidate_lattice(
    init_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    args: argparse.Namespace,
    *,
    max_candidates: int,
) -> torch.Tensor:
    _cost, _residual, trans_err, rot_err = pose_costs_and_residual_targets(
        init_pose[:, None].float(),
        pose_gt.float(),
        rot_cost_weight=0.0,
    )
    rows = []
    for row_idx in range(init_pose.shape[0]):
        trans_cm, rot_deg = adaptive_lattice_spec_for_error(
            float(trans_err[row_idx, 0].detach().cpu()),
            float(rot_err[row_idx, 0].detach().cpu()),
        )
        row_bank = build_local_pose_lattice_candidates(
            init_pose[row_idx : row_idx + 1],
            trans_cm=trans_cm,
            rot_deg=rot_deg,
            include_identity=bool(args.include_identity_candidate),
            max_candidates=max_candidates,
            limit_strategy=args.limit_strategy,
            combine_trans_rot=True,
            direction_mode=args.lattice_direction_mode,
        )
        rows.append(row_bank[0])
    return torch.stack(rows, dim=0)


def build_nvs_candidate_bank(init_pose: torch.Tensor, pose_gt: torch.Tensor, args: argparse.Namespace, *, train: bool) -> torch.Tensor:
    max_candidates = int(args.topk)
    bank_mode = str(getattr(args, "candidate_bank_mode", "lattice") or "lattice").lower()
    if bank_mode in {"adaptive", "adaptive_balanced", "bucket_adaptive", "adaptive_direction_balanced"}:
        lattice = _adaptive_candidate_lattice(init_pose, pose_gt, args, max_candidates=max_candidates)
        if bank_mode == "adaptive_direction_balanced" and bool(train):
            directed = _direction_balanced_candidates(init_pose, pose_gt, args)
            lattice = torch.cat([directed, lattice], dim=1)[:, :max_candidates]
    elif bank_mode == "direction_balanced" and bool(train):
        directed = _direction_balanced_candidates(init_pose, pose_gt, args)
        lattice = build_local_pose_lattice_candidates(
            init_pose,
            trans_cm=parse_float_csv(args.lattice_trans_cm),
            rot_deg=parse_float_csv(args.lattice_rot_deg),
            include_identity=bool(args.include_identity_candidate),
            max_candidates=max_candidates,
            limit_strategy=args.limit_strategy,
            combine_trans_rot=True,
            direction_mode=args.lattice_direction_mode,
        )
        lattice = torch.cat([directed, lattice], dim=1)[:, :max_candidates]
    elif bank_mode in {"balanced", "direction_balanced"}:
        lattice = build_local_pose_lattice_candidates(
            init_pose,
            trans_cm=parse_float_csv(args.lattice_trans_cm),
            rot_deg=parse_float_csv(args.lattice_rot_deg),
            include_identity=bool(args.include_identity_candidate),
            max_candidates=max_candidates,
            limit_strategy=args.limit_strategy,
            combine_trans_rot=True,
            direction_mode=args.lattice_direction_mode,
        )
    else:
        lattice = build_local_pose_lattice_candidates(
            init_pose,
            trans_cm=parse_float_csv(args.lattice_trans_cm),
            rot_deg=parse_float_csv(args.lattice_rot_deg),
            include_identity=bool(args.include_identity_candidate),
            max_candidates=max_candidates,
            limit_strategy=args.limit_strategy,
            combine_trans_rot=bool(args.combine_trans_rot),
            direction_mode=args.lattice_direction_mode,
        )
    append_gt = bool(args.train_append_gt_candidate) if train else bool(args.eval_append_gt_candidate)
    if not append_gt:
        return lattice
    bank = torch.cat([pose_gt[:, None].to(device=lattice.device, dtype=lattice.dtype), lattice], dim=1)
    return bank[:, :max_candidates]


def _resize_feature(feature: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    return F.interpolate(feature.float(), size=hw, mode="bilinear", align_corners=False)


def _resize_mask(mask: torch.Tensor | None, hw: tuple[int, int]) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim == 5:
        bsz, num, channels, height, width = mask.shape
        mask = mask.reshape(bsz * num, channels, height, width)
        if tuple(mask.shape[-2:]) != hw:
            mask = F.interpolate(mask.float(), size=hw, mode="nearest")
        return mask.reshape(bsz, num, channels, *hw).float()
    if tuple(mask.shape[-2:]) != hw:
        mask = F.interpolate(mask.float(), size=hw, mode="nearest")
    return mask.float()


def _resize_feature_bank(feature: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    if feature.ndim == 4:
        return F.interpolate(feature.float(), size=hw, mode="bilinear", align_corners=False)
    if feature.ndim == 5:
        bsz, count, channels, height, width = feature.shape
        flat = feature.reshape(bsz * count, channels, height, width)
        flat = F.interpolate(flat.float(), size=hw, mode="bilinear", align_corners=False)
        return flat.reshape(bsz, count, channels, *hw)
    raise ValueError(f"feature must be 4D or 5D, got {tuple(feature.shape)}")


def resize_score_candidate_tensors(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None,
    score_weight: torch.Tensor | None,
    *,
    score_hw: tuple[int, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Resize candidate-side tensors for the pose scoring path only."""
    if score_hw is None:
        return query_feature, render_feature, mask, score_weight
    render_feature = _resize_feature_bank(render_feature, score_hw)
    mask = _resize_mask(mask, score_hw) if mask is not None else None
    score_weight = _resize_mask(score_weight, score_hw) if score_weight is not None else None
    return query_feature, render_feature, mask, score_weight


def masked_dense_cosine(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return candidate-wise mean dense cosine.

    query_feature: (B,C,H,W), render_feature: (B,K,C,H,W) or (B,C,H,W).
    """
    squeeze_candidate = False
    if render_feature.ndim == 4:
        render_feature = render_feature[:, None]
        squeeze_candidate = True
    if query_feature.ndim != 4 or render_feature.ndim != 5:
        raise ValueError("expected query (B,C,H,W) and render (B,K,C,H,W)")
    target_hw = tuple(render_feature.shape[-2:])
    query = _resize_feature(query_feature.float(), target_hw)
    render = render_feature.float()
    channels = min(int(query.shape[1]), int(render.shape[2]))
    query = F.normalize(query[:, :channels], dim=1, eps=1.0e-6)
    render = F.normalize(render[:, :, :channels], dim=2, eps=1.0e-6)
    cosine = (query[:, None] * render).sum(dim=2, keepdim=True)
    if mask is not None and mask.ndim == 4 and render_feature.ndim == 5 and mask.shape[1] == render_feature.shape[1]:
        mask = mask[:, :, None]
    valid = _resize_mask(mask, target_hw)
    if valid is None:
        score = cosine.flatten(2).mean(dim=2)
    else:
        if valid.ndim == 4:
            valid = valid[:, None]
        valid = valid.to(device=cosine.device, dtype=cosine.dtype)
        denom = valid.flatten(2).sum(dim=2).clamp(min=1.0)
        score = (cosine * valid).flatten(2).sum(dim=2) / denom
    return score[:, 0] if squeeze_candidate else score


def masked_dense_alignment_loss(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = masked_dense_cosine(query_feature, render_feature, mask=mask)
    loss = F.relu(float(margin) + 1.0 - cosine).mean()
    return loss, cosine.mean()


def observability_contrast_loss(
    query_feature: torch.Tensor,
    gt_render_feature: torch.Tensor,
    candidate_render_feature: torch.Tensor,
    pose_cost: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gt_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    margin: float = 0.05,
    min_negative_cost_m: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Surrogate pose-observability loss for localization features.

    The GT render is the positive pose-energy sample.  Nearby but wrong
    candidate renders are negatives.  This directly encourages the adapted
    feature space to form a sharp local pose energy instead of merely matching
    RADIO/DCFF teacher features in isolation.
    """
    if pose_cost.ndim != 2 or valid_mask.shape != pose_cost.shape:
        raise ValueError("pose_cost and valid_mask must have shape (B,K)")
    gt_score = masked_dense_cosine(query_feature, gt_render_feature, mask=gt_mask)
    if gt_score.ndim == 2:
        if gt_score.shape[1] != 1:
            raise ValueError("gt_render_feature must contain exactly one positive render")
        gt_score = gt_score[:, 0]
    negative_scores = masked_dense_cosine(query_feature, candidate_render_feature, mask=candidate_mask)
    return observability_score_contrast_loss(
        gt_score,
        negative_scores,
        pose_cost,
        valid_mask,
        margin=margin,
        min_negative_cost_m=min_negative_cost_m,
    )


def observability_score_contrast_loss(
    gt_score: torch.Tensor,
    negative_scores: torch.Tensor,
    pose_cost: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    min_negative_cost_m: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Surrogate observability loss directly on candidate selection scores."""
    if gt_score.ndim == 2:
        if gt_score.shape[1] != 1:
            raise ValueError("gt_score with shape (B,K) must have K=1")
        gt_score = gt_score[:, 0]
    if gt_score.ndim != 1:
        raise ValueError("gt_score must have shape (B,) or (B,1)")
    if negative_scores.ndim != 2:
        raise ValueError("negative_scores must have shape (B,K)")
    if negative_scores.shape != pose_cost.shape:
        raise ValueError(
            f"candidate scores shape {tuple(negative_scores.shape)} does not match pose cost {tuple(pose_cost.shape)}"
        )
    if gt_score.shape[0] != negative_scores.shape[0]:
        raise ValueError("gt_score batch size must match negative_scores")
    valid_negative = (
        valid_mask.to(device=negative_scores.device).bool()
        & torch.isfinite(pose_cost.to(device=negative_scores.device))
        & (pose_cost.to(device=negative_scores.device, dtype=negative_scores.dtype) >= float(min_negative_cost_m))
    )
    masked_negatives = negative_scores.float().masked_fill(~valid_negative, -1.0e6)
    hard_negative_score = masked_negatives.max(dim=1).values
    has_negative = valid_negative.any(dim=1)
    per_row = F.relu(float(margin) + hard_negative_score - gt_score.float())
    if bool(has_negative.any()):
        loss = per_row[has_negative].mean()
        active = has_negative.float().mean()
        hard_mean = hard_negative_score[has_negative].mean()
        gap = (gt_score.float()[has_negative] - hard_negative_score[has_negative]).mean()
    else:
        loss = negative_scores.new_zeros(())
        active = negative_scores.new_zeros(())
        hard_mean = negative_scores.new_zeros(())
        gap = negative_scores.new_zeros(())
    return {
        "loss": loss,
        "active": active,
        "gt_score": gt_score.float().mean().detach(),
        "hard_negative_score": hard_mean.detach(),
        "gap": gap.detach(),
    }


def candidate_observability_margin_loss(
    candidate_scores: torch.Tensor,
    pose_cost: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    margin: float = 0.05,
    min_cost_gap_m: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Hard candidate-internal pose-observability margin.

    The positive is the best valid pose candidate in the local bank.  Negatives
    are candidates that are clearly worse in GT pose cost.  This optimizes the
    same score surface used at evaluation time, unlike the GT-render contrast
    which can improve a side objective without improving topK reranking.
    """
    if candidate_scores.ndim != 2 or pose_cost.shape != candidate_scores.shape:
        raise ValueError("candidate_scores and pose_cost must have shape (B,K)")
    if valid_mask.shape != candidate_scores.shape:
        raise ValueError("valid_mask must match candidate_scores")
    valid = valid_mask.to(device=candidate_scores.device).bool() & torch.isfinite(pose_cost)
    cost = pose_cost.to(device=candidate_scores.device, dtype=candidate_scores.dtype)
    scores = candidate_scores.float().masked_fill(~valid, -1.0e6)
    masked_cost = cost.masked_fill(~valid, float("inf"))
    best_idx = masked_cost.argmin(dim=1)
    batch_idx = torch.arange(candidate_scores.shape[0], device=candidate_scores.device)
    best_cost = masked_cost[batch_idx, best_idx]
    best_score = scores[batch_idx, best_idx]
    hard_negative = valid & torch.isfinite(cost) & (cost >= best_cost[:, None] + float(min_cost_gap_m))
    hard_scores = scores.masked_fill(~hard_negative, -1.0e6)
    hard_score = hard_scores.max(dim=1).values
    has_negative = hard_negative.any(dim=1)
    per_row = F.relu(float(margin) + hard_score - best_score)
    if bool(has_negative.any()):
        loss = per_row[has_negative].mean()
        active = has_negative.float().mean()
        gap = (best_score[has_negative] - hard_score[has_negative]).mean()
        hard_mean = hard_score[has_negative].mean()
    else:
        loss = candidate_scores.new_zeros(())
        active = candidate_scores.new_zeros(())
        gap = candidate_scores.new_zeros(())
        hard_mean = candidate_scores.new_zeros(())
    return {
        "loss": loss,
        "active": active,
        "gap": gap.detach(),
        "best_score": best_score.detach().mean(),
        "hard_negative_score": hard_mean.detach(),
        "best_index": best_idx.detach(),
    }


def hard_flow_candidate_selection_mask(
    candidate_scores: torch.Tensor,
    pose_cost: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str = "best_and_hard_negative",
    min_cost_gap_m: float = 0.10,
) -> Dict[str, torch.Tensor]:
    """Select candidates for geometry flow supervision.

    `all` preserves the original behavior.  `best` only supervises the oracle
    candidate.  `best_and_hard_negative` adds the currently high-score wrong
    candidate whose GT pose cost is sufficiently worse than the oracle.
    """
    if candidate_scores.ndim != 2 or pose_cost.shape != candidate_scores.shape:
        raise ValueError("candidate_scores and pose_cost must have shape (B,K)")
    if valid_mask.shape != candidate_scores.shape:
        raise ValueError("valid_mask must match candidate_scores")
    mode = str(mode or "all").lower()
    valid = valid_mask.to(device=candidate_scores.device).bool() & torch.isfinite(
        pose_cost.to(device=candidate_scores.device)
    )
    cost = pose_cost.to(device=candidate_scores.device, dtype=candidate_scores.dtype)
    scores = candidate_scores.float().masked_fill(~valid, -1.0e6)
    masked_cost = cost.masked_fill(~valid, float("inf"))
    best_idx = masked_cost.argmin(dim=1)
    batch_idx = torch.arange(candidate_scores.shape[0], device=candidate_scores.device)
    best_cost = masked_cost[batch_idx, best_idx]
    best_valid = torch.isfinite(best_cost)

    if mode == "all":
        selection = valid.clone()
        hard_idx = best_idx
        has_hard = torch.zeros_like(best_valid)
    elif mode in ("best", "best_only"):
        selection = torch.zeros_like(valid)
        selection[batch_idx, best_idx] = best_valid
        hard_idx = best_idx
        has_hard = torch.zeros_like(best_valid)
    elif mode in ("best_and_hard_negative", "hard_negative", "hard"):
        selection = torch.zeros_like(valid)
        selection[batch_idx, best_idx] = best_valid
        hard_mask = valid & torch.isfinite(cost) & (cost >= best_cost[:, None] + float(min_cost_gap_m))
        hard_scores = scores.masked_fill(~hard_mask, -1.0e6)
        hard_idx = hard_scores.argmax(dim=1)
        has_hard = hard_mask.any(dim=1)
        selection[batch_idx, hard_idx] = selection[batch_idx, hard_idx] | has_hard
        hard_idx = torch.where(has_hard, hard_idx, best_idx)
    else:
        raise ValueError(f"Unknown local_flow_nce_candidate_mode: {mode}")

    return {
        "selection_mask": selection,
        "best_index": best_idx.detach(),
        "hard_negative_index": hard_idx.detach(),
        "best_active": best_valid.float().mean().detach(),
        "hard_negative_active": has_hard.float().mean().detach(),
        "selected_fraction": selection.float().mean().detach(),
    }


def pose_threshold_success_metrics(
    trans_err_m: torch.Tensor,
    rot_err_rad: torch.Tensor,
    *,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """Return CPR-style success fractions for common local-refinement thresholds."""
    trans = trans_err_m.float()
    rot_deg = rot_err_rad.float() * (180.0 / math.pi)
    if trans.shape != rot_deg.shape:
        raise ValueError("translation and rotation error tensors must have the same shape")
    return {
        f"{prefix}success_5cm_2deg": ((trans <= 0.05) & (rot_deg <= 2.0)).float().mean(),
        f"{prefix}success_10cm_5deg": ((trans <= 0.10) & (rot_deg <= 5.0)).float().mean(),
        f"{prefix}success_25cm_10deg": ((trans <= 0.25) & (rot_deg <= 10.0)).float().mean(),
        f"{prefix}success_50cm_10deg": ((trans <= 0.50) & (rot_deg <= 10.0)).float().mean(),
    }


def selected_candidate_pose_metrics(
    pose_cost: torch.Tensor,
    trans_err: torch.Tensor,
    rot_err: torch.Tensor,
    oracle_cost: torch.Tensor,
    selected_idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Gather pose metrics for the candidate chosen by the active selector."""
    if pose_cost.ndim != 2:
        raise ValueError("pose_cost must have shape (B,K)")
    if trans_err.shape != pose_cost.shape or rot_err.shape != pose_cost.shape:
        raise ValueError("trans_err and rot_err must match pose_cost shape")
    if oracle_cost.ndim != 1 or oracle_cost.shape[0] != pose_cost.shape[0]:
        raise ValueError("oracle_cost must have shape (B,)")
    if selected_idx.ndim != 1 or selected_idx.shape[0] != pose_cost.shape[0]:
        raise ValueError("selected_idx must have shape (B,)")
    gather_idx = selected_idx.to(device=pose_cost.device, dtype=torch.long)[:, None]
    selected_cost = pose_cost.gather(1, gather_idx).squeeze(1)
    selected_trans = trans_err.gather(1, gather_idx).squeeze(1)
    selected_rot = rot_err.gather(1, gather_idx).squeeze(1)
    return {
        "selected_cost": selected_cost,
        "selected_trans": selected_trans,
        "selected_rot": selected_rot,
        "selected_oracle_gap": selected_cost - oracle_cost.to(device=pose_cost.device, dtype=pose_cost.dtype),
    }


def candidate_selection_bias_metrics(
    candidate_scores: torch.Tensor,
    pose_cost: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_idx: torch.Tensor,
    *,
    identity_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Measure whether selection is biased toward identity or score-high wrong poses."""
    if candidate_scores.ndim != 2 or pose_cost.shape != candidate_scores.shape:
        raise ValueError("candidate_scores and pose_cost must have shape (B,K)")
    if valid_mask.shape != candidate_scores.shape:
        raise ValueError("valid_mask must match candidate_scores")
    if selected_idx.ndim != 1 or selected_idx.shape[0] != candidate_scores.shape[0]:
        raise ValueError("selected_idx must have shape (B,)")
    device = candidate_scores.device
    valid = valid_mask.to(device=device).bool() & torch.isfinite(pose_cost.to(device=device))
    scores = candidate_scores.float().masked_fill(~valid, -1.0e6)
    costs = pose_cost.to(device=device, dtype=torch.float32).masked_fill(~valid, float("inf"))
    best_idx = costs.argmin(dim=1)
    batch_idx = torch.arange(candidate_scores.shape[0], device=device)
    best_score = scores[batch_idx, best_idx]
    selected_score = scores[batch_idx, selected_idx.to(device=device, dtype=torch.long)]

    if identity_mask is None:
        selected_identity_frac = candidate_scores.new_zeros(())
        best_minus_identity = candidate_scores.new_zeros(())
    else:
        identity = identity_mask.to(device=device).bool() & valid
        has_identity = identity.any(dim=1)
        identity_score = scores.masked_fill(~identity, -1.0e6).max(dim=1).values
        selected_identity = identity.gather(1, selected_idx.to(device=device, dtype=torch.long)[:, None]).squeeze(1)
        selected_identity_frac = selected_identity.float().mean()
        if bool(has_identity.any()):
            best_minus_identity = (best_score[has_identity] - identity_score[has_identity]).mean()
        else:
            best_minus_identity = candidate_scores.new_zeros(())

    return {
        "selected_identity_frac": selected_identity_frac.detach(),
        "score_best_minus_score_identity": best_minus_identity.detach(),
        "score_best_minus_score_selected": (best_score - selected_score).mean().detach(),
    }


def pose_energy_logits_with_base_prior(
    logits: torch.Tensor,
    base_scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    weight: float,
    mode: str = "zscore",
) -> torch.Tensor:
    """Add a frozen candidate-score prior so PoseEnergyNet learns a residual ranker."""
    if logits.shape != base_scores.shape:
        raise ValueError("logits and base_scores must have the same shape")
    valid = torch.ones_like(logits, dtype=torch.bool) if valid_mask is None else valid_mask.to(device=logits.device).bool()
    if valid.shape != logits.shape:
        raise ValueError("valid_mask must match logits shape")
    base = base_scores.to(device=logits.device, dtype=logits.dtype).detach()
    base = torch.where(valid, base, torch.zeros_like(base))
    mode_key = str(mode or "zscore").lower()
    if mode_key == "raw":
        prior = base
    else:
        count = valid.to(dtype=base.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (base * valid.to(dtype=base.dtype)).sum(dim=1, keepdim=True) / count
        centered = torch.where(valid, base - mean, torch.zeros_like(base))
        if mode_key == "centered":
            prior = centered
        elif mode_key == "zscore":
            var = (centered.square() * valid.to(dtype=base.dtype)).sum(dim=1, keepdim=True) / count
            prior = centered / var.clamp(min=1.0e-6).sqrt()
        else:
            raise ValueError(f"Unknown pose energy base prior mode: {mode}")
    combined = logits.float() + float(weight) * prior.float()
    return combined.masked_fill(~valid, -1.0e6).to(dtype=logits.dtype)


def _candidate_feature_to_hw(feature: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if tuple(feature.shape[-2:]) == tuple(hw):
        return feature
    if feature.ndim != 5:
        raise ValueError("candidate feature must have shape (B,K,C,H,W)")
    bsz, num, channels, height, width = feature.shape
    flat = feature.reshape(bsz * num, channels, height, width)
    flat = F.interpolate(flat.float(), size=hw, mode="bilinear", align_corners=False)
    return flat.reshape(bsz, num, channels, *hw)


def _intrinsics_to_components(
    intrinsics: torch.Tensor,
    *,
    batch_size: int,
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    if intrinsics.ndim == 2 and intrinsics.shape[-1] == 4:
        intrinsics = intrinsics[:, None].expand(-1, num_candidates, -1)
        fx, fy, cx, cy = [intrinsics[..., idx] for idx in range(4)]
    elif intrinsics.ndim == 3 and intrinsics.shape[-1] == 4:
        if intrinsics.shape[1] == 1 and num_candidates != 1:
            intrinsics = intrinsics.expand(-1, num_candidates, -1)
        fx, fy, cx, cy = [intrinsics[..., idx] for idx in range(4)]
    elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        intrinsics = intrinsics[:, None].expand(-1, num_candidates, -1, -1)
        fx, fy, cx, cy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    elif intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3):
        if intrinsics.shape[1] == 1 and num_candidates != 1:
            intrinsics = intrinsics.expand(-1, num_candidates, -1, -1)
        fx, fy, cx, cy = intrinsics[..., 0, 0], intrinsics[..., 1, 1], intrinsics[..., 0, 2], intrinsics[..., 1, 2]
    else:
        raise ValueError(f"unsupported intrinsics shape {tuple(intrinsics.shape)}")
    if fx.shape[:2] != (batch_size, num_candidates):
        raise ValueError(f"intrinsics batch/candidate shape {tuple(fx.shape)} does not match {(batch_size, num_candidates)}")
    return (
        fx[..., None, None],
        fy[..., None, None],
        cx[..., None, None],
        cy[..., None, None],
    )


def _scale_intrinsics_between_hw(
    intrinsics: torch.Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """Scale intrinsics when the scoring feature grid is resized."""
    source_h, source_w = int(source_hw[0]), int(source_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if (source_h, source_w) == (target_h, target_w):
        return intrinsics
    if source_h <= 0 or source_w <= 0 or target_h <= 0 or target_w <= 0:
        raise ValueError("source_hw and target_hw must be positive")
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    scaled = intrinsics.clone()
    if scaled.shape[-1] == 4:
        scaled[..., 0] = scaled[..., 0] * sx
        scaled[..., 1] = scaled[..., 1] * sy
        scaled[..., 2] = scaled[..., 2] * sx
        scaled[..., 3] = scaled[..., 3] * sy
    elif scaled.shape[-2:] == (3, 3):
        scaled[..., 0, 0] = scaled[..., 0, 0] * sx
        scaled[..., 1, 1] = scaled[..., 1, 1] * sy
        scaled[..., 0, 2] = scaled[..., 0, 2] * sx
        scaled[..., 1, 2] = scaled[..., 1, 2] * sy
    else:
        raise ValueError(f"unsupported intrinsics shape {tuple(intrinsics.shape)}")
    return scaled


def _flow_geometry_for_score_hw(
    render_position: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    source_hw: tuple[int, int],
    score_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    position = _candidate_feature_to_hw(render_position.float(), score_hw)
    scaled_intrinsics = _scale_intrinsics_between_hw(intrinsics, source_hw, score_hw)
    return position, scaled_intrinsics


def _sample_target_map_at_grid(target: torch.Tensor, grid: torch.Tensor, *, mode: str) -> torch.Tensor:
    if target.ndim == 3:
        target = target[:, None]
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError("target map must have shape (B,C,H,W)")
    bsz, num, height, width = grid.shape[:4]
    target = target.to(device=grid.device, dtype=torch.float32)
    flat_target = target[:, None].expand(-1, num, -1, -1, -1).reshape(bsz * num, target.shape[1], *target.shape[-2:])
    flat_grid = grid.reshape(bsz * num, height, width, 2)
    sampled = F.grid_sample(
        flat_target,
        flat_grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(bsz, num, target.shape[1], height, width)


def project_world_positions_to_feature_grid(
    position_world: torch.Tensor,
    target_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    feature_hw: tuple[int, int],
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    depth_abs_tolerance_m: float = 0.08,
    depth_rel_tolerance: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project candidate world positions into the target/query feature image.

    Returns a grid for ``grid_sample``, a visibility/occlusion mask, and the
    projected target-camera depth for each candidate pixel.
    """
    squeeze_candidate = False
    if position_world.ndim == 4:
        position_world = position_world[:, None]
        squeeze_candidate = True
    if position_world.ndim != 5 or position_world.shape[2] != 3:
        raise ValueError("position_world must have shape (B,K,3,H,W) or (B,3,H,W)")
    bsz, num, _channels, height, width = position_world.shape
    device = position_world.device
    dtype = position_world.dtype
    target_pose = target_pose.to(device=device, dtype=dtype)
    if target_pose.ndim != 3 or target_pose.shape[0] != bsz or target_pose.shape[-2:] != (4, 4):
        raise ValueError(f"target_pose must have shape (B,4,4), got {tuple(target_pose.shape)}")

    world = position_world.permute(0, 1, 3, 4, 2).contiguous()
    rotation = target_pose[:, :3, :3]
    translation = target_pose[:, :3, 3]
    cam = torch.einsum("bij,bkhwj->bkhwi", rotation, world) + translation[:, None, None, None, :]
    x = cam[..., 0]
    y = cam[..., 1]
    z = cam[..., 2].clamp(min=1.0e-6)

    fx, fy, cx, cy = _intrinsics_to_components(
        intrinsics,
        batch_size=bsz,
        num_candidates=num,
        device=device,
        dtype=dtype,
    )
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    target_h, target_w = int(feature_hw[0]), int(feature_hw[1])
    denom_w = max(target_w - 1, 1)
    denom_h = max(target_h - 1, 1)
    grid_x = 2.0 * (u / float(denom_w)) - 1.0
    grid_y = 2.0 * (v / float(denom_h)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    in_bounds = (
        torch.isfinite(grid_x)
        & torch.isfinite(grid_y)
        & (cam[..., 2] > 1.0e-6)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    valid = in_bounds[:, :, None]
    z_map = z[:, :, None]
    if target_mask is not None:
        sampled_mask = _sample_target_map_at_grid(target_mask.float(), grid, mode="nearest")
        valid = valid & (sampled_mask > 0.5)
    if target_depth is not None:
        sampled_depth = _sample_target_map_at_grid(target_depth.float(), grid, mode="bilinear")
        depth_tol = float(depth_abs_tolerance_m) + float(depth_rel_tolerance) * sampled_depth.abs()
        valid = valid & torch.isfinite(sampled_depth) & (sampled_depth > 1.0e-6) & ((sampled_depth - z_map).abs() <= depth_tol)
    if squeeze_candidate:
        return grid[:, 0], valid[:, 0], z_map[:, 0]
    return grid, valid, z_map


def warped_candidate_alignment_loss(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    render_position: torch.Tensor,
    target_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    depth_abs_tolerance_m: float = 0.08,
    depth_rel_tolerance: float = 0.08,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if query_feature.ndim != 4 or render_feature.ndim != 5:
        raise ValueError("expected query (B,C,H,W) and render (B,K,C,H,W)")
    if render_position.ndim != 5:
        raise ValueError("render_position must have shape (B,K,3,H,W)")
    pos_hw = tuple(render_position.shape[-2:])
    render = _candidate_feature_to_hw(render_feature.float(), pos_hw)
    query = query_feature.float()
    grid, projected_valid, _depth = project_world_positions_to_feature_grid(
        render_position.float(),
        target_pose.float(),
        intrinsics,
        feature_hw=tuple(query.shape[-2:]),
        target_depth=target_depth,
        target_mask=target_mask,
        depth_abs_tolerance_m=depth_abs_tolerance_m,
        depth_rel_tolerance=depth_rel_tolerance,
    )
    bsz, num, _channels, height, width = render.shape
    flat_query = query[:, None].expand(-1, num, -1, -1, -1).reshape(bsz * num, query.shape[1], *query.shape[-2:])
    sampled_query = F.grid_sample(
        flat_query,
        grid.reshape(bsz * num, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(bsz, num, query.shape[1], height, width)
    channels = min(int(sampled_query.shape[2]), int(render.shape[2]))
    sampled_query = F.normalize(sampled_query[:, :, :channels], dim=2, eps=1.0e-6)
    render = F.normalize(render[:, :, :channels], dim=2, eps=1.0e-6)
    cosine = (sampled_query * render).sum(dim=2, keepdim=True)
    valid = projected_valid
    if candidate_mask is not None:
        cand_valid = _resize_mask(candidate_mask, pos_hw)
        if cand_valid.ndim == 4:
            cand_valid = cand_valid[:, None]
        valid = valid & (cand_valid.to(device=valid.device) > 0.5)
    valid_f = valid.to(device=cosine.device, dtype=cosine.dtype)
    denom = valid_f.sum().clamp(min=1.0)
    loss = ((1.0 - cosine) * valid_f).sum() / denom
    metrics = {
        "warp_align_loss": loss.detach(),
        "warp_align_cos": ((cosine * valid_f).sum() / denom).detach(),
        "warp_valid_frac": valid_f.mean().detach(),
        "warp_in_bounds_frac": projected_valid.float().mean().detach(),
    }
    return loss, metrics


def _local_offset_norms(radius: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    offsets = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            offsets.append(math.sqrt(float(dx * dx + dy * dy)))
    return torch.tensor(offsets, device=device, dtype=dtype)


def local_correlation_volume(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    radius: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a local query-render correlation volume around same-pixel alignment."""
    if query_feature.ndim != 4 or render_feature.ndim != 5:
        raise ValueError("expected query (B,C,H,W) and render (B,K,C,H,W)")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    target_hw = tuple(render_feature.shape[-2:])
    query = _resize_feature(query_feature.float(), target_hw)
    render = render_feature.float()
    channels = min(int(query.shape[1]), int(render.shape[2]))
    query = F.normalize(query[:, :channels], dim=1, eps=1.0e-6)
    render = F.normalize(render[:, :, :channels], dim=2, eps=1.0e-6)
    bsz, num, _channels, height, width = render.shape
    kernel = 2 * radius + 1
    patches = F.unfold(query, kernel_size=kernel, padding=radius)
    patches = patches.view(bsz, channels, kernel * kernel, height, width)
    corr = torch.einsum("bkchw,bcphw->bkphw", render, patches)
    valid = torch.ones((bsz, num, 1, height, width), device=render.device, dtype=torch.bool)
    if mask is not None:
        valid_mask = _resize_mask(mask, target_hw)
        if valid_mask.ndim == 4:
            valid_mask = valid_mask[:, None]
        valid = valid & (valid_mask.to(device=render.device) > 0.5)
    return corr, valid


def local_zero_offset_scores_from_corr(
    corr: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    peak_gap_weight: float,
    offset_weight: float,
    score_mode: str = "local_zero_offset",
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if corr.ndim != 5:
        raise ValueError("corr must have shape (B,K,P,H,W)")
    radius = int(radius)
    zero_idx = (2 * radius + 1) * radius + radius
    zero = corr[:, :, zero_idx]
    peak, peak_idx = corr.max(dim=2)
    offset_norms = _local_offset_norms(radius, device=corr.device, dtype=corr.dtype)
    probs = F.softmax(corr / max(float(temperature), 1.0e-6), dim=2)
    expected_offset = (probs * offset_norms.view(1, 1, -1, 1, 1)).sum(dim=2)
    peak_offset = offset_norms[peak_idx]
    score_mode = str(score_mode or "local_zero_offset")
    if score_mode == "local_zero_only":
        score_map = zero
    elif score_mode == "local_neg_expected_offset":
        score_map = -expected_offset
    elif score_mode == "local_neg_peak_offset":
        score_map = -peak_offset
    else:
        score_map = zero - float(peak_gap_weight) * (peak - zero).clamp_min(0.0) - float(offset_weight) * expected_offset
    if valid.ndim == 5:
        valid_2d = valid[:, :, 0]
    else:
        valid_2d = valid
    valid_f = valid_2d.to(device=corr.device, dtype=corr.dtype)
    if weight is not None:
        weight_f = _resize_mask(weight, tuple(corr.shape[-2:]))
        if weight_f.ndim == 5:
            weight_f = weight_f[:, :, 0]
        elif weight_f.ndim == 4:
            if weight_f.shape[1] == corr.shape[1]:
                pass
            elif weight_f.shape[1] == 1:
                weight_f = weight_f[:, 0][:, None].expand(-1, corr.shape[1], -1, -1)
            else:
                raise ValueError(
                    f"weight channel/candidate dimension {weight_f.shape[1]} cannot broadcast to K={corr.shape[1]}"
                )
        else:
            raise ValueError("weight must have shape (B,K,H,W), (B,K,1,H,W), or (B,1,H,W)")
        valid_f = valid_f * weight_f.to(device=corr.device, dtype=corr.dtype).clamp_min(0.0)
    denom = valid_f.flatten(2).sum(dim=2).clamp(min=1.0)
    scores = (score_map * valid_f).flatten(2).sum(dim=2) / denom
    stats = {
        "local_zero_score": ((zero * valid_f).flatten(2).sum(dim=2) / denom).mean().detach(),
        "local_peak_score": ((peak * valid_f).flatten(2).sum(dim=2) / denom).mean().detach(),
        "local_peak_gap": (((peak - zero).clamp_min(0.0) * valid_f).flatten(2).sum(dim=2) / denom).mean().detach(),
        "local_peak_offset_px": ((peak_offset * valid_f).flatten(2).sum(dim=2) / denom).mean().detach(),
        "local_expected_offset_px": ((expected_offset * valid_f).flatten(2).sum(dim=2) / denom).mean().detach(),
    }
    return scores, stats


def local_zero_offset_correlation_scores(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    radius: int = 3,
    temperature: float = 0.07,
    peak_gap_weight: float = 0.5,
    offset_weight: float = 0.05,
    score_mode: str = "local_zero_offset",
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    corr, valid = local_correlation_volume(query_feature, render_feature, mask=mask, radius=radius)
    return local_zero_offset_scores_from_corr(
        corr,
        valid,
        radius=radius,
        temperature=temperature,
        peak_gap_weight=peak_gap_weight,
        offset_weight=offset_weight,
        score_mode=score_mode,
        weight=weight,
    )


def pair_matcher_local_candidate_score_maps(
    matcher: PairConditionedLocalMatcher,
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    radius: int = 3,
    stride: int = 8,
    temperature: float = 0.05,
    chunk_points: int = 65536,
    candidate_chunk_size: int = 0,
    offset_chunk_size: int = 0,
    candidate_score_mode: str = "center_logprob_margin",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sparse pair-matcher heatmap evidence for pose candidates.

    A sparse regular grid is used to keep memory bounded.  For each sampled
    query pixel, the matcher sees the candidate render patch centered at the
    same pixel.  Good candidates should put probability mass on the zero-offset
    center; bad candidates should move the peak away from center.

    The output channels are:
      0. selected candidate evidence score map;
      1. center log probability;
      2. center-vs-hard-negative margin.
    """
    if query_feature.ndim != 4 or render_feature.ndim != 5:
        raise ValueError("expected query (B,C,H,W) and render (B,K,C,H,W)")
    radius = max(int(radius), 0)
    stride = max(int(stride), 1)
    chunk_points = max(int(chunk_points), 1)
    target_hw = tuple(render_feature.shape[-2:])
    query = _resize_feature(query_feature.float(), target_hw)
    render = render_feature.float()
    channels = min(int(query.shape[1]), int(render.shape[2]))
    query = query[:, :channels]
    render = render[:, :, :channels]
    bsz, num_candidates, _channels, height, width = render.shape
    candidate_chunk_size = int(candidate_chunk_size or 0)
    if candidate_chunk_size <= 0 or candidate_chunk_size > num_candidates:
        candidate_chunk_size = num_candidates
    offset_chunk_size = int(offset_chunk_size or 0)
    ys = torch.arange(stride // 2, height, stride, device=query.device, dtype=query.dtype)
    xs = torch.arange(stride // 2, width, stride, device=query.device, dtype=query.dtype)
    if ys.numel() == 0:
        ys = torch.arange(0, height, device=query.device, dtype=query.dtype)
    if xs.numel() == 0:
        xs = torch.arange(0, width, device=query.device, dtype=query.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_h, grid_w = int(grid_y.shape[0]), int(grid_y.shape[1])
    xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    num_points = int(xy.shape[0])
    valid_grid = torch.ones((bsz, num_points), device=query.device, dtype=query.dtype)
    query_xy = xy[None].expand(bsz, -1, -1)
    query_sparse, query_in = sample_feature_at_xy(query, query_xy, valid_grid, source_hw=None)

    render_flat = render.reshape(bsz * num_candidates, channels, height, width)
    render_xy = xy[None].expand(bsz * num_candidates, -1, -1)
    offset_axis = torch.arange(-radius, radius + 1, device=query.device, dtype=query.dtype)
    dy, dx = torch.meshgrid(offset_axis, offset_axis, indexing="ij")
    offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
    center_index = int(((offsets[:, 0] == 0) & (offsets[:, 1] == 0)).nonzero(as_tuple=False)[0].item())

    mask_2d = None
    if mask is not None:
        mask_2d = _resize_mask(mask, target_hw)
        if mask_2d.ndim == 5:
            mask_2d = mask_2d[:, :, 0]
        elif mask_2d.ndim == 4:
            if mask_2d.shape[1] == 1:
                mask_2d = mask_2d[:, 0][:, None].expand(-1, num_candidates, -1, -1)
            elif mask_2d.shape[1] != num_candidates:
                raise ValueError(f"mask candidate dimension {mask_2d.shape[1]} cannot broadcast to K={num_candidates}")
        else:
            raise ValueError("mask must have shape (B,K,H,W), (B,K,1,H,W), or (B,1,H,W)")

    score_map_chunks = []
    valid_chunks = []
    for cand_start in range(0, num_candidates, candidate_chunk_size):
        cand_end = min(num_candidates, cand_start + candidate_chunk_size)
        chunk_k = cand_end - cand_start
        render_chunk = render[:, cand_start:cand_end].contiguous()
        render_flat = render_chunk.reshape(bsz * chunk_k, channels, height, width)
        render_xy = xy[None].expand(bsz * chunk_k, -1, -1)
        patch_xy = render_xy[:, :, None, :] + offsets[None, None, :, :]
        patch_sparse, patch_in = sample_feature_at_xy(
            render_flat,
            patch_xy.reshape(bsz * chunk_k, num_points * offsets.shape[0], 2),
            torch.ones(
                (bsz * chunk_k, num_points * offsets.shape[0]),
                device=query.device,
                dtype=query.dtype,
            ),
            source_hw=None,
        )
        patch_sparse = patch_sparse.reshape(bsz, chunk_k, num_points, offsets.shape[0], channels)
        patch_in = patch_in.reshape(bsz, chunk_k, num_points, offsets.shape[0])

        center_valid = query_in[:, None, :] & patch_in[:, :, :, center_index]
        if mask_2d is not None:
            mask_flat = mask_2d[:, cand_start:cand_end].contiguous().reshape(bsz * chunk_k, 1, height, width)
            mask_sparse, mask_in = sample_feature_at_xy(
                mask_flat,
                render_xy,
                torch.ones((bsz * chunk_k, num_points), device=query.device, dtype=query.dtype),
                source_hw=None,
            )
            mask_sparse = mask_sparse[..., 0].reshape(bsz, chunk_k, num_points)
            mask_in = mask_in.reshape(bsz, chunk_k, num_points)
            center_valid = center_valid & mask_in & (mask_sparse > 0.5)

        q_all = query_sparse[:, None].expand(-1, chunk_k, -1, -1).reshape(
            bsz * chunk_k * num_points,
            channels,
        )
        patch_all = patch_sparse.reshape(bsz * chunk_k * num_points, offsets.shape[0], channels)
        patch_valid_all = patch_in.reshape(bsz * chunk_k * num_points, offsets.shape[0])
        if 0 < offset_chunk_size < int(offsets.shape[0]):
            temp = max(float(temperature), 1.0e-6)
            with torch.no_grad():
                q_norm = F.normalize(q_all.float(), dim=-1, eps=1.0e-6)
                patch_norm = F.normalize(patch_all.float(), dim=-1, eps=1.0e-6)
                full_base_cos = (q_norm[:, None, :] * patch_norm).sum(dim=-1)
                full_row_mean = full_base_cos.mean(dim=1, keepdim=True)
                full_row_max = full_base_cos.max(dim=1, keepdim=True).values
                del q_norm, patch_norm, full_base_cos
            center_logit_flat = None
            logsumexp_flat = None
            hard_neg_flat = None
            for offset_start in range(0, int(offsets.shape[0]), offset_chunk_size):
                offset_end = min(int(offsets.shape[0]), offset_start + offset_chunk_size)
                logits_chunks = []
                for start in range(0, q_all.shape[0], chunk_points):
                    logits_chunks.append(
                        matcher(
                            q_all[start : start + chunk_points],
                            patch_all[start : start + chunk_points, offset_start:offset_end],
                            offsets=offsets[offset_start:offset_end],
                            patch_valid=patch_valid_all[start : start + chunk_points, offset_start:offset_end],
                            row_mean=full_row_mean[start : start + chunk_points],
                            row_max=full_row_max[start : start + chunk_points],
                        )
                    )
                logits_chunk = torch.cat(logits_chunks, dim=0)
                chunk_logsumexp = torch.logsumexp(logits_chunk / temp, dim=1)
                logsumexp_flat = (
                    chunk_logsumexp
                    if logsumexp_flat is None
                    else torch.logaddexp(logsumexp_flat, chunk_logsumexp)
                )
                if offset_start <= center_index < offset_end:
                    center_local = int(center_index - offset_start)
                    center_logit_flat = logits_chunk[:, center_local]
                    logits_for_hard = logits_chunk.clone()
                    logits_for_hard[:, center_local] = -1.0e4
                else:
                    logits_for_hard = logits_chunk
                chunk_hard = logits_for_hard.max(dim=1).values
                hard_neg_flat = chunk_hard if hard_neg_flat is None else torch.maximum(hard_neg_flat, chunk_hard)
            if center_logit_flat is None or logsumexp_flat is None or hard_neg_flat is None:
                raise RuntimeError("pair matcher offset chunking failed to cover center offset")
            center_logit = center_logit_flat.reshape(bsz, chunk_k, num_points)
            center_log_prob = (center_logit_flat / temp - logsumexp_flat).reshape(bsz, chunk_k, num_points)
            has_neg = patch_valid_all.clone()
            has_neg[:, center_index] = False
            hard_neg = hard_neg_flat.reshape(bsz, chunk_k, num_points)
            has_neg = has_neg.any(dim=1).reshape(bsz, chunk_k, num_points)
        else:
            logits_chunks = []
            for start in range(0, q_all.shape[0], chunk_points):
                logits_chunks.append(
                    matcher(
                        q_all[start : start + chunk_points],
                        patch_all[start : start + chunk_points],
                        offsets=offsets,
                        patch_valid=patch_valid_all[start : start + chunk_points],
                    )
                )
            logits = torch.cat(logits_chunks, dim=0).reshape(bsz, chunk_k, num_points, offsets.shape[0])
            log_probs = F.log_softmax(logits / max(float(temperature), 1.0e-6), dim=-1)
            center_log_prob = log_probs[:, :, :, center_index]
            center_logit = logits[:, :, :, center_index]
            other_mask = patch_in.clone()
            other_mask[:, :, :, center_index] = False
            hard_neg = logits.masked_fill(~other_mask, -1.0e4).max(dim=-1).values
            has_neg = other_mask.any(dim=-1)
        hard_neg = torch.where(has_neg, hard_neg, torch.zeros_like(hard_neg))
        score_mode = str(candidate_score_mode or "center_logprob_margin").lower()
        center_margin = center_logit - hard_neg
        if score_mode == "center_margin":
            score_map = center_margin
        elif score_mode == "center_logprob_margin":
            score_map = center_log_prob + center_margin
        else:
            raise ValueError(f"Unknown pair matcher candidate score mode: {candidate_score_mode}")
        score_map_chunks.append(
            torch.stack([score_map, center_log_prob, center_margin], dim=2).reshape(
                bsz,
                chunk_k,
                3,
                grid_h,
                grid_w,
            )
        )
        valid_chunks.append(center_valid.reshape(bsz, chunk_k, grid_h, grid_w))

    return torch.cat(score_map_chunks, dim=1), torch.cat(valid_chunks, dim=1)


def pair_matcher_local_candidate_scores(
    matcher: PairConditionedLocalMatcher,
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    radius: int = 3,
    stride: int = 8,
    temperature: float = 0.05,
    chunk_points: int = 65536,
    candidate_chunk_size: int = 0,
    offset_chunk_size: int = 0,
    candidate_score_mode: str = "center_logprob_margin",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Score pose candidates with pair-conditioned local heatmap evidence."""
    score_maps, valid_map = pair_matcher_local_candidate_score_maps(
        matcher,
        query_feature,
        render_feature,
        mask=mask,
        radius=radius,
        stride=stride,
        temperature=temperature,
        chunk_points=chunk_points,
        candidate_chunk_size=candidate_chunk_size,
        offset_chunk_size=offset_chunk_size,
        candidate_score_mode=candidate_score_mode,
    )
    valid_f = valid_map.to(dtype=score_maps.dtype)
    denom = valid_f.flatten(2).sum(dim=-1).clamp(min=1.0)
    score_map = score_maps[:, :, 0]
    center_log_prob = score_maps[:, :, 1]
    center_margin = score_maps[:, :, 2]
    scores = (score_map * valid_f).flatten(2).sum(dim=-1) / denom
    stats = {
        "local_zero_score": ((center_log_prob * valid_f).flatten(2).sum(dim=-1) / denom).mean().detach(),
        "local_peak_score": ((score_map * valid_f).flatten(2).sum(dim=-1) / denom).mean().detach(),
        "local_peak_gap": ((-center_margin * valid_f).flatten(2).sum(dim=-1) / denom).mean().detach(),
        "local_peak_offset_px": score_maps.new_zeros(()).detach(),
        "local_expected_offset_px": score_maps.new_zeros(()).detach(),
    }
    return scores, stats


def local_flow_nce_loss_from_corr(
    corr: torch.Tensor,
    corr_valid: torch.Tensor,
    render_position: torch.Tensor,
    target_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    candidate_selection_mask: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    radius: int = 3,
    temperature: float = 0.07,
    depth_abs_tolerance_m: float = 0.08,
    depth_rel_tolerance: float = 0.08,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if corr.ndim != 5:
        raise ValueError("corr must have shape (B,K,P,H,W)")
    radius = int(radius)
    bsz, num, _patches, height, width = corr.shape
    grid, projected_valid, _depth = project_world_positions_to_feature_grid(
        render_position.float(),
        target_pose.float(),
        intrinsics,
        feature_hw=(height, width),
        target_depth=target_depth,
        target_mask=target_mask,
        depth_abs_tolerance_m=depth_abs_tolerance_m,
        depth_rel_tolerance=depth_rel_tolerance,
    )
    device = corr.device
    dtype = corr.dtype
    x_coords = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width).expand(bsz, num, height, width)
    y_coords = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1).expand(bsz, num, height, width)
    u = (grid[..., 0].to(dtype=dtype) + 1.0) * 0.5 * float(max(width - 1, 1))
    v = (grid[..., 1].to(dtype=dtype) + 1.0) * 0.5 * float(max(height - 1, 1))
    du = torch.round(u - x_coords).long()
    dv = torch.round(v - y_coords).long()
    in_window = (du >= -radius) & (du <= radius) & (dv >= -radius) & (dv <= radius)
    target_index = (dv + radius) * (2 * radius + 1) + (du + radius)
    valid = projected_valid
    if corr_valid.ndim == 5:
        valid = valid & corr_valid.bool()
    if candidate_mask is not None:
        cand_valid = _resize_mask(candidate_mask, (height, width))
        if cand_valid.ndim == 4:
            cand_valid = cand_valid[:, None]
        valid = valid & (cand_valid.to(device=device) > 0.5)
    selected_frac = corr.new_tensor(1.0)
    if candidate_selection_mask is not None:
        if candidate_selection_mask.shape != (bsz, num):
            raise ValueError(
                f"candidate_selection_mask must have shape {(bsz, num)}, got {tuple(candidate_selection_mask.shape)}"
            )
        selection = candidate_selection_mask.to(device=device).bool()
        valid = valid & selection[:, :, None, None, None]
        selected_frac = selection.float().mean().to(device=device, dtype=corr.dtype)
    valid = valid[:, :, 0] & in_window
    logits = (corr / max(float(temperature), 1.0e-6)).permute(0, 1, 3, 4, 2).reshape(-1, corr.shape[2])
    targets = target_index.reshape(-1).clamp(min=0, max=corr.shape[2] - 1)
    valid_flat = valid.reshape(-1)
    if bool(valid_flat.any()):
        loss = F.cross_entropy(logits[valid_flat], targets[valid_flat])
        offset_norm = torch.sqrt((du.float() ** 2) + (dv.float() ** 2))
        target_offset = offset_norm[valid].mean()
    else:
        loss = corr.sum() * 0.0
        target_offset = corr.new_zeros(())
    metrics = {
        "local_flow_nce_loss": loss.detach(),
        "local_flow_valid_frac": valid.float().mean().detach(),
        "local_flow_target_offset_px": target_offset.detach(),
        "local_flow_candidate_selected_frac": selected_frac.detach(),
    }
    return loss, metrics


def local_flow_nce_loss(
    query_feature: torch.Tensor,
    render_feature: torch.Tensor,
    render_position: torch.Tensor,
    target_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    candidate_mask: torch.Tensor | None = None,
    candidate_selection_mask: torch.Tensor | None = None,
    target_depth: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    radius: int = 3,
    temperature: float = 0.07,
    depth_abs_tolerance_m: float = 0.08,
    depth_rel_tolerance: float = 0.08,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    corr, corr_valid = local_correlation_volume(query_feature, render_feature, mask=candidate_mask, radius=radius)
    return local_flow_nce_loss_from_corr(
        corr,
        corr_valid,
        render_position,
        target_pose,
        intrinsics,
        candidate_mask=candidate_mask,
        candidate_selection_mask=candidate_selection_mask,
        target_depth=target_depth,
        target_mask=target_mask,
        radius=radius,
        temperature=temperature,
        depth_abs_tolerance_m=depth_abs_tolerance_m,
        depth_rel_tolerance=depth_rel_tolerance,
    )


def feature_drift_loss(
    query_loc: torch.Tensor,
    query_base: torch.Tensor,
    render_loc: torch.Tensor,
    render_base: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    query_loss, _ = masked_dense_alignment_loss(query_loc, query_base, None)
    render_loss, _ = masked_dense_alignment_loss(render_loc[:, 0], render_base[:, 0], mask[:, 0] if mask is not None and mask.ndim == 5 else mask)
    return 0.5 * (query_loss + render_loss)


def variance_floor_loss(feature: torch.Tensor, min_std: float) -> torch.Tensor:
    if feature.ndim == 5:
        channels = feature.shape[2]
        values = feature.permute(0, 1, 3, 4, 2).reshape(-1, channels).float()
    elif feature.ndim == 4:
        channels = feature.shape[1]
        values = feature.permute(0, 2, 3, 1).reshape(-1, channels).float()
    else:
        raise ValueError("feature must be 4D or 5D")
    std = values.std(dim=0)
    return F.relu(float(min_std) - std).mean()


def rank_losses_from_scores(
    scores: torch.Tensor,
    pose_cost: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature_m: float,
    pairwise_weight: float,
    pairwise_min_gap_m: float,
    pairwise_logit_margin: float,
) -> Dict[str, torch.Tensor]:
    valid = valid.bool() & torch.isfinite(pose_cost)
    logits = (scores.float() / max(float(temperature_m), 1.0e-6)).masked_fill(~valid, -1.0e6)
    cost = pose_cost.float()
    target_logits = (-cost / max(float(temperature_m), 1.0e-6)).masked_fill(~valid, -1.0e6)
    target_probs = F.softmax(target_logits, dim=1).detach()
    ce_loss = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    target_idx = cost.masked_fill(~valid, float("inf")).argmin(dim=1)
    best_cost = cost.gather(1, target_idx[:, None])
    best_logit = logits.gather(1, target_idx[:, None])
    worse = valid & (cost > best_cost + float(pairwise_min_gap_m))
    pairwise = scores.new_zeros(())
    if bool(worse.any()):
        pairwise = F.softplus(logits - best_logit + float(pairwise_logit_margin))[worse].mean()
    pred_idx = logits.argmax(dim=1)
    pred_cost = cost.gather(1, pred_idx[:, None]).squeeze(1)
    oracle_cost = cost.gather(1, target_idx[:, None]).squeeze(1)
    total = ce_loss + float(pairwise_weight) * pairwise
    return {
        "rank_loss": total,
        "rank_ce_loss": ce_loss,
        "rank_pairwise_loss": pairwise,
        "rank_pairwise_active": worse.float().mean(),
        "pred_index": pred_idx.detach(),
        "target_index": target_idx.detach(),
        "top1_acc": (pred_idx == target_idx).float().mean(),
        "pred_cost": pred_cost,
        "oracle_cost": oracle_cost,
        "oracle_gap": pred_cost - oracle_cost,
    }


def pose_energy_score_monotonicity_loss(
    before_score: torch.Tensor,
    after_score: torch.Tensor,
    *,
    before_cost: torch.Tensor,
    after_cost: torch.Tensor,
    margin: float = 0.0,
    improved_only: bool = True,
) -> Dict[str, torch.Tensor]:
    before_score = before_score.float()
    after_score = after_score.to(device=before_score.device, dtype=before_score.dtype)
    before_cost = before_cost.to(device=before_score.device, dtype=before_score.dtype)
    after_cost = after_cost.to(device=before_score.device, dtype=before_score.dtype)
    valid = torch.isfinite(before_score) & torch.isfinite(after_score) & torch.isfinite(before_cost) & torch.isfinite(after_cost)
    if bool(improved_only):
        valid = valid & (after_cost < before_cost)
    score_gain = after_score - before_score
    loss_per = F.relu(float(margin) - score_gain)
    if bool(valid.any()):
        loss = loss_per[valid].mean()
        mean_gain = score_gain[valid].mean()
    else:
        loss = before_score.new_zeros(())
        mean_gain = before_score.new_zeros(())
    return {
        "loss": loss,
        "active": valid.float().mean(),
        "score_gain": mean_gain,
    }


def pose_energy_direction_pairwise_loss(
    logits: torch.Tensor,
    candidate_pose: torch.Tensor,
    init_pose: torch.Tensor,
    pose_gt: torch.Tensor,
    target_index: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    min_cos_gap: float = 0.25,
    logit_margin: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Penalize candidates whose correction direction is worse than the GT-nearest candidate."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B,K)")
    bsz, num_candidates = logits.shape
    valid = torch.ones((bsz, num_candidates), device=logits.device, dtype=torch.bool)
    if valid_mask is not None:
        valid = valid_mask.to(device=logits.device).bool()
    init_centers = camera_centers_from_w2c(init_pose.to(device=logits.device).float())
    gt_centers = camera_centers_from_w2c(pose_gt.to(device=logits.device).float())
    cand_centers = camera_centers_from_w2c(candidate_pose.to(device=logits.device).float().reshape(-1, 4, 4)).reshape(
        bsz,
        num_candidates,
        3,
    )
    target_vec = gt_centers - init_centers
    cand_vec = cand_centers - init_centers[:, None]
    target_norm = torch.linalg.norm(target_vec, dim=-1, keepdim=True)
    cand_norm = torch.linalg.norm(cand_vec, dim=-1)
    cos = (cand_vec * target_vec[:, None]).sum(dim=-1) / (cand_norm * target_norm).clamp(min=1.0e-8)
    cos = torch.where((cand_norm > 1.0e-6) & (target_norm > 1.0e-6), cos, torch.zeros_like(cos))
    batch_idx = torch.arange(bsz, device=logits.device)
    target_index = target_index.to(device=logits.device).long()
    target_cos = cos[batch_idx, target_index][:, None]
    target_logit = logits[batch_idx, target_index][:, None]
    bad_direction = valid & (cos + float(min_cos_gap) < target_cos)
    active = bad_direction.float().mean()
    if bool(bad_direction.any()):
        loss = F.softplus(logits - target_logit + float(logit_margin))[bad_direction].mean()
    else:
        loss = logits.new_zeros(())
    pred_idx = logits.masked_fill(~valid, -1.0e6).argmax(dim=1)
    pred_cos = cos[batch_idx, pred_idx]
    return {
        "loss": loss,
        "active": active,
        "pred_cos": pred_cos.mean(),
        "target_cos": cos[batch_idx, target_index].mean(),
    }


def pose_energy_correction_cosine_soft_label_loss(
    logits: torch.Tensor,
    correction_cos: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    target_temperature: float = 0.1,
    min_cos: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Train candidate logits to select candidates moving from init toward GT.

    This complements pose-cost ranking. Several candidates can have similar
    pose cost in a local lattice, but candidates moving opposite the GT
    correction should never win the topK selector.
    """
    if logits.ndim != 2 or correction_cos.ndim != 2:
        raise ValueError("logits and correction_cos must have shape (B,K)")
    if logits.shape != correction_cos.shape:
        raise ValueError("logits and correction_cos shapes must match")
    valid = torch.isfinite(logits) & torch.isfinite(correction_cos)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=logits.device).bool()
    logits = logits.float()
    correction_cos = correction_cos.to(device=logits.device, dtype=logits.dtype)
    positive = (correction_cos - float(min_cos)).clamp_min(0.0).masked_fill(~valid, 0.0)
    has_positive = positive.sum(dim=1, keepdim=True) > 1.0e-8
    target_logits = positive / max(float(target_temperature), 1.0e-6)
    target_logits = target_logits.masked_fill(~valid, -1.0e6)
    uniform_logits = torch.zeros_like(target_logits).masked_fill(~valid, -1.0e6)
    target_probs = torch.where(
        has_positive,
        F.softmax(target_logits, dim=1),
        F.softmax(uniform_logits, dim=1),
    ).detach()
    selection_log_probs = F.log_softmax(logits.masked_fill(~valid, -1.0e6), dim=1)
    loss_per = -(target_probs * selection_log_probs).sum(dim=1)
    active_rows = valid.any(dim=1)
    if bool(active_rows.any()):
        loss = loss_per[active_rows].mean()
    else:
        loss = logits.new_zeros(())
    masked_logits = logits.masked_fill(~valid, -1.0e6)
    masked_target = positive.masked_fill(~valid, -1.0e6)
    pred_idx = masked_logits.argmax(dim=1)
    target_idx = masked_target.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    pred_cos = correction_cos[batch_idx, pred_idx]
    target_cos = correction_cos[batch_idx, target_idx]
    return {
        "loss": loss,
        "active": active_rows.float().mean(),
        "pred_cos": pred_cos.mean(),
        "target_cos": target_cos.mean(),
        "pred_index": pred_idx.detach(),
        "target_index": target_idx.detach(),
    }


def score_pose_improvement_soft_label_loss(
    logits: torch.Tensor,
    candidate_cost: torch.Tensor,
    init_cost: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    target_temperature_m: float = 0.05,
    min_improvement_m: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Train candidate scores to prefer hypotheses that improve over T0."""
    if logits.ndim != 2 or candidate_cost.ndim != 2:
        raise ValueError("logits and candidate_cost must have shape (B,K)")
    if logits.shape != candidate_cost.shape:
        raise ValueError("logits and candidate_cost shapes must match")
    if init_cost.ndim == 1:
        init_cost = init_cost[:, None]
    if init_cost.ndim != 2 or init_cost.shape[0] != logits.shape[0] or init_cost.shape[1] != 1:
        raise ValueError("init_cost must have shape (B,) or (B,1)")

    logits = logits.float()
    candidate_cost = candidate_cost.to(device=logits.device, dtype=logits.dtype)
    init_cost = init_cost.to(device=logits.device, dtype=logits.dtype)
    valid = torch.isfinite(logits) & torch.isfinite(candidate_cost) & torch.isfinite(init_cost)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=logits.device).bool()

    raw_improvement = init_cost - candidate_cost
    positive_improvement = (raw_improvement - float(min_improvement_m)).clamp_min(0.0).masked_fill(~valid, 0.0)
    has_positive = positive_improvement.sum(dim=1, keepdim=True) > 1.0e-8
    temp = max(float(target_temperature_m), 1.0e-6)
    improvement_target_logits = (positive_improvement / temp).masked_fill(~valid, -1.0e6)
    fallback_target_logits = (-candidate_cost / temp).masked_fill(~valid, -1.0e6)
    target_logits = torch.where(has_positive, improvement_target_logits, fallback_target_logits)
    target_probs = F.softmax(target_logits, dim=1).detach()

    selection_log_probs = F.log_softmax(logits.masked_fill(~valid, -1.0e6), dim=1)
    loss_per = -(target_probs * selection_log_probs).sum(dim=1)
    active_rows = valid.any(dim=1)
    loss = loss_per[active_rows].mean() if bool(active_rows.any()) else logits.new_zeros(())

    masked_logits = logits.masked_fill(~valid, -1.0e6)
    target_score = torch.where(has_positive, positive_improvement, -candidate_cost).masked_fill(~valid, -1.0e6)
    pred_idx = masked_logits.argmax(dim=1)
    target_idx = target_score.argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    pred_improvement = raw_improvement[batch_idx, pred_idx]
    target_improvement = raw_improvement[batch_idx, target_idx]
    return {
        "loss": loss,
        "active": has_positive.squeeze(1).float().mean(),
        "pred_improvement_m": pred_improvement.mean(),
        "target_improvement_m": target_improvement.mean(),
        "pred_cost_m": candidate_cost[batch_idx, pred_idx].mean(),
        "target_cost_m": candidate_cost[batch_idx, target_idx].mean(),
        "pred_index": pred_idx.detach(),
        "target_index": target_idx.detach(),
    }


def candidate_teacher_quality_scores_from_batch(
    batch: Dict,
    *,
    valid_mask: torch.Tensor,
    target_mode: str = "pnp_composite",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-candidate teacher quality derived from render-LoFTR/PnP fields.

    These fields are offline teacher evidence, not final evaluation labels.  A
    high score means the query-render pair produced many confident, geometrically
    consistent matches, which is the signal the localization feature should
    learn to expose.
    """
    valid = valid_mask.bool()
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must have shape (B,K), got {tuple(valid.shape)}")
    has_teacher = any(
        key in batch
        for key in (
            "retrieval_pnp_success_candidates",
            "retrieval_pnp_num_inliers_candidates",
            "retrieval_pnp_num_matches_candidates",
            "retrieval_pnp_inlier_ratio_candidates",
            "retrieval_pnp_inlier_conf_mean_candidates",
        )
    )
    zeros = torch.zeros(valid.shape, device=valid.device, dtype=torch.float32)
    if not has_teacher:
        return zeros, torch.zeros_like(valid)

    features, names = candidate_quality_features_from_batch(batch, valid_mask=valid)
    features = features.to(device=valid.device, dtype=torch.float32)
    by_name = {name: features[..., idx] for idx, name in enumerate(names)}
    mode = str(target_mode or "pnp_composite").lower()
    if mode in {"inliers", "num_inliers", "pnp_inliers"}:
        quality = by_name["retrieval_pnp_num_inliers_log_z"]
    elif mode in {"inlier_ratio", "ratio"}:
        quality = by_name["retrieval_pnp_inlier_ratio"]
    elif mode in {"confidence", "conf", "inlier_conf"}:
        quality = by_name["retrieval_pnp_inlier_conf_mean"]
    elif mode in {"reproj", "reproj_quality", "median_reproj"}:
        quality = by_name["retrieval_pnp_reproj_median_quality"]
    elif mode in {"pnp_composite", "composite", "render_loftr_pnp"}:
        quality = (
            1.25 * by_name["retrieval_pnp_num_inliers_log_z"]
            + 0.50 * by_name["retrieval_pnp_num_matches_log_z"]
            + 1.00 * by_name["retrieval_pnp_inlier_ratio"]
            + 0.75 * by_name["retrieval_pnp_inlier_conf_mean"]
            + 0.50 * by_name["retrieval_pnp_reproj_median_quality"]
            + 0.50 * by_name["retrieval_pnp_success"]
        )
    else:
        raise ValueError(f"unsupported candidate teacher quality target mode: {target_mode}")
    quality = torch.where(torch.isfinite(quality) & valid, quality, zeros)
    row_span = quality.masked_fill(~valid, -1.0e6).max(dim=1).values - quality.masked_fill(~valid, 1.0e6).min(dim=1).values
    active = valid & torch.isfinite(quality) & (row_span[:, None] > 1.0e-6)
    return quality, active


def candidate_teacher_quality_listwise_loss(
    logits: torch.Tensor,
    batch: Dict,
    *,
    valid_mask: torch.Tensor,
    target_mode: str = "pnp_composite",
    temperature: float = 0.5,
    pairwise_weight: float = 0.1,
    pairwise_min_gap: float = 0.25,
    logit_margin: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Train candidate scores to match offline render-LoFTR/PnP teacher quality."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (B,K), got {tuple(logits.shape)}")
    valid = valid_mask.to(device=logits.device).bool()
    if valid.shape != logits.shape:
        raise ValueError(f"valid_mask must have shape {tuple(logits.shape)}, got {tuple(valid.shape)}")
    quality, active = candidate_teacher_quality_scores_from_batch(batch, valid_mask=valid, target_mode=target_mode)
    quality = quality.to(device=logits.device, dtype=logits.dtype)
    active = active.to(device=logits.device).bool()
    active_rows = active.any(dim=1)
    zero = logits.new_zeros(())
    if not bool(active_rows.any()):
        return {
            "loss": zero,
            "ce_loss": zero,
            "pairwise_loss": zero,
            "active": zero,
            "top1_acc": zero,
            "pred_quality": zero,
            "target_quality": zero,
            "pred_index": torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long),
            "target_index": torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long),
        }

    temp = max(float(temperature), 1.0e-6)
    masked_logits = (logits.float() / temp).masked_fill(~active, -1.0e6)
    target_logits = (quality.float() / temp).masked_fill(~active, -1.0e6)
    target_probs = F.softmax(target_logits, dim=1).detach()
    log_probs = F.log_softmax(masked_logits, dim=1)
    ce_per_row = -(target_probs * log_probs).sum(dim=1)
    ce_loss = ce_per_row[active_rows].mean()

    target_idx = quality.masked_fill(~active, -1.0e6).argmax(dim=1)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    target_quality = quality[batch_idx, target_idx][:, None]
    target_logit = masked_logits[batch_idx, target_idx][:, None]
    worse = active & (quality + float(pairwise_min_gap) < target_quality)
    pairwise_loss = zero
    if bool(worse.any()):
        pairwise_loss = F.softplus(masked_logits - target_logit + float(logit_margin))[worse].mean()
    pred_idx = masked_logits.argmax(dim=1)
    pred_quality = quality[batch_idx, pred_idx]
    total = ce_loss + float(pairwise_weight) * pairwise_loss
    return {
        "loss": total,
        "ce_loss": ce_loss,
        "pairwise_loss": pairwise_loss,
        "active": active_rows.float().mean(),
        "top1_acc": ((pred_idx == target_idx) & active_rows).float().sum() / active_rows.float().sum().clamp(min=1.0),
        "pred_quality": pred_quality[active_rows].mean(),
        "target_quality": quality[batch_idx, target_idx][active_rows].mean(),
        "pred_index": pred_idx.detach(),
        "target_index": target_idx.detach(),
    }


def effective_candidate_teacher_quality_weight(args: argparse.Namespace) -> float:
    """Return the current teacher-quality auxiliary weight.

    Stage 1/2 should learn pose/correction ordering first and only then let
    LoFTR/PnP quality act as reliability evidence.  This avoids letting a
    noisy matching teacher dominate the pose-cost target from step 0.
    """
    base = float(getattr(args, "candidate_teacher_quality_weight", 0.0) or 0.0)
    if base <= 0.0:
        return 0.0
    step = int(getattr(args, "current_step", 0) or 0)
    start = int(getattr(args, "candidate_teacher_quality_start_step", 0) or 0)
    warmup = int(getattr(args, "candidate_teacher_quality_warmup_steps", 0) or 0)
    if step < start:
        return 0.0
    if warmup <= 0:
        return base
    progress = min(max((step - start) / float(warmup), 0.0), 1.0)
    return base * progress


def score_anti_identity_loss(
    logits: torch.Tensor,
    candidate_cost: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    identity_mask: torch.Tensor | None = None,
    identity_index: int = 0,
    min_gap_m: float = 0.03,
    logit_margin: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Penalize selecting the center/identity candidate when a better candidate exists."""
    if logits.ndim != 2 or candidate_cost.ndim != 2:
        raise ValueError("logits and candidate_cost must have shape (B,K)")
    if logits.shape != candidate_cost.shape:
        raise ValueError("logits and candidate_cost shapes must match")
    identity_index = int(identity_index)
    if identity_index < 0 or identity_index >= logits.shape[1]:
        raise ValueError(f"identity_index={identity_index} is out of range for K={logits.shape[1]}")

    logits = logits.float()
    candidate_cost = candidate_cost.to(device=logits.device, dtype=logits.dtype)
    valid = torch.isfinite(logits) & torch.isfinite(candidate_cost)
    if valid_mask is not None:
        valid = valid & valid_mask.to(device=logits.device).bool()

    if identity_mask is not None:
        if identity_mask.shape != logits.shape:
            raise ValueError("identity_mask must have shape (B,K)")
        identity_valid = identity_mask.to(device=logits.device).bool() & valid
    else:
        identity_index = int(identity_index)
        if identity_index < 0 or identity_index >= logits.shape[1]:
            raise ValueError(f"identity_index={identity_index} is out of range for K={logits.shape[1]}")
        identity_valid = torch.zeros_like(valid)
        identity_valid[:, identity_index] = valid[:, identity_index]

    has_identity = identity_valid.any(dim=1, keepdim=True)
    identity_cost = candidate_cost.masked_fill(~identity_valid, float("inf")).min(dim=1, keepdim=True).values
    identity_logit = logits.masked_fill(~identity_valid, -1.0e6).max(dim=1, keepdim=True).values
    better = valid & has_identity & ~identity_valid & (candidate_cost + float(min_gap_m) < identity_cost)
    active = better.float().mean()
    if bool(better.any()):
        loss = F.softplus(identity_logit - logits + float(logit_margin))[better].mean()
        better_margin = (logits - identity_logit)[better].mean()
    else:
        loss = logits.new_zeros(())
        better_margin = logits.new_zeros(())

    pred_idx = logits.masked_fill(~valid, -1.0e6).argmax(dim=1)
    selected_identity = identity_valid.gather(1, pred_idx[:, None]).squeeze(1)
    return {
        "loss": loss,
        "active": active,
        "better_margin": better_margin,
        "selected_identity_frac": selected_identity.float().mean(),
    }


def _rename_teacher_metrics(metrics: Dict[str, torch.Tensor], *, source_prefix: str, target_prefix: str) -> Dict[str, torch.Tensor]:
    renamed: Dict[str, torch.Tensor] = {}
    for key, value in metrics.items():
        new_key = str(key)
        if new_key.startswith(source_prefix):
            new_key = target_prefix + new_key[len(source_prefix) :]
        renamed[new_key] = value
    return renamed


def nvs_teacher_correspondence_loss(
    query_loc: torch.Tensor,
    gt_render_loc: torch.Tensor,
    batch: Dict,
    *,
    weight: float = 0.0,
    patch_weight: float = 0.0,
    temperature: float = 0.07,
    min_confidence: float = 0.0,
    min_points: int = 4,
    negative_exclusion_px: float = 0.0,
    positive_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.1,
    patch_radius: int = 2,
    patch_temperature: float | None = None,
    patch_min_points: int | None = None,
    patch_positive_weight: float = 0.0,
    patch_margin_weight: float = 0.0,
    patch_margin: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train NVS loc features from offline matching-teacher correspondences.

    The teacher supplies sparse query↔GT-render matches.  This loss supervises
    the query/render localization adapter directly, before candidate scoring
    compresses the evidence into scalar logits.
    """
    if gt_render_loc.ndim == 5:
        if gt_render_loc.shape[1] != 1:
            raise ValueError("gt_render_loc must have one GT render candidate when 5D")
        gt_render_feat = gt_render_loc[:, 0]
    elif gt_render_loc.ndim == 4:
        gt_render_feat = gt_render_loc
    else:
        raise ValueError("gt_render_loc must have shape (B,C,H,W) or (B,1,C,H,W)")

    required = (
        "teacher_corr_query_xy",
        "teacher_corr_map_xy",
        "teacher_corr_conf",
        "teacher_corr_valid",
    )
    missing = [key for key in required if key not in batch]
    zero = query_loc.new_zeros(())
    metrics: Dict[str, torch.Tensor] = {
        "nvs_teacher_corr_missing": query_loc.new_tensor(1.0 if missing else 0.0),
        "nvs_teacher_corr_loss": zero.detach(),
        "nvs_teacher_patch_loss": zero.detach(),
        "nvs_teacher_corr_weighted_loss": zero.detach(),
        "nvs_teacher_patch_weighted_loss": zero.detach(),
    }
    if missing:
        return zero, metrics

    corr_loss = zero
    if float(weight) > 0.0:
        corr_loss, corr_metrics = sparse_teacher_correspondence_loss(
            query_loc,
            gt_render_feat,
            batch["teacher_corr_query_xy"],
            batch["teacher_corr_map_xy"],
            batch["teacher_corr_conf"],
            batch["teacher_corr_valid"],
            xy_source_hw=batch.get("teacher_corr_hw"),
            temperature=float(temperature),
            min_confidence=float(min_confidence),
            min_points=int(min_points),
            negative_exclusion_px=float(negative_exclusion_px),
            positive_weight=float(positive_weight),
            margin_weight=float(margin_weight),
            margin=float(margin),
        )
        metrics.update(
            _rename_teacher_metrics(
                corr_metrics,
                source_prefix="map_teacher_corr",
                target_prefix="nvs_teacher_corr",
            )
        )

    patch_loss = zero
    if float(patch_weight) > 0.0:
        patch_loss, patch_metrics = sparse_teacher_local_patch_loss(
            query_loc,
            gt_render_feat,
            batch["teacher_corr_query_xy"],
            batch["teacher_corr_map_xy"],
            batch["teacher_corr_conf"],
            batch["teacher_corr_valid"],
            xy_source_hw=batch.get("teacher_corr_hw"),
            radius=int(patch_radius),
            temperature=float(patch_temperature if patch_temperature is not None else temperature),
            min_confidence=float(min_confidence),
            min_points=int(patch_min_points if patch_min_points is not None else min_points),
            positive_weight=float(patch_positive_weight),
            margin_weight=float(patch_margin_weight),
            margin=float(patch_margin),
        )
        metrics.update(
            _rename_teacher_metrics(
                patch_metrics,
                source_prefix="map_teacher_patch",
                target_prefix="nvs_teacher_patch",
            )
        )

    total = float(weight) * corr_loss + float(patch_weight) * patch_loss
    metrics.update(
        {
            "nvs_teacher_corr_loss": corr_loss.detach(),
            "nvs_teacher_patch_loss": patch_loss.detach(),
            "nvs_teacher_corr_weighted_loss": (float(weight) * corr_loss).detach(),
            "nvs_teacher_patch_weighted_loss": (float(patch_weight) * patch_loss).detach(),
        }
    )
    return total, metrics


def nvs_teacher_pair_match_loss(
    matcher: PairConditionedLocalMatcher | None,
    query_loc: torch.Tensor,
    gt_render_loc: torch.Tensor,
    batch: Dict,
    *,
    weight: float = 0.0,
    radius: int = 3,
    temperature: float = 0.05,
    min_confidence: float = 0.0,
    min_points: int = 4,
    positive_weight: float = 0.0,
    margin_weight: float = 0.0,
    margin: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Teacher-supervised pair-conditioned local heatmap loss.

    Unlike the scalar candidate score losses, this keeps the local matching
    evidence structured: each teacher query point must select the center of a
    rendered-map patch through a learned pair-conditioned matcher.
    """
    zero = query_loc.new_zeros(())
    metrics: Dict[str, torch.Tensor] = {
        "nvs_pair_match_missing": query_loc.new_tensor(0.0 if matcher is not None else 1.0),
        "nvs_pair_match_loss": zero.detach(),
        "nvs_pair_match_weighted_loss": zero.detach(),
        "nvs_pair_match_acc": zero.detach(),
        "nvs_pair_match_pos": zero.detach(),
        "nvs_pair_match_neg": zero.detach(),
        "nvs_pair_match_gap": zero.detach(),
        "nvs_pair_match_soft_epe": zero.detach(),
        "nvs_pair_match_points": zero.detach(),
        "nvs_pair_match_cov": zero.detach(),
        "nvs_pair_match_skipped_no_points": query_loc.new_tensor(1.0),
    }
    if matcher is None or float(weight) <= 0.0:
        return zero, metrics
    if gt_render_loc.ndim == 5:
        if gt_render_loc.shape[1] != 1:
            raise ValueError("gt_render_loc must have one GT render candidate when 5D")
        map_feat = gt_render_loc[:, 0]
    elif gt_render_loc.ndim == 4:
        map_feat = gt_render_loc
    else:
        raise ValueError("gt_render_loc must have shape (B,C,H,W) or (B,1,C,H,W)")
    required = (
        "teacher_corr_query_xy",
        "teacher_corr_map_xy",
        "teacher_corr_conf",
        "teacher_corr_valid",
    )
    missing = [key for key in required if key not in batch]
    metrics["nvs_pair_match_missing"] = query_loc.new_tensor(1.0 if missing else 0.0)
    if missing:
        return zero, metrics

    with torch.cuda.amp.autocast(enabled=False):
        q_feat = query_loc.float()
        m_feat = map_feat.float()
        query_xy = batch["teacher_corr_query_xy"].to(device=q_feat.device).float()
        map_xy = batch["teacher_corr_map_xy"].to(device=q_feat.device).float()
        confidence = batch["teacher_corr_conf"].to(device=q_feat.device).float()
        valid = batch["teacher_corr_valid"].to(device=q_feat.device).float()
        if confidence.ndim == 3:
            confidence = confidence.squeeze(-1)
        if valid.ndim == 3:
            valid = valid.squeeze(-1)
        xy_source_hw = batch.get("teacher_corr_hw")

        q_sparse, q_in = sample_feature_at_xy(q_feat, query_xy, valid, source_hw=xy_source_hw)
        _, m_in = sample_feature_at_xy(m_feat, map_xy, valid, source_hw=xy_source_hw)
        if xy_source_hw is not None:
            map_xy_feat = _scale_xy_tensor(map_xy, xy_source_hw, m_feat.shape[-2:])
        else:
            map_xy_feat = map_xy
        usable = (valid > 0) & q_in & m_in & torch.isfinite(confidence) & (confidence >= float(min_confidence))
        usable = usable & torch.isfinite(q_sparse).all(dim=-1) & torch.isfinite(map_xy_feat).all(dim=-1)

        r = max(int(radius), 0)
        offset_axis = torch.arange(-r, r + 1, device=q_feat.device, dtype=q_feat.dtype)
        dy, dx = torch.meshgrid(offset_axis, offset_axis, indexing="ij")
        offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
        center_index = int(((offsets[:, 0] == 0) & (offsets[:, 1] == 0)).nonzero(as_tuple=False)[0].item())

        losses = []
        acc_values = []
        pos_values = []
        neg_values = []
        gap_values = []
        soft_epe_values = []
        point_counts = []
        temp = max(float(temperature), 1.0e-6)
        for batch_idx in range(q_sparse.shape[0]):
            mask_b = usable[batch_idx]
            if int(mask_b.sum().item()) < int(min_points):
                continue
            q_b = q_sparse[batch_idx, mask_b]
            xy_b = map_xy_feat[batch_idx, mask_b].to(device=q_feat.device, dtype=q_feat.dtype)
            patch_xy = xy_b[:, None, :] + offsets[None, :, :]
            flat_xy = patch_xy.reshape(1, -1, 2)
            flat_valid = torch.ones(1, flat_xy.shape[1], device=q_feat.device, dtype=q_feat.dtype)
            patch_sparse, patch_in = sample_feature_at_xy(
                m_feat[batch_idx : batch_idx + 1],
                flat_xy,
                flat_valid,
                source_hw=None,
            )
            n_points = q_b.shape[0]
            n_offsets = offsets.shape[0]
            patch_sparse = patch_sparse.view(n_points, n_offsets, q_b.shape[-1])
            patch_in = patch_in.view(n_points, n_offsets)
            center_valid = patch_in[:, center_index]
            if int(center_valid.sum().item()) < int(min_points):
                continue
            q_b = q_b[center_valid]
            patch_sparse = patch_sparse[center_valid]
            patch_in = patch_in[center_valid]
            w_b = confidence[batch_idx, mask_b][center_valid].clamp(min=0.0)
            if float(w_b.sum().item()) <= 0.0:
                w_b = torch.ones_like(w_b)

            q_center_n = F.normalize(q_b, dim=-1, eps=1.0e-6)
            patch_center_n = F.normalize(patch_sparse[:, center_index], dim=-1, eps=1.0e-6)
            center_pos_cos = (q_center_n * patch_center_n).sum(dim=-1)
            logits = matcher(q_b, patch_sparse, offsets=offsets, patch_valid=patch_in)
            targets = torch.full((logits.shape[0],), center_index, device=logits.device, dtype=torch.long)
            per_point = F.cross_entropy(logits / temp, targets, reduction="none")
            pos = logits[:, center_index]
            other_mask = patch_in.clone()
            other_mask[:, center_index] = False
            hard_neg = logits.masked_fill(~other_mask, -1.0e4).max(dim=1).values
            has_neg = other_mask.any(dim=1)
            hard_neg = torch.where(has_neg, hard_neg, torch.zeros_like(hard_neg))
            if float(positive_weight) > 0.0:
                per_point = per_point + float(positive_weight) * (1.0 - center_pos_cos)
            if float(margin_weight) > 0.0:
                margin_loss = F.relu(hard_neg - pos + float(margin))
                per_point = per_point + float(margin_weight) * margin_loss * has_neg.float()
            losses.append((per_point * w_b).sum() / w_b.sum().clamp(min=1.0e-6))

            with torch.no_grad():
                acc_values.append(((logits.argmax(dim=1) == center_index).float() * w_b).sum() / w_b.sum().clamp(min=1.0e-6))
                pos_values.append((pos * w_b).sum() / w_b.sum().clamp(min=1.0e-6))
                neg_values.append((hard_neg * w_b).sum() / w_b.sum().clamp(min=1.0e-6))
                gap_values.append(((pos - hard_neg) * w_b).sum() / w_b.sum().clamp(min=1.0e-6))
                probs = F.softmax(logits / temp, dim=1)
                expected_offset = probs @ offsets.to(device=logits.device, dtype=logits.dtype)
                soft_epe = torch.linalg.vector_norm(expected_offset, dim=1)
                soft_epe_values.append((soft_epe * w_b).sum() / w_b.sum().clamp(min=1.0e-6))
                point_counts.append(center_valid.float().sum())

        if losses:
            loss = torch.stack(losses).mean()
            metrics.update(
                {
                    "nvs_pair_match_loss": loss.detach(),
                    "nvs_pair_match_weighted_loss": (float(weight) * loss).detach(),
                    "nvs_pair_match_acc": torch.stack(acc_values).mean().detach(),
                    "nvs_pair_match_pos": torch.stack(pos_values).mean().detach(),
                    "nvs_pair_match_neg": torch.stack(neg_values).mean().detach(),
                    "nvs_pair_match_gap": torch.stack(gap_values).mean().detach(),
                    "nvs_pair_match_soft_epe": torch.stack(soft_epe_values).mean().detach(),
                    "nvs_pair_match_points": torch.stack(point_counts).mean().detach(),
                    "nvs_pair_match_cov": usable.float().mean().detach(),
                    "nvs_pair_match_skipped_no_points": q_feat.new_tensor(0.0),
                }
            )
        else:
            loss = zero
            metrics["nvs_pair_match_cov"] = usable.float().mean().detach()
    return float(weight) * loss, metrics


def forward_batch(
    model,
    adapter: PoseFeatureDomainAdapter,
    energy_net: PoseEnergyNet | None,
    pair_matcher: PairConditionedLocalMatcher | None,
    map_renderer,
    batch,
    cfg: Dict,
    args: argparse.Namespace,
    *,
    train: bool,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = next(adapter.parameters()).device
    batch = move_batch_to_device(batch, device)
    pose_gt = map_pose_gt_for_batch(map_renderer, batch, device)
    batch, pose_gt, synthetic_mask = maybe_replace_with_synthetic_queries(batch, map_renderer, pose_gt, cfg, args)
    external_candidate_valid = None
    if str(getattr(args, "candidate_bank_mode", "lattice") or "lattice").lower() == "cache":
        if "pose_init_candidates" not in batch:
            raise KeyError("candidate_bank_mode=cache requires batch['pose_init_candidates']")
        candidate_poses = batch["pose_init_candidates"].to(device=device, dtype=torch.float32)
        topk = max(1, min(int(args.topk), int(candidate_poses.shape[1])))
        candidate_poses = candidate_poses[:, :topk]
        if "candidate_valid_mask" in batch:
            external_candidate_valid = batch["candidate_valid_mask"].to(device=device).bool()[:, :topk]
        else:
            external_candidate_valid = torch.ones(candidate_poses.shape[:2], device=device, dtype=torch.bool)
        if "pose_init" in batch:
            init_pose = batch["pose_init"].to(device=device, dtype=torch.float32)
        else:
            init_pose = candidate_poses[:, 0].detach().clone()
    else:
        init_pose = candidate_center_pose(pose_gt, args)
        candidate_poses = build_nvs_candidate_bank(init_pose, pose_gt, args, train=train)

    with torch.no_grad():
        gt_batch = map_renderer.attach_pose_candidate_renders(
            dict(batch),
            pose_gt[:, None],
            prefix="nvs_gt",
            require_grad=False,
            feature="all",
            include_aux=True,
        )
        cand_batch = map_renderer.attach_pose_candidate_renders(
            dict(batch),
            candidate_poses,
            prefix="nvs_candidate",
            require_grad=False,
            candidate_valid_mask=external_candidate_valid,
            feature="all",
            include_aux=True,
        )
    query_fine_key = args.query_fine_key or cfg.get("map_supervision", {}).get("query_fine_key", "fine")
    train_query_model = train and any(param.requires_grad for param in model.parameters())
    with torch.set_grad_enabled(train_query_model):
        with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
            if bool(cfg.get("model", {}).get("teacher_fine_condition", False)):
                outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
            else:
                outputs = model(batch["rgb"])
    query_base = outputs[query_fine_key].float()
    if not train_query_model:
        query_base = query_base.detach()
    gt_render_base = gt_batch["nvs_gt_fine"].float().detach()
    cand_render_base = cand_batch["nvs_candidate_fine"].float().detach()
    gt_mask = gt_batch.get("nvs_gt_mask")
    cand_mask = cand_batch.get("nvs_candidate_mask")
    gt_depth = gt_batch.get("nvs_gt_depth")
    cand_depth = cand_batch.get("nvs_candidate_depth")
    cand_position = cand_batch.get("nvs_candidate_position")
    gt_rgb = gt_batch.get("nvs_gt_rgb")
    cand_rgb = cand_batch.get("nvs_candidate_rgb")
    gt_intrinsics = gt_batch.get("nvs_gt_intrinsics", cand_batch.get("nvs_candidate_intrinsics"))
    candidate_pose = cand_batch["nvs_candidate_pose"].float().detach()

    energy_use_uncertainty = bool(args.pose_energy_use_uncertainty)
    score_use_uncertainty = bool(args.score_use_uncertainty)
    project_uncertainty = energy_use_uncertainty or score_use_uncertainty
    render_chunk_size = int(cfg.get("map_supervision", {}).get("candidate_render_score_projector_chunk_size", 0) or 0)
    adapter_needs_rgb = bool(getattr(args, "pose_feature_adapter_rgb_context_enabled", False)) or bool(
        getattr(args, "pose_feature_adapter_texture_branch_enabled", False)
    )
    query_rgb = batch.get("rgb") if adapter_needs_rgb else None
    gt_project_rgb = gt_rgb if adapter_needs_rgb else None
    cand_project_rgb = cand_rgb if adapter_needs_rgb else None
    if project_uncertainty:
        query_loc, query_uncertainty = adapter.project_query_with_uncertainty(query_base, rgb=query_rgb)
        gt_render_loc, _gt_uncertainty = project_render_bank_with_uncertainty(
            adapter,
            gt_render_base,
            render_rgb=gt_project_rgb,
            render_chunk_size=0,
        )
        cand_render_loc, cand_uncertainty = project_render_bank_with_uncertainty(
            adapter,
            cand_render_base,
            render_rgb=cand_project_rgb,
            render_chunk_size=render_chunk_size,
        )
    else:
        query_uncertainty = None
        cand_uncertainty = None
        query_loc = adapter.project_query(query_base, rgb=query_rgb)
        gt_render_loc = project_render_bank(
            adapter,
            gt_render_base,
            render_rgb=gt_project_rgb,
            render_chunk_size=0,
        )
        cand_render_loc = project_render_bank(
            adapter,
            cand_render_base,
            render_rgb=cand_project_rgb,
            render_chunk_size=render_chunk_size,
        )
    score_weight = None
    if score_use_uncertainty and query_uncertainty is not None and cand_uncertainty is not None:
        query_weight = _resize_feature(query_uncertainty.float(), tuple(cand_render_loc.shape[-2:]))
        score_weight = query_weight[:, None] * cand_uncertainty.float()
        if cand_mask is not None:
            cand_weight = _resize_mask(cand_mask, tuple(cand_render_loc.shape[-2:]))
            if cand_weight.ndim == 4:
                cand_weight = cand_weight[:, :, None] if cand_weight.shape[1] == cand_render_loc.shape[1] else cand_weight[:, None]
            score_weight = score_weight * cand_weight.to(device=score_weight.device, dtype=score_weight.dtype)
    score_feature_hw = parse_score_feature_hw(getattr(args, "score_feature_hw", None))
    score_query_loc, score_cand_render_loc, score_cand_mask, score_weight = resize_score_candidate_tensors(
        query_loc,
        cand_render_loc,
        cand_mask,
        score_weight,
        score_hw=score_feature_hw,
    )

    align_loss, align_cos = masked_dense_alignment_loss(
        query_loc,
        gt_render_loc,
        mask=gt_mask,
        margin=float(args.align_margin),
    )
    if float(args.warp_align_weight) > 0.0:
        if cand_position is None or gt_intrinsics is None:
            raise KeyError("warp alignment requires nvs_candidate_position and nvs_gt_intrinsics")
        target_intrinsics = gt_intrinsics[:, 0] if gt_intrinsics.ndim == 3 and gt_intrinsics.shape[-1] == 4 else gt_intrinsics
        warp_loss, warp_metrics = warped_candidate_alignment_loss(
            query_loc,
            cand_render_loc,
            cand_position.float().detach(),
            pose_gt.float(),
            target_intrinsics,
            candidate_mask=cand_mask,
            target_depth=gt_depth,
            target_mask=gt_mask,
            depth_abs_tolerance_m=float(args.warp_depth_tolerance_m),
            depth_rel_tolerance=float(args.warp_depth_tolerance_rel),
        )
    else:
        warp_loss = query_loc.new_zeros(())
        warp_metrics = {
            "warp_align_loss": warp_loss.detach(),
            "warp_align_cos": warp_loss.detach(),
            "warp_valid_frac": warp_loss.detach(),
            "warp_in_bounds_frac": warp_loss.detach(),
        }
        target_intrinsics = gt_intrinsics[:, 0] if gt_intrinsics is not None and gt_intrinsics.ndim == 3 and gt_intrinsics.shape[-1] == 4 else gt_intrinsics

    local_stats = {
        "local_zero_score": query_loc.new_zeros(()),
        "local_peak_score": query_loc.new_zeros(()),
        "local_peak_gap": query_loc.new_zeros(()),
        "local_peak_offset_px": query_loc.new_zeros(()),
        "local_expected_offset_px": query_loc.new_zeros(()),
    }
    flow_loss = query_loc.new_zeros(())
    flow_metrics = {
        "local_flow_nce_loss": flow_loss.detach(),
        "local_flow_valid_frac": flow_loss.detach(),
        "local_flow_target_offset_px": flow_loss.detach(),
        "local_flow_candidate_selected_frac": flow_loss.detach(),
        "local_flow_best_active": flow_loss.detach(),
        "local_flow_hard_negative_active": flow_loss.detach(),
    }
    flow_inputs = None
    score_mode = str(args.score_mode or "same_pixel")
    need_local_corr = score_mode.startswith("local_") or float(args.local_flow_nce_weight) > 0.0
    if score_mode == "pair_matcher_local":
        if pair_matcher is None:
            raise ValueError("score_mode=pair_matcher_local requires --pair-matcher-enabled")
        cand_scores, local_stats = pair_matcher_local_candidate_scores(
            pair_matcher,
            score_query_loc,
            score_cand_render_loc,
            mask=score_cand_mask,
            radius=int(args.pair_matcher_radius),
            stride=int(args.pair_matcher_score_stride),
            temperature=float(args.pair_matcher_temperature),
            chunk_points=int(args.pair_matcher_score_chunk_points),
            candidate_chunk_size=int(args.pair_matcher_score_candidate_chunk_size),
            offset_chunk_size=int(args.pair_matcher_score_offset_chunk_size),
            candidate_score_mode=str(args.pair_matcher_candidate_score_mode),
        )
        if float(args.local_flow_nce_weight) > 0.0:
            if cand_position is None or gt_intrinsics is None:
                raise KeyError("local flow NCE requires nvs_candidate_position and nvs_gt_intrinsics")
            local_corr, local_valid = local_correlation_volume(
                score_query_loc,
                score_cand_render_loc,
                mask=score_cand_mask,
                radius=int(args.local_corr_radius),
            )
            flow_position, flow_intrinsics = _flow_geometry_for_score_hw(
                cand_position.float().detach(),
                target_intrinsics,
                source_hw=tuple(query_loc.shape[-2:]),
                score_hw=tuple(score_cand_render_loc.shape[-2:]),
            )
            flow_inputs = (local_corr, local_valid, flow_position, flow_intrinsics, score_cand_mask)
    elif need_local_corr:
        local_corr, local_valid = local_correlation_volume(
            score_query_loc,
            score_cand_render_loc,
            mask=score_cand_mask,
            radius=int(args.local_corr_radius),
        )
        if score_mode == "local_zero_offset":
            cand_scores, local_stats = local_zero_offset_scores_from_corr(
                local_corr,
                local_valid,
                radius=int(args.local_corr_radius),
                temperature=float(args.local_corr_temperature),
                peak_gap_weight=float(args.local_zero_peak_gap_weight),
                offset_weight=float(args.local_zero_offset_weight),
                score_mode=score_mode,
                weight=score_weight,
            )
        elif score_mode in ("local_zero_only", "local_neg_expected_offset", "local_neg_peak_offset"):
            cand_scores, local_stats = local_zero_offset_scores_from_corr(
                local_corr,
                local_valid,
                radius=int(args.local_corr_radius),
                temperature=float(args.local_corr_temperature),
                peak_gap_weight=float(args.local_zero_peak_gap_weight),
                offset_weight=float(args.local_zero_offset_weight),
                score_mode=score_mode,
                weight=score_weight,
            )
        else:
            cand_scores = masked_dense_cosine(
                score_query_loc,
                score_cand_render_loc,
                mask=score_weight if score_weight is not None else score_cand_mask,
            )
        if float(args.local_flow_nce_weight) > 0.0:
            if cand_position is None or gt_intrinsics is None:
                raise KeyError("local flow NCE requires nvs_candidate_position and nvs_gt_intrinsics")
            flow_position, flow_intrinsics = _flow_geometry_for_score_hw(
                cand_position.float().detach(),
                target_intrinsics,
                source_hw=tuple(query_loc.shape[-2:]),
                score_hw=tuple(score_cand_render_loc.shape[-2:]),
            )
            flow_inputs = (local_corr, local_valid, flow_position, flow_intrinsics, score_cand_mask)
    else:
        cand_scores = masked_dense_cosine(
            score_query_loc,
            score_cand_render_loc,
            mask=score_weight if score_weight is not None else score_cand_mask,
        )
    teacher_corr_loss, teacher_corr_metrics = nvs_teacher_correspondence_loss(
        query_loc,
        gt_render_loc,
        batch,
        weight=float(args.teacher_corr_weight),
        patch_weight=float(args.teacher_corr_local_patch_weight),
        temperature=float(args.teacher_corr_temperature),
        min_confidence=float(args.teacher_corr_min_confidence),
        min_points=int(args.teacher_corr_min_points),
        negative_exclusion_px=float(args.teacher_corr_negative_exclusion_px),
        positive_weight=float(args.teacher_corr_positive_weight),
        margin_weight=float(args.teacher_corr_margin_weight),
        margin=float(args.teacher_corr_margin),
        patch_radius=int(args.teacher_corr_local_patch_radius),
        patch_temperature=float(args.teacher_corr_local_patch_temperature),
        patch_min_points=int(args.teacher_corr_local_patch_min_points),
        patch_positive_weight=float(args.teacher_corr_local_patch_positive_weight),
        patch_margin_weight=float(args.teacher_corr_local_patch_margin_weight),
        patch_margin=float(args.teacher_corr_local_patch_margin),
    )
    pair_match_loss, pair_match_metrics = nvs_teacher_pair_match_loss(
        pair_matcher,
        query_loc,
        gt_render_loc,
        batch,
        weight=float(args.pair_matcher_weight) if bool(args.pair_matcher_enabled) else 0.0,
        radius=int(args.pair_matcher_radius),
        temperature=float(args.pair_matcher_temperature),
        min_confidence=float(args.pair_matcher_min_confidence),
        min_points=int(args.pair_matcher_min_points),
        positive_weight=float(args.pair_matcher_positive_weight),
        margin_weight=float(args.pair_matcher_margin_weight),
        margin=float(args.pair_matcher_margin),
    )
    cand_valid = torch.isfinite(cand_scores)
    if cand_mask is not None:
        valid_mask = _resize_mask(score_cand_mask, tuple(score_cand_render_loc.shape[-2:]))
        cand_valid = cand_valid & (valid_mask.flatten(2).sum(dim=2) > 1.0)
    if external_candidate_valid is not None:
        cand_valid = cand_valid & external_candidate_valid.to(device=cand_valid.device).bool()
    pose_cost, _residual_target, trans_err, rot_err = pose_costs_and_residual_targets(
        candidate_pose,
        pose_gt.float(),
        valid_mask=cand_valid,
        rot_cost_weight=float(args.rot_cost_weight),
    )
    init_pose_cost, _init_residual, init_trans, init_rot = pose_costs_and_residual_targets(
        init_pose[:, None].float(),
        pose_gt.float(),
        rot_cost_weight=float(args.rot_cost_weight),
    )
    if flow_inputs is not None:
        flow_selection = None
        candidate_selection_mask = None
        flow_candidate_mode = str(getattr(args, "local_flow_nce_candidate_mode", "all") or "all").lower()
        if flow_candidate_mode != "all":
            flow_selection = hard_flow_candidate_selection_mask(
                cand_scores.detach(),
                pose_cost.detach(),
                cand_valid,
                mode=flow_candidate_mode,
                min_cost_gap_m=float(args.local_flow_nce_hard_negative_min_cost_gap_m),
            )
            candidate_selection_mask = flow_selection["selection_mask"]
        local_corr, local_valid, flow_position, flow_intrinsics, flow_candidate_mask = flow_inputs
        flow_loss, flow_metrics = local_flow_nce_loss_from_corr(
            local_corr,
            local_valid,
            flow_position,
            pose_gt.float(),
            flow_intrinsics,
            candidate_mask=flow_candidate_mask,
            candidate_selection_mask=candidate_selection_mask,
            target_depth=gt_depth,
            target_mask=gt_mask,
            radius=int(args.local_corr_radius),
            temperature=float(args.local_corr_temperature),
            depth_abs_tolerance_m=float(args.warp_depth_tolerance_m),
            depth_rel_tolerance=float(args.warp_depth_tolerance_rel),
        )
        if flow_selection is None:
            flow_metrics["local_flow_best_active"] = cand_valid.any(dim=1).float().mean().detach()
            flow_metrics["local_flow_hard_negative_active"] = flow_loss.new_zeros(()).detach()
        else:
            flow_metrics["local_flow_best_active"] = flow_selection["best_active"].detach()
            flow_metrics["local_flow_hard_negative_active"] = flow_selection["hard_negative_active"].detach()
    correction_cos = candidate_correction_cosines(candidate_pose, init_pose, pose_gt.float()).to(device=device)
    identity_mask = candidate_identity_mask(candidate_pose, init_pose).to(device=device)
    rank = rank_losses_from_scores(
        cand_scores,
        pose_cost.detach(),
        cand_valid,
        temperature_m=float(args.rank_temperature_m),
        pairwise_weight=float(args.rank_pairwise_weight),
        pairwise_min_gap_m=float(args.rank_pairwise_min_gap_m),
        pairwise_logit_margin=float(args.rank_pairwise_logit_margin),
    )
    score_correction_cosine = {
        "loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "pred_cos": query_loc.new_zeros(()),
        "target_cos": query_loc.new_zeros(()),
    }
    if float(args.score_correction_cosine_weight) > 0.0:
        score_correction_cosine = pose_energy_correction_cosine_soft_label_loss(
            cand_scores,
            correction_cos.detach(),
            valid_mask=cand_valid,
            target_temperature=float(args.score_correction_cosine_temperature),
            min_cos=float(args.score_correction_cosine_min_cos),
        )
    score_pose_improvement = {
        "loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "pred_improvement_m": query_loc.new_zeros(()),
        "target_improvement_m": query_loc.new_zeros(()),
        "pred_cost_m": query_loc.new_zeros(()),
        "target_cost_m": query_loc.new_zeros(()),
    }
    if float(args.score_pose_improvement_weight) > 0.0:
        score_pose_improvement = score_pose_improvement_soft_label_loss(
            cand_scores,
            pose_cost.detach(),
            init_pose_cost.detach(),
            valid_mask=cand_valid,
            target_temperature_m=float(args.score_pose_improvement_temperature_m),
            min_improvement_m=float(args.score_pose_improvement_min_improvement_m),
        )
    candidate_teacher_quality = {
        "loss": query_loc.new_zeros(()),
        "ce_loss": query_loc.new_zeros(()),
        "pairwise_loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "top1_acc": query_loc.new_zeros(()),
        "pred_quality": query_loc.new_zeros(()),
        "target_quality": query_loc.new_zeros(()),
    }
    candidate_teacher_quality_weight = effective_candidate_teacher_quality_weight(args)
    if candidate_teacher_quality_weight > 0.0:
        candidate_teacher_quality = candidate_teacher_quality_listwise_loss(
            cand_scores,
            batch,
            valid_mask=cand_valid,
            target_mode=str(args.candidate_teacher_quality_target_mode),
            temperature=float(args.candidate_teacher_quality_temperature),
            pairwise_weight=float(args.candidate_teacher_quality_pairwise_weight),
            pairwise_min_gap=float(args.candidate_teacher_quality_pairwise_min_gap),
            logit_margin=float(args.candidate_teacher_quality_logit_margin),
        )
    score_anti_identity = {
        "loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "better_margin": query_loc.new_zeros(()),
        "selected_identity_frac": query_loc.new_zeros(()),
    }
    if float(args.score_anti_identity_weight) > 0.0:
        score_anti_identity = score_anti_identity_loss(
            cand_scores / max(float(args.rank_temperature_m), 1.0e-6),
            pose_cost.detach(),
            valid_mask=cand_valid,
            identity_mask=identity_mask,
            identity_index=int(args.score_anti_identity_index),
            min_gap_m=float(args.score_anti_identity_min_gap_m),
            logit_margin=float(args.score_anti_identity_logit_margin),
        )
    candidate_observability = {
        "loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "gap": query_loc.new_zeros(()),
        "best_score": query_loc.new_zeros(()),
        "hard_negative_score": query_loc.new_zeros(()),
    }
    if float(args.candidate_observability_weight) > 0.0:
        candidate_observability = candidate_observability_margin_loss(
            cand_scores,
            pose_cost.detach(),
            cand_valid,
            margin=float(args.candidate_observability_margin),
            min_cost_gap_m=float(args.candidate_observability_min_cost_gap_m),
        )
    observability = {
        "loss": query_loc.new_zeros(()),
        "active": query_loc.new_zeros(()),
        "gt_score": query_loc.new_zeros(()),
        "hard_negative_score": query_loc.new_zeros(()),
        "gap": query_loc.new_zeros(()),
    }
    if float(args.observability_contrast_weight) > 0.0:
        obs_source = str(getattr(args, "observability_contrast_score_source", "dense") or "dense").lower()
        if obs_source == "selection_score" and score_mode == "pair_matcher_local":
            if pair_matcher is None:
                raise ValueError("selection_score observability with pair_matcher_local requires pair_matcher")
            gt_scores, _gt_local_stats = pair_matcher_local_candidate_scores(
                pair_matcher,
                query_loc,
                gt_render_loc,
                mask=gt_mask,
                radius=int(args.pair_matcher_radius),
                stride=int(args.pair_matcher_score_stride),
                temperature=float(args.pair_matcher_temperature),
                chunk_points=int(args.pair_matcher_score_chunk_points),
                candidate_chunk_size=1,
                offset_chunk_size=int(args.pair_matcher_score_offset_chunk_size),
                candidate_score_mode=str(args.pair_matcher_candidate_score_mode),
            )
            observability = observability_score_contrast_loss(
                gt_scores,
                cand_scores,
                pose_cost.detach(),
                cand_valid,
                margin=float(args.observability_contrast_margin),
                min_negative_cost_m=float(args.observability_negative_min_cost_m),
            )
        elif obs_source == "dense":
            observability = observability_contrast_loss(
                query_loc,
                gt_render_loc,
                cand_render_loc,
                pose_cost.detach(),
                cand_valid,
                gt_mask=gt_mask,
                candidate_mask=cand_mask,
                margin=float(args.observability_contrast_margin),
                min_negative_cost_m=float(args.observability_negative_min_cost_m),
            )
        else:
            raise ValueError(f"Unsupported observability_contrast_score_source={obs_source!r} for score_mode={score_mode!r}")
    energy_loss = query_loc.new_zeros(())
    pose_energy_metrics = {
        "pose_energy_loss": energy_loss.detach(),
        "pose_energy_energy_loss": energy_loss.detach(),
        "pose_energy_residual_loss": energy_loss.detach(),
        "pose_energy_translation_loss": energy_loss.detach(),
        "pose_energy_rotation_loss": energy_loss.detach(),
        "pose_energy_joint_loss": energy_loss.detach(),
        "pose_energy_confidence_loss": energy_loss.detach(),
        "pose_energy_improve_loss": energy_loss.detach(),
        "pose_energy_monotonicity_loss": energy_loss.detach(),
        "pose_energy_monotonicity_active": energy_loss.detach(),
        "pose_energy_monotonicity_score_gain": energy_loss.detach(),
        "pose_energy_anti_identity_loss": energy_loss.detach(),
        "pose_energy_anti_identity_active": energy_loss.detach(),
        "pose_energy_pairwise_rank_loss": energy_loss.detach(),
        "pose_energy_pairwise_rank_active": energy_loss.detach(),
        "pose_energy_direction_loss": energy_loss.detach(),
        "pose_energy_direction_active": energy_loss.detach(),
        "pose_energy_direction_pred_cos": energy_loss.detach(),
        "pose_energy_direction_target_cos": energy_loss.detach(),
        "pose_energy_correction_cosine_loss": energy_loss.detach(),
        "pose_energy_correction_cosine_active": energy_loss.detach(),
        "pose_energy_correction_cosine_pred_cos": energy_loss.detach(),
        "pose_energy_correction_cosine_target_cos": energy_loss.detach(),
        "pose_energy_top1_acc": energy_loss.detach(),
        "pose_energy_spearman": energy_loss.detach(),
        "pose_energy_good_bad_auc": energy_loss.detach(),
        "pose_energy_pred_cost_m": energy_loss.detach(),
        "pose_energy_oracle_cost_m": energy_loss.detach(),
        "pose_energy_oracle_gap_m": energy_loss.detach(),
        "pose_energy_factorized_pred_cost_m": energy_loss.detach(),
        "pose_energy_factorized_oracle_gap_m": energy_loss.detach(),
        "pose_energy_factorized_trans_err_m": energy_loss.detach(),
        "pose_energy_factorized_rot_deg": energy_loss.detach(),
        "pose_energy_selected_correction_cos": energy_loss.detach(),
        "pose_energy_oracle_correction_cos": energy_loss.detach(),
        "pose_energy_factorized_correction_cos": energy_loss.detach(),
        "pose_energy_factorized_translation_top1_acc": energy_loss.detach(),
        "pose_energy_factorized_rotation_top1_acc": energy_loss.detach(),
        "pose_energy_residual_pred_cost_m": energy_loss.detach(),
        "pose_energy_residual_cost_gain_m": energy_loss.detach(),
        "pose_energy_residual_trans_err_m": energy_loss.detach(),
        "pose_energy_residual_rot_deg": energy_loss.detach(),
    }
    energy_pred_idx = None
    if energy_net is not None and bool(args.pose_energy_enabled):
        score_source = str(getattr(args, "pose_energy_score_source", "local_corr") or "local_corr").lower()
        use_score_prior = float(getattr(args, "pose_energy_base_score_prior_weight", 0.0) or 0.0) > 0.0
        feature_pack = fine_candidate_selector_features(
            query_loc,
            cand_render_loc,
            candidate_pose,
            init_pose=init_pose,
            coarse_logits=cand_scores.detach() if use_score_prior else None,
            query_rgb=batch.get("rgb"),
            candidate_rgb=cand_rgb,
            depth=cand_depth,
            mask=cand_mask,
            candidate_valid_mask=cand_valid,
            mode="local",
            radius=int(args.local_corr_radius),
            preprocess=str(args.pose_energy_score_preprocess),
            highpass_kernel=int(args.pose_energy_score_highpass_kernel),
            score_map_mode=str(args.pose_energy_score_map_mode),
            return_score_maps=True,
            use_coarse_logits=use_score_prior,
            use_candidate_delta=bool(args.pose_energy_use_candidate_delta),
            use_delta_vector=bool(args.pose_energy_use_delta_vector),
            use_center_delta_vector=bool(args.pose_energy_use_center_delta_vector),
            use_depth=True,
            use_mask=True,
            use_rgb=bool(args.pose_energy_use_rgb),
            query_uncertainty=query_uncertainty,
            candidate_uncertainty=cand_uncertainty,
            use_uncertainty=energy_use_uncertainty,
        )
        if score_source == "pair_matcher_heatmap":
            if pair_matcher is None:
                raise ValueError("pose_energy_score_source=pair_matcher_heatmap requires --pair-matcher-enabled")
            pair_score_maps, pair_valid_map = pair_matcher_local_candidate_score_maps(
                pair_matcher,
                query_loc,
                cand_render_loc,
                mask=cand_mask,
                radius=int(args.pair_matcher_radius),
                stride=int(args.pair_matcher_score_stride),
                temperature=float(args.pair_matcher_temperature),
                chunk_points=int(args.pair_matcher_score_chunk_points),
                candidate_chunk_size=int(args.pair_matcher_score_candidate_chunk_size),
                offset_chunk_size=int(args.pair_matcher_score_offset_chunk_size),
                candidate_score_mode=str(args.pair_matcher_candidate_score_mode),
            )
            feature_pack["score_maps"] = pair_score_maps
            feature_pack["valid"] = feature_pack["valid"] & pair_valid_map.flatten(2).any(dim=2)
        elif score_source != "local_corr":
            raise ValueError(f"Unknown pose_energy_score_source: {args.pose_energy_score_source}")
        energy_out = energy_net(
            feature_pack["score_maps"],
            feature_pack["features"],
            valid_mask=feature_pack["valid"],
        )
        base_prior_weight = float(getattr(args, "pose_energy_base_score_prior_weight", 0.0) or 0.0)
        if base_prior_weight != 0.0:
            energy_out = dict(energy_out)
            combined_logits = pose_energy_logits_with_base_prior(
                energy_out["energy_logits"],
                cand_scores.detach(),
                feature_pack["valid"],
                weight=base_prior_weight,
                mode=str(getattr(args, "pose_energy_base_score_prior_mode", "zscore") or "zscore"),
            )
            energy_out["energy_logits"] = combined_logits
            if "joint_energy_logits" in energy_out:
                energy_out["joint_energy_logits"] = combined_logits
        energy_pack = pose_energy_losses(
            energy_out,
            candidate_pose,
            pose_gt.float(),
            valid_mask=feature_pack["valid"],
            target_temperature_m=float(args.pose_energy_target_temperature_m),
            rot_cost_weight=float(args.rot_cost_weight),
            hard_ce_weight=float(args.pose_energy_hard_ce_weight),
            component_hard_ce_weight=float(args.pose_energy_component_hard_ce_weight),
            residual_weight=float(args.pose_energy_residual_weight),
            residual_target_mode=str(args.pose_energy_residual_target_mode),
            residual_soft_topk_temperature_m=float(args.pose_energy_residual_soft_topk_temperature_m),
            improve_weight=float(args.pose_energy_improve_weight),
            improve_margin_m=float(args.pose_energy_improve_margin_m),
            update_scale=float(args.pose_energy_update_scale),
            residual_trans_scale_m=float(args.pose_energy_residual_trans_scale_m),
            residual_rot_scale_rad=math.radians(float(args.pose_energy_residual_rot_scale_deg)),
            anti_identity_weight=float(args.pose_energy_anti_identity_weight),
            anti_identity_margin=float(args.pose_energy_anti_identity_margin),
            anti_identity_min_gap_m=float(args.pose_energy_anti_identity_min_gap_m),
            identity_index=int(args.pose_energy_identity_index),
            pairwise_rank_weight=float(args.pose_energy_pairwise_rank_weight),
            pairwise_rank_min_gap_m=float(args.pose_energy_pairwise_rank_min_gap_m),
            pairwise_rank_logit_margin=float(args.pose_energy_pairwise_rank_logit_margin),
            translation_energy_weight=float(args.pose_energy_translation_weight),
            rotation_energy_weight=float(args.pose_energy_rotation_weight),
            joint_energy_weight=float(args.pose_energy_joint_weight),
            confidence_weight=float(args.pose_energy_confidence_weight),
            confidence_temperature_m=args.pose_energy_confidence_temperature_m,
        )
        energy_loss = energy_pack["loss"]
        direction_loss = {
            "loss": energy_loss.new_zeros(()),
            "active": energy_loss.new_zeros(()),
            "pred_cos": energy_loss.new_zeros(()),
            "target_cos": energy_loss.new_zeros(()),
        }
        correction_cosine_loss = {
            "loss": energy_loss.new_zeros(()),
            "active": energy_loss.new_zeros(()),
            "pred_cos": energy_loss.new_zeros(()),
            "target_cos": energy_loss.new_zeros(()),
        }
        if float(args.pose_energy_direction_weight) > 0.0:
            direction_loss = pose_energy_direction_pairwise_loss(
                energy_out["energy_logits"],
                candidate_pose,
                init_pose,
                pose_gt.float(),
                energy_pack["target_index"],
                valid_mask=feature_pack["valid"],
                min_cos_gap=float(args.pose_energy_direction_min_cos_gap),
                logit_margin=float(args.pose_energy_direction_logit_margin),
            )
            energy_loss = energy_loss + float(args.pose_energy_direction_weight) * direction_loss["loss"]
        if float(args.pose_energy_correction_cosine_weight) > 0.0:
            correction_cosine_loss = pose_energy_correction_cosine_soft_label_loss(
                energy_out["energy_logits"],
                correction_cos,
                valid_mask=feature_pack["valid"],
                target_temperature=float(args.pose_energy_correction_cosine_temperature),
                min_cos=float(args.pose_energy_correction_cosine_min_cos),
            )
            energy_loss = (
                energy_loss
                + float(args.pose_energy_correction_cosine_weight) * correction_cosine_loss["loss"]
            )
        energy_selection_scores = pose_energy_selection_scores(
            energy_out,
            confidence_weight=float(args.pose_energy_selection_confidence_weight),
            residual_norm_weight=float(args.pose_energy_selection_residual_norm_weight),
            residual_trans_scale_m=float(args.pose_energy_residual_trans_scale_m),
            residual_rot_scale_rad=math.radians(float(args.pose_energy_residual_rot_scale_deg)),
        ).masked_fill(~feature_pack["valid"], -1.0e6)
        energy_pred_idx = energy_selection_scores.argmax(dim=1)
        energy_batch_idx = torch.arange(pose_gt.shape[0], device=device)
        selected_pose = candidate_pose[energy_batch_idx, energy_pred_idx]
        selected_delta = energy_out["residual_delta"][energy_batch_idx, energy_pred_idx].float()
        residual_pose = apply_pose_energy_residual_update(
            selected_pose.float(),
            selected_delta,
            update_scale=float(args.pose_energy_update_scale),
        )
        residual_cost, _residual_targets, residual_trans, residual_rot = pose_costs_and_residual_targets(
            residual_pose[:, None],
            pose_gt.float(),
            valid_mask=torch.ones((pose_gt.shape[0], 1), device=device, dtype=torch.bool),
            rot_cost_weight=float(args.rot_cost_weight),
        )
        residual_cost = residual_cost[:, 0]
        residual_trans = residual_trans[:, 0]
        residual_rot = residual_rot[:, 0]
        monotonicity = {
            "loss": energy_loss.new_zeros(()),
            "active": energy_loss.new_zeros(()),
            "score_gain": energy_loss.new_zeros(()),
        }
        if float(args.pose_energy_monotonicity_weight) > 0.0:
            with torch.no_grad():
                residual_batch = map_renderer.attach_pose_candidate_renders(
                    dict(batch),
                    residual_pose[:, None].detach(),
                    prefix="nvs_residual",
                    require_grad=False,
                    feature="all",
                    include_aux=True,
                )
            residual_render_base = residual_batch["nvs_residual_fine"].float().detach()
            residual_depth = residual_batch.get("nvs_residual_depth")
            residual_mask = residual_batch.get("nvs_residual_mask")
            if energy_use_uncertainty:
                residual_render_loc, residual_uncertainty = project_render_bank_with_uncertainty(
                    adapter,
                    residual_render_base,
                    render_chunk_size=render_chunk_size,
                )
            else:
                residual_uncertainty = None
                residual_render_loc = apply_pose_feature_adapter(
                    adapter,
                    query_base,
                    residual_render_base,
                    render_chunk_size=render_chunk_size,
                )[1]
            residual_valid = torch.ones((pose_gt.shape[0], 1), device=device, dtype=torch.bool)
            residual_feature_pack = fine_candidate_selector_features(
                query_loc,
                residual_render_loc,
                residual_pose[:, None].detach(),
                init_pose=init_pose,
                query_rgb=batch.get("rgb"),
                candidate_rgb=residual_batch.get("nvs_residual_rgb"),
                depth=residual_depth,
                mask=residual_mask,
                candidate_valid_mask=residual_valid,
                mode="local",
                radius=int(args.local_corr_radius),
                preprocess=str(args.pose_energy_score_preprocess),
                highpass_kernel=int(args.pose_energy_score_highpass_kernel),
                score_map_mode=str(args.pose_energy_score_map_mode),
                return_score_maps=True,
                use_coarse_logits=False,
                use_candidate_delta=bool(args.pose_energy_use_candidate_delta),
                use_delta_vector=bool(args.pose_energy_use_delta_vector),
                use_center_delta_vector=bool(args.pose_energy_use_center_delta_vector),
                use_depth=True,
                use_mask=True,
                use_rgb=bool(args.pose_energy_use_rgb),
                query_uncertainty=query_uncertainty,
                candidate_uncertainty=residual_uncertainty,
                use_uncertainty=energy_use_uncertainty,
            )
            residual_energy_out = energy_net(
                residual_feature_pack["score_maps"],
                residual_feature_pack["features"],
                valid_mask=residual_feature_pack["valid"],
            )
            before_score = energy_out["energy_logits"][energy_batch_idx, energy_pred_idx]
            after_score = residual_energy_out["energy_logits"][:, 0]
            monotonicity = pose_energy_score_monotonicity_loss(
                before_score,
                after_score,
                before_cost=energy_pack["pose_cost"].gather(1, energy_pred_idx[:, None]).squeeze(1).detach(),
                after_cost=residual_cost.detach(),
                margin=float(args.pose_energy_monotonicity_margin),
                improved_only=bool(args.pose_energy_monotonicity_improved_only),
            )
            energy_loss = energy_loss + float(args.pose_energy_monotonicity_weight) * monotonicity["loss"]
        energy_spearman = _spearman_rows(
            energy_selection_scores,
            energy_pack["pose_cost"],
            feature_pack["valid"],
        )
        energy_auc = _good_bad_auc_rows(
            energy_selection_scores,
            energy_pack["pose_cost"],
            feature_pack["valid"],
            good_m=float(args.auc_good_m),
            bad_m=float(args.auc_bad_m),
        )
        energy_pred_cost = energy_pack["pose_cost"].gather(1, energy_pred_idx[:, None]).squeeze(1)
        energy_selected_correction_cos = correction_cos.gather(1, energy_pred_idx[:, None]).squeeze(1)
        energy_oracle_correction_cos = correction_cos.gather(1, energy_pack["target_index"][:, None]).squeeze(1)
        factorized_metrics = pose_energy_factorized_selection_metrics(
            energy_out,
            candidate_pose,
            pose_gt.float(),
            valid_mask=feature_pack["valid"],
            rot_cost_weight=float(args.rot_cost_weight),
        )
        factorized_pose, _factorized_trans_idx, _factorized_rot_idx = pose_energy_factorized_selection(
            energy_out,
            candidate_pose,
            valid_mask=feature_pack["valid"],
        )
        factorized_correction_cos = candidate_correction_cosines(
            factorized_pose[:, None],
            init_pose,
            pose_gt.float(),
        )[:, 0]
        pose_energy_metrics = {
            "pose_energy_loss": energy_loss.detach(),
            "pose_energy_energy_loss": energy_pack["energy_loss"].detach(),
            "pose_energy_residual_loss": energy_pack["residual_loss"].detach(),
            "pose_energy_translation_loss": energy_pack["translation_energy_loss"].detach(),
            "pose_energy_rotation_loss": energy_pack["rotation_energy_loss"].detach(),
            "pose_energy_joint_loss": energy_pack["joint_energy_loss"].detach(),
            "pose_energy_component_hard_ce_loss": energy_pack["component_hard_ce_loss"].detach(),
            "pose_energy_confidence_loss": energy_pack["confidence_loss"].detach(),
            "pose_energy_improve_loss": energy_pack["improve_loss"].detach(),
            "pose_energy_monotonicity_loss": monotonicity["loss"].detach(),
            "pose_energy_monotonicity_active": monotonicity["active"].detach(),
            "pose_energy_monotonicity_score_gain": monotonicity["score_gain"].detach(),
            "pose_energy_anti_identity_loss": energy_pack["anti_identity_loss"].detach(),
            "pose_energy_anti_identity_active": energy_pack["anti_identity_active"].detach(),
            "pose_energy_pairwise_rank_loss": energy_pack["pairwise_rank_loss"].detach(),
            "pose_energy_pairwise_rank_active": energy_pack["pairwise_rank_active"].detach(),
            "pose_energy_direction_loss": direction_loss["loss"].detach(),
            "pose_energy_direction_active": direction_loss["active"].detach(),
            "pose_energy_direction_pred_cos": direction_loss["pred_cos"].detach(),
            "pose_energy_direction_target_cos": direction_loss["target_cos"].detach(),
            "pose_energy_correction_cosine_loss": correction_cosine_loss["loss"].detach(),
            "pose_energy_correction_cosine_active": correction_cosine_loss["active"].detach(),
            "pose_energy_correction_cosine_pred_cos": correction_cosine_loss["pred_cos"].detach(),
            "pose_energy_correction_cosine_target_cos": correction_cosine_loss["target_cos"].detach(),
            "pose_energy_top1_acc": (energy_pred_idx == energy_pack["target_index"]).float().mean().detach(),
            "pose_energy_spearman": energy_spearman.detach(),
            "pose_energy_good_bad_auc": energy_auc.detach(),
            "pose_energy_pred_cost_m": energy_pred_cost.detach().mean(),
            "pose_energy_oracle_cost_m": energy_pack["oracle_cost"].detach().mean(),
            "pose_energy_oracle_gap_m": (energy_pred_cost.detach() - energy_pack["oracle_cost"].detach()).mean(),
            "pose_energy_factorized_pred_cost_m": factorized_metrics["factorized_pred_cost_m"].detach().mean(),
            "pose_energy_factorized_oracle_gap_m": factorized_metrics["factorized_oracle_gap_m"].detach().mean(),
            "pose_energy_factorized_trans_err_m": factorized_metrics["factorized_trans_err_m"].detach().mean(),
            "pose_energy_factorized_rot_deg": (
                factorized_metrics["factorized_rot_err_rad"].detach().mean() * (180.0 / math.pi)
            ),
            "pose_energy_selected_correction_cos": energy_selected_correction_cos.detach().mean(),
            "pose_energy_oracle_correction_cos": energy_oracle_correction_cos.detach().mean(),
            "pose_energy_factorized_correction_cos": factorized_correction_cos.detach().mean(),
            "pose_energy_factorized_translation_top1_acc": factorized_metrics[
                "factorized_translation_top1_acc"
            ].detach(),
            "pose_energy_factorized_rotation_top1_acc": factorized_metrics[
                "factorized_rotation_top1_acc"
            ].detach(),
            "pose_energy_residual_pred_cost_m": residual_cost.detach().mean(),
            "pose_energy_residual_cost_gain_m": (energy_pred_cost.detach() - residual_cost.detach()).mean(),
            "pose_energy_residual_trans_err_m": residual_trans.detach().mean(),
            "pose_energy_residual_rot_deg": (residual_rot.detach().mean() * (180.0 / math.pi)),
        }
    drift = feature_drift_loss(query_loc, query_base.detach(), gt_render_loc, gt_render_base, gt_mask)
    var_loss = 0.5 * (
        variance_floor_loss(query_loc, float(args.variance_min_std))
        + variance_floor_loss(gt_render_loc, float(args.variance_min_std))
    )
    loss = (
        float(args.align_weight) * align_loss
        + float(args.warp_align_weight) * warp_loss
        + float(args.local_flow_nce_weight) * flow_loss
        + teacher_corr_loss
        + pair_match_loss
        + float(args.rank_ce_weight) * rank["rank_loss"]
        + float(args.score_correction_cosine_weight) * score_correction_cosine["loss"]
        + float(args.score_pose_improvement_weight) * score_pose_improvement["loss"]
        + candidate_teacher_quality_weight * candidate_teacher_quality["loss"]
        + float(args.score_anti_identity_weight) * score_anti_identity["loss"]
        + float(args.candidate_observability_weight) * candidate_observability["loss"]
        + float(args.observability_contrast_weight) * observability["loss"]
        + float(args.pose_energy_weight) * energy_loss
        + float(args.drift_weight) * drift
        + float(args.variance_weight) * var_loss
    )
    pred_idx = energy_pred_idx if bool(args.pose_energy_select_for_metric) and energy_pred_idx is not None else rank["pred_index"].long()
    batch_idx = torch.arange(pose_gt.shape[0], device=device)
    rank_pred_idx = rank["pred_index"].long()
    rank_target_idx = rank["target_index"].long()
    selected_metrics = selected_candidate_pose_metrics(
        pose_cost,
        trans_err,
        rot_err,
        rank["oracle_cost"].detach(),
        pred_idx.long(),
    )
    pred_trans = selected_metrics["selected_trans"]
    pred_rot = selected_metrics["selected_rot"]
    oracle_trans = trans_err[batch_idx, rank_target_idx]
    oracle_rot = rot_err[batch_idx, rank_target_idx]
    identity_idx = int(args.score_anti_identity_index)
    if bool(identity_mask.any()):
        selected_identity_frac = identity_mask.gather(1, pred_idx.long()[:, None]).squeeze(1).float().mean()
        oracle_identity_frac = identity_mask.gather(1, rank_target_idx[:, None]).squeeze(1).float().mean()
    else:
        selected_identity_frac = query_loc.new_zeros(())
        oracle_identity_frac = query_loc.new_zeros(())
    selection_bias = candidate_selection_bias_metrics(
        cand_scores.detach(),
        pose_cost.detach(),
        cand_valid,
        pred_idx.long(),
        identity_mask=identity_mask,
    )
    selected_pose_for_delta = candidate_pose[batch_idx, pred_idx.long()]
    oracle_pose_for_delta = candidate_pose[batch_idx, rank_target_idx]
    _selected_delta_loss, selected_delta_rot, selected_delta_trans = pose_error_tensors(
        selected_pose_for_delta.float(),
        init_pose.float(),
    )
    _oracle_delta_loss, oracle_delta_rot, oracle_delta_trans = pose_error_tensors(
        oracle_pose_for_delta.float(),
        init_pose.float(),
    )
    spearman = _spearman_rows(cand_scores, pose_cost, cand_valid)
    auc = _good_bad_auc_rows(
        cand_scores,
        pose_cost,
        cand_valid,
        good_m=float(args.auc_good_m),
        bad_m=float(args.auc_bad_m),
    )
    pose_obs_metrics = pose_observability_diagnostic_metrics(
        query_loc,
        gt_render_loc,
        gt_depth,
        target_intrinsics,
        mask=gt_mask,
    ) if bool(args.pose_observability_diagnostic_enabled) else {
        "query_pose_obs_logdet": query_loc.new_zeros(()),
        "query_pose_obs_trace": query_loc.new_zeros(()),
        "query_pose_obs_trace_inv": query_loc.new_zeros(()),
        "query_pose_obs_condition": query_loc.new_zeros(()),
        "query_pose_obs_valid_frac": query_loc.new_zeros(()),
        "map_pose_obs_logdet": query_loc.new_zeros(()),
        "map_pose_obs_trace": query_loc.new_zeros(()),
        "map_pose_obs_trace_inv": query_loc.new_zeros(()),
        "map_pose_obs_condition": query_loc.new_zeros(()),
        "map_pose_obs_valid_frac": query_loc.new_zeros(()),
        "query_pose_obs_unit_logdet": query_loc.new_zeros(()),
        "query_pose_obs_unit_trace": query_loc.new_zeros(()),
        "query_pose_obs_unit_trace_inv": query_loc.new_zeros(()),
        "query_pose_obs_unit_condition": query_loc.new_zeros(()),
        "map_pose_obs_unit_logdet": query_loc.new_zeros(()),
        "map_pose_obs_unit_trace": query_loc.new_zeros(()),
        "map_pose_obs_unit_trace_inv": query_loc.new_zeros(()),
        "map_pose_obs_unit_condition": query_loc.new_zeros(()),
        "pose_obs_diagnostic_missing": query_loc.new_ones(()),
    }
    metrics = {
        "loss": loss.detach(),
        "align_loss": align_loss.detach(),
        "align_cos": align_cos.detach(),
        "warp_align_loss": warp_metrics["warp_align_loss"].detach(),
        "warp_align_cos": warp_metrics["warp_align_cos"].detach(),
        "warp_valid_frac": warp_metrics["warp_valid_frac"].detach(),
        "warp_in_bounds_frac": warp_metrics["warp_in_bounds_frac"].detach(),
        "local_flow_nce_loss": flow_metrics["local_flow_nce_loss"].detach(),
        "local_flow_valid_frac": flow_metrics["local_flow_valid_frac"].detach(),
        "local_flow_target_offset_px": flow_metrics["local_flow_target_offset_px"].detach(),
        "local_flow_candidate_selected_frac": flow_metrics["local_flow_candidate_selected_frac"].detach(),
        "local_flow_best_active": flow_metrics["local_flow_best_active"].detach(),
        "local_flow_hard_negative_active": flow_metrics["local_flow_hard_negative_active"].detach(),
        **{key: value.detach() for key, value in teacher_corr_metrics.items()},
        **{key: value.detach() for key, value in pair_match_metrics.items()},
        "local_zero_score": local_stats["local_zero_score"].detach(),
        "local_peak_score": local_stats["local_peak_score"].detach(),
        "local_peak_gap": local_stats["local_peak_gap"].detach(),
        "local_peak_offset_px": local_stats["local_peak_offset_px"].detach(),
        "local_expected_offset_px": local_stats["local_expected_offset_px"].detach(),
        "rank_loss": rank["rank_loss"].detach(),
        "rank_ce_loss": rank["rank_ce_loss"].detach(),
        "rank_pairwise_loss": rank["rank_pairwise_loss"].detach(),
        "rank_pairwise_active": rank["rank_pairwise_active"].detach(),
        "score_correction_cosine_loss": score_correction_cosine["loss"].detach(),
        "score_correction_cosine_active": score_correction_cosine["active"].detach(),
        "score_correction_cosine_pred_cos": score_correction_cosine["pred_cos"].detach(),
        "score_correction_cosine_target_cos": score_correction_cosine["target_cos"].detach(),
        "score_pose_improvement_loss": score_pose_improvement["loss"].detach(),
        "score_pose_improvement_active": score_pose_improvement["active"].detach(),
        "score_pose_improvement_pred_m": score_pose_improvement["pred_improvement_m"].detach(),
        "score_pose_improvement_target_m": score_pose_improvement["target_improvement_m"].detach(),
        "score_pose_improvement_pred_cost_m": score_pose_improvement["pred_cost_m"].detach(),
        "score_pose_improvement_target_cost_m": score_pose_improvement["target_cost_m"].detach(),
        "candidate_teacher_quality_loss": candidate_teacher_quality["loss"].detach(),
        "candidate_teacher_quality_weight": query_loc.new_tensor(candidate_teacher_quality_weight).detach(),
        "candidate_teacher_quality_ce_loss": candidate_teacher_quality["ce_loss"].detach(),
        "candidate_teacher_quality_pairwise_loss": candidate_teacher_quality["pairwise_loss"].detach(),
        "candidate_teacher_quality_active": candidate_teacher_quality["active"].detach(),
        "candidate_teacher_quality_top1_acc": candidate_teacher_quality["top1_acc"].detach(),
        "candidate_teacher_quality_pred": candidate_teacher_quality["pred_quality"].detach(),
        "candidate_teacher_quality_target": candidate_teacher_quality["target_quality"].detach(),
        "score_anti_identity_loss": score_anti_identity["loss"].detach(),
        "score_anti_identity_active": score_anti_identity["active"].detach(),
        "score_anti_identity_better_margin": score_anti_identity["better_margin"].detach(),
        "candidate_observability_loss": candidate_observability["loss"].detach(),
        "candidate_observability_active": candidate_observability["active"].detach(),
        "candidate_observability_gap": candidate_observability["gap"].detach(),
        "candidate_observability_best_score": candidate_observability["best_score"].detach(),
        "candidate_observability_hard_negative_score": candidate_observability["hard_negative_score"].detach(),
        "selected_identity_frac": selected_identity_frac.detach(),
        "oracle_identity_frac": oracle_identity_frac.detach(),
        "candidate_identity_frac": identity_mask.float().mean().detach(),
        "selected_delta_trans_m": selected_delta_trans.detach().mean(),
        "selected_delta_rot_deg": selected_delta_rot.detach().mean(),
        "oracle_delta_trans_m": oracle_delta_trans.detach().mean(),
        "oracle_delta_rot_deg": oracle_delta_rot.detach().mean(),
        "score_best_minus_score_identity": selection_bias["score_best_minus_score_identity"].detach(),
        "score_best_minus_score_selected": selection_bias["score_best_minus_score_selected"].detach(),
        "observability_contrast_loss": observability["loss"].detach(),
        "observability_active": observability["active"].detach(),
        "observability_gt_score": observability["gt_score"].detach(),
        "observability_hard_negative_score": observability["hard_negative_score"].detach(),
        "observability_gap": observability["gap"].detach(),
        "drift_loss": drift.detach(),
        "variance_loss": var_loss.detach(),
        "rank_top1_acc": rank["top1_acc"].detach(),
        "top1_acc": (pred_idx.long() == rank_target_idx).float().mean().detach(),
        "spearman": spearman.detach(),
        "good_bad_auc": auc.detach(),
        "rank_pred_cost_m": rank["pred_cost"].detach().mean(),
        "pred_cost_m": selected_metrics["selected_cost"].detach().mean(),
        "oracle_cost_m": rank["oracle_cost"].detach().mean(),
        "rank_oracle_gap_m": rank["oracle_gap"].detach().mean(),
        "oracle_gap_m": selected_metrics["selected_oracle_gap"].detach().mean(),
        "pred_trans_m": pred_trans.detach().mean(),
        "pred_rot_deg": (pred_rot.detach().mean() * (180.0 / math.pi)),
        **{key: value.detach() for key, value in pose_threshold_success_metrics(pred_trans, pred_rot, prefix="pred_").items()},
        **{
            key: value.detach()
            for key, value in pose_threshold_success_metrics(oracle_trans, oracle_rot, prefix="oracle_").items()
        },
        **{
            key: value.detach()
            for key, value in pose_threshold_success_metrics(init_trans[:, 0], init_rot[:, 0], prefix="init_").items()
        },
        "rank_selected_correction_cos": correction_cos[batch_idx, rank_pred_idx].detach().mean(),
        "selected_correction_cos": correction_cos[batch_idx, pred_idx.long()].detach().mean(),
        "oracle_correction_cos": correction_cos[batch_idx, rank_target_idx].detach().mean(),
        "candidate_score_mean": cand_scores.detach().mean(),
        "candidate_score_std": cand_scores.detach().std(),
        "score_feature_h": query_loc.new_tensor(float(score_feature_hw[0] if score_feature_hw is not None else 0)),
        "score_feature_w": query_loc.new_tensor(float(score_feature_hw[1] if score_feature_hw is not None else 0)),
        "synthetic_fraction": synthetic_mask.float().mean().detach(),
        **{key: value.detach() for key, value in pose_obs_metrics.items()},
    }
    metrics.update(pose_energy_metrics)
    return loss, metrics


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in rows[0].keys()}


@torch.no_grad()
def evaluate(
    model,
    adapter,
    energy_net,
    pair_matcher,
    loader,
    map_renderer,
    cfg: Dict,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    adapter.eval()
    if energy_net is not None:
        energy_net.eval()
    if pair_matcher is not None:
        pair_matcher.eval()
    eval_args = argparse.Namespace(**vars(args))
    if args.eval_synthetic_ratio is not None:
        eval_args.synthetic_ratio = float(args.eval_synthetic_ratio)
    rows = []
    for batch_idx, batch in enumerate(loader):
        _loss, metrics = forward_batch(
            model,
            adapter,
            energy_net,
            pair_matcher,
            map_renderer,
            batch,
            cfg,
            eval_args,
            train=False,
        )
        rows.append({key: float(value.detach().cpu()) for key, value in metrics.items()})
        if args.eval_max_samples is not None and (batch_idx + 1) * int(args.batch_size) >= int(args.eval_max_samples):
            break
    return mean_metrics(rows)


def _prefixed_state_dict(model: torch.nn.Module, prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    allowed = tuple(prefix for prefix in prefixes if prefix)
    if not allowed:
        return {}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if any(key.startswith(prefix) for prefix in allowed)
    }


def _gradient_stats(parameters: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    total = 0
    nonfinite = 0
    max_abs = 0.0
    grad_norm_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += int(grad.numel())
        finite = torch.isfinite(grad)
        nonfinite += int((~finite).sum().item())
        finite_grad = grad[finite]
        if finite_grad.numel() == 0:
            continue
        max_abs = max(max_abs, float(finite_grad.abs().max().cpu()))
        grad_norm_sq += float((finite_grad.float() * finite_grad.float()).sum().cpu())
    return {
        "grad_total_elems": float(total),
        "grad_nonfinite_elems": float(nonfinite),
        "grad_max_abs": float(max_abs),
        "grad_norm": float(math.sqrt(max(grad_norm_sq, 0.0))),
    }


def collect_trainable_parameters(
    adapter: PoseFeatureDomainAdapter,
    energy_net: PoseEnergyNet | None,
    pair_matcher: PairConditionedLocalMatcher | None,
    model: torch.nn.Module,
    *,
    train_adapter: bool = True,
    adapter_domains: str = "all",
    train_energy_net: bool = True,
    train_pair_matcher: bool = True,
) -> List[torch.nn.Parameter]:
    domains = str(adapter_domains or "all").lower()
    if domains not in {"all", "query", "render", "none"}:
        raise ValueError("adapter_domains must be all, query, render, or none")
    for _name, param in adapter.named_parameters():
        param.requires_grad_(False)
    if bool(train_adapter) and domains != "none":
        for name, param in adapter.named_parameters():
            if domains == "all":
                param.requires_grad_(True)
            elif domains == "query" and name.startswith("query_"):
                param.requires_grad_(True)
            elif domains == "render" and name.startswith("render_"):
                param.requires_grad_(True)
    params: List[torch.nn.Parameter] = []
    if bool(train_adapter):
        params.extend(param for param in adapter.parameters() if param.requires_grad)
    if energy_net is not None:
        for param in energy_net.parameters():
            param.requires_grad_(bool(train_energy_net))
        if bool(train_energy_net):
            params.extend(energy_net.parameters())
    if pair_matcher is not None:
        for param in pair_matcher.parameters():
            param.requires_grad_(bool(train_pair_matcher))
        if bool(train_pair_matcher):
            params.extend(pair_matcher.parameters())
    params.extend(param for param in model.parameters() if param.requires_grad)
    return params


def save_checkpoint(
    path: Path,
    adapter,
    model,
    optimizer,
    step: int,
    metrics: Dict[str, float],
    cfg: Dict,
    args,
    *,
    energy_net=None,
    pair_matcher=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    query_prefixes = []
    if bool(getattr(args, "train_projector", False)):
        query_prefixes.append("local_corr_projector.")
    query_prefixes.extend(parse_str_csv(getattr(args, "train_model_prefixes", "")))
    query_state = _prefixed_state_dict(model, query_prefixes)
    torch.save(
        {
            "step": int(step),
            "pose_feature_adapter_state_dict": adapter.state_dict(),
            "query_projector_state_dict": _prefixed_state_dict(model, ["local_corr_projector."]),
            "query_model_state_dict": query_state,
            "query_model_prefixes": query_prefixes,
            "pose_energy_net_state_dict": energy_net.state_dict() if energy_net is not None else None,
            "pair_matcher_state_dict": pair_matcher.state_dict() if pair_matcher is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg,
            "args": vars(args),
        },
        path,
    )


def load_adapter_checkpoint(
    path: str,
    adapter,
    model=None,
    optimizer=None,
    energy_net=None,
    pair_matcher=None,
    *,
    strict_adapter: bool = True,
) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    adapter_state = checkpoint.get("pose_feature_adapter_state_dict", checkpoint)
    adapter_load = adapter.load_state_dict(adapter_state, strict=bool(strict_adapter))
    checkpoint["missing_adapter_keys"] = list(adapter_load.missing_keys)
    checkpoint["unexpected_adapter_keys"] = list(adapter_load.unexpected_keys)
    query_state = checkpoint.get("query_model_state_dict") or checkpoint.get("query_projector_state_dict") or {}
    if model is not None and query_state:
        model.load_state_dict(query_state, strict=False)
    energy_state = checkpoint.get("pose_energy_net_state_dict") or checkpoint.get("pose_energy_state_dict")
    if energy_net is not None and energy_state:
        energy_net.load_state_dict(energy_state, strict=True)
    pair_state = checkpoint.get("pair_matcher_state_dict")
    if pair_matcher is not None and pair_state:
        pair_matcher.load_state_dict(pair_state, strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _metric_is_better(value: float, best: float, mode: str) -> bool:
    if str(mode) == "max":
        return value > best
    return value < best


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    args = apply_config_defaults(args, cfg)
    (out_dir / "resolved_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    train_args = argparse.Namespace(**vars(args))
    train_args.split = args.train_split
    train_args.max_samples = args.max_samples
    model, train_loader, map_renderer = build_model_and_data(cfg, train_args, device)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.split = args.eval_split
    eval_args.max_samples = args.eval_max_samples
    _eval_model, eval_loader, _eval_renderer = build_model_and_data(cfg, eval_args, device)
    del _eval_model, _eval_renderer

    train_prefixes = []
    if bool(args.train_projector):
        train_prefixes.append("local_corr_projector.")
    train_prefixes.extend(parse_str_csv(args.train_model_prefixes))
    _set_trainable(model, train_prefixes)
    adapter = build_pose_feature_adapter(args, model, cfg, device)
    if adapter is None:
        raise ValueError("NVS pose feature training requires pose_feature_adapter_enabled=true")
    energy_net = None
    if bool(args.pose_energy_enabled):
        energy_net = PoseEnergyNet(
            vector_dim=nvs_pose_energy_vector_dim(args),
            score_map_channels=3,
            map_channels=int(args.pose_energy_map_channels),
            grid_size=int(args.pose_energy_grid_size),
            hidden_dim=int(args.pose_energy_hidden_dim),
            context_layers=int(args.pose_energy_context_layers),
            context_heads=int(args.pose_energy_context_heads),
            zero_init_heads=bool(args.pose_energy_zero_init_heads),
            zero_init_residual_head=bool(args.pose_energy_zero_init_residual_head),
            factorized_heads=bool(args.pose_energy_factorized_heads),
        ).to(device)
    pair_matcher = None
    if bool(args.pair_matcher_enabled):
        pair_matcher = PairConditionedLocalMatcher(
            channels=int(adapter.channels),
            hidden_dim=int(args.pair_matcher_hidden_dim),
            offset_radius=int(args.pair_matcher_radius),
            zero_init_residual=bool(args.pair_matcher_zero_init_residual),
            base_dot_weight=float(args.pair_matcher_base_dot_weight),
        ).to(device)
    trainable = collect_trainable_parameters(
        adapter,
        energy_net,
        pair_matcher,
        model,
        train_adapter=bool(args.train_pose_feature_adapter),
        adapter_domains=str(args.train_pose_feature_adapter_domains),
        train_energy_net=bool(args.train_pose_energy_net),
        train_pair_matcher=bool(args.train_pair_matcher),
    )
    if not trainable:
        raise ValueError("No trainable parameters selected for NVS pose feature adapter training")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    if args.resume_adapter:
        loaded = load_adapter_checkpoint(
            args.resume_adapter,
            adapter,
            model=model,
            optimizer=None,
            energy_net=energy_net,
            pair_matcher=pair_matcher,
            strict_adapter=bool(args.resume_adapter_strict),
        )
        print(f"loaded adapter checkpoint: {args.resume_adapter} step={loaded.get('step', 'unknown')}", flush=True)
        if loaded.get("missing_adapter_keys") or loaded.get("unexpected_adapter_keys"):
            print(
                "adapter checkpoint loaded non-strict: "
                f"missing={loaded.get('missing_adapter_keys', [])} "
                f"unexpected={loaded.get('unexpected_adapter_keys', [])}",
                flush=True,
            )

    if args.eval_only:
        metrics = evaluate(model, adapter, energy_net, pair_matcher, eval_loader, map_renderer, cfg, args)
        summary = {"checkpoint": args.resume_adapter, "metrics": metrics}
        (out_dir / "eval_only_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return

    log_path = out_dir / "train_log.jsonl"
    best_metric = -float("inf") if str(args.best_metric_mode) == "max" else float("inf")
    step = 0
    consecutive_skipped_steps = 0
    train_iter = iter(train_loader)
    while step < int(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        model.train(bool(train_prefixes))
        adapter.train(bool(args.train_pose_feature_adapter))
        if energy_net is not None:
            energy_net.train(bool(args.train_pose_energy_net))
        if pair_matcher is not None:
            pair_matcher.train(bool(args.train_pair_matcher))
        optimizer.zero_grad(set_to_none=True)
        args.current_step = step
        with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
            loss, metrics = forward_batch(model, adapter, energy_net, pair_matcher, map_renderer, batch, cfg, args, train=True)
        scale_before = float(scaler.get_scale()) if bool(args.amp and device.type == "cuda") else 1.0
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0.0:
            scaler.unscale_(optimizer)
        grad_stats = _gradient_stats(trainable)
        if float(grad_stats["grad_nonfinite_elems"]) > 0.0 and not bool(args.amp and device.type == "cuda"):
            raise RuntimeError(
                f"Non-finite gradients without AMP at step {step + 1}: "
                f"{int(grad_stats['grad_nonfinite_elems'])}/{int(grad_stats['grad_total_elems'])}"
            )
        if float(args.grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale()) if bool(args.amp and device.type == "cuda") else 1.0
        optimizer_step = 1.0
        if bool(args.amp and device.type == "cuda") and scale_after < scale_before:
            optimizer_step = 0.0
            consecutive_skipped_steps += 1
            print(
                f"warning: AMP skipped optimizer step {step + 1}; "
                f"nonfinite_grad={int(grad_stats['grad_nonfinite_elems'])}/"
                f"{int(grad_stats['grad_total_elems'])}, scale {scale_before:g}->{scale_after:g}",
                flush=True,
            )
            if consecutive_skipped_steps >= int(args.max_skipped_optimizer_steps):
                raise RuntimeError(
                    f"AMP skipped {consecutive_skipped_steps} optimizer steps in a row. "
                    "Disable AMP for this NVS pose-energy path with --no-amp."
                )
        else:
            consecutive_skipped_steps = 0
        step += 1

        row = {"step": step, "split": "train"}
        row.update({key: float(value.detach().cpu()) for key, value in metrics.items()})
        row.update(grad_stats)
        row["optimizer_step"] = optimizer_step
        row["amp_scale_before"] = scale_before
        row["amp_scale_after"] = scale_after
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 10 == 0:
            print(
                "step={step} loss={loss:.4f} align={align_loss:.4f}/{align_cos:.3f} "
                "warp={warp_align_loss:.4f}/{warp_align_cos:.3f}/{warp_valid_frac:.2f} "
                "flow={local_flow_nce_loss:.3f}/{local_flow_valid_frac:.2f} "
                "rank={rank_loss:.4f} pred={pred_cost_m:.3f} oracle={oracle_cost_m:.3f} "
                "pe={pose_energy_pred_cost_m:.3f}/{pose_energy_residual_pred_cost_m:.3f} "
                "gap={oracle_gap_m:.3f} sp={spearman:.3f} auc={good_bad_auc:.3f} syn={synthetic_fraction:.2f}".format(
                    **row
                ),
                flush=True,
            )
        if step % int(args.eval_every) == 0 or step == int(args.max_steps):
            args.current_step = step
            eval_metrics = evaluate(model, adapter, energy_net, pair_matcher, eval_loader, map_renderer, cfg, args)
            eval_row = {"step": step, "split": "eval"}
            eval_row.update(eval_metrics)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(eval_row) + "\n")
            print("eval step={step} ".format(step=step) + json.dumps(eval_metrics, sort_keys=True), flush=True)
            metric = float(eval_metrics.get(str(args.best_metric), float("inf")))
            if _metric_is_better(metric, best_metric, str(args.best_metric_mode)):
                best_metric = metric
                save_checkpoint(
                    out_dir / "checkpoints" / "best.pth",
                    adapter,
                    model,
                    optimizer,
                    step,
                    eval_metrics,
                    cfg,
                    args,
                    energy_net=energy_net,
                    pair_matcher=pair_matcher,
                )
        if step % int(args.save_every) == 0:
            save_checkpoint(
                out_dir / "checkpoints" / f"step_{step:06d}.pth",
                adapter,
                model,
                optimizer,
                step,
                row,
                cfg,
                args,
                energy_net=energy_net,
                pair_matcher=pair_matcher,
            )

    summary = {"best_metric": best_metric, "best_metric_name": args.best_metric, "steps": step, "out_dir": str(out_dir)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
