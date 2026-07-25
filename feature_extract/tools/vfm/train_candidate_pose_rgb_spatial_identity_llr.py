"""Train a target-free candidate/view identity LLR on real image evidence.

This is intentionally a separate stage from the local RGB spatial density.
It learns whether a fixed query point and a fixed candidate support observation
represent the same physical landmark, while a different module remains
responsible for sub-pixel alignment.  The encoder never receives pose,
projection offsets, residuals, labels, track IDs, ranks, or coarse scores.

Train-only targets are joined only after the candidate/view LLR has been
emitted.  In particular, the checkpoint-selection gate selects points from
the target-free layout before it reads registered identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatBatch,
    HardRepeatQueryTargets,
    TrainQueryGroup,
    _DistributedState,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _select_group_points,
    _source_table,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    training_gate_decision,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CandidatePoseRGBSpatialHardRepeatTargets,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CandidatePoseRGBSpatialIdentityLLR,
    CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    candidate_pose_rgb_spatial_identity_component_edge_usable,
    marginalize_candidate_pose_rgb_spatial_identity_llr,
    resolve_candidate_pose_rgb_spatial_identity_context_windows,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_support_appearance,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    TARGET_FREE_SELECTOR_POLICIES,
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_pose_rgb_spatial_identity_llr_checkpoint_v3"
REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_GATE_VERSION = (
    "combined_visual_controls_plus_conditional_source_visual_ablation_and_hard_pose_v5"
)
REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_FORMAT = (
    "candidate_pose_rgb_spatial_identity_llr_observation_pretrain_v2"
)
_STATIC_SELECTOR_POLICIES = tuple(
    policy for policy in TARGET_FREE_SELECTOR_POLICIES if "rgb" not in str(policy)
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-stage",
        choices=("identity_l0", "hard_pose_contrastive"),
        default="hard_pose_contrastive",
        help=(
            "identity_l0 gates candidate/null/view evidence before any pose-group objective; "
            "hard_pose_contrastive retains the later pose-group calibration protocol."
        ),
    )
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--initialization-only",
        action="store_true",
        help=(
            "Run the frozen P1 zero-shot transfer audit and exit before any P1 "
            "target can update the broad visual initializer."
        ),
    )
    parser.add_argument(
        "--texture-observation-pretrain-checkpoint",
        default="",
        help="Gate-approved broad real-RGB observation pretrain used only to initialize the FPN.",
    )
    parser.add_argument(
        "--identity-observation-pretrain-checkpoint",
        default="",
        help=(
            "Strictly gate-approved fold-matched broad identity checkpoint. When set, "
            "it initializes the entire candidate-set RGB/context LLR model."
        ),
    )
    parser.add_argument("--rgb-context-radius-px", type=float, default=48.0)
    parser.add_argument("--rgb-step-px", type=float, default=1.0)
    parser.add_argument(
        "--train-query-anchor-jitter-radius-px",
        type=float,
        default=0.0,
        help=(
            "Train-only target-free query-anchor perturbation radius. It moves only "
            "the query RGB/context crop inside its image while leaving the frozen "
            "candidate set, supports, priors, and targets unchanged."
        ),
    )
    parser.add_argument(
        "--allow-full-identity-l0-feature-finetune",
        action="store_true",
        help=(
            "Explicitly permit full encoder fine-tuning during identity_l0. This is "
            "reserved for the detector-anchor robustness ablation and requires a "
            "positive train-query anchor jitter radius."
        ),
    )
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument(
        "--candidate-prior-logit-weight",
        type=float,
        default=0.0,
        help=(
            "Fixed log candidate-prior weight added only after the visual LLR is emitted. "
            "Zero preserves the prior-free diagnostic objective."
        ),
    )
    parser.add_argument(
        "--posterior-cross-entropy-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for exact-track candidate-plus-null posterior cross entropy in the "
            "fixed-prior residual score space."
        ),
    )
    parser.add_argument(
        "--posterior-candidate-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Relative optimization weight for exact-candidate posterior rows. "
            "It does not alter the target-free runtime posterior."
        ),
    )
    parser.add_argument(
        "--posterior-null-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Relative optimization weight for explicit-null posterior rows. "
            "It does not alter the immutable runtime null prior."
        ),
    )
    parser.add_argument(
        "--rgb-identity-posterior-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Direct exact-candidate posterior loss for the RGB-only identity expert. "
            "It uses fixed support-view and null factors, never the fused scorer."
        ),
    )
    parser.add_argument(
        "--rgb-identity-permutation-loss-weight",
        type=float,
        default=0.25,
        help="Support-appearance derangement control for the RGB identity expert.",
    )
    parser.add_argument(
        "--context-coherence-hard-repeat-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Hard coherent-repeat margin for the RADIO/ALIKE context expert using "
            "only its own edge LLR."
        ),
    )
    parser.add_argument(
        "--context-coherence-permutation-loss-weight",
        type=float,
        default=0.25,
        help="Support-appearance derangement control for the context coherence expert.",
    )
    parser.add_argument(
        "--reset-edge-head-final-for-prior-residual",
        action="store_true",
        help=(
            "After loading the broad visual initializer, reset only its final LLR layer so "
            "P1 residual training starts from the frozen candidate prior."
        ),
    )
    parser.add_argument(
        "--reset-current-p1-scalar-head-finals",
        action="store_true",
        help=(
            "Reset only the RGB/context/view/null scalar-head final layers after "
            "loading the broad encoder. This lets a current-P1 pose-group loss "
            "calibrate a new likelihood ratio without inheriting a broad "
            "candidate-pool scalar calibration."
        ),
    )
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=21)
    parser.add_argument("--max-points-per-query", type=int, default=96)
    parser.add_argument(
        "--validation-selector-policy",
        choices=_STATIC_SELECTOR_POLICIES,
        default="support_coverage",
    )
    parser.add_argument("--validation-selector-point-budget", type=int, default=96)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--minimum-inner-eligible-query-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-inner-hard-repeat-eligible-query-fraction", type=float, default=0.9)
    parser.add_argument(
        "--max-hard-repeat-edges-per-query",
        type=int,
        default=64,
        help=(
            "Maximum train-only coherent-repeat edges joined per query; zero keeps "
            "every eligible edge for a strict full-pool audit."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=0.0,
        help=(
            "Edge-LLR head learning rate. Zero inherits --learning-rate; use a larger "
            "value when the deliberately neutral zero-initialized head needs to adapt."
        ),
    )
    parser.add_argument(
        "--projector-learning-rate",
        type=float,
        default=0.0,
        help="Context/RGB projector learning rate; zero inherits --learning-rate.",
    )
    parser.add_argument(
        "--context-learning-rate",
        type=float,
        default=0.0,
        help="RADIO/ALIKE context encoder learning rate; zero inherits --learning-rate.",
    )
    parser.add_argument(
        "--texture-learning-rate",
        type=float,
        default=0.0,
        help="Pretrained real-RGB FPN learning rate; zero inherits --learning-rate.",
    )
    parser.add_argument(
        "--feature-training-mode",
        choices=("auto", "frozen_scalar_heads", "full"),
        default="auto",
        help=(
            "For identity_l0, auto freezes the pretrained multi-scale encoders and trains "
            "only FP32 identity/view/null heads. Full fine-tuning is reserved for an "
            "explicit later ablation after the scalar likelihood has passed its gate."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-margin", type=float, default=0.25)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.25)
    parser.add_argument("--hard-repeat-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--hard-pose-margin",
        type=float,
        default=0.25,
        help=(
            "Margin between the correct multi-point identity score and each "
            "coherent-wrong pose group."
        ),
    )
    parser.add_argument(
        "--hard-pose-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the train-only coherent-pose group contrastive loss.",
    )
    parser.add_argument(
        "--hard-pose-permutation-loss-weight",
        type=float,
        default=0.25,
        help=(
            "Weight for the support-derangement control applied to the coherent-pose "
            "group score."
        ),
    )
    parser.add_argument(
        "--minimum-hard-pose-points",
        type=int,
        default=4,
        help="Minimum distinct query anchors required for one coherent-pose group.",
    )
    parser.add_argument("--permutation-margin", type=float, default=0.05)
    parser.add_argument("--permutation-loss-weight", type=float, default=0.25)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=1)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument(
        "--minimum-posterior-candidate-top1-fraction",
        type=float,
        default=0.10,
        help=(
            "Train-only identity-L0 floor for exact-candidate top-1, separate from "
            "the explicit-null diagnostic and never used by the runtime scorer."
        ),
    )
    parser.add_argument(
        "--minimum-rgb-identity-posterior-top1-fraction",
        type=float,
        default=0.10,
        help="Train-only gate for the isolated RGB identity expert's exact-candidate top-1.",
    )
    parser.add_argument(
        "--minimum-rgb-identity-permutation-gap",
        type=float,
        default=0.05,
        help="Required isolated RGB score drop under support-appearance permutation.",
    )
    parser.add_argument(
        "--minimum-context-coherence-hard-repeat-gap",
        type=float,
        default=0.05,
        help="Required isolated context-expert margin over coherent repeated candidates.",
    )
    parser.add_argument(
        "--minimum-context-coherence-permutation-gap",
        type=float,
        default=0.05,
        help="Required isolated context score drop under support-appearance permutation.",
    )
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--minimum-position-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-repeat-gap", type=float, default=0.05)
    parser.add_argument("--minimum-inner-hard-pose-eligible-query-fraction", type=float, default=0.9)
    parser.add_argument("--minimum-hard-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-pose-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=65536.0,
        help=(
            "Initial GradScaler scale. Full P1 visual fine-tuning may use a lower "
            "explicit value to avoid losing early updates to FP16 overflow."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.rgb_context_radius_px),
        float(args.rgb_step_px),
        float(args.train_query_anchor_jitter_radius_px),
        float(args.max_abs_log_ratio),
        float(args.candidate_prior_logit_weight),
        float(args.posterior_cross_entropy_loss_weight),
        float(args.posterior_candidate_loss_weight),
        float(args.posterior_null_loss_weight),
        float(args.rgb_identity_posterior_loss_weight),
        float(args.rgb_identity_permutation_loss_weight),
        float(args.context_coherence_hard_repeat_loss_weight),
        float(args.context_coherence_permutation_loss_weight),
        float(args.learning_rate),
        float(args.head_learning_rate),
        float(args.projector_learning_rate),
        float(args.context_learning_rate),
        float(args.texture_learning_rate),
        float(args.weight_decay),
        float(args.identity_margin),
        float(args.identity_loss_weight),
        float(args.hard_repeat_margin),
        float(args.hard_repeat_loss_weight),
        float(args.hard_pose_margin),
        float(args.hard_pose_loss_weight),
        float(args.hard_pose_permutation_loss_weight),
        float(args.permutation_margin),
        float(args.permutation_loss_weight),
        float(args.minimum_inner_eligible_query_fraction),
        float(args.minimum_inner_hard_repeat_eligible_query_fraction),
        float(args.minimum_inner_hard_pose_eligible_query_fraction),
        float(args.minimum_win_fraction),
        float(args.minimum_posterior_candidate_top1_fraction),
        float(args.minimum_rgb_identity_posterior_top1_fraction),
        float(args.minimum_rgb_identity_permutation_gap),
        float(args.minimum_context_coherence_hard_repeat_gap),
        float(args.minimum_context_coherence_permutation_gap),
        float(args.minimum_normal_gap),
        float(args.minimum_visual_gap_delta),
        float(args.minimum_position_visual_gap),
        float(args.minimum_hard_repeat_win_fraction),
        float(args.minimum_hard_repeat_gap),
        float(args.minimum_hard_pose_win_fraction),
        float(args.minimum_hard_pose_gap),
        float(args.minimum_hard_pose_visual_gap_delta),
        float(args.gradient_clip_norm),
        float(args.rgb_cache_gb),
        float(args.amp_init_scale),
    )
    if (
        int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.max_points_per_query) < 4
        or int(args.validation_selector_point_budget) < 4
        or int(args.validation_selector_grid_rows) <= 0
        or int(args.validation_selector_grid_columns) <= 0
        or int(args.max_hard_repeat_edges_per_query) < 0
        or int(args.minimum_hard_pose_points) < 2
        or int(args.epochs) < 0
        or (not bool(args.initialization_only) and int(args.epochs) <= 0)
        or int(args.permutation_control_shift) <= 0
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or not all(math.isfinite(value) for value in values)
        or float(args.rgb_context_radius_px) <= 0.0
        or float(args.rgb_step_px) <= 0.0
        or float(args.train_query_anchor_jitter_radius_px) < 0.0
        or float(args.max_abs_log_ratio) <= 0.0
        or float(args.candidate_prior_logit_weight) < 0.0
        or float(args.posterior_cross_entropy_loss_weight) < 0.0
        or float(args.posterior_candidate_loss_weight) < 0.0
        or float(args.posterior_null_loss_weight) < 0.0
        or float(args.rgb_identity_posterior_loss_weight) < 0.0
        or float(args.rgb_identity_permutation_loss_weight) < 0.0
        or float(args.context_coherence_hard_repeat_loss_weight) < 0.0
        or float(args.context_coherence_permutation_loss_weight) < 0.0
        or (
            bool(args.reset_edge_head_final_for_prior_residual)
            and float(args.candidate_prior_logit_weight) <= 0.0
        )
        or float(args.learning_rate) <= 0.0
        or float(args.head_learning_rate) < 0.0
        or float(args.projector_learning_rate) < 0.0
        or float(args.context_learning_rate) < 0.0
        or float(args.texture_learning_rate) < 0.0
        or float(args.weight_decay) < 0.0
        or float(args.identity_margin) < 0.0
        or float(args.identity_loss_weight) < 0.0
        or float(args.hard_repeat_margin) < 0.0
        or float(args.hard_repeat_loss_weight) < 0.0
        or float(args.hard_pose_margin) < 0.0
        or float(args.hard_pose_loss_weight) < 0.0
        or float(args.hard_pose_permutation_loss_weight) < 0.0
        or float(args.permutation_margin) < 0.0
        or float(args.permutation_loss_weight) < 0.0
        or not 0.0 < float(args.minimum_inner_eligible_query_fraction) <= 1.0
        or not 0.0 < float(args.minimum_inner_hard_repeat_eligible_query_fraction) <= 1.0
        or not 0.0 < float(args.minimum_inner_hard_pose_eligible_query_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_posterior_candidate_top1_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_rgb_identity_posterior_top1_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_repeat_win_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_pose_win_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or float(args.minimum_rgb_identity_permutation_gap) < 0.0
        or float(args.minimum_context_coherence_hard_repeat_gap) < 0.0
        or float(args.minimum_context_coherence_permutation_gap) < 0.0
        or float(args.minimum_visual_gap_delta) < 0.0
        or float(args.minimum_position_visual_gap) < 0.0
        or float(args.minimum_hard_repeat_gap) < 0.0
        or float(args.minimum_hard_pose_gap) < 0.0
        or float(args.minimum_hard_pose_visual_gap_delta) < 0.0
        or float(args.gradient_clip_norm) <= 0.0
        or float(args.rgb_cache_gb) <= 0.0
        or float(args.amp_init_scale) <= 0.0
        or not (
            str(args.texture_observation_pretrain_checkpoint).strip()
            or str(args.identity_observation_pretrain_checkpoint).strip()
        )
    ):
        raise ValueError("identity LLR training arguments are invalid")
    if float(args.posterior_cross_entropy_loss_weight) > 0.0 and (
        float(args.posterior_candidate_loss_weight) <= 0.0
        or float(args.posterior_null_loss_weight) <= 0.0
    ):
        raise ValueError(
            "candidate/null posterior supervision requires positive weights for both classes"
        )
    if str(args.training_stage) == "identity_l0" and (
        float(args.hard_pose_loss_weight) > 0.0
        or float(args.hard_pose_permutation_loss_weight) > 0.0
        or float(args.posterior_cross_entropy_loss_weight) <= 0.0
        or float(args.candidate_prior_logit_weight) <= 0.0
        or float(args.rgb_identity_posterior_loss_weight) <= 0.0
        or float(args.rgb_identity_permutation_loss_weight) <= 0.0
        or float(args.context_coherence_hard_repeat_loss_weight) <= 0.0
        or float(args.context_coherence_permutation_loss_weight) <= 0.0
    ):
        raise ValueError(
            "identity_l0 requires both independent expert losses, explicit candidate/null posterior supervision, and no pose-group loss"
        )
    if (
        str(args.training_stage) == "identity_l0"
        and str(args.feature_training_mode) == "full"
        and not bool(args.allow_full_identity_l0_feature_finetune)
    ):
        raise ValueError(
            "full identity_l0 feature fine-tuning requires its explicit detector-anchor ablation flag"
        )
    if float(args.train_query_anchor_jitter_radius_px) > 0.0 and (
        str(args.training_stage) != "identity_l0"
        or str(args.feature_training_mode) != "full"
        or not bool(args.allow_full_identity_l0_feature_finetune)
    ):
        raise ValueError(
            "query-anchor jitter requires the explicit full identity_l0 detector-anchor ablation"
        )
    if bool(args.allow_full_identity_l0_feature_finetune) and (
        str(args.training_stage) != "identity_l0"
        or str(args.feature_training_mode) != "full"
        or float(args.train_query_anchor_jitter_radius_px) <= 0.0
    ):
        raise ValueError(
            "the full identity_l0 feature-finetune flag requires identity_l0, full mode, and positive query-anchor jitter"
        )
    return resolve_candidate_pose_rgb_spatial_identity_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _reduce(state: _DistributedState, value: torch.Tensor) -> torch.Tensor:
    output = value.detach().clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def _effective_learning_rate(*, requested: float, fallback: float) -> float:
    """Resolve an optional per-module rate without silently changing old runs."""

    value = float(requested)
    base = float(fallback)
    if not math.isfinite(value) or not math.isfinite(base) or value < 0.0 or base <= 0.0:
        raise ValueError("identity LLR learning rate is invalid")
    return base if value == 0.0 else value


def _resolve_feature_training_mode(args: argparse.Namespace) -> str:
    """Resolve the stage-safe default without changing later pose experiments."""

    requested = str(args.feature_training_mode)
    if requested not in {"auto", "frozen_scalar_heads", "full"}:
        raise ValueError("identity LLR feature-training mode is invalid")
    if requested != "auto":
        return requested
    return "frozen_scalar_heads" if str(args.training_stage) == "identity_l0" else "full"


_IDENTITY_LLR_SCALAR_HEAD_PREFIXES = (
    "rgb_identity_head.",
    "context_coherence_head.",
    "support_view_head.",
    "null_head.",
)
_IDENTITY_LLR_EXPERT_HEAD_PREFIXES = {
    "rgb_identity": ("rgb_identity_head.",),
    "context_coherence": ("context_coherence_head.",),
}


def _configure_identity_llr_feature_training(
    *, model: CandidatePoseRGBSpatialIdentityLLR, mode: str
) -> dict[str, object]:
    """Freeze unstable feature backprop during the sparse identity-L0 stage.

    The broad RADIO/ALIKE/RGB encoder is already trained before P1 fitting.
    Under sparse P1 hard repeats, AMP gradients through the attention/FPN
    feature stack can overflow and make GradScaler skip the *entire* update.
    ``frozen_scalar_heads`` therefore calibrates only the target-free scalar
    likelihood/view/null heads.  Frozen modules also stay in ``eval`` mode so
    BatchNorm statistics cannot drift while their weights are fixed.
    """

    training_mode = str(mode)
    if training_mode not in {"frozen_scalar_heads", "full"}:
        raise ValueError("identity LLR feature-training mode is invalid")
    scalar_parameters = 0
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        is_scalar = name.startswith(_IDENTITY_LLR_SCALAR_HEAD_PREFIXES)
        # ``edge_head`` is the retained v2 fused-head state slot.  V3 never
        # consumes it in ``forward``; leaving it trainable makes DDP wait for
        # a gradient that cannot exist.  It remains frozen even in a future
        # full feature ablation and is kept solely for checkpoint migration.
        is_legacy_fused = name.startswith("edge_head.")
        trainable = (training_mode == "full" or is_scalar) and not is_legacy_fused
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_parameters += 1
            scalar_parameters += int(is_scalar)
        else:
            frozen_parameters += 1
    if scalar_parameters == 0:
        raise RuntimeError("identity LLR scalar heads were not made trainable")
    # Activation checkpointing only helps when gradients must flow through the
    # expensive feature stack.  Disabling it for frozen encoders avoids a
    # needless second forward and lowers the L0 runtime cost.
    model.activation_checkpointing = training_mode == "full"
    return {
        "mode": training_mode,
        "trainable_parameter_tensor_count": int(trainable_parameters),
        "scalar_parameter_tensor_count": int(scalar_parameters),
        "frozen_parameter_tensor_count": int(frozen_parameters),
        "activation_checkpointing": bool(model.activation_checkpointing),
    }


def _set_identity_llr_train_mode(*, model: nn.Module, feature_training_mode: str) -> None:
    """Enter training mode while keeping frozen feature modules behaviorally fixed."""

    model.train()
    if str(feature_training_mode) != "frozen_scalar_heads":
        return
    core = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(core, CandidatePoseRGBSpatialIdentityLLR):
        raise TypeError("identity LLR train-mode setup requires the identity model")
    for module in (
        core.context_encoders,
        core.global_projections,
        core.texture_encoder,
        core.context_projector,
        core.rgb_projector,
    ):
        module.eval()


def _query_anchor_jitter_seed(*, seed: int, epoch: int, query_id: str) -> int:
    """Return a stable target-free seed for one train query and epoch."""

    if int(seed) < 0 or int(epoch) < 0 or not str(query_id):
        raise ValueError("query-anchor jitter seed inputs are invalid")
    digest = hashlib.sha256(
        f"identity_llr_query_anchor_jitter_v1\0{int(seed)}\0{int(epoch)}\0{query_id}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _jitter_runtime_query_anchors(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    radius_px: float,
    coordinate_image_size: tuple[int, int],
    interior_margin_px: float,
    seed: int,
) -> CandidatePoseRGBSpatialRuntime:
    """Perturb only train-time query crops while preserving the frozen P1 layout.

    The perturbation is sampled uniformly in radius and angle, matching the
    observed detector-anchor displacement scale without using a registered
    observation, pose, residual, track, or candidate score.  Candidate tracks,
    support observations, maplet masses, and coarse/null priors remain exactly
    unchanged.  Clamping uses the RGB crop interior rather than letting a
    boundary artifact become an accidental training signal.
    """

    active = runtime
    radius = float(radius_px)
    width, height = (int(coordinate_image_size[0]), int(coordinate_image_size[1]))
    margin = float(interior_margin_px)
    if (
        not math.isfinite(radius)
        or not math.isfinite(margin)
        or radius < 0.0
        or margin < 0.0
        or width <= 0
        or height <= 0
        or width <= 2.0 * margin
        or height <= 2.0 * margin
        or int(seed) < 0
    ):
        raise ValueError("query-anchor jitter inputs are invalid")
    if radius == 0.0:
        return active
    generator = np.random.default_rng(int(seed))
    count = int(active.point_count)
    angles = generator.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
    distances = generator.uniform(0.0, radius, size=count).astype(np.float32)
    offsets = np.stack(
        (distances * np.cos(angles), distances * np.sin(angles)), axis=1
    ).astype(np.float32, copy=False)
    delta = torch.as_tensor(offsets, dtype=torch.float32, device=active.query_xy.device)
    minimum = torch.tensor((margin, margin), dtype=torch.float32, device=delta.device)
    maximum = torch.tensor(
        (float(width - 1) - margin, float(height - 1) - margin),
        dtype=torch.float32,
        device=delta.device,
    )
    query_xy = torch.maximum(torch.minimum(active.query_xy + delta, maximum), minimum)
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=active.query_image_indices,
        query_xy=query_xy,
        support_image_indices=active.support_image_indices,
        support_xy=active.support_xy,
        support_view_valid=active.support_view_valid,
        candidate_view_weights=active.candidate_view_weights,
        candidate_probabilities=active.candidate_probabilities,
        null_probabilities=active.null_probabilities,
    )


def _select_identity_l0_group_points(
    *,
    group: TrainQueryGroup,
    max_points: int,
    seed: int,
    required_source_point_ids: Sequence[int] = (),
) -> np.ndarray:
    """Preserve both exact-candidate and explicit-null rows for L0 training.

    The generic spatial sampler keeps observed rows and coherent-repeat
    sources, which is correct for a density loss but can starve the new null
    posterior.  L0 reserves up to one quarter of its immutable point budget
    for complete all-dustbin candidate groups while retaining the rest for
    exact identities and current hard-repeat sources.  This policy is strictly
    train-side; validation still selects target-free layout rows first.
    """

    limit = int(max_points)
    count = int(group.point_count)
    if limit < 4 or count <= 0:
        raise ValueError("identity L0 point sampling arguments are invalid")
    if limit >= count:
        return np.arange(count, dtype=np.int64)
    observed_mask = np.asarray(
        group.spatial_target_observed & group.spatial_target_supervised, dtype=bool
    )
    dustbin = np.asarray(group.spatial_target_dustbin, dtype=bool)
    supervised = np.asarray(group.spatial_target_supervised, dtype=bool)
    if (
        observed_mask.ndim != 2
        or observed_mask.shape[0] != count
        or dustbin.shape != observed_mask.shape
        or supervised.shape != observed_mask.shape
    ):
        raise ValueError("identity L0 target rows are incompatible with the runtime group")
    observed = np.flatnonzero(np.any(observed_mask, axis=1))
    null_rows = np.flatnonzero(
        np.all(supervised, axis=1)
        & np.all(dustbin, axis=1)
        & ~np.any(observed_mask, axis=1)
    )
    source_ids = np.asarray(group.source_point_ids, dtype=np.int64).reshape(-1)
    if source_ids.shape != (count,):
        raise ValueError("identity L0 source-point ids are invalid")
    positions_by_source = {int(value): index for index, value in enumerate(source_ids.tolist())}
    required_ids = np.unique(np.asarray(required_source_point_ids, dtype=np.int64).reshape(-1))
    if any(int(value) not in positions_by_source for value in required_ids.tolist()):
        raise ValueError("identity L0 hard-repeat source is absent from its query group")
    required = np.asarray(
        [positions_by_source[int(value)] for value in required_ids.tolist()], dtype=np.int64
    )
    positive_priority = np.unique(np.concatenate([observed, required])).astype(np.int64)
    # The seed is stable across processes and independent of target values.
    query_hash = sum((index + 1) * ord(char) for index, char in enumerate(str(group.query_id)))
    generator = np.random.default_rng(int(seed) ^ int(query_hash))

    def take(values: np.ndarray, amount: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        if amount <= 0 or len(values) == 0:
            return np.empty((0,), dtype=np.int64)
        if len(values) <= amount:
            return values
        return np.asarray(generator.choice(values, size=amount, replace=False), dtype=np.int64)

    if len(positive_priority) + len(null_rows) <= limit:
        selected = np.concatenate([positive_priority, null_rows])
    else:
        null_budget = min(len(null_rows), max(1, limit // 4))
        positive_budget = limit - null_budget
        selected_positive = take(positive_priority, positive_budget)
        selected_null = take(null_rows, limit - len(selected_positive))
        selected = np.concatenate([selected_positive, selected_null])
    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(count, dtype=np.int64), selected, assume_unique=False)
        selected = np.concatenate([selected, take(remaining, limit - len(selected))])
    if len(selected) != limit:
        raise RuntimeError("identity L0 point sampler did not fill its fixed budget")
    return np.sort(np.unique(selected)).astype(np.int64)


def _identity_llr_optimizer_parameter_groups(
    *, model: CandidatePoseRGBSpatialIdentityLLR, args: argparse.Namespace
) -> list[dict[str, object]]:
    """Keep the neutral LLR head fast while preserving the RGB FPN initializer.

    The final LLR layer starts at zero so that an untrained model is exactly
    neutral.  Giving every parameter the small FPN fine-tuning rate leaves
    that head effectively frozen for the short, scarce hard-repeat protocol.
    This grouping is optimization-only: it neither changes encoder inputs nor
    exposes target-bearing fields to the runtime scorer.
    """

    rates = {
        "edge_head": _effective_learning_rate(
            requested=float(args.head_learning_rate), fallback=float(args.learning_rate)
        ),
        "projectors": _effective_learning_rate(
            requested=float(args.projector_learning_rate), fallback=float(args.learning_rate)
        ),
        "context": _effective_learning_rate(
            requested=float(args.context_learning_rate), fallback=float(args.learning_rate)
        ),
        "texture": _effective_learning_rate(
            requested=float(args.texture_learning_rate), fallback=float(args.learning_rate)
        ),
    }
    buckets: dict[str, list[nn.Parameter]] = {name: [] for name in rates}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(
            (
                "edge_head.",
                "rgb_identity_head.",
                "context_coherence_head.",
                "support_view_head.",
                "null_head.",
            )
        ):
            bucket = "edge_head"
        elif name.startswith("context_projector.") or name.startswith("rgb_projector."):
            bucket = "projectors"
        elif name.startswith("texture_encoder."):
            bucket = "texture"
        elif name.startswith("context_encoders.") or name.startswith("global_projections."):
            bucket = "context"
        else:  # Keep newly introduced trainable parameters explicit by default.
            raise ValueError(f"identity LLR optimizer has unclassified parameter: {name}")
        buckets[bucket].append(parameter)
    seen = [id(parameter) for parameters in buckets.values() for parameter in parameters]
    if not seen or len(seen) != len(set(seen)):
        raise RuntimeError("identity LLR optimizer parameter grouping is incomplete or duplicated")
    return [
        {"params": parameters, "lr": rates[name], "group_name": name}
        for name, parameters in buckets.items()
        if parameters
    ]


def _identity_llr_head_statistics(model: nn.Module) -> dict[str, float]:
    """Expose the neutral-head escape diagnostic in every epoch record."""

    core = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(core, CandidatePoseRGBSpatialIdentityLLR):
        raise TypeError("identity LLR diagnostics require the identity LLR model")
    output: dict[str, float] = {}
    for name in ("edge_head", "rgb_identity_head", "context_coherence_head"):
        head = getattr(core, name)
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise RuntimeError("identity LLR final edge head is invalid")
        weight = final.weight.detach().float()
        bias = final.bias.detach().float()
        output.update(
            {
                f"{name}_final_weight_abs_mean": float(weight.abs().mean().item()),
                f"{name}_final_weight_l2": float(weight.norm().item()),
                f"{name}_final_bias_abs": float(bias.abs().mean().item()),
            }
        )
    return output


def _identity_llr_scalar_head_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return the small calibration-head tensors whose updates must be observable."""

    core = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(core, CandidatePoseRGBSpatialIdentityLLR):
        raise TypeError("identity LLR scalar diagnostics require the identity model")
    output = {
        name: parameter
        for name, parameter in core.named_parameters()
        if name.startswith(_IDENTITY_LLR_SCALAR_HEAD_PREFIXES)
    }
    if not output:
        raise RuntimeError("identity LLR scalar diagnostics found no calibration heads")
    return output


def _identity_llr_scalar_gradient_diagnostics(model: nn.Module) -> tuple[float, bool]:
    """Measure post-unscale scalar gradients before GradScaler decides a step."""

    parameters = _identity_llr_scalar_head_parameters(model)
    squared_norm = 0.0
    finite = True
    for parameter in parameters.values():
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach().float()
        parameter_finite = bool(torch.isfinite(detached).all())
        finite = finite and parameter_finite
        if parameter_finite:
            squared_norm += float(detached.square().sum().item())
    return math.sqrt(squared_norm), finite


def _identity_llr_expert_update_l2(
    *,
    model: nn.Module,
    before: Mapping[str, torch.Tensor],
    expert: str,
) -> float:
    """Measure a specific expert's actual optimizer update after one step.

    A nonzero aggregate scalar update can hide a dead RGB or context branch.
    The identity-L0 smoke protocol therefore records both expert update norms
    and rejects a frozen-head epoch if either independently supervised head
    remained unchanged.
    """

    prefixes = _IDENTITY_LLR_EXPERT_HEAD_PREFIXES.get(str(expert))
    if prefixes is None:
        raise ValueError("identity LLR expert update name is invalid")
    current = _identity_llr_scalar_head_parameters(model)
    names = tuple(name for name in current if name.startswith(prefixes))
    if not names or any(name not in before for name in names):
        raise RuntimeError("identity LLR expert update parameters are incomplete")
    squared = 0.0
    for name in names:
        delta = current[name].detach().float() - before[name].detach().float()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("identity LLR expert update is non-finite")
        squared += float(delta.square().sum().item())
    return math.sqrt(squared)


def _reset_identity_llr_final_edge_head(model: CandidatePoseRGBSpatialIdentityLLR) -> None:
    """Start a fixed-prior residual fit from neutral visual evidence only."""

    if not isinstance(model, CandidatePoseRGBSpatialIdentityLLR):
        raise TypeError("identity LLR residual reset requires the identity LLR model")
    with torch.no_grad():
        for name in ("edge_head", "rgb_identity_head", "context_coherence_head"):
            final = getattr(model, name)[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("identity LLR final edge head is invalid")
            final.weight.zero_()
            final.bias.zero_()


def _reset_identity_l0_scalar_head_finals(
    model: CandidatePoseRGBSpatialIdentityLLR,
) -> tuple[str, ...]:
    """Start current-P1 identity calibration from neutral scalar evidence.

    Broad observation pretraining supplies the multi-scale encoders and head
    hidden representations, but its scalar likelihood was trained with a
    different candidate pool and may have seen a different source-availability
    pattern.  Reusing those final logits would make a current hard-repeat run
    look like a zero-shot likelihood transfer when it is not.  Keep the visual
    representation, but learn RGB/context/view/null calibration only from the
    current train-only P1 contract.
    """

    names = (
        "rgb_identity_head",
        "context_coherence_head",
        "support_view_head",
        "null_head",
    )
    with torch.no_grad():
        for name in names:
            final = getattr(model, name)[-1]
            if not isinstance(final, nn.Linear):
                raise RuntimeError("identity L0 scalar head final layer is invalid")
            final.weight.zero_()
            final.bias.zero_()
    return names


def _write_json_atomically(path: Path, value: object) -> None:
    """Persist interrupt-safe diagnostics without exposing partial JSON."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def _load_texture_observation_pretrain(
    *, model: CandidatePoseRGBSpatialIdentityLLR, path: Path, texture_feature_dim: int, hidden_dim: int
) -> dict[str, object]:
    """Copy only the gate-approved real-RGB FPN into the identity model."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RGB observation pretrain is absent: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("RGB observation pretrain is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format") != CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
    ):
        raise ValueError("RGB observation pretrain violates the target-free contract")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    config = metadata.get("config")
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(config, Mapping)
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
        or config.get("rgb_cost_volume_only") is not True
    ):
        raise ValueError("RGB observation pretrain is not compatible with identity LLR texture initialization")
    prefix = "texture_encoder."
    texture_state = {
        str(name)[len(prefix) :]: value
        for name, value in state_dict.items()
        if str(name).startswith(prefix)
    }
    if not texture_state:
        raise ValueError("RGB observation pretrain lacks texture encoder weights")
    try:
        model.texture_encoder.load_state_dict(texture_state, strict=True)
    except RuntimeError as error:
        raise ValueError("RGB observation pretrain texture encoder is incompatible") from error
    return {
        "path": str(checkpoint),
        "sha256": file_sha256_short(checkpoint),
        "selected_epoch": int(inner.get("selected_epoch", -1)),
    }


def _load_identity_observation_pretrain(
    *,
    model: CandidatePoseRGBSpatialIdentityLLR,
    path: Path,
    candidate_count: int,
    context_windows: Mapping[str, int],
    source_lineage: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Load only a fold-matched broad checkpoint that passed all visual gates."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"identity observation pretrain is absent: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("identity observation pretrain is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format") != REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("candidate_slot_permutation_equivariant") is not True
        or metadata.get("visual_evidence_gate_version")
        != REQUIRED_IDENTITY_OBSERVATION_PRETRAIN_GATE_VERSION
        or metadata.get("p1_initialization_allowed") is not True
        or int(metadata.get("fixed_candidate_count", -1)) != int(candidate_count)
    ):
        raise ValueError("identity observation pretrain violates the strict initialization contract")
    config = metadata.get("config")
    training = metadata.get("training")
    inner = training.get("inner_validation") if isinstance(training, Mapping) else None
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    fold = metadata.get("observation_pair_inner_validation")
    lineage = metadata.get("lineage")
    required_lineage = (
        "radio_final_context_cache_sha256",
        "radio_intermediate_context_cache_sha256",
        "alike_spatial_context_cache_sha256",
        "source_image_manifest_sha256",
        "rgb_coordinate_bridge",
    )
    if (
        not isinstance(config, Mapping)
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(fold, Mapping)
        or not isinstance(lineage, Mapping)
        or any(lineage.get(name) != source_lineage.get(name) for name in required_lineage)
        or int(fold.get("fold_count", -1)) != int(args.inner_validation_fold_count)
        or int(fold.get("fold_index", -1)) != int(args.inner_validation_fold_index)
        or not math.isclose(
            float(config.get("rgb_context_radius_px", float("nan"))),
            float(args.rgb_context_radius_px),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(config.get("rgb_step_px", float("nan"))),
            float(args.rgb_step_px),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or int(config.get("texture_feature_dim", -1)) != int(args.texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(args.hidden_dim)
        or not math.isclose(
            float(config.get("max_abs_log_ratio", float("nan"))),
            float(args.max_abs_log_ratio),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or dict(config.get("context_windows", {})) != dict(context_windows)
    ):
        raise ValueError("identity observation pretrain configuration, lineage, or fold is incompatible")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("identity observation pretrain state is incompatible") from error
    return {
        "kind": "fold_matched_broad_identity_observation_pretrain",
        "path": str(checkpoint),
        "sha256": file_sha256_short(checkpoint),
        "selected_epoch": int(inner.get("selected_epoch", -1)),
        "observation_pair_inner_validation": dict(fold),
        "independent_expert_initialization": "loaded_from_gate_approved_v3_source_masked_experts_v1",
    }


def _validate_identity_target_contract(
    *,
    targets: CandidatePoseRGBSpatialTrainingTargets,
    hard_targets: CandidatePoseRGBSpatialHardRepeatTargets,
) -> None:
    """Forbid projection-only labels in the candidate identity stage.

    The identity LLR is trained to separate the exact observed landmark from a
    coherent-repeat landmark.  A generic correct-pose projection target is a
    different supervision problem and can silently make a hard-repeat artifact
    from the registered-identity contract incompatible with its main loss.
    """

    target_metadata = targets.metadata
    hard_metadata = hard_targets.metadata
    accepted_hard_repeat_selections = {
        "registered_exact_track_correct_local_candidate_vs_different_"
        "coherent_wrong_local_candidate_min_linf_then_frozen_prior_v1",
        "registered_exact_track_correct_local_candidate_vs_all_or_capped_"
        "distinct_coherent_wrong_local_candidates_min_linf_then_frozen_prior_v2",
    }
    if (
        target_metadata.get("spatial_supervision_mode") != "registered_exact_identity"
        or hard_metadata.get("positive_requires_registered_exact_track") is not True
        or str(hard_metadata.get("selection", "")) not in accepted_hard_repeat_selections
    ):
        raise ValueError(
            "identity LLR requires registered-exact targets and matching coherent-repeat targets"
        )


def _candidate_llrs(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    edge_usable_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_prediction = prediction
    if edge_usable_override is not None:
        override = torch.as_tensor(
            edge_usable_override,
            dtype=torch.bool,
            device=prediction.edge_log_likelihood_ratios.device,
        )
        if override.shape != prediction.edge_usable.shape:
            raise ValueError("identity LLR edge-availability override is incompatible")
        active_prediction = CandidatePoseRGBSpatialIdentityLLREdgePrediction(
            edge_log_likelihood_ratios=prediction.edge_log_likelihood_ratios,
            edge_usable=prediction.edge_usable & override,
            support_view_logits=prediction.support_view_logits,
            point_null_log_likelihood_ratios=prediction.point_null_log_likelihood_ratios,
            rgb_edge_usable=prediction.rgb_edge_usable & override,
            context_edge_usable=prediction.context_edge_usable & override,
        )
    values = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=active_prediction,
        runtime=runtime,
        missing_edge_log_likelihood_ratio=0.0,
    )
    active_runtime = runtime.to(values.device)
    usable = torch.any(
        active_prediction.edge_usable & (active_runtime.candidate_view_weights > 0.0),
        dim=2,
    )
    if values.shape != usable.shape:
        raise RuntimeError("identity LLR candidate marginalization drifted")
    return values, usable


def _component_edge_only_prediction(
    *,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    component: str,
    edge_usable_override: torch.Tensor | None = None,
) -> CandidatePoseRGBSpatialIdentityLLREdgePrediction:
    """Expose one expert without learned view/null or fused-edge shortcuts.

    Expert supervision must not accidentally use the combined LLR, learned
    support-view posterior, or learned null residual.  This adapter preserves
    only the immutable runtime view masses used by the normal marginalizer and
    puts every learned non-edge scalar at its neutral value.  It is therefore
    safe to use for both the RGB identity and context coherence objectives.
    """

    fields = {
        "rgb_identity": "rgb_identity_edge_log_likelihood_ratios",
        "context_coherence": "context_coherence_edge_log_likelihood_ratios",
    }
    field = fields.get(str(component))
    if field is None:
        raise ValueError("identity LLR component name is invalid")
    values = torch.as_tensor(getattr(prediction, field))
    usable = candidate_pose_rgb_spatial_identity_component_edge_usable(
        prediction=prediction, component=str(component)
    ).to(device=values.device)
    if edge_usable_override is not None:
        override = torch.as_tensor(
            edge_usable_override,
            dtype=torch.bool,
            device=values.device,
        )
        if override.shape != usable.shape:
            raise ValueError("identity LLR component edge-availability override is incompatible")
        usable = usable & override
    if values.shape != usable.shape:
        raise ValueError("identity LLR component edge shape is invalid")
    return CandidatePoseRGBSpatialIdentityLLREdgePrediction(
        edge_log_likelihood_ratios=values,
        edge_usable=usable,
        support_view_logits=torch.zeros_like(values),
        point_null_log_likelihood_ratios=torch.zeros(
            (len(values),), dtype=values.dtype, device=values.device
        ),
        rgb_edge_usable=usable if str(component) == "rgb_identity" else torch.zeros_like(usable),
        context_edge_usable=(
            usable if str(component) == "context_coherence" else torch.zeros_like(usable)
        ),
    )


def _common_component_edge_usable(
    *,
    component: str,
    predictions: Sequence[CandidatePoseRGBSpatialIdentityLLREdgePrediction],
) -> torch.Tensor:
    """Intersect availability for one expert without requiring other sources."""

    if not predictions:
        raise ValueError("identity LLR component availability requires predictions")
    common = candidate_pose_rgb_spatial_identity_component_edge_usable(
        prediction=predictions[0], component=str(component)
    ).clone()
    for prediction in predictions[1:]:
        common &= candidate_pose_rgb_spatial_identity_component_edge_usable(
            prediction=prediction, component=str(component)
        ).to(device=common.device)
    return common


def _candidate_scores(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    candidate_prior_logit_weight: float,
    edge_usable_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse immutable candidate mass with a prior-free visual residual.

    ``prediction`` remains a target-free visual LLR.  The frozen coarse prior
    is added strictly after that LLR has been emitted, so it can participate in
    posterior-consistent supervision and runtime ranking without becoming an
    encoder shortcut.  A zero weight retains the legacy prior-free score space.
    """

    weight = float(candidate_prior_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("candidate-prior logit weight is invalid")
    values, usable = _candidate_llrs(
        runtime=runtime,
        prediction=prediction,
        edge_usable_override=edge_usable_override,
    )
    if weight == 0.0:
        return values, usable
    active = runtime.to(values.device)
    prior = active.candidate_probabilities.to(dtype=values.dtype)
    if prior.shape != values.shape or torch.any(prior < 0.0):
        raise ValueError("candidate prior does not match identity LLR scores")
    positive_prior = prior > 0.0
    scores = values + weight * torch.log(prior.clamp_min(torch.finfo(values.dtype).tiny))
    usable = usable & positive_prior
    return torch.where(positive_prior, scores, torch.zeros_like(scores)), usable


def _candidate_plus_null_logits(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    candidate_prior_logit_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join target-free visual residuals with immutable candidate/null mass.

    The returned null column is the fixed runtime null prior plus a learned
    candidate-set visual residual.  The residual is emitted before target
    labels are read, so explicit-null supervision cannot leak into inference.
    """

    weight = float(candidate_prior_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("candidate-plus-null prior weight is invalid")
    candidate_scores, candidate_usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=weight,
    )
    active = runtime.to(candidate_scores.device)
    null_prior = active.null_probabilities.to(dtype=candidate_scores.dtype)
    null_residual = torch.as_tensor(
        prediction.point_null_log_likelihood_ratios,
        dtype=candidate_scores.dtype,
        device=candidate_scores.device,
    ).reshape(-1)
    if (
        null_prior.shape != (len(candidate_scores),)
        or null_residual.shape != (len(candidate_scores),)
        or torch.any(null_prior < 0.0)
        or not torch.isfinite(null_residual).all()
    ):
        raise ValueError("candidate-plus-null visual residual inputs are invalid")
    candidate_logits = candidate_scores.masked_fill(~candidate_usable, -torch.inf)
    if weight == 0.0:
        null_logits = null_residual
    else:
        null_logits = torch.where(
            null_prior > 0.0,
            null_residual
            + weight * torch.log(null_prior.clamp_min(torch.finfo(candidate_scores.dtype).tiny)),
            torch.full_like(null_prior, -torch.inf),
        )
    logits = torch.cat([candidate_logits, null_logits[:, None]], dim=1)
    if not torch.all(torch.isfinite(logits) | torch.isneginf(logits)):
        raise ValueError("candidate-plus-null logits are non-finite")
    return logits, candidate_usable


def _registered_identity_or_null_labels(
    *,
    observed_candidate_mask: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build train-only candidate-or-null labels from the exact P1 contract."""

    observed = torch.as_tensor(observed_candidate_mask, dtype=torch.bool)
    dustbin = torch.as_tensor(target_dustbin, dtype=torch.bool, device=observed.device)
    supervised = torch.as_tensor(target_supervised, dtype=torch.bool, device=observed.device)
    if (
        observed.ndim != 2
        or dustbin.shape != observed.shape
        or supervised.shape != observed.shape
        or torch.any(observed & ~supervised)
        or torch.any(observed.sum(dim=1) > 1)
    ):
        raise ValueError("registered candidate-or-null targets are invalid")
    row_supervised = torch.any(supervised, dim=1)
    # The registered P1 identity artifact deliberately exposes whole candidate
    # groups or no label at all.  Partial rows would make a null label mean
    # "unobserved" instead of the intended "none of this frozen top-L set".
    if torch.any(row_supervised & ~torch.all(supervised, dim=1)):
        raise ValueError("registered candidate-or-null targets require complete candidate rows")
    observed_rows = torch.any(observed, dim=1)
    null_rows = row_supervised & ~observed_rows & torch.all(dustbin, dim=1)
    unresolved = row_supervised & ~observed_rows & ~null_rows
    if bool(unresolved.any()):
        raise ValueError("registered candidate-or-null target row has no unambiguous label")
    labels = torch.full(
        (len(observed),), -1, dtype=torch.long, device=observed.device
    )
    if bool(observed_rows.any()):
        labels[observed_rows] = torch.argmax(
            observed[observed_rows].to(dtype=torch.long), dim=1
        )
    labels[null_rows] = int(observed.shape[1])
    return labels, observed_rows, null_rows


def _registered_identity_or_null_posterior_cross_entropy_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    observed_candidate_mask: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
    candidate_prior_logit_weight: float,
    candidate_loss_weight: float = 1.0,
    null_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train exact candidate retrieval and explicit top-L null jointly.

    A correct candidate row is supervised against all fixed candidates plus the
    immutable null.  A complete all-dustbin row is supervised as null.  Rows
    outside the train-only exact-identity contract are excluded entirely.
    """

    if (
        not math.isfinite(float(candidate_loss_weight))
        or not math.isfinite(float(null_loss_weight))
        or float(candidate_loss_weight) <= 0.0
        or float(null_loss_weight) <= 0.0
    ):
        raise ValueError("candidate/null posterior loss weights must be finite and positive")
    logits, candidate_usable = _candidate_plus_null_logits(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
    )
    labels, observed_rows, null_rows = _registered_identity_or_null_labels(
        observed_candidate_mask=observed_candidate_mask,
        target_dustbin=target_dustbin,
        target_supervised=target_supervised,
    )
    labels = labels.to(device=logits.device)
    observed_rows = observed_rows.to(device=logits.device)
    null_rows = null_rows.to(device=logits.device)
    if labels.shape != (len(logits),):
        raise ValueError("candidate-or-null labels do not match prediction rows")
    candidate_labels = labels.clamp(min=0, max=logits.shape[1] - 2)
    observed_usable = candidate_usable.gather(1, candidate_labels[:, None]).squeeze(1)
    active = (observed_rows & observed_usable) | null_rows
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "candidate_active": 0.0,
            "null_active": 0.0,
            "loss": 0.0,
            "optimization_loss": 0.0,
            "candidate_loss": 0.0,
            "null_loss": 0.0,
            "mean_target_probability": 0.0,
            "top1_fraction": 0.0,
            "candidate_top1_fraction": 0.0,
            "null_top1_fraction": 0.0,
        }
    active_logits = logits[active]
    active_labels = labels[active]
    if torch.any(active_labels < 0) or torch.any(active_labels >= active_logits.shape[1]):
        raise RuntimeError("candidate-or-null active labels are unresolved")
    per_row_loss = F.cross_entropy(active_logits, active_labels, reduction="none")
    probabilities = F.log_softmax(active_logits, dim=1)
    target_probability = probabilities[
        torch.arange(len(active_labels), device=logits.device), active_labels
    ].exp()
    top1 = active_logits.detach().argmax(dim=1) == active_labels
    active_candidates = observed_rows[active]
    active_null = null_rows[active]
    row_weights = torch.where(
        active_candidates,
        torch.full_like(per_row_loss, float(candidate_loss_weight)),
        torch.full_like(per_row_loss, float(null_loss_weight)),
    )
    loss = (per_row_loss * row_weights).sum() / row_weights.sum().clamp_min(
        torch.finfo(per_row_loss.dtype).tiny
    )
    unweighted_loss = per_row_loss.mean()
    return loss, {
        "active": float(len(active_labels)),
        "candidate_active": float(active_candidates.sum().item()),
        "null_active": float(active_null.sum().item()),
        "loss": float(unweighted_loss.detach().item()),
        "optimization_loss": float(loss.detach().item()),
        "candidate_loss": float(
            per_row_loss[active_candidates].detach().mean().item()
            if bool(active_candidates.any())
            else 0.0
        ),
        "null_loss": float(
            per_row_loss[active_null].detach().mean().item()
            if bool(active_null.any())
            else 0.0
        ),
        "mean_target_probability": float(target_probability.detach().mean().item()),
        "top1_fraction": float(top1.float().mean().item()),
        "candidate_top1_fraction": float(
            top1[active_candidates].float().mean().item() if bool(active_candidates.any()) else 0.0
        ),
        "null_top1_fraction": float(
            top1[active_null].float().mean().item() if bool(active_null.any()) else 0.0
        ),
    }


def _registered_identity_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    observed_candidate_mask: torch.Tensor,
    margin: float,
    candidate_prior_logit_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank registered exact identity above all usable distinct candidates."""

    target = torch.as_tensor(
        observed_candidate_mask, dtype=torch.bool, device=prediction.edge_log_likelihood_ratios.device
    )
    values, usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
    )
    if target.shape != values.shape or torch.any(target.sum(dim=1) > 1):
        raise ValueError("registered identity targets do not match fixed candidate layout")
    point_indices, positive_indices = torch.nonzero(target, as_tuple=True)
    if len(point_indices) == 0:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "loss": 0.0,
            "mean_gap": 0.0,
            "win_fraction": 0.0,
        }
    selected_usable = usable[point_indices, positive_indices]
    negative_mask = usable[point_indices].clone()
    negative_mask[torch.arange(len(point_indices), device=values.device), positive_indices] = False
    has_negative = torch.any(negative_mask, dim=1)
    active = selected_usable & has_negative
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "loss": 0.0,
            "mean_gap": 0.0,
            "win_fraction": 0.0,
        }
    positive = values[point_indices[active], positive_indices[active]]
    negative = values[point_indices[active]].masked_fill(
        ~negative_mask[active], -torch.inf
    ).amax(dim=1)
    gaps = positive - negative
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(active.sum().item()),
        "loss": float(loss.detach().item()),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


def _registered_identity_posterior_cross_entropy_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    observed_candidate_mask: torch.Tensor,
    candidate_prior_logit_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calibrate a candidate-plus-null posterior from fixed prior and visual LLR.

    Only rows with an exact registered observation are supervised.  The null
    mass is immutable and receives no visual residual; rows without an exact
    observed candidate remain excluded rather than being incorrectly taught as
    dustbin examples.
    """

    weight = float(candidate_prior_logit_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("posterior candidate-prior logit weight is invalid")
    target = torch.as_tensor(
        observed_candidate_mask,
        dtype=torch.bool,
        device=prediction.edge_log_likelihood_ratios.device,
    )
    scores, usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=weight,
    )
    if target.shape != scores.shape or torch.any(target.sum(dim=1) > 1):
        raise ValueError("posterior identity targets do not match fixed candidate layout")
    points, candidates = torch.nonzero(target, as_tuple=True)
    if len(points) == 0:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "loss": 0.0,
            "mean_target_probability": 0.0,
            "top1_fraction": 0.0,
        }
    active = usable[points, candidates]
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "loss": 0.0,
            "mean_target_probability": 0.0,
            "top1_fraction": 0.0,
        }
    active_points = points[active]
    active_candidates = candidates[active]
    candidate_logits = scores[active_points].masked_fill(~usable[active_points], -torch.inf)
    active_runtime = runtime.to(scores.device)
    null_prior = active_runtime.null_probabilities[active_points].to(dtype=scores.dtype)
    if torch.any(null_prior < 0.0):
        raise ValueError("posterior identity null mass cannot be negative")
    null_logits = torch.where(
        null_prior > 0.0,
        weight * torch.log(null_prior.clamp_min(torch.finfo(scores.dtype).tiny)),
        torch.full_like(null_prior, -torch.inf),
    )
    logits = torch.cat([candidate_logits, null_logits[:, None]], dim=1)
    loss = F.cross_entropy(logits, active_candidates)
    log_probabilities = F.log_softmax(logits, dim=1)
    target_probability = log_probabilities[
        torch.arange(len(active_candidates), device=logits.device), active_candidates
    ].exp()
    return loss, {
        "active": float(len(active_candidates)),
        "loss": float(loss.detach().item()),
        "mean_target_probability": float(target_probability.detach().mean().item()),
        "top1_fraction": float(
            (logits.detach().argmax(dim=1) == active_candidates).float().mean().item()
        ),
    }


def _hard_repeat_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    candidate_prior_logit_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train exact candidates against the strongest current repeat per point.

    A V2 train-only target may expose several distinct wrong candidates for a
    single fixed query point.  The runtime identity factor is pose-independent,
    so treating those rows as independent losses would let a repeated easy
    negative change the objective weight.  Collapse them to the highest-score
    usable candidate: every point must beat its hardest coherent repeat.
    """

    if hard_batch is None:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active": 0.0, "loss": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    values, usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
    )
    points = hard_batch.point_indices.to(device=values.device, dtype=torch.long)
    positive = hard_batch.positive_candidate_indices.to(device=values.device, dtype=torch.long)
    negative = hard_batch.negative_candidate_indices.to(device=values.device, dtype=torch.long)
    if (
        points.ndim != 1
        or positive.shape != points.shape
        or negative.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= len(values))
        or torch.any(positive < 0)
        or torch.any(positive >= values.shape[1])
        or torch.any(negative < 0)
        or torch.any(negative >= values.shape[1])
    ):
        raise ValueError("hard-repeat identity indices are invalid")
    active_edges = usable[points, positive] & usable[points, negative]
    if not bool(active_edges.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active": 0.0, "loss": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    active_points = points[active_edges]
    active_positive = positive[active_edges]
    active_negative = negative[active_edges]
    group_keys = active_points * int(values.shape[1]) + active_positive
    gaps_by_group: list[torch.Tensor] = []
    for group_key in torch.unique(group_keys, sorted=True):
        members = group_keys == group_key
        first = torch.nonzero(members, as_tuple=False)[0, 0]
        point_index = active_points[first]
        positive_index = active_positive[first]
        positive_value = values[point_index, positive_index]
        strongest_negative = values[point_index, active_negative[members]].amax()
        gaps_by_group.append(positive_value - strongest_negative)
    gaps = torch.stack(gaps_by_group)
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(len(gaps_by_group)),
        "loss": float(loss.detach().item()),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


def _hard_pose_group_gap_values(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    hard_batch: HardRepeatBatch | None,
    minimum_points: int,
    candidate_prior_logit_weight: float,
    edge_usable_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Aggregate train-only hard edges into mutually coherent pose groups.

    A ``pair_id`` represents one mined coherent-wrong pose.  The learned model
    still sees only target-free query/support appearance; pair membership is
    joined after edge LLRs have been emitted.  For every point in that pose
    mode, V2 multi-negative rows collapse to the strongest wrong candidate.
    The group score is the mean point LLR gap, preventing longer pose groups
    from changing the objective weight while preserving their multi-point
    evidence requirement.
    """

    required = int(minimum_points)
    if required < 2:
        raise ValueError("hard-pose group requires at least two distinct points")
    device = prediction.edge_log_likelihood_ratios.device
    empty_pairs = torch.empty((0,), dtype=torch.long, device=device)
    empty_gaps = torch.empty((0,), dtype=prediction.edge_log_likelihood_ratios.dtype, device=device)
    if hard_batch is None or hard_batch.pair_ids is None:
        return empty_pairs, empty_gaps, 0
    values, usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
        edge_usable_override=edge_usable_override,
    )
    points = hard_batch.point_indices.to(device=values.device, dtype=torch.long)
    positive = hard_batch.positive_candidate_indices.to(device=values.device, dtype=torch.long)
    negative = hard_batch.negative_candidate_indices.to(device=values.device, dtype=torch.long)
    pair_ids = hard_batch.pair_ids.to(device=values.device, dtype=torch.long)
    if (
        points.ndim != 1
        or positive.shape != points.shape
        or negative.shape != points.shape
        or pair_ids.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= len(values))
        or torch.any(positive < 0)
        or torch.any(positive >= values.shape[1])
        or torch.any(negative < 0)
        or torch.any(negative >= values.shape[1])
        or torch.any(pair_ids < 0)
    ):
        raise ValueError("hard-pose identity indices are invalid")
    active_edges = usable[points, positive] & usable[points, negative]
    if not bool(active_edges.any()):
        return empty_pairs, empty_gaps, 0
    active_pairs = pair_ids[active_edges]
    active_points = points[active_edges]
    active_positive = positive[active_edges]
    active_negative = negative[active_edges]
    pose_ids: list[torch.Tensor] = []
    pose_gaps: list[torch.Tensor] = []
    active_point_count = 0
    for pair_id in torch.unique(active_pairs, sorted=True):
        pair_members = active_pairs == pair_id
        pair_points = active_points[pair_members]
        pair_positive = active_positive[pair_members]
        pair_negative = active_negative[pair_members]
        point_gaps: list[torch.Tensor] = []
        for point_index in torch.unique(pair_points, sorted=True):
            point_members = pair_points == point_index
            point_positive = pair_positive[point_members]
            if len(torch.unique(point_positive)) != 1:
                raise ValueError("hard-pose target changes the positive identity within one point")
            positive_index = point_positive[0]
            positive_value = values[point_index, positive_index]
            strongest_negative = values[point_index, pair_negative[point_members]].amax()
            point_gaps.append(positive_value - strongest_negative)
        if len(point_gaps) < required:
            continue
        pose_ids.append(pair_id)
        pose_gaps.append(torch.stack(point_gaps).mean())
        active_point_count += len(point_gaps)
    if not pose_gaps:
        return empty_pairs, empty_gaps, 0
    return torch.stack(pose_ids), torch.stack(pose_gaps), active_point_count


def _hard_pose_group_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    minimum_points: int,
    candidate_prior_logit_weight: float = 0.0,
    edge_usable_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank correct multi-point evidence above every coherent-wrong pose."""

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("hard-pose margin is invalid")
    _, gaps, active_points = _hard_pose_group_gap_values(
        runtime=runtime,
        prediction=prediction,
        hard_batch=hard_batch,
        minimum_points=int(minimum_points),
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
        edge_usable_override=edge_usable_override,
    )
    if len(gaps) == 0:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "active_points": 0.0,
            "loss": 0.0,
            "mean_gap": 0.0,
            "win_fraction": 0.0,
        }
    loss = F.softplus(torch.as_tensor(target_margin, device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(len(gaps)),
        "active_points": float(active_points),
        "loss": float(loss.detach().item()),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


def _hard_pose_group_permutation_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    hard_batch: HardRepeatBatch | None,
    margin: float,
    minimum_points: int,
    candidate_prior_logit_weight: float = 0.0,
    common_edge_usable: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require coherent-pose evidence to fall under support-view derangement."""

    target_margin = float(margin)
    if not math.isfinite(target_margin) or target_margin < 0.0:
        raise ValueError("hard-pose permutation margin is invalid")
    normal_ids, normal_gaps, normal_points = _hard_pose_group_gap_values(
        runtime=runtime,
        prediction=prediction,
        hard_batch=hard_batch,
        minimum_points=int(minimum_points),
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
        edge_usable_override=common_edge_usable,
    )
    permuted_ids, permuted_gaps, permuted_points = _hard_pose_group_gap_values(
        runtime=permuted_runtime,
        prediction=permuted_prediction,
        hard_batch=hard_batch,
        minimum_points=int(minimum_points),
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
        edge_usable_override=common_edge_usable,
    )
    if len(normal_gaps) == 0 and len(permuted_gaps) == 0:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {
            "active": 0.0,
            "active_points": 0.0,
            "loss": 0.0,
            "mean_gap": 0.0,
            "win_fraction": 0.0,
        }
    if (
        not torch.equal(normal_ids, permuted_ids)
        or len(normal_gaps) != len(permuted_gaps)
        or normal_points != permuted_points
    ):
        raise RuntimeError("hard-pose permutation control changed target availability")
    gaps = normal_gaps - permuted_gaps
    loss = F.softplus(torch.as_tensor(target_margin, device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(len(gaps)),
        "active_points": float(normal_points),
        "loss": float(loss.detach().item()),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


def _registered_permutation_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: CandidatePoseRGBSpatialIdentityLLREdgePrediction,
    observed_candidate_mask: torch.Tensor,
    margin: float,
    candidate_prior_logit_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require exact identity evidence to fall under support-view derangement."""

    target = torch.as_tensor(
        observed_candidate_mask, dtype=torch.bool, device=prediction.edge_log_likelihood_ratios.device
    )
    normal_values, normal_usable = _candidate_scores(
        runtime=runtime,
        prediction=prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
    )
    permuted_values, permuted_usable = _candidate_scores(
        runtime=permuted_runtime,
        prediction=permuted_prediction,
        candidate_prior_logit_weight=float(candidate_prior_logit_weight),
    )
    if target.shape != normal_values.shape or normal_values.shape != permuted_values.shape:
        raise ValueError("support permutation identity targets are invalid")
    points, candidates = torch.nonzero(target, as_tuple=True)
    if len(points) == 0:
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active": 0.0, "loss": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    active = normal_usable[points, candidates] & permuted_usable[points, candidates]
    if not bool(active.any()):
        zero = prediction.edge_log_likelihood_ratios.sum() * 0.0
        return zero, {"active": 0.0, "loss": 0.0, "mean_gap": 0.0, "win_fraction": 0.0}
    gaps = normal_values[points[active], candidates[active]] - permuted_values[
        points[active], candidates[active]
    ]
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(active.sum().item()),
        "loss": float(loss.detach().item()),
        "mean_gap": float(gaps.detach().mean().item()),
        "win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
    }


@torch.no_grad()
def _evaluate_inner_validation_target_free(
    *,
    model: nn.Module,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, TrainQueryGroup],
    hard_by_query: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    rgb_context_radius_px: float,
    rgb_step_px: float,
    cache: TensorImageLRUCache,
    state: _DistributedState,
    selector_policy: str,
    selector_point_budget: int,
    selector_grid_rows: int,
    selector_grid_columns: int,
    identity_margin: float,
    hard_repeat_margin: float,
    hard_pose_margin: float,
    minimum_hard_pose_points: int,
    candidate_prior_logit_weight: float,
    max_hard_repeat_edges_per_query: int,
    seed: int,
    permutation_shift: int,
    amp_enabled: bool,
    visual_source_scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Score train-only inner queries after target-free static selection only.

    ``visual_source_scales`` is audit-only appearance ablation metadata.  It
    cannot alter geometry, priors, candidate/view ownership, or target timing.
    The normal training and gate paths leave it unset.
    """

    if str(selector_policy) not in _STATIC_SELECTOR_POLICIES:
        raise ValueError("identity LLR validation selector is not static target-free")
    model.eval()
    totals = torch.zeros((59,), dtype=torch.float64, device=state.device)
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("identity LLR validation query group is unresolved")
        # This is the only selection step.  It reads frozen layout fields, not
        # observed tracks, hard pairs, projections, or residuals.
        selector_input = selector_input_from_target_free_layout(
            layout=layout,
            rows=group.layout_rows,
            rgb_context_radius_px=float(rgb_context_radius_px),
            coordinate_image_size=coordinate_image_size,
        )
        if int(selector_point_budget) > selector_input.point_count:
            raise ValueError("identity LLR validation selector budget exceeds frozen pool")
        positions = select_target_free_spatial_quota(
            selector_input=selector_input,
            quality_scores=target_free_selector_scores(
                selector_input=selector_input, policy=str(selector_policy)
            ),
            point_budget=int(selector_point_budget),
            grid_rows=int(selector_grid_rows),
            grid_columns=int(selector_grid_columns),
            image_size=coordinate_image_size,
        )
        # Only after selection do we materialize target-bearing masks.
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=positions,
            device=state.device,
        )
        hard_batch = (
            None
            if hard_by_query.get(str(query_id)) is None
            else _hard_repeat_batch_from_group(
                hard_targets=hard_by_query[str(query_id)],
                group=group,
                point_positions=positions,
                device=state.device,
                max_edges=int(max_hard_repeat_edges_per_query),
                seed=int(seed),
            )
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(rgb_context_radius_px),
            step_px=float(rgb_step_px),
            cache=cache,
            device=state.device,
        )
        permuted_runtime = permute_runtime_support_appearance(
            batch.runtime, shift=int(permutation_shift)
        )
        permuted_patches = permute_support_patch_appearance(
            runtime=permuted_runtime,
            support_patches=support_patches,
            shift=int(permutation_shift),
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                visual_source_scales=visual_source_scales,
            )
            permuted_prediction = model(
                runtime=permuted_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_patches,
                visual_source_scales=visual_source_scales,
            )
            position_only_prediction = model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                visual_content_scale=0.0,
            )
            rgb_normal_prediction = _component_edge_only_prediction(
                prediction=normal_prediction,
                component="rgb_identity",
            )
            context_normal_prediction = _component_edge_only_prediction(
                prediction=normal_prediction,
                component="context_coherence",
            )
            rgb_component_permutation_usable = _common_component_edge_usable(
                component="rgb_identity",
                predictions=(normal_prediction, permuted_prediction),
            )
            context_component_permutation_usable = _common_component_edge_usable(
                component="context_coherence",
                predictions=(normal_prediction, permuted_prediction),
            )
            rgb_normal_permutation_prediction = _component_edge_only_prediction(
                prediction=normal_prediction,
                component="rgb_identity",
                edge_usable_override=rgb_component_permutation_usable,
            )
            rgb_permuted_prediction = _component_edge_only_prediction(
                prediction=permuted_prediction,
                component="rgb_identity",
                edge_usable_override=rgb_component_permutation_usable,
            )
            context_normal_permutation_prediction = _component_edge_only_prediction(
                prediction=normal_prediction,
                component="context_coherence",
                edge_usable_override=context_component_permutation_usable,
            )
            context_permuted_prediction = _component_edge_only_prediction(
                prediction=permuted_prediction,
                component="context_coherence",
                edge_usable_override=context_component_permutation_usable,
            )
            normal_loss, normal = _registered_identity_metrics(
                runtime=batch.runtime,
                prediction=normal_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                margin=float(identity_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            posterior_loss, posterior = _registered_identity_or_null_posterior_cross_entropy_metrics(
                runtime=batch.runtime,
                prediction=normal_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                target_dustbin=batch.spatial_target_dustbin,
                target_supervised=batch.spatial_target_supervised,
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            permuted_loss, permuted = _registered_identity_metrics(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                margin=float(identity_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            position_only_loss, position_only = _registered_identity_metrics(
                runtime=batch.runtime,
                prediction=position_only_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                margin=float(identity_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            hard_loss, hard = _hard_repeat_metrics(
                runtime=batch.runtime,
                prediction=normal_prediction,
                hard_batch=hard_batch,
                margin=float(hard_repeat_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            hard_position_only_loss, hard_position_only = _hard_repeat_metrics(
                runtime=batch.runtime,
                prediction=position_only_prediction,
                hard_batch=hard_batch,
                margin=float(hard_repeat_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            # Candidate/view availability must be identical across every
            # visual-control branch.  Otherwise a support crop boundary could
            # masquerade as a pose-level visual advantage.
            common_hard_edge_usable = (
                normal_prediction.edge_usable
                & permuted_prediction.edge_usable
                & position_only_prediction.edge_usable
            )
            hard_pose_loss, hard_pose = _hard_pose_group_metrics(
                runtime=batch.runtime,
                prediction=normal_prediction,
                hard_batch=hard_batch,
                margin=float(hard_pose_margin),
                minimum_points=int(minimum_hard_pose_points),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
                edge_usable_override=common_hard_edge_usable,
            )
            hard_pose_permuted_loss, hard_pose_permuted = _hard_pose_group_metrics(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                hard_batch=hard_batch,
                margin=float(hard_pose_margin),
                minimum_points=int(minimum_hard_pose_points),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
                edge_usable_override=common_hard_edge_usable,
            )
            hard_pose_position_only_loss, hard_pose_position_only = _hard_pose_group_metrics(
                runtime=batch.runtime,
                prediction=position_only_prediction,
                hard_batch=hard_batch,
                margin=float(hard_pose_margin),
                minimum_points=int(minimum_hard_pose_points),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
                edge_usable_override=common_hard_edge_usable,
            )
            rgb_posterior_loss, rgb_posterior = (
                _registered_identity_posterior_cross_entropy_metrics(
                    runtime=batch.runtime,
                    prediction=rgb_normal_prediction,
                    observed_candidate_mask=batch.spatial_target_observed,
                    candidate_prior_logit_weight=float(candidate_prior_logit_weight),
                )
            )
            rgb_permutation_loss, rgb_permutation = _registered_permutation_metrics(
                runtime=batch.runtime,
                prediction=rgb_normal_permutation_prediction,
                permuted_runtime=permuted_runtime,
                permuted_prediction=rgb_permuted_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                margin=float(identity_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            context_hard_loss, context_hard = _hard_repeat_metrics(
                runtime=batch.runtime,
                prediction=context_normal_prediction,
                hard_batch=hard_batch,
                margin=float(hard_repeat_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
            context_permutation_loss, context_permutation = _registered_permutation_metrics(
                runtime=batch.runtime,
                prediction=context_normal_permutation_prediction,
                permuted_runtime=permuted_runtime,
                permuted_prediction=context_permuted_prediction,
                observed_candidate_mask=batch.spatial_target_observed,
                margin=float(identity_margin),
                candidate_prior_logit_weight=float(candidate_prior_logit_weight),
            )
        if (
            normal["active"] <= 0.0
            or permuted["active"] <= 0.0
            or position_only["active"] <= 0.0
        ):
            totals[8] += 1.0
            continue
        if float(posterior["candidate_active"]) != float(normal["active"]):
            raise RuntimeError("candidate-or-null posterior changed registered target availability")
        # RGB and full-map context have deliberately different valid domains.
        # Their individual controls already use their own normal/permuted
        # intersections, so requiring their active counts to match would turn
        # an RGB crop boundary into an invalidation of context evidence.
        if float(rgb_posterior["active"]) > float(normal["active"]):
            raise RuntimeError("RGB identity expert exceeds fused registered availability")
        hard_active = float(hard["active"])
        hard_position_active = float(hard_position_only["active"])
        if (hard_active > 0.0) != (hard_position_active > 0.0):
            raise RuntimeError("position-only hard-repeat control changed target availability")
        hard_pose_active = float(hard_pose["active"])
        hard_pose_permuted_active = float(hard_pose_permuted["active"])
        hard_pose_position_active = float(hard_pose_position_only["active"])
        if not (
            hard_pose_active == hard_pose_permuted_active == hard_pose_position_active
            and float(hard_pose["active_points"])
            == float(hard_pose_permuted["active_points"])
            == float(hard_pose_position_only["active_points"])
        ):
            raise RuntimeError("hard-pose visual controls changed target availability")
        totals += torch.tensor(
            [
                float(normal_loss.item()),
                float(normal["mean_gap"]),
                float(normal["win_fraction"]),
                float(permuted_loss.item()),
                float(permuted["mean_gap"]),
                float(permuted["win_fraction"]),
                1.0,
                float(normal["active"]),
                0.0,
                float(hard_loss.item()) if hard_active > 0.0 else 0.0,
                float(hard["mean_gap"]) if hard_active > 0.0 else 0.0,
                float(hard["win_fraction"]) if hard_active > 0.0 else 0.0,
                hard_active,
                1.0 if hard_active > 0.0 else 0.0,
                float(position_only_loss.item()),
                float(position_only["mean_gap"]),
                float(position_only["win_fraction"]),
                float(hard_position_only_loss.item()) if hard_active > 0.0 else 0.0,
                float(hard_position_only["mean_gap"]) if hard_active > 0.0 else 0.0,
                float(hard_position_only["win_fraction"]) if hard_active > 0.0 else 0.0,
                float(hard_pose_loss.item()) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose["mean_gap"]) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose["win_fraction"]) if hard_pose_active > 0.0 else 0.0,
                hard_pose_active,
                float(hard_pose["active_points"]),
                float(hard_pose_permuted_loss.item()) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose_permuted["mean_gap"]) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose_permuted["win_fraction"]) if hard_pose_active > 0.0 else 0.0,
                hard_pose_permuted_active,
                float(hard_pose_permuted["active_points"]),
                float(hard_pose_position_only_loss.item()) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose_position_only["mean_gap"]) if hard_pose_active > 0.0 else 0.0,
                float(hard_pose_position_only["win_fraction"]) if hard_pose_active > 0.0 else 0.0,
                hard_pose_position_active,
                float(hard_pose_position_only["active_points"]),
                1.0 if hard_pose_active > 0.0 else 0.0,
                float(posterior_loss.item()),
                float(posterior["mean_target_probability"]),
                float(posterior["top1_fraction"]),
                float(posterior["candidate_active"]),
                float(posterior["null_active"]),
                float(posterior["null_top1_fraction"] * posterior["null_active"]),
                float(
                    posterior["candidate_top1_fraction"] * posterior["candidate_active"]
                ),
                float(rgb_posterior_loss.item()),
                float(rgb_posterior["mean_target_probability"]),
                float(rgb_posterior["top1_fraction"]),
                float(rgb_posterior["active"]),
                float(rgb_permutation_loss.item()),
                float(rgb_permutation["mean_gap"]),
                float(rgb_permutation["win_fraction"]),
                float(rgb_permutation["active"]),
                float(context_hard_loss.item()) if hard_active > 0.0 else 0.0,
                float(context_hard["mean_gap"]) if hard_active > 0.0 else 0.0,
                float(context_hard["win_fraction"]) if hard_active > 0.0 else 0.0,
                float(context_hard["active"]) if hard_active > 0.0 else 0.0,
                float(context_permutation_loss.item()),
                float(context_permutation["mean_gap"]),
                float(context_permutation["win_fraction"]),
                float(context_permutation["active"]),
            ],
            dtype=torch.float64,
            device=state.device,
        )
    totals = _reduce(state, totals)
    evaluated = float(totals[6].item())
    if evaluated <= 0.0:
        raise RuntimeError("identity LLR target-free inner validation had no registered rows")
    requested = float(len(query_ids))
    hard_evaluated = float(totals[13].item())
    hard_pose_evaluated = float(totals[35].item())
    return {
        "normal_query_grouped_loss": float((totals[0] / evaluated).item()),
        "normal_mean_correct_minus_hardest_wrong": float((totals[1] / evaluated).item()),
        "normal_correct_win_fraction": float((totals[2] / evaluated).item()),
        "permuted_query_grouped_loss": float((totals[3] / evaluated).item()),
        "permuted_mean_correct_minus_hardest_wrong": float((totals[4] / evaluated).item()),
        "permuted_correct_win_fraction": float((totals[5] / evaluated).item()),
        "position_only_query_grouped_loss": float((totals[14] / evaluated).item()),
        "position_only_mean_correct_minus_hardest_wrong": float(
            (totals[15] / evaluated).item()
        ),
        "position_only_correct_win_fraction": float((totals[16] / evaluated).item()),
        "posterior_cross_entropy_loss": float((totals[36] / evaluated).item()),
        "posterior_mean_target_probability": float((totals[37] / evaluated).item()),
        "posterior_top1_fraction": float((totals[38] / evaluated).item()),
        "posterior_candidate_active_per_query": float((totals[39] / evaluated).item()),
        "posterior_candidate_top1_fraction": float(
            (totals[42] / totals[39]).item() if float(totals[39].item()) > 0.0 else 0.0
        ),
        "posterior_null_active_per_query": float((totals[40] / evaluated).item()),
        "posterior_null_top1_fraction": float(
            (totals[41] / totals[40]).item() if float(totals[40].item()) > 0.0 else 0.0
        ),
        "rgb_identity_posterior_cross_entropy_loss": float(
            (totals[43] / evaluated).item()
        ),
        "rgb_identity_posterior_mean_target_probability": float(
            (totals[44] / evaluated).item()
        ),
        "rgb_identity_posterior_top1_fraction": float(
            (totals[45] / totals[46]).item() if float(totals[46].item()) > 0.0 else 0.0
        ),
        "rgb_identity_posterior_active_per_query": float((totals[46] / evaluated).item()),
        "rgb_identity_permutation_query_grouped_loss": float(
            (totals[47] / evaluated).item()
        ),
        "rgb_identity_permutation_mean_gap": float(
            (totals[48] / totals[50]).item() if float(totals[50].item()) > 0.0 else 0.0
        ),
        "rgb_identity_permutation_win_fraction": float(
            (totals[49] / totals[50]).item() if float(totals[50].item()) > 0.0 else 0.0
        ),
        "rgb_identity_permutation_active_per_query": float((totals[50] / evaluated).item()),
        "context_coherence_hard_repeat_query_grouped_loss": float(
            (totals[51] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "context_coherence_hard_repeat_mean_gap": float(
            (totals[52] / totals[54]).item() if float(totals[54].item()) > 0.0 else 0.0
        ),
        "context_coherence_hard_repeat_win_fraction": float(
            (totals[53] / totals[54]).item() if float(totals[54].item()) > 0.0 else 0.0
        ),
        "context_coherence_hard_repeat_active_per_query": float(
            (totals[54] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "context_coherence_permutation_query_grouped_loss": float(
            (totals[55] / evaluated).item()
        ),
        "context_coherence_permutation_mean_gap": float(
            (totals[56] / totals[58]).item() if float(totals[58].item()) > 0.0 else 0.0
        ),
        "context_coherence_permutation_win_fraction": float(
            (totals[57] / totals[58]).item() if float(totals[58].item()) > 0.0 else 0.0
        ),
        "context_coherence_permutation_active_per_query": float(
            (totals[58] / evaluated).item()
        ),
        "query_count": evaluated,
        "eligible_query_fraction": float(evaluated / requested),
        "mean_registered_edges_per_eligible_query": float((totals[7] / evaluated).item()),
        "skipped_query_count": float(totals[8].item()),
        "hard_repeat_query_count": hard_evaluated,
        "hard_repeat_eligible_query_fraction": float(hard_evaluated / requested),
        "hard_repeat_mean_correct_minus_coherent_wrong": float(
            (totals[10] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "hard_repeat_correct_win_fraction": float(
            (totals[11] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "hard_repeat_position_only_query_grouped_loss": float(
            (totals[17] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "hard_repeat_position_only_mean_correct_minus_coherent_wrong": float(
            (totals[18] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "hard_repeat_position_only_correct_win_fraction": float(
            (totals[19] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "mean_hard_repeat_edges_per_eligible_query": float(
            (totals[12] / hard_evaluated).item() if hard_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_query_count": hard_pose_evaluated,
        "hard_pose_group_eligible_query_fraction": float(hard_pose_evaluated / requested),
        "hard_pose_group_query_grouped_loss": float(
            (totals[20] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_mean_correct_minus_coherent_wrong": float(
            (totals[21] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_correct_win_fraction": float(
            (totals[22] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "mean_hard_pose_groups_per_eligible_query": float(
            (totals[23] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "mean_hard_pose_points_per_eligible_query": float(
            (totals[24] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_permuted_query_grouped_loss": float(
            (totals[25] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_permuted_mean_correct_minus_coherent_wrong": float(
            (totals[26] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_permuted_correct_win_fraction": float(
            (totals[27] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_position_only_query_grouped_loss": float(
            (totals[30] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_position_only_mean_correct_minus_coherent_wrong": float(
            (totals[31] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
        "hard_pose_group_position_only_correct_win_fraction": float(
            (totals[32] / hard_pose_evaluated).item() if hard_pose_evaluated > 0.0 else 0.0
        ),
    }


def _is_better(
    *,
    candidate: Mapping[str, float],
    incumbent: Mapping[str, float] | None,
    candidate_passed: bool,
    incumbent_passed: bool,
    training_stage: str = "hard_pose_contrastive",
    minimum_posterior_candidate_top1_fraction: float = 0.0,
) -> bool:
    if incumbent is None:
        return True
    stage = str(training_stage)
    if stage == "identity_l0":
        candidate_ready = int(
            float(candidate.get("posterior_candidate_top1_fraction", 0.0))
            >= float(minimum_posterior_candidate_top1_fraction)
        )
        incumbent_ready = int(
            float(incumbent.get("posterior_candidate_top1_fraction", 0.0))
            >= float(minimum_posterior_candidate_top1_fraction)
        )
        candidate_key = (
            int(bool(candidate_passed)),
            candidate_ready,
            float(candidate.get("rgb_identity_posterior_top1_fraction", 0.0)),
            float(candidate.get("rgb_identity_permutation_mean_gap", 0.0)),
            float(candidate.get("context_coherence_hard_repeat_mean_gap", 0.0)),
            float(candidate.get("context_coherence_permutation_mean_gap", 0.0)),
            float(candidate["hard_repeat_mean_correct_minus_coherent_wrong"]),
            float(candidate["hard_repeat_correct_win_fraction"]),
            float(candidate["hard_repeat_mean_correct_minus_coherent_wrong"])
            - float(candidate["hard_repeat_position_only_mean_correct_minus_coherent_wrong"]),
            float(candidate["normal_mean_correct_minus_hardest_wrong"])
            - float(candidate["permuted_mean_correct_minus_hardest_wrong"]),
            float(candidate["normal_mean_correct_minus_hardest_wrong"])
            - float(candidate["position_only_mean_correct_minus_hardest_wrong"]),
            float(candidate["normal_mean_correct_minus_hardest_wrong"]),
            float(candidate["normal_correct_win_fraction"]),
            float(candidate.get("posterior_candidate_top1_fraction", 0.0)),
        )
        incumbent_key = (
            int(bool(incumbent_passed)),
            incumbent_ready,
            float(incumbent.get("rgb_identity_posterior_top1_fraction", 0.0)),
            float(incumbent.get("rgb_identity_permutation_mean_gap", 0.0)),
            float(incumbent.get("context_coherence_hard_repeat_mean_gap", 0.0)),
            float(incumbent.get("context_coherence_permutation_mean_gap", 0.0)),
            float(incumbent["hard_repeat_mean_correct_minus_coherent_wrong"]),
            float(incumbent["hard_repeat_correct_win_fraction"]),
            float(incumbent["hard_repeat_mean_correct_minus_coherent_wrong"])
            - float(incumbent["hard_repeat_position_only_mean_correct_minus_coherent_wrong"]),
            float(incumbent["normal_mean_correct_minus_hardest_wrong"])
            - float(incumbent["permuted_mean_correct_minus_hardest_wrong"]),
            float(incumbent["normal_mean_correct_minus_hardest_wrong"])
            - float(incumbent["position_only_mean_correct_minus_hardest_wrong"]),
            float(incumbent["normal_mean_correct_minus_hardest_wrong"]),
            float(incumbent["normal_correct_win_fraction"]),
            float(incumbent.get("posterior_candidate_top1_fraction", 0.0)),
        )
        return candidate_key > incumbent_key
    if stage != "hard_pose_contrastive":
        raise ValueError("identity LLR checkpoint stage is invalid")
    candidate_key = (
        int(bool(candidate_passed)),
        float(candidate["hard_pose_group_mean_correct_minus_coherent_wrong"]),
        float(candidate["hard_pose_group_correct_win_fraction"]),
        float(candidate["hard_pose_group_mean_correct_minus_coherent_wrong"])
        - float(candidate["hard_pose_group_permuted_mean_correct_minus_coherent_wrong"]),
        float(candidate["hard_pose_group_mean_correct_minus_coherent_wrong"])
        - float(candidate["hard_pose_group_position_only_mean_correct_minus_coherent_wrong"]),
        float(candidate["normal_mean_correct_minus_hardest_wrong"])
        - float(candidate["permuted_mean_correct_minus_hardest_wrong"]),
        float(candidate["normal_mean_correct_minus_hardest_wrong"])
        - float(candidate["position_only_mean_correct_minus_hardest_wrong"]),
        float(candidate["normal_mean_correct_minus_hardest_wrong"]),
        float(candidate["normal_correct_win_fraction"]),
    )
    incumbent_key = (
        int(bool(incumbent_passed)),
        float(incumbent["hard_pose_group_mean_correct_minus_coherent_wrong"]),
        float(incumbent["hard_pose_group_correct_win_fraction"]),
        float(incumbent["hard_pose_group_mean_correct_minus_coherent_wrong"])
        - float(incumbent["hard_pose_group_permuted_mean_correct_minus_coherent_wrong"]),
        float(incumbent["hard_pose_group_mean_correct_minus_coherent_wrong"])
        - float(incumbent["hard_pose_group_position_only_mean_correct_minus_coherent_wrong"]),
        float(incumbent["normal_mean_correct_minus_hardest_wrong"])
        - float(incumbent["permuted_mean_correct_minus_hardest_wrong"]),
        float(incumbent["normal_mean_correct_minus_hardest_wrong"])
        - float(incumbent["position_only_mean_correct_minus_hardest_wrong"]),
        float(incumbent["normal_mean_correct_minus_hardest_wrong"]),
        float(incumbent["normal_correct_win_fraction"]),
    )
    return candidate_key > incumbent_key


def _checkpoint_gate(
    *, metrics: Mapping[str, float], args: argparse.Namespace
) -> dict[str, object]:
    decision = training_gate_decision(
        metrics,
        minimum_win_fraction=float(args.minimum_win_fraction),
        minimum_normal_gap=float(args.minimum_normal_gap),
        minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
    )
    coverage = float(metrics["eligible_query_fraction"])
    hard_pose_coverage = float(metrics["hard_pose_group_eligible_query_fraction"])
    hard_pose_win = float(metrics["hard_pose_group_correct_win_fraction"])
    hard_pose_gap = float(metrics["hard_pose_group_mean_correct_minus_coherent_wrong"])
    hard_pose_permuted_gap = float(
        metrics["hard_pose_group_permuted_mean_correct_minus_coherent_wrong"]
    )
    hard_pose_position_only_gap = float(
        metrics["hard_pose_group_position_only_mean_correct_minus_coherent_wrong"]
    )
    position_gap = float(metrics["normal_mean_correct_minus_hardest_wrong"]) - float(
        metrics["position_only_mean_correct_minus_hardest_wrong"]
    )
    hard_pose_visual_gap = hard_pose_gap - hard_pose_permuted_gap
    hard_pose_position_gap = hard_pose_gap - hard_pose_position_only_gap
    stage = str(args.training_stage)
    if stage == "identity_l0":
        hard_repeat_coverage = float(metrics["hard_repeat_eligible_query_fraction"])
        hard_repeat_win = float(metrics["hard_repeat_correct_win_fraction"])
        hard_repeat_gap = float(metrics["hard_repeat_mean_correct_minus_coherent_wrong"])
        hard_repeat_position_gap = hard_repeat_gap - float(
            metrics["hard_repeat_position_only_mean_correct_minus_coherent_wrong"]
        )
        candidate_top1 = float(metrics.get("posterior_candidate_top1_fraction", 0.0))
        rgb_candidate_top1 = float(metrics.get("rgb_identity_posterior_top1_fraction", 0.0))
        rgb_permutation_gap = float(metrics.get("rgb_identity_permutation_mean_gap", 0.0))
        context_hard_repeat_gap = float(
            metrics.get("context_coherence_hard_repeat_mean_gap", 0.0)
        )
        context_permutation_gap = float(
            metrics.get("context_coherence_permutation_mean_gap", 0.0)
        )
        decision["training_stage"] = stage
        decision["eligible_query_fraction"] = coverage
        decision["minimum_inner_eligible_query_fraction"] = float(
            args.minimum_inner_eligible_query_fraction
        )
        decision["hard_repeat_eligible_query_fraction"] = hard_repeat_coverage
        decision["minimum_inner_hard_repeat_eligible_query_fraction"] = float(
            args.minimum_inner_hard_repeat_eligible_query_fraction
        )
        decision["hard_repeat_correct_win_fraction"] = hard_repeat_win
        decision["minimum_hard_repeat_win_fraction"] = float(args.minimum_hard_repeat_win_fraction)
        decision["hard_repeat_mean_correct_minus_coherent_wrong"] = hard_repeat_gap
        decision["minimum_hard_repeat_gap"] = float(args.minimum_hard_repeat_gap)
        decision["posterior_candidate_top1_fraction"] = candidate_top1
        decision["minimum_posterior_candidate_top1_fraction"] = float(
            args.minimum_posterior_candidate_top1_fraction
        )
        decision["rgb_identity_posterior_top1_fraction"] = rgb_candidate_top1
        decision["minimum_rgb_identity_posterior_top1_fraction"] = float(
            args.minimum_rgb_identity_posterior_top1_fraction
        )
        decision["rgb_identity_permutation_mean_gap"] = rgb_permutation_gap
        decision["minimum_rgb_identity_permutation_gap"] = float(
            args.minimum_rgb_identity_permutation_gap
        )
        decision["context_coherence_hard_repeat_mean_gap"] = context_hard_repeat_gap
        decision["minimum_context_coherence_hard_repeat_gap"] = float(
            args.minimum_context_coherence_hard_repeat_gap
        )
        decision["context_coherence_permutation_mean_gap"] = context_permutation_gap
        decision["minimum_context_coherence_permutation_gap"] = float(
            args.minimum_context_coherence_permutation_gap
        )
        decision["normal_minus_position_only_gap"] = position_gap
        decision["hard_repeat_minus_position_only_gap"] = hard_repeat_position_gap
        decision["minimum_position_visual_gap"] = float(args.minimum_position_visual_gap)
        decision["passed"] = bool(
            decision["passed"]
            and coverage >= float(args.minimum_inner_eligible_query_fraction)
            and hard_repeat_coverage
            >= float(args.minimum_inner_hard_repeat_eligible_query_fraction)
            and hard_repeat_win >= float(args.minimum_hard_repeat_win_fraction)
            and hard_repeat_gap >= float(args.minimum_hard_repeat_gap)
            and candidate_top1 >= float(args.minimum_posterior_candidate_top1_fraction)
            and rgb_candidate_top1
            >= float(args.minimum_rgb_identity_posterior_top1_fraction)
            and rgb_permutation_gap >= float(args.minimum_rgb_identity_permutation_gap)
            and context_hard_repeat_gap
            >= float(args.minimum_context_coherence_hard_repeat_gap)
            and context_permutation_gap
            >= float(args.minimum_context_coherence_permutation_gap)
            and position_gap >= float(args.minimum_position_visual_gap)
            and hard_repeat_position_gap >= float(args.minimum_position_visual_gap)
        )
        return decision
    if stage != "hard_pose_contrastive":
        raise ValueError("identity LLR checkpoint stage is invalid")
    decision["training_stage"] = stage
    decision["eligible_query_fraction"] = coverage
    decision["minimum_inner_eligible_query_fraction"] = float(
        args.minimum_inner_eligible_query_fraction
    )
    decision["hard_repeat_point_diagnostic"] = {
        "eligible_query_fraction": float(metrics["hard_repeat_eligible_query_fraction"]),
        "correct_win_fraction": float(metrics["hard_repeat_correct_win_fraction"]),
        "mean_correct_minus_coherent_wrong": float(
            metrics["hard_repeat_mean_correct_minus_coherent_wrong"]
        ),
    }
    decision["hard_pose_group_eligible_query_fraction"] = hard_pose_coverage
    decision["minimum_inner_hard_pose_eligible_query_fraction"] = float(
        args.minimum_inner_hard_pose_eligible_query_fraction
    )
    decision["hard_pose_group_correct_win_fraction"] = hard_pose_win
    decision["minimum_hard_pose_win_fraction"] = float(args.minimum_hard_pose_win_fraction)
    decision["hard_pose_group_mean_correct_minus_coherent_wrong"] = hard_pose_gap
    decision["minimum_hard_pose_gap"] = float(args.minimum_hard_pose_gap)
    decision["hard_pose_group_minus_permuted_gap"] = hard_pose_visual_gap
    decision["minimum_hard_pose_visual_gap_delta"] = float(
        args.minimum_hard_pose_visual_gap_delta
    )
    decision["normal_minus_position_only_gap"] = position_gap
    decision["hard_pose_group_minus_position_only_gap"] = hard_pose_position_gap
    decision["minimum_position_visual_gap"] = float(args.minimum_position_visual_gap)
    decision["passed"] = bool(
        decision["passed"]
        and coverage >= float(args.minimum_inner_eligible_query_fraction)
        and hard_pose_coverage >= float(args.minimum_inner_hard_pose_eligible_query_fraction)
        and hard_pose_win >= float(args.minimum_hard_pose_win_fraction)
        and hard_pose_gap >= float(args.minimum_hard_pose_gap)
        and hard_pose_visual_gap >= float(args.minimum_hard_pose_visual_gap_delta)
        and position_gap >= float(args.minimum_position_visual_gap)
        and hard_pose_position_gap >= float(args.minimum_position_visual_gap)
    )
    return decision


def _prepare_output_directory(
    *, output_dir: Path, force: bool, state: _DistributedState
) -> None:
    """Create a run directory exactly once under DDP.

    The output-exists decision belongs to rank zero.  Checking it independently
    on every rank races with rank zero's ``mkdir``: a nonzero rank can observe
    the newly created directory and incorrectly reject the same run.  Broadcast
    the owner decision before any rank writes or enters the startup barrier.
    """

    owner_refuses = bool(
        state.rank == 0 and output_dir.exists() and not bool(force)
    )
    refusal = torch.tensor(
        [int(owner_refuses)], dtype=torch.int64, device=state.device
    )
    if state.enabled:
        distributed.broadcast(refusal, src=0)
    if bool(int(refusal.item())):
        raise FileExistsError(f"refusing to overwrite identity LLR output: {output_dir}")
    if state.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomically(
            output_dir / "progress.json",
            {
                "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                "status": "running",
                "completed_epochs": 0,
                "interruption_safe_history": "history.partial.json",
            },
        )
    if state.enabled:
        distributed.barrier()


def train_candidate_pose_rgb_spatial_identity_llr(args: argparse.Namespace) -> dict[str, object]:
    """Fit one real-image identity-LLR experiment under a strict inner gate."""

    windows = _validate_args(args)
    feature_training_mode = _resolve_feature_training_mode(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        _prepare_output_directory(
            output_dir=output_dir, force=bool(args.force), state=state
        )
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover
            pass

        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        hard_path = Path(args.hard_repeat_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        layout_sha = file_sha256_short(layout_path)
        targets_sha = file_sha256_short(targets_path)
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha
        )
        hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(hard_path)
        _validate_identity_target_contract(targets=targets, hard_targets=hard_targets)
        groups = build_train_query_groups(layout=layout, targets=targets)
        hard_by_query = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=targets_sha,
        )
        train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=tuple(sorted(groups)),
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        if state.enabled and len(train_ids) % state.world_size != 0:
            raise ValueError("DDP identity LLR training requires evenly partitioned train query IDs")

        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("identity LLR requires common frozen context image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        identity_source_lineage = {
            "radio_final_context_cache_sha256": file_sha256_short(
                Path(args.radio_final_context_cache)
            ),
            "radio_intermediate_context_cache_sha256": file_sha256_short(
                Path(args.radio_intermediate_context_cache)
            ),
            "alike_spatial_context_cache_sha256": file_sha256_short(
                Path(args.alike_spatial_context_cache)
            ),
            "source_image_manifest_sha256": str(
                sources[0].metadata.get("source_image_manifest_sha256", "")
            ),
            "rgb_coordinate_bridge": rgb_bridge,
        }
        model = CandidatePoseRGBSpatialIdentityLLR(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            rgb_context_radius_px=float(args.rgb_context_radius_px),
            rgb_step_px=float(args.rgb_step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=True,
            context_windows=windows,
        )
        if str(args.identity_observation_pretrain_checkpoint).strip():
            initialization = _load_identity_observation_pretrain(
                model=model,
                path=Path(args.identity_observation_pretrain_checkpoint),
                candidate_count=int(layout.candidate_count),
                context_windows=windows,
                source_lineage=identity_source_lineage,
                args=args,
            )
        else:
            initialization = {
                "kind": "texture_only_rgb_observation_pretrain",
                **_load_texture_observation_pretrain(
                    model=model,
                    path=Path(args.texture_observation_pretrain_checkpoint),
                    texture_feature_dim=int(args.texture_feature_dim),
                    hidden_dim=int(args.hidden_dim),
                ),
            }
        if str(args.training_stage) == "identity_l0" or bool(
            args.reset_current_p1_scalar_head_finals
        ):
            initialization = {
                **initialization,
                "neutral_current_p1_scalar_head_final_layers": list(
                    _reset_identity_l0_scalar_head_finals(model)
                ),
                "scalar_likelihood_transfer": (
                    "final_layers_reset_current_p1_pose_group_or_identity_l0_v2"
                ),
            }
        if bool(args.reset_edge_head_final_for_prior_residual):
            _reset_identity_llr_final_edge_head(model)
            initialization = {
                **initialization,
                "prior_residual_final_edge_head_reset": True,
            }
        feature_training = _configure_identity_llr_feature_training(
            model=model,
            mode=feature_training_mode,
        )
        model = model.to(state.device)
        model_for_train: nn.Module
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        optimizer = torch.optim.AdamW(
            _identity_llr_optimizer_parameter_groups(model=model, args=args),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled, init_scale=float(args.amp_init_scale)
        )
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        image_cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        # Audit the broad initializer on the actual frozen P1 candidate set
        # before any P1 target can update it.  This separates transfer from
        # sparse hard-repeat fine-tuning and keeps the evidence chain
        # inspectable when a later checkpoint improves or regresses.
        initialization_inner = _evaluate_inner_validation_target_free(
            model=model_for_train,
            layout=layout,
            groups=groups,
            hard_by_query=hard_by_query,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            rgb_context_radius_px=float(args.rgb_context_radius_px),
            rgb_step_px=float(args.rgb_step_px),
            cache=image_cache,
            state=state,
            selector_policy=str(args.validation_selector_policy),
            selector_point_budget=int(args.validation_selector_point_budget),
            selector_grid_rows=int(args.validation_selector_grid_rows),
            selector_grid_columns=int(args.validation_selector_grid_columns),
            identity_margin=float(args.identity_margin),
            hard_repeat_margin=float(args.hard_repeat_margin),
            hard_pose_margin=float(args.hard_pose_margin),
            minimum_hard_pose_points=int(args.minimum_hard_pose_points),
            candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
            max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
            seed=int(args.seed),
            permutation_shift=int(args.permutation_control_shift),
            amp_enabled=amp_enabled,
        )
        initialization_inner_gate = _checkpoint_gate(metrics=initialization_inner, args=args)
        if state.rank == 0:
            initialization = {
                **initialization,
                "p1_zero_shot_inner_validation": {
                    "metrics": initialization_inner,
                    "gate": initialization_inner_gate,
                    "selection_before_target_join": True,
                    "model_weights_updated": False,
                },
            }
            _write_json_atomically(
                output_dir / "initialization_inner_validation.json",
                initialization["p1_zero_shot_inner_validation"],
            )
            _write_json_atomically(
                output_dir / "progress.json",
                {
                    "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                    "status": "running",
                    "completed_epochs": 0,
                    "initialization_inner_gate_passed": bool(initialization_inner_gate["passed"]),
                    "initialization_inner_validation": initialization_inner,
                },
            )
        if state.enabled:
            distributed.barrier()
        if bool(args.initialization_only):
            summary: dict[str, object] = {}
            if state.rank == 0:
                summary = {
                    "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                    "checkpoint": None,
                    "initialization": initialization,
                    "checkpoint_selection": {
                        "selected_epoch": 0,
                        "selected_checkpoint_kind": "zero_shot_initializer",
                        "inner_validation": initialization_inner,
                        "gate": initialization_inner_gate,
                    },
                    "history": [],
                    "rgb_cache_rank0": image_cache.summary(),
                    "protocol": {
                        "initialization_only": True,
                        "model_weights_updated": False,
                        "no_render": True,
                        "no_image_retrieval_or_submap": True,
                        "runtime_layout_remains_target_free": True,
                        "pose_or_ground_truth_not_available_to_runtime_encoder": True,
                        "train_only_target_artifact": True,
                        "inner_validation_selector_target_free": True,
                        "inner_validation_observed_target_sampler_used": False,
                        "heldout_validation_or_test_not_run": True,
                    },
                }
                _write_json_atomically(output_dir / "history.json", [])
                _write_json_atomically(output_dir / "summary.json", summary)
                _write_json_atomically(
                    output_dir / "progress.json",
                    {
                        "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                        "status": "complete",
                        "completed_epochs": 0,
                        "selected_epoch": 0,
                        "selected_checkpoint_kind": "zero_shot_initializer",
                        "inner_gate_passed": bool(initialization_inner_gate["passed"]),
                        "initialization_only": True,
                    },
                )
            if state.enabled:
                distributed.barrier()
            return summary
        # The broad initializer is a valid target-free candidate in the same
        # frozen P1 audit.  Sparse hard-repeat updates must beat it under the
        # exact same gate; otherwise preserving the initializer is safer than
        # exporting a degraded fine-tuned checkpoint.
        best_state: dict[str, torch.Tensor] | None = None
        best_metrics: dict[str, float] | None = None
        best_epoch = -1
        best_checkpoint_kind = ""
        if state.rank == 0:
            core = (
                model_for_train.module
                if isinstance(model_for_train, DistributedDataParallel)
                else model_for_train
            )
            best_state = {
                name: value.detach().cpu().clone() for name, value in core.state_dict().items()
            }
            best_metrics = dict(initialization_inner)
            best_epoch = 0
            best_checkpoint_kind = "zero_shot_initializer"
        history: list[dict[str, object]] = []
        local_train_ids = tuple(train_ids[state.rank :: state.world_size])
        for epoch in range(int(args.epochs)):
            epoch_start = time.time()
            _set_identity_llr_train_mode(
                model=model_for_train,
                feature_training_mode=feature_training_mode,
            )
            totals = torch.zeros((55,), dtype=torch.float64, device=state.device)
            for query_id in local_train_ids:
                group = groups[str(query_id)]
                hard_targets_for_query = hard_by_query.get(str(query_id))
                required_source_ids = (
                    ()
                    if hard_targets_for_query is None
                    else tuple(int(value) for value in hard_targets_for_query.source_point_ids.tolist())
                )
                if str(args.training_stage) == "identity_l0":
                    positions = _select_identity_l0_group_points(
                        group=group,
                        max_points=int(args.max_points_per_query),
                        seed=int(args.seed) + int(epoch),
                        required_source_point_ids=required_source_ids,
                    )
                else:
                    positions = _select_group_points(
                        group=group,
                        max_points=int(args.max_points_per_query),
                        seed=int(args.seed) + int(epoch),
                        required_source_point_ids=(
                            None if hard_targets_for_query is None else required_source_ids
                        ),
                    )
                batch = _query_batch_from_group(
                    group=group,
                    complete_runtime=complete_runtime,
                    point_positions=positions,
                    device=state.device,
                )
                train_runtime = _jitter_runtime_query_anchors(
                    runtime=batch.runtime,
                    radius_px=float(args.train_query_anchor_jitter_radius_px),
                    coordinate_image_size=coordinate_image_size,
                    interior_margin_px=float(args.rgb_context_radius_px),
                    seed=_query_anchor_jitter_seed(
                        seed=int(args.seed), epoch=int(epoch), query_id=str(query_id)
                    ),
                )
                hard_batch = (
                    None
                    if hard_targets_for_query is None
                    else _hard_repeat_batch_from_group(
                        hard_targets=hard_targets_for_query,
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_hard_repeat_edges_per_query),
                        seed=int(args.seed) + int(epoch),
                    )
                )
                query_patches, support_patches = _crop_runtime_rgb_patches(
                    runtime=train_runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=float(args.rgb_context_radius_px),
                    step_px=float(args.rgb_step_px),
                    cache=image_cache,
                    device=state.device,
                )
                permuted_runtime = permute_runtime_support_appearance(
                    train_runtime, shift=int(args.permutation_control_shift)
                )
                permuted_patches = permute_support_patch_appearance(
                    runtime=permuted_runtime,
                    support_patches=support_patches,
                    shift=int(args.permutation_control_shift),
                )
                optimizer.zero_grad(set_to_none=True)
                scalar_before = {
                    name: parameter.detach().clone()
                    for name, parameter in _identity_llr_scalar_head_parameters(
                        model_for_train
                    ).items()
                }
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(
                        runtime=train_runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                    )
                    rgb_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="rgb_identity",
                    )
                    context_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="context_coherence",
                    )
                    identity_loss, identity = _registered_identity_metrics(
                        runtime=train_runtime,
                        prediction=prediction,
                        observed_candidate_mask=batch.spatial_target_observed,
                        margin=float(args.identity_margin),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                    )
                    posterior_loss, posterior = _registered_identity_or_null_posterior_cross_entropy_metrics(
                        runtime=train_runtime,
                        prediction=prediction,
                        observed_candidate_mask=batch.spatial_target_observed,
                        target_dustbin=batch.spatial_target_dustbin,
                        target_supervised=batch.spatial_target_supervised,
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                        candidate_loss_weight=float(args.posterior_candidate_loss_weight),
                        null_loss_weight=float(args.posterior_null_loss_weight),
                    )
                    hard_loss, hard = _hard_repeat_metrics(
                        runtime=train_runtime,
                        prediction=prediction,
                        hard_batch=hard_batch,
                        margin=float(args.hard_repeat_margin),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                    )
                    permuted_prediction = model_for_train(
                        runtime=permuted_runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=permuted_patches,
                    )
                    rgb_component_permutation_usable = _common_component_edge_usable(
                        component="rgb_identity",
                        predictions=(prediction, permuted_prediction),
                    )
                    context_component_permutation_usable = _common_component_edge_usable(
                        component="context_coherence",
                        predictions=(prediction, permuted_prediction),
                    )
                    rgb_permutation_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="rgb_identity",
                        edge_usable_override=rgb_component_permutation_usable,
                    )
                    rgb_permuted_prediction = _component_edge_only_prediction(
                        prediction=permuted_prediction,
                        component="rgb_identity",
                        edge_usable_override=rgb_component_permutation_usable,
                    )
                    context_permutation_prediction = _component_edge_only_prediction(
                        prediction=prediction,
                        component="context_coherence",
                        edge_usable_override=context_component_permutation_usable,
                    )
                    context_permuted_prediction = _component_edge_only_prediction(
                        prediction=permuted_prediction,
                        component="context_coherence",
                        edge_usable_override=context_component_permutation_usable,
                    )
                    rgb_posterior_loss, rgb_posterior = (
                        _registered_identity_posterior_cross_entropy_metrics(
                            runtime=train_runtime,
                            prediction=rgb_prediction,
                            observed_candidate_mask=batch.spatial_target_observed,
                            candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                        )
                    )
                    rgb_permutation_loss, rgb_permutation = _registered_permutation_metrics(
                        runtime=train_runtime,
                        prediction=rgb_permutation_prediction,
                        permuted_runtime=permuted_runtime,
                        permuted_prediction=rgb_permuted_prediction,
                        observed_candidate_mask=batch.spatial_target_observed,
                        margin=float(args.permutation_margin),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                    )
                    context_hard_loss, context_hard = _hard_repeat_metrics(
                        runtime=train_runtime,
                        prediction=context_prediction,
                        hard_batch=hard_batch,
                        margin=float(args.hard_repeat_margin),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                    )
                    context_permutation_loss, context_permutation = (
                        _registered_permutation_metrics(
                            runtime=train_runtime,
                            prediction=context_permutation_prediction,
                            permuted_runtime=permuted_runtime,
                            permuted_prediction=context_permuted_prediction,
                            observed_candidate_mask=batch.spatial_target_observed,
                            margin=float(args.permutation_margin),
                            candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                        )
                    )
                    permutation_loss, permutation = _registered_permutation_metrics(
                        runtime=train_runtime,
                        prediction=prediction,
                        permuted_runtime=permuted_runtime,
                        permuted_prediction=permuted_prediction,
                        observed_candidate_mask=batch.spatial_target_observed,
                        margin=float(args.permutation_margin),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                    )
                    common_hard_edge_usable = (
                        prediction.edge_usable & permuted_prediction.edge_usable
                    )
                    hard_pose_loss, hard_pose = _hard_pose_group_metrics(
                        runtime=train_runtime,
                        prediction=prediction,
                        hard_batch=hard_batch,
                        margin=float(args.hard_pose_margin),
                        minimum_points=int(args.minimum_hard_pose_points),
                        candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                        edge_usable_override=common_hard_edge_usable,
                    )
                    hard_pose_permutation_loss, hard_pose_permutation = (
                        _hard_pose_group_permutation_metrics(
                            runtime=train_runtime,
                            prediction=prediction,
                            permuted_runtime=permuted_runtime,
                            permuted_prediction=permuted_prediction,
                            hard_batch=hard_batch,
                            margin=float(args.permutation_margin),
                            minimum_points=int(args.minimum_hard_pose_points),
                            candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                            common_edge_usable=common_hard_edge_usable,
                        )
                    )
                    total_loss = (
                        float(args.identity_loss_weight) * identity_loss
                        + float(args.posterior_cross_entropy_loss_weight) * posterior_loss
                        + float(args.hard_repeat_loss_weight) * hard_loss
                        + float(args.hard_pose_loss_weight) * hard_pose_loss
                        + float(args.permutation_loss_weight) * permutation_loss
                        + float(args.hard_pose_permutation_loss_weight)
                        * hard_pose_permutation_loss
                        + float(args.rgb_identity_posterior_loss_weight) * rgb_posterior_loss
                        + float(args.rgb_identity_permutation_loss_weight)
                        * rgb_permutation_loss
                        + float(args.context_coherence_hard_repeat_loss_weight)
                        * context_hard_loss
                        + float(args.context_coherence_permutation_loss_weight)
                        * context_permutation_loss
                    )
                    # A pose-only ablation can deliberately set every direct
                    # null/posterior term to zero. Keep all scalar heads in
                    # the ordinary DDP graph with an exactly zero objective
                    # contribution, rather than enabling find-unused on a
                    # module that is forwarded twice (normal/permuted) per
                    # iteration. The latter makes older PyTorch DDP mark the
                    # same parameter ready twice.
                    scalar_graph_anchor = (
                        prediction.edge_log_likelihood_ratios.sum()
                        + prediction.support_view_logits.sum()
                        + prediction.point_null_log_likelihood_ratios.sum()
                    ) * 0.0
                    total_loss = total_loss + scalar_graph_anchor
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                scalar_gradient_l2, scalar_gradients_finite = (
                    _identity_llr_scalar_gradient_diagnostics(model_for_train)
                )
                if (
                    feature_training_mode == "frozen_scalar_heads"
                    and not scalar_gradients_finite
                ):
                    raise FloatingPointError(
                        "identity L0 scalar-head gradients became non-finite after AMP unscale"
                    )
                torch.nn.utils.clip_grad_norm_(model_for_train.parameters(), float(args.gradient_clip_norm))
                scaler_scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scalar_update_squared = 0.0
                for name, parameter in _identity_llr_scalar_head_parameters(
                    model_for_train
                ).items():
                    scalar_update_squared += float(
                        (parameter.detach().float() - scalar_before[name].float()).square().sum().item()
                    )
                scalar_update_l2 = math.sqrt(scalar_update_squared)
                rgb_identity_update_l2 = _identity_llr_expert_update_l2(
                    model=model_for_train,
                    before=scalar_before,
                    expert="rgb_identity",
                )
                context_coherence_update_l2 = _identity_llr_expert_update_l2(
                    model=model_for_train,
                    before=scalar_before,
                    expert="context_coherence",
                )
                scaler_scale_after = float(scaler.get_scale())
                totals += torch.tensor(
                    [
                        float(total_loss.detach().item()),
                        float(identity["loss"]),
                        float(identity["mean_gap"]),
                        float(identity["win_fraction"]),
                        float(identity["active"]),
                        float(hard["loss"]),
                        float(hard["mean_gap"]),
                        float(hard["win_fraction"]),
                        float(hard["active"]),
                        float(permutation["loss"]),
                        float(permutation["mean_gap"]),
                        float(permutation["win_fraction"]),
                        float(permutation["active"]),
                        float(hard_pose["loss"]),
                        float(hard_pose["mean_gap"]),
                        float(hard_pose["win_fraction"]),
                        float(hard_pose["active"]),
                        float(hard_pose["active_points"]),
                        float(hard_pose_permutation["loss"]),
                        float(hard_pose_permutation["mean_gap"]),
                        float(hard_pose_permutation["win_fraction"]),
                        float(hard_pose_permutation["active"]),
                        float(posterior["loss"]),
                        float(posterior["mean_target_probability"]),
                        float(posterior["top1_fraction"]),
                        float(posterior["active"]),
                        1.0,
                        float(posterior["null_active"]),
                        float(posterior["null_top1_fraction"] * posterior["null_active"]),
                        float(posterior["candidate_active"]),
                        float(scalar_gradient_l2),
                        float(scalar_update_l2),
                        float(not scalar_gradients_finite),
                        float(scaler_scale_after < scaler_scale_before),
                        float(
                            posterior["candidate_top1_fraction"]
                            * posterior["candidate_active"]
                        ),
                        float(posterior["candidate_loss"] * posterior["candidate_active"]),
                        float(posterior["null_loss"] * posterior["null_active"]),
                        float(posterior["optimization_loss"]),
                        float(rgb_posterior["loss"]),
                        float(rgb_posterior["top1_fraction"]),
                        float(rgb_posterior["active"]),
                        float(rgb_permutation["loss"]),
                        float(rgb_permutation["mean_gap"]),
                        float(rgb_permutation["win_fraction"]),
                        float(rgb_permutation["active"]),
                        float(context_hard["loss"]),
                        float(context_hard["mean_gap"]),
                        float(context_hard["win_fraction"]),
                        float(context_hard["active"]),
                        float(context_permutation["loss"]),
                        float(context_permutation["mean_gap"]),
                        float(context_permutation["win_fraction"]),
                        float(context_permutation["active"]),
                        float(rgb_identity_update_l2),
                        float(context_coherence_update_l2),
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce(state, totals)
            global_steps = float(totals[26].item())
            if global_steps <= 0.0:
                raise RuntimeError("identity LLR epoch had no train query step")
            if feature_training_mode == "frozen_scalar_heads" and float(totals[31].item()) <= 0.0:
                raise RuntimeError(
                    "identity L0 epoch made no scalar-head parameter update; refusing invalid evidence run"
                )
            if feature_training_mode == "frozen_scalar_heads" and (
                float(totals[53].item()) <= 0.0
                or float(totals[54].item()) <= 0.0
            ):
                raise RuntimeError(
                    "identity L0 epoch left an independently supervised RGB or context expert unchanged"
                )
            inner = _evaluate_inner_validation_target_free(
                model=model_for_train,
                layout=layout,
                groups=groups,
                hard_by_query=hard_by_query,
                complete_runtime=complete_runtime,
                query_ids=inner_validation_ids,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                rgb_context_radius_px=float(args.rgb_context_radius_px),
                rgb_step_px=float(args.rgb_step_px),
                cache=image_cache,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                identity_margin=float(args.identity_margin),
                hard_repeat_margin=float(args.hard_repeat_margin),
                hard_pose_margin=float(args.hard_pose_margin),
                minimum_hard_pose_points=int(args.minimum_hard_pose_points),
                candidate_prior_logit_weight=float(args.candidate_prior_logit_weight),
                max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
                seed=int(args.seed),
                permutation_shift=int(args.permutation_control_shift),
                amp_enabled=amp_enabled,
            )
            if state.rank == 0:
                gate = _checkpoint_gate(metrics=inner, args=args)
                incumbent_gate = None if best_metrics is None else _checkpoint_gate(
                    metrics=best_metrics, args=args
                )
                epoch_metrics: dict[str, object] = {
                    "epoch": int(epoch + 1),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_identity_loss": float((totals[1] / global_steps).item()),
                    "train_identity_mean_gap": float((totals[2] / global_steps).item()),
                    "train_identity_win_fraction": float((totals[3] / global_steps).item()),
                    "train_identity_active_edges_per_step": float((totals[4] / global_steps).item()),
                    "train_hard_repeat_loss": float((totals[5] / global_steps).item()),
                    "train_hard_repeat_mean_gap": float((totals[6] / global_steps).item()),
                    "train_hard_repeat_win_fraction": float((totals[7] / global_steps).item()),
                    "train_hard_repeat_active_edges_per_step": float((totals[8] / global_steps).item()),
                    "train_permutation_loss": float((totals[9] / global_steps).item()),
                    "train_permutation_mean_gap": float((totals[10] / global_steps).item()),
                    "train_permutation_win_fraction": float((totals[11] / global_steps).item()),
                    "train_permutation_active_edges_per_step": float((totals[12] / global_steps).item()),
                    "train_hard_pose_group_loss": float((totals[13] / global_steps).item()),
                    "train_hard_pose_group_mean_gap": float((totals[14] / global_steps).item()),
                    "train_hard_pose_group_win_fraction": float((totals[15] / global_steps).item()),
                    "train_hard_pose_groups_per_step": float((totals[16] / global_steps).item()),
                    "train_hard_pose_points_per_step": float((totals[17] / global_steps).item()),
                    "train_hard_pose_permutation_loss": float((totals[18] / global_steps).item()),
                    "train_hard_pose_permutation_mean_gap": float((totals[19] / global_steps).item()),
                    "train_hard_pose_permutation_win_fraction": float((totals[20] / global_steps).item()),
                    "train_hard_pose_permutation_groups_per_step": float((totals[21] / global_steps).item()),
                    "train_posterior_cross_entropy_loss": float((totals[22] / global_steps).item()),
                    "train_posterior_optimization_loss": float(
                        (totals[37] / global_steps).item()
                    ),
                    "train_posterior_mean_target_probability": float(
                        (totals[23] / global_steps).item()
                    ),
                    "train_posterior_top1_fraction": float((totals[24] / global_steps).item()),
                    "train_posterior_active_edges_per_step": float((totals[25] / global_steps).item()),
                    "train_posterior_null_active_edges_per_step": float(
                        (totals[27] / global_steps).item()
                    ),
                    "train_posterior_null_top1_fraction": float(
                        (totals[28] / totals[27]).item()
                        if float(totals[27].item()) > 0.0
                        else 0.0
                    ),
                    "train_posterior_candidate_active_edges_per_step": float(
                        (totals[29] / global_steps).item()
                    ),
                    "train_posterior_candidate_top1_fraction": float(
                        (totals[34] / totals[29]).item()
                        if float(totals[29].item()) > 0.0
                        else 0.0
                    ),
                    "train_posterior_candidate_cross_entropy_loss": float(
                        (totals[35] / totals[29]).item()
                        if float(totals[29].item()) > 0.0
                        else 0.0
                    ),
                    "train_posterior_null_cross_entropy_loss": float(
                        (totals[36] / totals[27]).item()
                        if float(totals[27].item()) > 0.0
                        else 0.0
                    ),
                    "train_rgb_identity_posterior_loss": float(
                        (totals[38] / global_steps).item()
                    ),
                    "train_rgb_identity_posterior_top1_fraction": float(
                        (totals[39] / totals[40]).item()
                        if float(totals[40].item()) > 0.0
                        else 0.0
                    ),
                    "train_rgb_identity_posterior_active_edges_per_step": float(
                        (totals[40] / global_steps).item()
                    ),
                    "train_rgb_identity_permutation_loss": float(
                        (totals[41] / global_steps).item()
                    ),
                    "train_rgb_identity_permutation_mean_gap": float(
                        (totals[42] / totals[44]).item()
                        if float(totals[44].item()) > 0.0
                        else 0.0
                    ),
                    "train_rgb_identity_permutation_win_fraction": float(
                        (totals[43] / totals[44]).item()
                        if float(totals[44].item()) > 0.0
                        else 0.0
                    ),
                    "train_rgb_identity_permutation_active_edges_per_step": float(
                        (totals[44] / global_steps).item()
                    ),
                    "train_context_coherence_hard_repeat_loss": float(
                        (totals[45] / global_steps).item()
                    ),
                    "train_context_coherence_hard_repeat_mean_gap": float(
                        (totals[46] / totals[48]).item()
                        if float(totals[48].item()) > 0.0
                        else 0.0
                    ),
                    "train_context_coherence_hard_repeat_win_fraction": float(
                        (totals[47] / totals[48]).item()
                        if float(totals[48].item()) > 0.0
                        else 0.0
                    ),
                    "train_context_coherence_hard_repeat_active_edges_per_step": float(
                        (totals[48] / global_steps).item()
                    ),
                    "train_context_coherence_permutation_loss": float(
                        (totals[49] / global_steps).item()
                    ),
                    "train_context_coherence_permutation_mean_gap": float(
                        (totals[50] / totals[52]).item()
                        if float(totals[52].item()) > 0.0
                        else 0.0
                    ),
                    "train_context_coherence_permutation_win_fraction": float(
                        (totals[51] / totals[52]).item()
                        if float(totals[52].item()) > 0.0
                        else 0.0
                    ),
                    "train_context_coherence_permutation_active_edges_per_step": float(
                        (totals[52] / global_steps).item()
                    ),
                    "train_rgb_identity_head_update_l2": float(totals[53].item()),
                    "train_context_coherence_head_update_l2": float(totals[54].item()),
                    "train_scalar_head_gradient_l2_per_step": float(
                        (totals[30] / global_steps).item()
                    ),
                    "train_scalar_head_update_l2": float(totals[31].item()),
                    "train_scalar_head_nonfinite_gradient_steps": int(totals[32].item()),
                    "train_amp_skipped_steps": int(totals[33].item()),
                    "global_query_steps": int(global_steps),
                    "epoch_seconds": float(time.time() - epoch_start),
                    **_identity_llr_head_statistics(model_for_train),
                    **{f"inner_{name}": value for name, value in inner.items()},
                    "inner_gate_passed": bool(gate["passed"]),
                }
                if _is_better(
                    candidate=inner,
                    incumbent=best_metrics,
                    candidate_passed=bool(gate["passed"]),
                    incumbent_passed=bool(False if incumbent_gate is None else incumbent_gate["passed"]),
                    training_stage=str(args.training_stage),
                    minimum_posterior_candidate_top1_fraction=float(
                        args.minimum_posterior_candidate_top1_fraction
                    ),
                ):
                    best_metrics = dict(inner)
                    best_epoch = int(epoch + 1)
                    best_checkpoint_kind = "p1_finetune_epoch"
                    core = model_for_train.module if isinstance(model_for_train, DistributedDataParallel) else model_for_train
                    best_state = {
                        name: value.detach().cpu().clone() for name, value in core.state_dict().items()
                    }
                history.append(epoch_metrics)
                _write_json_atomically(output_dir / "history.partial.json", history)
                _write_json_atomically(
                    output_dir / "progress.json",
                    {
                        "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                        "status": "running",
                        "completed_epochs": int(epoch + 1),
                        "last_epoch": epoch_metrics,
                    },
                )
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank != 0:
            return {}
        if best_state is None or best_metrics is None or best_epoch < 0:
            raise RuntimeError("identity LLR training did not select a checkpoint")
        gate = _checkpoint_gate(metrics=best_metrics, args=args)
        input_paths = {
            "rgb_spatial_layout": layout_path,
            "training_targets": targets_path,
            "hard_repeat_targets": hard_path,
            "radio_final_context_cache": Path(args.radio_final_context_cache),
            "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
            "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        }
        if str(args.identity_observation_pretrain_checkpoint).strip():
            input_paths["identity_observation_pretrain_checkpoint"] = Path(
                args.identity_observation_pretrain_checkpoint
            )
        else:
            input_paths["texture_observation_pretrain_checkpoint"] = Path(
                args.texture_observation_pretrain_checkpoint
            )
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "model_format": CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
            "architecture": "candidate_set_conditioned_independently_calibrated_rgb_identity_plus_multiscale_context_coherence_llr_experts_with_source_specific_availability_learned_support_view_posterior_and_explicit_visual_null_residual_v6",
            "contains_target_fields": False,
            "checkpoint_contains_train_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_used_by_runtime_scorer": False,
            "render": False,
            "image_retrieval_or_submap_used": False,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": int(layout.candidate_count),
            "fixed_support_view_count": int(layout.support_view_count),
            "explicit_null": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "candidate_set_conditioned": True,
            "edge_llr_factorization": "bounded_sum_of_independent_rgb_identity_and_multiscale_context_coherence_raw_llrs_v1",
            "edge_source_availability": {
                "rgb_identity": "rgb_crop_usable_only_else_neutral_llr_v1",
                "context_coherence": "full_map_radio_alike_context_usable_only_else_neutral_llr_v1",
                "combined": "union_of_independent_source_masks_no_cross_source_invalidation_v1",
            },
            "candidate_slot_permutation_equivariant": True,
            "support_view_embeddings_directly_averaged": False,
            "support_view_posterior": "learned_visual_logits_over_usable_fixed_views_with_immutable_coverage_base_mass_v1",
            "candidate_null_visual_residual": "permutation_invariant_visual_candidate_set_summary_added_after_immutable_null_prior_v1",
            "visual_evidence_gate_version": (
                "normal_and_independent_rgb_context_coherence_candidate_null_visual_controls_l0_v2"
                if str(args.training_stage) == "identity_l0"
                else "normal_and_coherent_pose_group_fixed_prior_residual_visual_controls_v4"
            ),
            "diagnostic_only": True,
            "promotion_allowed": False,
            "raw_scores_must_not_feed_pnp": True,
            "initialization": initialization,
            "train_only_inner_gate_passed": bool(gate["passed"]),
            "holdout_evaluation_allowed": bool(gate["passed"]),
            "encoder_inputs": [
                "frozen_query_anchor_xy",
                "fixed_support_observation_xy",
                "full_2d_radio_final_absolute_context_crop",
                "full_2d_radio_intermediate_structure_crop",
                "full_2d_alike_phase_context_crop",
                "real_rgb_query_and_fixed_support_region_patches",
            ],
            "encoder_excludes": [
                "pose_matrix",
                "projection_offset",
                "reprojection_residual",
                "ground_truth_label",
                "track_id",
                "candidate_rank",
                "coarse_score",
                "candidate_prior_probability",
            ],
            "lineage": {
                "layout_sha256": layout_sha,
                "training_targets_sha256": targets_sha,
                "hard_repeat_targets_sha256": file_sha256_short(hard_path),
                # Persist the exact source contract used to construct the
                # encoder.  The candidate layout itself intentionally has no
                # image-feature cache lineage, so using its metadata here
                # would silently erase the source manifest.
                **identity_source_lineage,
                "inputs": {
                    name: {"path": str(path), "sha256": file_sha256_short(path)}
                    for name, path in input_paths.items()
                },
            },
            "config": {
                "rgb_context_radius_px": float(args.rgb_context_radius_px),
                "rgb_step_px": float(args.rgb_step_px),
                "texture_feature_dim": int(args.texture_feature_dim),
                "hidden_dim": int(args.hidden_dim),
                "max_abs_log_ratio": float(args.max_abs_log_ratio),
                "candidate_prior_logit_weight": float(args.candidate_prior_logit_weight),
                "candidate_prior_fusion": "fixed_log_prior_added_after_prior_free_visual_llr_v1",
                "explicit_null": "immutable_input_null_log_prior_plus_target_free_visual_residual_v2",
                "support_view_posterior": "learned_after_edge_llr_before_candidate_marginalization_v1",
                "edge_llr_factorization": {
                    "rgb_identity": "source-masked_candidate_set_relative_features_v1",
                    "context_coherence": "source-masked_candidate_set_relative_features_v1",
                    "combination": "sum_raw_llrs_then_bound_once_v1",
                    "expert_losses_use_fused_view_or_null": False,
                },
                "edge_source_availability": "per_source_masks_union_neutral_missing_v1",
                "scalar_head_precision": "fp32_under_amp_for_likelihood_stability_v1",
                "training_stage": str(args.training_stage),
                "feature_training": feature_training,
                "train_query_anchor_jitter": {
                    "radius_px": float(args.train_query_anchor_jitter_radius_px),
                    "distribution": "uniform_radius_uniform_angle_v1",
                    "applied_to": "train_query_rgb_and_context_crops_only",
                    "leaves_candidate_support_prior_and_target_arrays_unchanged": True,
                    "validation_or_runtime_applies_jitter": False,
                },
                "amp_init_scale": float(args.amp_init_scale),
                "edge_chunk_size": int(args.edge_chunk_size),
                "context_windows": dict(windows),
                "validation_selector": {
                    "policy": str(args.validation_selector_policy),
                    "point_budget": int(args.validation_selector_point_budget),
                    "grid_rows": int(args.validation_selector_grid_rows),
                    "grid_columns": int(args.validation_selector_grid_columns),
                    "target_free_static_only": True,
                },
            },
            "training": {
                "objective": (
                    "registered_exact_candidate_or_explicit_null_class_weighted_fixed_prior_residual_cross_entropy_plus_"
                    "direct_current_coherent_repeat_identity_margin_plus_independent_rgb_identity_candidate_posterior_and_support_derangement_plus_"
                    "independent_context_coherence_hard_repeat_and_support_derangement_controls_l0_v2"
                    if str(args.training_stage) == "identity_l0"
                    else "registered_exact_candidate_or_explicit_null_fixed_prior_residual_margin_plus_"
                    "candidate_null_posterior_cross_entropy_plus_pair_id_grouped_correct_vs_coherent_wrong_"
                    "pose_margin_plus_support_derangement_controls_v5"
                ),
                "training_stage": str(args.training_stage),
                "feature_training": feature_training,
                "allow_full_identity_l0_feature_finetune": bool(
                    args.allow_full_identity_l0_feature_finetune
                ),
                "train_query_anchor_jitter_radius_px": float(
                    args.train_query_anchor_jitter_radius_px
                ),
                "amp_init_scale": float(args.amp_init_scale),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "learning_rate_groups": {
                    str(group["group_name"]): float(group["lr"])
                    for group in optimizer.param_groups
                },
                "weight_decay": float(args.weight_decay),
                "identity_margin": float(args.identity_margin),
                "identity_loss_weight": float(args.identity_loss_weight),
                "posterior_cross_entropy_loss_weight": float(
                    args.posterior_cross_entropy_loss_weight
                ),
                "posterior_candidate_loss_weight": float(args.posterior_candidate_loss_weight),
                "posterior_null_loss_weight": float(args.posterior_null_loss_weight),
                "rgb_identity_posterior_loss_weight": float(
                    args.rgb_identity_posterior_loss_weight
                ),
                "rgb_identity_permutation_loss_weight": float(
                    args.rgb_identity_permutation_loss_weight
                ),
                "context_coherence_hard_repeat_loss_weight": float(
                    args.context_coherence_hard_repeat_loss_weight
                ),
                "context_coherence_permutation_loss_weight": float(
                    args.context_coherence_permutation_loss_weight
                ),
                "hard_repeat_margin": float(args.hard_repeat_margin),
                "hard_repeat_loss_weight": float(args.hard_repeat_loss_weight),
                "hard_pose_group_margin": float(args.hard_pose_margin),
                "hard_pose_group_loss_weight": float(args.hard_pose_loss_weight),
                "hard_pose_group_permutation_loss_weight": float(
                    args.hard_pose_permutation_loss_weight
                ),
                "hard_pose_group_minimum_distinct_points": int(args.minimum_hard_pose_points),
                "permutation_margin": float(args.permutation_margin),
                "permutation_loss_weight": float(args.permutation_loss_weight),
                "hard_repeat_target_semantics": "frozen_current_target_free_system_mode_then_train_only_registered_exact_track_vs_distinct_wrong_candidate_v1",
                "hard_pose_group_target_contract": "pair_id_is_train_only_coherent_wrong_pose_membership_joined_after_target_free_candidate_edge_llr_v1",
                "pose_or_group_id_not_available_to_runtime_encoder": True,
                "candidate_prior_not_available_to_runtime_encoder": True,
                "prior_residual_final_edge_head_reset": bool(
                    args.reset_edge_head_final_for_prior_residual
                ),
                "current_p1_scalar_head_final_reset": bool(
                    args.reset_current_p1_scalar_head_finals
                ),
                "pointwise_hard_repeat_diagnostic_thresholds": {
                    "minimum_hard_repeat_win_fraction": float(
                        args.minimum_hard_repeat_win_fraction
                    ),
                    "minimum_hard_repeat_gap": float(args.minimum_hard_repeat_gap),
                    "minimum_inner_hard_repeat_eligible_query_fraction": float(
                        args.minimum_inner_hard_repeat_eligible_query_fraction
                    ),
                },
                "inner_gate_thresholds": {
                    "minimum_win_fraction": float(args.minimum_win_fraction),
                    "minimum_posterior_candidate_top1_fraction": float(
                        args.minimum_posterior_candidate_top1_fraction
                    ),
                    "minimum_rgb_identity_posterior_top1_fraction": float(
                        args.minimum_rgb_identity_posterior_top1_fraction
                    ),
                    "minimum_rgb_identity_permutation_gap": float(
                        args.minimum_rgb_identity_permutation_gap
                    ),
                    "minimum_context_coherence_hard_repeat_gap": float(
                        args.minimum_context_coherence_hard_repeat_gap
                    ),
                    "minimum_context_coherence_permutation_gap": float(
                        args.minimum_context_coherence_permutation_gap
                    ),
                    "minimum_normal_gap": float(args.minimum_normal_gap),
                    "minimum_visual_gap_delta": float(args.minimum_visual_gap_delta),
                    "minimum_position_visual_gap": float(args.minimum_position_visual_gap),
                    "minimum_inner_eligible_query_fraction": float(
                        args.minimum_inner_eligible_query_fraction
                    ),
                    "minimum_inner_hard_pose_eligible_query_fraction": float(
                        args.minimum_inner_hard_pose_eligible_query_fraction
                    ),
                    "minimum_hard_pose_win_fraction": float(
                        args.minimum_hard_pose_win_fraction
                    ),
                    "minimum_hard_pose_gap": float(args.minimum_hard_pose_gap),
                    "minimum_hard_pose_visual_gap_delta": float(
                        args.minimum_hard_pose_visual_gap_delta
                    ),
                },
                "inner_validation": {
                    "selected_epoch": int(best_epoch),
                    "selected_checkpoint_kind": str(best_checkpoint_kind),
                    "metrics": best_metrics,
                    "gate": gate,
                    "selector_target_free": True,
                    "selection_before_target_join": True,
                    "observed_target_preserving_sampler_used": False,
                },
            },
        }
        checkpoint_path = output_dir / "candidate_pose_rgb_spatial_identity_llr.pt"
        torch.save({"format": CHECKPOINT_FORMAT, "metadata": metadata, "state_dict": best_state}, checkpoint_path)
        summary = {
            "stage": "train_candidate_pose_rgb_spatial_identity_llr",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "initialization": initialization,
            "checkpoint_selection": {
                "selected_epoch": int(best_epoch),
                "selected_checkpoint_kind": str(best_checkpoint_kind),
                "inner_validation": best_metrics,
                "gate": gate,
            },
            "history": history,
            "rgb_cache_rank0": image_cache.summary(),
            "protocol": {
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "runtime_layout_remains_target_free": True,
                "pose_or_ground_truth_not_available_to_runtime_encoder": True,
                "train_only_target_artifact": True,
                "inner_validation_selector_target_free": True,
                "inner_validation_observed_target_sampler_used": False,
                "train_query_anchor_jitter_is_target_free": True,
                "train_query_anchor_jitter_applied_to_validation_or_runtime": False,
                "training_layout_runtime_forbidden": bool(
                    layout.metadata.get("runtime_scorer_must_not_load_this_layout", False)
                ),
                "heldout_validation_or_test_not_run": True,
            },
        }
        _write_json_atomically(output_dir / "history.json", history)
        _write_json_atomically(output_dir / "summary.json", summary)
        _write_json_atomically(
            output_dir / "progress.json",
            {
                "stage": "train_candidate_pose_rgb_spatial_identity_llr",
                "status": "complete",
                "completed_epochs": int(args.epochs),
                "selected_epoch": int(best_epoch),
                "selected_checkpoint_kind": str(best_checkpoint_kind),
                "inner_gate_passed": bool(gate["passed"]),
            },
        )
        return summary
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    summary = train_candidate_pose_rgb_spatial_identity_llr(parse_args(argv))
    if summary:
        print(
            json.dumps(
                {
                    "checkpoint": summary.get("checkpoint"),
                    "initialization_only": bool(
                        summary.get("protocol", {}).get("initialization_only", False)
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
