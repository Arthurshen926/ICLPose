"""Train L0 candidate identity evidence on the actual frozen P1 layout.

The model sees only target-free query/support image-space context.  Strict
track identity and mined coherent-repeat targets are joined after that forward
pass, never stored in the runtime layout or exposed to inference scoring.

This is intentionally the identity/support-view phase before RGB spatial
density, pose-level losses, PnP, or external evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _source_table,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    coherent_hard_repeat_context_margin_loss,
    load_context_observation_pretrain_initialization_checkpoint,
    training_gate_decision,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    query_grouped_pose_margin_loss,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    candidate_pose_rgb_spatial_score_component_prediction,
    context_candidate_logit_mixture,
    context_identity_cross_entropy_loss,
    context_identity_support_permutation_margin_loss,
    permute_runtime_support_image_appearance_only,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
    selected_candidate_view_context_log_likelihood_ratio,
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
CHECKPOINT_FORMAT = "candidate_pose_context_identity_l0_checkpoint_v2"
FIXED_FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"
OBJECTIVE = (
    "strict_observed_track_candidate_cross_entropy_plus_train_mined_"
    "coherent_repeat_context_margin_plus_optional_correct_vs_coherent_wrong_"
    "fixed_mixture_pose_margin_plus_geometry_fixed_support_derangement_l0_v3"
)

_CONTEXT_STATIC_SELECTOR_POLICIES = tuple(
    policy for policy in TARGET_FREE_SELECTOR_POLICIES if "rgb" not in policy
)


def fixed_final_epoch_checkpoint_selection(*, epochs: int) -> dict[str, object]:
    """Declare the non-data-dependent checkpoint selection policy."""

    if int(epochs) <= 0:
        raise ValueError("P1 L0 fixed-final-epoch selection requires positive epochs")
    return {
        "policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "selected_epoch": int(epochs),
        "inner_validation_used_for_model_selection": False,
    }


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_geometry_fixed_support_image_control(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: object,
    permuted_prediction: object,
) -> None:
    """Reject a visual control if it changes geometry or crop availability."""

    invariants = (
        (runtime.query_image_indices, permuted_runtime.query_image_indices),
        (runtime.query_xy, permuted_runtime.query_xy),
        (runtime.support_xy, permuted_runtime.support_xy),
        (runtime.support_view_valid, permuted_runtime.support_view_valid),
        (runtime.candidate_view_weights, permuted_runtime.candidate_view_weights),
        (runtime.candidate_probabilities, permuted_runtime.candidate_probabilities),
        (runtime.null_probabilities, permuted_runtime.null_probabilities),
    )
    if any(not torch.equal(left, right) for left, right in invariants):
        raise RuntimeError("P1 L0 support control changed geometry or mixture mass")
    per_point_valid = runtime.support_view_valid.reshape(runtime.point_count, -1).sum(dim=1)
    if bool(torch.any(per_point_valid > 1)) and torch.equal(
        runtime.support_image_indices, permuted_runtime.support_image_indices
    ):
        raise RuntimeError("P1 L0 support control did not derange image content")
    normal_usable = getattr(normal_prediction, "context_edge_usable", None)
    permuted_usable = getattr(permuted_prediction, "context_edge_usable", None)
    if (
        normal_usable is None
        or permuted_usable is None
        or not torch.equal(normal_usable, permuted_usable)
    ):
        raise RuntimeError("P1 L0 support control changed context crop availability")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-repeat-targets", default="")
    parser.add_argument(
        "--context-observation-pretrain-checkpoint",
        default="",
        help=(
            "Optional gate-approved absolute-context L0 checkpoint. Only its context encoders "
            "and identity head are loaded; RGB and spatial-density weights are never copied."
        ),
    )
    parser.add_argument(
        "--context-observation-pairs",
        default="",
        help=(
            "Train-only source pair artifact used to prove that the pretrain split exactly "
            "matches this P1 inner fold before any context weights are loaded."
        ),
    )
    parser.add_argument("--search-radius-px", type=float, default=None)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument(
        "--context-encoder-arch",
        default="absolute_cross_attention_v3",
        help=(
            "L0 requires explicit full-image coordinates so repeated local crops can be "
            "tested against a position-only control."
        ),
    )
    parser.add_argument(
        "--trainable-context-scope",
        choices=("head_only", "all_context"),
        default="all_context",
        help="P1 small-data adaptation scope after optional broad-context initialization.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--identity-support-permutation-loss-weight", type=float, default=0.25)
    parser.add_argument("--identity-support-permutation-margin", type=float, default=0.25)
    parser.add_argument("--identity-support-permutation-shift", type=int, default=2)
    parser.add_argument("--hard-repeat-context-loss-weight", type=float, default=0.0)
    parser.add_argument("--hard-repeat-context-margin", type=float, default=0.25)
    parser.add_argument(
        "--pose-context-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only correct-pose versus coherent-wrong fixed-mixture margin. "
            "The visual encoder remains pose-free."
        ),
    )
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=64)
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument(
        "--validation-selector-policy",
        choices=_CONTEXT_STATIC_SELECTOR_POLICIES,
        default="coarse_margin",
        help="Strict target-free static point policy used only for the held-out P1 gate.",
    )
    parser.add_argument("--validation-selector-point-budget", type=int, default=64)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--pose-margin", type=float, default=0.25)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--minimum-position-only-gap", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.context_radius_px),
        float(args.step_px),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.identity_loss_weight),
        float(args.identity_support_permutation_loss_weight),
        float(args.identity_support_permutation_margin),
        float(args.hard_repeat_context_loss_weight),
        float(args.hard_repeat_context_margin),
        float(args.pose_context_loss_weight),
        float(args.max_abs_context_log_ratio),
        float(args.pose_margin),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_visual_gap_delta),
        float(args.minimum_position_only_gap),
        float(args.gradient_clip_norm),
    )
    if (
        int(args.epochs) <= 0
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.max_points_per_query) < 4
        or int(args.validation_selector_point_budget) < 4
        or int(args.validation_selector_grid_rows) <= 0
        or int(args.validation_selector_grid_columns) <= 0
        or int(args.max_hard_repeat_edges_per_query) < 0
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or int(args.identity_support_permutation_shift) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.identity_loss_weight) < 0.0
        or float(args.identity_support_permutation_loss_weight) < 0.0
        or float(args.identity_support_permutation_margin) < 0.0
        or float(args.hard_repeat_context_loss_weight) < 0.0
        or float(args.hard_repeat_context_margin) < 0.0
        or float(args.pose_context_loss_weight) < 0.0
        or float(args.max_abs_context_log_ratio) <= 0.0
        or float(args.pose_margin) < 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or float(args.minimum_visual_gap_delta) < 0.0
        or float(args.minimum_position_only_gap) < 0.0
        or float(args.gradient_clip_norm) <= 0.0
    ):
        raise ValueError("P1 L0 context-identity arguments are invalid")
    if (
        float(args.identity_loss_weight)
        + float(args.identity_support_permutation_loss_weight)
        + float(args.hard_repeat_context_loss_weight)
        <= 0.0
    ):
        raise ValueError("P1 L0 context-identity training has no active objective")
    if float(args.hard_repeat_context_loss_weight) > 0.0 and not str(
        args.hard_repeat_targets
    ).strip():
        raise ValueError("hard-repeat context loss requires train-only hard-repeat targets")
    if str(args.context_observation_pretrain_checkpoint).strip() and not str(
        args.context_observation_pairs
    ).strip():
        raise ValueError(
            "P1 L0 context pretrain initialization requires its train-only observation-pair artifact"
        )
    if str(args.context_observation_pairs).strip() and not str(
        args.context_observation_pretrain_checkpoint
    ).strip():
        raise ValueError("context observation pairs are meaningful only with a pretrain checkpoint")
    if (
        resolve_candidate_pose_rgb_spatial_context_encoder_arch(args.context_encoder_arch)
        != "absolute_cross_attention_v3"
    ):
        raise ValueError(
            "P1 L0 context identity requires absolute_cross_attention_v3 for its position-only control"
        )
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _distributed_output_conflict(
    *, state: _DistributedState, paths: Sequence[Path]
) -> bool:
    conflict = bool(any(Path(path).exists() for path in paths)) if state.rank == 0 else False
    if state.enabled:
        value = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(value, src=0)
        conflict = bool(value.item())
    return conflict


def _rank_query_ids(
    *, query_ids: Sequence[str], state: _DistributedState, seed: int, epoch: int
) -> tuple[str, ...]:
    ids = np.asarray(sorted(set(str(value) for value in query_ids))).astype(str)
    if len(ids) == 0:
        raise ValueError("P1 L0 context-identity has no train queries")
    ordered = np.random.default_rng(int(seed) + int(epoch) * 1000003).permutation(ids)
    per_rank = int(math.ceil(len(ordered) / float(state.world_size)))
    padded = np.resize(ordered, per_rank * int(state.world_size))
    begin = int(state.rank) * per_rank
    return tuple(str(value) for value in padded[begin : begin + per_rank].tolist())


def select_identity_group_points(
    *,
    group: object,
    max_points: int,
    seed: int,
    required_source_point_ids: Sequence[int] = (),
) -> np.ndarray:
    """Retain strict identities and direct hard-repeat rows before random fill.

    The P1 hard-repeat objective is the only supervision that directly names a
    coherent wrong identity.  Selecting detector points first and then joining
    those targets can silently erase that supervision, so caller-supplied
    train-only source IDs are treated as required rows whenever they fit in the
    fixed per-query budget.
    """

    count = int(getattr(group, "point_count"))
    observed = np.asarray(getattr(group, "spatial_target_observed"), dtype=bool)
    limit = int(max_points)
    source_ids = np.asarray(getattr(group, "source_point_ids"), dtype=np.int64).reshape(-1)
    if (
        observed.ndim != 2
        or observed.shape[0] != count
        or source_ids.shape != (count,)
        or limit < 1
    ):
        raise ValueError("P1 L0 identity point selection inputs are invalid")
    if limit >= count:
        return np.arange(count, dtype=np.int64)
    positives = np.flatnonzero(np.any(observed, axis=1))
    required_ids = np.asarray(required_source_point_ids, dtype=np.int64).reshape(-1)
    source_position = {int(source_id): position for position, source_id in enumerate(source_ids.tolist())}
    required = np.asarray(
        [source_position[int(source_id)] for source_id in np.unique(required_ids).tolist() if int(source_id) in source_position],
        dtype=np.int64,
    )
    if len(required_ids) and len(required) != len(np.unique(required_ids)):
        raise ValueError("P1 L0 required hard-repeat source point is absent from its query group")
    priority = np.unique(np.concatenate([positives, required]))
    generator = np.random.default_rng(int(seed))
    if len(priority) >= limit:
        return np.sort(generator.choice(priority, size=limit, replace=False)).astype(np.int64)
    remaining = np.setdiff1d(np.arange(count, dtype=np.int64), priority, assume_unique=True)
    extra = generator.choice(remaining, size=limit - len(priority), replace=False)
    return np.sort(np.concatenate([priority, extra])).astype(np.int64)


def resolve_target_free_validation_point_budget(
    *, requested_point_budget: int, available_point_count: int
) -> int:
    """Clamp a static selector budget to one query's frozen target-free pool.

    A query may legitimately have fewer detector points than the nominal global
    budget. Dropping it from validation would favor easy images, so retain all
    available target-free points instead. Pools smaller than four remain
    invalid because they cannot support meaningful spatially diverse checking.
    """

    requested = int(requested_point_budget)
    available = int(available_point_count)
    if requested < 4 or available < 4:
        raise ValueError("P1 L0 validation selector requires at least four target-free points")
    return min(requested, available)


def context_pose_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: object,
    batch: object,
    pose_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Join train-only pose targets only after the target-free context forward.

    This is the missing bridge between a candidate-edge LLR and the actual
    inference quantity. Each query point keeps its fixed top-L/null mixture;
    the loss only asks the correct pose score to beat the coherent-wrong pose
    pool after the visual evidence has been emitted.
    """

    context_prediction = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=prediction,
        component="context_only",
    )
    correct = score_candidate_pose_rgb_spatial_batch(
        runtime=runtime,
        prediction=context_prediction,
        candidate_projection_offsets_xy=torch.as_tensor(
            getattr(batch, "correct_projection_offsets_xy"),
            device=context_prediction.joint_log_probabilities.device,
        ).unsqueeze(0),
        candidate_projection_valid=torch.as_tensor(
            getattr(batch, "correct_projection_valid"),
            device=context_prediction.joint_log_probabilities.device,
        ).unsqueeze(0),
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_log_likelihood_ratio=6.0,
    ).pose_log_likelihood_ratios
    wrong = score_candidate_pose_rgb_spatial_batch(
        runtime=runtime,
        prediction=context_prediction,
        candidate_projection_offsets_xy=torch.as_tensor(
            getattr(batch, "wrong_projection_offsets_xy"),
            device=context_prediction.joint_log_probabilities.device,
        ),
        candidate_projection_valid=torch.as_tensor(
            getattr(batch, "wrong_projection_valid"),
            device=context_prediction.joint_log_probabilities.device,
        ),
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_log_likelihood_ratio=6.0,
    ).pose_log_likelihood_ratios
    if correct.shape != (1,) or wrong.ndim != 1 or len(wrong) < 1:
        raise ValueError("P1 L0 pose targets do not define one correct and at least one wrong mode")
    return query_grouped_pose_margin_loss(
        correct_scores=correct,
        coherent_wrong_scores=wrong.reshape(1, -1),
        margin=float(pose_margin),
    )


def _context_only_validation(
    *,
    model: torch.nn.Module,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, object],
    complete_runtime: object,
    query_ids: Sequence[str],
    image_size: tuple[int, int],
    state: _DistributedState,
    selector_policy: str,
    selector_point_budget: int,
    selector_grid_rows: int,
    selector_grid_columns: int,
    pose_margin: float,
    amp_enabled: bool,
    context_appearance_mode: str = "visual",
) -> dict[str, float]:
    """Evaluate L0 with target-free static selection before target joins.

    Training may retain sparse registered rows so its supervised gradients are
    not lost. Validation must not: the policy below sees only frozen layout
    fields, then correct/wrong projections are joined after visual forwards.
    """

    if (
        not query_ids
        or str(selector_policy) not in _CONTEXT_STATIC_SELECTOR_POLICIES
        or int(selector_point_budget) < 4
        or int(selector_grid_rows) <= 0
        or int(selector_grid_columns) <= 0
        or str(context_appearance_mode) not in {"visual", "position_only"}
    ):
        raise ValueError("P1 L0 target-free validation arguments are invalid")
    model.eval()
    totals = torch.zeros((7,), dtype=torch.float64, device=state.device)
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("P1 L0 target-free validation query is unresolved")
        selector_input = selector_input_from_target_free_layout(
            layout=layout, rows=group.layout_rows
        )
        point_budget = resolve_target_free_validation_point_budget(
            requested_point_budget=int(selector_point_budget),
            available_point_count=int(selector_input.point_count),
        )
        selector_scores = target_free_selector_scores(
            selector_input=selector_input,
            policy=str(selector_policy),
        )
        positions = select_target_free_spatial_quota(
            selector_input=selector_input,
            quality_scores=selector_scores,
            point_budget=point_budget,
            grid_rows=int(selector_grid_rows),
            grid_columns=int(selector_grid_columns),
            image_size=image_size,
        )
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=positions,
            device=state.device,
        )
        permuted_runtime = permute_runtime_support_image_appearance_only(batch.runtime, shift=1)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = candidate_pose_rgb_spatial_score_component_prediction(
                prediction=model(
                    runtime=batch.runtime,
                    context_only=True,
                    context_appearance_mode=str(context_appearance_mode),
                ),
                component="context_only",
            )
            permuted_prediction = candidate_pose_rgb_spatial_score_component_prediction(
                prediction=model(
                    runtime=permuted_runtime,
                    context_only=True,
                    context_appearance_mode=str(context_appearance_mode),
                ),
                component="context_only",
            )
            _assert_geometry_fixed_support_image_control(
                runtime=batch.runtime,
                permuted_runtime=permuted_runtime,
                normal_prediction=normal_prediction,
                permuted_prediction=permuted_prediction,
            )
            normal_correct = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=normal_prediction,
                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=6.0,
            ).pose_log_likelihood_ratios
            normal_wrong = score_candidate_pose_rgb_spatial_batch(
                runtime=batch.runtime,
                prediction=normal_prediction,
                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                candidate_projection_valid=batch.wrong_projection_valid,
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=6.0,
            ).pose_log_likelihood_ratios
            normal_loss, normal_metrics = query_grouped_pose_margin_loss(
                correct_scores=normal_correct,
                coherent_wrong_scores=normal_wrong.reshape(1, -1),
                margin=float(pose_margin),
            )
            permuted_correct = score_candidate_pose_rgb_spatial_batch(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                candidate_projection_offsets_xy=batch.correct_projection_offsets_xy.unsqueeze(0),
                candidate_projection_valid=batch.correct_projection_valid.unsqueeze(0),
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=6.0,
            ).pose_log_likelihood_ratios
            permuted_wrong = score_candidate_pose_rgb_spatial_batch(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                candidate_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                candidate_projection_valid=batch.wrong_projection_valid,
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=6.0,
            ).pose_log_likelihood_ratios
            permuted_loss, permuted_metrics = query_grouped_pose_margin_loss(
                correct_scores=permuted_correct,
                coherent_wrong_scores=permuted_wrong.reshape(1, -1),
                margin=float(pose_margin),
            )
        totals += torch.tensor(
            [
                float(normal_loss.item()),
                float(normal_metrics["query_mean_correct_minus_hardest_wrong"]),
                float(normal_metrics["query_correct_win_fraction"]),
                float(permuted_loss.item()),
                float(permuted_metrics["query_mean_correct_minus_hardest_wrong"]),
                float(permuted_metrics["query_correct_win_fraction"]),
                1.0,
            ],
            dtype=torch.float64,
            device=state.device,
        )
    if state.enabled:
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    if float(totals[-1].item()) <= 0.0:
        raise RuntimeError("P1 L0 target-free validation did not evaluate a query")
    return {
        "normal_query_grouped_loss": float((totals[0] / totals[-1]).item()),
        "normal_mean_correct_minus_hardest_wrong": float((totals[1] / totals[-1]).item()),
        "normal_correct_win_fraction": float((totals[2] / totals[-1]).item()),
        "permuted_query_grouped_loss": float((totals[3] / totals[-1]).item()),
        "permuted_mean_correct_minus_hardest_wrong": float((totals[4] / totals[-1]).item()),
        "permuted_correct_win_fraction": float((totals[5] / totals[-1]).item()),
        "query_count": float(totals[-1].item()),
    }


@torch.no_grad()
def _context_only_direct_evidence_validation(
    *,
    model: torch.nn.Module,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, object],
    complete_runtime: object,
    hard_repeat_groups: Mapping[str, object],
    query_ids: Sequence[str],
    image_size: tuple[int, int],
    state: _DistributedState,
    selector_policy: str,
    selector_point_budget: int,
    selector_grid_rows: int,
    selector_grid_columns: int,
    max_hard_repeat_edges_per_query: int,
    amp_enabled: bool,
) -> dict[str, float]:
    """Audit L0 candidate evidence before it is folded into a pose score.

    The three forwards share exactly the same target-free selected P1 points.
    Only after all visual predictions are available do we join strict identity
    labels and mined coherent-repeat pairs.  Comparing the normal branch with
    a support derangement and a descriptor-zeroed position-only branch makes
    this a direct test of candidate-specific visual evidence, rather than a
    self-confirming PnP or pose-residual diagnostic.
    """

    if (
        not query_ids
        or str(selector_policy) not in _CONTEXT_STATIC_SELECTOR_POLICIES
        or int(selector_point_budget) < 4
        or int(selector_grid_rows) <= 0
        or int(selector_grid_columns) <= 0
        or int(max_hard_repeat_edges_per_query) < 0
    ):
        raise ValueError("P1 L0 direct evidence validation arguments are invalid")
    model.eval()
    # Each sum is first reduced to one value per query, so dense repeated
    # facade rows cannot outweigh the rest of the inner-validation images.
    identity_totals = torch.zeros((11,), dtype=torch.float64, device=state.device)
    hard_repeat_totals = torch.zeros((8,), dtype=torch.float64, device=state.device)

    def identity_row_statistics(
        *,
        logits: torch.Tensor,
        candidate_usable: torch.Tensor,
        labels: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~candidate_usable, -torch.inf)
        selected = masked[active].gather(1, labels[active, None]).squeeze(1)
        competing = masked[active].clone()
        competing.scatter_(1, labels[active, None], -torch.inf)
        cross_entropy = torch.logsumexp(masked[active], dim=1) - selected
        margin = selected - torch.amax(competing, dim=1)
        win = (torch.argmax(masked[active], dim=1) == labels[active]).to(
            dtype=torch.float32
        )
        return cross_entropy.mean(), margin.mean(), win.mean()

    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("P1 L0 direct evidence query is unresolved")
        selector_input = selector_input_from_target_free_layout(
            layout=layout, rows=group.layout_rows
        )
        point_budget = resolve_target_free_validation_point_budget(
            requested_point_budget=int(selector_point_budget),
            available_point_count=int(selector_input.point_count),
        )
        positions = select_target_free_spatial_quota(
            selector_input=selector_input,
            quality_scores=target_free_selector_scores(
                selector_input=selector_input, policy=str(selector_policy)
            ),
            point_budget=point_budget,
            grid_rows=int(selector_grid_rows),
            grid_columns=int(selector_grid_columns),
            image_size=image_size,
        )
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=positions,
            device=state.device,
        )
        permuted_runtime = permute_runtime_support_image_appearance_only(batch.runtime, shift=1)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = model(runtime=batch.runtime, context_only=True)
            permuted_prediction = model(runtime=permuted_runtime, context_only=True)
            position_only_prediction = model(
                runtime=batch.runtime,
                context_only=True,
                context_appearance_mode="position_only",
            )
            normal_logits, normal_usable = context_candidate_logit_mixture(
                runtime=batch.runtime, prediction=normal_prediction
            )
            permuted_logits, permuted_usable = context_candidate_logit_mixture(
                runtime=permuted_runtime, prediction=permuted_prediction
            )
            position_logits, position_usable = context_candidate_logit_mixture(
                runtime=batch.runtime, prediction=position_only_prediction
            )
        _assert_geometry_fixed_support_image_control(
            runtime=batch.runtime,
            permuted_runtime=permuted_runtime,
            normal_prediction=normal_prediction,
            permuted_prediction=permuted_prediction,
        )
        observed = torch.as_tensor(
            batch.spatial_target_observed, dtype=torch.bool, device=state.device
        )
        if (
            observed.shape != normal_logits.shape
            or permuted_logits.shape != normal_logits.shape
            or position_logits.shape != normal_logits.shape
            or torch.any(observed.sum(dim=1) > 1)
        ):
            raise ValueError("P1 L0 direct identity targets are incompatible")
        labels = torch.argmax(observed.to(dtype=torch.long), dim=1)
        common_candidate_usable = normal_usable & permuted_usable & position_usable
        active_identity = (
            torch.any(observed, dim=1)
            & common_candidate_usable.gather(1, labels[:, None]).squeeze(1)
            & (common_candidate_usable.sum(dim=1) >= 2)
        )
        if bool(active_identity.any()):
            normal_ce, normal_margin, normal_win = identity_row_statistics(
                logits=normal_logits,
                candidate_usable=common_candidate_usable,
                labels=labels,
                active=active_identity,
            )
            permuted_ce, permuted_margin, permuted_win = identity_row_statistics(
                logits=permuted_logits,
                candidate_usable=common_candidate_usable,
                labels=labels,
                active=active_identity,
            )
            position_ce, position_margin, position_win = identity_row_statistics(
                logits=position_logits,
                candidate_usable=common_candidate_usable,
                labels=labels,
                active=active_identity,
            )
            identity_totals += torch.tensor(
                [
                    float(normal_ce.item()),
                    float(normal_margin.item()),
                    float(normal_win.item()),
                    float(permuted_ce.item()),
                    float(permuted_margin.item()),
                    float(permuted_win.item()),
                    float(position_ce.item()),
                    float(position_margin.item()),
                    float(position_win.item()),
                    float(active_identity.sum().item()),
                    1.0,
                ],
                dtype=torch.float64,
                device=state.device,
            )

        hard_targets = hard_repeat_groups.get(str(query_id))
        hard_batch = (
            None
            if hard_targets is None
            else _hard_repeat_batch_from_group(
                hard_targets=hard_targets,
                group=group,
                point_positions=positions,
                device=state.device,
                max_edges=int(max_hard_repeat_edges_per_query),
                seed=0,
            )
        )
        if hard_batch is None:
            continue
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_positive, normal_positive_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.positive_candidate_indices,
                )
            )
            normal_negative, normal_negative_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=batch.runtime,
                    prediction=normal_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.negative_candidate_indices,
                )
            )
            permuted_positive, permuted_positive_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=permuted_runtime,
                    prediction=permuted_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.positive_candidate_indices,
                )
            )
            permuted_negative, permuted_negative_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=permuted_runtime,
                    prediction=permuted_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.negative_candidate_indices,
                )
            )
            position_positive, position_positive_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=batch.runtime,
                    prediction=position_only_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.positive_candidate_indices,
                )
            )
            position_negative, position_negative_usable = (
                selected_candidate_view_context_log_likelihood_ratio(
                    runtime=batch.runtime,
                    prediction=position_only_prediction,
                    point_indices=hard_batch.point_indices,
                    candidate_indices=hard_batch.negative_candidate_indices,
                )
            )
        active_hard_repeat = (
            normal_positive_usable
            & normal_negative_usable
            & permuted_positive_usable
            & permuted_negative_usable
            & position_positive_usable
            & position_negative_usable
        )
        if bool(active_hard_repeat.any()):
            normal_gap = (normal_positive - normal_negative)[active_hard_repeat]
            permuted_gap = (permuted_positive - permuted_negative)[active_hard_repeat]
            position_gap = (position_positive - position_negative)[active_hard_repeat]
            hard_repeat_totals += torch.tensor(
                [
                    float(normal_gap.mean().item()),
                    float((normal_gap > 0.0).to(dtype=torch.float32).mean().item()),
                    float(permuted_gap.mean().item()),
                    float((permuted_gap > 0.0).to(dtype=torch.float32).mean().item()),
                    float(position_gap.mean().item()),
                    float((position_gap > 0.0).to(dtype=torch.float32).mean().item()),
                    float(active_hard_repeat.sum().item()),
                    1.0,
                ],
                dtype=torch.float64,
                device=state.device,
            )
    if state.enabled:
        distributed.all_reduce(identity_totals, op=distributed.ReduceOp.SUM)
        distributed.all_reduce(hard_repeat_totals, op=distributed.ReduceOp.SUM)
    identity_count = float(identity_totals[-1].item())
    hard_repeat_count = float(hard_repeat_totals[7].item())

    def identity_value(index: int) -> float:
        return 0.0 if identity_count <= 0.0 else float((identity_totals[index] / identity_count).item())

    def hard_repeat_value(index: int) -> float:
        return (
            0.0
            if hard_repeat_count <= 0.0
            else float((hard_repeat_totals[index] / hard_repeat_count).item())
        )

    normal_identity_margin = identity_value(1)
    permuted_identity_margin = identity_value(4)
    position_identity_margin = identity_value(7)
    normal_hard_repeat_gap = hard_repeat_value(0)
    permuted_hard_repeat_gap = hard_repeat_value(2)
    position_hard_repeat_gap = hard_repeat_value(4)
    return {
        "identity_query_count": identity_count,
        "identity_active_rows_per_query": identity_value(9),
        "identity_normal_cross_entropy": identity_value(0),
        "identity_normal_mean_margin": normal_identity_margin,
        "identity_normal_top1_accuracy": identity_value(2),
        "identity_permuted_cross_entropy": identity_value(3),
        "identity_permuted_mean_margin": permuted_identity_margin,
        "identity_permuted_top1_accuracy": identity_value(5),
        "identity_position_only_cross_entropy": identity_value(6),
        "identity_position_only_mean_margin": position_identity_margin,
        "identity_position_only_top1_accuracy": identity_value(8),
        "identity_normal_minus_permuted_margin": normal_identity_margin
        - permuted_identity_margin,
        "identity_normal_minus_position_only_margin": normal_identity_margin
        - position_identity_margin,
        "hard_repeat_query_count": hard_repeat_count,
        "hard_repeat_active_edges_per_query": hard_repeat_value(6),
        "hard_repeat_normal_mean_gap": normal_hard_repeat_gap,
        "hard_repeat_normal_win_fraction": hard_repeat_value(1),
        "hard_repeat_permuted_mean_gap": permuted_hard_repeat_gap,
        "hard_repeat_permuted_win_fraction": hard_repeat_value(3),
        "hard_repeat_position_only_mean_gap": position_hard_repeat_gap,
        "hard_repeat_position_only_win_fraction": hard_repeat_value(5),
        "hard_repeat_normal_minus_permuted_gap": normal_hard_repeat_gap
        - permuted_hard_repeat_gap,
        "hard_repeat_normal_minus_position_only_gap": normal_hard_repeat_gap
        - position_hard_repeat_gap,
    }


def p1_context_hard_repeat_evidence_gate(
    *,
    metrics: Mapping[str, float],
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
    minimum_position_only_gap: float,
) -> dict[str, object]:
    """Require direct coherent-repeat evidence before L1 spatial training."""

    required = (
        "hard_repeat_query_count",
        "hard_repeat_normal_mean_gap",
        "hard_repeat_normal_win_fraction",
        "hard_repeat_permuted_mean_gap",
        "hard_repeat_position_only_mean_gap",
    )
    try:
        values = {name: float(metrics[name]) for name in required}
        threshold_win = float(minimum_win_fraction)
        threshold_gap = float(minimum_normal_gap)
        threshold_visual = float(minimum_visual_gap_delta)
        threshold_position = float(minimum_position_only_gap)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("P1 direct hard-repeat gate metrics are incomplete") from error
    if (
        not all(math.isfinite(value) for value in (*values.values(), threshold_win, threshold_gap, threshold_visual, threshold_position))
        or not 0.0 <= threshold_win <= 1.0
        or min(threshold_gap, threshold_visual, threshold_position) < 0.0
    ):
        raise ValueError("P1 direct hard-repeat gate metrics are invalid")
    support_gap = values["hard_repeat_normal_mean_gap"] - values[
        "hard_repeat_permuted_mean_gap"
    ]
    position_gap = values["hard_repeat_normal_mean_gap"] - values[
        "hard_repeat_position_only_mean_gap"
    ]
    checks = {
        "target_free_selected_hard_repeat_coverage": values["hard_repeat_query_count"] >= 1.0,
        "normal_gap": values["hard_repeat_normal_mean_gap"] >= threshold_gap,
        "normal_win_fraction": values["hard_repeat_normal_win_fraction"] >= threshold_win,
        "support_appearance_gap": support_gap >= threshold_visual,
        "descriptor_over_position_gap": position_gap >= threshold_position,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "normal_minus_permuted_gap": float(support_gap),
        "normal_minus_position_only_gap": float(position_gap),
        "thresholds": {
            "minimum_win_fraction": threshold_win,
            "minimum_normal_gap": threshold_gap,
            "minimum_visual_gap_delta": threshold_visual,
            "minimum_position_only_gap": threshold_position,
        },
    }


def p1_context_identity_gate(
    *,
    visual_metrics: Mapping[str, float],
    position_only_metrics: Mapping[str, float],
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
    minimum_position_only_gap: float,
) -> dict[str, object]:
    """Require P1 pose evidence beyond both support and position controls."""

    base = training_gate_decision(
        visual_metrics,
        minimum_win_fraction=float(minimum_win_fraction),
        minimum_normal_gap=float(minimum_normal_gap),
        minimum_visual_gap_delta=float(minimum_visual_gap_delta),
    )
    try:
        visual_gap = float(visual_metrics["normal_mean_correct_minus_hardest_wrong"])
        position_gap = float(position_only_metrics["normal_mean_correct_minus_hardest_wrong"])
        visual_queries = int(visual_metrics["query_count"])
        position_queries = int(position_only_metrics["query_count"])
        threshold = float(minimum_position_only_gap)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("P1 position-only gate metrics are incomplete") from error
    if (
        not math.isfinite(visual_gap)
        or not math.isfinite(position_gap)
        or not math.isfinite(threshold)
        or threshold < 0.0
        or visual_queries < 1
        or visual_queries != position_queries
    ):
        raise ValueError("P1 position-only gate metrics are invalid")
    descriptor_over_position = visual_gap - position_gap
    checks = {
        "normal_and_support_permutation": bool(base["passed"]),
        "descriptor_over_position": descriptor_over_position >= threshold,
    }
    return {
        **base,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "normal_minus_position_only_gap": float(descriptor_over_position),
        "minimum_position_only_gap": float(threshold),
    }


def load_context_identity_l0_initialization_checkpoint(
    *,
    path: Path,
    model: CandidatePoseRGBSpatialLikelihood,
    layout_sha256: str,
    targets_sha256: str,
    candidate_count: int,
    support_view_count: int,
    source_cache_paths: Mapping[str, Path],
    source_image_manifest_sha256: str,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    texture_feature_dim: int,
    hidden_dim: int,
    max_abs_context_log_ratio: float,
    context_windows: Mapping[str, int],
    context_encoder_arch: str,
    allow_diagnostic_ineligible: bool = False,
) -> dict[str, object]:
    """Load one exact P1 L0 checkpoint for a context-only frozen audit.

    This intentionally does not reuse the generic RGB likelihood loader.  An
    L0 checkpoint has a distinct format and only has meaning for the RADIO /
    ALIKE context component; treating it as a complete RGB likelihood would
    silently make random frozen RGB heads appear valid.  A failed inner gate
    is permitted only for an explicitly diagnostic replay.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"P1 L0 context checkpoint is absent: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch releases
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1 L0 context checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or not isinstance(metadata, Mapping)
        or not isinstance(state_dict, Mapping)
        or metadata.get("format") != CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or metadata.get("appearance_control_geometry_fixed") is not True
        or metadata.get("checkpoint_selection_policy")
        != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or metadata.get("inner_validation_used_for_model_selection") is not False
    ):
        raise ValueError("P1 L0 context checkpoint is not target-free compatible")
    excluded = metadata.get("encoder_excludes")
    required_excluded = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
        "rgb_patch",
    }
    if (
        not isinstance(excluded, Sequence)
        or isinstance(excluded, (str, bytes))
        or not required_excluded.issubset({str(value) for value in excluded})
    ):
        raise ValueError("P1 L0 context checkpoint exclusion contract is incomplete")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    training = metadata.get("training")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(config, Mapping)
        or not isinstance(training, Mapping)
    ):
        raise ValueError("P1 L0 context checkpoint lacks lineage/config/training")
    if (
        str(lineage.get("rgb_spatial_layout_sha256", "")) != str(layout_sha256)
        or str(lineage.get("training_targets_sha256", "")) != str(targets_sha256)
    ):
        raise ValueError("P1 L0 context checkpoint lineage is stale")
    expected_cache_paths = {
        "radio_final": "radio_final_context_cache_sha256",
        "radio_intermediate": "radio_intermediate_context_cache_sha256",
        "alike": "alike_spatial_context_cache_sha256",
    }
    if set(source_cache_paths) != set(expected_cache_paths):
        raise ValueError("P1 L0 context checkpoint source-cache set is incomplete")
    for source_name, lineage_name in expected_cache_paths.items():
        if str(lineage.get(lineage_name, "")) != file_sha256_short(
            Path(source_cache_paths[source_name])
        ):
            raise ValueError("P1 L0 context checkpoint source cache lineage differs")
    if str(lineage.get("source_image_manifest_sha256", "")) != str(
        source_image_manifest_sha256
    ):
        raise ValueError("P1 L0 context checkpoint source image manifest differs")
    expected_floats = {
        "search_radius_px": float(search_radius_px),
        "context_radius_px": float(context_radius_px),
        "step_px": float(step_px),
        "max_abs_context_log_ratio": float(max_abs_context_log_ratio),
    }
    for name, expected in expected_floats.items():
        try:
            observed = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("P1 L0 context checkpoint config is incomplete") from error
        if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("P1 L0 context checkpoint config differs")
    if (
        config.get("context_only") is not True
        or int(config.get("texture_feature_dim", -1)) != int(texture_feature_dim)
        or int(config.get("hidden_dim", -1)) != int(hidden_dim)
        or int(config.get("candidate_count", -1)) != int(candidate_count)
        or int(config.get("support_view_count", -1)) != int(support_view_count)
    ):
        raise ValueError("P1 L0 context checkpoint model config differs")
    try:
        observed_windows = resolve_candidate_pose_rgb_spatial_context_windows(
            config.get("context_windows")
        )
        expected_windows = resolve_candidate_pose_rgb_spatial_context_windows(context_windows)
        observed_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            config.get("context_encoder_arch", "conv_v1")
        )
        expected_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            context_encoder_arch
        )
    except (TypeError, ValueError) as error:
        raise ValueError("P1 L0 context checkpoint context configuration is invalid") from error
    if (
        observed_windows != expected_windows
        or observed_arch != expected_arch
        or observed_arch != "absolute_cross_attention_v3"
    ):
        raise ValueError("P1 L0 context checkpoint context configuration differs")
    inner = training.get("inner_validation")
    gate = inner.get("gate") if isinstance(inner, Mapping) else None
    selection = inner.get("checkpoint_selection") if isinstance(inner, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or not isinstance(selection, Mapping)
        or selection.get("policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or selection.get("inner_validation_used_for_model_selection") is not False
        or int(selection.get("selected_epoch", -1)) != int(inner.get("selected_epoch", -1))
        or int(inner.get("selected_epoch", -1)) < 1
    ):
        raise ValueError("P1 L0 context checkpoint inner gate is incomplete")
    passed = gate.get("passed") is True
    if metadata.get("l1_spatial_training_allowed") is not passed:
        raise ValueError("P1 L0 context checkpoint gate metadata is inconsistent")
    if not passed and not bool(allow_diagnostic_ineligible):
        raise ValueError("P1 L0 context checkpoint inner gate did not pass")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("P1 L0 context checkpoint state dict is incompatible") from error
    return {
        "kind": "p1_context_identity_l0_context_only",
        "path": str(checkpoint_path),
        "sha256": file_sha256_short(checkpoint_path),
        "selected_epoch": int(inner["selected_epoch"]),
        "target_free_checkpoint_gate_passed": bool(passed),
        "eligible_for_p1_finetune": bool(passed),
        "diagnostic_ineligible_override": bool(not passed),
        "loaded_parameter_scope": "full_target_free_model_context_component_only",
    }


def train_candidate_pose_context_identity_l0(args: argparse.Namespace) -> dict[str, object]:
    """Run the P1-distribution-matched L0 identity training protocol."""

    context_windows = _validate_args(args)
    context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
        args.context_encoder_arch
    )
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_context_identity_l0.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if not bool(args.force) and _distributed_output_conflict(
            state=state, paths=(checkpoint_path, history_path, summary_path)
        ):
            raise FileExistsError("refusing to overwrite P1 L0 context-identity output")
        if state.enabled:
            distributed.barrier()
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - older torch releases
            pass

        layout_path = Path(args.rgb_spatial_layout)
        target_path = Path(args.training_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(target_path)
        validate_training_layout_and_targets(
            layout=layout,
            targets=targets,
            layout_sha256=file_sha256_short(layout_path),
        )
        target_radius = float(targets.metadata["spatial_search_radius_px"])
        search_radius = target_radius if args.search_radius_px is None else float(args.search_radius_px)
        if not math.isclose(search_radius, target_radius, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("P1 L0 search radius differs from train-only targets")
        groups = build_train_query_groups(layout=layout, targets=targets)
        train_query_ids, validation_query_ids = _partition_train_queries_for_inner_validation(
            query_ids=tuple(sorted(groups)),
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        context_cache_paths = {
            "radio_final": Path(args.radio_final_context_cache),
            "radio_intermediate": Path(args.radio_intermediate_context_cache),
            "alike": Path(args.alike_spatial_context_cache),
        }
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("P1 L0 context identity requires common image dimensions")
        image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)

        hard_repeat_groups: dict[str, object] = {}
        hard_repeat_path = Path(args.hard_repeat_targets) if str(args.hard_repeat_targets).strip() else None
        if hard_repeat_path is not None:
            hard_repeat_groups = build_hard_repeat_query_targets(
                layout=layout,
                targets=targets,
                hard_repeat_targets=load_candidate_pose_rgb_spatial_hard_repeat_targets(
                    hard_repeat_path
                ),
                layout_sha256=file_sha256_short(layout_path),
                targets_sha256=file_sha256_short(target_path),
            )
        if float(args.hard_repeat_context_loss_weight) > 0.0 and not hard_repeat_groups:
            raise ValueError("P1 L0 hard-repeat context objective has no train targets")

        model = CandidatePoseRGBSpatialLikelihood(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            search_radius_px=search_radius,
            context_radius_px=float(args.context_radius_px),
            step_px=float(args.step_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=True,
            context_windows=context_windows,
            context_encoder_arch=context_encoder_arch,
        )
        context_initialization: dict[str, object] | None = None
        if str(args.context_observation_pretrain_checkpoint).strip():
            context_initialization = load_context_observation_pretrain_initialization_checkpoint(
                path=Path(args.context_observation_pretrain_checkpoint),
                model=model,
                source_cache_paths=context_cache_paths,
                source_image_manifest_sha256=str(
                    sources[0].metadata.get("source_image_manifest_sha256", "")
                ),
                search_radius_px=search_radius,
                context_radius_px=float(args.context_radius_px),
                step_px=float(args.step_px),
                texture_feature_dim=int(args.texture_feature_dim),
                hidden_dim=int(args.hidden_dim),
                max_abs_context_log_ratio=float(args.max_abs_context_log_ratio),
                context_windows=context_windows,
                context_encoder_arch=context_encoder_arch,
                observation_pairs_path=Path(args.context_observation_pairs),
                expected_pretrain_train_query_ids=train_query_ids,
                expected_pretrain_validation_query_ids=validation_query_ids,
            )
        for module in (model.texture_encoder, model.spatial_residual_head, model.non_dustbin_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if str(args.trainable_context_scope) == "head_only":
            for parameter in model.context_encoders.parameters():
                parameter.requires_grad_(False)
        else:
            for parameter in model.context_encoders.parameters():
                parameter.requires_grad_(True)
        for parameter in model.context_identity_head.parameters():
            parameter.requires_grad_(True)
        model = model.to(state.device)
        if state.enabled:
            model_for_train: torch.nn.Module = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        core_model = model_for_train.module if state.enabled else model_for_train
        assert isinstance(core_model, CandidatePoseRGBSpatialLikelihood)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model_for_train.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        final_metrics: dict[str, float] | None = None
        final_position_only_metrics: dict[str, float] | None = None
        final_direct_evidence_metrics: dict[str, float] | None = None
        final_state_dict: dict[str, torch.Tensor] | None = None
        final_epoch = -1
        history: list[dict[str, object]] = []
        started = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            local_query_ids = _rank_query_ids(
                query_ids=train_query_ids, state=state, seed=int(args.seed), epoch=int(epoch)
            )
            totals = torch.zeros((15,), dtype=torch.float64, device=state.device)
            epoch_started = time.time()
            for step, query_id in enumerate(local_query_ids):
                group = groups[str(query_id)]
                hard_targets = hard_repeat_groups.get(str(query_id))
                positions = select_identity_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + int(epoch) * 100003 + int(step),
                    required_source_point_ids=(
                        () if hard_targets is None else hard_targets.source_point_ids
                    ),
                )
                batch = _query_batch_from_group(
                    group=group,
                    complete_runtime=complete_runtime,
                    point_positions=positions,
                    device=state.device,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(runtime=batch.runtime, context_only=True)
                    identity_loss, identity_metrics = context_identity_cross_entropy_loss(
                        runtime=batch.runtime,
                        prediction=prediction,
                        target_observed=batch.spatial_target_observed,
                    )
                    permutation_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                    permutation_metrics = {
                        "context_identity_permutation_active_rows": 0.0,
                        "context_identity_permutation_mean_gap": 0.0,
                        "context_identity_permutation_win_fraction": 0.0,
                        "context_identity_permutation_margin_loss": 0.0,
                    }
                    if float(args.identity_support_permutation_loss_weight) > 0.0:
                        permuted_runtime = permute_runtime_support_image_appearance_only(
                            batch.runtime, shift=int(args.identity_support_permutation_shift)
                        )
                        permuted_prediction = model_for_train(
                            runtime=permuted_runtime, context_only=True
                        )
                        _assert_geometry_fixed_support_image_control(
                            runtime=batch.runtime,
                            permuted_runtime=permuted_runtime,
                            normal_prediction=prediction,
                            permuted_prediction=permuted_prediction,
                        )
                        permutation_loss, permutation_metrics = (
                            context_identity_support_permutation_margin_loss(
                                runtime=batch.runtime,
                                prediction=prediction,
                                permuted_runtime=permuted_runtime,
                                permuted_prediction=permuted_prediction,
                                target_observed=batch.spatial_target_observed,
                                margin=float(args.identity_support_permutation_margin),
                            )
                        )
                    hard_repeat_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                    hard_repeat_metrics = {
                        "hard_repeat_context_active_edges": 0.0,
                        "hard_repeat_context_mean_positive_minus_negative": 0.0,
                        "hard_repeat_context_correct_win_fraction": 0.0,
                        "hard_repeat_context_margin_loss": 0.0,
                    }
                    if float(args.hard_repeat_context_loss_weight) > 0.0:
                        hard_batch = (
                            None
                            if hard_targets is None
                            else _hard_repeat_batch_from_group(
                                hard_targets=hard_targets,
                                group=group,
                                point_positions=positions,
                                device=state.device,
                                max_edges=int(args.max_hard_repeat_edges_per_query),
                                seed=int(args.seed) + int(epoch) * 100003 + int(step),
                            )
                        )
                        hard_repeat_loss, hard_repeat_metrics = (
                            coherent_hard_repeat_context_margin_loss(
                                runtime=batch.runtime,
                                prediction=prediction,
                                hard_batch=hard_batch,
                                margin=float(args.hard_repeat_context_margin),
                                missing_edge_log_likelihood_ratio=0.0,
                            )
                        )
                    pose_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                    pose_metrics = {
                        "query_mean_correct_minus_hardest_wrong": 0.0,
                        "query_correct_win_fraction": 0.0,
                    }
                    if float(args.pose_context_loss_weight) > 0.0:
                        pose_loss, pose_metrics = context_pose_margin_loss(
                            runtime=batch.runtime,
                            prediction=prediction,
                            batch=batch,
                            pose_margin=float(args.pose_margin),
                        )
                    loss = (
                        float(args.identity_loss_weight) * identity_loss
                        + float(args.identity_support_permutation_loss_weight) * permutation_loss
                        + float(args.hard_repeat_context_loss_weight) * hard_repeat_loss
                        + float(args.pose_context_loss_weight) * pose_loss
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model_for_train.parameters(), float(args.gradient_clip_norm)
                )
                scaler.step(optimizer)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(loss.detach().item()),
                        float(identity_loss.detach().item()),
                        float(identity_metrics["context_identity_top1_accuracy"]),
                        float(identity_metrics["context_identity_mean_margin"]),
                        float(identity_metrics["context_identity_active_rows"]),
                        float(permutation_metrics["context_identity_permutation_margin_loss"]),
                        float(permutation_metrics["context_identity_permutation_mean_gap"]),
                        float(permutation_metrics["context_identity_permutation_win_fraction"]),
                        float(hard_repeat_metrics["hard_repeat_context_margin_loss"]),
                        float(hard_repeat_metrics["hard_repeat_context_mean_positive_minus_negative"]),
                        float(hard_repeat_metrics["hard_repeat_context_correct_win_fraction"]),
                        float(pose_loss.detach().item()),
                        float(pose_metrics["query_mean_correct_minus_hardest_wrong"]),
                        float(pose_metrics["query_correct_win_fraction"]),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            if state.enabled:
                distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
            global_steps = int(len(local_query_ids) * state.world_size)
            validation = _context_only_validation(
                model=model_for_train,
                layout=layout,
                groups=groups,
                complete_runtime=complete_runtime,
                query_ids=validation_query_ids,
                image_size=image_size,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                pose_margin=float(args.pose_margin),
                amp_enabled=amp_enabled,
            )
            position_only_validation = _context_only_validation(
                model=model_for_train,
                layout=layout,
                groups=groups,
                complete_runtime=complete_runtime,
                query_ids=validation_query_ids,
                image_size=image_size,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                pose_margin=float(args.pose_margin),
                amp_enabled=amp_enabled,
                context_appearance_mode="position_only",
            )
            direct_evidence_validation = _context_only_direct_evidence_validation(
                model=model_for_train,
                layout=layout,
                groups=groups,
                complete_runtime=complete_runtime,
                hard_repeat_groups=hard_repeat_groups,
                query_ids=validation_query_ids,
                image_size=image_size,
                state=state,
                selector_policy=str(args.validation_selector_policy),
                selector_point_budget=int(args.validation_selector_point_budget),
                selector_grid_rows=int(args.validation_selector_grid_rows),
                selector_grid_columns=int(args.validation_selector_grid_columns),
                max_hard_repeat_edges_per_query=0,
                amp_enabled=amp_enabled,
            )
            if state.rank == 0:
                # The train-query fold is a gate and telemetry only. It must
                # never choose an epoch, which would make its later gate value
                # optimistically selected.
                final_metrics = dict(validation)
                final_position_only_metrics = dict(position_only_validation)
                final_direct_evidence_metrics = dict(direct_evidence_validation)
                final_epoch = int(epoch + 1)
                final_state_dict = _model_state_cpu(core_model)
                record = {
                    "epoch": int(epoch + 1),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_identity_cross_entropy": float((totals[1] / global_steps).item()),
                    "train_identity_top1_accuracy": float((totals[2] / global_steps).item()),
                    "train_identity_mean_margin": float((totals[3] / global_steps).item()),
                    "train_identity_active_rows_per_step": float((totals[4] / global_steps).item()),
                    "train_permutation_margin_loss": float((totals[5] / global_steps).item()),
                    "train_permutation_mean_gap": float((totals[6] / global_steps).item()),
                    "train_permutation_win_fraction": float((totals[7] / global_steps).item()),
                    "train_hard_repeat_margin_loss": float((totals[8] / global_steps).item()),
                    "train_hard_repeat_mean_positive_minus_negative": float(
                        (totals[9] / global_steps).item()
                    ),
                    "train_hard_repeat_win_fraction": float((totals[10] / global_steps).item()),
                    "train_pose_margin_loss": float((totals[11] / global_steps).item()),
                    "train_pose_mean_correct_minus_hardest_wrong": float(
                        (totals[12] / global_steps).item()
                    ),
                    "train_pose_correct_win_fraction": float((totals[13] / global_steps).item()),
                    "global_query_steps": int(global_steps),
                    "epoch_seconds": float(time.time() - epoch_started),
                    **{f"inner_{key}": value for key, value in validation.items()},
                    **{
                        f"inner_position_only_{key}": value
                        for key, value in position_only_validation.items()
                    },
                    **{
                        f"inner_direct_{key}": value
                        for key, value in direct_evidence_validation.items()
                    },
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()

        if state.rank == 0:
            if (
                final_metrics is None
                or final_position_only_metrics is None
                or final_direct_evidence_metrics is None
                or final_state_dict is None
                or final_epoch < 1
            ):
                raise RuntimeError("P1 L0 context identity did not retain final-epoch telemetry")
            pose_gate = p1_context_identity_gate(
                visual_metrics=final_metrics,
                position_only_metrics=final_position_only_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
                minimum_position_only_gap=float(args.minimum_position_only_gap),
            )
            direct_evidence_gate = p1_context_hard_repeat_evidence_gate(
                metrics=final_direct_evidence_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
                minimum_position_only_gap=float(args.minimum_position_only_gap),
            )
            gate = {
                "passed": bool(pose_gate["passed"] and direct_evidence_gate["passed"]),
                "checks": {
                    "pose_level_context_evidence": bool(pose_gate["passed"]),
                    "direct_candidate_hard_repeat_evidence": bool(
                        direct_evidence_gate["passed"]
                    ),
                },
                "pose_gate": pose_gate,
                "direct_candidate_evidence_gate": direct_evidence_gate,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "architecture": "p1_top20_two_view_absolute_context_only_identity_l0_v2",
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "l1_spatial_training_allowed": bool(gate["passed"]),
                "raw_scores_must_not_feed_pnp": True,
                "appearance_control_geometry_fixed": True,
                "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
                "inner_validation_used_for_model_selection": False,
                "encoder_inputs": [
                    "query_image_xy",
                    "fixed_top20_support_image_xy",
                    "full_2d_radio_final_context_crop",
                    "full_2d_radio_intermediate_context_crop",
                    "full_2d_alike_context_crop",
                ],
                "encoder_excludes": [
                    "pose_matrix",
                    "projection_offset",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "candidate_rank",
                    "coarse_score",
                    "rgb_patch",
                ],
                "config": {
                    "search_radius_px": float(search_radius),
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_context_log_ratio": float(args.max_abs_context_log_ratio),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "context_windows": dict(context_windows),
                    "context_encoder_arch": context_encoder_arch,
                    "context_only": True,
                    "candidate_count": int(layout.candidate_count),
                    "support_view_count": int(layout.support_view_count),
                    "trainable_context_scope": str(args.trainable_context_scope),
                    "validation_selector": {
                        "policy": str(args.validation_selector_policy),
                        "point_budget": int(args.validation_selector_point_budget),
                        "grid_rows": int(args.validation_selector_grid_rows),
                        "grid_columns": int(args.validation_selector_grid_columns),
                        "target_free_static_only": True,
                    },
                },
                "training": {
                    "objective": OBJECTIVE,
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "identity_loss_weight": float(args.identity_loss_weight),
                    "identity_support_permutation_loss_weight": float(
                        args.identity_support_permutation_loss_weight
                    ),
                    "identity_support_permutation_margin": float(
                        args.identity_support_permutation_margin
                    ),
                    "identity_support_permutation_training_shift": int(
                        args.identity_support_permutation_shift
                    ),
                    "support_appearance_control": (
                        "support_image_index_derangement_keep_support_xy_and_mixture_mass_v2"
                    ),
                    "hard_repeat_context_loss_weight": float(
                        args.hard_repeat_context_loss_weight
                    ),
                    "hard_repeat_context_margin": float(args.hard_repeat_context_margin),
                    "pose_context_loss_weight": float(args.pose_context_loss_weight),
                    "context_observation_initialization": context_initialization,
                    "trainable_context_scope": str(args.trainable_context_scope),
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only_disjoint_query_images",
                        "selected_epoch": int(final_epoch),
                        "final_epoch_metrics": final_metrics,
                        "position_only_final_epoch_metrics": final_position_only_metrics,
                        "direct_candidate_evidence_final_epoch_metrics": final_direct_evidence_metrics,
                        "checkpoint_selection": fixed_final_epoch_checkpoint_selection(
                            epochs=int(args.epochs)
                        ),
                        "support_permutation_control_shift": 1,
                        "selector": {
                            "policy": str(args.validation_selector_policy),
                            "point_budget": int(args.validation_selector_point_budget),
                            "grid_rows": int(args.validation_selector_grid_rows),
                            "grid_columns": int(args.validation_selector_grid_columns),
                            "target_free_static_only": True,
                        },
                        "selection_before_target_join": True,
                        "observed_target_preserving_sampler_used": False,
                        "position_only_control": (
                            "zero_frozen_descriptors_keep_absolute_coordinates_v1"
                        ),
                        "gate": gate,
                    },
                },
                "lineage": {
                    "rgb_spatial_layout": str(layout_path.resolve()),
                    "rgb_spatial_layout_sha256": file_sha256_short(layout_path),
                    "training_targets_sha256": file_sha256_short(target_path),
                    "hard_repeat_targets_sha256": (
                        "" if hard_repeat_path is None else file_sha256_short(hard_repeat_path)
                    ),
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
                    "context_observation_pretrain_checkpoint_sha256": (
                        ""
                        if context_initialization is None
                        else str(context_initialization["sha256"])
                    ),
                },
            }
            torch.save(
                {"format": CHECKPOINT_FORMAT, "state_dict": final_state_dict, "metadata": metadata},
                checkpoint_path,
            )
            summary = {
                "stage": "train_candidate_pose_context_identity_l0",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - started),
                "history": history,
                "checkpoint_selection": {
                    **fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs)),
                    "inner_validation": final_metrics,
                    "inner_position_only_validation": final_position_only_metrics,
                    "inner_direct_candidate_evidence": final_direct_evidence_metrics,
                    "gate": gate,
                },
                "protocol": {
                    "train_only_identity_and_hard_repeat_targets": True,
                    "runtime_checkpoint_target_free": True,
                    "context_only_no_rgb_crop_or_encoder": True,
                    "position_only_control": True,
                    "support_appearance_control_geometry_fixed": True,
                    "inner_validation_used_for_model_selection": False,
                    "hard_repeat_source_points_retained_before_random_fill": True,
                    "training_sampler_may_use_train_targets": True,
                    "inner_validation_selector_target_free": True,
                    "inner_validation_observed_target_sampler_used": False,
                    "validation_or_test_labels_used_by_fit": False,
                    "pnp_or_external_pose_not_run": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                },
            }
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return {"checkpoint": str(checkpoint_path), "rank": int(state.rank)}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = train_candidate_pose_context_identity_l0(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
