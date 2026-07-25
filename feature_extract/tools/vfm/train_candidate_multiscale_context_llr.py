"""Train one independently calibrated multiscale context likelihood.

The frozen P1 layout supplies only query anchors, global top-L candidate
tracks, and fixed maplet support observations.  Correct/wrong pose projections
and hard-repeat identities are joined *after* the target-free visual forward.
This keeps the learned scalar a real appearance LLR rather than a shortcut for
candidate rank, pose geometry, target identity, or image coordinates.

Each invocation trains exactly one source (RADIO-final, RADIO-intermediate, or
ALIKE).  It is deliberately a diagnostic stage: a checkpoint is not promoted
to held-out pose ranking unless it passes normal-support, support-deranged,
and zero-visual train-only controls on a query-disjoint inner fold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
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
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _select_group_points,
    _source_table,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_multiscale_context_llr import (
    CANDIDATE_MULTISCALE_CONTEXT_LLR_FORMAT,
    CANDIDATE_MULTISCALE_CONTEXT_SOURCES,
    CandidateMultiscaleContextLLR,
    CandidateMultiscaleContextLLRPrediction,
    score_fixed_global_topl_phase_pose,
    source_candidate_log_likelihood_ratios,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    query_grouped_pose_margin_loss,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CandidatePoseRGBSpatialHardRepeatTargets,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_runtime_support_image_appearance_only,
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


CHECKPOINT_FORMAT = "candidate_multiscale_context_llr_checkpoint_v2"
FIXED_FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"
_STATIC_SELECTOR_POLICIES = tuple(
    policy for policy in TARGET_FREE_SELECTOR_POLICIES if "rgb" not in str(policy)
)


def fixed_final_epoch_checkpoint_selection(*, epochs: int) -> dict[str, object]:
    """Declare the non-data-dependent checkpoint selection policy."""

    if int(epochs) <= 0:
        raise ValueError("fixed-final-epoch context checkpoint selection requires positive epochs")
    return {
        "policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "selected_epoch": int(epochs),
        "inner_validation_used_for_model_selection": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", choices=CANDIDATE_MULTISCALE_CONTEXT_SOURCES, required=True)
    parser.add_argument("--radio-final-context-window", type=int, default=5)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=9)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument("--radio-final-phase-shift-radius", type=int, default=1)
    parser.add_argument("--radio-intermediate-phase-shift-radius", type=int, default=2)
    parser.add_argument("--alike-phase-shift-radius", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument("--edge-chunk-size", type=int, default=1024)
    parser.add_argument("--max-points-per-query", type=int, default=32)
    parser.add_argument(
        "--validation-selector-policy",
        choices=_STATIC_SELECTOR_POLICIES,
        default="coarse_margin",
        help="Target-free point policy used before inner-fold targets are joined.",
    )
    parser.add_argument("--validation-selector-point-budget", type=int, default=32)
    parser.add_argument("--validation-selector-grid-rows", type=int, default=4)
    parser.add_argument("--validation-selector-grid-columns", type=int, default=4)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pose-margin", type=float, default=0.15)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.15)
    parser.add_argument("--appearance-control-margin", type=float, default=0.05)
    parser.add_argument("--identity-loss-weight", type=float, default=0.25)
    parser.add_argument("--hard-repeat-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--hard-repeat-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--local-radius-px", type=float, default=8.0)
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=256)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=1024.0,
        help="Conservative FP16 loss scale for grid-sampled 1,280-D context features.",
    )
    parser.add_argument("--minimum-pose-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-pose-gap", type=float, default=0.05)
    parser.add_argument("--minimum-pose-permutation-delta", type=float, default=0.05)
    parser.add_argument("--minimum-pose-zero-visual-delta", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-hard-repeat-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-repeat-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-permutation-delta", type=float, default=0.05)
    parser.add_argument("--minimum-hard-repeat-zero-visual-delta", type=float, default=0.05)
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _context_windows(args: argparse.Namespace) -> dict[str, int]:
    return {
        "radio_final": int(args.radio_final_context_window),
        "radio_intermediate": int(args.radio_intermediate_context_window),
        "alike": int(args.alike_context_window),
    }


def _phase_shift_radii(args: argparse.Namespace) -> dict[str, int]:
    return {
        "radio_final": int(args.radio_final_phase_shift_radius),
        "radio_intermediate": int(args.radio_intermediate_phase_shift_radius),
        "alike": int(args.alike_phase_shift_radius),
    }


def source_weights(source: str) -> dict[str, float]:
    """Select exactly one independently calibrated visual family."""

    selected = str(source)
    if selected not in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
        raise ValueError("multiscale context source is invalid")
    return {
        name: 1.0 if name == selected else 0.0
        for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES
    }


def configure_single_source_trainable_parameters(
    *, model: CandidateMultiscaleContextLLR, source: str
) -> tuple[str, ...]:
    """Freeze all but one source head before DDP registers parameters."""

    if not isinstance(model, CandidateMultiscaleContextLLR):
        raise ValueError("multiscale context trainable scope requires its model")
    selected = str(source)
    if selected not in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
        raise ValueError("multiscale context trainable source is invalid")
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(f"heads.{selected}.")
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("multiscale context selected source has no trainable head")
    return tuple(trainable)


def _source_candidate_scores(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscaleContextLLRPrediction,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one source's fixed-view candidate LLR and visual availability."""

    name = str(source)
    if name not in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
        raise ValueError("multiscale context score source is invalid")
    values = source_candidate_log_likelihood_ratios(prediction=prediction, runtime=runtime)[name]
    active = runtime.to(values.device)
    edge_usable = prediction.source_edge_usable[name]
    usable = torch.any(
        edge_usable & (active.candidate_view_weights > 0.0), dim=2
    )
    if values.shape != usable.shape or not torch.isfinite(values).all():
        raise RuntimeError("multiscale context candidate score construction drifted")
    return values, usable


def source_candidate_plus_null_logits(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscaleContextLLRPrediction,
    source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join target-free source LLRs to immutable candidate/null prior mass."""

    scores, usable = _source_candidate_scores(
        runtime=runtime, prediction=prediction, source=source
    )
    active = runtime.to(scores.device)
    candidate_prior = active.candidate_probabilities.to(dtype=scores.dtype)
    null_prior = active.null_probabilities.to(dtype=scores.dtype)
    candidate_logits = torch.where(
        (candidate_prior > 0.0) & usable,
        torch.log(candidate_prior.clamp_min(torch.finfo(scores.dtype).tiny)) + scores,
        torch.full_like(scores, -torch.inf),
    )
    null_logits = torch.where(
        null_prior > 0.0,
        torch.log(null_prior.clamp_min(torch.finfo(scores.dtype).tiny)),
        torch.full_like(null_prior, -torch.inf),
    )
    logits = torch.cat((candidate_logits, null_logits[:, None]), dim=1)
    if not torch.all(torch.isfinite(logits) | torch.isneginf(logits)):
        raise RuntimeError("multiscale context candidate/null logits are non-finite")
    return logits, usable


def registered_identity_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscaleContextLLRPrediction,
    source: str,
    observed_candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train exact registered rows only; null remains a fixed mixture term."""

    target = torch.as_tensor(
        observed_candidate_mask,
        dtype=torch.bool,
        device=prediction.source_edge_log_likelihood_ratios[str(source)].device,
    )
    logits, usable = source_candidate_plus_null_logits(
        runtime=runtime, prediction=prediction, source=source
    )
    if target.shape != (logits.shape[0], logits.shape[1] - 1) or torch.any(
        target.sum(dim=1) > 1
    ):
        raise ValueError("registered identity target does not match fixed candidate layout")
    points, candidates = torch.nonzero(target, as_tuple=True)
    if len(points) == 0:
        return logits.sum() * 0.0, {"active": 0.0, "loss": 0.0, "top1": 0.0}
    active = usable[points, candidates]
    if not bool(active.any()):
        return logits.sum() * 0.0, {"active": 0.0, "loss": 0.0, "top1": 0.0}
    selected_logits = logits[points[active]]
    selected_targets = candidates[active]
    loss = F.cross_entropy(selected_logits, selected_targets)
    return loss, {
        "active": float(len(selected_targets)),
        "loss": float(loss.detach().item()),
        "top1": float(
            (selected_logits.detach().argmax(dim=1) == selected_targets)
            .to(dtype=torch.float32)
            .mean()
            .item()
        ),
    }


def _grouped_hard_repeat_gaps(
    *,
    scores: torch.Tensor,
    common_usable: torch.Tensor,
    hard_batch: HardRepeatBatch | None,
) -> torch.Tensor:
    """Collapse multi-negative rows to one strongest wrong candidate per point."""

    if hard_batch is None:
        return scores.new_empty((0,))
    points = hard_batch.point_indices.to(device=scores.device, dtype=torch.long)
    positive = hard_batch.positive_candidate_indices.to(device=scores.device, dtype=torch.long)
    negative = hard_batch.negative_candidate_indices.to(device=scores.device, dtype=torch.long)
    usable = torch.as_tensor(common_usable, dtype=torch.bool, device=scores.device)
    if (
        scores.ndim != 2
        or usable.shape != scores.shape
        or points.ndim != 1
        or positive.shape != points.shape
        or negative.shape != points.shape
        or torch.any(points < 0)
        or torch.any(points >= len(scores))
        or torch.any(positive < 0)
        or torch.any(positive >= scores.shape[1])
        or torch.any(negative < 0)
        or torch.any(negative >= scores.shape[1])
    ):
        raise ValueError("multiscale context hard-repeat rows are invalid")
    active = usable[points, positive] & usable[points, negative]
    if not bool(active.any()):
        return scores.new_empty((0,))
    active_points = points[active]
    active_positive = positive[active]
    active_negative = negative[active]
    group_keys = active_points * int(scores.shape[1]) + active_positive
    gaps: list[torch.Tensor] = []
    for key in torch.unique(group_keys, sorted=True):
        members = group_keys == key
        first = torch.nonzero(members, as_tuple=False)[0, 0]
        point = active_points[first]
        correct = scores[point, active_positive[first]]
        wrong = scores[point, active_negative[members]].amax()
        gaps.append(correct - wrong)
    return torch.stack(gaps) if gaps else scores.new_empty((0,))


def hard_repeat_margin_loss(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscaleContextLLRPrediction,
    source: str,
    hard_batch: HardRepeatBatch | None,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train source appearance against current coherent-repeat candidates."""

    scores, usable = _source_candidate_scores(
        runtime=runtime, prediction=prediction, source=source
    )
    gaps = _grouped_hard_repeat_gaps(
        scores=scores, common_usable=usable, hard_batch=hard_batch
    )
    if len(gaps) == 0:
        return scores.sum() * 0.0, {"active": 0.0, "loss": 0.0, "gap": 0.0, "win": 0.0}
    loss = F.softplus(torch.as_tensor(float(margin), device=gaps.device) - gaps).mean()
    return loss, {
        "active": float(len(gaps)),
        "loss": float(loss.detach().item()),
        "gap": float(gaps.detach().mean().item()),
        "win": float((gaps.detach() > 0.0).to(dtype=torch.float32).mean().item()),
    }


def hard_repeat_control_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: CandidateMultiscaleContextLLRPrediction,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: CandidateMultiscaleContextLLRPrediction,
    zero_prediction: CandidateMultiscaleContextLLRPrediction,
    source: str,
    hard_batch: HardRepeatBatch | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score hard-repeat groups on one common visual-availability set."""

    normal_scores, normal_usable = _source_candidate_scores(
        runtime=runtime, prediction=normal_prediction, source=source
    )
    permuted_scores, permuted_usable = _source_candidate_scores(
        runtime=permuted_runtime, prediction=permuted_prediction, source=source
    )
    zero_scores, zero_usable = _source_candidate_scores(
        runtime=runtime, prediction=zero_prediction, source=source
    )
    common = normal_usable & permuted_usable & zero_usable
    return (
        _grouped_hard_repeat_gaps(
            scores=normal_scores, common_usable=common, hard_batch=hard_batch
        ),
        _grouped_hard_repeat_gaps(
            scores=permuted_scores, common_usable=common, hard_batch=hard_batch
        ),
        _grouped_hard_repeat_gaps(
            scores=zero_scores, common_usable=common, hard_batch=hard_batch
        ),
    )


def pose_margin_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateMultiscaleContextLLRPrediction,
    correct_projection_offsets_xy: torch.Tensor,
    correct_projection_valid: torch.Tensor,
    wrong_projection_offsets_xy: torch.Tensor,
    wrong_projection_valid: torch.Tensor,
    source: str,
    local_radius_px: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Rank the correct pose above its frozen coherent-wrong pose pool."""

    weights = source_weights(str(source))
    correct = score_fixed_global_topl_phase_pose(
        prediction=prediction,
        runtime=runtime,
        candidate_projection_offsets_xy=torch.as_tensor(
            correct_projection_offsets_xy, device=prediction.source_edge_log_likelihood_ratios[str(source)].device
        ).unsqueeze(0),
        candidate_projection_valid=torch.as_tensor(
            correct_projection_valid, device=prediction.source_edge_log_likelihood_ratios[str(source)].device
        ).unsqueeze(0),
        local_radius_px=float(local_radius_px),
        source_weights=weights,
    ).pose_log_likelihood_ratios
    wrong = score_fixed_global_topl_phase_pose(
        prediction=prediction,
        runtime=runtime,
        candidate_projection_offsets_xy=wrong_projection_offsets_xy,
        candidate_projection_valid=wrong_projection_valid,
        local_radius_px=float(local_radius_px),
        source_weights=weights,
    ).pose_log_likelihood_ratios
    loss, metrics = query_grouped_pose_margin_loss(
        correct_scores=correct,
        coherent_wrong_scores=wrong.reshape(1, -1),
        margin=float(margin),
    )
    raw_gap = correct - wrong.max().reshape(1)
    if raw_gap.shape != (1,) or not torch.isfinite(raw_gap).all():
        raise RuntimeError("multiscale context pose gap construction drifted")
    return loss, {
        "loss": float(loss.detach().item()),
        "gap": float(metrics["query_mean_correct_minus_hardest_wrong"]),
        "win": float(metrics["query_correct_win_fraction"]),
    }, raw_gap


def appearance_control_margin_loss(
    *, normal_gaps: torch.Tensor, permuted_gaps: torch.Tensor, zero_gaps: torch.Tensor, margin: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Demand appearance evidence beyond both deranged and zero-content controls."""

    normal = torch.as_tensor(normal_gaps)
    permuted = torch.as_tensor(permuted_gaps, dtype=normal.dtype, device=normal.device)
    zero = torch.as_tensor(zero_gaps, dtype=normal.dtype, device=normal.device)
    if (
        normal.ndim != 1
        or permuted.shape != normal.shape
        or zero.shape != normal.shape
        or len(normal) == 0
        or not torch.isfinite(normal).all()
        or not torch.isfinite(permuted).all()
        or not torch.isfinite(zero).all()
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("multiscale context appearance-control inputs are invalid")
    strongest_control = torch.maximum(permuted, zero)
    deltas = normal - strongest_control
    loss = F.softplus(torch.as_tensor(float(margin), device=normal.device) - deltas).mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "normal_minus_permuted": float((normal.detach() - permuted.detach()).mean().item()),
        "normal_minus_zero": float((normal.detach() - zero.detach()).mean().item()),
    }


def _zero_visual_source_scales() -> dict[str, float]:
    return {name: 0.0 for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES}


def _source_visual_scales(source: str) -> dict[str, float]:
    return {name: 1.0 if name == str(source) else 0.0 for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES}


def _select_validation_positions(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    group: TrainQueryGroup,
    args: argparse.Namespace,
    coordinate_image_size: tuple[int, int],
) -> np.ndarray:
    selector_input = selector_input_from_target_free_layout(
        layout=layout, rows=group.layout_rows
    )
    budget = int(args.validation_selector_point_budget)
    if budget <= 0 or budget > selector_input.point_count:
        raise ValueError("multiscale context validation selector budget is invalid")
    return select_target_free_spatial_quota(
        selector_input=selector_input,
        quality_scores=target_free_selector_scores(
            selector_input=selector_input, policy=str(args.validation_selector_policy)
        ),
        point_budget=budget,
        grid_rows=int(args.validation_selector_grid_rows),
        grid_columns=int(args.validation_selector_grid_columns),
        image_size=coordinate_image_size,
    )


@torch.no_grad()
def evaluate_inner_validation(
    *,
    model: nn.Module,
    layout: CandidatePoseRGBSpatialLayout,
    groups: Mapping[str, TrainQueryGroup],
    hard_by_query: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    args: argparse.Namespace,
    coordinate_image_size: tuple[int, int],
    state: _DistributedState,
    amp_enabled: bool,
) -> dict[str, float]:
    """Evaluate only after target-free selection and three visual forwards."""

    if not query_ids:
        raise ValueError("multiscale context inner validation has no queries")
    model.eval()
    # normal / permuted / zero pose loss, gap, win, query count; exact-identity
    # diagnostic; hard-repeat normal/permuted/zero grouped gaps and coverage.
    totals = torch.zeros((21,), dtype=torch.float64, device=state.device)
    source = str(args.source)
    source_scales = _source_visual_scales(source)
    zero_scales = _zero_visual_source_scales()
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("multiscale context validation query is unresolved")
        positions = _select_validation_positions(
            layout=layout,
            group=group,
            args=args,
            coordinate_image_size=coordinate_image_size,
        )
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
                max_edges=0,
                seed=int(args.seed),
            )
        )
        permuted_runtime = permute_runtime_support_image_appearance_only(
            batch.runtime, shift=int(args.support_permutation_shift)
        )
        if (
            torch.equal(batch.runtime.support_image_indices, permuted_runtime.support_image_indices)
            or not torch.equal(batch.runtime.support_xy, permuted_runtime.support_xy)
            or not torch.equal(batch.runtime.support_view_valid, permuted_runtime.support_view_valid)
            or not torch.equal(
                batch.runtime.candidate_view_weights, permuted_runtime.candidate_view_weights
            )
        ):
            raise RuntimeError("multiscale context support appearance control changed geometry or no image content")
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = model(
                runtime=batch.runtime, visual_source_scales=source_scales
            )
            permuted_prediction = model(
                runtime=permuted_runtime, visual_source_scales=source_scales
            )
            zero_prediction = model(
                runtime=batch.runtime, visual_source_scales=zero_scales
            )
            for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
                if not torch.equal(
                    normal_prediction.source_edge_usable[name],
                    permuted_prediction.source_edge_usable[name],
                ):
                    raise RuntimeError("multiscale context appearance control changed visual availability")
            normal_loss, normal, _ = pose_margin_metrics(
                runtime=batch.runtime,
                prediction=normal_prediction,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                source=source,
                local_radius_px=float(args.local_radius_px),
                margin=float(args.pose_margin),
            )
            permuted_loss, permuted, _ = pose_margin_metrics(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                source=source,
                local_radius_px=float(args.local_radius_px),
                margin=float(args.pose_margin),
            )
            zero_loss, zero, _ = pose_margin_metrics(
                runtime=batch.runtime,
                prediction=zero_prediction,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                source=source,
                local_radius_px=float(args.local_radius_px),
                margin=float(args.pose_margin),
            )
            _, identity = registered_identity_loss(
                runtime=batch.runtime,
                prediction=normal_prediction,
                source=source,
                observed_candidate_mask=batch.spatial_target_observed,
            )
            hard_normal, hard_permuted, hard_zero = hard_repeat_control_metrics(
                runtime=batch.runtime,
                normal_prediction=normal_prediction,
                permuted_runtime=permuted_runtime,
                permuted_prediction=permuted_prediction,
                zero_prediction=zero_prediction,
                source=source,
                hard_batch=hard_batch,
            )
        values = torch.tensor(
            [
                float(normal_loss.item()),
                float(normal["gap"]),
                float(normal["win"]),
                float(permuted_loss.item()),
                float(permuted["gap"]),
                float(permuted["win"]),
                float(zero_loss.item()),
                float(zero["gap"]),
                float(zero["win"]),
                1.0,
                float(identity["loss"] * identity["active"]),
                float(identity["top1"] * identity["active"]),
                float(identity["active"]),
                float(hard_normal.sum().item()) if len(hard_normal) else 0.0,
                float((hard_normal > 0.0).sum().item()) if len(hard_normal) else 0.0,
                float(hard_permuted.sum().item()) if len(hard_permuted) else 0.0,
                float((hard_permuted > 0.0).sum().item()) if len(hard_permuted) else 0.0,
                float(hard_zero.sum().item()) if len(hard_zero) else 0.0,
                float((hard_zero > 0.0).sum().item()) if len(hard_zero) else 0.0,
                float(len(hard_normal)),
                1.0 if len(hard_normal) else 0.0,
            ],
            dtype=torch.float64,
            device=state.device,
        )
        totals += values
    if state.enabled:
        distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    query_count = float(totals[9].item())
    if query_count <= 0.0:
        raise RuntimeError("multiscale context inner validation did not run a query")
    hard_count = float(totals[19].item())
    identity_count = float(totals[12].item())
    return {
        "normal_pose_loss": float((totals[0] / query_count).item()),
        "normal_pose_gap": float((totals[1] / query_count).item()),
        "normal_pose_win_fraction": float((totals[2] / query_count).item()),
        "permuted_pose_loss": float((totals[3] / query_count).item()),
        "permuted_pose_gap": float((totals[4] / query_count).item()),
        "permuted_pose_win_fraction": float((totals[5] / query_count).item()),
        "zero_visual_pose_loss": float((totals[6] / query_count).item()),
        "zero_visual_pose_gap": float((totals[7] / query_count).item()),
        "zero_visual_pose_win_fraction": float((totals[8] / query_count).item()),
        "normal_minus_permuted_pose_gap": float(((totals[1] - totals[4]) / query_count).item()),
        "normal_minus_zero_visual_pose_gap": float(((totals[1] - totals[7]) / query_count).item()),
        "identity_loss": float((totals[10] / identity_count).item()) if identity_count else 0.0,
        "identity_top1": float((totals[11] / identity_count).item()) if identity_count else 0.0,
        "identity_active": identity_count,
        "hard_repeat_gap": float((totals[13] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_win_fraction": float((totals[14] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_permuted_gap": float((totals[15] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_permuted_win_fraction": float((totals[16] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_zero_visual_gap": float((totals[17] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_zero_visual_win_fraction": float((totals[18] / hard_count).item()) if hard_count else 0.0,
        "hard_repeat_common_groups": hard_count,
        "hard_repeat_eligible_query_fraction": float((totals[20] / query_count).item()),
        "hard_repeat_normal_minus_permuted_gap": float(
            ((totals[13] - totals[15]) / hard_count).item() if hard_count else 0.0
        ),
        "hard_repeat_normal_minus_zero_visual_gap": float(
            ((totals[13] - totals[17]) / hard_count).item() if hard_count else 0.0
        ),
        "query_count": query_count,
    }


def inner_gate_decision(
    *, metrics: Mapping[str, float], args: argparse.Namespace
) -> dict[str, float | bool]:
    """Require both pose-level and identity-level controls to carry evidence."""

    required_names = (
        "normal_pose_win_fraction",
        "normal_pose_gap",
        "normal_minus_permuted_pose_gap",
        "normal_minus_zero_visual_pose_gap",
        "hard_repeat_eligible_query_fraction",
        "hard_repeat_win_fraction",
        "hard_repeat_gap",
        "hard_repeat_normal_minus_permuted_gap",
        "hard_repeat_normal_minus_zero_visual_gap",
    )
    try:
        values = {name: float(metrics[name]) for name in required_names}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("multiscale context inner gate metrics are incomplete") from error
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("multiscale context inner gate metrics are non-finite")
    thresholds = {
        "minimum_pose_win_fraction": float(args.minimum_pose_win_fraction),
        "minimum_pose_gap": float(args.minimum_pose_gap),
        "minimum_pose_permutation_delta": float(args.minimum_pose_permutation_delta),
        "minimum_pose_zero_visual_delta": float(args.minimum_pose_zero_visual_delta),
        "minimum_hard_repeat_eligible_query_fraction": float(
            args.minimum_hard_repeat_eligible_query_fraction
        ),
        "minimum_hard_repeat_win_fraction": float(args.minimum_hard_repeat_win_fraction),
        "minimum_hard_repeat_gap": float(args.minimum_hard_repeat_gap),
        "minimum_hard_repeat_permutation_delta": float(
            args.minimum_hard_repeat_permutation_delta
        ),
        "minimum_hard_repeat_zero_visual_delta": float(
            args.minimum_hard_repeat_zero_visual_delta
        ),
    }
    if (
        not all(math.isfinite(value) for value in thresholds.values())
        or not 0.0 <= thresholds["minimum_pose_win_fraction"] <= 1.0
        or not 0.0 < thresholds["minimum_hard_repeat_eligible_query_fraction"] <= 1.0
        or not 0.0 <= thresholds["minimum_hard_repeat_win_fraction"] <= 1.0
        or any(value < 0.0 for name, value in thresholds.items() if "fraction" not in name)
    ):
        raise ValueError("multiscale context inner gate thresholds are invalid")
    pose_passed = (
        values["normal_pose_win_fraction"] >= thresholds["minimum_pose_win_fraction"]
        and values["normal_pose_gap"] >= thresholds["minimum_pose_gap"]
        and values["normal_minus_permuted_pose_gap"]
        >= thresholds["minimum_pose_permutation_delta"]
        and values["normal_minus_zero_visual_pose_gap"]
        >= thresholds["minimum_pose_zero_visual_delta"]
    )
    hard_repeat_passed = (
        values["hard_repeat_eligible_query_fraction"]
        >= thresholds["minimum_hard_repeat_eligible_query_fraction"]
        and values["hard_repeat_win_fraction"] >= thresholds["minimum_hard_repeat_win_fraction"]
        and values["hard_repeat_gap"] >= thresholds["minimum_hard_repeat_gap"]
        and values["hard_repeat_normal_minus_permuted_gap"]
        >= thresholds["minimum_hard_repeat_permutation_delta"]
        and values["hard_repeat_normal_minus_zero_visual_gap"]
        >= thresholds["minimum_hard_repeat_zero_visual_delta"]
    )
    return {
        **values,
        **thresholds,
        "pose_passed": bool(pose_passed),
        "hard_repeat_passed": bool(hard_repeat_passed),
        "passed": bool(pose_passed and hard_repeat_passed),
    }


def _is_better(
    *,
    candidate: Mapping[str, float],
    candidate_gate: Mapping[str, float | bool],
    incumbent: Mapping[str, float] | None,
    incumbent_gate: Mapping[str, float | bool] | None,
) -> bool:
    if incumbent is None or incumbent_gate is None:
        return True
    candidate_key = (
        int(bool(candidate_gate["passed"])),
        float(candidate_gate["hard_repeat_normal_minus_zero_visual_gap"]),
        float(candidate_gate["hard_repeat_normal_minus_permuted_gap"]),
        float(candidate_gate["normal_minus_zero_visual_pose_gap"]),
        float(candidate_gate["normal_minus_permuted_pose_gap"]),
        float(candidate["hard_repeat_gap"]),
        float(candidate["normal_pose_gap"]),
    )
    incumbent_key = (
        int(bool(incumbent_gate["passed"])),
        float(incumbent_gate["hard_repeat_normal_minus_zero_visual_gap"]),
        float(incumbent_gate["hard_repeat_normal_minus_permuted_gap"]),
        float(incumbent_gate["normal_minus_zero_visual_pose_gap"]),
        float(incumbent_gate["normal_minus_permuted_pose_gap"]),
        float(incumbent["hard_repeat_gap"]),
        float(incumbent["normal_pose_gap"]),
    )
    return candidate_key > incumbent_key


def _write_json_atomically(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _prepare_output_directory(*, output_dir: Path, force: bool, state: _DistributedState) -> None:
    conflict = output_dir.exists() and any(output_dir.iterdir())
    if state.enabled:
        flag = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(flag, src=0)
        conflict = bool(int(flag.item()))
    if conflict and not bool(force):
        raise FileExistsError("refusing to overwrite multiscale context output")
    if state.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if state.enabled:
        distributed.barrier()


def _validate_args(args: argparse.Namespace) -> None:
    windows = _context_windows(args)
    radii = _phase_shift_radii(args)
    positive_values = (
        int(args.hidden_dim),
        int(args.edge_chunk_size),
        int(args.max_points_per_query),
        int(args.validation_selector_point_budget),
        int(args.validation_selector_grid_rows),
        int(args.validation_selector_grid_columns),
        int(args.inner_validation_fold_count),
        int(args.epochs),
    )
    scalars = (
        float(args.max_abs_log_ratio),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.pose_margin),
        float(args.hard_repeat_margin),
        float(args.appearance_control_margin),
        float(args.identity_loss_weight),
        float(args.hard_repeat_loss_weight),
        float(args.pose_loss_weight),
        float(args.pose_control_loss_weight),
        float(args.hard_repeat_control_loss_weight),
        float(args.local_radius_px),
        float(args.gradient_clip_norm),
        float(args.amp_init_scale),
    )
    if (
        str(args.source) not in CANDIDATE_MULTISCALE_CONTEXT_SOURCES
        or str(args.validation_selector_policy) not in _STATIC_SELECTOR_POLICIES
        or any(value <= 0 for value in positive_values)
        or int(args.inner_validation_fold_index) < 0
        or int(args.inner_validation_fold_index) >= int(args.inner_validation_fold_count)
        or int(args.max_hard_repeat_edges_per_query) < 0
        or int(args.support_permutation_shift) == 0
        or any(not math.isfinite(value) for value in scalars)
        or any(value < 0.0 for value in scalars[2:])
        or float(args.max_abs_log_ratio) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.local_radius_px) <= 0.0
        or float(args.gradient_clip_norm) <= 0.0
        or float(args.amp_init_scale) <= 0.0
        or any(value < 3 or value % 2 == 0 for value in windows.values())
        or any(value < 0 for value in radii.values())
    ):
        raise ValueError("multiscale context training arguments are invalid")


def _validate_target_contract(
    *,
    targets: CandidatePoseRGBSpatialTrainingTargets,
    hard_targets: CandidatePoseRGBSpatialHardRepeatTargets,
    local_radius_px: float,
) -> None:
    """Reject generic/old target artifacts before a P1 LLR experiment starts."""

    target_metadata = targets.metadata
    hard_metadata = hard_targets.metadata
    target_radius = float(target_metadata.get("spatial_search_radius_px", 0.0))
    if (
        target_metadata.get("spatial_supervision_mode") != "registered_exact_identity"
        or target_metadata.get("full_pose_hard_target_format")
        != "candidate_pose_rgb_spatial_full_pool_hard_modes_v1"
        or hard_metadata.get("positive_requires_registered_exact_track") is not True
        or hard_metadata.get("negative_selection")
        != "all_or_capped_distinct_coherent_wrong_local_candidates_v1"
        or not math.isfinite(target_radius)
        or float(local_radius_px) > target_radius
    ):
        raise ValueError(
            "multiscale context LLR requires current registered-identity full-pose hard targets"
        )


def _model_state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    return {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}


def train_candidate_multiscale_context_llr(args: argparse.Namespace) -> dict[str, object]:
    """Fit a source-specific phase LLR and retain the fixed final epoch."""

    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        _prepare_output_directory(output_dir=output_dir, force=bool(args.force), state=state)
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if state.device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - old torch
            pass

        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        hard_path = Path(args.hard_repeat_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(hard_path)
        layout_sha = file_sha256_short(layout_path)
        targets_sha = file_sha256_short(targets_path)
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha
        )
        _validate_target_contract(
            targets=targets,
            hard_targets=hard_targets,
            local_radius_px=float(args.local_radius_px),
        )
        groups = build_train_query_groups(layout=layout, targets=targets)
        hard_by_query = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=targets_sha,
        )
        train_ids, inner_ids = _partition_train_queries_for_inner_validation(
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
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("multiscale context source images do not share one coordinate frame")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        model = CandidateMultiscaleContextLLR(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            context_windows=_context_windows(args),
            phase_shift_radii=_phase_shift_radii(args),
            hidden_dim=int(args.hidden_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
        )
        trainable_names = configure_single_source_trainable_parameters(
            model=model, source=str(args.source)
        )
        model = model.to(state.device)
        model_for_train: nn.Module
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
                # Every parameter still requiring gradients belongs to the
                # selected head and is used by the normal visual forward.
                # Keeping unused-parameter discovery off is required because
                # normal/permuted/zero controls deliberately reuse that head
                # before their common backward pass.
                find_unused_parameters=False,
            )
        else:
            model_for_train = model
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled, init_scale=float(args.amp_init_scale)
        )

        initial_metrics = evaluate_inner_validation(
            model=model_for_train,
            layout=layout,
            groups=groups,
            hard_by_query=hard_by_query,
            complete_runtime=complete_runtime,
            query_ids=inner_ids,
            args=args,
            coordinate_image_size=coordinate_image_size,
            state=state,
            amp_enabled=amp_enabled,
        )
        initial_gate = inner_gate_decision(metrics=initial_metrics, args=args)
        if state.rank == 0:
            _write_json_atomically(
                output_dir / "initialization_inner_validation.json",
                {"metrics": initial_metrics, "gate": initial_gate},
            )
        if state.enabled:
            distributed.barrier()

        # The query-disjoint fold is telemetry, not an epoch selector.  The
        # following calibration/gate can only be interpreted independently if
        # this trainer always returns the predeclared final epoch.
        final_metrics: dict[str, float] | None = dict(initial_metrics) if state.rank == 0 else None
        final_gate: dict[str, float | bool] | None = dict(initial_gate) if state.rank == 0 else None
        history: list[dict[str, object]] = []
        local_ids = tuple(train_ids[state.rank :: state.world_size])
        if not local_ids:
            raise RuntimeError("multiscale context DDP rank has no train query")
        steps_per_epoch = max(
            math.ceil(len(train_ids) / state.world_size), 1
        )
        source_scales = _source_visual_scales(str(args.source))
        zero_scales = _zero_visual_source_scales()
        for epoch in range(int(args.epochs)):
            epoch_start = time.time()
            model_for_train.train()
            # The model contains only scalar source heads. BatchNorm/dropout
            # are absent, so train/eval mode cannot alter visual controls.
            totals = torch.zeros((24,), dtype=torch.float64, device=state.device)
            for step in range(steps_per_epoch):
                padding = step >= len(local_ids)
                query_id = local_ids[step % len(local_ids)]
                group = groups[str(query_id)]
                hard_targets_for_query = hard_by_query.get(str(query_id))
                required_sources = (
                    None
                    if hard_targets_for_query is None
                    else hard_targets_for_query.source_point_ids
                )
                positions = _select_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + int(epoch),
                    required_source_point_ids=required_sources,
                )
                batch = _query_batch_from_group(
                    group=group,
                    complete_runtime=complete_runtime,
                    point_positions=positions,
                    device=state.device,
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
                permuted_runtime = permute_runtime_support_image_appearance_only(
                    batch.runtime, shift=int(args.support_permutation_shift)
                )
                if (
                    torch.equal(batch.runtime.support_image_indices, permuted_runtime.support_image_indices)
                    or not torch.equal(batch.runtime.support_xy, permuted_runtime.support_xy)
                    or not torch.equal(batch.runtime.support_view_valid, permuted_runtime.support_view_valid)
                    or not torch.equal(
                        batch.runtime.candidate_view_weights, permuted_runtime.candidate_view_weights
                    )
                ):
                    raise RuntimeError(
                        "multiscale context support appearance control changed geometry or no image content"
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    normal_prediction = model_for_train(
                        runtime=batch.runtime, visual_source_scales=source_scales
                    )
                    permuted_prediction = model_for_train(
                        runtime=permuted_runtime, visual_source_scales=source_scales
                    )
                    zero_prediction = model_for_train(
                        runtime=batch.runtime, visual_source_scales=zero_scales
                    )
                    for name in CANDIDATE_MULTISCALE_CONTEXT_SOURCES:
                        if not torch.equal(
                            normal_prediction.source_edge_usable[name],
                            permuted_prediction.source_edge_usable[name],
                        ):
                            raise RuntimeError(
                                "multiscale context appearance control changed visual availability"
                            )
                    identity_loss, identity = registered_identity_loss(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        source=str(args.source),
                        observed_candidate_mask=batch.spatial_target_observed,
                    )
                    hard_loss, hard = hard_repeat_margin_loss(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        source=str(args.source),
                        hard_batch=hard_batch,
                        margin=float(args.hard_repeat_margin),
                    )
                    pose_loss, pose, normal_pose_gap = pose_margin_metrics(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        source=str(args.source),
                        local_radius_px=float(args.local_radius_px),
                        margin=float(args.pose_margin),
                    )
                    _, permuted_pose, permuted_pose_gap = pose_margin_metrics(
                        runtime=permuted_runtime,
                        prediction=permuted_prediction,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        source=str(args.source),
                        local_radius_px=float(args.local_radius_px),
                        margin=float(args.pose_margin),
                    )
                    _, zero_pose, zero_pose_gap = pose_margin_metrics(
                        runtime=batch.runtime,
                        prediction=zero_prediction,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        source=str(args.source),
                        local_radius_px=float(args.local_radius_px),
                        margin=float(args.pose_margin),
                    )
                    pose_control, pose_control_metrics = appearance_control_margin_loss(
                        normal_gaps=normal_pose_gap,
                        permuted_gaps=permuted_pose_gap,
                        zero_gaps=zero_pose_gap,
                        margin=float(args.appearance_control_margin),
                    )
                    hard_normal, hard_permuted, hard_zero = hard_repeat_control_metrics(
                        runtime=batch.runtime,
                        normal_prediction=normal_prediction,
                        permuted_runtime=permuted_runtime,
                        permuted_prediction=permuted_prediction,
                        zero_prediction=zero_prediction,
                        source=str(args.source),
                        hard_batch=hard_batch,
                    )
                    if len(hard_normal):
                        hard_control, hard_control_metrics = appearance_control_margin_loss(
                            normal_gaps=hard_normal,
                            permuted_gaps=hard_permuted,
                            zero_gaps=hard_zero,
                            margin=float(args.appearance_control_margin),
                        )
                    else:
                        hard_control = normal_prediction.source_edge_log_likelihood_ratios[
                            str(args.source)
                        ].sum() * 0.0
                        hard_control_metrics = {
                            "loss": 0.0,
                            "normal_minus_permuted": 0.0,
                            "normal_minus_zero": 0.0,
                        }
                    total_loss = (
                        float(args.identity_loss_weight) * identity_loss
                        + float(args.hard_repeat_loss_weight) * hard_loss
                        + float(args.pose_loss_weight) * pose_loss
                        + float(args.pose_control_loss_weight) * pose_control
                        + float(args.hard_repeat_control_loss_weight) * hard_control
                    )
                    if padding:
                        # Every DDP rank must participate in the same all-reduce
                        # count; duplicate padding cannot alter the objective.
                        total_loss = total_loss * 0.0
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                trainable_parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                gradients_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in trainable_parameters
                )
                if gradients_finite:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=float(args.gradient_clip_norm),
                    )
                else:
                    # ``GradScaler.step`` will consume the inf/nan flag set by
                    # ``unscale_`` and skip this update.  Do not run clipping
                    # on bad gradients, which would turn a skipped AMP update
                    # into a misleading NaN training statistic.
                    gradient_norm = torch.zeros((), dtype=torch.float32, device=state.device)
                scaler.step(optimizer)
                scaler.update()
                if not padding:
                    totals += torch.tensor(
                        [
                            float(total_loss.detach().item()),
                            float(identity_loss.detach().item()),
                            float(identity["active"]),
                            float(identity["top1"] * identity["active"]),
                            float(hard_loss.detach().item()),
                            float(hard["active"]),
                            float(hard["gap"] * hard["active"]),
                            float(hard["win"] * hard["active"]),
                            float(pose_loss.detach().item()),
                            float(pose["gap"]),
                            float(pose["win"]),
                            float(pose_control.detach().item()),
                            float(pose_control_metrics["normal_minus_permuted"]),
                            float(pose_control_metrics["normal_minus_zero"]),
                            float(hard_control.detach().item()),
                            float(len(hard_normal)),
                            float(hard_normal.sum().detach().item()) if len(hard_normal) else 0.0,
                            float(hard_permuted.sum().detach().item()) if len(hard_normal) else 0.0,
                            float(hard_zero.sum().detach().item()) if len(hard_normal) else 0.0,
                            float(gradient_norm.detach().item()),
                            1.0,
                            float(not gradients_finite),
                            float(scaler.get_scale()),
                            0.0,
                        ],
                        dtype=torch.float64,
                        device=state.device,
                    )
            if state.enabled:
                distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
            steps = float(totals[20].item())
            if steps <= 0.0:
                raise RuntimeError("multiscale context epoch had no real training step")
            inner = evaluate_inner_validation(
                model=model_for_train,
                layout=layout,
                groups=groups,
                hard_by_query=hard_by_query,
                complete_runtime=complete_runtime,
                query_ids=inner_ids,
                args=args,
                coordinate_image_size=coordinate_image_size,
                state=state,
                amp_enabled=amp_enabled,
            )
            if state.rank == 0:
                gate = inner_gate_decision(metrics=inner, args=args)
                epoch_metrics: dict[str, object] = {
                    "epoch": int(epoch + 1),
                    "train_total_loss": float((totals[0] / steps).item()),
                    "train_identity_loss": float((totals[1] / steps).item()),
                    "train_identity_active_per_step": float((totals[2] / steps).item()),
                    "train_identity_top1": float(
                        (totals[3] / totals[2]).item() if float(totals[2].item()) else 0.0
                    ),
                    "train_hard_repeat_loss": float((totals[4] / steps).item()),
                    "train_hard_repeat_active_per_step": float((totals[5] / steps).item()),
                    "train_hard_repeat_gap": float(
                        (totals[6] / totals[5]).item() if float(totals[5].item()) else 0.0
                    ),
                    "train_hard_repeat_win_fraction": float(
                        (totals[7] / totals[5]).item() if float(totals[5].item()) else 0.0
                    ),
                    "train_pose_loss": float((totals[8] / steps).item()),
                    "train_pose_gap": float((totals[9] / steps).item()),
                    "train_pose_win_fraction": float((totals[10] / steps).item()),
                    "train_pose_control_loss": float((totals[11] / steps).item()),
                    "train_pose_normal_minus_permuted": float((totals[12] / steps).item()),
                    "train_pose_normal_minus_zero": float((totals[13] / steps).item()),
                    "train_hard_repeat_control_loss": float((totals[14] / steps).item()),
                    "train_hard_repeat_control_groups_per_step": float((totals[15] / steps).item()),
                    "train_hard_repeat_normal_gap": float(
                        (totals[16] / totals[15]).item() if float(totals[15].item()) else 0.0
                    ),
                    "train_hard_repeat_permuted_gap": float(
                        (totals[17] / totals[15]).item() if float(totals[15].item()) else 0.0
                    ),
                    "train_hard_repeat_zero_gap": float(
                        (totals[18] / totals[15]).item() if float(totals[15].item()) else 0.0
                    ),
                    "train_gradient_norm": float((totals[19] / steps).item()),
                    "train_nonfinite_gradient_steps": int(totals[21].item()),
                    "train_amp_scale": float((totals[22] / steps).item()),
                    "global_query_steps": int(steps),
                    "epoch_seconds": float(time.time() - epoch_start),
                    **{f"inner_{name}": value for name, value in inner.items()},
                    "inner_gate_passed": bool(gate["passed"]),
                }
                # Keep final-epoch telemetry only. It never affects weights or
                # checkpoint choice.
                final_metrics = dict(inner)
                final_gate = dict(gate)
                history.append(epoch_metrics)
                _write_json_atomically(output_dir / "history.partial.json", history)
                _write_json_atomically(
                    output_dir / "progress.json",
                    {
                        "stage": "train_candidate_multiscale_context_llr",
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
        if final_metrics is None or final_gate is None:
            raise RuntimeError("multiscale context training did not retain final telemetry")
        final_state = _model_state_cpu(model_for_train)
        checkpoint_selection = fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs))
        input_paths = {
            "rgb_spatial_layout": layout_path,
            "training_targets": targets_path,
            "hard_repeat_targets": hard_path,
            "radio_final_context_cache": Path(args.radio_final_context_cache),
            "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
            "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        }
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "model_format": CANDIDATE_MULTISCALE_CONTEXT_LLR_FORMAT,
            "architecture": "one_independently_calibrated_full_2d_phase_llr_head_per_radio_final_radio_intermediate_or_alike_source_with_fixed_maplet_view_mixture_v1",
            "source": str(args.source),
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
            "support_view_mixture": "fixed_maplet_coverage_weights_with_neutral_missing_view_v1",
            "out_of_window_projection": "fixed_neutral_llr_not_learned_dustbin_v1",
            "candidate_slot_permutation_equivariant": True,
            "appearance_control_geometry_fixed": True,
            "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
            "inner_validation_used_for_model_selection": False,
            "source_heads_independently_calibrated": True,
            "source_weights_at_runtime": source_weights(str(args.source)),
            "encoder_inputs": [
                "frozen_query_anchor_xy",
                "fixed_support_observation_xy",
                "full_2d_radio_final_context_crop",
                "full_2d_radio_intermediate_context_crop",
                "full_2d_alike_spatial_phase_crop",
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
                "absolute_image_coordinate",
            ],
            "config": {
                "context_windows": _context_windows(args),
                "phase_shift_radii": _phase_shift_radii(args),
                "hidden_dim": int(args.hidden_dim),
                "max_abs_log_ratio": float(args.max_abs_log_ratio),
                "edge_chunk_size": int(args.edge_chunk_size),
                "local_radius_px": float(args.local_radius_px),
                "amp_init_scale": float(args.amp_init_scale),
            },
            "lineage": {
                "layout_sha256": layout_sha,
                "training_targets_sha256": targets_sha,
                "hard_repeat_targets_sha256": file_sha256_short(hard_path),
                "projection_space_id": str(layout.metadata["projection_space_id"]),
                "descriptor_space_id": str(layout.metadata["descriptor_space_id"]),
                "source_caches": {
                    name: file_sha256_short(path) for name, path in input_paths.items()
                    if name.endswith("context_cache")
                },
                "source_image_manifest_sha256": str(
                    sources[0].metadata.get("source_image_manifest_sha256", "")
                ),
            },
            "training": {
                "selected_epoch": int(args.epochs),
                "checkpoint_selection": checkpoint_selection,
                "epoch_zero_inner_validation": initial_metrics,
                "epoch_zero_inner_gate": initial_gate,
                "final_epoch_inner_validation": final_metrics,
                "final_epoch_inner_gate": final_gate,
                "trainable_parameters": list(trainable_names),
                "train_query_count": int(len(train_ids)),
                "inner_validation_query_count": int(len(inner_ids)),
                "fold_count": int(args.inner_validation_fold_count),
                "fold_index": int(args.inner_validation_fold_index),
            },
            "promotion_allowed": False,
            "heldout_evaluation_allowed": bool(final_gate["passed"]),
            "pnp_integration_allowed": False,
            "next_required_gate": "query_disjoint_validation_and_late_paired_audit_before_any_pose_or_pnp_promotion_v1",
        }
        checkpoint_path = output_dir / "checkpoint.pt"
        torch.save(
            {"format": CHECKPOINT_FORMAT, "state_dict": final_state, "metadata": metadata},
            checkpoint_path,
        )
        summary = {
            "stage": "train_candidate_multiscale_context_llr",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256_short(checkpoint_path),
            "selected_epoch": int(args.epochs),
            "checkpoint_selection": checkpoint_selection,
            "source": str(args.source),
            "inner_validation": final_metrics,
            "inner_gate": final_gate,
            "history": history,
            "protocol": {
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "fixed_global_topl": True,
                "runtime_layout_remains_target_free": True,
                "train_targets_joined_only_after_visual_forward": True,
                "normal_support_image_derangement_with_fixed_geometry_and_zero_visual_controls": True,
                "checkpoint_selection_did_not_consult_inner_validation": True,
                "heldout_validation_or_pose_not_run": True,
            },
        }
        _write_json_atomically(output_dir / "history.json", history)
        _write_json_atomically(output_dir / "summary.json", summary)
        _write_json_atomically(
            output_dir / "progress.json",
            {
                "stage": "train_candidate_multiscale_context_llr",
                "status": "complete",
                "completed_epochs": int(args.epochs),
                "selected_epoch": int(args.epochs),
                "inner_gate_passed": bool(final_gate["passed"]),
            },
        )
        return summary
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    summary = train_candidate_multiscale_context_llr(parse_args(argv))
    if summary:
        print(
            json.dumps(
                {
                    "checkpoint": summary.get("checkpoint"),
                    "source": summary.get("source"),
                    "inner_gate_passed": summary.get("inner_gate", {}).get("passed"),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
