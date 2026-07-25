"""DDP pretrain candidate-specific RGB likelihood on real coherent-wrong poses.

Unlike center-only observation identity pretraining, this objective evaluates
the same fixed visual candidate layout at train-only correct and coherent-wrong
pose projections.  The neural encoder never receives a pose, residual, track
ID, rank, or label.  Those targets are joined only after target-free edge
densities have been emitted.
"""

from __future__ import annotations

import argparse
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_likelihood import (
    _crop_pair_rgb_patches,
    _discover_rgb_image_size,
    _initialize_distributed,
    _finalize_distributed,
    _source_table,
    _validate_rgb_coordinate_bridge,
    _DistributedState,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    CandidatePoseRGBSpatialHardPosePairs,
    load_candidate_pose_rgb_spatial_hard_pose_pairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS,
    CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    context_candidate_logit_mixture,
    context_identity_cross_entropy_loss,
    context_identity_support_permutation_margin_loss,
    permute_runtime_support_appearance,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    score_candidate_pose_rgb_spatial_batch,
    spatial_density_nll,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import TensorImageLRUCache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-pose-pairs", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-radius-px", type=float, default=8.0)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--max-abs-pose-log-ratio", type=float, default=6.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument("--radio-final-context-window", type=int, default=9)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=9)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument(
        "--context-encoder-arch",
        default="conv_v1",
        help="Explicit local 2D context encoder architecture serialized into checkpoint lineage.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pose-margin", type=float, default=0.25)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--density-loss-weight", type=float, default=0.5)
    parser.add_argument("--dustbin-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="L0 identity phase: skip the RGB spatial encoder and train only context identity.",
    )
    parser.add_argument(
        "--identity-loss-weight",
        type=float,
        default=0.0,
        help="Strict observed-track candidate cross-entropy on target-free context logits.",
    )
    parser.add_argument(
        "--identity-support-permutation-loss-weight",
        type=float,
        default=0.0,
        help="Train-only support-appearance derangement contrast for the strict identity head.",
    )
    parser.add_argument("--identity-support-permutation-margin", type=float, default=0.25)
    parser.add_argument(
        "--identity-support-permutation-shift",
        type=int,
        default=2,
        help="Nonzero candidate-slot derangement reserved away from the shift-1 gate.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--rgb-cache-gb", type=float, default=10.0)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _reduce_sum(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def _distributed_output_conflict(
    *, state: _DistributedState, paths: Sequence[Path]
) -> bool:
    """Make an output collision fail on every rank instead of deadlocking DDP."""

    conflict = bool(any(Path(path).exists() for path in paths)) if state.rank == 0 else False
    if state.enabled:
        decision = torch.tensor(
            [int(conflict)], dtype=torch.int64, device=state.device
        )
        distributed.broadcast(decision, src=0)
        conflict = bool(int(decision.item()))
    return conflict


def _runtime_from_hard_pose_rows(
    *,
    pairs: CandidatePoseRGBSpatialHardPosePairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
) -> CandidatePoseRGBSpatialRuntime:
    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(selected) == 0 or np.any(selected < 0) or np.any(selected >= pairs.row_count):
        raise ValueError("hard-pose runtime rows are invalid")
    query_ids = pairs.query_image_ids[selected]
    support_ids = pairs.support_image_ids[selected]
    try:
        query_indices = np.asarray(
            [image_index_by_id[str(value)] for value in query_ids.tolist()], dtype=np.int64
        )
        support_indices = np.asarray(
            [[image_index_by_id[str(value)] for value in row] for row in support_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("hard-pose pair image is absent from context sources") from error
    candidate_count = int(pairs.candidate_count)
    if support_indices.shape != (len(selected), candidate_count):
        raise RuntimeError("hard-pose support ownership has an unexpected shape")
    null_mass = float(CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS)
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.from_numpy(query_indices),
        query_xy=torch.from_numpy(np.asarray(pairs.query_xy[selected], dtype=np.float32)),
        support_image_indices=torch.from_numpy(support_indices[:, :, None]),
        support_xy=torch.from_numpy(
            np.asarray(pairs.support_xy[selected], dtype=np.float32)[:, :, None, :]
        ),
        support_view_valid=torch.ones((len(selected), candidate_count, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones(
            (len(selected), candidate_count, 1), dtype=torch.float32
        ),
        candidate_probabilities=torch.full(
            (len(selected), candidate_count),
            (1.0 - null_mass) / float(candidate_count),
            dtype=torch.float32,
        ),
        null_probabilities=torch.full((len(selected),), null_mass, dtype=torch.float32),
    )


def hard_pose_group_margin_loss(
    *, correct_scores: torch.Tensor, wrong_scores: torch.Tensor, margin: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require a grouped correct pose to beat its coherent wrong pose."""

    correct = torch.as_tensor(correct_scores, dtype=torch.float32).reshape(-1)
    wrong = torch.as_tensor(wrong_scores, dtype=torch.float32, device=correct.device).reshape(-1)
    required_margin = float(margin)
    if (
        len(correct) == 0
        or wrong.shape != correct.shape
        or not torch.isfinite(correct).all()
        or not torch.isfinite(wrong).all()
        or not math.isfinite(required_margin)
        or required_margin < 0.0
    ):
        raise ValueError("hard-pose grouped margin inputs are invalid")
    gaps = correct - wrong
    loss = F.relu(required_margin - gaps).mean()
    return loss, {
        "group_count": float(len(gaps)),
        "mean_correct_minus_wrong": float(gaps.detach().mean().item()),
        "correct_win_fraction": float((gaps.detach() > 0.0).float().mean().item()),
        "margin_loss": float(loss.detach().item()),
    }


def hard_pose_pretrain_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, object]:
    required = {
        "correct_win_fraction": float(metrics["normal_correct_win_fraction"]),
        "normal_gap": float(metrics["normal_mean_correct_minus_wrong"]),
        "visual_gap_delta": float(metrics["visual_gap_delta"]),
    }
    if not all(math.isfinite(value) for value in required.values()):
        raise ValueError("hard-pose pretrain gate metrics are invalid")
    checks = {
        "correct_win_fraction": required["correct_win_fraction"] >= float(minimum_win_fraction),
        "normal_gap": required["normal_gap"] >= float(minimum_normal_gap),
        "visual_gap_delta": required["visual_gap_delta"]
        >= float(minimum_visual_gap_delta),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "minimum_win_fraction": float(minimum_win_fraction),
            "minimum_normal_gap": float(minimum_normal_gap),
            "minimum_visual_gap_delta": float(minimum_visual_gap_delta),
        },
    }


def _rank_group_rows(
    *,
    group_ids: np.ndarray,
    rows_by_group: Mapping[int, np.ndarray],
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    groups = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if len(groups) == 0 or int(world_size) <= 0 or not 0 <= int(rank) < int(world_size):
        raise ValueError("hard-pose DDP group partition is invalid")
    ordered = np.random.default_rng(int(seed) + int(epoch) * 1000003).permutation(groups)
    per_rank = int(math.ceil(len(ordered) / float(world_size)))
    padded = np.resize(ordered, per_rank * int(world_size))
    local = padded[int(rank) * per_rank : (int(rank) + 1) * per_rank]
    return [np.asarray(rows_by_group[int(group)], dtype=np.int64) for group in local.tolist()]


def _score_pose_group(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: object,
    correct_offsets_xy: torch.Tensor,
    correct_valid: torch.Tensor,
    wrong_offsets_xy: torch.Tensor,
    wrong_valid: torch.Tensor,
    max_abs_pose_log_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = score_candidate_pose_rgb_spatial_batch(
        runtime=runtime,
        prediction=prediction,
        candidate_projection_offsets_xy=correct_offsets_xy.unsqueeze(0),
        candidate_projection_valid=correct_valid.unsqueeze(0),
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    wrong = score_candidate_pose_rgb_spatial_batch(
        runtime=runtime,
        prediction=prediction,
        candidate_projection_offsets_xy=wrong_offsets_xy.unsqueeze(0),
        candidate_projection_valid=wrong_valid.unsqueeze(0),
        missing_edge_log_likelihood_ratio=0.0,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    return normal, wrong


@torch.no_grad()
def _context_identity_control_metrics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: object,
    permuted_runtime: CandidatePoseRGBSpatialRuntime,
    permuted_prediction: object,
    target_observed: torch.Tensor,
) -> dict[str, float]:
    """Measure target-free context identity before and after support derangement.

    The model emits both context tensors without labels or pose projections.
    Only after both normal and deranged forwards are complete do we join the
    train-only observed-track indicators.  This distinguishes a context branch
    that has learned candidate identity from one whose pose gain comes entirely
    from the RGB spatial density.
    """

    normal_logits, normal_usable = context_candidate_logit_mixture(
        runtime=runtime,
        prediction=prediction,
    )
    permuted_logits, permuted_usable = context_candidate_logit_mixture(
        runtime=permuted_runtime,
        prediction=permuted_prediction,
    )
    targets = torch.as_tensor(
        target_observed, dtype=torch.bool, device=normal_logits.device
    )
    if (
        targets.shape != normal_logits.shape
        or permuted_logits.shape != normal_logits.shape
        or normal_usable.shape != normal_logits.shape
        or permuted_usable.shape != normal_logits.shape
        or torch.any(targets.sum(dim=1) > 1)
    ):
        raise ValueError("context identity control targets are invalid")
    labels = torch.argmax(targets.to(dtype=torch.long), dim=1)
    observed_rows = torch.any(targets, dim=1)
    normal_label_usable = normal_usable.gather(1, labels[:, None]).squeeze(1)
    permuted_label_usable = permuted_usable.gather(1, labels[:, None]).squeeze(1)
    active = (
        observed_rows
        & normal_label_usable
        & permuted_label_usable
        & (normal_usable.sum(dim=1) >= 2)
        & (permuted_usable.sum(dim=1) >= 2)
    )
    if not bool(active.any()):
        return {
            "context_identity_active_rows": 0.0,
            "context_identity_normal_top1_sum": 0.0,
            "context_identity_normal_margin_sum": 0.0,
            "context_identity_permuted_top1_sum": 0.0,
            "context_identity_permuted_margin_sum": 0.0,
            "context_identity_target_score_gap_sum": 0.0,
        }

    def _masked_margin(logits: torch.Tensor, usable: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = logits.masked_fill(~usable, -torch.inf)
        selected = masked.gather(1, labels[:, None]).squeeze(1)
        competing = masked.clone()
        competing.scatter_(1, labels[:, None], -torch.inf)
        margin = selected - torch.amax(competing, dim=1)
        return selected, margin

    normal_selected, normal_margin = _masked_margin(normal_logits, normal_usable)
    permuted_selected, permuted_margin = _masked_margin(
        permuted_logits, permuted_usable
    )
    normal_masked = normal_logits.masked_fill(~normal_usable, -torch.inf)
    permuted_masked = permuted_logits.masked_fill(~permuted_usable, -torch.inf)
    return {
        "context_identity_active_rows": float(active.sum().item()),
        "context_identity_normal_top1_sum": float(
            (torch.argmax(normal_masked[active], dim=1) == labels[active])
            .to(dtype=torch.float32)
            .sum()
            .item()
        ),
        "context_identity_normal_margin_sum": float(normal_margin[active].sum().item()),
        "context_identity_permuted_top1_sum": float(
            (torch.argmax(permuted_masked[active], dim=1) == labels[active])
            .to(dtype=torch.float32)
            .sum()
            .item()
        ),
        "context_identity_permuted_margin_sum": float(
            permuted_margin[active].sum().item()
        ),
        "context_identity_target_score_gap_sum": float(
            (normal_selected[active] - permuted_selected[active]).sum().item()
        ),
    }


@torch.no_grad()
def _evaluate_inner_validation(
    *,
    model: CandidatePoseRGBSpatialLikelihood,
    pairs: CandidatePoseRGBSpatialHardPosePairs,
    group_ids: Sequence[int],
    rows_by_group: Mapping[int, np.ndarray],
    image_index_by_id: Mapping[str, int],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    patch_radius_px: float,
    step_px: float,
    cache: TensorImageLRUCache,
    device: torch.device,
    pose_margin: float,
    dustbin_loss_weight: float,
    max_abs_pose_log_ratio: float,
    amp_enabled: bool,
    context_only: bool,
) -> dict[str, float]:
    model.eval()
    totals = np.zeros((15,), dtype=np.float64)
    for group_id in group_ids:
        rows = np.asarray(rows_by_group[int(group_id)], dtype=np.int64)
        runtime = _runtime_from_hard_pose_rows(
            pairs=pairs, rows=rows, image_index_by_id=image_index_by_id
        )
        if bool(context_only):
            query_patches = None
            support_patches = None
        else:
            query_patches, support_patches = _crop_pair_rgb_patches(
                runtime=runtime,
                image_ids=image_ids,
                image_root=image_root,
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=patch_radius_px,
                step_px=step_px,
                cache=cache,
                device=device,
            )
        correct_offsets = torch.from_numpy(
            np.asarray(pairs.correct_projection_offsets_xy[rows], dtype=np.float32)
        ).to(device)
        correct_valid = torch.from_numpy(
            np.asarray(pairs.correct_projection_valid[rows], dtype=bool)
        ).to(device)
        wrong_offsets = torch.from_numpy(
            np.asarray(pairs.coherent_wrong_projection_offsets_xy[rows], dtype=np.float32)
        ).to(device)
        wrong_valid = torch.from_numpy(
            np.asarray(pairs.coherent_wrong_projection_valid[rows], dtype=bool)
        ).to(device)
        target_observed = torch.from_numpy(
            np.asarray(pairs.spatial_target_observed[rows], dtype=bool)
        ).to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prediction = model(
                runtime=runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                context_only=bool(context_only),
            )
            if bool(context_only):
                density = prediction.context_log_likelihood_ratios.sum() * 0.0
            else:
                density, _ = spatial_density_nll(
                    prediction=prediction,
                    target_offsets_xy=correct_offsets,
                    target_dustbin=~target_observed,
                    target_supervised=torch.ones_like(target_observed),
                    dustbin_weight=float(dustbin_loss_weight),
                    balance_observed_and_dustbin=True,
                )
            normal_correct, normal_wrong = _score_pose_group(
                runtime=runtime,
                prediction=prediction,
                correct_offsets_xy=correct_offsets,
                correct_valid=correct_valid,
                wrong_offsets_xy=wrong_offsets,
                wrong_valid=wrong_valid,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
            )
            normal_loss, normal_metrics = hard_pose_group_margin_loss(
                correct_scores=normal_correct, wrong_scores=normal_wrong, margin=pose_margin
            )
            permuted_runtime = permute_runtime_support_appearance(runtime, shift=1)
            permuted_prediction = model(
                runtime=permuted_runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=(
                    None
                    if support_patches is None
                    else torch.roll(support_patches, shifts=1, dims=1)
                ),
                context_only=bool(context_only),
            )
            permuted_correct, permuted_wrong = _score_pose_group(
                runtime=permuted_runtime,
                prediction=permuted_prediction,
                correct_offsets_xy=correct_offsets,
                correct_valid=correct_valid,
                wrong_offsets_xy=wrong_offsets,
                wrong_valid=wrong_valid,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
            )
            permuted_loss, permuted_metrics = hard_pose_group_margin_loss(
                correct_scores=permuted_correct,
                wrong_scores=permuted_wrong,
                margin=pose_margin,
            )
            context_metrics = _context_identity_control_metrics(
                runtime=runtime,
                prediction=prediction,
                permuted_runtime=permuted_runtime,
                permuted_prediction=permuted_prediction,
                target_observed=target_observed,
            )
        totals += np.asarray(
            [
                float(normal_metrics["mean_correct_minus_wrong"]),
                float(normal_metrics["correct_win_fraction"]),
                float(normal_loss.item()),
                float(permuted_metrics["mean_correct_minus_wrong"]),
                float(permuted_metrics["correct_win_fraction"]),
                float(permuted_loss.item()),
                float(density.item()),
                1.0,
                float(len(rows)),
                float(context_metrics["context_identity_active_rows"]),
                float(context_metrics["context_identity_normal_top1_sum"]),
                float(context_metrics["context_identity_normal_margin_sum"]),
                float(context_metrics["context_identity_permuted_top1_sum"]),
                float(context_metrics["context_identity_permuted_margin_sum"]),
                float(context_metrics["context_identity_target_score_gap_sum"]),
            ],
            dtype=np.float64,
        )
    if totals[7] <= 0.0:
        raise RuntimeError("hard-pose inner validation has no group")
    count = totals[7]
    return {
        "normal_mean_correct_minus_wrong": float(totals[0] / count),
        "normal_correct_win_fraction": float(totals[1] / count),
        "normal_margin_loss": float(totals[2] / count),
        "permuted_mean_correct_minus_wrong": float(totals[3] / count),
        "permuted_correct_win_fraction": float(totals[4] / count),
        "permuted_margin_loss": float(totals[5] / count),
        "spatial_density_nll": float(totals[6] / count),
        "visual_gap_delta": float((totals[0] - totals[3]) / count),
        "context_identity_active_rows": float(totals[9]),
        "context_identity_normal_top1_accuracy": float(totals[10] / max(totals[9], 1.0)),
        "context_identity_normal_mean_margin": float(totals[11] / max(totals[9], 1.0)),
        "context_identity_permuted_top1_accuracy": float(totals[12] / max(totals[9], 1.0)),
        "context_identity_permuted_mean_margin": float(totals[13] / max(totals[9], 1.0)),
        "context_identity_target_score_gap": float(totals[14] / max(totals[9], 1.0)),
        "group_count": float(count),
        "row_count": float(totals[8]),
    }


def _is_better_epoch(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None) -> bool:
    if incumbent is None:
        return True
    candidate_loss = float(candidate["normal_margin_loss"])
    incumbent_loss = float(incumbent["normal_margin_loss"])
    return candidate_loss < incumbent_loss - 1e-12 or (
        abs(candidate_loss - incumbent_loss) <= 1e-12
        and float(candidate["normal_correct_win_fraction"])
        > float(incumbent["normal_correct_win_fraction"])
    )


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.search_radius_px),
        float(args.context_radius_px),
        float(args.step_px),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.pose_margin),
        float(args.pose_loss_weight),
        float(args.density_loss_weight),
        float(args.dustbin_loss_weight),
        float(args.identity_loss_weight),
        float(args.identity_support_permutation_loss_weight),
        float(args.identity_support_permutation_margin),
        float(args.gradient_clip_norm),
        float(args.rgb_cache_gb),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_visual_gap_delta),
        float(args.max_abs_context_log_ratio),
        float(args.max_abs_pose_log_ratio),
    )
    if (
        int(args.epochs) <= 0
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or not all(math.isfinite(value) for value in values)
        or float(args.search_radius_px) < float(args.step_px)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.pose_margin) < 0.0
        or float(args.pose_loss_weight) < 0.0
        or float(args.density_loss_weight) < 0.0
        or float(args.dustbin_loss_weight) <= 0.0
        or float(args.identity_loss_weight) < 0.0
        or float(args.identity_support_permutation_loss_weight) < 0.0
        or float(args.identity_support_permutation_margin) < 0.0
        or int(args.identity_support_permutation_shift) <= 0
        or float(args.gradient_clip_norm) <= 0.0
        or float(args.rgb_cache_gb) <= 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or float(args.minimum_visual_gap_delta) < 0.0
        or float(args.max_abs_context_log_ratio) <= 0.0
        or float(args.max_abs_pose_log_ratio) <= 0.0
    ):
        raise ValueError("hard-pose pretraining arguments are invalid")
    if (
        float(args.pose_loss_weight)
        + float(args.density_loss_weight)
        + float(args.identity_loss_weight)
        + float(args.identity_support_permutation_loss_weight)
        <= 0.0
    ):
        raise ValueError("hard-pose pretraining has no active objective")
    if bool(args.context_only) and (
        float(args.pose_loss_weight) > 0.0 or float(args.density_loss_weight) > 0.0
    ):
        raise ValueError("context-only L0 pretraining cannot include pose or spatial-density loss")
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def pretrain_candidate_pose_rgb_spatial_hard_pose_likelihood(
    args: argparse.Namespace,
) -> dict[str, object]:
    """Fit a target-free visual model using train-only hard pose targets."""

    context_windows = _validate_args(args)
    context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
        args.context_encoder_arch
    )
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_rgb_spatial_hard_pose_pretrain.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if not bool(args.force) and _distributed_output_conflict(
            state=state,
            paths=(checkpoint_path, history_path, summary_path),
        ):
            raise FileExistsError("refusing to overwrite hard-pose pretraining output")
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
        except AttributeError:  # pragma: no cover
            pass

        pair_path = Path(args.hard_pose_pairs)
        pairs = load_candidate_pose_rgb_spatial_hard_pose_pairs(pair_path)
        train_group_ids = np.unique(
            pairs.pose_pair_ids[pairs.split_names == "inner_train"]
        ).astype(np.int64)
        validation_group_ids = np.unique(
            pairs.pose_pair_ids[pairs.split_names == "inner_validation"]
        ).astype(np.int64)
        rows_by_group = {
            int(group_id): np.flatnonzero(pairs.pose_pair_ids == int(group_id)).astype(np.int64)
            for group_id in np.unique(pairs.pose_pair_ids).tolist()
        }
        if len(train_group_ids) == 0 or len(validation_group_ids) == 0:
            raise ValueError("hard-pose pairs lack an inner train or validation group")
        if not all(
            np.all(pairs.split_names[rows_by_group[int(group_id)]] == "inner_train")
            for group_id in train_group_ids.tolist()
        ) or not all(
            np.all(pairs.split_names[rows_by_group[int(group_id)]] == "inner_validation")
            for group_id in validation_group_ids.tolist()
        ):
            raise ValueError("hard-pose group split contract is inconsistent")
        if not math.isclose(
            float(pairs.metadata["spatial_search_radius_px"]),
            float(args.search_radius_px),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("hard-pose pair search radius differs from model support")
        expected_patch_radius = float(args.search_radius_px) + float(args.context_radius_px)
        if not math.isclose(
            float(pairs.metadata["rgb_patch_radius_px"]),
            expected_patch_radius,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("hard-pose pair patch radius differs from model support")
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
            raise ValueError("hard-pose pretraining currently requires common image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = _validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        image_index_by_id = {
            str(image_id): index for index, image_id in enumerate(image_ids.tolist())
        }
        referenced_ids = set(pairs.query_image_ids.tolist())
        referenced_ids.update(pairs.support_image_ids.reshape(-1).tolist())
        if not referenced_ids.issubset(image_index_by_id):
            raise ValueError("hard-pose pairs reference an image absent from context sources")
        model = CandidatePoseRGBSpatialLikelihood(
            sources=source_tensors,
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            search_radius_px=float(args.search_radius_px),
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
        if bool(args.context_only):
            # L0 must not obtain a spatial shortcut or leave unused trainable
            # parameters under DDP.  Keep the modules in the checkpoint for
            # later L1 compatibility, but freeze and bypass them here.
            for module in (
                model.texture_encoder,
                model.spatial_residual_head,
                model.non_dustbin_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        model = model.to(state.device)
        model_for_train: torch.nn.Module
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
                broadcast_buffers=False,
            )
        else:
            model_for_train = model
        core_model = model_for_train.module if state.enabled else model_for_train
        assert isinstance(core_model, CandidatePoseRGBSpatialLikelihood)
        trainable_parameters = [
            parameter for parameter in model_for_train.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("hard-pose pretraining has no trainable model parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3), storage_dtype=torch.float16
        )
        best_metrics: dict[str, float] | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        best_epoch = -1
        history: list[dict[str, object]] = []
        start_time = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            group_rows = _rank_group_rows(
                group_ids=train_group_ids,
                rows_by_group=rows_by_group,
                rank=state.rank,
                world_size=state.world_size,
                seed=int(args.seed),
                epoch=int(epoch),
            )
            totals = torch.zeros((14,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for rows in group_rows:
                runtime = _runtime_from_hard_pose_rows(
                    pairs=pairs, rows=rows, image_index_by_id=image_index_by_id
                )
                if bool(args.context_only):
                    query_patches = None
                    support_patches = None
                else:
                    query_patches, support_patches = _crop_pair_rgb_patches(
                        runtime=runtime,
                        image_ids=image_ids,
                        image_root=Path(args.image_root),
                        coordinate_image_size=coordinate_image_size,
                        rgb_image_size=rgb_image_size,
                        radius_px=expected_patch_radius,
                        step_px=float(args.step_px),
                        cache=cache,
                        device=state.device,
                    )
                correct_offsets = torch.from_numpy(
                    np.asarray(pairs.correct_projection_offsets_xy[rows], dtype=np.float32)
                ).to(state.device)
                correct_valid = torch.from_numpy(
                    np.asarray(pairs.correct_projection_valid[rows], dtype=bool)
                ).to(state.device)
                wrong_offsets = torch.from_numpy(
                    np.asarray(pairs.coherent_wrong_projection_offsets_xy[rows], dtype=np.float32)
                ).to(state.device)
                wrong_valid = torch.from_numpy(
                    np.asarray(pairs.coherent_wrong_projection_valid[rows], dtype=bool)
                ).to(state.device)
                observed = torch.from_numpy(
                    np.asarray(pairs.spatial_target_observed[rows], dtype=bool)
                ).to(state.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(
                        runtime=runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        context_only=bool(args.context_only),
                    )
                    if bool(args.context_only):
                        density_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                        density_metrics = {"spatial_density_active_edges": 0.0}
                    else:
                        density_loss, density_metrics = spatial_density_nll(
                            prediction=prediction,
                            target_offsets_xy=correct_offsets,
                            target_dustbin=~observed,
                            target_supervised=torch.ones_like(observed),
                            dustbin_weight=float(args.dustbin_loss_weight),
                            balance_observed_and_dustbin=True,
                        )
                    correct_score, wrong_score = _score_pose_group(
                        runtime=runtime,
                        prediction=prediction,
                        correct_offsets_xy=correct_offsets,
                        correct_valid=correct_valid,
                        wrong_offsets_xy=wrong_offsets,
                        wrong_valid=wrong_valid,
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    pose_loss, pose_metrics = hard_pose_group_margin_loss(
                        correct_scores=correct_score,
                        wrong_scores=wrong_score,
                        margin=float(args.pose_margin),
                    )
                    identity_loss, identity_metrics = context_identity_cross_entropy_loss(
                        runtime=runtime,
                        prediction=prediction,
                        target_observed=observed,
                    )
                    permutation_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                    permutation_metrics = {
                        "context_identity_permutation_active_rows": 0.0,
                        "context_identity_permutation_mean_gap": 0.0,
                        "context_identity_permutation_win_fraction": 0.0,
                        "context_identity_permutation_margin_loss": 0.0,
                    }
                    if float(args.identity_support_permutation_loss_weight) > 0.0:
                        permuted_runtime = permute_runtime_support_appearance(
                            runtime,
                            shift=int(args.identity_support_permutation_shift),
                        )
                        if torch.equal(
                            permuted_runtime.support_image_indices, runtime.support_image_indices
                        ) and torch.equal(permuted_runtime.support_xy, runtime.support_xy):
                            raise RuntimeError("identity support permutation did not change appearances")
                        permuted_prediction = model_for_train(
                            runtime=permuted_runtime,
                            context_only=True,
                        )
                        permutation_loss, permutation_metrics = (
                            context_identity_support_permutation_margin_loss(
                                runtime=runtime,
                                prediction=prediction,
                                permuted_runtime=permuted_runtime,
                                permuted_prediction=permuted_prediction,
                                target_observed=observed,
                                margin=float(args.identity_support_permutation_margin),
                            )
                        )
                    loss = (
                        float(args.pose_loss_weight) * pose_loss
                        + float(args.density_loss_weight) * density_loss
                        + float(args.identity_loss_weight) * identity_loss
                        + float(args.identity_support_permutation_loss_weight) * permutation_loss
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
                        float(pose_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(pose_metrics["mean_correct_minus_wrong"]),
                        float(pose_metrics["correct_win_fraction"]),
                        float(density_metrics["spatial_density_active_edges"]),
                        float(identity_loss.detach().item()),
                        float(identity_metrics["context_identity_top1_accuracy"]),
                        float(identity_metrics["context_identity_mean_margin"]),
                        float(identity_metrics["context_identity_active_rows"]),
                        float(permutation_metrics["context_identity_permutation_margin_loss"]),
                        float(permutation_metrics["context_identity_permutation_mean_gap"]),
                        float(permutation_metrics["context_identity_permutation_win_fraction"]),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce_sum(state, totals)
            global_steps = int(len(group_rows) * state.world_size)
            if state.rank == 0:
                inner = _evaluate_inner_validation(
                    model=core_model,
                    pairs=pairs,
                    group_ids=validation_group_ids.tolist(),
                    rows_by_group=rows_by_group,
                    image_index_by_id=image_index_by_id,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    patch_radius_px=expected_patch_radius,
                    step_px=float(args.step_px),
                    cache=cache,
                    device=state.device,
                    pose_margin=float(args.pose_margin),
                    dustbin_loss_weight=float(args.dustbin_loss_weight),
                    max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    amp_enabled=amp_enabled,
                    context_only=bool(args.context_only),
                )
                if _is_better_epoch(inner, best_metrics):
                    best_metrics = dict(inner)
                    best_epoch = int(epoch + 1)
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in core_model.state_dict().items()
                    }
                record = {
                    "epoch": int(epoch + 1),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_pose_margin_loss": float((totals[1] / global_steps).item()),
                    "train_spatial_density_loss": float((totals[2] / global_steps).item()),
                    "train_mean_correct_minus_wrong": float((totals[3] / global_steps).item()),
                    "train_correct_win_fraction": float((totals[4] / global_steps).item()),
                    "train_spatial_active_edges_per_step": float((totals[5] / global_steps).item()),
                    "train_context_identity_cross_entropy": float(
                        (totals[6] / global_steps).item()
                    ),
                    "train_context_identity_top1_accuracy": float(
                        (totals[7] / global_steps).item()
                    ),
                    "train_context_identity_mean_margin": float(
                        (totals[8] / global_steps).item()
                    ),
                    "train_context_identity_active_rows_per_step": float(
                        (totals[9] / global_steps).item()
                    ),
                    "train_context_identity_permutation_margin_loss": float(
                        (totals[10] / global_steps).item()
                    ),
                    "train_context_identity_permutation_mean_gap": float(
                        (totals[11] / global_steps).item()
                    ),
                    "train_context_identity_permutation_win_fraction": float(
                        (totals[12] / global_steps).item()
                    ),
                    "global_group_steps": int(global_steps),
                    "epoch_seconds": float(time.time() - epoch_start),
                    **{f"inner_{key}": value for key, value in inner.items()},
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank == 0:
            if best_state_dict is None or best_metrics is None or best_epoch < 1:
                raise RuntimeError("hard-pose pretraining did not select an inner checkpoint")
            gate = hard_pose_pretrain_gate(
                best_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata: dict[str, object] = {
                "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "architecture": (
                    "l0_candidate_specific_radio_final_intermediate_and_alike_"
                    "context_identity_pretrain_v1"
                    if bool(args.context_only)
                    else "local_2d_radio_final_intermediate_and_alike_context_plus_"
                    "high_resolution_real_rgb_cost_volume_hard_pose_group_pretrain_v1"
                ),
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "p1_finetune_allowed": bool(gate["passed"]),
                "raw_scores_must_not_feed_pnp": True,
                "encoder_inputs": [
                    "query_image_xy",
                    "fixed_support_image_xy",
                    "full_2d_radio_final_context_crop",
                    "full_2d_radio_intermediate_context_crop",
                    "full_2d_alike_context_crop",
                    *(
                        []
                        if bool(args.context_only)
                        else ["real_rgb_query_and_support_patches"]
                    ),
                ],
                "encoder_excludes": [
                    "pose_matrix",
                    "projection_offset",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "candidate_rank",
                    "coarse_score",
                ],
                "config": {
                    "search_radius_px": float(args.search_radius_px),
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_context_log_ratio": float(args.max_abs_context_log_ratio),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "context_windows": dict(context_windows),
                    "context_encoder_arch": context_encoder_arch,
                    "context_only": bool(args.context_only),
                    "fixed_candidate_null_mass": float(
                        CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_NULL_MASS
                    ),
                },
                "training": {
                    "objective": (
                        CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE
                        if bool(args.context_only)
                        else CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_OBJECTIVE
                    ),
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "pose_margin": float(args.pose_margin),
                    "pose_loss_weight": float(args.pose_loss_weight),
                    "density_loss_weight": float(args.density_loss_weight),
                    "dustbin_loss_weight": float(args.dustbin_loss_weight),
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
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only_disjoint_query_images",
                        "selected_epoch": int(best_epoch),
                        "selected_metrics": best_metrics,
                        "support_permutation_control": "candidate_slot_cyclic_shift_1_"
                        "reserved_for_inner_gate_v1",
                        "gate": gate,
                    },
                },
                "lineage": {
                    "hard_pose_pairs": str(pair_path.resolve()),
                    "hard_pose_pairs_sha256": file_sha256_short(pair_path),
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
                },
            }
            torch.save(
                {
                    "format": CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PRETRAIN_CHECKPOINT_FORMAT,
                    "state_dict": best_state_dict,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            summary = {
                "stage": "pretrain_candidate_pose_rgb_spatial_hard_pose_likelihood",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "checkpoint_selection": {
                    "selected_epoch": int(best_epoch),
                    "inner_validation": best_metrics,
                    "gate": gate,
                },
                "rgb_cache_rank0": cache.summary(),
                "protocol": {
                    "train_only_hard_pose_targets": True,
                    "runtime_checkpoint_target_free": True,
                    "validation_or_test_labels_used_by_fit": False,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "pnp_or_heldout_pose_not_run": True,
                    "l0_context_only": bool(args.context_only),
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
    args = parse_args(argv)
    result = pretrain_candidate_pose_rgb_spatial_hard_pose_likelihood(args)
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
