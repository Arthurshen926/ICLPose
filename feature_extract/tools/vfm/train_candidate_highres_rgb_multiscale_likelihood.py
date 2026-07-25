"""Train a target-free, multi-scale real-RGB candidate pose likelihood.

The runtime network sees only frozen P1 query anchors, fixed global top-L
candidate support observations, and real RGB patches.  Correct and coherent
wrong pose projections are joined only after target-free visual densities have
been emitted.  The command deliberately stops at a train-query-disjoint gate:
no validation/test pose result and no PnP integration is permitted from this
checkpoint unless a separate paired held-out audit is run afterwards.

The two RGB scales are intentionally selectable independently.  This avoids
mistaking an apparent combined gain for a valid broad-context improvement when
the fine local measurement alone is doing all useful work.
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
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _query_batch_from_group,
    _select_group_points,
    _stable_query_hash,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_highres_rgb_multiscale_likelihood import (
    CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
    CANDIDATE_HIGHRES_RGB_SOURCES,
    CandidateHighresRGBMultiscaleLikelihood,
    CandidateHighresRGBMultiscalePrediction,
    highres_rgb_spatial_density_nll,
    score_candidate_highres_rgb_multiscale_batch,
    selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    permute_support_patch_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_highres_rgb_multiscale_likelihood_checkpoint_v2"
FIXED_FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"


def fixed_final_epoch_checkpoint_selection(*, epochs: int) -> dict[str, object]:
    """Describe the predeclared checkpoint choice used by this trainer.

    The query-disjoint inner fold is useful diagnostic telemetry, but it must
    never decide which epoch is handed to the subsequent temperature
    calibration.  Otherwise that same fold would be reused as a calibration
    validation set and its reported gate would no longer be independent.
    """

    if int(epochs) <= 0:
        raise ValueError("fixed-final-epoch checkpoint selection requires a positive epoch count")
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
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source",
        choices=("fine", "broad", "combined"),
        required=True,
        help="Visual source trained and audited in this independent run.",
    )
    parser.add_argument("--fine-search-radius-px", type=float, default=8.0)
    parser.add_argument("--fine-context-radius-px", type=float, default=12.0)
    parser.add_argument("--broad-search-radius-px", type=float, default=8.0)
    parser.add_argument("--broad-context-radius-px", type=float, default=24.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--rgb-temperature", type=float, default=10.0)
    parser.add_argument("--max-abs-edge-log-ratio", type=float, default=3.0)
    parser.add_argument("--max-abs-pose-log-ratio", type=float, default=6.0)
    parser.add_argument("--max-points-per-query", type=int, default=32)
    parser.add_argument("--max-hard-repeat-edges-per-query", type=int, default=256)
    parser.add_argument(
        "--max-train-queries",
        type=int,
        default=0,
        help="Diagnostic smoke limit only; zero retains every train query.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pose-margin", type=float, default=0.20)
    parser.add_argument("--hard-repeat-margin", type=float, default=0.20)
    parser.add_argument("--appearance-control-margin", type=float, default=0.05)
    parser.add_argument("--density-loss-weight", type=float, default=0.25)
    parser.add_argument("--dustbin-loss-weight", type=float, default=0.50)
    parser.add_argument("--pose-loss-weight", type=float, default=1.0)
    parser.add_argument("--soft-hard-loss-weight", type=float, default=0.50)
    parser.add_argument("--soft-hard-temperature", type=float, default=0.35)
    parser.add_argument("--hard-repeat-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--hard-repeat-appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--registered-observation-appearance-control-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Train-only direct normal-vs-permuted contrast on registered positive "
            "candidate observations. Zero preserves the pose-only control ablation."
        ),
    )
    parser.add_argument(
        "--registered-observation-appearance-control-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=4096.0,
        help="Conservative initial AMP loss scale for the high-variance pose margin objective.",
    )
    parser.add_argument("--rgb-cache-gb", type=float, default=6.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument(
        "--rgb-cache-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Decoded full-image cache placement. CPU keeps the complete uint8 map "
            "out of activation memory; RGB owner batches are still cropped on GPU."
        ),
    )
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
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


def source_weights(source: str) -> dict[str, float]:
    """Return the audited source blend without data-dependent weighting."""

    name = str(source).strip().lower()
    if name == "fine":
        return {"fine": 1.0, "broad": 0.0}
    if name == "broad":
        return {"fine": 0.0, "broad": 1.0}
    if name == "combined":
        return {"fine": 0.5, "broad": 0.5}
    raise ValueError("high-resolution RGB training source is invalid")


def active_source_names(source: str) -> tuple[str, ...]:
    """Return the scale forwards required by one independent experiment."""

    return tuple(name for name, weight in source_weights(source).items() if weight > 0.0)


def target_free_query_owner_costs(
    *, layout: CandidatePoseRGBSpatialLayout, groups: Mapping[str, TrainQueryGroup]
) -> dict[str, int]:
    """Estimate RGB-transfer work from the target-free frozen support layout.

    Every P1 group has the same point/edge count, whereas the number of unique
    owner images can vary by more than 2x.  CPU-cache crop time is dominated by
    transferring those owners to the crop GPU.  The estimate reads only query
    IDs, fixed support image IDs, and view-valid flags already available at
    runtime; it never inspects train targets, pose, residuals, or identities.
    """

    if not isinstance(layout, CandidatePoseRGBSpatialLayout) or not groups:
        raise ValueError("high-resolution RGB owner-cost inputs are invalid")
    query_ids = np.asarray(layout.query_ids).astype(str)
    support_ids = np.asarray(layout.support_image_ids).astype(str)
    support_valid = np.asarray(layout.support_view_valid, dtype=bool)
    costs: dict[str, int] = {}
    for query_id, group in groups.items():
        rows = np.asarray(group.layout_rows, dtype=np.int64).reshape(-1)
        if len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(query_ids)):
            raise ValueError("high-resolution RGB owner-cost layout rows are invalid")
        owners = set(query_ids[rows].tolist())
        owners.update(support_ids[rows][support_valid[rows]].tolist())
        if not owners:
            raise ValueError("high-resolution RGB owner-cost group is empty")
        costs[str(query_id)] = int(len(owners))
    return costs


def balanced_ddp_query_schedules(
    *,
    query_ids: Sequence[str],
    owner_costs: Mapping[str, int],
    world_size: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Pair ranks with similar target-free RGB owner-transfer workload.

    DDP synchronizes gradients after every query.  Randomly pairing a query
    with 200 source owners on one rank against one with 500 owners on the other
    leaves a GPU idle during crop/forward work.  The function keeps the epoch's
    query set intact, groups similar-cost queries into one DDP step, randomizes
    rank assignment and batch order, and pads only when the world size does not
    divide the query count.
    """

    ids = tuple(sorted(set(str(query_id) for query_id in query_ids)))
    workers = int(world_size)
    if not ids or workers <= 0 or any(query_id not in owner_costs for query_id in ids):
        raise ValueError("high-resolution RGB DDP schedule inputs are invalid")
    costs = {query_id: int(owner_costs[query_id]) for query_id in ids}
    if any(cost <= 0 for cost in costs.values()):
        raise ValueError("high-resolution RGB DDP owner cost is invalid")
    generator = np.random.default_rng(int(seed))
    if workers == 1:
        order = generator.permutation(len(ids))
        return (tuple(ids[int(index)] for index in order.tolist()),)
    tie_break = {query_id: int(value) for query_id, value in zip(ids, generator.permutation(len(ids)).tolist())}
    ordered = sorted(ids, key=lambda query_id: (costs[query_id], tie_break[query_id]))
    step_groups: list[list[str]] = []
    for start in range(0, len(ordered), workers):
        group = list(ordered[start : start + workers])
        if len(group) < workers:
            # Duplicate the final nearest-cost example only for the incomplete
            # DDP step. This matches the historical fixed-step behavior while
            # avoiding a low-cost query paired with an unrelated high-cost row.
            while len(group) < workers:
                group.append(group[-1])
        if len(group) != workers:
            raise RuntimeError("high-resolution RGB DDP schedule padding failed")
        generator.shuffle(group)
        step_groups.append(group)
    generator.shuffle(step_groups)
    return tuple(
        tuple(group[rank] for group in step_groups)
        for rank in range(workers)
    )


def ddp_owner_cost_balance_metrics(
    *, schedules: Sequence[Sequence[str]], owner_costs: Mapping[str, int]
) -> dict[str, float]:
    """Summarize per-step source-owner imbalance for training telemetry."""

    if not schedules or not all(len(schedule) == len(schedules[0]) for schedule in schedules):
        raise ValueError("high-resolution RGB DDP balance schedules are invalid")
    differences: list[float] = []
    for step in range(len(schedules[0])):
        values = [
            float(owner_costs[str(schedules[rank][step])])
            for rank in range(len(schedules))
        ]
        differences.append(max(values) - min(values))
    return {
        "mean_abs_owner_cost_difference": float(np.mean(differences)),
        "max_abs_owner_cost_difference": float(np.max(differences)),
    }


def configure_trainable_parameters(
    *, model: CandidateHighresRGBMultiscaleLikelihood, source: str
) -> tuple[str, ...]:
    """Train a shared RGB FPN plus only the selected scale calibrators."""

    if not isinstance(model, CandidateHighresRGBMultiscaleLikelihood):
        raise ValueError("high-resolution RGB trainable scope requires its model")
    enabled_sources = set(active_source_names(source))
    names: list[str] = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith("texture_encoder.") or any(
            name.startswith(f"calibrators.{source_name}.") for source_name in enabled_sources
        )
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names:
        raise RuntimeError("high-resolution RGB selected source has no trainable parameters")
    return tuple(names)


def _source_prediction_loss(
    *,
    prediction: CandidateHighresRGBMultiscalePrediction,
    target_offsets_xy: torch.Tensor,
    target_dustbin: torch.Tensor,
    target_supervised: torch.Tensor,
    source: str,
    dustbin_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = source_weights(source)
    values: list[torch.Tensor] = []
    metrics = {"active_edges": 0.0, "observed_edges": 0.0, "dustbin_edges": 0.0}
    for name in CANDIDATE_HIGHRES_RGB_SOURCES:
        weight = float(weights[name])
        if weight <= 0.0:
            continue
        loss, component = highres_rgb_spatial_density_nll(
            scale_prediction=prediction.sources[name],
            target_offsets_xy=target_offsets_xy,
            target_dustbin=target_dustbin,
            target_supervised=target_supervised,
            dustbin_weight=float(dustbin_weight),
            balance_observed_and_dustbin=True,
        )
        values.append(weight * loss)
        for key in metrics:
            metrics[key] += weight * float(component[key])
    if not values:
        raise RuntimeError("high-resolution RGB source loss has no enabled source")
    loss = torch.stack(values).sum()
    metrics["mean_nll"] = float(loss.detach().item())
    return loss, metrics


def _pose_scores(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    correct_projection_offsets_xy: torch.Tensor,
    correct_projection_valid: torch.Tensor,
    wrong_projection_offsets_xy: torch.Tensor,
    wrong_projection_valid: torch.Tensor,
    source: str,
    max_abs_pose_log_ratio: float,
    candidate_prior_temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    correct = score_candidate_highres_rgb_multiscale_batch(
        runtime=runtime,
        prediction=prediction,
        candidate_projection_offsets_xy=correct_projection_offsets_xy.unsqueeze(0),
        candidate_projection_valid=correct_projection_valid.unsqueeze(0),
        source=source,
        candidate_prior_temperature=float(candidate_prior_temperature),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    wrong = score_candidate_highres_rgb_multiscale_batch(
        runtime=runtime,
        prediction=prediction,
        candidate_projection_offsets_xy=wrong_projection_offsets_xy,
        candidate_projection_valid=wrong_projection_valid,
        source=source,
        candidate_prior_temperature=float(candidate_prior_temperature),
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    ).pose_log_likelihood_ratios
    if correct.shape != (1,) or wrong.ndim != 1 or len(wrong) == 0:
        raise ValueError("high-resolution RGB query pose score shapes are invalid")
    return correct, wrong


def pose_margin_terms(
    *, correct_scores: torch.Tensor, wrong_scores: torch.Tensor, margin: float, temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hardest and soft-hard pose losses plus the raw hardest gap."""

    if (
        correct_scores.shape != (1,)
        or wrong_scores.ndim != 1
        or len(wrong_scores) == 0
        or not math.isfinite(float(margin))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("high-resolution RGB pose margin inputs are invalid")
    gap = correct_scores - torch.amax(wrong_scores, dim=0, keepdim=True)
    hardest = F.softplus(float(margin) - gap).mean()
    normalized_soft_wrong = float(temperature) * torch.logsumexp(
        wrong_scores / float(temperature), dim=0
    ) - float(temperature) * math.log(float(len(wrong_scores)))
    soft_gap = correct_scores - normalized_soft_wrong
    soft = F.softplus(float(margin) - soft_gap).mean()
    return hardest, soft, gap


def _hard_repeat_gaps(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: CandidateHighresRGBMultiscalePrediction,
    hard_batch: HardRepeatBatch | None,
    source: str,
    max_abs_pose_log_ratio: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return direct true-minus-coherent-repeat edge gaps and common mask."""

    if hard_batch is None:
        return None, None
    positive, positive_usable = selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.positive_candidate_indices,
        offsets_xy=hard_batch.positive_offsets_xy,
        source=source,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    negative, negative_usable = selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=prediction,
        point_indices=hard_batch.point_indices,
        candidate_indices=hard_batch.negative_candidate_indices,
        offsets_xy=hard_batch.negative_offsets_xy,
        source=source,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    usable = positive_usable & negative_usable
    return positive - negative, usable


def appearance_control_margin_loss(
    *, normal_gaps: torch.Tensor, permuted_gaps: torch.Tensor, margin: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Require real candidate-specific RGB pairing to beat a derangement."""

    if (
        normal_gaps.shape != permuted_gaps.shape
        or normal_gaps.numel() == 0
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("high-resolution RGB appearance control inputs are invalid")
    delta = normal_gaps - permuted_gaps
    loss = F.softplus(float(margin) - delta).mean()
    return loss, {
        "normal_gap": float(normal_gaps.detach().mean().item()),
        "permuted_gap": float(permuted_gaps.detach().mean().item()),
        "normal_minus_permuted": float(delta.detach().mean().item()),
    }


def registered_observation_appearance_control_terms(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    normal_prediction: CandidateHighresRGBMultiscalePrediction,
    permuted_prediction: CandidateHighresRGBMultiscalePrediction,
    target_offsets_xy: torch.Tensor,
    target_observed: torch.Tensor,
    target_supervised: torch.Tensor,
    source: str,
    max_abs_pose_log_ratio: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Contrast real support RGB with a geometry-fixed patch derangement.

    The existing pose-level control may average away the few identity-bearing
    candidates inside a 20-way mixture.  This train-only term therefore joins
    exact registered observations to their candidate slot and compares the
    *same* local likelihood under normal and support-patch-permuted RGB.  It
    never alters the runtime layout and skips invalid/out-of-window samples
    rather than treating them as dustbin evidence.
    """

    if (
        not isinstance(runtime, CandidatePoseRGBSpatialRuntime)
        or not isinstance(normal_prediction, CandidateHighresRGBMultiscalePrediction)
        or not isinstance(permuted_prediction, CandidateHighresRGBMultiscalePrediction)
    ):
        raise ValueError("registered-observation appearance control inputs are invalid")
    device = normal_prediction.sources["fine"].joint_log_probabilities.device
    offsets = torch.as_tensor(target_offsets_xy, dtype=torch.float32, device=device)
    observed = torch.as_tensor(target_observed, dtype=torch.bool, device=device)
    supervised = torch.as_tensor(target_supervised, dtype=torch.bool, device=device)
    expected = (runtime.point_count, runtime.candidate_count)
    if (
        offsets.shape != (*expected, 2)
        or observed.shape != expected
        or supervised.shape != expected
        or not torch.isfinite(offsets).all()
    ):
        raise ValueError("registered-observation appearance-control targets are invalid")
    selected = observed & supervised
    points, candidates = torch.nonzero(selected, as_tuple=True)
    if len(points) == 0:
        zero = normal_prediction.sources["fine"].spatial_logits.sum() * 0.0
        return zero, {
            "observed_candidate_count": 0.0,
            "usable_candidate_count": 0.0,
            "normal_minus_permuted": 0.0,
        }
    selected_offsets = offsets[points, candidates]
    projection_valid = torch.ones((len(points),), dtype=torch.bool, device=device)
    normal_values, normal_usable = selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=normal_prediction,
        point_indices=points,
        candidate_indices=candidates,
        offsets_xy=selected_offsets,
        projection_valid=projection_valid,
        source=source,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    permuted_values, permuted_usable = selected_candidate_highres_rgb_log_likelihood_ratio_at_offsets(
        runtime=runtime,
        prediction=permuted_prediction,
        point_indices=points,
        candidate_indices=candidates,
        offsets_xy=selected_offsets,
        projection_valid=projection_valid,
        source=source,
        max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
    )
    usable = normal_usable & permuted_usable
    if not bool(usable.any()):
        return normal_values.sum() * 0.0, {
            "observed_candidate_count": float(len(points)),
            "usable_candidate_count": 0.0,
            "normal_minus_permuted": 0.0,
        }
    loss, metrics = appearance_control_margin_loss(
        normal_gaps=normal_values[usable],
        permuted_gaps=permuted_values[usable],
        margin=float(margin),
    )
    return loss, {
        **metrics,
        "observed_candidate_count": float(len(points)),
        "usable_candidate_count": float(usable.sum().item()),
    }


def inner_gate_decision(*, metrics: Mapping[str, float], args: argparse.Namespace) -> dict[str, float | bool]:
    """Require pose, repeat, permutation, and zero-visual evidence together."""

    required = (
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
    if any(name not in metrics or not math.isfinite(float(metrics[name])) for name in required):
        raise ValueError("high-resolution RGB inner gate metrics are incomplete")
    checks = {
        "pose_win": float(metrics["normal_pose_win_fraction"]) >= float(args.minimum_pose_win_fraction),
        "pose_gap": float(metrics["normal_pose_gap"]) >= float(args.minimum_pose_gap),
        "pose_permutation": float(metrics["normal_minus_permuted_pose_gap"])
        >= float(args.minimum_pose_permutation_delta),
        "pose_zero": float(metrics["normal_minus_zero_visual_pose_gap"])
        >= float(args.minimum_pose_zero_visual_delta),
        "repeat_coverage": float(metrics["hard_repeat_eligible_query_fraction"])
        >= float(args.minimum_hard_repeat_eligible_query_fraction),
        "repeat_win": float(metrics["hard_repeat_win_fraction"])
        >= float(args.minimum_hard_repeat_win_fraction),
        "repeat_gap": float(metrics["hard_repeat_gap"])
        >= float(args.minimum_hard_repeat_gap),
        "repeat_permutation": float(metrics["hard_repeat_normal_minus_permuted_gap"])
        >= float(args.minimum_hard_repeat_permutation_delta),
        "repeat_zero": float(metrics["hard_repeat_normal_minus_zero_visual_gap"])
        >= float(args.minimum_hard_repeat_zero_visual_delta),
    }
    return {**checks, "passed": bool(all(checks.values()))}


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    core = model.module if isinstance(model, DistributedDataParallel) else model
    return {name: value.detach().cpu().clone() for name, value in core.state_dict().items()}


def _reduce(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def _output_conflict(*, state: _DistributedState, paths: Sequence[Path]) -> bool:
    conflict = bool(any(Path(path).exists() for path in paths)) if state.rank == 0 else False
    if state.enabled:
        value = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(value, src=0)
        conflict = bool(int(value.item()))
    return conflict


@torch.no_grad()
def evaluate_inner_gate(
    *,
    model: nn.Module,
    groups: Mapping[str, TrainQueryGroup],
    hard_repeat_groups: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    patch_radius_px: float,
    cache: TensorImageLRUCache,
    cache_device: torch.device,
    state: _DistributedState,
    source: str,
    max_points_per_query: int,
    max_hard_repeat_edges_per_query: int,
    pose_margin: float,
    max_abs_pose_log_ratio: float,
    support_permutation_shift: int,
    amp_enabled: bool,
    seed: int,
    candidate_prior_temperature: float = 1.0,
) -> dict[str, float]:
    """Evaluate only a query-disjoint train fold under three visual controls."""

    if not query_ids or int(max_points_per_query) <= 0 or int(support_permutation_shift) <= 0:
        raise ValueError("high-resolution RGB inner evaluation configuration is invalid")
    # Queries are intentionally partitioned across ranks for inference.  A
    # DDP wrapper performs collectives in ``forward`` even with gradients off,
    # so calling it only on the ranks that own a query deadlocks against the
    # later metric all-reduce.  Evaluation has no backward pass: unwrap it and
    # let each rank run its independent target-free visual forwards.
    core_model = model.module if isinstance(model, DistributedDataParallel) else model
    core_model.eval()
    # normal loss, normal gap/win, permuted loss/gap/win, query count,
    # hard normal gap/win, hard permuted gap/win, hard edge count, and the
    # number of queries retaining a normal/permuted common hard-repeat edge.
    totals = torch.zeros((13,), dtype=torch.float64, device=state.device)
    for position, query_id in enumerate(query_ids):
        if position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        if group is None:
            raise ValueError("high-resolution RGB inner query is unresolved")
        # This is a fixed P1 layout selected before target construction.  Do
        # not call the train sampler here: it retains observed rows using
        # train-only fields.  Every fixed P1 row contributes to the gate.
        if group.point_count > int(max_points_per_query):
            raise ValueError("inner gate point budget would require target-dependent selection")
        point_positions = np.arange(group.point_count, dtype=np.int64)
        batch = _query_batch_from_group(
            group=group,
            complete_runtime=complete_runtime,
            point_positions=point_positions,
            device=state.device,
        )
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=batch.runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(patch_radius_px),
            step_px=1.0,
            cache=cache,
            device=state.device,
            cache_device=cache_device,
        )
        # The raw-RGB branch consumes support coordinates only for its fixed
        # in-window mask.  A visual control must leave that geometry intact
        # and derange RGB patches alone; otherwise border availability can
        # masquerade as an appearance effect.
        permuted_patches = permute_support_patch_appearance(
            runtime=batch.runtime,
            support_patches=support_patches,
            shift=int(support_permutation_shift),
        )
        if torch.equal(permuted_patches, support_patches):
            raise ValueError("high-resolution RGB support permutation did not alter appearance")
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            normal_prediction = core_model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
                active_sources=active_source_names(source),
            )
            permuted_prediction = core_model(
                runtime=batch.runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=permuted_patches,
                active_sources=active_source_names(source),
            )
            normal_correct, normal_wrong = _pose_scores(
                runtime=batch.runtime,
                prediction=normal_prediction,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                source=source,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
                candidate_prior_temperature=float(candidate_prior_temperature),
            )
            permuted_correct, permuted_wrong = _pose_scores(
                runtime=batch.runtime,
                prediction=permuted_prediction,
                correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                correct_projection_valid=batch.correct_projection_valid,
                wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                wrong_projection_valid=batch.wrong_projection_valid,
                source=source,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
                candidate_prior_temperature=float(candidate_prior_temperature),
            )
            normal_loss, _, normal_gap = pose_margin_terms(
                correct_scores=normal_correct,
                wrong_scores=normal_wrong,
                margin=float(pose_margin),
                temperature=0.35,
            )
            permuted_loss, _, permuted_gap = pose_margin_terms(
                correct_scores=permuted_correct,
                wrong_scores=permuted_wrong,
                margin=float(pose_margin),
                temperature=0.35,
            )
        totals[:7] += torch.tensor(
            [
                float(normal_loss.item()),
                float(normal_gap.item()),
                float((normal_gap > 0.0).to(dtype=torch.float32).item()),
                float(permuted_loss.item()),
                float(permuted_gap.item()),
                float((permuted_gap > 0.0).to(dtype=torch.float32).item()),
                1.0,
            ],
            dtype=torch.float64,
            device=state.device,
        )
        hard = hard_repeat_groups.get(str(query_id))
        if hard is not None:
            hard_batch = _hard_repeat_batch_from_group(
                hard_targets=hard,
                group=group,
                point_positions=point_positions,
                device=state.device,
                max_edges=int(max_hard_repeat_edges_per_query),
                seed=int(seed),
            )
            normal_hard, normal_usable = _hard_repeat_gaps(
                runtime=batch.runtime,
                prediction=normal_prediction,
                hard_batch=hard_batch,
                source=source,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
            )
            permuted_hard, permuted_usable = _hard_repeat_gaps(
                runtime=batch.runtime,
                prediction=permuted_prediction,
                hard_batch=hard_batch,
                source=source,
                max_abs_pose_log_ratio=max_abs_pose_log_ratio,
            )
            if (
                normal_hard is not None
                and normal_usable is not None
                and permuted_hard is not None
                and permuted_usable is not None
            ):
                common = normal_usable & permuted_usable
                if bool(common.any()):
                    values = normal_hard[common]
                    permuted_values = permuted_hard[common]
                    totals[7:13] += torch.tensor(
                        [
                            float(values.sum().item()),
                            float((values > 0.0).to(dtype=torch.float32).sum().item()),
                            float(permuted_values.sum().item()),
                            float((permuted_values > 0.0).to(dtype=torch.float32).sum().item()),
                            float(common.sum().item()),
                            1.0,
                        ],
                        dtype=torch.float64,
                        device=state.device,
                    )
    totals = _reduce(state, totals)
    query_count = float(totals[6].item())
    if query_count <= 0.0:
        raise RuntimeError("high-resolution RGB inner gate evaluated no query")
    hard_edge_count = float(totals[11].item())
    hard_eligible_query_count = float(totals[12].item())
    normal_gap = float((totals[1] / query_count).item())
    permuted_gap = float((totals[4] / query_count).item())
    if hard_edge_count > 0.0:
        hard_gap = float((totals[7] / hard_edge_count).item())
        hard_win = float((totals[8] / hard_edge_count).item())
        hard_permuted_gap = float((totals[9] / hard_edge_count).item())
        hard_permuted_win = float((totals[10] / hard_edge_count).item())
    else:
        hard_gap = 0.0
        hard_win = 0.0
        hard_permuted_gap = 0.0
        hard_permuted_win = 0.0
    return {
        "normal_pose_margin_loss": float((totals[0] / query_count).item()),
        "normal_pose_gap": normal_gap,
        "normal_pose_win_fraction": float((totals[2] / query_count).item()),
        "permuted_pose_margin_loss": float((totals[3] / query_count).item()),
        "permuted_pose_gap": permuted_gap,
        "permuted_pose_win_fraction": float((totals[5] / query_count).item()),
        "normal_minus_permuted_pose_gap": normal_gap - permuted_gap,
        # ``zero_appearance`` returns an exact target-free neutral likelihood,
        # so its correct-minus-wrong pose gap is structurally zero.
        "normal_minus_zero_visual_pose_gap": normal_gap,
        "query_count": query_count,
        "hard_repeat_common_active_edges": hard_edge_count,
        "hard_repeat_eligible_query_fraction": hard_eligible_query_count / query_count,
        "hard_repeat_gap": hard_gap,
        "hard_repeat_win_fraction": hard_win,
        "hard_repeat_permuted_gap": hard_permuted_gap,
        "hard_repeat_permuted_win_fraction": hard_permuted_win,
        "hard_repeat_normal_minus_permuted_gap": hard_gap - hard_permuted_gap,
        "hard_repeat_normal_minus_zero_visual_gap": hard_gap,
    }


def _validate_args(args: argparse.Namespace) -> None:
    finite_positive = (
        "fine_search_radius_px",
        "fine_context_radius_px",
        "broad_search_radius_px",
        "broad_context_radius_px",
        "rgb_temperature",
        "max_abs_edge_log_ratio",
        "max_abs_pose_log_ratio",
        "learning_rate",
        "soft_hard_temperature",
        "rgb_cache_gb",
        "amp_init_scale",
    )
    if (
        any(not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) <= 0.0 for name in finite_positive)
        or int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.max_points_per_query) <= 0
        or int(args.max_hard_repeat_edges_per_query) <= 0
        or int(args.max_train_queries) < 0
        or int(args.epochs) <= 0
        or int(args.inner_validation_fold_count) < 2
        or int(args.inner_validation_fold_index) < 0
        or int(args.inner_validation_fold_index) >= int(args.inner_validation_fold_count)
        or int(args.support_permutation_shift) <= 0
        or any(
            not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) < 0.0
            for name in (
                "pose_margin",
                "hard_repeat_margin",
                "appearance_control_margin",
                "density_loss_weight",
                "dustbin_loss_weight",
                "pose_loss_weight",
                "soft_hard_loss_weight",
                "hard_repeat_loss_weight",
                "pose_appearance_control_loss_weight",
                "hard_repeat_appearance_control_loss_weight",
                "registered_observation_appearance_control_loss_weight",
                "registered_observation_appearance_control_margin",
                "gradient_clip_norm",
            )
        )
    ):
        raise ValueError("high-resolution RGB training arguments are invalid")


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_candidate_highres_rgb_multiscale_likelihood(args: argparse.Namespace) -> dict[str, object]:
    """Fit the real-RGB branch and retain only a train-only diagnostic checkpoint."""

    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_highres_rgb_multiscale_likelihood.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if _output_conflict(state=state, paths=(checkpoint_path, history_path, summary_path)) and not bool(args.force):
            raise FileExistsError("refusing to overwrite high-resolution RGB likelihood output")
        random.seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        torch.manual_seed(int(args.seed) + state.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed) + state.rank)
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:  # pragma: no cover - older torch
            pass

        layout_path = Path(args.rgb_spatial_layout)
        targets_path = Path(args.training_targets)
        hard_repeat_path = Path(args.hard_repeat_targets)
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        targets = load_candidate_pose_rgb_spatial_training_targets(targets_path)
        validate_training_layout_and_targets(
            layout=layout,
            targets=targets,
            layout_sha256=file_sha256_short(layout_path),
        )
        if not math.isclose(
            float(targets.metadata.get("spatial_search_radius_px", float("nan"))),
            float(args.fine_search_radius_px),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(args.fine_search_radius_px), float(args.broad_search_radius_px), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("high-resolution RGB search radius must match the frozen target support")
        groups = build_train_query_groups(layout=layout, targets=targets)
        owner_costs = target_free_query_owner_costs(layout=layout, groups=groups)
        hard_repeat_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=targets,
            hard_repeat_targets=load_candidate_pose_rgb_spatial_hard_repeat_targets(hard_repeat_path),
            layout_sha256=file_sha256_short(layout_path),
            targets_sha256=file_sha256_short(targets_path),
        )
        all_query_ids = tuple(sorted(groups))
        if int(args.max_train_queries) > 0:
            if int(args.max_train_queries) < 2:
                raise ValueError("high-resolution RGB max-train-queries must be zero or at least two")
            all_query_ids = tuple(
                sorted(all_query_ids, key=lambda value: (_stable_query_hash(str(value)), str(value)))[: int(args.max_train_queries)]
            )
        inner_train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=all_query_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )
        headers = load_context_attention_source_headers(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
        )
        image_ids = np.asarray(headers.image_ids).astype(str)
        image_sizes = np.asarray(headers.image_sizes, dtype=np.int64)
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("high-resolution RGB likelihood requires one processed coordinate size")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=Path(args.image_root), image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=headers.metadata_by_name["radio_final"],
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        model = CandidateHighresRGBMultiscaleLikelihood(
            image_sizes=torch.from_numpy(image_sizes.astype(np.float32)),
            fine_search_radius_px=float(args.fine_search_radius_px),
            fine_context_radius_px=float(args.fine_context_radius_px),
            broad_search_radius_px=float(args.broad_search_radius_px),
            broad_context_radius_px=float(args.broad_context_radius_px),
            texture_feature_dim=int(args.texture_feature_dim),
            hidden_dim=int(args.hidden_dim),
            edge_chunk_size=int(args.edge_chunk_size),
            rgb_temperature=float(args.rgb_temperature),
            max_abs_edge_log_ratio=float(args.max_abs_edge_log_ratio),
        ).to(state.device)
        trainable_parameter_names = configure_trainable_parameters(model=model, source=str(args.source))
        if state.enabled:
            model_for_train: nn.Module = DistributedDataParallel(
                model,
                device_ids=[state.local_rank],
                output_device=state.local_rank,
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
            enabled=amp_enabled,
            init_scale=float(args.amp_init_scale),
        )
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else state.device
        patch_radius_px = float(model.full_patch_radius_px)
        if any(group.point_count > int(args.max_points_per_query) for group in (groups[q] for q in inner_validation_ids)):
            raise ValueError("inner gate requires all frozen P1 rows; increase max-points-per-query")

        # Epoch zero is diagnostic telemetry only.  The checkpoint is always
        # the predeclared final epoch, so this fold cannot affect model choice.
        start_time = time.time()
        baseline = evaluate_inner_gate(
            model=model_for_train,
            groups=groups,
            hard_repeat_groups=hard_repeat_groups,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            patch_radius_px=patch_radius_px,
            cache=cache,
            cache_device=cache_device,
            state=state,
            source=str(args.source),
            max_points_per_query=int(args.max_points_per_query),
            max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
            pose_margin=float(args.pose_margin),
            max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
            support_permutation_shift=int(args.support_permutation_shift),
            amp_enabled=amp_enabled,
            seed=int(args.seed),
        )
        history: list[dict[str, object]] = []
        baseline_gate = inner_gate_decision(metrics=baseline, args=args)
        final_metrics = dict(baseline)
        final_gate = dict(baseline_gate)
        steps_per_rank = int(math.ceil(len(inner_train_ids) / state.world_size))
        for epoch in range(1, int(args.epochs) + 1):
            model_for_train.train()
            schedules = balanced_ddp_query_schedules(
                query_ids=inner_train_ids,
                owner_costs=owner_costs,
                world_size=state.world_size,
                seed=int(args.seed) + epoch,
            )
            local_query_ids = schedules[state.rank]
            if len(local_query_ids) != steps_per_rank:
                raise RuntimeError("high-resolution RGB DDP schedule step count drifted")
            balance_metrics = ddp_owner_cost_balance_metrics(
                schedules=schedules,
                owner_costs=owner_costs,
            )
            totals = torch.zeros((16,), dtype=torch.float64, device=state.device)
            epoch_start = time.time()
            for local_step, query_id in enumerate(local_query_ids):
                group = groups[str(query_id)]
                hard_target = hard_repeat_groups.get(str(query_id))
                required_sources = None if hard_target is None else hard_target.source_point_ids
                positions = _select_group_points(
                    group=group,
                    max_points=int(args.max_points_per_query),
                    seed=int(args.seed) + epoch * 100003 + local_step,
                    required_source_point_ids=required_sources,
                )
                batch = _query_batch_from_group(
                    group=group,
                    complete_runtime=complete_runtime,
                    point_positions=positions,
                    device=state.device,
                )
                query_patches, support_patches = _crop_runtime_rgb_patches(
                    runtime=batch.runtime,
                    image_ids=image_ids,
                    image_root=Path(args.image_root),
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    radius_px=patch_radius_px,
                    step_px=1.0,
                    cache=cache,
                    device=state.device,
                    cache_device=cache_device,
                )
                permuted_patches = permute_support_patch_appearance(
                    runtime=batch.runtime,
                    support_patches=support_patches,
                    shift=int(args.support_permutation_shift),
                )
                if torch.equal(permuted_patches, support_patches):
                    raise ValueError("high-resolution RGB support permutation did not alter appearance")
                hard_batch = (
                    None
                    if hard_target is None
                    else _hard_repeat_batch_from_group(
                        hard_targets=hard_target,
                        group=group,
                        point_positions=positions,
                        device=state.device,
                        max_edges=int(args.max_hard_repeat_edges_per_query),
                        seed=int(args.seed) + epoch * 100003 + local_step,
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    normal_prediction = model_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=support_patches,
                        active_sources=active_source_names(str(args.source)),
                    )
                    permuted_prediction = model_for_train(
                        runtime=batch.runtime,
                        query_rgb_patches=query_patches,
                        support_rgb_patches=permuted_patches,
                        active_sources=active_source_names(str(args.source)),
                    )
                    density_loss, density_metrics = _source_prediction_loss(
                        prediction=normal_prediction,
                        target_offsets_xy=batch.spatial_target_offsets_xy,
                        target_dustbin=batch.spatial_target_dustbin,
                        target_supervised=batch.spatial_target_supervised,
                        source=str(args.source),
                        dustbin_weight=float(args.dustbin_loss_weight),
                    )
                    normal_correct, normal_wrong = _pose_scores(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        source=str(args.source),
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    permuted_correct, permuted_wrong = _pose_scores(
                        runtime=batch.runtime,
                        prediction=permuted_prediction,
                        correct_projection_offsets_xy=batch.correct_projection_offsets_xy,
                        correct_projection_valid=batch.correct_projection_valid,
                        wrong_projection_offsets_xy=batch.wrong_projection_offsets_xy,
                        wrong_projection_valid=batch.wrong_projection_valid,
                        source=str(args.source),
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    pose_loss, soft_hard_loss, normal_gap = pose_margin_terms(
                        correct_scores=normal_correct,
                        wrong_scores=normal_wrong,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    _, _, permuted_gap = pose_margin_terms(
                        correct_scores=permuted_correct,
                        wrong_scores=permuted_wrong,
                        margin=float(args.pose_margin),
                        temperature=float(args.soft_hard_temperature),
                    )
                    pose_control_loss, pose_control_metrics = appearance_control_margin_loss(
                        normal_gaps=normal_gap,
                        permuted_gaps=permuted_gap,
                        margin=float(args.appearance_control_margin),
                    )
                    registered_control_loss = density_loss * 0.0
                    registered_control_metrics = {
                        "observed_candidate_count": 0.0,
                        "usable_candidate_count": 0.0,
                        "normal_minus_permuted": 0.0,
                    }
                    if float(args.registered_observation_appearance_control_loss_weight) > 0.0:
                        registered_control_loss, registered_control_metrics = (
                            registered_observation_appearance_control_terms(
                                runtime=batch.runtime,
                                normal_prediction=normal_prediction,
                                permuted_prediction=permuted_prediction,
                                target_offsets_xy=batch.spatial_target_offsets_xy,
                                target_observed=batch.spatial_target_observed,
                                target_supervised=batch.spatial_target_supervised,
                                source=str(args.source),
                                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                                margin=float(
                                    args.registered_observation_appearance_control_margin
                                ),
                            )
                        )
                    hard_normal, hard_normal_usable = _hard_repeat_gaps(
                        runtime=batch.runtime,
                        prediction=normal_prediction,
                        hard_batch=hard_batch,
                        source=str(args.source),
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    hard_permuted, hard_permuted_usable = _hard_repeat_gaps(
                        runtime=batch.runtime,
                        prediction=permuted_prediction,
                        hard_batch=hard_batch,
                        source=str(args.source),
                        max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                    )
                    zero = density_loss * 0.0
                    hard_loss = zero
                    hard_control_loss = zero
                    hard_gap_value = 0.0
                    hard_win_value = 0.0
                    hard_active = 0.0
                    if (
                        hard_normal is not None
                        and hard_normal_usable is not None
                        and hard_permuted is not None
                        and hard_permuted_usable is not None
                    ):
                        common = hard_normal_usable & hard_permuted_usable
                        if bool(common.any()):
                            values = hard_normal[common]
                            permuted_values = hard_permuted[common]
                            hard_loss = F.softplus(float(args.hard_repeat_margin) - values).mean()
                            hard_control_loss, _ = appearance_control_margin_loss(
                                normal_gaps=values,
                                permuted_gaps=permuted_values,
                                margin=float(args.appearance_control_margin),
                            )
                            hard_gap_value = float(values.detach().mean().item())
                            hard_win_value = float((values.detach() > 0.0).float().mean().item())
                            hard_active = float(common.sum().item())
                    total_loss = (
                        float(args.density_loss_weight) * density_loss
                        + float(args.pose_loss_weight) * pose_loss
                        + float(args.soft_hard_loss_weight) * soft_hard_loss
                        + float(args.hard_repeat_loss_weight) * hard_loss
                        + float(args.pose_appearance_control_loss_weight) * pose_control_loss
                        + float(args.hard_repeat_appearance_control_loss_weight) * hard_control_loss
                        + float(args.registered_observation_appearance_control_loss_weight)
                        * registered_control_loss
                    )
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                locally_finite = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                finite_flag = torch.tensor([int(locally_finite)], dtype=torch.int64, device=state.device)
                if state.enabled:
                    distributed.all_reduce(finite_flag, op=distributed.ReduceOp.MIN)
                if bool(int(finite_flag.item())):
                    if float(args.gradient_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in model.parameters() if parameter.requires_grad],
                            float(args.gradient_clip_norm),
                        )
                    scaler.step(optimizer)
                else:
                    optimizer.zero_grad(set_to_none=True)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(total_loss.detach().item()),
                        float(density_loss.detach().item()),
                        float(pose_loss.detach().item()),
                        float(soft_hard_loss.detach().item()),
                        float(normal_gap.detach().item()),
                        float(pose_control_metrics["normal_minus_permuted"]),
                        float(hard_loss.detach().item()),
                        hard_gap_value,
                        hard_win_value,
                        hard_active,
                        float(hard_control_loss.detach().item()),
                        1.0 - float(int(finite_flag.item())),
                        1.0,
                        float(registered_control_loss.detach().item()),
                        float(registered_control_metrics["normal_minus_permuted"]),
                        float(registered_control_metrics["usable_candidate_count"]),
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce(state, totals)
            global_steps = float(steps_per_rank * state.world_size)
            inner_metrics = evaluate_inner_gate(
                model=model_for_train,
                groups=groups,
                hard_repeat_groups=hard_repeat_groups,
                complete_runtime=complete_runtime,
                query_ids=inner_validation_ids,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                patch_radius_px=patch_radius_px,
                cache=cache,
                cache_device=cache_device,
                state=state,
                source=str(args.source),
                max_points_per_query=int(args.max_points_per_query),
                max_hard_repeat_edges_per_query=int(args.max_hard_repeat_edges_per_query),
                pose_margin=float(args.pose_margin),
                max_abs_pose_log_ratio=float(args.max_abs_pose_log_ratio),
                support_permutation_shift=int(args.support_permutation_shift),
                amp_enabled=amp_enabled,
                seed=int(args.seed) + epoch,
            )
            gate = inner_gate_decision(metrics=inner_metrics, args=args)
            if state.rank == 0:
                # This records the current fixed-final-epoch telemetry.  It
                # intentionally does not influence optimization or state
                # selection; the final update is retained below unconditionally.
                final_metrics = dict(inner_metrics)
                final_gate = dict(gate)
                record: dict[str, object] = {
                    "epoch": int(epoch),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_density_loss": float((totals[1] / global_steps).item()),
                    "train_pose_loss": float((totals[2] / global_steps).item()),
                    "train_soft_hard_loss": float((totals[3] / global_steps).item()),
                    "train_pose_gap": float((totals[4] / global_steps).item()),
                    "train_pose_appearance_delta": float((totals[5] / global_steps).item()),
                    "train_hard_repeat_loss": float((totals[6] / global_steps).item()),
                    "train_hard_repeat_gap": float((totals[7] / global_steps).item()),
                    "train_hard_repeat_win_fraction": float((totals[8] / global_steps).item()),
                    "train_hard_repeat_active_edges": float((totals[9] / global_steps).item()),
                    "train_hard_repeat_appearance_loss": float((totals[10] / global_steps).item()),
                    "train_nonfinite_update_fraction": float((totals[11] / global_steps).item()),
                    "train_registered_observation_appearance_loss": float(
                        (totals[13] / global_steps).item()
                    ),
                    "train_registered_observation_appearance_delta": float(
                        (totals[14] / global_steps).item()
                    ),
                    "train_registered_observation_appearance_active_candidates": float(
                        (totals[15] / global_steps).item()
                    ),
                    "global_query_steps": int(global_steps),
                    "ddp_owner_cost_mean_abs_difference": float(
                        balance_metrics["mean_abs_owner_cost_difference"]
                    ),
                    "ddp_owner_cost_max_abs_difference": float(
                        balance_metrics["max_abs_owner_cost_difference"]
                    ),
                    "epoch_seconds": float(time.time() - epoch_start),
                    **{f"inner_{key}": value for key, value in inner_metrics.items()},
                    "inner_gate_passed": bool(gate["passed"]),
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank == 0:
            final_state = _model_state(model_for_train)
            checkpoint_selection = fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs))
            metadata: dict[str, object] = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_HIGHRES_RGB_MULTISCALE_LIKELIHOOD_FORMAT,
                "architecture": "real_rgb_fpn_fine1px_plus_broad2px_candidate_specific_density_v1",
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
                "projection_after_network_only": True,
                "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
                "scale_combination": "conservative_convex_edge_llr_blend_not_independent_product_v1",
                "appearance_control_geometry_fixed": True,
                "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
                "inner_validation_used_for_model_selection": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "pnp_integration_allowed": False,
                "heldout_evaluation_allowed": bool(final_gate["passed"]),
                "raw_scores_must_not_feed_pnp": True,
                "train_only_inner_gate_passed": bool(final_gate["passed"]),
                "encoder_inputs": [
                    "frozen_query_anchor_xy_for_rgb_crop_only",
                    "fixed_support_observation_xy_for_rgb_crop_only",
                    "real_rgb_query_and_fixed_support_patches",
                ],
                "encoder_excludes": [
                    "pose_matrix",
                    "projection_offset",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "candidate_rank",
                    "coarse_score",
                    "radio_or_alike_descriptor_values",
                ],
                "config": {
                    "source": str(args.source),
                    "fine_search_radius_px": float(args.fine_search_radius_px),
                    "fine_context_radius_px": float(args.fine_context_radius_px),
                    "fine_step_px": 1.0,
                    "broad_search_radius_px": float(args.broad_search_radius_px),
                    "broad_context_radius_px": float(args.broad_context_radius_px),
                    "broad_feature_step_px": 2.0,
                    "broad_output_step_px": 2.0,
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "amp_init_scale": float(args.amp_init_scale),
                    "rgb_temperature": float(args.rgb_temperature),
                    "max_abs_edge_log_ratio": float(args.max_abs_edge_log_ratio),
                    "max_abs_pose_log_ratio": float(args.max_abs_pose_log_ratio),
                    "registered_observation_appearance_control_loss_weight": float(
                        args.registered_observation_appearance_control_loss_weight
                    ),
                    "registered_observation_appearance_control_margin": float(
                        args.registered_observation_appearance_control_margin
                    ),
                    "trainable_parameter_scope": "shared_texture_encoder_plus_selected_scale_calibrators_v1",
                    "trainable_parameter_names": list(trainable_parameter_names),
                },
                "training": {
                    "objective": "target_free_multiscale_rgb_density_plus_correct_vs_full_coherent_wrong_pose_and_repeat_margin_with_geometry_fixed_support_permutation_and_registered_observation_appearance_controls_v3",
                    "epochs": int(args.epochs),
                    "world_size": int(state.world_size),
                    "ddp_schedule": "target_free_support_owner_cost_balanced_v1",
                    "owner_cost_range": {
                        "min": int(min(owner_costs.values())),
                        "max": int(max(owner_costs.values())),
                    },
                    "inner_train_query_count": len(inner_train_ids),
                    "inner_validation_query_count": len(inner_validation_ids),
                    "checkpoint_selection": checkpoint_selection,
                    "selected_epoch": int(args.epochs),
                    "epoch_zero_inner_validation_metrics": baseline,
                    "epoch_zero_inner_gate": baseline_gate,
                    "final_epoch_inner_validation_metrics": final_metrics,
                    "final_epoch_inner_gate": final_gate,
                    "zero_visual_control": "structural_neutral_no_rgb_evidence_v1",
                    "support_permutation_control": "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2",
                    "registered_observation_appearance_control": {
                        "enabled": bool(
                            float(args.registered_observation_appearance_control_loss_weight)
                            > 0.0
                        ),
                        "semantics": "train_only_registered_candidate_local_llr_normal_minus_geometry_fixed_support_rgb_derangement_v1",
                    },
                },
                "lineage": {
                    "layout_sha256": file_sha256_short(layout_path),
                    "training_targets_sha256": file_sha256_short(targets_path),
                    "hard_repeat_targets_sha256": file_sha256_short(hard_repeat_path),
                    "source_image_manifest_sha256": str(
                        headers.metadata_by_name["radio_final"].get("source_image_manifest_sha256", "")
                    ),
                    "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                    "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
                    "rgb_coordinate_bridge": rgb_bridge,
                },
            }
            _save_checkpoint(
                checkpoint_path,
                {"format": CHECKPOINT_FORMAT, "state_dict": final_state, "metadata": metadata},
            )
            history_path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
            summary: dict[str, object] = {
                "stage": "train_candidate_highres_rgb_multiscale_likelihood",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "checkpoint_selection": {
                    **checkpoint_selection,
                    "final_epoch_inner_validation": final_metrics,
                    "final_epoch_gate": final_gate,
                },
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "protocol": {
                    "runtime_layout_remains_target_free": True,
                    "pose_or_ground_truth_not_available_to_runtime_encoder": True,
                    "train_only_target_artifact": True,
                    "inner_validation_query_disjoint": True,
                    "inner_gate_uses_all_frozen_p1_rows": True,
                    "checkpoint_selection_did_not_consult_inner_validation": True,
                    "support_permutation_control": "fixed_runtime_geometry_and_validity_with_rgb_patch_derangement_only_v2",
                    "heldout_validation_or_test_not_run": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                },
                "rgb_cache_device": str(cache_device),
                "rgb_cache_rank0": cache.summary(),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            return summary
        return {}
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_candidate_highres_rgb_multiscale_likelihood(args)


if __name__ == "__main__":  # pragma: no cover - command entry point
    main()
