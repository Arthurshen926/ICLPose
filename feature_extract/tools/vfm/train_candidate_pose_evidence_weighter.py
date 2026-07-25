"""Train a target-free continuous token weighter against coherent wrong poses.

The frozen RADIO-phase/RGB spatial likelihood is evaluated first from the
target-free P1 layout.  Only then are train-only correct/coherent-wrong pose
projections joined to optimize a small evidence weighter.  The weighter never
sees pose, projections, residuals, landmark identity/rank, coarse score, or
labels at runtime.  It outputs all-token continuous weights and a query-level
uniform fallback mass; it does not perform hard top-K landmark selection.

This remains a train-only diagnostic.  A pass can permit a separately frozen
outer pose-rank audit, never direct PnP integration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_candidate_multiscale_phase_identity_llr import (
    _checkpoint_partition,
    _load_checkpoint as _load_phase_checkpoint,
    _load_model as _load_phase_model,
    _validate_checkpoint as _validate_phase_checkpoint,
)
from feature_extract.tools.vfm.audit_candidate_phase_identity_spatial_selector import (
    _load_rgb_model,
    _read_hybrid_checkpoint,
    summarize_pose_gap_distribution,
)
from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    _source_table,
)
from feature_extract.tools.vfm.train_candidate_phase_identity_spatial_likelihood import (
    CHECKPOINT_FORMAT as HYBRID_CHECKPOINT_FORMAT,
    _hybrid_pose_scores,
    _phase_control_common_availability,
)
from feature_extract.tools.vfm.train_candidate_highres_rgb_multiscale_likelihood import (
    pose_margin_terms,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    TrainQueryGroup,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _query_batch_from_group,
    _slice_runtime,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_evidence_selector import (
    runtime_visual_edge_availability,
)
from feature_extract.vfm.localization.candidate_pose_evidence_weighter import (
    CANDIDATE_POSE_EVIDENCE_WEIGHTER_FORMAT,
    TargetFreePoseEvidenceWeighter,
    TargetFreePoseEvidenceFeatures,
    TARGET_FREE_EVIDENCE_FEATURE_POLICIES,
    apply_target_free_feature_policy,
    build_target_free_pose_evidence_features,
    effective_sample_size,
    fit_target_free_feature_normalizer,
    relative_spatial_coverage_divergence,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


CHECKPOINT_FORMAT = "candidate_pose_evidence_weighter_checkpoint_v1"
PREPARED_QUERY_FORMAT = "candidate_pose_evidence_weighter_prepared_query_v1"


@dataclass(frozen=True)
class PreparedEvidenceWeighterQuery:
    """One query's target-free features plus train-only detached score targets.

    This object is intentionally in-memory only.  It is never serialized into
    the runtime checkpoint, preventing correct/wrong projections or derived
    target scores from leaking into inference artifacts.
    """

    query_id: str
    features: TargetFreePoseEvidenceFeatures
    xy: torch.Tensor
    normal_correct_points: torch.Tensor
    normal_wrong_points: torch.Tensor
    rgb_deranged_correct_points: torch.Tensor
    rgb_deranged_wrong_points: torch.Tensor

    def __post_init__(self) -> None:
        query_id = str(self.query_id)
        xy = torch.as_tensor(self.xy, dtype=torch.float32)
        count = self.features.point_count
        correct = torch.as_tensor(self.normal_correct_points, dtype=torch.float32).reshape(-1)
        wrong = torch.as_tensor(self.normal_wrong_points, dtype=torch.float32)
        deranged_correct = torch.as_tensor(self.rgb_deranged_correct_points, dtype=torch.float32).reshape(-1)
        deranged_wrong = torch.as_tensor(self.rgb_deranged_wrong_points, dtype=torch.float32)
        if (
            not query_id
            or xy.shape != (count, 2)
            or correct.shape != (count,)
            or wrong.ndim != 2
            or wrong.shape[0] == 0
            or wrong.shape[1] != count
            or deranged_correct.shape != (count,)
            or deranged_wrong.shape != wrong.shape
            or not torch.isfinite(xy).all()
            or not torch.isfinite(correct).all()
            or not torch.isfinite(wrong).all()
            or not torch.isfinite(deranged_correct).all()
            or not torch.isfinite(deranged_wrong).all()
        ):
            raise ValueError("prepared evidence weighter query is invalid")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "xy", xy.detach().cpu())
        object.__setattr__(self, "normal_correct_points", correct.detach().cpu())
        object.__setattr__(self, "normal_wrong_points", wrong.detach().cpu())
        object.__setattr__(self, "rgb_deranged_correct_points", deranged_correct.detach().cpu())
        object.__setattr__(self, "rgb_deranged_wrong_points", deranged_wrong.detach().cpu())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--phase-checkpoint", required=True)
    parser.add_argument("--hybrid-checkpoint", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--pose-margin", type=float, default=0.20)
    parser.add_argument("--soft-hard-temperature", type=float, default=0.35)
    parser.add_argument("--soft-hard-loss-weight", type=float, default=0.50)
    parser.add_argument("--rgb-appearance-control-loss-weight", type=float, default=0.25)
    parser.add_argument("--rgb-appearance-control-margin", type=float, default=0.02)
    parser.add_argument("--coverage-loss-weight", type=float, default=0.10)
    parser.add_argument("--coverage-grid-rows", type=int, default=4)
    parser.add_argument("--coverage-grid-columns", type=int, default=4)
    parser.add_argument("--effective-sample-size-loss-weight", type=float, default=0.25)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=96.0)
    parser.add_argument("--uniform-fallback-loss-weight", type=float, default=0.05)
    parser.add_argument("--minimum-uniform-mass", type=float, default=0.75)
    parser.add_argument("--maximum-uniform-mass", type=float, default=0.98)
    parser.add_argument("--initial-uniform-mass", type=float, default=0.90)
    parser.add_argument("--max-abs-local-logit", type=float, default=2.0)
    parser.add_argument(
        "--target-free-feature-policy",
        choices=TARGET_FREE_EVIDENCE_FEATURE_POLICIES,
        default="rgb_only",
        help="Use only RGB local-mode statistics unless a phase ablation is explicitly requested.",
    )
    parser.add_argument(
        "--uniform-regret-loss-weight",
        type=float,
        default=10.0,
        help="Train-only penalty when a learned weight is worse than that query's uniform baseline.",
    )
    parser.add_argument(
        "--maximum-train-uniform-gap-degradation",
        type=float,
        default=0.0,
        help="Allowed train-only learned-minus-uniform pose-gap regression before regret is penalized.",
    )
    parser.add_argument(
        "--uniform-win-safety-loss-weight",
        type=float,
        default=10.0,
        help="Extra train-only protection against turning a uniform-positive query negative.",
    )
    parser.add_argument(
        "--minimum-train-uniform-win-gap",
        type=float,
        default=0.0,
        help="Required learned pose gap for a train query whose uniform gap is positive.",
    )
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("uint8", "float16"), default="uint8")
    parser.add_argument("--rgb-cache-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--minimum-mean-gap-improvement", type=float, default=0.002)
    parser.add_argument("--minimum-median-gap-improvement", type=float, default=0.001)
    parser.add_argument("--maximum-p10-gap-degradation", type=float, default=0.002)
    parser.add_argument("--maximum-minimum-gap-degradation", type=float, default=0.010)
    parser.add_argument("--minimum-rgb-visual-gap-improvement", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        args.epochs,
        args.hidden_dim,
        args.coverage_grid_rows,
        args.coverage_grid_columns,
    )
    positive_values = (
        args.learning_rate,
        args.gradient_clip_norm,
        args.soft_hard_temperature,
        args.minimum_effective_sample_size,
        args.rgb_cache_gb,
        args.max_abs_local_logit,
    )
    nonnegative_values = (
        args.weight_decay,
        args.soft_hard_loss_weight,
        args.rgb_appearance_control_loss_weight,
        args.rgb_appearance_control_margin,
        args.coverage_loss_weight,
        args.effective_sample_size_loss_weight,
        args.uniform_fallback_loss_weight,
        args.uniform_regret_loss_weight,
        args.maximum_train_uniform_gap_degradation,
        args.uniform_win_safety_loss_weight,
        args.minimum_train_uniform_win_gap,
        args.minimum_mean_gap_improvement,
        args.minimum_median_gap_improvement,
        args.maximum_p10_gap_degradation,
        args.maximum_minimum_gap_degradation,
        args.minimum_rgb_visual_gap_improvement,
    )
    if (
        any(int(value) <= 0 for value in positive_ints)
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in positive_values)
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative_values)
        or not math.isfinite(float(args.pose_margin))
        or int(args.support_permutation_shift) == 0
        or not 0.0 <= float(args.minimum_uniform_mass) < float(args.maximum_uniform_mass) <= 1.0
        or not float(args.minimum_uniform_mass)
        <= float(args.initial_uniform_mass)
        <= float(args.maximum_uniform_mass)
    ):
        raise ValueError("pose evidence weighter training arguments are invalid")


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
    }


def _assert_runtime_alignment(
    *, visual_runtime: CandidatePoseRGBSpatialRuntime, target_runtime: CandidatePoseRGBSpatialRuntime
) -> None:
    names = (
        "query_image_indices",
        "query_xy",
        "support_image_indices",
        "support_xy",
        "support_view_valid",
        "candidate_view_weights",
        "candidate_probabilities",
        "null_probabilities",
    )
    if any(
        not torch.equal(getattr(visual_runtime, name), getattr(target_runtime, name))
        for name in names
    ):
        raise RuntimeError("evidence weighter target rows do not align to target-free visual runtime")


def aggregate_weighted_point_pose_scores(
    *, correct_points: torch.Tensor, wrong_points: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate fixed point LLRs with one static target-free weight vector."""

    correct = torch.as_tensor(correct_points, dtype=torch.float32)
    wrong = torch.as_tensor(wrong_points, dtype=torch.float32, device=correct.device)
    static = torch.as_tensor(weights, dtype=torch.float32, device=correct.device).reshape(-1)
    if (
        correct.ndim != 1
        or wrong.ndim != 2
        or wrong.shape[0] == 0
        or wrong.shape[1] != len(correct)
        or static.shape != correct.shape
        or not torch.isfinite(correct).all()
        or not torch.isfinite(wrong).all()
        or not torch.isfinite(static).all()
        or torch.any(static < 0.0)
        or not bool(static.sum() > 0.0)
    ):
        raise ValueError("weighted point pose score inputs are invalid")
    normalized = static / static.sum().clamp_min(torch.finfo(static.dtype).tiny)
    return (correct * normalized).sum().reshape(1), wrong @ normalized


def _query_terms(
    *,
    prepared: PreparedEvidenceWeighterQuery,
    weights: torch.Tensor,
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    device = weights.device
    correct, wrong = aggregate_weighted_point_pose_scores(
        correct_points=prepared.normal_correct_points.to(device),
        wrong_points=prepared.normal_wrong_points.to(device),
        weights=weights,
    )
    deranged_correct, deranged_wrong = aggregate_weighted_point_pose_scores(
        correct_points=prepared.rgb_deranged_correct_points.to(device),
        wrong_points=prepared.rgb_deranged_wrong_points.to(device),
        weights=weights,
    )
    hard_loss, soft_loss, normal_gap = pose_margin_terms(
        correct_scores=correct,
        wrong_scores=wrong,
        margin=float(args.pose_margin),
        temperature=float(args.soft_hard_temperature),
    )
    uniform = torch.full_like(weights, 1.0 / float(len(weights)))
    uniform_correct, uniform_wrong = aggregate_weighted_point_pose_scores(
        correct_points=prepared.normal_correct_points.to(device),
        wrong_points=prepared.normal_wrong_points.to(device),
        weights=uniform,
    )
    _uniform_hard, _uniform_soft, uniform_normal_gap = pose_margin_terms(
        correct_scores=uniform_correct,
        wrong_scores=uniform_wrong,
        margin=float(args.pose_margin),
        temperature=float(args.soft_hard_temperature),
    )
    # These two losses are train-only robust-risk controls.  They use the
    # already frozen uniform aggregation as a per-query reference, not a
    # runtime input.  A model may improve an easy query, but it is explicitly
    # charged for regressing any query and especially for flipping a
    # previously positive uniform pose gap below zero.
    uniform_regret = F.relu(
        uniform_normal_gap
        - normal_gap
        - float(args.maximum_train_uniform_gap_degradation)
    ).mean()
    uniform_won = (uniform_normal_gap > 0.0).to(dtype=normal_gap.dtype)
    uniform_win_safety = (
        uniform_won
        * F.relu(float(args.minimum_train_uniform_win_gap) - normal_gap)
    ).mean()
    _deranged_hard, _deranged_soft, deranged_gap = pose_margin_terms(
        correct_scores=deranged_correct,
        wrong_scores=deranged_wrong,
        margin=float(args.pose_margin),
        temperature=float(args.soft_hard_temperature),
    )
    rgb_delta = normal_gap - deranged_gap
    rgb_control = F.softplus(float(args.rgb_appearance_control_margin) - rgb_delta).mean()
    ess = effective_sample_size(weights)
    ess_penalty = F.relu(float(args.minimum_effective_sample_size) - ess) / float(
        args.minimum_effective_sample_size
    )
    coverage = relative_spatial_coverage_divergence(
        weights=weights,
        xy=prepared.xy.to(device),
        image_size=image_size,
        grid_rows=int(args.coverage_grid_rows),
        grid_columns=int(args.coverage_grid_columns),
    )
    return {
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "normal_gap": normal_gap,
        "uniform_normal_gap": uniform_normal_gap,
        "uniform_regret": uniform_regret,
        "uniform_win_safety": uniform_win_safety,
        "deranged_gap": deranged_gap,
        "rgb_delta": rgb_delta,
        "rgb_control": rgb_control,
        "ess": ess,
        "ess_penalty": ess_penalty,
        "coverage": coverage,
    }


def _objective(
    *, terms: Mapping[str, torch.Tensor], uniform_mass: torch.Tensor, args: argparse.Namespace
) -> torch.Tensor:
    mass = torch.as_tensor(uniform_mass, dtype=torch.float32, device=terms["hard_loss"].device)
    if mass.shape != (1,) or not torch.isfinite(mass).all() or bool(mass[0] < 0.0) or bool(mass[0] > 1.0):
        raise ValueError("pose evidence weighter uniform mass is invalid")
    return (
        terms["hard_loss"]
        + float(args.soft_hard_loss_weight) * terms["soft_loss"]
        + float(args.rgb_appearance_control_loss_weight) * terms["rgb_control"]
        + float(args.coverage_loss_weight) * terms["coverage"]
        + float(args.effective_sample_size_loss_weight) * terms["ess_penalty"]
        + float(args.uniform_fallback_loss_weight) * (1.0 - mass[0])
        + float(args.uniform_regret_loss_weight) * terms["uniform_regret"]
        + float(args.uniform_win_safety_loss_weight) * terms["uniform_win_safety"]
    )


@torch.no_grad()
def _prepare_query(
    *,
    group: TrainQueryGroup,
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    phase_model,
    rgb_model,
    cache: TensorImageLRUCache,
    cache_device: torch.device,
    device: torch.device,
    score_args: argparse.Namespace,
    support_permutation_shift: int,
    target_free_feature_policy: str,
) -> PreparedEvidenceWeighterQuery:
    points = np.arange(group.point_count, dtype=np.int64)
    visual_runtime = _slice_runtime(complete_runtime, group.layout_rows[points])
    query_patches, support_patches = _crop_runtime_rgb_patches(
        runtime=visual_runtime,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=float(rgb_model.full_patch_radius_px),
        step_px=1.0,
        cache=cache,
        device=device,
        cache_device=cache_device,
    )
    from feature_extract.vfm.localization.candidate_phase_identity_spatial_likelihood import (
        permute_support_patches_with_phase_identity_point_blocks,
    )

    rgb_deranged_patches = permute_support_patches_with_phase_identity_point_blocks(
        runtime=visual_runtime,
        support_patches=support_patches,
        shift=int(support_permutation_shift),
    )
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        rgb_normal = rgb_model(
            runtime=visual_runtime,
            query_rgb_patches=query_patches,
            support_rgb_patches=support_patches,
            active_sources=("fine",),
        )
        rgb_deranged = rgb_model(
            runtime=visual_runtime,
            query_rgb_patches=query_patches,
            support_rgb_patches=rgb_deranged_patches,
            active_sources=("fine",),
        )
    phase_normal = phase_model(runtime=visual_runtime)
    phase_deranged = phase_model(
        runtime=visual_runtime, support_permutation_shift=int(support_permutation_shift)
    )
    runtime_availability = runtime_visual_edge_availability(
        runtime=visual_runtime,
        phase_prediction=phase_normal,
        rgb_prediction=rgb_normal,
    )
    # The visual feature object is fully constructed before any target-bearing
    # batch exists in this function.
    feature_object = build_target_free_pose_evidence_features(
        runtime=visual_runtime,
        phase_prediction=phase_normal,
        rgb_prediction=rgb_normal,
        edge_availability_override=runtime_availability,
    )
    features = apply_target_free_feature_policy(
        features=feature_object,
        policy=str(target_free_feature_policy),
    ).values.detach().cpu()
    target_batch = _query_batch_from_group(
        group=group, complete_runtime=complete_runtime, point_positions=points, device=device
    )
    _assert_runtime_alignment(visual_runtime=visual_runtime, target_runtime=target_batch.runtime)
    paired_availability = _phase_control_common_availability(
        phase_normal=phase_normal,
        phase_deranged=phase_deranged,
        rgb_normal=rgb_normal,
        rgb_deranged=rgb_deranged,
    )
    normal_correct, normal_wrong = _hybrid_pose_scores(
        runtime=target_batch.runtime,
        phase_prediction=phase_normal,
        rgb_prediction=rgb_normal,
        availability=paired_availability,
        correct_offsets_xy=target_batch.correct_projection_offsets_xy,
        correct_valid=target_batch.correct_projection_valid,
        wrong_offsets_xy=target_batch.wrong_projection_offsets_xy,
        wrong_valid=target_batch.wrong_projection_valid,
        args=score_args,
    )
    rgb_correct, rgb_wrong = _hybrid_pose_scores(
        runtime=target_batch.runtime,
        phase_prediction=phase_normal,
        rgb_prediction=rgb_deranged,
        availability=paired_availability,
        correct_offsets_xy=target_batch.correct_projection_offsets_xy,
        correct_valid=target_batch.correct_projection_valid,
        wrong_offsets_xy=target_batch.wrong_projection_offsets_xy,
        wrong_valid=target_batch.wrong_projection_valid,
        args=score_args,
    )
    return PreparedEvidenceWeighterQuery(
        query_id=group.query_id,
        features=TargetFreePoseEvidenceFeatures(values=features),
        xy=visual_runtime.query_xy,
        normal_correct_points=normal_correct.point_log_likelihood_ratios[0],
        normal_wrong_points=normal_wrong.point_log_likelihood_ratios,
        rgb_deranged_correct_points=rgb_correct.point_log_likelihood_ratios[0],
        rgb_deranged_wrong_points=rgb_wrong.point_log_likelihood_ratios,
    )


@torch.no_grad()
def evaluate_prepared_queries(
    *,
    model: TargetFreePoseEvidenceWeighter,
    queries: Sequence[PreparedEvidenceWeighterQuery],
    image_size: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, object]:
    if not queries:
        raise ValueError("evidence weighter evaluation requires at least one query")
    records: list[dict[str, object]] = []
    for prepared in queries:
        prediction = model(prepared.features.values.to(model.feature_center.device))
        learned = _query_terms(prepared=prepared, weights=prediction.weights, image_size=image_size, args=args)
        uniform = torch.full_like(prediction.weights, 1.0 / float(len(prediction.weights)))
        baseline = _query_terms(prepared=prepared, weights=uniform, image_size=image_size, args=args)
        records.append(
            {
                "query_id": prepared.query_id,
                "normal_pose_gap": float(learned["normal_gap"].item()),
                "rgb_deranged_pose_gap": float(learned["deranged_gap"].item()),
                "normal_minus_rgb_deranged_pose_gap": float(learned["rgb_delta"].item()),
                "effective_sample_size": float(learned["ess"].item()),
                "coverage_divergence": float(learned["coverage"].item()),
                "uniform_mass": float(prediction.uniform_mass.item()),
                "uniform_normal_pose_gap": float(baseline["normal_gap"].item()),
                "uniform_rgb_deranged_pose_gap": float(baseline["deranged_gap"].item()),
                "uniform_normal_minus_rgb_deranged_pose_gap": float(baseline["rgb_delta"].item()),
            }
        )
    normal = np.asarray([float(record["normal_pose_gap"]) for record in records], dtype=np.float64)
    visual = np.asarray(
        [float(record["normal_minus_rgb_deranged_pose_gap"]) for record in records], dtype=np.float64
    )
    uniform = np.asarray([float(record["uniform_normal_pose_gap"]) for record in records], dtype=np.float64)
    result: dict[str, object] = {
        "query_count": len(records),
        "mean_normal_pose_gap": float(np.mean(normal)),
        "normal_pose_win_fraction": float(np.mean(normal > 0.0)),
        "mean_normal_minus_rgb_deranged_pose_gap": float(np.mean(visual)),
        "mean_effective_sample_size": float(
            np.mean([float(record["effective_sample_size"]) for record in records])
        ),
        "minimum_effective_sample_size": float(
            np.min([float(record["effective_sample_size"]) for record in records])
        ),
        "mean_coverage_divergence": float(
            np.mean([float(record["coverage_divergence"]) for record in records])
        ),
        "mean_uniform_mass": float(np.mean([float(record["uniform_mass"]) for record in records])),
        "records": records,
    }
    result.update(
        summarize_pose_gap_distribution(
            normal_pose_gaps=normal,
            normal_minus_rgb_deranged_pose_gaps=visual,
            uniform_normal_pose_gaps=uniform,
        )
    )
    result["uniform"] = {
        "mean_normal_pose_gap": float(np.mean(uniform)),
        "normal_pose_win_fraction": float(np.mean(uniform > 0.0)),
        "mean_normal_minus_rgb_deranged_pose_gap": float(
            np.mean(
                [float(record["uniform_normal_minus_rgb_deranged_pose_gap"]) for record in records]
            )
        ),
        "median_normal_pose_gap": float(np.median(uniform)),
        "p10_normal_pose_gap": float(np.quantile(uniform, 0.10)),
        "minimum_normal_pose_gap": float(np.min(uniform)),
    }
    return result


def inner_gate_decision(*, metrics: Mapping[str, object], args: argparse.Namespace) -> dict[str, bool]:
    baseline = metrics.get("uniform")
    if not isinstance(baseline, Mapping):
        raise ValueError("evidence weighter gate requires a uniform baseline")
    try:
        checks = {
            "mean_gap": float(metrics["mean_normal_pose_gap"])
            >= float(baseline["mean_normal_pose_gap"]) + float(args.minimum_mean_gap_improvement),
            "median_gap": float(metrics["median_normal_pose_gap"])
            >= float(baseline["median_normal_pose_gap"]) + float(args.minimum_median_gap_improvement),
            "win_fraction": float(metrics["normal_pose_win_fraction"])
            >= float(baseline["normal_pose_win_fraction"]),
            "p10_tail": float(metrics["p10_normal_pose_gap"])
            >= float(baseline["p10_normal_pose_gap"]) - float(args.maximum_p10_gap_degradation),
            "minimum_tail": float(metrics["minimum_normal_pose_gap"])
            >= float(baseline["minimum_normal_pose_gap"]) - float(args.maximum_minimum_gap_degradation),
            "no_uniform_win_to_loss": int(metrics["uniform_win_to_selector_loss_count"]) == 0,
            "rgb_visual_delta": float(metrics["mean_normal_minus_rgb_deranged_pose_gap"])
            >= float(baseline["mean_normal_minus_rgb_deranged_pose_gap"])
            + float(args.minimum_rgb_visual_gap_improvement),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("evidence weighter gate metrics are invalid") from error
    checks["passed"] = bool(all(checks.values()))
    return checks


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_training(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("evidence weighter requested CUDA but CUDA is unavailable")
    _seed_everything(int(args.seed))
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "candidate_pose_evidence_weighter.pt"
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.force):
        raise FileExistsError(f"evidence weighter output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = Path(args.rgb_spatial_layout)
    geometry_path = Path(args.geometry_training_targets)
    phase_path = Path(args.phase_checkpoint)
    hybrid_path = Path(args.hybrid_checkpoint)
    source_paths = _source_paths(args)
    if not all(path.is_file() for path in (layout_path, geometry_path, phase_path, hybrid_path, *source_paths.values())):
        raise FileNotFoundError("evidence weighter input is absent")
    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    targets = load_candidate_pose_rgb_spatial_training_targets(geometry_path)
    layout_sha = file_sha256_short(layout_path)
    geometry_sha = file_sha256_short(geometry_path)
    groups = build_train_query_groups(layout=layout, targets=targets)
    phase_checkpoint = _load_phase_checkpoint(phase_path)
    phase_partition, heldout_query_ids = _checkpoint_partition(
        phase_checkpoint, all_train_query_ids=tuple(sorted(groups))
    )
    train_query_ids = tuple(str(value) for value in phase_partition["inner_train"]["query_ids"])
    if not train_query_ids or set(train_query_ids).intersection(heldout_query_ids):
        raise ValueError("evidence weighter phase partition is invalid")
    state_dict, hybrid_metadata = _read_hybrid_checkpoint(hybrid_path)
    if hybrid_metadata.get("format") not in {None, HYBRID_CHECKPOINT_FORMAT}:
        raise ValueError("evidence weighter hybrid checkpoint format is invalid")
    lineage = hybrid_metadata.get("lineage")
    training = hybrid_metadata.get("training")
    if (
        not isinstance(lineage, Mapping)
        or str(lineage.get("layout_sha256", "")) != layout_sha
        or str(lineage.get("geometry_training_targets_sha256", "")) != geometry_sha
        or not isinstance(training, Mapping)
        or training.get("phase_train_query_partition") != phase_partition
    ):
        raise ValueError("evidence weighter parent hybrid lineage is stale")
    sources = load_context_attention_sources(
        radio_final_context_cache=source_paths["radio_final"],
        radio_intermediate_context_cache=source_paths["radio_intermediate"],
        alike_spatial_context_cache=source_paths["alike"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    image_ids, image_sizes, source_grids = _source_table(sources)
    source_manifest = str(sources[0].metadata.get("source_image_manifest_sha256", ""))
    if not source_manifest:
        raise ValueError("evidence weighter source image manifest is absent")
    _validate_phase_checkpoint(
        checkpoint=phase_checkpoint,
        checkpoint_path=phase_path,
        layout_path=layout_path,
        layout_metadata=layout.metadata,
        source_paths=source_paths,
        source_manifest_sha256=source_manifest,
    )
    phase_model = _load_phase_model(
        checkpoint=phase_checkpoint, source_grids=source_grids, image_sizes=image_sizes, device=device
    ).eval()
    rgb_model = _load_rgb_model(
        state_dict=state_dict, metadata=hybrid_metadata, image_sizes=image_sizes, device=device
    ).eval()
    unique_sizes = np.unique(image_sizes, axis=0)
    if unique_sizes.shape != (1, 2):
        raise ValueError("evidence weighter requires one processed coordinate image size")
    coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    rgb_image_size = _discover_rgb_image_size(image_root=Path(args.image_root), image_id=str(image_ids[0]))
    rgb_bridge = validate_rgb_coordinate_bridge(
        source_metadata=sources[0].metadata,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
    hybrid_config = hybrid_metadata.get("config")
    if not isinstance(hybrid_config, Mapping):
        raise ValueError("evidence weighter parent hybrid config is invalid")
    score_args = SimpleNamespace(
        identity_weight=float(hybrid_config["identity_weight"]),
        spatial_weight=float(hybrid_config["spatial_weight"]),
        max_abs_pose_log_ratio=float(hybrid_config["max_abs_pose_log_ratio"]),
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
    )
    cache_device = torch.device("cpu") if str(args.rgb_cache_device) == "cpu" else device
    prepared: dict[str, PreparedEvidenceWeighterQuery] = {}
    for index, query_id in enumerate(tuple(sorted(groups))):
        prepared[str(query_id)] = _prepare_query(
            group=groups[str(query_id)],
            complete_runtime=complete_runtime,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            phase_model=phase_model,
            rgb_model=rgb_model,
            cache=cache,
            cache_device=cache_device,
            device=device,
            score_args=score_args,
            support_permutation_shift=int(args.support_permutation_shift),
            target_free_feature_policy=str(args.target_free_feature_policy),
        )
        if (index + 1) % 8 == 0 or index + 1 == len(groups):
            print(
                json.dumps(
                    {"prepared_queries": int(index + 1), "total_queries": int(len(groups))}, sort_keys=True
                ),
                flush=True,
            )
    train_prepared = [prepared[query_id] for query_id in train_query_ids]
    heldout_prepared = [prepared[query_id] for query_id in heldout_query_ids]
    center, scale = fit_target_free_feature_normalizer([item.features for item in train_prepared])
    model = TargetFreePoseEvidenceWeighter(
        feature_center=center,
        feature_scale=scale,
        hidden_dim=int(args.hidden_dim),
        minimum_uniform_mass=float(args.minimum_uniform_mass),
        maximum_uniform_mass=float(args.maximum_uniform_mass),
        initial_uniform_mass=float(args.initial_uniform_mass),
        max_abs_local_logit=float(args.max_abs_local_logit),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay)
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        order = list(train_prepared)
        random.Random(int(args.seed) + epoch * 104729).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        tracked: dict[str, list[float]] = {
            "normal_gap": [],
            "uniform_regret": [],
            "uniform_win_safety": [],
            "rgb_delta": [],
            "ess": [],
            "coverage": [],
            "uniform_mass": [],
        }
        for item in order:
            prediction = model(item.features.values.to(device))
            terms = _query_terms(
                prepared=item, weights=prediction.weights, image_size=coordinate_image_size, args=args
            )
            losses.append(_objective(terms=terms, uniform_mass=prediction.uniform_mass, args=args))
            for name, value in (
                ("normal_gap", terms["normal_gap"]),
                ("uniform_regret", terms["uniform_regret"]),
                ("uniform_win_safety", terms["uniform_win_safety"]),
                ("rgb_delta", terms["rgb_delta"]),
                ("ess", terms["ess"]),
                ("coverage", terms["coverage"]),
                ("uniform_mass", prediction.uniform_mass[0]),
            ):
                tracked[name].append(float(value.detach().item()))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("evidence weighter emitted a nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip_norm))
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == int(args.epochs):
            record: dict[str, float | int] = {
                "epoch": int(epoch + 1),
                "train_total_loss": float(loss.detach().item()),
                "train_normal_pose_gap": float(np.mean(tracked["normal_gap"])),
                "train_uniform_regret": float(np.mean(tracked["uniform_regret"])),
                "train_uniform_win_safety": float(np.mean(tracked["uniform_win_safety"])),
                "train_normal_minus_rgb_deranged_pose_gap": float(np.mean(tracked["rgb_delta"])),
                "train_effective_sample_size": float(np.mean(tracked["ess"])),
                "train_coverage_divergence": float(np.mean(tracked["coverage"])),
                "train_uniform_mass": float(np.mean(tracked["uniform_mass"])),
            }
            history.append(record)
            print(json.dumps({"evidence_weighter_train": record}, sort_keys=True), flush=True)
    model.eval()
    heldout_metrics = evaluate_prepared_queries(
        model=model, queries=heldout_prepared, image_size=coordinate_image_size, args=args
    )
    gate = inner_gate_decision(metrics=heldout_metrics, args=args)
    metadata: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_EVIDENCE_WEIGHTER_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "runtime_layout_is_target_free": True,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "pnp_integration_allowed": False,
        "fixed_global_topl": True,
        "explicit_null": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "projection_after_network_only": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "selector_availability": "normal_runtime_phase_rgb_intersection_only",
        "paired_score_availability": "normal_deranged_common_intersection_only",
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
            "hidden_dim": int(args.hidden_dim),
            "minimum_uniform_mass": float(args.minimum_uniform_mass),
            "maximum_uniform_mass": float(args.maximum_uniform_mass),
            "initial_uniform_mass": float(args.initial_uniform_mass),
            "max_abs_local_logit": float(args.max_abs_local_logit),
            "target_free_feature_policy": str(args.target_free_feature_policy),
            "coverage_grid": [int(args.coverage_grid_rows), int(args.coverage_grid_columns)],
            "loss_weights": {
                "soft_hard": float(args.soft_hard_loss_weight),
                "rgb_appearance_control": float(args.rgb_appearance_control_loss_weight),
                "coverage": float(args.coverage_loss_weight),
                "effective_sample_size": float(args.effective_sample_size_loss_weight),
                "uniform_fallback": float(args.uniform_fallback_loss_weight),
                "uniform_regret": float(args.uniform_regret_loss_weight),
                "uniform_win_safety": float(args.uniform_win_safety_loss_weight),
            },
            "uniform_safety": {
                "maximum_train_uniform_gap_degradation": float(
                    args.maximum_train_uniform_gap_degradation
                ),
                "minimum_train_uniform_win_gap": float(args.minimum_train_uniform_win_gap),
            },
        },
        "lineage": {
            "layout_sha256": layout_sha,
            "geometry_training_targets_sha256": geometry_sha,
            "phase_checkpoint_sha256": file_sha256_short(phase_path),
            "hybrid_checkpoint_sha256": file_sha256_short(hybrid_path),
            "source_cache_sha256": {name: file_sha256_short(path) for name, path in source_paths.items()},
            "source_image_manifest_sha256": source_manifest,
            "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
            "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
            "rgb_coordinate_bridge": rgb_bridge,
        },
        "training": {
            "objective": "frozen_target_free_rgb_visual_statistics_to_continuous_all_token_weights_and_query_uniform_fallback_direct_correct_vs_coherent_wrong_pose_margin_with_train_only_uniform_regret_and_uniform_positive_safety_v2",
            "targets_materialized_only_after_visual_feature_forward": True,
            "heldout_phase_inner_validation_queries_never_enter_optimizer": True,
            "phase_train_query_partition": phase_partition,
            "parent_hybrid_train_only_gate_passed": hybrid_metadata.get("train_only_inner_gate_passed") is True,
            "epochs": int(args.epochs),
            "final_inner_validation_metrics": heldout_metrics,
            "final_inner_gate": gate,
            "seed": int(args.seed),
        },
    }
    _atomic_torch_save(
        checkpoint_path,
        {"format": CHECKPOINT_FORMAT, "state_dict": model.state_dict(), "metadata": metadata},
    )
    summary = {
        "stage": "train_candidate_pose_evidence_weighter",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256_short(checkpoint_path),
        "prepared_query_format": PREPARED_QUERY_FORMAT,
        "prepared_query_count": len(prepared),
        "final_inner_validation": heldout_metrics,
        "final_inner_gate": gate,
        "history": history,
        "rgb_cache": cache.summary(),
        "elapsed_seconds": float(time.time() - start),
        "next_step": (
            "frozen_outer_pose_rank_audit_only"
            if bool(gate["passed"])
            else "diagnostic_only_reject_weighter_and_redesign_target_free_evidence"
        ),
        "protocol": {
            "target_free_runtime": True,
            "targets_joined_after_visual_feature_forward": True,
            "fixed_global_topl_and_explicit_null": True,
            "no_render": True,
            "no_image_retrieval_or_submap": True,
            "no_pnp": True,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evidence_weighter": heldout_metrics, "gate": gate, "output": str(output_dir)}, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run_training(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
