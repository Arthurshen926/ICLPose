"""Train latent identity plus candidate-specific pose-alignment evidence.

The model sees only real RADIO final/intermediate and ALIKE feature crops.  A
train-only registered SfM observation target teaches a soft identity posterior
at each observed query coordinate.  Separately, train-only correct and
coherent-wrong pose pairs teach a candidate-specific alignment likelihood.

At runtime neither target source is loaded: identity is evaluated before a
pose is read, then its soft top-L posterior and token weights are frozen while
the alignment branch scores projected candidate locations.

Use both local GPUs with::

    torchrun --standalone --nproc_per_node=2 \
      feature_extract/tools/vfm/train_candidate_pose_latent_evidence.py ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from feature_extract.tools.vfm.train_candidate_pose_llr import (
    _DistributedState,
    _QueryRuntime,
    _finalize_distributed,
    _group_train_pairs_by_query,
    _initialize_distributed,
    _load_train_pairs,
    _partition_train_queries_for_inner_validation,
    _poses_for_query_group,
    _prepare_query_runtimes,
    _project_candidate_positions,
    _reduce_statistics,
    _source_manifest,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.candidate_pose_latent_evidence import (
    CANDIDATE_POSE_LATENT_EVIDENCE_FORMAT,
    CandidatePoseLatentEvidence,
    candidate_pose_point_log_mixture,
    direct_candidate_alignment_scores,
    same_track_alignment_margin_loss,
    weighted_pose_log_likelihood_ratio,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    validate_serialized_grouped_hypothesis_semantic_lineage,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    MixedVerificationPoints,
    load_mixed_verification_points,
    mixed_verification_points_scoring_compatibility,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


CHECKPOINT_FORMAT = "candidate_pose_latent_evidence_checkpoint_v1"
_METRIC_COUNT = 16


@dataclass(frozen=True)
class _IdentityTrainTargets:
    """Train-only exact-track classes aligned to every verification point."""

    target_classes: np.ndarray
    candidate_count: int
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        classes = np.asarray(self.target_classes, dtype=np.int64).reshape(-1)
        candidate_count = int(self.candidate_count)
        if (
            len(classes) == 0
            or candidate_count <= 1
            or np.any(classes < -1)
            or np.any(classes > candidate_count)
            or not isinstance(self.metadata, Mapping)
        ):
            raise ValueError("latent identity train targets are invalid")
        object.__setattr__(self, "target_classes", classes)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "metadata", dict(self.metadata))


def identity_target_class_masks(
    *, target_classes: torch.Tensor, candidate_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split train-only target classes without conflating ``-1`` and top-L."""

    targets = torch.as_tensor(target_classes, dtype=torch.long).reshape(-1)
    count = int(candidate_count)
    if (
        len(targets) == 0
        or count <= 1
        or torch.any(targets < -1)
        or torch.any(targets > count)
    ):
        raise ValueError("latent identity target classes are invalid")
    return (
        (targets >= 0) & (targets < count),
        targets == count,
        targets < 0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-points", required=True)
    parser.add_argument("--train-pairs", required=True)
    parser.add_argument("--maplet-support-index", required=True)
    parser.add_argument("--support-geometry-index", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--support-view-count", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-identity-residual", type=float, default=3.0)
    parser.add_argument("--max-abs-alignment-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--null-uniformity-weight", type=float, default=0.25)
    parser.add_argument("--alignment-loss-weight", type=float, default=1.0)
    parser.add_argument("--alignment-margin", type=float, default=0.25)
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument(
        "--development-query-limit",
        type=int,
        default=0,
        help="nonzero deterministic train-query prefix for smoke fits only",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _build_train_identity_targets(
    *,
    points: MixedVerificationPoints,
    colmap_model_dir: Path,
    registered_identity_radius_px: float,
) -> _IdentityTrainTargets:
    """Join only train image observations to the fixed top-L point layout."""

    radius = float(registered_identity_radius_px)
    candidate_count = int(points.candidate_track_ids.shape[1])
    if radius <= 0.0 or candidate_count <= 1:
        raise ValueError("latent identity target radius or candidate count is invalid")
    train_rows = np.flatnonzero(np.asarray(points.split_names).astype(str) == "train")
    if len(train_rows) == 0:
        raise ValueError("latent identity supervision has no train verification rows")
    images_path = Path(colmap_model_dir) / "images.bin"
    if not images_path.is_file():
        raise FileNotFoundError(f"latent identity model lacks {images_path}")
    images = read_colmap_images_binary(images_path)
    train_query_ids = set(np.asarray(points.query_ids)[train_rows].astype(str).tolist())
    images_by_name = {
        str(image.image_name): image
        for image in images.values()
        if str(image.image_name) in train_query_ids
    }
    targets = registered_query_observation_targets(
        query_ids=np.asarray(points.query_ids)[train_rows].astype(str),
        query_xy=np.asarray(points.xy, dtype=np.float32)[train_rows],
        images_by_name=images_by_name,
        max_distance_px=radius,
    )
    labels = registered_candidate_identity_labels(
        np.asarray(points.candidate_track_ids, dtype=np.int64)[train_rows], targets
    )
    membership = registered_candidate_identity_target_membership(
        np.asarray(points.candidate_track_ids, dtype=np.int64)[train_rows], targets
    )
    supervised = np.asarray(targets.supervised, dtype=bool)
    if not np.any(supervised):
        raise ValueError("latent identity supervision found no registered train observations")
    classes = np.full((len(points.query_ids),), -1, dtype=np.int64)
    selected_membership = membership[supervised]
    if selected_membership.shape != (int(np.sum(supervised)), candidate_count + 1):
        raise RuntimeError("latent identity membership has an invalid explicit-null layout")
    if np.any(np.sum(selected_membership, axis=1) != 1):
        raise RuntimeError("latent identity membership is not singleton-or-null")
    classes[train_rows[supervised]] = np.argmax(selected_membership, axis=1).astype(
        np.int64
    )
    exact_tensor, _fixed_null_tensor, _unlabelled_tensor = identity_target_class_masks(
        target_classes=torch.from_numpy(classes[train_rows]),
        candidate_count=candidate_count,
    )
    exact = exact_tensor.numpy()
    metadata = {
        "supervision": "registered_query_observation_exact_track_or_fixed_null_selector_v1",
        "registered_identity_radius_px": radius,
        "train_split_row_count": int(len(train_rows)),
        "registered_supervised_train_row_count": int(np.sum(supervised)),
        "registered_supervised_train_row_rate": float(np.mean(supervised)),
        "exact_track_retrieved_train_row_count": int(np.sum(exact)),
        "exact_track_retrieved_given_registered_train_rate": float(np.mean(exact[supervised])),
        "fixed_null_registered_train_row_count": int(
            np.sum(classes[train_rows][supervised] == candidate_count)
        ),
        "unlabelled_train_row_count": int(np.sum(classes[train_rows] < 0)),
        "registered_identity_target_coverage": summarize_registered_candidate_identity(
            labels, targets
        ),
        "colmap_images_sha256": file_sha256_short(images_path),
        "validation_or_test_identity_targets_loaded": False,
    }
    return _IdentityTrainTargets(
        target_classes=classes,
        candidate_count=candidate_count,
        metadata=metadata,
    )


def conditional_identity_supervision_loss(
    *,
    conditional_probabilities: torch.Tensor,
    target_classes: torch.Tensor,
    null_uniformity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit exact top-L identity and make known-null selector rows ambiguous.

    The explicit null mass is intentionally fixed by the frozen coarse
    posterior.  A registered track absent from top-L therefore cannot teach a
    nonexistent learned null probability.  It instead receives a uniform
    conditional-candidate objective, which trains the pose-independent token
    selector to avoid a spurious hard identity.
    """

    conditional = torch.as_tensor(conditional_probabilities, dtype=torch.float32)
    targets = torch.as_tensor(
        target_classes, dtype=torch.long, device=conditional.device
    ).reshape(-1)
    weight = float(null_uniformity_weight)
    if (
        conditional.ndim != 2
        or conditional.shape[0] == 0
        or targets.shape != (conditional.shape[0],)
        or conditional.shape[1] <= 1
        or not torch.isfinite(conditional).all()
        or torch.any(conditional < 0.0)
        or torch.any(torch.abs(conditional.sum(dim=1) - 1.0) > 1e-4)
        or torch.any(targets < -1)
        or torch.any(targets > conditional.shape[1])
        or not np.isfinite(weight)
        or weight < 0.0
    ):
        raise ValueError("latent conditional identity supervision inputs are invalid")
    candidate_count = int(conditional.shape[1])
    exact, fixed_null, unlabelled = identity_target_class_masks(
        target_classes=targets, candidate_count=candidate_count
    )
    log_probability = torch.log(conditional.clamp_min(torch.finfo(conditional.dtype).tiny))
    zero = conditional.sum() * 0.0
    exact_loss = (
        F.nll_loss(log_probability[exact], targets[exact]) if torch.any(exact) else zero
    )
    null_uniform_kl = (
        -log_probability[fixed_null].mean(dim=1).mean() - math.log(float(candidate_count))
        if torch.any(fixed_null)
        else zero
    )
    loss = exact_loss + weight * null_uniform_kl
    exact_count = int(exact.sum().item())
    top1_accuracy = (
        float(
            (
                conditional[exact].argmax(dim=1) == targets[exact]
            )
            .to(dtype=torch.float32)
            .mean()
            .item()
        )
        if exact_count
        else 0.0
    )
    return loss, {
        "exact_candidate_target_count": float(exact_count),
        "fixed_null_target_count": float(fixed_null.sum().item()),
        "unlabelled_target_count": float(unlabelled.sum().item()),
        "exact_candidate_nll": float(exact_loss.detach().item()),
        "fixed_null_uniform_kl": float(null_uniform_kl.detach().item()),
        "exact_candidate_top1_accuracy": top1_accuracy,
    }


def exact_identity_alignment_weights(
    *,
    selector_weights: torch.Tensor,
    target_classes: torch.Tensor,
    candidate_count: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return train-only alignment weights for tokens with a true top-L track."""

    selector = torch.as_tensor(selector_weights, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(
        target_classes, dtype=torch.long, device=selector.device
    ).reshape(-1)
    count = int(candidate_count)
    if (
        selector.shape != targets.shape
        or len(selector) == 0
        or count <= 1
        or not torch.isfinite(selector).all()
        or torch.any(selector < 0.0)
        or torch.any(targets < -1)
        or torch.any(targets > count)
    ):
        raise ValueError("latent alignment training weights are invalid")
    exact, _fixed_null, _unlabelled = identity_target_class_masks(
        target_classes=targets, candidate_count=count
    )
    weights = selector.detach() * exact.to(dtype=selector.dtype)
    return weights, {
        "exact_identity_alignment_token_count": float(exact.sum().item()),
        "exact_identity_alignment_weight_sum": float(weights.sum().item()),
    }


def direct_candidate_alignment_margin_loss(
    *,
    candidate_alignment: torch.Tensor,
    target_classes: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train pose alignment on known candidate edges before runtime mixing.

    This target-bearing objective is intentionally separate from the runtime
    soft top-L/null mixture.  It gives a correct landmark edge a usable
    correct-versus-wrong gradient instead of reducing it by every unrelated
    candidate's fixed prior mass.  The scorer never imports this function.
    """

    scores, exact = direct_candidate_alignment_scores(
        candidate_alignment=candidate_alignment,
        target_classes=target_classes,
    )
    loss, metrics = same_track_alignment_margin_loss(
        correct_alignment=scores[:1],
        coherent_wrong_alignment=scores[1:].reshape(1, -1),
        margin=float(margin),
    )
    return loss, {
        **metrics,
        "exact_candidate_token_count": float(exact.sum().item()),
    }


def _query_loss(
    *,
    model: torch.nn.Module,
    query: _QueryRuntime,
    observed_xy: np.ndarray,
    target_classes: np.ndarray,
    poses_w2c: np.ndarray,
    device: torch.device,
    identity_loss_weight: float,
    null_uniformity_weight: float,
    alignment_loss_weight: float,
    alignment_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one query-grouped train or inner-validation objective."""

    targets = torch.as_tensor(target_classes, dtype=torch.long, device=device).reshape(-1)
    observed = torch.as_tensor(observed_xy, dtype=torch.float32, device=device)
    poses = torch.as_tensor(poses_w2c, dtype=torch.float32, device=device)
    candidate_count = int(query.runtime.candidate_probabilities.shape[1])
    if (
        observed.shape != (len(targets), 2)
        or targets.shape != (len(query.runtime.query_image_indices),)
        or poses.ndim != 3
        or poses.shape[0] < 2
        or poses.shape[1:] != (4, 4)
        or candidate_count <= 1
    ):
        raise ValueError("latent query loss inputs are incompatible")
    projected_xy, projected_valid = _project_candidate_positions(
        query=query, poses_w2c=poses, device=device
    )
    alignment_mask, _fixed_null, _unlabelled = identity_target_class_masks(
        target_classes=targets, candidate_count=candidate_count
    )
    identity, pose, has_alignment_tokens = model(
        runtime=query.runtime,
        observed_xy=observed,
        candidate_query_xy=projected_xy,
        candidate_projection_valid=projected_valid,
        alignment_selector_mask=alignment_mask,
    )
    identity_loss, identity_metrics = conditional_identity_supervision_loss(
        conditional_probabilities=identity.conditional_probabilities,
        target_classes=targets,
        null_uniformity_weight=float(null_uniformity_weight),
    )
    weights, alignment_weight_metrics = exact_identity_alignment_weights(
        selector_weights=identity.selector_weights,
        target_classes=targets,
        candidate_count=candidate_count,
    )
    if bool(has_alignment_tokens.detach().item()) != bool(weights.sum().item() > 0.0):
        raise RuntimeError("latent identity and alignment selector masks diverged")
    if bool(has_alignment_tokens.detach().item()):
        alignment_loss, alignment_metrics = direct_candidate_alignment_margin_loss(
            candidate_alignment=pose.candidate_log_likelihood_ratios,
            target_classes=targets,
            margin=float(alignment_margin),
        )
        # ``pose`` was produced with the train-only exact-track mask so that
        # direct supervision has a bounded group layout.  Reuse its immutable
        # candidate edge tensor, but recompute this diagnostic with the full
        # target-free identity selector used by scoring.
        runtime_point = candidate_pose_point_log_mixture(
            candidate_alignment=pose.candidate_log_likelihood_ratios.detach(),
            candidate_probabilities=identity.candidate_probabilities.detach(),
            null_probabilities=identity.null_probabilities.detach(),
        )
        runtime_pose = weighted_pose_log_likelihood_ratio(
            point_log_likelihood_ratios=runtime_point,
            selector_weights=identity.selector_weights.detach(),
        )
        runtime_alignment_loss, runtime_alignment_metrics = same_track_alignment_margin_loss(
            correct_alignment=runtime_pose[:1],
            coherent_wrong_alignment=runtime_pose[1:].reshape(1, -1),
            margin=float(alignment_margin),
        )
        alignment_active = 1.0
    else:
        # The forward still ran the alignment branch so DDP sees every
        # parameter; this query simply has no true top-L identity to supervise
        # its correct-versus-wrong alignment margin.
        alignment_loss = pose.pose_log_likelihood_ratios.sum() * 0.0
        alignment_metrics = {
            "correct_win_fraction": 0.0,
            "mean_correct_minus_hardest_wrong": 0.0,
            "exact_candidate_token_count": 0.0,
        }
        runtime_alignment_loss = pose.pose_log_likelihood_ratios.sum() * 0.0
        runtime_alignment_metrics = {
            "correct_win_fraction": 0.0,
            "mean_correct_minus_hardest_wrong": 0.0,
        }
        alignment_active = 0.0
    total = (
        float(identity_loss_weight) * identity_loss
        + float(alignment_loss_weight) * alignment_loss
    )
    return total, {
        "identity_loss": float(identity_loss.detach().item()),
        "identity_exact_candidate_nll": float(identity_metrics["exact_candidate_nll"]),
        "identity_fixed_null_uniform_kl": float(identity_metrics["fixed_null_uniform_kl"]),
        "identity_exact_candidate_target_count": float(
            identity_metrics["exact_candidate_target_count"]
        ),
        "identity_fixed_null_target_count": float(identity_metrics["fixed_null_target_count"]),
        "identity_unlabelled_target_count": float(identity_metrics["unlabelled_target_count"]),
        "identity_exact_candidate_correct_count": float(
            identity_metrics["exact_candidate_target_count"]
            * identity_metrics["exact_candidate_top1_accuracy"]
        ),
        "alignment_loss": float(alignment_loss.detach().item()),
        "alignment_mean_correct_minus_hardest_wrong": float(
            alignment_metrics["mean_correct_minus_hardest_wrong"]
        ),
        "alignment_correct_win_fraction": float(alignment_metrics["correct_win_fraction"]),
        "runtime_alignment_loss": float(runtime_alignment_loss.detach().item()),
        "runtime_alignment_mean_correct_minus_hardest_wrong": float(
            runtime_alignment_metrics["mean_correct_minus_hardest_wrong"]
        ),
        "runtime_alignment_correct_win_fraction": float(
            runtime_alignment_metrics["correct_win_fraction"]
        ),
        "alignment_active": alignment_active,
        "alignment_selector_weight_sum": float(
            alignment_weight_metrics["exact_identity_alignment_weight_sum"]
        ),
    }


def _metric_tensor(
    *, loss: torch.Tensor, metrics: Mapping[str, float], device: torch.device
) -> torch.Tensor:
    values = (
        float(loss.detach().item()),
        float(metrics["identity_loss"]),
        float(metrics["identity_exact_candidate_nll"]),
        float(metrics["identity_fixed_null_uniform_kl"]),
        float(metrics["identity_exact_candidate_target_count"]),
        float(metrics["identity_fixed_null_target_count"]),
        float(metrics["identity_unlabelled_target_count"]),
        float(metrics["identity_exact_candidate_correct_count"]),
        float(metrics["alignment_loss"]),
        float(metrics["alignment_mean_correct_minus_hardest_wrong"]),
        float(metrics["alignment_correct_win_fraction"]),
        float(metrics["runtime_alignment_loss"]),
        float(metrics["runtime_alignment_mean_correct_minus_hardest_wrong"]),
        float(metrics["runtime_alignment_correct_win_fraction"]),
        float(metrics["alignment_active"]),
        1.0,
    )
    if len(values) != _METRIC_COUNT or not np.isfinite(values).all():
        raise ValueError("latent training metrics are non-finite")
    return torch.tensor(values, dtype=torch.float64, device=device)


def _summarize_metric_tensor(values: torch.Tensor) -> dict[str, float]:
    stats = torch.as_tensor(values, dtype=torch.float64).detach().cpu().numpy()
    if stats.shape != (_METRIC_COUNT,) or not np.isfinite(stats).all() or stats[-1] <= 0.0:
        raise ValueError("latent metric accumulator is invalid")
    query_count = float(stats[-1])
    active = float(stats[14])
    exact_count = float(stats[4])
    return {
        "query_count": query_count,
        "total_loss": float(stats[0] / query_count),
        "identity_loss": float(stats[1] / query_count),
        "identity_exact_candidate_nll": float(stats[2] / query_count),
        "identity_fixed_null_uniform_kl": float(stats[3] / query_count),
        "identity_exact_candidate_target_count": exact_count,
        "identity_fixed_null_target_count": float(stats[5]),
        "identity_unlabelled_target_count": float(stats[6]),
        "identity_exact_candidate_top1_accuracy": (
            0.0 if exact_count <= 0.0 else float(stats[7] / exact_count)
        ),
        "alignment_active_query_count": active,
        "alignment_loss": 1e9 if active <= 0.0 else float(stats[8] / active),
        "alignment_mean_correct_minus_hardest_wrong": (
            0.0 if active <= 0.0 else float(stats[9] / active)
        ),
        "alignment_correct_win_fraction": (
            0.0 if active <= 0.0 else float(stats[10] / active)
        ),
        "runtime_alignment_loss": 1e9 if active <= 0.0 else float(stats[11] / active),
        "runtime_alignment_mean_correct_minus_hardest_wrong": (
            0.0 if active <= 0.0 else float(stats[12] / active)
        ),
        "runtime_alignment_correct_win_fraction": (
            0.0 if active <= 0.0 else float(stats[13] / active)
        ),
    }


@torch.no_grad()
def _evaluate_inner_validation(
    *,
    model: CandidatePoseLatentEvidence,
    query_runtimes: Mapping[str, _QueryRuntime],
    query_xy: Mapping[str, np.ndarray],
    query_targets: Mapping[str, np.ndarray],
    pairs: object,
    groups: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    device: torch.device,
    identity_loss_weight: float,
    null_uniformity_weight: float,
    alignment_loss_weight: float,
    alignment_margin: float,
    amp_enabled: bool,
) -> dict[str, float]:
    if not query_ids:
        raise ValueError("latent inner validation has no query IDs")
    totals = torch.zeros((_METRIC_COUNT,), dtype=torch.float64, device=device)
    model.eval()
    for query_id in query_ids:
        query = query_runtimes.get(str(query_id))
        if query is None or str(query_id) not in query_xy or str(query_id) not in query_targets:
            raise ValueError("latent inner validation query runtime is missing")
        poses = _poses_for_query_group(pairs=pairs, pair_indices=groups[str(query_id)])
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss, metrics = _query_loss(
                model=model,
                query=query,
                observed_xy=query_xy[str(query_id)],
                target_classes=query_targets[str(query_id)],
                poses_w2c=poses,
                device=device,
                identity_loss_weight=float(identity_loss_weight),
                null_uniformity_weight=float(null_uniformity_weight),
                alignment_loss_weight=float(alignment_loss_weight),
                alignment_margin=float(alignment_margin),
            )
        totals += _metric_tensor(loss=loss, metrics=metrics, device=device)
    summary = _summarize_metric_tensor(totals)
    if summary["alignment_active_query_count"] <= 0.0:
        raise RuntimeError("latent inner validation has no exact top-L alignment tokens")
    return summary


def _is_better_inner_validation_epoch(
    *, candidate: Mapping[str, float], incumbent: Mapping[str, float] | None
) -> bool:
    """Select by held-out-train pose discrimination before auxiliary identity loss."""

    for key in (
        "alignment_loss",
        "runtime_alignment_loss",
        "alignment_correct_win_fraction",
        "identity_loss",
    ):
        if not np.isfinite(float(candidate[key])):
            raise ValueError("latent inner validation metric is non-finite")
    if incumbent is None:
        return True
    if float(candidate["alignment_loss"]) < float(incumbent["alignment_loss"]) - 1e-12:
        return True
    if abs(float(candidate["alignment_loss"]) - float(incumbent["alignment_loss"])) <= 1e-12:
        if float(candidate["alignment_correct_win_fraction"]) > float(
            incumbent["alignment_correct_win_fraction"]
        ):
            return True
        if abs(
            float(candidate["alignment_correct_win_fraction"])
            - float(incumbent["alignment_correct_win_fraction"])
        ) <= 1e-12:
            return float(candidate["identity_loss"]) < float(incumbent["identity_loss"])
    return False


def train_candidate_pose_latent_evidence(args: argparse.Namespace) -> dict[str, object]:
    """Run DDP-safe train-only fitting and save a target-free checkpoint."""

    state = _initialize_distributed(str(args.device))
    try:
        positive = (
            float(args.learning_rate),
            float(args.max_abs_identity_residual),
            float(args.max_abs_alignment_log_ratio),
            float(args.registered_identity_radius_px),
        )
        nonnegative = (
            float(args.weight_decay),
            float(args.identity_loss_weight),
            float(args.null_uniformity_weight),
            float(args.alignment_loss_weight),
            float(args.alignment_margin),
            float(args.gradient_clip_norm),
        )
        if (
            int(args.epochs) <= 0
            or int(args.support_view_count) != 2
            or int(args.edge_chunk_size) <= 0
            or int(args.hidden_dim) < 4
            or int(args.development_query_limit) < 0
            or any(not np.isfinite(value) or value <= 0.0 for value in positive)
            or any(not np.isfinite(value) or value < 0.0 for value in nonnegative)
        ):
            raise ValueError("latent evidence training arguments are invalid")
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_latent_evidence.pt"
        history_path = output_dir / "history.json"
        if state.rank == 0 and (checkpoint_path.exists() or history_path.exists()) and not bool(args.force):
            raise FileExistsError("refusing to overwrite latent evidence output")
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
        except AttributeError:  # pragma: no cover - older torch.
            pass

        points_path = Path(args.verification_points)
        points = load_mixed_verification_points(points_path)
        if (
            points.metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
            or points.metadata.get("contains_ground_truth") is not False
            or points.metadata.get("pose_or_ground_truth_used") is not False
            or points.metadata.get("render") is not False
            or points.metadata.get("image_retrieval_or_submap_used") is not False
        ):
            raise ValueError("latent evidence points violate the real-image target-free contract")
        pairs_path = Path(args.train_pairs)
        pairs = _load_train_pairs(pairs_path)
        lineage = pairs.metadata.get("hypothesis_semantic_lineage")
        try:
            validate_serialized_grouped_hypothesis_semantic_lineage(lineage)
        except ValueError as exc:
            raise ValueError("latent evidence train pairs lack current hypothesis lineage") from exc
        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        query_runtimes = _prepare_query_runtimes(
            points=points,
            sources=sources,
            maplet_support_index=Path(args.maplet_support_index),
            support_geometry_index=Path(args.support_geometry_index),
            projected_landmark_bank=Path(args.projected_landmark_bank),
            colmap_model_dir=Path(args.colmap_model_dir),
            support_view_count=int(args.support_view_count),
            required_split="train",
        )
        identity_targets = _build_train_identity_targets(
            points=points,
            colmap_model_dir=Path(args.colmap_model_dir),
            registered_identity_radius_px=float(args.registered_identity_radius_px),
        )
        groups = _group_train_pairs_by_query(pairs)
        all_query_ids = tuple(sorted(groups))
        if int(args.development_query_limit) > 0:
            all_query_ids = all_query_ids[: int(args.development_query_limit)]
        if len(all_query_ids) < 2:
            raise ValueError("latent evidence needs at least two train query groups")
        missing = sorted(set(all_query_ids).difference(query_runtimes))
        if missing:
            raise ValueError(f"latent train query lacks a runtime: {missing[:5]}")
        query_xy: dict[str, np.ndarray] = {}
        query_targets: dict[str, np.ndarray] = {}
        for query_id in all_query_ids:
            rows = points.rows_for_query(query_id)
            if len(rows) == 0 or np.any(points.split_names[rows] != "train"):
                raise ValueError("latent train query does not map to train-only points")
            query_xy[query_id] = np.asarray(points.xy[rows], dtype=np.float32)
            query_targets[query_id] = np.asarray(
                identity_targets.target_classes[rows], dtype=np.int64
            )
            if len(query_xy[query_id]) != len(query_runtimes[query_id].runtime.query_image_indices):
                raise RuntimeError("latent train query point ordering differs from its runtime")
        inner_train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=all_query_ids,
            fold_count=int(args.inner_validation_fold_count),
            fold_index=int(args.inner_validation_fold_index),
        )

        source_by_name = {str(source.name): source for source in sources}
        model = CandidatePoseLatentEvidence(
            sources={
                name: torch.from_numpy(np.asarray(source.grid, dtype=np.float32))
                for name, source in source_by_name.items()
            },
            image_sizes=torch.from_numpy(
                np.asarray(source_by_name["radio_final"].image_sizes, dtype=np.float32)
            ),
            hidden_dim=int(args.hidden_dim),
            max_abs_identity_residual=float(args.max_abs_identity_residual),
            max_abs_alignment_log_ratio=float(args.max_abs_alignment_log_ratio),
            edge_chunk_size=int(args.edge_chunk_size),
            activation_checkpointing=True,
        ).to(state.device)
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
        assert isinstance(core_model, CandidatePoseLatentEvidence)
        optimizer = torch.optim.AdamW(
            model_for_train.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        steps_per_rank = int(math.ceil(len(inner_train_ids) / state.world_size))
        history: list[dict[str, float]] = []
        best_inner_validation: dict[str, float] | None = None
        best_epoch = -1
        best_state_dict: dict[str, torch.Tensor] | None = None
        start_time = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            order = np.random.default_rng(int(args.seed) + epoch).permutation(
                len(inner_train_ids)
            )
            local_ids = tuple(
                inner_train_ids[
                    int(order[(state.rank + step * state.world_size) % len(order)])
                ]
                for step in range(steps_per_rank)
            )
            totals = torch.zeros((_METRIC_COUNT,), dtype=torch.float64, device=state.device)
            for query_id in local_ids:
                query = query_runtimes[query_id]
                poses = _poses_for_query_group(
                    pairs=pairs, pair_indices=groups[query_id]
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss, metrics = _query_loss(
                        model=model_for_train,
                        query=query,
                        observed_xy=query_xy[query_id],
                        target_classes=query_targets[query_id],
                        poses_w2c=poses,
                        device=state.device,
                        identity_loss_weight=float(args.identity_loss_weight),
                        null_uniformity_weight=float(args.null_uniformity_weight),
                        alignment_loss_weight=float(args.alignment_loss_weight),
                        alignment_margin=float(args.alignment_margin),
                    )
                scaler.scale(loss).backward()
                if float(args.gradient_clip_norm) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_for_train.parameters(), float(args.gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
                totals += _metric_tensor(loss=loss, metrics=metrics, device=state.device)
            totals = _reduce_statistics(state, totals)
            epoch_metrics = {"epoch": float(epoch + 1)}
            epoch_metrics.update(
                {f"train_{key}": value for key, value in _summarize_metric_tensor(totals).items()}
            )
            if state.enabled:
                distributed.barrier()
            if state.rank == 0:
                inner_validation = _evaluate_inner_validation(
                    model=core_model,
                    query_runtimes=query_runtimes,
                    query_xy=query_xy,
                    query_targets=query_targets,
                    pairs=pairs,
                    groups=groups,
                    query_ids=inner_validation_ids,
                    device=state.device,
                    identity_loss_weight=float(args.identity_loss_weight),
                    null_uniformity_weight=float(args.null_uniformity_weight),
                    alignment_loss_weight=float(args.alignment_loss_weight),
                    alignment_margin=float(args.alignment_margin),
                    amp_enabled=amp_enabled,
                )
                epoch_metrics.update(
                    {f"inner_validation_{key}": value for key, value in inner_validation.items()}
                )
                if _is_better_inner_validation_epoch(
                    candidate=inner_validation, incumbent=best_inner_validation
                ):
                    best_inner_validation = dict(inner_validation)
                    best_epoch = int(epoch + 1)
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in core_model.state_dict().items()
                    }
                history.append(epoch_metrics)
                print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()
        if state.rank == 0:
            if best_state_dict is None or best_inner_validation is None or best_epoch < 1:
                raise RuntimeError("latent evidence did not select an inner validation checkpoint")
            output_dir.mkdir(parents=True, exist_ok=True)
            scoring_inputs = {
                "verification_points": points_path,
                "maplet_support_index": Path(args.maplet_support_index),
                "support_geometry_index": Path(args.support_geometry_index),
                "projected_landmark_bank": Path(args.projected_landmark_bank),
                "radio_final_context_cache": Path(args.radio_final_context_cache),
                "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
                "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
                "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
                "colmap_images_bin": Path(args.colmap_model_dir) / "images.bin",
            }
            metadata = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_LATENT_EVIDENCE_FORMAT,
                "architecture": "full_2d_radio_final_plus_intermediate_plus_alike_identity_posterior_then_static_soft_candidate_pose_alignment_v1",
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "training_target_source_is_train_only": True,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "raw_scores_must_not_feed_pnp": True,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "fixed_global_topl": True,
                "fixed_candidate_top_k": int(identity_targets.candidate_count),
                "fixed_support_view_count": int(args.support_view_count),
                "candidate_reselection_per_pose": False,
                "support_reselection_per_pose": False,
                "explicit_null": True,
                "identity_nonnull_null_mass": "fixed_from_frozen_coarse_prior",
                "identity_candidate_mixture": "soft_observed_coordinate_posterior_detached_before_pose",
                "alignment_token_weights": "identity_max_conditional_posterior_detached_before_pose",
                "alignment_training_mask": "exact_registered_track_present_in_fixed_topl_train_only",
                "known_null_training": "conditional_candidate_uniformity_without_learned_null_mass",
                "runtime_excludes": [
                    "pose_matrix",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "coarse_posterior_as_neural_input",
                    "train_alignment_selector_mask",
                ],
                "hypothesis_semantic_lineage": lineage,
                "hidden_dim": int(args.hidden_dim),
                "max_abs_identity_residual": float(args.max_abs_identity_residual),
                "max_abs_alignment_log_ratio": float(args.max_abs_alignment_log_ratio),
                "edge_chunk_size": int(args.edge_chunk_size),
                "activation_checkpointing_during_training": True,
                "training": {
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "identity_loss_weight": float(args.identity_loss_weight),
                    "null_uniformity_weight": float(args.null_uniformity_weight),
                    "alignment_loss_weight": float(args.alignment_loss_weight),
                    "alignment_margin": float(args.alignment_margin),
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "development_query_limit": int(args.development_query_limit),
                    "inner_validation": {
                        "split": "train_query_only",
                        "fold_count": int(args.inner_validation_fold_count),
                        "fold_index": int(args.inner_validation_fold_index),
                        "selected_epoch": int(best_epoch),
                        "selected_metrics": best_inner_validation,
                    },
                },
                "train_identity_target_audit": identity_targets.metadata,
                "verification_points_scoring_compatibility": (
                    mixed_verification_points_scoring_compatibility(points.metadata)
                ),
                "inputs": _source_manifest(scoring_inputs),
                "train_only_inputs": _source_manifest(
                    {
                        "train_pairs": pairs_path,
                        "registered_identity_colmap_images_bin": Path(args.colmap_model_dir)
                        / "images.bin",
                    }
                ),
                "train_pair_count": int(len(pairs.query_ids)),
                "train_query_count": int(len(all_query_ids)),
                "inner_train_query_count": int(len(inner_train_ids)),
                "inner_validation_query_count": int(len(inner_validation_ids)),
            }
            torch.save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "state_dict": best_state_dict,
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            summary: dict[str, object] = {
                "stage": "train_candidate_pose_latent_evidence",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - start_time),
                "history": history,
                "checkpoint_selection": {
                    "selected_epoch": int(best_epoch),
                    "inner_validation": best_inner_validation,
                },
                "protocol": {
                    "train_only_query_grouped_supervision": True,
                    "target_free_runtime_scorer_required": True,
                    "support_descriptor_permutation_control_required": True,
                    "pnp_integration_forbidden_until_validation_and_late_audit_gate": True,
                },
            }
            history_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return (
            json.loads(history_path.read_text())
            if state.rank == 0
            else {"stage": "train_candidate_pose_latent_evidence", "rank": state.rank}
        )
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_candidate_pose_latent_evidence(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
