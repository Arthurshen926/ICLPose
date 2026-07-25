"""Train a target-free, permutation-equivariant top-L identity posterior.

The model sees only real-image RADIO-final/intermediate and ALIKE phase
fields for a fixed query/support/candidate layout.  Exact candidate labels and
current coherent-repeat failures are joined strictly after the visual forward.
The inner gate first asks whether the visual posterior improves candidate
identity itself; it never promotes a checkpoint directly to PnP.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.train_candidate_multiscale_phase_identity_llr import (
    ExactIdentityQueryTargets,
    PhaseIdentityBatch,
    _rank_query_batches,
    build_exact_identity_query_targets,
    build_phase_identity_batch,
    filter_static_hard_repeat_groups_to_registered_exact_identity,
    train_query_partition_manifest,
    validate_external_frozen_current_hard_targets,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _finalize_distributed,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _slice_runtime,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_group_phase_identity_llr import (
    CANDIDATE_GROUP_PHASE_IDENTITY_LLR_FORMAT,
    CandidateGroupPhaseIdentityLLR,
    CandidateGroupPhaseIdentityPrediction,
    candidate_group_identity_plus_null_logits,
    current_group_hard_repeat_identity_margin_loss,
    exact_group_identity_or_null_cross_entropy,
)
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    canonical_registered_identity_or_null_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
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


CHECKPOINT_FORMAT = "candidate_group_phase_identity_llr_checkpoint_v1"
GATE_FORMAT = "candidate_group_phase_identity_static_hard_and_rank_gate_v1"
FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--geometry-training-targets", required=True)
    parser.add_argument("--registered-identity-targets", required=True)
    parser.add_argument("--current-hard-repeat-targets", required=True)
    parser.add_argument("--current-hard-mining-checkpoint", required=True)
    parser.add_argument("--static-hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-embedding-dim", type=int, default=48)
    parser.add_argument("--group-embedding-dim", type=int, default=96)
    parser.add_argument("--max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument("--source-storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--learn-null-log-likelihood",
        action="store_true",
        help="Diagnostic-only null calibration; disabled by default during candidate-rank gating.",
    )
    parser.add_argument("--queries-per-step", type=int, default=2)
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument("--max-current-hard-edges-per-query", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-prior-logit-weight", type=float, default=1.0)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-hard-margin", type=float, default=0.25)
    parser.add_argument("--inner-fold-count", type=int, default=5)
    parser.add_argument("--inner-fold-index", type=int, default=1)
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--minimum-hard-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--minimum-hard-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-hard-gap", type=float, default=0.05)
    parser.add_argument("--minimum-hard-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--minimum-identity-top1-lift", type=float, default=0.02)
    parser.add_argument("--minimum-identity-mean-rank-reduction", type=float, default=0.25)
    parser.add_argument("--minimum-identity-correct-probability-lift", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        args.source_embedding_dim,
        args.group_embedding_dim,
        args.queries_per_step,
        args.max_points_per_query,
        args.epochs,
        args.inner_fold_count,
    )
    values = (
        args.max_abs_log_ratio,
        args.learning_rate,
        args.weight_decay,
        args.identity_prior_logit_weight,
        args.identity_loss_weight,
        args.current_hard_loss_weight,
        args.current_hard_margin,
        args.minimum_hard_eligible_query_fraction,
        args.minimum_hard_win_fraction,
        args.minimum_hard_gap,
        args.minimum_hard_visual_gap_delta,
        args.minimum_identity_top1_lift,
        args.minimum_identity_mean_rank_reduction,
        args.minimum_identity_correct_probability_lift,
        args.gradient_clip_norm,
    )
    if (
        any(int(value) <= 0 for value in positive_ints)
        or int(args.max_points_per_query) < 4
        or int(args.max_current_hard_edges_per_query) < 0
        or int(args.support_permutation_shift) <= 0
        or not 0 <= int(args.inner_fold_index) < int(args.inner_fold_count)
        or not all(math.isfinite(float(value)) for value in values)
        or float(args.max_abs_log_ratio) <= 0.0
        or float(args.learning_rate) <= 0.0
        or any(float(value) < 0.0 for value in values[2:])
        or not 0.0 < float(args.minimum_hard_eligible_query_fraction) <= 1.0
        or not 0.0 <= float(args.minimum_hard_win_fraction) <= 1.0
        or float(args.identity_loss_weight) <= 0.0
        or float(args.current_hard_loss_weight) <= 0.0
    ):
        raise ValueError("candidate-group identity trainer arguments are invalid")


def _source_table(
    sources: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor], str]:
    by_name = {str(getattr(source, "name")): source for source in sources}
    if set(by_name) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("candidate-group identity sources are incomplete")
    reference = by_name["radio_final"]
    image_ids = np.asarray(getattr(reference, "image_ids")).astype(str)
    image_sizes = np.asarray(getattr(reference, "image_sizes"), dtype=np.int64)
    if len(image_ids) == 0 or image_sizes.shape != (len(image_ids), 2):
        raise ValueError("candidate-group identity source image table is invalid")
    grids: dict[str, torch.Tensor] = {}
    for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        source = by_name[name]
        if not np.array_equal(np.asarray(getattr(source, "image_ids")).astype(str), image_ids) or not np.array_equal(
            np.asarray(getattr(source, "image_sizes"), dtype=np.int64), image_sizes
        ):
            raise ValueError("candidate-group identity source image ownership differs")
        grids[name] = torch.from_numpy(np.asarray(getattr(source, "grid"), dtype=np.float32))
    metadata = getattr(reference, "metadata")
    manifest = str(metadata.get("source_image_manifest_sha256", "")) if isinstance(metadata, Mapping) else ""
    if not manifest:
        raise ValueError("candidate-group identity source image manifest is absent")
    return image_ids, image_sizes, grids, manifest


def _reduce_sum(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _atomic_json_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _group_training_loss(
    *,
    model: torch.nn.Module,
    batch: PhaseIdentityBatch,
    device: torch.device,
    identity_prior_logit_weight: float,
    identity_loss_weight: float,
    current_hard_loss_weight: float,
    current_hard_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = model(runtime=batch.runtime)
    if not isinstance(prediction, CandidateGroupPhaseIdentityPrediction):
        raise RuntimeError("candidate-group identity model returned an unexpected prediction")
    identity_loss, identity_metrics = exact_group_identity_or_null_cross_entropy(
        runtime=batch.runtime,
        prediction=prediction,
        observed_candidate_mask=batch.observed_candidate_mask.to(device=device),
        candidate_dustbin_mask=batch.candidate_dustbin_mask.to(device=device),
        candidate_supervised_mask=batch.candidate_supervised_mask.to(device=device),
        candidate_prior_logit_weight=float(identity_prior_logit_weight),
        balance_observed_and_null=True,
    )
    if len(batch.current_hard_point_indices):
        hard_loss, hard_metrics = current_group_hard_repeat_identity_margin_loss(
            runtime=batch.runtime,
            prediction=prediction,
            point_indices=batch.current_hard_point_indices,
            positive_candidate_indices=batch.current_hard_positive_candidate_indices,
            negative_candidate_indices=batch.current_hard_negative_candidate_indices,
            margin=float(current_hard_margin),
            candidate_prior_logit_weight=float(identity_prior_logit_weight),
        )
    else:
        hard_loss = prediction.candidate_log_likelihood_ratios.sum() * 0.0
        hard_metrics = {
            "hard_repeat_active": 0.0,
            "hard_repeat_margin_loss": 0.0,
            "hard_repeat_mean_gap": 0.0,
            "hard_repeat_win_fraction": 0.0,
        }
    total = float(identity_loss_weight) * identity_loss + float(current_hard_loss_weight) * hard_loss
    if not torch.isfinite(total):
        raise RuntimeError("candidate-group identity loss became non-finite")
    return total, {
        "identity_loss": float(identity_loss.detach().item()),
        **{str(key): float(value) for key, value in identity_metrics.items()},
        **{str(key): float(value) for key, value in hard_metrics.items()},
        "total_loss": float(total.detach().item()),
    }


def group_identity_gate_decision(
    *,
    hard_metrics: Mapping[str, float],
    rank_metrics: Mapping[str, float],
    minimum_hard_eligible_query_fraction: float,
    minimum_hard_win_fraction: float,
    minimum_hard_gap: float,
    minimum_hard_visual_gap_delta: float,
    minimum_identity_top1_lift: float,
    minimum_identity_mean_rank_reduction: float,
    minimum_identity_correct_probability_lift: float,
) -> dict[str, object]:
    """Require visual hard-repeat evidence and real posterior rank improvement."""

    required_hard = (
        "eligible_query_fraction",
        "normal_win_fraction",
        "normal_mean_positive_minus_negative",
        "permuted_mean_positive_minus_negative",
    )
    required_rank = (
        "base_top1",
        "group_top1",
        "base_mean_rank",
        "group_mean_rank",
        "base_correct_probability",
        "group_correct_probability",
        "observed_count",
    )
    if not all(name in hard_metrics for name in required_hard) or not all(
        name in rank_metrics for name in required_rank
    ):
        raise ValueError("candidate-group gate metrics are incomplete")
    values = [
        *(float(hard_metrics[name]) for name in required_hard),
        *(float(rank_metrics[name]) for name in required_rank),
        float(minimum_hard_eligible_query_fraction),
        float(minimum_hard_win_fraction),
        float(minimum_hard_gap),
        float(minimum_hard_visual_gap_delta),
        float(minimum_identity_top1_lift),
        float(minimum_identity_mean_rank_reduction),
        float(minimum_identity_correct_probability_lift),
    ]
    if not all(math.isfinite(value) for value in values) or float(rank_metrics["observed_count"]) <= 0.0:
        raise ValueError("candidate-group gate values are invalid")
    visual_delta = float(hard_metrics["normal_mean_positive_minus_negative"]) - float(
        hard_metrics["permuted_mean_positive_minus_negative"]
    )
    top1_lift = float(rank_metrics["group_top1"]) - float(rank_metrics["base_top1"])
    rank_reduction = float(rank_metrics["base_mean_rank"]) - float(rank_metrics["group_mean_rank"])
    probability_lift = float(rank_metrics["group_correct_probability"]) - float(
        rank_metrics["base_correct_probability"]
    )
    hard_passed = bool(
        float(hard_metrics["eligible_query_fraction"]) >= float(minimum_hard_eligible_query_fraction)
        and float(hard_metrics["normal_win_fraction"]) >= float(minimum_hard_win_fraction)
        and float(hard_metrics["normal_mean_positive_minus_negative"]) >= float(minimum_hard_gap)
        and visual_delta >= float(minimum_hard_visual_gap_delta)
    )
    rank_passed = bool(
        top1_lift >= float(minimum_identity_top1_lift)
        and rank_reduction >= float(minimum_identity_mean_rank_reduction)
        and probability_lift >= float(minimum_identity_correct_probability_lift)
    )
    return {
        "format": GATE_FORMAT,
        "hard_repeat_passed": hard_passed,
        "posterior_rank_passed": rank_passed,
        "passed": bool(hard_passed and rank_passed),
        "normal_minus_permuted_hard_gap": visual_delta,
        "identity_top1_lift": top1_lift,
        "identity_mean_rank_reduction": rank_reduction,
        "identity_correct_probability_lift": probability_lift,
        "thresholds": {
            "minimum_hard_eligible_query_fraction": float(minimum_hard_eligible_query_fraction),
            "minimum_hard_win_fraction": float(minimum_hard_win_fraction),
            "minimum_hard_gap": float(minimum_hard_gap),
            "minimum_hard_visual_gap_delta": float(minimum_hard_visual_gap_delta),
            "minimum_identity_top1_lift": float(minimum_identity_top1_lift),
            "minimum_identity_mean_rank_reduction": float(minimum_identity_mean_rank_reduction),
            "minimum_identity_correct_probability_lift": float(
                minimum_identity_correct_probability_lift
            ),
        },
    }


@torch.no_grad()
def _evaluate_static_hard(
    *,
    model: CandidateGroupPhaseIdentityLLR,
    groups: Mapping[str, object],
    static_hard_by_query: Mapping[str, object],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    state: _DistributedState,
    support_permutation_shift: int,
    identity_prior_logit_weight: float,
) -> dict[str, float]:
    """Evaluate exact correct-vs-coherent-wrong pairs under paired appearance control."""

    totals = torch.zeros((7,), dtype=torch.float64, device=state.device)
    model.eval()
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        hard = static_hard_by_query.get(str(query_id))
        if group is None or hard is None:
            raise ValueError("candidate-group static hard query is unresolved")
        runtime = _slice_runtime(complete_runtime, group.layout_rows)
        normal = model(runtime=runtime)
        permuted = model(runtime=runtime, support_permutation_shift=int(support_permutation_shift))
        normal_logits, normal_usable = candidate_group_identity_plus_null_logits(
            runtime=runtime,
            prediction=normal,
            candidate_prior_logit_weight=float(identity_prior_logit_weight),
        )
        permuted_logits, permuted_usable = candidate_group_identity_plus_null_logits(
            runtime=runtime,
            prediction=permuted,
            candidate_prior_logit_weight=float(identity_prior_logit_weight),
        )
        source_position = {
            int(source_id): index for index, source_id in enumerate(group.source_point_ids.tolist())
        }
        points = torch.tensor(
            [source_position[int(source_id)] for source_id in hard.source_point_ids.tolist()],
            dtype=torch.long,
            device=state.device,
        )
        positive = torch.from_numpy(np.asarray(hard.positive_candidate_indices, dtype=np.int64)).to(
            device=state.device
        )
        negative = torch.from_numpy(np.asarray(hard.negative_candidate_indices, dtype=np.int64)).to(
            device=state.device
        )
        active = (
            normal_usable[points, positive]
            & normal_usable[points, negative]
            & permuted_usable[points, positive]
            & permuted_usable[points, negative]
        )
        totals[6] += 1.0
        if bool(active.any()):
            normal_gap = normal_logits[points[active], positive[active]] - normal_logits[
                points[active], negative[active]
            ]
            permuted_gap = permuted_logits[points[active], positive[active]] - permuted_logits[
                points[active], negative[active]
            ]
            totals[0] += normal_gap.double().sum()
            totals[1] += (normal_gap > 0.0).double().sum()
            totals[2] += permuted_gap.double().sum()
            totals[3] += (permuted_gap > 0.0).double().sum()
            totals[4] += active.double().sum()
            totals[5] += 1.0
    totals = _reduce_sum(state, totals)
    count = float(totals[4].item())
    queries = float(totals[6].item())
    if count <= 0.0 or queries <= 0.0:
        return {
            "active_edge_count": count,
            "eligible_query_fraction": 0.0,
            "normal_mean_positive_minus_negative": 0.0,
            "normal_win_fraction": 0.0,
            "permuted_mean_positive_minus_negative": 0.0,
            "permuted_win_fraction": 0.0,
            "query_count": queries,
        }
    return {
        "active_edge_count": count,
        "eligible_query_fraction": float(totals[5].item() / queries),
        "normal_mean_positive_minus_negative": float(totals[0].item() / count),
        "normal_win_fraction": float(totals[1].item() / count),
        "permuted_mean_positive_minus_negative": float(totals[2].item() / count),
        "permuted_win_fraction": float(totals[3].item() / count),
        "query_count": queries,
    }


@torch.no_grad()
def _evaluate_identity_rank(
    *,
    model: CandidateGroupPhaseIdentityLLR,
    groups: Mapping[str, object],
    exact_by_query: Mapping[str, ExactIdentityQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    state: _DistributedState,
    identity_prior_logit_weight: float,
) -> dict[str, float]:
    """Compare frozen-prior and visual posterior ranks on all observed rows."""

    totals = torch.zeros((8,), dtype=torch.float64, device=state.device)
    model.eval()
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        exact = exact_by_query.get(str(query_id))
        if group is None or exact is None:
            raise ValueError("candidate-group rank audit query is unresolved")
        runtime = _slice_runtime(complete_runtime, group.layout_rows)
        prediction = model(runtime=runtime)
        visual_logits, candidate_usable = candidate_group_identity_plus_null_logits(
            runtime=runtime,
            prediction=prediction,
            candidate_prior_logit_weight=float(identity_prior_logit_weight),
        )
        observed, _, supervised = canonical_registered_identity_or_null_targets(
            observed_candidate_mask=torch.from_numpy(exact.observed_candidate_mask),
            candidate_dustbin_mask=torch.from_numpy(exact.candidate_dustbin_mask),
            candidate_supervised_mask=torch.from_numpy(exact.candidate_supervised_mask),
        )
        observed = observed.to(device=state.device)
        supervised = supervised.to(device=state.device)
        labels = observed.to(dtype=torch.long).argmax(dim=1)
        target_usable = candidate_usable.gather(1, labels[:, None]).squeeze(1)
        target_rows = (
            supervised
            & observed.any(dim=1)
            & target_usable
            & (candidate_usable.sum(dim=1) >= 2)
        )
        if not bool(target_rows.any()):
            continue
        target = labels[target_rows]
        prior_logits = torch.cat(
            (
                float(identity_prior_logit_weight)
                * torch.log(runtime.candidate_probabilities.to(state.device).clamp_min(torch.finfo(torch.float32).tiny)),
                (
                    float(identity_prior_logit_weight)
                    * torch.log(runtime.null_probabilities.to(state.device).clamp_min(torch.finfo(torch.float32).tiny))
                )[:, None],
            ),
            dim=1,
        )[target_rows]
        visual = visual_logits[target_rows]
        target_column = torch.arange(len(target), device=state.device)
        base_target = prior_logits[target_column, target]
        group_target = visual[target_column, target]
        base_rank = 1 + (prior_logits[:, :-1] > base_target[:, None]).sum(dim=1)
        group_rank = 1 + (visual[:, :-1] > group_target[:, None]).sum(dim=1)
        base_probability = torch.softmax(prior_logits, dim=1)[target_column, target]
        group_probability = torch.softmax(visual, dim=1)[target_column, target]
        totals[0] += float(len(target))
        totals[1] += (prior_logits.argmax(dim=1) == target).double().sum()
        totals[2] += (visual.argmax(dim=1) == target).double().sum()
        totals[3] += base_rank.double().sum()
        totals[4] += group_rank.double().sum()
        totals[5] += base_probability.double().sum()
        totals[6] += group_probability.double().sum()
        totals[7] += 1.0
    totals = _reduce_sum(state, totals)
    count = float(totals[0].item())
    if count <= 0.0:
        raise RuntimeError("candidate-group rank audit retained no observed targets")
    return {
        "observed_count": count,
        "base_top1": float(totals[1].item() / count),
        "group_top1": float(totals[2].item() / count),
        "base_mean_rank": float(totals[3].item() / count),
        "group_mean_rank": float(totals[4].item() / count),
        "base_correct_probability": float(totals[5].item() / count),
        "group_correct_probability": float(totals[6].item() / count),
        "query_count": float(totals[7].item()),
    }


def train_candidate_group_phase_identity_llr(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(int(args.seed) + state.rank)
        np.random.seed(int(args.seed) + state.rank)
        random.seed(int(args.seed) + state.rank)
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_group_phase_identity_llr.pt"
        summary_path = output_dir / "summary.json"
        if checkpoint_path.exists() and not bool(args.force):
            raise FileExistsError(f"candidate-group checkpoint already exists: {checkpoint_path}")
        if state.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if state.enabled:
            distributed.barrier()

        layout_path = Path(args.rgb_spatial_layout)
        geometry_path = Path(args.geometry_training_targets)
        identity_path = Path(args.registered_identity_targets)
        current_hard_path = Path(args.current_hard_repeat_targets)
        static_hard_path = Path(args.static_hard_repeat_targets)
        mining_checkpoint_path = Path(args.current_hard_mining_checkpoint)
        source_paths = {
            "radio_final": Path(args.radio_final_context_cache),
            "radio_intermediate": Path(args.radio_intermediate_context_cache),
            "alike": Path(args.alike_spatial_context_cache),
        }
        if not all(path.is_file() for path in (layout_path, geometry_path, identity_path, current_hard_path, static_hard_path, mining_checkpoint_path, *source_paths.values())):
            raise FileNotFoundError("candidate-group identity input artifact is absent")
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        geometry_targets = load_candidate_pose_rgb_spatial_training_targets(geometry_path)
        identity_targets = load_candidate_pose_rgb_spatial_training_targets(identity_path)
        layout_sha = file_sha256_short(layout_path)
        geometry_sha = file_sha256_short(geometry_path)
        identity_sha = file_sha256_short(identity_path)
        validate_training_layout_and_targets(layout=layout, targets=geometry_targets, layout_sha256=layout_sha)
        validate_training_layout_and_targets(layout=layout, targets=identity_targets, layout_sha256=layout_sha)
        groups = build_train_query_groups(layout=layout, targets=geometry_targets)
        exact_by_query = build_exact_identity_query_targets(groups=groups, identity_targets=identity_targets)
        all_train_query_ids = tuple(sorted(groups))
        inner_train_ids, inner_validation_ids = _partition_train_queries_for_inner_validation(
            query_ids=all_train_query_ids,
            fold_count=int(args.inner_fold_count),
            fold_index=int(args.inner_fold_index),
        )
        partition = train_query_partition_manifest(
            all_query_ids=all_train_query_ids,
            inner_train_query_ids=inner_train_ids,
            inner_validation_query_ids=inner_validation_ids,
            fold_count=int(args.inner_fold_count),
            fold_index=int(args.inner_fold_index),
        )
        current_hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(current_hard_path)
        current_hard_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=current_hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        current_hard_provenance = validate_external_frozen_current_hard_targets(
            mined_targets=current_hard_targets,
            mined_groups=current_hard_groups,
            registered_identity_targets=identity_targets,
            registered_identity_targets_sha256=identity_sha,
            expected_partition=partition,
            mining_checkpoint_path=mining_checkpoint_path,
        )
        static_hard_targets = load_candidate_pose_rgb_spatial_hard_repeat_targets(static_hard_path)
        static_hard_geometry_groups = build_hard_repeat_query_targets(
            layout=layout,
            targets=geometry_targets,
            hard_repeat_targets=static_hard_targets,
            layout_sha256=layout_sha,
            targets_sha256=geometry_sha,
        )
        static_hard_groups, static_hard_filter = filter_static_hard_repeat_groups_to_registered_exact_identity(
            static_groups=static_hard_geometry_groups,
            registered_identity_targets=identity_targets,
        )
        if not set(inner_validation_ids).issubset(static_hard_groups):
            raise ValueError("candidate-group static hard target lacks an inner-validation query")

        sources = load_context_attention_sources(
            radio_final_context_cache=source_paths["radio_final"],
            radio_intermediate_context_cache=source_paths["radio_intermediate"],
            alike_spatial_context_cache=source_paths["alike"],
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_grids, source_manifest = _source_table(sources)
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        storage_dtype = torch.float16 if str(args.source_storage_dtype) == "float16" else torch.float32
        model = CandidateGroupPhaseIdentityLLR(
            sources=source_grids,
            image_sizes=torch.from_numpy(image_sizes),
            source_embedding_dim=int(args.source_embedding_dim),
            group_embedding_dim=int(args.group_embedding_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
            source_storage_dtype=storage_dtype,
            learn_null_log_likelihood=bool(args.learn_null_log_likelihood),
        ).to(state.device)
        model_for_train: torch.nn.Module = model
        if state.enabled:
            model_for_train = DistributedDataParallel(
                model,
                device_ids=[state.local_rank] if state.device.type == "cuda" else None,
                output_device=state.local_rank if state.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model_for_train.parameters() if parameter.requires_grad],
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        history: list[dict[str, object]] = []
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            query_batches = _rank_query_batches(
                query_ids=inner_train_ids,
                state=state,
                queries_per_step=int(args.queries_per_step),
                seed=int(args.seed) + epoch * 1000003,
            )
            sums = torch.zeros((7,), dtype=torch.float64, device=state.device)
            for step, query_batch in enumerate(query_batches):
                batch = build_phase_identity_batch(
                    query_ids=query_batch,
                    groups=groups,
                    exact_by_query=exact_by_query,
                    current_hard_by_query=current_hard_groups,
                    complete_runtime=complete_runtime,
                    max_points_per_query=int(args.max_points_per_query),
                    max_current_hard_edges_per_query=int(args.max_current_hard_edges_per_query),
                    seed=int(args.seed) + epoch * 1000003 + step * 9176,
                    device=state.device,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _group_training_loss(
                    model=model_for_train,
                    batch=batch,
                    device=state.device,
                    identity_prior_logit_weight=float(args.identity_prior_logit_weight),
                    identity_loss_weight=float(args.identity_loss_weight),
                    current_hard_loss_weight=float(args.current_hard_loss_weight),
                    current_hard_margin=float(args.current_hard_margin),
                )
                loss.backward()
                if float(args.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_for_train.parameters(), max_norm=float(args.gradient_clip_norm)
                    )
                optimizer.step()
                active = float(metrics["hard_repeat_active"])
                sums += torch.tensor(
                    [
                        float(metrics["total_loss"]),
                        float(metrics["identity_loss"]),
                        float(metrics["hard_repeat_margin_loss"]),
                        float(metrics["hard_repeat_mean_gap"]) * active,
                        float(metrics["hard_repeat_win_fraction"]) * active,
                        active,
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            sums = _reduce_sum(state, sums)
            if state.rank == 0:
                epoch_log = {
                    "epoch": int(epoch + 1),
                    "train_step_count": int(sums[6].item()),
                    "mean_total_loss": float((sums[0] / sums[6].clamp_min(1.0)).item()),
                    "mean_identity_loss": float((sums[1] / sums[6].clamp_min(1.0)).item()),
                    "mean_current_hard_margin_loss": float((sums[2] / sums[6].clamp_min(1.0)).item()),
                    "current_hard_active_edges": float(sums[5].item()),
                    "current_hard_mean_gap": float((sums[3] / sums[5].clamp_min(1.0)).item()),
                    "current_hard_win_fraction": float((sums[4] / sums[5].clamp_min(1.0)).item()),
                }
                history.append(epoch_log)
                print(json.dumps({"candidate_group_identity_train": epoch_log}, sort_keys=True), flush=True)

        base_model = model_for_train.module if isinstance(model_for_train, DistributedDataParallel) else model
        hard_metrics = _evaluate_static_hard(
            model=base_model,
            groups=groups,
            static_hard_by_query=static_hard_groups,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            state=state,
            support_permutation_shift=int(args.support_permutation_shift),
            identity_prior_logit_weight=float(args.identity_prior_logit_weight),
        )
        rank_metrics = _evaluate_identity_rank(
            model=base_model,
            groups=groups,
            exact_by_query=exact_by_query,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            state=state,
            identity_prior_logit_weight=float(args.identity_prior_logit_weight),
        )
        gate = group_identity_gate_decision(
            hard_metrics=hard_metrics,
            rank_metrics=rank_metrics,
            minimum_hard_eligible_query_fraction=float(args.minimum_hard_eligible_query_fraction),
            minimum_hard_win_fraction=float(args.minimum_hard_win_fraction),
            minimum_hard_gap=float(args.minimum_hard_gap),
            minimum_hard_visual_gap_delta=float(args.minimum_hard_visual_gap_delta),
            minimum_identity_top1_lift=float(args.minimum_identity_top1_lift),
            minimum_identity_mean_rank_reduction=float(args.minimum_identity_mean_rank_reduction),
            minimum_identity_correct_probability_lift=float(
                args.minimum_identity_correct_probability_lift
            ),
        )
        if state.enabled:
            distributed.barrier()
        result: dict[str, object] = {
            "stage": "train_candidate_group_phase_identity_llr",
            "format": CHECKPOINT_FORMAT,
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection": {
                "policy": FINAL_EPOCH_SELECTION_POLICY,
                "selected_epoch": int(args.epochs),
                "inner_validation_used_for_model_selection": False,
                "hard_repeat": hard_metrics,
                "identity_rank": rank_metrics,
                "gate": gate,
            },
            "history": history,
            "protocol": {
                "target_free_runtime": True,
                "pose_or_ground_truth_not_available_to_visual_encoder": True,
                "fixed_global_topl_and_explicit_null": True,
                "candidate_order_permutation_equivariant": True,
                "no_render": True,
                "no_image_retrieval_or_submap": True,
                "heldout_validation_or_test_not_run": True,
                "pnp_integration_allowed": False,
            },
        }
        if state.rank == 0:
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_GROUP_PHASE_IDENTITY_LLR_FORMAT,
                "runtime_contract": {
                    "target_free_runtime": True,
                    "forbidden_encoder_inputs": [
                        "pose",
                        "projection_offset",
                        "residual",
                        "track_id",
                        "candidate_rank",
                        "coarse_score",
                        "training_label",
                    ],
                    "candidate_order_permutation_equivariant": True,
                    "render": False,
                    "image_retrieval_or_submap": False,
                    "visual_sources": list(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES),
                    "support_view_marginalization": "fixed_mass_neutral_missing_view_v1",
                },
                "model_config": {
                    "source_embedding_dim": int(args.source_embedding_dim),
                    "group_embedding_dim": int(args.group_embedding_dim),
                    "max_abs_log_ratio": float(args.max_abs_log_ratio),
                    "source_storage_dtype": str(args.source_storage_dtype),
                    "learn_null_log_likelihood": bool(args.learn_null_log_likelihood),
                    "source_configs": {
                        name: {
                            "name": str(config.name),
                            "window_size": int(config.window_size),
                            "shift_radius": int(config.shift_radius),
                            "region_bins": int(config.region_bins),
                        }
                        for name, config in base_model.source_configs.items()
                    },
                },
                "model_state_dict": base_model.state_dict(),
                "lineage": {
                    "layout_sha256": layout_sha,
                    "geometry_training_targets_sha256": geometry_sha,
                    "registered_identity_targets_sha256": identity_sha,
                    "current_hard_repeat_targets_sha256": file_sha256_short(current_hard_path),
                    "static_hard_repeat_targets_sha256": file_sha256_short(static_hard_path),
                    "static_hard_registered_exact_filter": static_hard_filter,
                    "source_cache_sha256": {
                        name: file_sha256_short(path) for name, path in source_paths.items()
                    },
                    "source_image_manifest_sha256": source_manifest,
                    "descriptor_space_id": str(layout.metadata.get("descriptor_space_id", "")),
                    "projection_space_id": str(layout.metadata.get("projection_space_id", "")),
                },
                "external_current_hard_negative_provenance": current_hard_provenance,
                "train_query_partition": partition,
                "selection_policy": FINAL_EPOCH_SELECTION_POLICY,
                "gate": gate,
                "training_args": vars(args),
            }
            _atomic_torch_save(checkpoint, checkpoint_path)
            _atomic_json_save(result, summary_path)
            print(json.dumps({"candidate_group_identity_gate": gate}, sort_keys=True), flush=True)
        return result
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> int:
    train_candidate_group_phase_identity_llr(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
