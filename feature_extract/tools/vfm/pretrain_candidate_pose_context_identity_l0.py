"""Pretrain absolute-context candidate identity on real SfM observation pairs.

This is L0 only.  The network receives fixed query/support image coordinates
and frozen RADIO/ALIKE feature grids, then emits one candidate-specific context
log-likelihood ratio per support edge.  It never sees RGB patches, pose,
residual, track ID, rank, coarse score, or target label.  Registered same-track
labels are joined only after this target-free forward pass.

The broad observation-pair artifact provides enough train-query examples to
test absolute visual phase without the sparse P1 exact-observation coverage.
Candidate slots are randomized before every forward pass, and evaluation
includes both support-appearance derangement and descriptor-zeroed
position-only controls.  Passing this broad gate is necessary but not
sufficient for any P1 or pose-level use.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_likelihood import (
    _DistributedState,
    _finalize_distributed,
    _initialize_distributed,
    _rank_batch_rows,
    _runtime_from_pair_rows,
    _source_table,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_OBJECTIVE,
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    context_candidate_logit_mixture,
    context_identity_cross_entropy_loss,
    context_identity_support_permutation_margin_loss,
    permute_runtime_candidate_slots,
    permute_runtime_support_image_appearance_only,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_observation_pairs import (
    CandidatePoseRGBSpatialObservationPairs,
    load_candidate_pose_rgb_spatial_observation_pairs,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)


CHECKPOINT_FORMAT = CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_CHECKPOINT_FORMAT
_POSITION_ONLY_MODE = "position_only"
FIXED_FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"


def fixed_final_epoch_checkpoint_selection(*, epochs: int) -> dict[str, object]:
    """Declare the non-data-dependent checkpoint selection policy."""

    if int(epochs) <= 0:
        raise ValueError("context observation fixed-final-epoch selection requires positive epochs")
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
    """Reject controls that alter geometry or source-crop availability."""

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
        raise RuntimeError("context observation support control changed geometry or mixture mass")
    per_point_valid = runtime.support_view_valid.reshape(runtime.point_count, -1).sum(dim=1)
    if bool(torch.any(per_point_valid > 1)) and torch.equal(
        runtime.support_image_indices, permuted_runtime.support_image_indices
    ):
        raise RuntimeError("context observation support control did not derange image content")
    normal_usable = getattr(normal_prediction, "context_edge_usable", None)
    permuted_usable = getattr(permuted_prediction, "context_edge_usable", None)
    if (
        normal_usable is None
        or permuted_usable is None
        or not torch.equal(normal_usable, permuted_usable)
    ):
        raise RuntimeError("context observation support control changed source availability")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-pairs", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--search-radius-px", type=float, default=8.0)
    parser.add_argument("--context-radius-px", type=float, default=12.0)
    parser.add_argument("--step-px", type=float, default=1.0)
    parser.add_argument("--texture-feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-abs-context-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--radio-final-context-window", type=int, default=15)
    parser.add_argument("--radio-intermediate-context-window", type=int, default=15)
    parser.add_argument("--alike-context-window", type=int, default=13)
    parser.add_argument("--context-encoder-arch", default="absolute_cross_attention_v3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--support-permutation-loss-weight", type=float, default=0.25)
    parser.add_argument("--support-permutation-margin", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Deterministic train-only smoke cap; zero uses every inner-train observation.",
    )
    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=0,
        help="Deterministic train-only gate cap; zero audits every inner-validation observation.",
    )
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-normal-margin", type=float, default=0.05)
    parser.add_argument("--minimum-support-visual-gap", type=float, default=0.05)
    parser.add_argument("--minimum-position-visual-gap", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> dict[str, int]:
    values = (
        float(args.search_radius_px),
        float(args.context_radius_px),
        float(args.step_px),
        float(args.learning_rate),
        float(args.weight_decay),
        float(args.identity_loss_weight),
        float(args.support_permutation_loss_weight),
        float(args.support_permutation_margin),
        float(args.max_abs_context_log_ratio),
        float(args.gradient_clip_norm),
        float(args.minimum_win_fraction),
        float(args.minimum_normal_margin),
        float(args.minimum_support_visual_gap),
        float(args.minimum_position_visual_gap),
    )
    if (
        int(args.texture_feature_dim) <= 0
        or int(args.hidden_dim) < 4
        or int(args.edge_chunk_size) <= 0
        or int(args.batch_size) < 2
        or int(args.epochs) <= 0
        or int(args.max_train_rows) < 0
        or int(args.max_validation_rows) < 0
        or not all(math.isfinite(value) for value in values)
        or float(args.search_radius_px) < float(args.step_px)
        or float(args.context_radius_px) <= 0.0
        or float(args.step_px) <= 0.0
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.identity_loss_weight) <= 0.0
        or float(args.support_permutation_loss_weight) < 0.0
        or float(args.support_permutation_margin) < 0.0
        or float(args.max_abs_context_log_ratio) <= 0.0
        or float(args.gradient_clip_norm) <= 0.0
        or not 0.0 <= float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_normal_margin) < 0.0
        or float(args.minimum_support_visual_gap) < 0.0
        or float(args.minimum_position_visual_gap) < 0.0
    ):
        raise ValueError("context observation L0 pretraining arguments are invalid")
    if (
        resolve_candidate_pose_rgb_spatial_context_encoder_arch(args.context_encoder_arch)
        != "absolute_cross_attention_v3"
    ):
        raise ValueError("context observation L0 requires absolute_cross_attention_v3")
    return resolve_candidate_pose_rgb_spatial_context_windows(
        {
            "radio_final": int(args.radio_final_context_window),
            "radio_intermediate": int(args.radio_intermediate_context_window),
            "alike": int(args.alike_context_window),
        }
    )


def _distributed_output_conflict(*, state: _DistributedState, paths: Sequence[Path]) -> bool:
    conflict = bool(any(Path(path).exists() for path in paths)) if state.rank == 0 else False
    if state.enabled:
        value = torch.tensor([int(conflict)], dtype=torch.int64, device=state.device)
        distributed.broadcast(value, src=0)
        conflict = bool(value.item())
    return conflict


def _limited_rows(rows: np.ndarray, *, limit: int, seed: int) -> np.ndarray:
    values = np.asarray(rows, dtype=np.int64).reshape(-1)
    if len(values) == 0 or int(limit) < 0:
        raise ValueError("context observation row cap inputs are invalid")
    if int(limit) == 0 or len(values) <= int(limit):
        return values
    selected = np.random.default_rng(int(seed)).choice(values, size=int(limit), replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def candidate_slot_permutations_from_anchor_ids(
    *, anchor_ids: np.ndarray, candidate_count: int, seed: int
) -> torch.Tensor:
    """Generate deterministic per-observation slot permutations after runtime creation."""

    anchors = np.asarray(anchor_ids, dtype=np.int64).reshape(-1)
    count = int(candidate_count)
    if len(anchors) == 0 or count < 2:
        raise ValueError("context observation candidate-slot inputs are invalid")
    output = np.empty((len(anchors), count), dtype=np.int64)
    for row, anchor in enumerate(anchors.tolist()):
        # No label participates in this seed.  DDP tail padding can repeat an
        # otherwise unique observation row in its final fixed-size batch; such
        # copies intentionally retain the same target-free ordering rather
        # than making the valid training protocol depend on batch position.
        local_seed = (int(seed) * 1000003 + int(anchor) * 9176) % (2**63 - 1)
        output[row] = np.random.default_rng(local_seed).permutation(count)
    return torch.from_numpy(output)


def positive_targets_after_candidate_slot_permutation(
    permutations: torch.Tensor,
) -> torch.Tensor:
    """Join the known observation-pair positive only after target-free reordering."""

    order = torch.as_tensor(permutations, dtype=torch.long)
    if (
        order.ndim != 2
        or order.shape[1] < 2
        or torch.any(order < 0)
        or torch.any((order == 0).sum(dim=1) != 1)
    ):
        raise ValueError("candidate-slot target permutation is invalid")
    labels = torch.argmax((order == 0).to(dtype=torch.long), dim=1)
    if not torch.all(order.gather(1, labels[:, None]).squeeze(1) == 0):
        raise ValueError("candidate-slot permutation does not contain the observation positive")
    targets = torch.zeros(order.shape, dtype=torch.bool, device=order.device)
    targets.scatter_(1, labels[:, None], True)
    return targets


def _row_identity_statistics(
    *,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: object,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-row CE, target margin, top-1, and active mask without priors."""

    logits, usable = context_candidate_logit_mixture(runtime=runtime, prediction=prediction)
    observed = torch.as_tensor(targets, dtype=torch.bool, device=logits.device)
    if (
        observed.shape != logits.shape
        or torch.any(observed.sum(dim=1) != 1)
        or usable.shape != logits.shape
    ):
        raise ValueError("context observation identity targets are incompatible")
    labels = torch.argmax(observed.to(dtype=torch.long), dim=1)
    active = usable.gather(1, labels[:, None]).squeeze(1) & (usable.sum(dim=1) >= 2)
    masked = logits.masked_fill(~usable, -torch.inf)
    ce = torch.zeros((len(masked),), dtype=masked.dtype, device=masked.device)
    if bool(active.any()):
        ce[active] = torch.nn.functional.cross_entropy(
            masked[active], labels[active], reduction="none"
        )
    target = masked.gather(1, labels[:, None]).squeeze(1)
    competitors = masked.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    margin = target - competitors.amax(dim=1)
    top1 = torch.argmax(masked, dim=1) == labels
    return ce, margin, top1, active


def _query_grouped_means(
    *,
    query_ids: Sequence[str],
    values: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for query_id, value, enabled in zip(query_ids, values.tolist(), active.tolist()):
        if bool(enabled):
            grouped[str(query_id)].append(float(value))
    if not grouped:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)


def observation_context_identity_gate(
    metrics: Mapping[str, float],
    *,
    minimum_win_fraction: float,
    minimum_normal_margin: float,
    minimum_support_visual_gap: float,
    minimum_position_visual_gap: float,
) -> dict[str, object]:
    """Require appearance to beat both support and coordinate-only controls."""

    normal_margin = float(metrics["normal_mean_margin"])
    support_margin = float(metrics["support_permuted_mean_margin"])
    position_margin = float(metrics["position_only_mean_margin"])
    win_fraction = float(metrics["normal_win_fraction"])
    support_gap = normal_margin - support_margin
    position_gap = normal_margin - position_margin
    checks = {
        "normal_margin": normal_margin >= float(minimum_normal_margin),
        "normal_win_fraction": win_fraction >= float(minimum_win_fraction),
        "support_appearance_gap": support_gap >= float(minimum_support_visual_gap),
        "descriptor_over_position_gap": position_gap >= float(minimum_position_visual_gap),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "normal_minus_support_permuted_margin": float(support_gap),
        "normal_minus_position_only_margin": float(position_gap),
        "thresholds": {
            "minimum_win_fraction": float(minimum_win_fraction),
            "minimum_normal_margin": float(minimum_normal_margin),
            "minimum_support_visual_gap": float(minimum_support_visual_gap),
            "minimum_position_visual_gap": float(minimum_position_visual_gap),
        },
    }


def _runtime_and_targets(
    *,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
    permutation_seed: int,
) -> tuple[CandidatePoseRGBSpatialRuntime, torch.Tensor]:
    """Build a target-free candidate layout, then join its permuted positive label."""

    selected = np.asarray(rows, dtype=np.int64).reshape(-1)
    runtime = _runtime_from_pair_rows(
        pairs=pairs, rows=selected, image_index_by_id=image_index_by_id
    )
    order = candidate_slot_permutations_from_anchor_ids(
        anchor_ids=pairs.anchor_ids[selected],
        candidate_count=runtime.candidate_count,
        seed=int(permutation_seed),
    )
    permuted = permute_runtime_candidate_slots(runtime, permutations=order)
    return permuted, positive_targets_after_candidate_slot_permutation(order)


def _evaluate_inner_validation(
    *,
    model: CandidatePoseRGBSpatialLikelihood,
    pairs: CandidatePoseRGBSpatialObservationPairs,
    rows: np.ndarray,
    image_index_by_id: Mapping[str, int],
    batch_size: int,
    seed: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    """Evaluate visual, deranged-support, and position-only evidence per query."""

    model.eval()
    per_query_ids: list[str] = []
    normal_ce: list[float] = []
    normal_margin: list[float] = []
    normal_win: list[float] = []
    support_margin: list[float] = []
    position_margin: list[float] = []
    active_values: list[bool] = []
    with torch.no_grad():
        for begin in range(0, len(rows), int(batch_size)):
            batch_rows = np.asarray(rows[begin : begin + int(batch_size)], dtype=np.int64)
            runtime, targets = _runtime_and_targets(
                pairs=pairs,
                rows=batch_rows,
                image_index_by_id=image_index_by_id,
                permutation_seed=int(seed) + 7919,
            )
            deranged = permute_runtime_support_image_appearance_only(runtime, shift=1)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                normal_prediction = model(runtime=runtime, context_only=True)
                deranged_prediction = model(runtime=deranged, context_only=True)
                position_prediction = model(
                    runtime=runtime,
                    context_only=True,
                    context_appearance_mode=_POSITION_ONLY_MODE,
                )
            _assert_geometry_fixed_support_image_control(
                runtime=runtime,
                permuted_runtime=deranged,
                normal_prediction=normal_prediction,
                permuted_prediction=deranged_prediction,
            )
            ce, margin, top1, active = _row_identity_statistics(
                runtime=runtime, prediction=normal_prediction, targets=targets
            )
            _unused, deranged_margin, _unused_top1, deranged_active = _row_identity_statistics(
                runtime=deranged, prediction=deranged_prediction, targets=targets
            )
            _unused, position_margin_values, _unused_top1, position_active = _row_identity_statistics(
                runtime=runtime, prediction=position_prediction, targets=targets
            )
            all_active = active & deranged_active & position_active
            per_query_ids.extend(pairs.query_image_ids[batch_rows].tolist())
            normal_ce.extend(ce.detach().cpu().numpy().astype(np.float64).tolist())
            normal_margin.extend(margin.detach().cpu().numpy().astype(np.float64).tolist())
            normal_win.extend(top1.detach().cpu().numpy().astype(np.float64).tolist())
            support_margin.extend(deranged_margin.detach().cpu().numpy().astype(np.float64).tolist())
            position_margin.extend(position_margin_values.detach().cpu().numpy().astype(np.float64).tolist())
            active_values.extend(all_active.detach().cpu().numpy().astype(bool).tolist())
    active = np.asarray(active_values, dtype=bool)
    if not len(active) or not np.any(active):
        raise RuntimeError("context observation inner validation has no usable candidate rows")
    query_ids = tuple(str(value) for value in per_query_ids)
    grouped_ce = _query_grouped_means(
        query_ids=query_ids, values=np.asarray(normal_ce), active=active
    )
    grouped_normal_margin = _query_grouped_means(
        query_ids=query_ids, values=np.asarray(normal_margin), active=active
    )
    grouped_normal_win = _query_grouped_means(
        query_ids=query_ids, values=np.asarray(normal_win), active=active
    )
    grouped_support_margin = _query_grouped_means(
        query_ids=query_ids, values=np.asarray(support_margin), active=active
    )
    grouped_position_margin = _query_grouped_means(
        query_ids=query_ids, values=np.asarray(position_margin), active=active
    )
    if not (
        len(grouped_ce)
        == len(grouped_normal_margin)
        == len(grouped_normal_win)
        == len(grouped_support_margin)
        == len(grouped_position_margin)
    ):
        raise RuntimeError("context observation query-grouped validation alignment differs")
    return {
        "query_count": float(len(grouped_ce)),
        "active_row_count": float(np.sum(active)),
        "normal_cross_entropy": float(np.mean(grouped_ce)),
        "normal_mean_margin": float(np.mean(grouped_normal_margin)),
        "normal_win_fraction": float(np.mean(grouped_normal_win)),
        "support_permuted_mean_margin": float(np.mean(grouped_support_margin)),
        "position_only_mean_margin": float(np.mean(grouped_position_margin)),
    }


def pretrain_candidate_pose_context_identity_l0(args: argparse.Namespace) -> dict[str, object]:
    """Fit broad train-observation identity evidence with no RGB/spatial path."""

    context_windows = _validate_args(args)
    context_encoder_arch = resolve_candidate_pose_rgb_spatial_context_encoder_arch(
        args.context_encoder_arch
    )
    state = _initialize_distributed(str(args.device))
    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_context_identity_observation_pretrain.pt"
        history_path = output_dir / "history.json"
        summary_path = output_dir / "summary.json"
        if not bool(args.force) and _distributed_output_conflict(
            state=state, paths=(checkpoint_path, history_path, summary_path)
        ):
            raise FileExistsError("refusing to overwrite context observation L0 output")
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
        except AttributeError:  # pragma: no cover - torch compatibility
            pass

        pair_path = Path(args.observation_pairs)
        pairs = load_candidate_pose_rgb_spatial_observation_pairs(pair_path)
        train_rows = _limited_rows(
            np.flatnonzero(pairs.split_names == "inner_train"),
            limit=int(args.max_train_rows),
            seed=int(args.seed) + 17,
        )
        validation_rows = _limited_rows(
            np.flatnonzero(pairs.split_names == "inner_validation"),
            limit=int(args.max_validation_rows),
            seed=int(args.seed) + 29,
        )
        if len(train_rows) == 0 or len(validation_rows) == 0:
            raise ValueError("context observation pairs lack an inner train or validation split")
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
            raise ValueError("context observation L0 requires common image dimensions")
        image_index_by_id = {str(image_id): index for index, image_id in enumerate(image_ids.tolist())}
        all_pair_images = set(pairs.query_image_ids.tolist())
        all_pair_images.update(pairs.positive_support_image_ids.tolist())
        all_pair_images.update(pairs.negative_support_image_ids.reshape(-1).tolist())
        if not all_pair_images.issubset(image_index_by_id):
            raise ValueError("context observation pairs reference an image absent from feature sources")

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
            activation_checkpointing=False,
            context_windows=context_windows,
            context_encoder_arch=context_encoder_arch,
        )
        # L0 must not accidentally update the RGB texture/density path.
        for module in (model.texture_encoder, model.spatial_residual_head, model.non_dustbin_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
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
        final_state_dict: dict[str, torch.Tensor] | None = None
        final_epoch = -1
        history: list[dict[str, object]] = []
        started = time.time()

        for epoch in range(int(args.epochs)):
            model_for_train.train()
            batches = _rank_batch_rows(
                rows=train_rows,
                batch_size=int(args.batch_size),
                rank=state.rank,
                world_size=state.world_size,
                seed=int(args.seed),
                epoch=int(epoch),
            )
            totals = torch.zeros((9,), dtype=torch.float64, device=state.device)
            epoch_started = time.time()
            for step, batch_rows in enumerate(batches):
                runtime, targets = _runtime_and_targets(
                    pairs=pairs,
                    rows=batch_rows,
                    image_index_by_id=image_index_by_id,
                    permutation_seed=int(args.seed) + int(epoch) * 1000003 + int(step),
                )
                deranged = permute_runtime_support_image_appearance_only(runtime, shift=1)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model_for_train(runtime=runtime, context_only=True)
                    identity_loss, identity_metrics = context_identity_cross_entropy_loss(
                        runtime=runtime, prediction=prediction, target_observed=targets
                    )
                    permutation_loss = prediction.context_log_likelihood_ratios.sum() * 0.0
                    permutation_metrics = {
                        "context_identity_permutation_active_rows": 0.0,
                        "context_identity_permutation_mean_gap": 0.0,
                        "context_identity_permutation_win_fraction": 0.0,
                        "context_identity_permutation_margin_loss": 0.0,
                    }
                    if float(args.support_permutation_loss_weight) > 0.0:
                        deranged_prediction = model_for_train(runtime=deranged, context_only=True)
                        _assert_geometry_fixed_support_image_control(
                            runtime=runtime,
                            permuted_runtime=deranged,
                            normal_prediction=prediction,
                            permuted_prediction=deranged_prediction,
                        )
                        permutation_loss, permutation_metrics = (
                            context_identity_support_permutation_margin_loss(
                                runtime=runtime,
                                prediction=prediction,
                                permuted_runtime=deranged,
                                permuted_prediction=deranged_prediction,
                                target_observed=targets,
                                margin=float(args.support_permutation_margin),
                            )
                        )
                    loss = (
                        float(args.identity_loss_weight) * identity_loss
                        + float(args.support_permutation_loss_weight) * permutation_loss
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
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            if state.enabled:
                distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
            global_steps = int(len(batches) * state.world_size)
            if state.rank == 0:
                validation = _evaluate_inner_validation(
                    model=core_model,
                    pairs=pairs,
                    rows=validation_rows,
                    image_index_by_id=image_index_by_id,
                    batch_size=int(args.batch_size),
                    seed=int(args.seed),
                    device=state.device,
                    amp_enabled=amp_enabled,
                )
                # Validation remains telemetry and a train-only gate only. It
                # cannot select a checkpoint or feed back into optimization.
                final_metrics = dict(validation)
                final_epoch = int(epoch + 1)
                final_state_dict = _model_state_cpu(core_model)
                record = {
                    "epoch": int(epoch + 1),
                    "epoch_seconds": float(time.time() - epoch_started),
                    "global_steps": int(global_steps),
                    "train_total_loss": float((totals[0] / global_steps).item()),
                    "train_identity_cross_entropy": float((totals[1] / global_steps).item()),
                    "train_identity_top1_accuracy": float((totals[2] / global_steps).item()),
                    "train_identity_mean_margin": float((totals[3] / global_steps).item()),
                    "train_identity_active_rows_per_step": float((totals[4] / global_steps).item()),
                    "train_support_permutation_margin_loss": float((totals[5] / global_steps).item()),
                    "train_support_permutation_mean_gap": float((totals[6] / global_steps).item()),
                    "train_support_permutation_win_fraction": float((totals[7] / global_steps).item()),
                    **{f"inner_{key}": value for key, value in validation.items()},
                }
                history.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            if state.enabled:
                distributed.barrier()

        if state.rank == 0:
            if final_metrics is None or final_state_dict is None or final_epoch < 1:
                raise RuntimeError("context observation L0 did not retain final-epoch telemetry")
            gate = observation_context_identity_gate(
                final_metrics,
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_normal_margin=float(args.minimum_normal_margin),
                minimum_support_visual_gap=float(args.minimum_support_visual_gap),
                minimum_position_visual_gap=float(args.minimum_position_visual_gap),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
                "architecture": "absolute_coordinate_multiscale_context_identity_l0_v1",
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "runtime_layout_is_target_free": True,
                "pose_or_ground_truth_used_by_runtime_scorer": False,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "p1_context_transfer_allowed": bool(gate["passed"]),
                "raw_scores_must_not_feed_pnp": True,
                "appearance_control_geometry_fixed": True,
                "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
                "inner_validation_used_for_model_selection": False,
                "encoder_inputs": [
                    "query_image_xy",
                    "fixed_support_image_xy",
                    "full_2d_radio_final_context_crop_with_absolute_image_coordinates",
                    "full_2d_radio_intermediate_context_crop_with_absolute_image_coordinates",
                    "full_2d_alike_context_crop_with_absolute_image_coordinates",
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
                    "search_radius_px": float(args.search_radius_px),
                    "context_radius_px": float(args.context_radius_px),
                    "step_px": float(args.step_px),
                    "texture_feature_dim": int(args.texture_feature_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_context_log_ratio": float(args.max_abs_context_log_ratio),
                    "edge_chunk_size": int(args.edge_chunk_size),
                    "context_windows": dict(context_windows),
                    "context_encoder_arch": context_encoder_arch,
                    "context_only": True,
                },
                "training": {
                    "objective": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_OBSERVATION_PRETRAIN_OBJECTIVE,
                    "legacy_l0_objective": CANDIDATE_POSE_RGB_SPATIAL_CONTEXT_IDENTITY_PRETRAIN_OBJECTIVE,
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "identity_loss_weight": float(args.identity_loss_weight),
                    "support_permutation_loss_weight": float(args.support_permutation_loss_weight),
                    "support_permutation_margin": float(args.support_permutation_margin),
                    "candidate_slot_permutation": "anchor_id_seeded_per_observation_no_label_input_v1",
                    "support_appearance_control": (
                        "support_image_index_derangement_keep_support_xy_and_mixture_mass_v2"
                    ),
                    "position_only_control": "zero_frozen_descriptors_keep_absolute_coordinates_v1",
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only_inner_validation_images",
                        "selected_epoch": int(final_epoch),
                        "final_epoch_metrics": final_metrics,
                        "checkpoint_selection": fixed_final_epoch_checkpoint_selection(
                            epochs=int(args.epochs)
                        ),
                        "gate": gate,
                    },
                },
                "lineage": {
                    "observation_pairs_sha256": file_sha256_short(pair_path),
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
                },
            }
            torch.save(
                {"format": CHECKPOINT_FORMAT, "state_dict": final_state_dict, "metadata": metadata},
                checkpoint_path,
            )
            summary = {
                "stage": "pretrain_candidate_pose_context_identity_l0",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": file_sha256_short(checkpoint_path),
                "elapsed_seconds": float(time.time() - started),
                "history": history,
                "checkpoint_selection": {
                    **fixed_final_epoch_checkpoint_selection(epochs=int(args.epochs)),
                    "inner_validation": final_metrics,
                    "gate": gate,
                },
                "protocol": {
                    "train_only_observation_targets": True,
                    "runtime_checkpoint_target_free": True,
                    "context_only_no_rgb_or_spatial_density": True,
                    "candidate_slots_randomized_before_target_join": True,
                    "support_appearance_control_geometry_fixed": True,
                    "inner_validation_used_for_model_selection": False,
                    "position_only_control": True,
                    "validation_or_test_labels_used_by_fit": False,
                    "pnp_or_heldout_pose_not_run": True,
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
    result = pretrain_candidate_pose_context_identity_l0(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
