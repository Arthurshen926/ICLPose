"""Train a target-free multiscale phase-field candidate identity likelihood.

This is intentionally a separate expert from RGB local measurement.  Its
forward receives only the frozen P1 query/support layout and real-image
RADIO-final, RADIO-intermediate, and ALIKE descriptor grids.  It does not
receive a pose, residual, track ID, candidate rank, coarse score, or target.

Training joins registered exact-track labels and frozen coherent-wrong edges
*after* that visual forward.  Current-system hard negatives are restricted to
the inner-train fold; the final gate uses a distinct static hard-repeat target
on the held-out train-query fold.  A checkpoint that fails this gate is
diagnostic-only and must not be used for pose ranking or PnP.

Typical invocation::

    torchrun --standalone --nproc_per_node=2 \
      feature_extract/tools/vfm/train_candidate_multiscale_phase_identity_llr.py \
      --rgb-spatial-layout .../layout.npz \
      --geometry-training-targets .../s1673.../targets.npz \
      --registered-identity-targets .../s1487.../targets.npz \
      --current-hard-repeat-targets .../s1704.../hard_repeat_targets.npz \
      --current-hard-mining-checkpoint .../s1687...pt \
      --static-hard-repeat-targets .../s1674.../hard_repeat_targets.npz \
      --radio-final-context-cache ... --radio-intermediate-context-cache ... \
      --alike-spatial-context-cache ... --output-dir ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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

from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
    TrainQueryGroup,
    _DistributedState,
    _finalize_distributed,
    _hard_repeat_batch_from_group,
    _initialize_distributed,
    _partition_train_queries_for_inner_validation,
    _slice_runtime,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    require_current_inner_gate_evaluator_manifest,
    train_query_partition_manifest,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_multiscale_phase_identity_llr import (
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
    CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES,
    CandidateMultiscalePhaseIdentityLLR,
    CandidateMultiscalePhaseIdentityPrediction,
    candidate_phase_identity_log_likelihood_ratios,
    canonical_registered_identity_or_null_targets,
    current_hard_repeat_identity_margin_loss,
    exact_identity_or_null_cross_entropy,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import load_candidate_pose_rgb_spatial_layout
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialRuntime,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_repeat import (
    CandidatePoseRGBSpatialHardRepeatTargets,
    load_candidate_pose_rgb_spatial_hard_repeat_targets,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_system_hard import (
    CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
    SUPPORTED_CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMATS,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ContextAttentionSource,
    load_context_attention_sources,
)


CHECKPOINT_FORMAT = "candidate_multiscale_phase_identity_llr_checkpoint_v1"
GATE_FORMAT = "candidate_multiscale_phase_identity_static_hard_gate_v1"
EXTERNAL_HARD_NEGATIVE_PROTOCOL = (
    "frozen_previous_system_inner_train_only_coherent_wrong_identity_edges_v1"
)
FINAL_EPOCH_SELECTION_POLICY = "fixed_final_epoch_without_inner_validation_model_selection_v1"


@dataclass(frozen=True)
class ExactIdentityQueryTargets:
    """Registered exact identity labels aligned to one geometry query group."""

    query_id: str
    source_point_ids: np.ndarray
    observed_candidate_mask: np.ndarray
    candidate_dustbin_mask: np.ndarray
    candidate_supervised_mask: np.ndarray

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        observed = np.asarray(self.observed_candidate_mask, dtype=bool)
        dustbin = np.asarray(self.candidate_dustbin_mask, dtype=bool)
        supervised = np.asarray(self.candidate_supervised_mask, dtype=bool)
        if (
            not str(self.query_id)
            or len(source_ids) == 0
            or len(np.unique(source_ids)) != len(source_ids)
            or observed.ndim != 2
            or observed.shape[0] != len(source_ids)
            or observed.shape[1] < 2
            or dustbin.shape != observed.shape
            or supervised.shape != observed.shape
        ):
            raise ValueError("registered exact identity query target arrays are invalid")
        # Validate the non-negotiable semantic contract before training can
        # reduce it to a point-level class label.
        canonical_registered_identity_or_null_targets(
            observed_candidate_mask=torch.from_numpy(observed),
            candidate_dustbin_mask=torch.from_numpy(dustbin),
            candidate_supervised_mask=torch.from_numpy(supervised),
        )
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "observed_candidate_mask", observed)
        object.__setattr__(self, "candidate_dustbin_mask", dustbin)
        object.__setattr__(self, "candidate_supervised_mask", supervised)


@dataclass(frozen=True)
class PhaseIdentityBatch:
    """A concatenated target-free runtime plus train-only labels after forward."""

    runtime: CandidatePoseRGBSpatialRuntime
    observed_candidate_mask: torch.Tensor
    candidate_dustbin_mask: torch.Tensor
    candidate_supervised_mask: torch.Tensor
    current_hard_point_indices: torch.Tensor
    current_hard_positive_candidate_indices: torch.Tensor
    current_hard_negative_candidate_indices: torch.Tensor

    def __post_init__(self) -> None:
        observed = torch.as_tensor(self.observed_candidate_mask, dtype=torch.bool)
        dustbin = torch.as_tensor(self.candidate_dustbin_mask, dtype=torch.bool)
        supervised = torch.as_tensor(self.candidate_supervised_mask, dtype=torch.bool)
        point = torch.as_tensor(self.current_hard_point_indices, dtype=torch.long).reshape(-1)
        positive = torch.as_tensor(
            self.current_hard_positive_candidate_indices, dtype=torch.long
        ).reshape(-1)
        negative = torch.as_tensor(
            self.current_hard_negative_candidate_indices, dtype=torch.long
        ).reshape(-1)
        if (
            not isinstance(self.runtime, CandidatePoseRGBSpatialRuntime)
            or observed.shape
            != (self.runtime.point_count, self.runtime.candidate_count)
            or dustbin.shape != observed.shape
            or supervised.shape != observed.shape
            or point.shape != positive.shape
            or point.shape != negative.shape
            or torch.any(point < 0)
            or torch.any(point >= self.runtime.point_count)
            or torch.any(positive < 0)
            or torch.any(positive >= self.runtime.candidate_count)
            or torch.any(negative < 0)
            or torch.any(negative >= self.runtime.candidate_count)
            or torch.any(positive == negative)
        ):
            raise ValueError("phase identity training batch is invalid")
        object.__setattr__(self, "observed_candidate_mask", observed)
        object.__setattr__(self, "candidate_dustbin_mask", dustbin)
        object.__setattr__(self, "candidate_supervised_mask", supervised)
        object.__setattr__(self, "current_hard_point_indices", point)
        object.__setattr__(self, "current_hard_positive_candidate_indices", positive)
        object.__setattr__(self, "current_hard_negative_candidate_indices", negative)


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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--queries-per-step", type=int, default=2)
    parser.add_argument("--max-points-per-query", type=int, default=64)
    parser.add_argument("--max-current-hard-edges-per-query", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--max-abs-log-ratio", type=float, default=4.0)
    parser.add_argument("--identity-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-identity-loss-weight", type=float, default=0.20)
    parser.add_argument("--current-hard-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-current-hard-loss-weight", type=float, default=0.20)
    parser.add_argument("--current-hard-margin", type=float, default=0.25)
    parser.add_argument(
        "--identity-prior-logit-weight",
        type=float,
        default=0.0,
        help="Fixed coarse prior weight in identity CE; zero keeps visual LLR training independent.",
    )
    parser.add_argument("--inner-fold-count", type=int, default=5)
    parser.add_argument("--inner-fold-index", type=int, default=1)
    parser.add_argument("--gate-minimum-eligible-query-fraction", type=float, default=0.90)
    parser.add_argument("--gate-minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--gate-minimum-gap", type=float, default=0.05)
    parser.add_argument("--gate-minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--support-permutation-shift", type=int, default=1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--source-storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "epochs": args.epochs,
        "queries_per_step": args.queries_per_step,
        "max_points_per_query": args.max_points_per_query,
        "hidden_dim": args.hidden_dim,
        "inner_fold_count": args.inner_fold_count,
    }
    if any(int(value) <= 0 for value in positive_ints.values()) or int(args.max_points_per_query) < 4:
        raise ValueError("phase identity trainer integer arguments are invalid")
    if int(args.max_current_hard_edges_per_query) < 0 or int(args.inner_fold_index) < 0:
        raise ValueError("phase identity trainer edge/fold arguments are invalid")
    floats = (
        args.learning_rate,
        args.weight_decay,
        args.max_abs_log_ratio,
        args.identity_loss_weight,
        args.source_identity_loss_weight,
        args.current_hard_loss_weight,
        args.source_current_hard_loss_weight,
        args.current_hard_margin,
        args.identity_prior_logit_weight,
        args.gate_minimum_eligible_query_fraction,
        args.gate_minimum_win_fraction,
        args.gate_minimum_gap,
        args.gate_minimum_visual_gap_delta,
        args.gradient_clip_norm,
    )
    if (
        not all(math.isfinite(float(value)) for value in floats)
        or float(args.learning_rate) <= 0.0
        or float(args.weight_decay) < 0.0
        or float(args.max_abs_log_ratio) <= 0.0
        or any(float(value) < 0.0 for value in floats[3:])
        or not 0.0 < float(args.gate_minimum_eligible_query_fraction) <= 1.0
        or not 0.0 <= float(args.gate_minimum_win_fraction) <= 1.0
        or int(args.support_permutation_shift) <= 0
        or not 0 <= int(args.inner_fold_index) < int(args.inner_fold_count)
    ):
        raise ValueError("phase identity trainer floating-point arguments are invalid")
    if float(args.current_hard_loss_weight) <= 0.0:
        raise ValueError("phase identity training requires direct current hard-repeat supervision")


def _source_table(
    sources: Sequence[ContextAttentionSource],
) -> tuple[np.ndarray, np.ndarray, dict[str, torch.Tensor]]:
    by_name = {str(source.name): source for source in sources}
    if set(by_name) != set(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES):
        raise ValueError("phase identity sources are incomplete")
    reference = by_name["radio_final"]
    image_ids = np.asarray(reference.image_ids).astype(str)
    image_sizes = np.asarray(reference.image_sizes, dtype=np.int64)
    if len(image_ids) == 0 or image_sizes.shape != (len(image_ids), 2):
        raise ValueError("phase identity image table is invalid")
    grids: dict[str, torch.Tensor] = {}
    for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        source = by_name[name]
        if not np.array_equal(np.asarray(source.image_ids).astype(str), image_ids) or not np.array_equal(
            np.asarray(source.image_sizes, dtype=np.int64), image_sizes
        ):
            raise ValueError("phase identity source image ownership differs")
        grids[name] = torch.from_numpy(np.asarray(source.grid, dtype=np.float32))
    return image_ids, image_sizes, grids


def build_exact_identity_query_targets(
    *,
    groups: Mapping[str, TrainQueryGroup],
    identity_targets: CandidatePoseRGBSpatialTrainingTargets,
) -> dict[str, ExactIdentityQueryTargets]:
    """Join strict registered labels to geometry groups by source-point ID."""

    metadata = identity_targets.metadata
    if (
        str(metadata.get("format", "")) != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(metadata.get("spatial_supervision_mode", "")) != "registered_exact_identity"
        or str(metadata.get("spatial_target_semantics", ""))
        != "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
    ):
        raise ValueError("phase identity training requires registered exact identity targets")
    row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(identity_targets.source_point_ids.tolist())
    }
    output: dict[str, ExactIdentityQueryTargets] = {}
    for query_id, group in groups.items():
        try:
            rows = np.asarray(
                [row_by_source[int(source_id)] for source_id in group.source_point_ids.tolist()],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("registered identity target misses a geometry group source point") from error
        if not np.all(identity_targets.query_ids[rows].astype(str) == str(query_id)):
            raise ValueError("registered identity target query ownership differs from geometry group")
        output[str(query_id)] = ExactIdentityQueryTargets(
            query_id=str(query_id),
            source_point_ids=np.asarray(group.source_point_ids, dtype=np.int64),
            observed_candidate_mask=np.asarray(identity_targets.spatial_target_observed[rows], dtype=bool),
            candidate_dustbin_mask=np.asarray(identity_targets.spatial_target_dustbin[rows], dtype=bool),
            candidate_supervised_mask=np.asarray(
                identity_targets.spatial_target_supervised[rows], dtype=bool
            ),
        )
    target_queries = set(identity_targets.query_ids.astype(str).tolist())
    if set(output) != target_queries:
        raise ValueError("registered identity target/group query coverage differs")
    return output


def validate_external_frozen_current_hard_targets(
    *,
    mined_targets: CandidatePoseRGBSpatialHardRepeatTargets,
    mined_groups: Mapping[str, HardRepeatQueryTargets],
    registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets,
    registered_identity_targets_sha256: str,
    expected_partition: Mapping[str, object],
    mining_checkpoint_path: Path,
) -> dict[str, object]:
    """Validate prior-system errors as external train-only supervision.

    This expert has a different architecture from the RGB system that mined
    ``s1704``.  It therefore must not claim checkpoint continuation.  The
    provenance check instead proves that the previous system was frozen and
    gate-approved, and that its mined examples exclude this expert's held-out
    fold.
    """

    metadata = mined_targets.metadata
    mining_format = str(metadata.get("mining_format", ""))
    if (
        mining_format not in SUPPORTED_CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMATS
        or str(metadata.get("score_component", "")) != "combined"
        or str(metadata.get("registered_identity_targets_sha256", ""))
        != str(registered_identity_targets_sha256)
    ):
        raise ValueError("external current hard targets have incompatible provenance")
    if mining_format == CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT:
        wrong_mode_selection = metadata.get("wrong_mode_selection")
        try:
            valid_multi_mode_contract = (
                isinstance(wrong_mode_selection, Mapping)
                and wrong_mode_selection.get("policy")
                == "target_free_combined_pose_llr_descending_pair_id_tiebreak_top_h_v1"
                and int(wrong_mode_selection.get("max_wrong_modes_per_query", 0)) > 1
                and wrong_mode_selection.get("target_join_after_mode_ranking") is True
                and wrong_mode_selection.get("label_based_mode_backfill") is False
            )
        except (TypeError, ValueError):
            valid_multi_mode_contract = False
        if not valid_multi_mode_contract:
            raise ValueError("external multi-mode current hard target contract is invalid")
    identity_metadata = registered_identity_targets.metadata
    if (
        str(identity_metadata.get("format", "")) != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(identity_metadata.get("spatial_supervision_mode", "")) != "registered_exact_identity"
    ):
        raise ValueError("external current hard targets lack registered identity lineage")
    frozen = metadata.get("frozen_checkpoint")
    partition = metadata.get("train_query_partition")
    config = metadata.get("mining_checkpoint_config")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("train_only_inner_gate_passed") is not True
        or not isinstance(partition, Mapping)
        or not isinstance(config, Mapping)
    ):
        raise ValueError("external current hard target lineage is incomplete")
    require_current_inner_gate_evaluator_manifest(
        metadata.get("inner_gate_evaluator_manifest"),
        subject="external current hard target",
    )
    require_current_inner_gate_evaluator_manifest(
        frozen.get("inner_gate_evaluator_manifest"),
        subject="external current hard frozen checkpoint",
    )
    try:
        rebuilt = train_query_partition_manifest(
            all_query_ids=partition["all_train"]["query_ids"],  # type: ignore[index]
            inner_train_query_ids=partition["inner_train"]["query_ids"],  # type: ignore[index]
            inner_validation_query_ids=partition["inner_validation"]["query_ids"],  # type: ignore[index]
            fold_count=int(partition["fold_count"]),  # type: ignore[index]
            fold_index=int(partition["fold_index"]),  # type: ignore[index]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("external current hard target partition is malformed") from error
    if dict(partition) != rebuilt or dict(rebuilt) != dict(expected_partition):
        raise ValueError("external current hard target partition differs from this fold")
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    config_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    checkpoint_sha = file_sha256_short(Path(mining_checkpoint_path))
    if (
        str(metadata.get("mining_checkpoint_config_sha256", "")) != config_sha
        or str(metadata.get("mining_checkpoint_sha256", "")) != checkpoint_sha
        or str(frozen.get("sha256", "")) != checkpoint_sha
    ):
        raise ValueError("external current hard target checkpoint lineage is stale")
    inner_train = set(rebuilt["inner_train"]["query_ids"])  # type: ignore[index]
    inner_validation = set(rebuilt["inner_validation"]["query_ids"])  # type: ignore[index]
    hard_queries = set(mined_targets.query_ids.astype(str).tolist())
    if (
        not hard_queries
        or not hard_queries.issubset(inner_train)
        or hard_queries.intersection(inner_validation)
        or hard_queries != set(mined_groups)
    ):
        raise ValueError("external current hard targets leak into the held-out fold")
    identity_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(registered_identity_targets.source_point_ids.tolist())
    }
    try:
        rows = np.asarray(
            [identity_row_by_source[int(source_id)] for source_id in mined_targets.source_point_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("external current hard edge is absent from registered identity targets") from error
    observed = np.asarray(registered_identity_targets.spatial_target_observed[rows], dtype=bool)
    if (
        np.any(observed.sum(axis=1) != 1)
        or np.any(
            registered_identity_targets.query_ids[rows].astype(str)
            != mined_targets.query_ids.astype(str)
        )
        or not bool(np.all(observed[np.arange(len(rows)), mined_targets.positive_candidate_indices]))
        or bool(np.any(observed[np.arange(len(rows)), mined_targets.negative_candidate_indices]))
    ):
        raise ValueError("external current hard targets do not preserve exact identity polarity")
    return {
        "protocol": EXTERNAL_HARD_NEGATIVE_PROTOCOL,
        "mining_checkpoint_sha256": checkpoint_sha,
        "mining_checkpoint_config_sha256": config_sha,
        "frozen_system_gate_approved": True,
        "selection_scope": "inner_train_only_excluding_this_expert_gate_fold_v1",
        "mined_query_count": int(len(hard_queries)),
        "mined_edge_count": int(mined_targets.count),
        "registered_identity_polarity_validation": "exact_positive_and_distinct_negative_v1",
    }


def filter_static_hard_repeat_groups_to_registered_exact_identity(
    *,
    static_groups: Mapping[str, HardRepeatQueryTargets],
    registered_identity_targets: CandidatePoseRGBSpatialTrainingTargets,
) -> tuple[dict[str, HardRepeatQueryTargets], dict[str, object]]:
    """Keep only static hard edges with an exact registered positive track.

    Older static hard-repeat artifacts were built from correct-pose geometric
    visibility.  Those are useful for a spatial-density branch but cannot be
    used as a gate for an identity LLR: a geometrically local candidate need
    not be the SfM track observed by the query.  This train-only filter makes
    the two semantics explicit without changing the frozen runtime layout.
    """

    metadata = registered_identity_targets.metadata
    if (
        str(metadata.get("format", "")) != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        or str(metadata.get("spatial_supervision_mode", "")) != "registered_exact_identity"
        or str(metadata.get("spatial_target_semantics", ""))
        != "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
    ):
        raise ValueError("static identity hard filter requires registered exact identity targets")
    row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(registered_identity_targets.source_point_ids.tolist())
    }
    observed = np.asarray(registered_identity_targets.spatial_target_observed, dtype=bool)
    supervised = np.asarray(registered_identity_targets.spatial_target_supervised, dtype=bool)
    identity_queries = registered_identity_targets.query_ids.astype(str)
    output: dict[str, HardRepeatQueryTargets] = {}
    input_count = 0
    retained_count = 0
    per_query: dict[str, int] = {}
    for query_id, hard in static_groups.items():
        try:
            rows = np.asarray(
                [row_by_source[int(source_id)] for source_id in hard.source_point_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("static hard edge is absent from registered identity targets") from error
        positive = np.asarray(hard.positive_candidate_indices, dtype=np.int64)
        negative = np.asarray(hard.negative_candidate_indices, dtype=np.int64)
        if (
            np.any(identity_queries[rows] != str(query_id))
            or np.any(observed.shape[1] <= positive)
            or np.any(observed.shape[1] <= negative)
        ):
            raise ValueError("static hard identity filter candidate/query lineage is invalid")
        mask = (
            supervised[rows].all(axis=1)
            & (observed[rows].sum(axis=1) == 1)
            & observed[rows, positive]
            & ~observed[rows, negative]
        )
        input_count += int(len(mask))
        retained_count += int(mask.sum())
        if not bool(mask.any()):
            continue
        selected = np.flatnonzero(mask)
        output[str(query_id)] = HardRepeatQueryTargets(
            query_id=str(query_id),
            source_point_ids=hard.source_point_ids[selected],
            pair_ids=hard.pair_ids[selected],
            positive_candidate_indices=hard.positive_candidate_indices[selected],
            negative_candidate_indices=hard.negative_candidate_indices[selected],
            positive_offsets_xy=hard.positive_offsets_xy[selected],
            negative_offsets_xy=hard.negative_offsets_xy[selected],
        )
        per_query[str(query_id)] = int(len(selected))
    if input_count == 0 or retained_count == 0 or not output:
        raise ValueError("static hard identity filter retained no exact candidate edges")
    return output, {
        "format": "registered_exact_identity_filter_of_geometry_static_hard_repeat_v1",
        "input_edge_count": int(input_count),
        "retained_edge_count": int(retained_count),
        "retained_fraction": float(retained_count / input_count),
        "retained_query_count": int(len(output)),
        "per_query_retained_edge_count": per_query,
        "positive_contract": "registered_exact_track_and_distinct_nonobserved_wrong_candidate_v1",
    }


def _concat_runtime(runtimes: Sequence[CandidatePoseRGBSpatialRuntime]) -> CandidatePoseRGBSpatialRuntime:
    if not runtimes:
        raise ValueError("phase identity runtime batch is empty")
    candidate_count = runtimes[0].candidate_count
    view_count = runtimes[0].support_view_count
    if any(
        runtime.candidate_count != candidate_count or runtime.support_view_count != view_count
        for runtime in runtimes
    ):
        raise ValueError("phase identity runtime batch candidate geometry differs")
    return CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.cat([runtime.query_image_indices for runtime in runtimes], dim=0),
        query_xy=torch.cat([runtime.query_xy for runtime in runtimes], dim=0),
        support_image_indices=torch.cat([runtime.support_image_indices for runtime in runtimes], dim=0),
        support_xy=torch.cat([runtime.support_xy for runtime in runtimes], dim=0),
        support_view_valid=torch.cat([runtime.support_view_valid for runtime in runtimes], dim=0),
        candidate_view_weights=torch.cat(
            [runtime.candidate_view_weights for runtime in runtimes], dim=0
        ),
        candidate_probabilities=torch.cat(
            [runtime.candidate_probabilities for runtime in runtimes], dim=0
        ),
        null_probabilities=torch.cat([runtime.null_probabilities for runtime in runtimes], dim=0),
    )


def _sample_training_positions(
    *,
    group: TrainQueryGroup,
    exact: ExactIdentityQueryTargets,
    current_hard: HardRepeatQueryTargets | None,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Keep train-only hard/registered points without placing labels in runtime."""

    limit = int(max_points)
    if limit <= 0 or limit >= group.point_count:
        return np.arange(group.point_count, dtype=np.int64)
    source_position = {int(source_id): row for row, source_id in enumerate(group.source_point_ids)}
    required_sources = (
        np.zeros((0,), dtype=np.int64)
        if current_hard is None
        else np.unique(np.asarray(current_hard.source_point_ids, dtype=np.int64))
    )
    try:
        required = np.asarray(
            [source_position[int(source_id)] for source_id in required_sources], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("current hard target source is absent from its query group") from error
    observed = np.flatnonzero(np.asarray(exact.observed_candidate_mask, dtype=bool).any(axis=1))
    rng = np.random.default_rng(int(seed) ^ int.from_bytes(
        hashlib.sha256(str(group.query_id).encode("utf-8")).digest()[:8], byteorder="big"
    ))
    required = np.unique(required)
    if len(required) >= limit:
        return np.sort(rng.choice(required, size=limit, replace=False)).astype(np.int64)
    priority = np.unique(np.concatenate([required, observed])).astype(np.int64)
    if len(priority) > limit:
        optional = np.setdiff1d(priority, required, assume_unique=False)
        chosen = np.concatenate(
            [required, rng.choice(optional, size=limit - len(required), replace=False)]
        )
        return np.sort(chosen).astype(np.int64)
    remaining = np.setdiff1d(
        np.arange(group.point_count, dtype=np.int64), priority, assume_unique=True
    )
    if len(priority) == limit:
        return np.sort(priority)
    return np.sort(
        np.concatenate([priority, rng.choice(remaining, size=limit - len(priority), replace=False)])
    ).astype(np.int64)


def build_phase_identity_batch(
    *,
    query_ids: Sequence[str],
    groups: Mapping[str, TrainQueryGroup],
    exact_by_query: Mapping[str, ExactIdentityQueryTargets],
    current_hard_by_query: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    max_points_per_query: int,
    max_current_hard_edges_per_query: int,
    seed: int,
    device: torch.device,
) -> PhaseIdentityBatch:
    """Build a multi-query GPU step while retaining hard-edge source mapping."""

    runtimes: list[CandidatePoseRGBSpatialRuntime] = []
    observed_parts: list[torch.Tensor] = []
    dustbin_parts: list[torch.Tensor] = []
    supervised_parts: list[torch.Tensor] = []
    hard_points: list[torch.Tensor] = []
    hard_positive: list[torch.Tensor] = []
    hard_negative: list[torch.Tensor] = []
    point_offset = 0
    for position, query_id in enumerate(query_ids):
        group = groups.get(str(query_id))
        exact = exact_by_query.get(str(query_id))
        if group is None or exact is None:
            raise ValueError("phase identity train query is unresolved")
        hard = current_hard_by_query.get(str(query_id))
        selected = _sample_training_positions(
            group=group,
            exact=exact,
            current_hard=hard,
            max_points=int(max_points_per_query),
            seed=int(seed) + position * 7919,
        )
        runtime = _slice_runtime(complete_runtime, group.layout_rows[selected])
        runtimes.append(runtime)
        observed_parts.append(torch.from_numpy(exact.observed_candidate_mask[selected]))
        dustbin_parts.append(torch.from_numpy(exact.candidate_dustbin_mask[selected]))
        supervised_parts.append(torch.from_numpy(exact.candidate_supervised_mask[selected]))
        if hard is not None:
            edge_batch = _hard_repeat_batch_from_group(
                hard_targets=hard,
                group=group,
                point_positions=selected,
                device=device,
                max_edges=int(max_current_hard_edges_per_query),
                seed=int(seed) + 104729 * (position + 1),
            )
            if edge_batch is not None:
                hard_points.append(edge_batch.point_indices + int(point_offset))
                hard_positive.append(edge_batch.positive_candidate_indices)
                hard_negative.append(edge_batch.negative_candidate_indices)
        point_offset += runtime.point_count
    if not runtimes:
        raise ValueError("phase identity train step has no query")
    empty = torch.zeros((0,), dtype=torch.long, device=device)
    return PhaseIdentityBatch(
        runtime=_concat_runtime(runtimes),
        observed_candidate_mask=torch.cat(observed_parts, dim=0),
        candidate_dustbin_mask=torch.cat(dustbin_parts, dim=0),
        candidate_supervised_mask=torch.cat(supervised_parts, dim=0),
        current_hard_point_indices=torch.cat(hard_points, dim=0) if hard_points else empty,
        current_hard_positive_candidate_indices=(
            torch.cat(hard_positive, dim=0) if hard_positive else empty
        ),
        current_hard_negative_candidate_indices=(
            torch.cat(hard_negative, dim=0) if hard_negative else empty
        ),
    )


def _phase_identity_losses(
    *,
    model: torch.nn.Module,
    batch: PhaseIdentityBatch,
    device: torch.device,
    identity_prior_logit_weight: float,
    identity_loss_weight: float,
    source_identity_loss_weight: float,
    current_hard_loss_weight: float,
    source_current_hard_loss_weight: float,
    current_hard_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Join all target-bearing objectives only after target-free inference."""

    prediction = model(runtime=batch.runtime)
    if not isinstance(prediction, CandidateMultiscalePhaseIdentityPrediction):
        raise RuntimeError("phase identity model returned an unexpected prediction")
    observed, dustbin, supervised = canonical_registered_identity_or_null_targets(
        observed_candidate_mask=batch.observed_candidate_mask.to(device=device),
        candidate_dustbin_mask=batch.candidate_dustbin_mask.to(device=device),
        candidate_supervised_mask=batch.candidate_supervised_mask.to(device=device),
    )
    combined_identity, combined_identity_metrics = exact_identity_or_null_cross_entropy(
        runtime=batch.runtime,
        prediction=prediction,
        observed_candidate_mask=observed,
        target_dustbin=dustbin,
        target_supervised=supervised,
        candidate_prior_logit_weight=float(identity_prior_logit_weight),
        balance_observed_and_null=True,
    )
    total = float(identity_loss_weight) * combined_identity
    metrics: dict[str, float] = {
        "identity_loss": float(combined_identity.detach().item()),
        **{f"identity_{key}": float(value) for key, value in combined_identity_metrics.items()},
    }
    for source_name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        source_identity, source_metrics = exact_identity_or_null_cross_entropy(
            runtime=batch.runtime,
            prediction=prediction,
            observed_candidate_mask=observed,
            target_dustbin=dustbin,
            target_supervised=supervised,
            candidate_prior_logit_weight=float(identity_prior_logit_weight),
            source_name=source_name,
            balance_observed_and_null=True,
        )
        total = total + float(source_identity_loss_weight) * source_identity
        metrics[f"{source_name}_identity_loss"] = float(source_identity.detach().item())
        metrics[f"{source_name}_identity_active"] = float(source_metrics["identity_active"])
    if len(batch.current_hard_point_indices):
        combined_hard, combined_hard_metrics = current_hard_repeat_identity_margin_loss(
            runtime=batch.runtime,
            prediction=prediction,
            point_indices=batch.current_hard_point_indices,
            positive_candidate_indices=batch.current_hard_positive_candidate_indices,
            negative_candidate_indices=batch.current_hard_negative_candidate_indices,
            margin=float(current_hard_margin),
        )
    else:
        combined_hard = prediction.edge_log_likelihood_ratios.sum() * 0.0
        combined_hard_metrics = {
            "hard_repeat_active": 0.0,
            "hard_repeat_mean_gap": 0.0,
            "hard_repeat_win_fraction": 0.0,
            "hard_repeat_margin_loss": 0.0,
        }
    total = total + float(current_hard_loss_weight) * combined_hard
    metrics.update(
        {f"current_hard_{key}": float(value) for key, value in combined_hard_metrics.items()}
    )
    for source_name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES:
        if len(batch.current_hard_point_indices):
            source_hard, source_hard_metrics = current_hard_repeat_identity_margin_loss(
                runtime=batch.runtime,
                prediction=prediction,
                point_indices=batch.current_hard_point_indices,
                positive_candidate_indices=batch.current_hard_positive_candidate_indices,
                negative_candidate_indices=batch.current_hard_negative_candidate_indices,
                margin=float(current_hard_margin),
                source_name=source_name,
            )
        else:
            source_hard = prediction.source_edge_log_likelihood_ratios[source_name].sum() * 0.0
            source_hard_metrics = {"hard_repeat_active": 0.0, "hard_repeat_mean_gap": 0.0}
        total = total + float(source_current_hard_loss_weight) * source_hard
        metrics[f"{source_name}_current_hard_loss"] = float(source_hard.detach().item())
        metrics[f"{source_name}_current_hard_active"] = float(
            source_hard_metrics["hard_repeat_active"]
        )
    if not torch.isfinite(total):
        raise RuntimeError("phase identity training loss became non-finite")
    metrics["total_loss"] = float(total.detach().item())
    return total, metrics


def _reduce_sum(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    output = values.clone()
    if state.enabled:
        distributed.all_reduce(output, op=distributed.ReduceOp.SUM)
    return output


def phase_identity_static_hard_gate_decision(
    metrics: Mapping[str, float],
    *,
    minimum_eligible_query_fraction: float,
    minimum_win_fraction: float,
    minimum_gap: float,
    minimum_visual_gap_delta: float,
) -> dict[str, float | bool]:
    """Apply one identical paired gate to fused and source-only evidence."""

    try:
        coverage = float(metrics["eligible_query_fraction"])
        win = float(metrics["normal_win_fraction"])
        gap = float(metrics["normal_mean_positive_minus_negative"])
        permuted = float(metrics["permuted_mean_positive_minus_negative"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("phase identity static hard metrics are incomplete") from error
    thresholds = (
        float(minimum_eligible_query_fraction),
        float(minimum_win_fraction),
        float(minimum_gap),
        float(minimum_visual_gap_delta),
    )
    if (
        not all(math.isfinite(value) for value in (coverage, win, gap, permuted, *thresholds))
        or not 0.0 < thresholds[0] <= 1.0
        or not 0.0 <= thresholds[1] <= 1.0
        or thresholds[2] < 0.0
        or thresholds[3] < 0.0
    ):
        raise ValueError("phase identity static hard gate values are invalid")
    visual_gap = gap - permuted
    return {
        "passed": bool(
            coverage >= thresholds[0]
            and win >= thresholds[1]
            and gap >= thresholds[2]
            and visual_gap >= thresholds[3]
        ),
        "eligible_query_fraction": coverage,
        "normal_win_fraction": win,
        "normal_mean_positive_minus_negative": gap,
        "permuted_mean_positive_minus_negative": permuted,
        "normal_minus_permuted_gap": visual_gap,
        "minimum_eligible_query_fraction": thresholds[0],
        "minimum_win_fraction": thresholds[1],
        "minimum_gap": thresholds[2],
        "minimum_visual_gap_delta": thresholds[3],
    }


@torch.no_grad()
def evaluate_static_hard_gate(
    *,
    model: CandidateMultiscalePhaseIdentityLLR,
    groups: Mapping[str, TrainQueryGroup],
    exact_by_query: Mapping[str, ExactIdentityQueryTargets],
    static_hard_by_query: Mapping[str, HardRepeatQueryTargets],
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    query_ids: Sequence[str],
    state: _DistributedState,
    support_permutation_shift: int,
    gate_thresholds: Mapping[str, float],
    static_hard_semantics: Mapping[str, object],
) -> dict[str, object]:
    """Score static held-out hard edges after an all-point visual forward.

    The forward runs every frozen P1 point in a query.  Static hard labels are
    joined only afterwards to measure paired normal/permuted candidate gaps;
    they never select the visual input pool.
    """

    if not query_ids or int(support_permutation_shift) <= 0:
        raise ValueError("phase identity static gate arguments are invalid")
    source_names = ("combined", *CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES)
    # normal gap sum, normal win sum, permuted gap sum, permuted win sum,
    # active edge count, eligible query count, all query count.
    totals = torch.zeros((len(source_names), 7), dtype=torch.float64, device=state.device)
    model.eval()
    for query_position, query_id in enumerate(query_ids):
        if query_position % state.world_size != state.rank:
            continue
        group = groups.get(str(query_id))
        hard = static_hard_by_query.get(str(query_id))
        exact = exact_by_query.get(str(query_id))
        if group is None or hard is None or exact is None:
            raise ValueError("phase identity static gate query is unresolved")
        # Do not assume query rows are contiguous: full layouts are currently
        # ordered that way, but an index-preserving manifest rebuild need not
        # retain this incidental storage order.
        runtime = _slice_runtime(complete_runtime, group.layout_rows)
        normal = model(runtime=runtime)
        permuted = model(
            runtime=runtime, support_permutation_shift=int(support_permutation_shift)
        )
        source_position = {
            int(source_id): index for index, source_id in enumerate(group.source_point_ids.tolist())
        }
        points = torch.tensor(
            [source_position[int(source_id)] for source_id in hard.source_point_ids.tolist()],
            dtype=torch.long,
            device=state.device,
        )
        positive = torch.from_numpy(
            np.asarray(hard.positive_candidate_indices, dtype=np.int64)
        ).to(device=state.device)
        negative = torch.from_numpy(
            np.asarray(hard.negative_candidate_indices, dtype=np.int64)
        ).to(device=state.device)
        for source_index, source_name in enumerate(source_names):
            name = None if source_name == "combined" else source_name
            normal_values, normal_usable = candidate_phase_identity_log_likelihood_ratios(
                runtime=runtime, prediction=normal, source_name=name
            )
            permuted_values, permuted_usable = candidate_phase_identity_log_likelihood_ratios(
                runtime=runtime, prediction=permuted, source_name=name
            )
            active = (
                normal_usable[points, positive]
                & normal_usable[points, negative]
                & permuted_usable[points, positive]
                & permuted_usable[points, negative]
            )
            totals[source_index, 6] += 1.0
            if bool(active.any()):
                normal_gap = normal_values[points[active], positive[active]] - normal_values[
                    points[active], negative[active]
                ]
                permuted_gap = permuted_values[points[active], positive[active]] - permuted_values[
                    points[active], negative[active]
                ]
                totals[source_index, 0] += normal_gap.double().sum()
                totals[source_index, 1] += (normal_gap > 0.0).double().sum()
                totals[source_index, 2] += permuted_gap.double().sum()
                totals[source_index, 3] += (permuted_gap > 0.0).double().sum()
                totals[source_index, 4] += active.double().sum()
                totals[source_index, 5] += 1.0
    totals = _reduce_sum(state, totals)
    results: dict[str, object] = {}
    for source_index, source_name in enumerate(source_names):
        count = float(totals[source_index, 4].item())
        queries = float(totals[source_index, 6].item())
        if count <= 0.0 or queries <= 0.0:
            values = {
                "active_edge_count": count,
                "eligible_query_fraction": 0.0,
                "normal_mean_positive_minus_negative": 0.0,
                "normal_win_fraction": 0.0,
                "permuted_mean_positive_minus_negative": 0.0,
                "permuted_win_fraction": 0.0,
                "query_count": queries,
            }
        else:
            values = {
                "active_edge_count": count,
                "eligible_query_fraction": float(totals[source_index, 5].item() / queries),
                "normal_mean_positive_minus_negative": float(totals[source_index, 0].item() / count),
                "normal_win_fraction": float(totals[source_index, 1].item() / count),
                "permuted_mean_positive_minus_negative": float(
                    totals[source_index, 2].item() / count
                ),
                "permuted_win_fraction": float(totals[source_index, 3].item() / count),
                "query_count": queries,
            }
        results[source_name] = {
            **values,
            "gate": phase_identity_static_hard_gate_decision(
                values,
                minimum_eligible_query_fraction=float(
                    gate_thresholds["minimum_eligible_query_fraction"]
                ),
                minimum_win_fraction=float(gate_thresholds["minimum_win_fraction"]),
                minimum_gap=float(gate_thresholds["minimum_gap"]),
                minimum_visual_gap_delta=float(gate_thresholds["minimum_visual_gap_delta"]),
            ),
        }
    combined_pass = bool(results["combined"]["gate"]["passed"])  # type: ignore[index]
    source_passes = {
        name: bool(results[name]["gate"]["passed"])  # type: ignore[index]
        for name in CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES
    }
    promotable_sources = tuple(name for name, passed in source_passes.items() if passed)
    return {
        "format": GATE_FORMAT,
        "selection": "all_frozen_p1_points_forward_then_registered_exact_static_hard_label_join_v1",
        "static_hard_semantics": dict(static_hard_semantics),
        "support_permutation": "distant_point_block_support_appearance_derangement_common_availability_v2",
        "thresholds": dict(gate_thresholds),
        "sources": results,
        "combined_passed": combined_pass,
        "independent_source_passes": source_passes,
        # The source heads are explicitly not assumed independent probability
        # factors. A failed equal-weight blend must not veto a source that has
        # independently passed normal/permuted exact-identity evidence. Any
        # downstream scorer must name that selected source explicitly rather
        # than silently falling back to the fused tensor.
        "promotable_source_names": list(promotable_sources),
        "promotion_mode": "combined_if_passed_else_explicit_gate_passed_source_only_v1",
        "passed": bool(combined_pass or promotable_sources),
    }


def _rank_query_batches(
    *, query_ids: Sequence[str], state: _DistributedState, queries_per_step: int, seed: int
) -> list[tuple[str, ...]]:
    ordered = list(sorted(str(query_id) for query_id in query_ids))
    random.Random(int(seed)).shuffle(ordered)
    local = ordered[state.rank :: state.world_size]
    batches = [tuple(local[start : start + int(queries_per_step)]) for start in range(0, len(local), int(queries_per_step))]
    if not batches:
        raise ValueError("phase identity rank received no train queries")
    local_count = torch.tensor([len(batches)], dtype=torch.int64, device=state.device)
    if state.enabled:
        distributed.all_reduce(local_count, op=distributed.ReduceOp.MAX)
    required = int(local_count.item())
    while len(batches) < required:
        batches.append(batches[len(batches) % len(batches)])
    return batches


def _serialize_source_configs(model: CandidateMultiscalePhaseIdentityLLR) -> dict[str, dict[str, int | str]]:
    return {
        name: {
            "name": str(config.name),
            "window_size": int(config.window_size),
            "shift_radius": int(config.shift_radius),
            "region_bins": int(config.region_bins),
        }
        for name, config in model.source_configs.items()
    }


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _atomic_json_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    state = _initialize_distributed(str(args.device))
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        random.seed(int(args.seed))
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_multiscale_phase_identity_llr.pt"
        report_path = output_dir / "training_report.json"
        if checkpoint_path.exists() and not bool(args.force):
            raise FileExistsError(f"phase identity checkpoint already exists: {checkpoint_path}")
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
        layout = load_candidate_pose_rgb_spatial_layout(layout_path)
        geometry_targets = load_candidate_pose_rgb_spatial_training_targets(geometry_path)
        identity_targets = load_candidate_pose_rgb_spatial_training_targets(identity_path)
        layout_sha = file_sha256_short(layout_path)
        geometry_sha = file_sha256_short(geometry_path)
        identity_sha = file_sha256_short(identity_path)
        validate_training_layout_and_targets(
            layout=layout, targets=geometry_targets, layout_sha256=layout_sha
        )
        validate_training_layout_and_targets(
            layout=layout, targets=identity_targets, layout_sha256=layout_sha
        )
        groups = build_train_query_groups(layout=layout, targets=geometry_targets)
        exact_by_query = build_exact_identity_query_targets(
            groups=groups, identity_targets=identity_targets
        )
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
        static_hard_groups, static_hard_filter = (
            filter_static_hard_repeat_groups_to_registered_exact_identity(
                static_groups=static_hard_geometry_groups,
                registered_identity_targets=identity_targets,
            )
        )
        if not set(inner_validation_ids).issubset(static_hard_groups):
            raise ValueError("static hard target lacks an inner-validation query")

        sources = load_context_attention_sources(
            radio_final_context_cache=Path(args.radio_final_context_cache),
            radio_intermediate_context_cache=Path(args.radio_intermediate_context_cache),
            alike_spatial_context_cache=Path(args.alike_spatial_context_cache),
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_grids = _source_table(sources)
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        storage_dtype = torch.float16 if str(args.source_storage_dtype) == "float16" else torch.float32
        model = CandidateMultiscalePhaseIdentityLLR(
            sources=source_grids,
            image_sizes=torch.from_numpy(image_sizes),
            hidden_dim=int(args.hidden_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
            source_storage_dtype=storage_dtype,
        ).to(state.device)
        model_for_train: torch.nn.Module = model
        if state.enabled:
            # The aligned descriptor grids are immutable, non-persistent
            # buffers loaded independently on each rank. Broadcasting them on
            # every DDP forward would dominate training time and is unnecessary.
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
        epoch_logs: list[dict[str, object]] = []
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            query_batches = _rank_query_batches(
                query_ids=inner_train_ids,
                state=state,
                queries_per_step=int(args.queries_per_step),
                seed=int(args.seed) + epoch * 1000003,
            )
            sum_loss = torch.zeros((), dtype=torch.float64, device=state.device)
            sum_identity_loss = torch.zeros((), dtype=torch.float64, device=state.device)
            sum_current_hard_loss = torch.zeros((), dtype=torch.float64, device=state.device)
            sum_hard_gap = torch.zeros((), dtype=torch.float64, device=state.device)
            sum_hard_win = torch.zeros((), dtype=torch.float64, device=state.device)
            sum_hard_active = torch.zeros((), dtype=torch.float64, device=state.device)
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
                loss, metrics = _phase_identity_losses(
                    model=model_for_train,
                    batch=batch,
                    device=state.device,
                    identity_prior_logit_weight=float(args.identity_prior_logit_weight),
                    identity_loss_weight=float(args.identity_loss_weight),
                    source_identity_loss_weight=float(args.source_identity_loss_weight),
                    current_hard_loss_weight=float(args.current_hard_loss_weight),
                    source_current_hard_loss_weight=float(args.source_current_hard_loss_weight),
                    current_hard_margin=float(args.current_hard_margin),
                )
                loss.backward()
                if float(args.gradient_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_for_train.parameters(), max_norm=float(args.gradient_clip_norm)
                    )
                optimizer.step()
                sum_loss += loss.detach().double()
                sum_identity_loss += float(metrics["identity_loss"])
                sum_current_hard_loss += float(metrics["current_hard_hard_repeat_margin_loss"])
                active = float(metrics["current_hard_hard_repeat_active"])
                sum_hard_gap += float(metrics["current_hard_hard_repeat_mean_gap"]) * active
                sum_hard_win += float(metrics["current_hard_hard_repeat_win_fraction"]) * active
                sum_hard_active += active
            totals = _reduce_sum(
                state,
                torch.stack(
                    [
                        sum_loss,
                        sum_identity_loss,
                        sum_current_hard_loss,
                        sum_hard_gap,
                        sum_hard_win,
                        sum_hard_active,
                        torch.tensor(
                            float(len(query_batches)), dtype=torch.float64, device=state.device
                        ),
                    ]
                ),
            )
            if state.rank == 0:
                epoch_log = {
                    "epoch": int(epoch + 1),
                    "train_step_count": int(totals[6].item()),
                    "mean_total_loss": float((totals[0] / totals[6].clamp_min(1.0)).item()),
                    "mean_identity_loss": float((totals[1] / totals[6].clamp_min(1.0)).item()),
                    "mean_current_hard_margin_loss": float(
                        (totals[2] / totals[6].clamp_min(1.0)).item()
                    ),
                    "current_hard_active_edges": float(totals[5].item()),
                    "current_hard_mean_gap": float(
                        (totals[3] / totals[5].clamp_min(1.0)).item()
                    ),
                    "current_hard_win_fraction": float(
                        (totals[4] / totals[5].clamp_min(1.0)).item()
                    ),
                }
                epoch_logs.append(epoch_log)
                print(json.dumps({"phase_identity_train": epoch_log}, sort_keys=True), flush=True)

        base_model = model_for_train.module if isinstance(model_for_train, DistributedDataParallel) else model
        gate_thresholds = {
            "minimum_eligible_query_fraction": float(args.gate_minimum_eligible_query_fraction),
            "minimum_win_fraction": float(args.gate_minimum_win_fraction),
            "minimum_gap": float(args.gate_minimum_gap),
            "minimum_visual_gap_delta": float(args.gate_minimum_visual_gap_delta),
        }
        gate = evaluate_static_hard_gate(
            model=base_model,
            groups=groups,
            exact_by_query=exact_by_query,
            static_hard_by_query=static_hard_groups,
            complete_runtime=complete_runtime,
            query_ids=inner_validation_ids,
            state=state,
            support_permutation_shift=int(args.support_permutation_shift),
            gate_thresholds=gate_thresholds,
            static_hard_semantics=static_hard_filter,
        )
        if state.enabled:
            distributed.barrier()
        if state.rank == 0:
            source_paths = {
                "radio_final": Path(args.radio_final_context_cache),
                "radio_intermediate": Path(args.radio_intermediate_context_cache),
                "alike": Path(args.alike_spatial_context_cache),
            }
            checkpoint = {
                "format": CHECKPOINT_FORMAT,
                "model_format": CANDIDATE_MULTISCALE_PHASE_IDENTITY_LLR_FORMAT,
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
                    "render": False,
                    "image_retrieval_or_submap": False,
                    "visual_sources": list(CANDIDATE_MULTISCALE_PHASE_IDENTITY_SOURCES),
                    "support_view_marginalization": "fixed_mass_neutral_missing_view_v1",
                },
                "model_config": {
                    "hidden_dim": int(args.hidden_dim),
                    "max_abs_log_ratio": float(args.max_abs_log_ratio),
                    "source_storage_dtype": str(args.source_storage_dtype),
                    "source_configs": _serialize_source_configs(base_model),
                    "source_weights": dict(base_model.source_weights),
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
                    "source_image_manifest_sha256": str(
                        sources[0].metadata.get("source_image_manifest_sha256", "")
                    ),
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
            report = {
                "format": CHECKPOINT_FORMAT,
                "checkpoint": str(checkpoint_path),
                "selection_policy": FINAL_EPOCH_SELECTION_POLICY,
                "epoch_logs": epoch_logs,
                "gate": gate,
                "promotable_to_frozen_pose_rank_audit": bool(gate["passed"]),
                "next_step": (
                    "run_frozen_top20_pose_rank_audit"
                    if bool(gate["passed"])
                    else "diagnostic_only_do_not_run_pose_ranking_or_pnp"
                ),
            }
            _atomic_json_save(report, report_path)
            print(json.dumps({"phase_identity_gate": gate}, sort_keys=True), flush=True)
        return 0
    finally:
        _finalize_distributed(state)


if __name__ == "__main__":
    raise SystemExit(main())
