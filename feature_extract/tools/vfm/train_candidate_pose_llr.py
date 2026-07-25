"""Train a candidate-specific correct-versus-coherent-wrong pose LLR.

This is deliberately a train-only diagnostic stage.  The network consumes
only real-image RADIO/ALIKE crop pairs at candidate-projected locations.  GT
and coherent-wrong poses exist only in the pair target artifact and are used
to form the pairwise objective; they are never serialized into the checkpoint
or available to the target-free score program.

Use both local GPUs with::

    torchrun --standalone --nproc_per_node=2 feature_extract/tools/vfm/train_candidate_pose_llr.py ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
from torch.nn.parallel import DistributedDataParallel

from feature_extract.tools.vfm.build_candidate_pose_llr_train_pairs import TRAIN_PAIR_FORMAT
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _fixed_candidate_views,
    _load_bank_xyz,
    _load_maplet_support_fields,
    _resolve_rows,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_FORMAT,
    CandidatePoseLLRRuntime,
    CandidateSpecificPoseLLR,
    query_grouped_pose_margin_loss,
    validate_serialized_grouped_hypothesis_semantic_lineage,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    build_fixed_candidate_context_runtime,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    MixedVerificationPoints,
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    project_simple_radial_torch,
)


CHECKPOINT_FORMAT = "candidate_pose_llr_checkpoint_v1"


@dataclass(frozen=True)
class _DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


@dataclass(frozen=True)
class _TrainPairs:
    query_ids: np.ndarray
    correct_poses_w2c: np.ndarray
    coherent_wrong_poses_w2c: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _QueryRuntime:
    runtime: CandidatePoseLLRRuntime
    candidate_xyz: torch.Tensor
    focal_length: float
    principal_x: float
    principal_y: float
    radial_k: float
    image_width: int
    image_height: int


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
    parser.add_argument("--max-abs-log-ratio", type=float, default=3.0)
    parser.add_argument("--edge-chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--inner-validation-fold-count", type=int, default=5)
    parser.add_argument("--inner-validation-fold-index", type=int, default=0)
    parser.add_argument("--missing-edge-log-likelihood-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _initialize_distributed(device_name: str) -> _DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP candidate pose-LLR training requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("candidate pose-LLR training requested CUDA but it is unavailable")
    return _DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        enabled=enabled,
    )


def _finalize_distributed(state: _DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        distributed.barrier()
        distributed.destroy_process_group()


def _load_train_pairs(path: Path) -> _TrainPairs:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "query_ids",
            "split_names",
            "correct_poses_w2c",
            "coherent_wrong_poses_w2c",
            "metadata_json",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"candidate pose-LLR pairs lack {sorted(missing)}")
        query_ids = np.asarray(payload["query_ids"]).astype(str)
        splits = np.asarray(payload["split_names"]).astype(str)
        correct = np.asarray(payload["correct_poses_w2c"], dtype=np.float64)
        wrong = np.asarray(payload["coherent_wrong_poses_w2c"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("format") != TRAIN_PAIR_FORMAT
        or metadata.get("training_only_target_artifact") is not True
        or metadata.get("contains_validation_or_test_targets") is not False
        or metadata.get("pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer") is not True
        or len(query_ids) == 0
        or splits.shape != query_ids.shape
        or np.any(splits != "train")
        or correct.shape != (len(query_ids), 4, 4)
        or wrong.shape != correct.shape
        or not np.isfinite(correct).all()
        or not np.isfinite(wrong).all()
    ):
        raise ValueError("candidate pose-LLR pairs violate the train-only contract")
    lineage = metadata.get("hypothesis_semantic_lineage")
    try:
        validate_serialized_grouped_hypothesis_semantic_lineage(lineage)
    except ValueError as exc:
        raise ValueError("candidate pose-LLR pairs lack a valid hypothesis semantic lineage") from exc
    return _TrainPairs(
        query_ids=query_ids,
        correct_poses_w2c=correct,
        coherent_wrong_poses_w2c=wrong,
        metadata=dict(metadata),
    )


def _group_train_pairs_by_query(pairs: _TrainPairs) -> dict[str, np.ndarray]:
    """Collect every coherent wrong mode under one train query."""

    groups: dict[str, np.ndarray] = {}
    for query_id in sorted(set(np.asarray(pairs.query_ids).astype(str).tolist())):
        indices = np.flatnonzero(np.asarray(pairs.query_ids).astype(str) == query_id)
        if len(indices) == 0:
            raise RuntimeError("train-pair query grouping lost a row")
        correct = np.asarray(pairs.correct_poses_w2c[indices], dtype=np.float64)
        if not np.all(correct == correct[:1]):
            raise ValueError("candidate pose-LLR query has inconsistent correct poses")
        groups[query_id] = indices.astype(np.int64, copy=False)
    if not groups:
        raise ValueError("candidate pose-LLR train pairs have no query groups")
    return groups


def _partition_train_queries_for_inner_validation(
    *, query_ids: Sequence[str], fold_count: int, fold_index: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Make a stable train-only query fold without sampling individual modes."""

    ids = tuple(sorted(set(str(query_id) for query_id in query_ids)))
    count = int(fold_count)
    index = int(fold_index)
    if len(ids) < 2 or count < 2 or count > len(ids) or not 0 <= index < count:
        raise ValueError("candidate pose-LLR inner validation fold is invalid")
    validation = tuple(
        query_id
        for query_id in ids
        if int.from_bytes(
            hashlib.sha256(query_id.encode("utf-8")).digest()[:8], byteorder="big"
        )
        % count
        == index
    )
    train = tuple(query_id for query_id in ids if query_id not in set(validation))
    if not train or not validation:
        raise ValueError("candidate pose-LLR inner validation fold is empty")
    return train, validation


def _poses_for_query_group(*, pairs: _TrainPairs, pair_indices: np.ndarray) -> np.ndarray:
    """Place the repeated correct pose before every wrong mode for one query."""

    indices = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError("candidate pose-LLR query group has no wrong poses")
    correct = np.asarray(pairs.correct_poses_w2c[indices], dtype=np.float64)
    wrong = np.asarray(pairs.coherent_wrong_poses_w2c[indices], dtype=np.float64)
    if (
        correct.shape != (len(indices), 4, 4)
        or wrong.shape != correct.shape
        or not np.all(correct == correct[:1])
    ):
        raise ValueError("candidate pose-LLR query-group poses are invalid")
    return np.concatenate((correct[:1], wrong), axis=0)


def _select_fixed_support_views(
    *,
    support_image_ids: np.ndarray,
    support_view_valid: np.ndarray,
    support_view_weights: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the coverage-sorted maplet prefix before any pose is evaluated."""

    ids = np.asarray(support_image_ids).astype(str)
    valid = np.asarray(support_view_valid, dtype=bool)
    weights = np.asarray(support_view_weights, dtype=np.float32)
    view_count = int(count)
    if (
        ids.ndim != 3
        or valid.shape != ids.shape
        or weights.shape != ids.shape
        or view_count <= 0
        or view_count > ids.shape[2]
        or np.any(weights < 0.0)
    ):
        raise ValueError("fixed support-view prefix inputs are invalid")
    ids = ids[:, :, :view_count].copy()
    valid = valid[:, :, :view_count].copy()
    weights = np.where(valid, weights[:, :, :view_count], 0.0).astype(np.float32)
    mass = weights.sum(axis=2, keepdims=True)
    positive = mass[..., 0] > 0.0
    if not np.all(positive):
        raise ValueError("fixed support-view prefix dropped all evidence for a candidate")
    weights = weights / mass
    return ids, valid, weights


def _load_support_geometry_index(path: Path) -> object:
    """Return the geometry index while retaining loader metadata out of runtime state."""

    index, _metadata = load_support_observation_geometry_index_npz(Path(path))
    return index


def _prepare_query_runtimes(
    *,
    points: MixedVerificationPoints,
    sources: Sequence[object],
    maplet_support_index: Path,
    support_geometry_index: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    support_view_count: int,
    required_split: str,
) -> dict[str, _QueryRuntime]:
    """Build fixed candidate/support arrays once, then slice by one source split."""

    if str(required_split) not in {"train", "validation", "test"}:
        raise ValueError("candidate pose-LLR runtime split is invalid")

    source_by_name = {str(source.name): source for source in sources}
    if set(source_by_name) != {"radio_final", "radio_intermediate", "alike"}:
        raise ValueError("candidate pose-LLR source set is incomplete")
    reference = source_by_name["radio_final"]
    for source in source_by_name.values():
        if not np.array_equal(source.image_ids, reference.image_ids) or not np.array_equal(
            source.image_sizes, reference.image_sizes
        ):
            raise ValueError("candidate pose-LLR image cache ownership differs across scales")
    maplet_tracks, support_ids, support_indices, support_coverage, _maplet_metadata = (
        _load_maplet_support_fields(Path(maplet_support_index))
    )
    fixed_views = _fixed_candidate_views(
        candidate_track_ids=np.asarray(points.candidate_track_ids, dtype=np.int64),
        candidate_probabilities=np.asarray(points.candidate_prior_probabilities, dtype=np.float32),
        maplet_track_ids=maplet_tracks,
        support_image_ids=support_ids,
        support_image_indices=support_indices,
        support_coverage_counts=support_coverage,
    )
    fixed_ids, fixed_valid, fixed_weights = _select_fixed_support_views(
        support_image_ids=fixed_views.support_image_ids,
        support_view_valid=fixed_views.valid,
        support_view_weights=fixed_views.weights,
        count=int(support_view_count),
    )
    support_geometry = _load_support_geometry_index(Path(support_geometry_index))
    context_runtime = build_fixed_candidate_context_runtime(
        query_ids=np.asarray(points.query_ids).astype(str),
        query_xy=np.asarray(points.xy, dtype=np.float32),
        candidate_track_ids=np.asarray(points.candidate_track_ids, dtype=np.int64),
        candidate_support_image_ids=fixed_ids,
        candidate_view_valid=fixed_valid,
        cache_image_ids=np.asarray(reference.image_ids).astype(str),
        support_geometry=support_geometry,
    )
    bank_tracks, bank_xyz, _bank_metadata = _load_bank_xyz(Path(projected_landmark_bank))
    candidate_rows = _resolve_rows(
        np.asarray(points.candidate_track_ids, dtype=np.int64), canonical_track_ids=bank_tracks
    )
    if np.any(candidate_rows < 0):
        raise ValueError("candidate pose-LLR points contain a track absent from the projected bank")
    candidate_xyz = bank_xyz[candidate_rows].astype(np.float32)
    complete_runtime = CandidatePoseLLRRuntime(
        query_image_indices=torch.from_numpy(context_runtime.query_image_indices),
        support_image_indices=torch.from_numpy(context_runtime.support_image_indices),
        support_xy=torch.from_numpy(context_runtime.support_xy),
        support_view_valid=torch.from_numpy(context_runtime.view_valid),
        candidate_view_weights=torch.from_numpy(fixed_weights),
        candidate_probabilities=torch.from_numpy(
            np.asarray(points.candidate_prior_probabilities, dtype=np.float32)
        ),
        null_probabilities=torch.from_numpy(np.asarray(points.null_probabilities, dtype=np.float32)),
    )
    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    camera_ids_by_name = read_colmap_image_camera_ids_binary(
        Path(colmap_model_dir) / "images.bin"
    )
    runtimes: dict[str, _QueryRuntime] = {}
    for query_id in sorted(set(np.asarray(points.query_ids).astype(str).tolist())):
        rows = points.rows_for_query(query_id)
        if len(rows) == 0 or np.any(points.split_names[rows] != str(required_split)):
            continue
        camera_id = camera_ids_by_name.get(str(query_id))
        if camera_id is None:
            raise ValueError(f"candidate pose-LLR query is absent from Colmap: {query_id}")
        camera = cameras.get(int(camera_id))
        if camera is None or int(camera.model_id) != 2 or len(camera.params) != 4:
            raise ValueError("candidate pose-LLR requires SIMPLE_RADIAL query cameras")
        rows_tensor = torch.from_numpy(np.asarray(rows, dtype=np.int64))
        runtimes[str(query_id)] = _QueryRuntime(
            runtime=CandidatePoseLLRRuntime(
                query_image_indices=complete_runtime.query_image_indices.index_select(0, rows_tensor),
                support_image_indices=complete_runtime.support_image_indices.index_select(0, rows_tensor),
                support_xy=complete_runtime.support_xy.index_select(0, rows_tensor),
                support_view_valid=complete_runtime.support_view_valid.index_select(0, rows_tensor),
                candidate_view_weights=complete_runtime.candidate_view_weights.index_select(0, rows_tensor),
                candidate_probabilities=complete_runtime.candidate_probabilities.index_select(0, rows_tensor),
                null_probabilities=complete_runtime.null_probabilities.index_select(0, rows_tensor),
            ),
            candidate_xyz=torch.from_numpy(candidate_xyz[rows]),
            focal_length=float(camera.params[0]),
            principal_x=float(camera.params[1]),
            principal_y=float(camera.params[2]),
            radial_k=float(camera.params[3]),
            image_width=int(camera.width),
            image_height=int(camera.height),
        )
    if not runtimes:
        raise ValueError(f"candidate pose-LLR has no {required_split} query runtimes")
    return runtimes


def _project_candidate_positions(
    *, query: _QueryRuntime, poses_w2c: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = query.candidate_xyz.to(device=device, dtype=torch.float32)
    poses = torch.as_tensor(poses_w2c, dtype=torch.float32, device=device)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("candidate pose-LLR pose batch is invalid")
    projected, valid = project_simple_radial_torch(
        xyz.reshape(-1, 3),
        poses,
        focal_length=float(query.focal_length),
        principal_x=float(query.principal_x),
        principal_y=float(query.principal_y),
        radial_k=float(query.radial_k),
        image_width=int(query.image_width),
        image_height=int(query.image_height),
    )
    return projected.reshape(len(poses), *xyz.shape[:2], 2), valid.reshape(
        len(poses), *xyz.shape[:2]
    )


def _source_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def _reduce_statistics(state: _DistributedState, values: torch.Tensor) -> torch.Tensor:
    result = values.detach().clone()
    if state.enabled:
        distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result


@torch.no_grad()
def _evaluate_inner_validation_query_groups(
    *,
    model: CandidateSpecificPoseLLR,
    query_runtimes: Mapping[str, _QueryRuntime],
    pairs: _TrainPairs,
    groups: Mapping[str, np.ndarray],
    query_ids: Sequence[str],
    device: torch.device,
    pairwise_margin: float,
    missing_edge_log_likelihood_ratio: float,
    amp_enabled: bool,
) -> dict[str, float]:
    """Evaluate the strongest coherent wrong pose once per train query."""

    if not query_ids:
        raise ValueError("candidate pose-LLR inner validation has no queries")
    model.eval()
    total_loss = 0.0
    total_gap = 0.0
    total_wins = 0.0
    for query_id in query_ids:
        pair_indices = groups.get(str(query_id))
        query = query_runtimes.get(str(query_id))
        if pair_indices is None or query is None:
            raise ValueError("candidate pose-LLR inner validation query is unresolved")
        poses = torch.from_numpy(
            _poses_for_query_group(pairs=pairs, pair_indices=pair_indices)
        )
        projected_xy, projected_valid = _project_candidate_positions(
            query=query, poses_w2c=poses, device=device
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pose_scores = model(
                query.runtime,
                projected_xy,
                projected_valid,
                float(missing_edge_log_likelihood_ratio),
            )
            loss, metrics = query_grouped_pose_margin_loss(
                correct_scores=pose_scores[:1],
                coherent_wrong_scores=pose_scores[1:].reshape(1, -1),
                margin=float(pairwise_margin),
            )
        total_loss += float(loss.item())
        total_gap += float(metrics["query_mean_correct_minus_hardest_wrong"])
        total_wins += float(metrics["query_correct_win_fraction"])
    count = float(len(query_ids))
    return {
        "query_grouped_loss": total_loss / count,
        "mean_correct_minus_hardest_wrong": total_gap / count,
        "correct_win_fraction": total_wins / count,
        "query_count": count,
    }


def _is_better_inner_validation_epoch(
    *, candidate: Mapping[str, float], incumbent: Mapping[str, float] | None
) -> bool:
    """Prefer a lower query-grouped validation loss, then more query wins."""

    candidate_loss = float(candidate["query_grouped_loss"])
    candidate_wins = float(candidate["correct_win_fraction"])
    if not np.isfinite(candidate_loss) or not np.isfinite(candidate_wins):
        raise ValueError("candidate pose-LLR inner validation metrics are invalid")
    if incumbent is None:
        return True
    incumbent_loss = float(incumbent["query_grouped_loss"])
    incumbent_wins = float(incumbent["correct_win_fraction"])
    if not np.isfinite(incumbent_loss) or not np.isfinite(incumbent_wins):
        raise ValueError("candidate pose-LLR incumbent validation metrics are invalid")
    if candidate_loss < incumbent_loss - 1e-12:
        return True
    return abs(candidate_loss - incumbent_loss) <= 1e-12 and candidate_wins > incumbent_wins


def train_candidate_pose_llr(args: argparse.Namespace) -> dict[str, object]:
    """Run a DDP-safe train-only pairwise LLR fit and write a checkpoint."""

    state = _initialize_distributed(str(args.device))
    try:
        if int(args.epochs) <= 0 or float(args.learning_rate) <= 0.0:
            raise ValueError("candidate pose-LLR epochs and learning rate must be positive")
        if int(args.support_view_count) <= 0 or int(args.edge_chunk_size) <= 0:
            raise ValueError("candidate pose-LLR fixed view count and chunk size must be positive")
        if not np.isfinite(float(args.missing_edge_log_likelihood_ratio)):
            raise ValueError("candidate pose-LLR missing edge score must be finite")
        output_dir = Path(args.output_dir)
        checkpoint_path = output_dir / "candidate_pose_llr.pt"
        history_path = output_dir / "history.json"
        if state.rank == 0 and (checkpoint_path.exists() or history_path.exists()) and not bool(args.force):
            raise FileExistsError("refusing to overwrite candidate pose-LLR checkpoint output")
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
        except AttributeError:  # pragma: no cover - older torch
            pass

        points_path = Path(args.verification_points)
        points = load_mixed_verification_points(points_path)
        if (
            points.metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
            or points.metadata.get("contains_ground_truth") is not False
            or points.metadata.get("pose_or_ground_truth_used") is not False
            or points.metadata.get("render") is not False
        ):
            raise ValueError("candidate pose-LLR verification points violate the real-image contract")
        pairs_path = Path(args.train_pairs)
        pairs = _load_train_pairs(pairs_path)
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
        missing_queries = sorted(set(pairs.query_ids.tolist()).difference(query_runtimes))
        if missing_queries:
            raise ValueError(f"candidate pose-LLR pair queries lack train runtimes: {missing_queries[:5]}")
        query_groups = _group_train_pairs_by_query(pairs)
        inner_train_query_ids, inner_validation_query_ids = (
            _partition_train_queries_for_inner_validation(
                query_ids=tuple(query_groups),
                fold_count=int(args.inner_validation_fold_count),
                fold_index=int(args.inner_validation_fold_index),
            )
        )
        model = CandidateSpecificPoseLLR(
            sources={source.name: torch.from_numpy(np.asarray(source.grid)) for source in sources},
            image_sizes=torch.from_numpy(np.asarray(sources[0].image_sizes, dtype=np.float32)),
            hidden_dim=int(args.hidden_dim),
            max_abs_log_ratio=float(args.max_abs_log_ratio),
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
        assert isinstance(core_model, CandidateSpecificPoseLLR)
        optimizer = torch.optim.AdamW(
            model_for_train.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        pair_count = len(pairs.query_ids)
        steps_per_rank = int(math.ceil(len(inner_train_query_ids) / state.world_size))
        history: list[dict[str, float]] = []
        best_inner_validation: dict[str, float] | None = None
        best_epoch = -1
        best_state_dict: dict[str, torch.Tensor] | None = None
        start_time = time.time()
        for epoch in range(int(args.epochs)):
            model_for_train.train()
            order = np.random.default_rng(int(args.seed) + epoch).permutation(
                len(inner_train_query_ids)
            )
            local_query_ids = tuple(
                inner_train_query_ids[
                    int(order[(state.rank + step * state.world_size) % len(order)])
                ]
                for step in range(steps_per_rank)
            )
            totals = torch.zeros((4,), dtype=torch.float64, device=state.device)
            for query_id in local_query_ids:
                query = query_runtimes[str(query_id)]
                poses = torch.from_numpy(
                    _poses_for_query_group(
                        pairs=pairs, pair_indices=query_groups[str(query_id)]
                    )
                )
                projected_xy, projected_valid = _project_candidate_positions(
                    query=query, poses_w2c=poses, device=state.device
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pose_scores = model_for_train(
                        query.runtime,
                        projected_xy,
                        projected_valid,
                        float(args.missing_edge_log_likelihood_ratio),
                    )
                    loss, metrics = query_grouped_pose_margin_loss(
                        correct_scores=pose_scores[:1],
                        coherent_wrong_scores=pose_scores[1:].reshape(1, -1),
                        margin=float(args.pairwise_margin),
                    )
                scaler.scale(loss).backward()
                if float(args.gradient_clip_norm) > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model_for_train.parameters(), float(args.gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
                totals += torch.tensor(
                    [
                        float(loss.detach().item()),
                        float(metrics["query_mean_correct_minus_hardest_wrong"]),
                        float(metrics["query_correct_win_fraction"]),
                        1.0,
                    ],
                    dtype=torch.float64,
                    device=state.device,
                )
            totals = _reduce_statistics(state, totals)
            epoch_metrics = {
                "epoch": float(epoch + 1),
                "query_grouped_loss": float((totals[0] / totals[3]).item()),
                "mean_correct_minus_hardest_wrong": float((totals[1] / totals[3]).item()),
                "correct_win_fraction": float((totals[2] / totals[3]).item()),
                "global_query_steps": float(totals[3].item()),
            }
            if state.enabled:
                distributed.barrier()
            if state.rank == 0:
                inner_validation = _evaluate_inner_validation_query_groups(
                    model=core_model,
                    query_runtimes=query_runtimes,
                    pairs=pairs,
                    groups=query_groups,
                    query_ids=inner_validation_query_ids,
                    device=state.device,
                    pairwise_margin=float(args.pairwise_margin),
                    missing_edge_log_likelihood_ratio=float(
                        args.missing_edge_log_likelihood_ratio
                    ),
                    amp_enabled=amp_enabled,
                )
                epoch_metrics.update(
                    {
                        f"inner_validation_{name}": float(value)
                        for name, value in inner_validation.items()
                    }
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
            output_dir.mkdir(parents=True, exist_ok=True)
            if best_state_dict is None or best_inner_validation is None or best_epoch < 1:
                raise RuntimeError("candidate pose-LLR did not select an inner validation checkpoint")
            input_paths = {
                "verification_points": points_path,
                "train_pairs": pairs_path,
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
                "model_format": CANDIDATE_POSE_LLR_FORMAT,
                "architecture": "candidate_specific_mask_aware_subpixel_full_2d_radio_context_plus_alike_shift_correlation_v3",
                "contains_target_fields": False,
                "checkpoint_contains_train_targets": False,
                "training_target_source_is_train_only_pair_artifact": True,
                "hypothesis_semantic_lineage": pairs.metadata[
                    "hypothesis_semantic_lineage"
                ],
                "candidate_pose_llr_is_independent_pose_likelihood": False,
                "diagnostic_only": True,
                "promotion_allowed": False,
                "raw_scores_must_not_feed_pnp": True,
                "render": False,
                "image_retrieval_or_submap_used": False,
                "fixed_global_topl": True,
                "fixed_candidate_top_k": int(points.candidate_track_ids.shape[1]),
                "fixed_support_view_count": int(args.support_view_count),
                "candidate_reselection_per_pose": False,
                "support_reselection_per_pose": False,
                "explicit_null": True,
                "missing_edge_semantics": "fixed_neutral_log_likelihood_ratio",
                "partial_crop_semantics": "subpixel_bilinear_mask_real_tokens_keep_valid_projection_edges_v2",
                "missing_edge_log_likelihood_ratio": float(args.missing_edge_log_likelihood_ratio),
                "encoder_inputs": [
                    "candidate_projected_query_crop_radio_final_9x9",
                    "candidate_projected_query_crop_radio_intermediate_13x13",
                    "candidate_projected_query_crop_alike_13x13",
                    "fixed_support_observation_crops",
                    "per-token_query_and_support_crop_validity_masks",
                    "alike_3x3_shift_correlation",
                ],
                "encoder_excludes": [
                    "pose_matrix",
                    "reprojection_residual",
                    "ground_truth_label",
                    "track_id",
                    "coarse_posterior",
                ],
                "hidden_dim": int(args.hidden_dim),
                "max_abs_log_ratio": float(args.max_abs_log_ratio),
                "edge_chunk_size": int(args.edge_chunk_size),
                "activation_checkpointing_during_training": True,
                "training": {
                    "objective": "query_grouped_strongest_coherent_wrong_margin",
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "pairwise_margin": float(args.pairwise_margin),
                    "world_size": int(state.world_size),
                    "seed": int(args.seed),
                    "inner_validation": {
                        "split": "train_query_only",
                        "fold_count": int(args.inner_validation_fold_count),
                        "fold_index": int(args.inner_validation_fold_index),
                        "selected_epoch": int(best_epoch),
                        "selected_metrics": best_inner_validation,
                    },
                },
                "inputs": _source_manifest(input_paths),
                "train_pair_count": int(pair_count),
                "train_query_count": int(len(query_groups)),
                "inner_train_query_count": int(len(inner_train_query_ids)),
                "inner_validation_query_count": int(len(inner_validation_query_ids)),
            }
            torch.save(
                {"format": CHECKPOINT_FORMAT, "state_dict": best_state_dict, "metadata": metadata},
                checkpoint_path,
            )
            summary = {
                "stage": "train_candidate_pose_llr",
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
                    "pnp_integration_forbidden_until_heldout_audit_gate": True,
                },
            }
            history_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if state.enabled:
            distributed.barrier()
        return (
            json.loads(history_path.read_text())
            if state.rank == 0
            else {"stage": "train_candidate_pose_llr", "rank": state.rank}
        )
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = train_candidate_pose_llr(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
