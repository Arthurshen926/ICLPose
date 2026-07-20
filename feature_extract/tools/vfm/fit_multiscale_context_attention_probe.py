"""Fit a train-only multiscale candidate-context cross-attention probe.

The input contract is frozen before this command starts.  It fixes query
tokens, top-L landmark candidates, support observations, and all real-image
descriptor grids.  Only train-image SfM observation identities are joined for
the loss; validation/test labels are never loaded by this command.

Launch with ``torchrun --standalone --nproc_per_node=2`` to use both local
GPUs.  DDP is used for fitting, while frozen inference is sharded across the
same ranks and reduced before a single target-free prediction artifact is
written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from feature_extract.tools.vfm.build_global_context_candidate_probe_features import (
    _array_sha256_short,
)
from feature_extract.tools.vfm.fit_multiscale_candidate_probe import (
    _load_base_overlay,
    _load_proposal_tracks,
    _replace_overlay_rows,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES,
    ANCHOR_RELATIVE_POSITION_ENCODING,
    CONTEXT_ATTENTION_FAMILIES,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
    CONTEXT_ATTENTION_SCALES,
    CandidateContextAttentionProbe,
    build_fixed_candidate_context_runtime,
    load_context_attention_frozen_layout,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_candidate_identity_target_membership,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


MODEL_FORMAT = "multiscale_context_attention_candidate_probe_v1"
PREDICTION_ARTIFACT_FORMAT = "multiscale_candidate_probe_predictions_v2"
OVERLAY_ARTIFACT_FORMAT = "multiscale_candidate_probe_prior_overlay_v2"
SUPERVISION_MODE = "registered_track_identity"
PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
TRAINING_OBJECTIVE = "registered_query_observation_exact_track_or_explicit_null_nll_v1"
FAMILY_PROFILES: dict[str, tuple[tuple[str, ...], str]] = {
    "relative_context_v1": (
        CONTEXT_ATTENTION_FAMILIES,
        ANCHOR_RELATIVE_POSITION_ENCODING,
    ),
    "absolute_phase_v1": (
        ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
}

_PROPOSAL_OVERLAY_CANDIDATE_INPUT = "proposal_overlay_v1"
_MIXED_POINTS_CANDIDATE_INPUT = "mixed_verification_points_embedded_coarse_prior_v1"


@dataclass(frozen=True)
class _DistributedState:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool


@dataclass(frozen=True)
class _CandidatePriorInput:
    """A fully validated target-free candidate table and its fixed prior."""

    kind: str
    source_path: Path
    source_sha256: str
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    metadata: Mapping[str, Any]
    base_prior_path: Path | None
    base_prior_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument("--proposals")
    candidate_source.add_argument("--verification_points")
    parser.add_argument(
        "--base_prior_overlay",
        default=None,
        help="required only with --proposals; mixed verification points carry their fixed prior",
    )
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--family_profile",
        choices=tuple(FAMILY_PROFILES),
        default="relative_context_v1",
        help="selects a frozen attribution family set and its position encoding",
    )
    parser.add_argument(
        "--families",
        default=None,
        help="optional exact spelling of the predeclared family profile",
    )
    parser.add_argument("--registered_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16, help="per-GPU row batch")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _metadata_file(path: Path, *, context: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{context} is not a JSON object")
    return value


def _family_profile(profile: str) -> tuple[tuple[str, ...], str]:
    value = FAMILY_PROFILES.get(str(profile))
    if value is None:
        raise ValueError("unsupported context-attention family profile")
    return value


def _parse_families(
    value: str | None, *, family_profile: str = "relative_context_v1"
) -> tuple[str, ...]:
    expected, _position_encoding = _family_profile(family_profile)
    if value is None:
        return expected
    families = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if set(families) != set(expected) or len(families) != len(expected):
        raise ValueError(
            "context-attention fitting requires every predeclared attribution family"
        )
    # Stable ordering makes output independent of CLI spelling.
    return expected


def _initialize_distributed(device_name: str) -> _DistributedState:
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 1:
            device = torch.device(str(device_name))
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"requested CUDA device is unavailable: {device}")
            return _DistributedState(
                rank=0, world_size=1, local_rank=0, device=device, enabled=False
            )
        if not torch.cuda.is_available():
            raise RuntimeError("distributed context-attention fitting requires CUDA")
        torch.cuda.set_device(local_rank)
        distributed.init_process_group(backend="nccl")
        return _DistributedState(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=torch.device(f"cuda:{local_rank}"),
            enabled=True,
        )
    device = torch.device(str(device_name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    return _DistributedState(rank=0, world_size=1, local_rank=0, device=device, enabled=False)


def _close_distributed(state: _DistributedState) -> None:
    if state.enabled and distributed.is_initialized():
        distributed.destroy_process_group()


def _broadcast_success_or_raise(state: _DistributedState, success: bool) -> None:
    if not state.enabled:
        if not success:
            raise FileExistsError("context-attention output directory already exists")
        return
    flag = torch.tensor([1 if success else 0], device=state.device, dtype=torch.int64)
    distributed.broadcast(flag, src=0)
    if not bool(flag.item()):
        raise FileExistsError("context-attention output directory already exists")


def _stable_family_seed(seed: int, family: str) -> int:
    digest = hashlib.sha256(str(family).encode("utf8")).digest()
    return int((int(seed) + int.from_bytes(digest[:4], "little")) % (2**31 - 1))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _contract_paths(contract: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Path]]:
    layout = Path(str(contract.get("frozen_layout_features", "")))
    geometry = Path(str(contract.get("support_geometry_index", "")))
    raw_sources = contract.get("source_scales")
    if not layout.is_file() or not geometry.is_file() or not isinstance(raw_sources, list):
        raise ValueError("context-attention contract source paths are invalid")
    sources: dict[str, Path] = {}
    for value, scale in zip(raw_sources, CONTEXT_ATTENTION_SCALES):
        if not isinstance(value, Mapping) or str(value.get("name")) != scale.name:
            raise ValueError("context-attention contract source scale order differs")
        path = Path(str(value.get("path", "")))
        if not path.is_file() or str(value.get("sha256", "")) != file_sha256_short(path):
            raise ValueError("context-attention contract source cache is stale")
        if int(value.get("grid_size", -1)) != int(scale.grid_size) or int(
            value.get("window_size", -1)
        ) != int(scale.window_size):
            raise ValueError("context-attention contract source scale configuration differs")
        sources[scale.name] = path
    if len(raw_sources) != len(CONTEXT_ATTENTION_SCALES):
        raise ValueError("context-attention contract has extra source scales")
    return layout, geometry, sources


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _metadata_file(Path(path), context="context-attention contract")
    if contract.get("format") != CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT:
        raise ValueError("unsupported context-attention contract format")
    for field in (
        "contains_ground_truth",
        "contains_target_errors",
        "pose_or_ground_truth_used",
        "image_retrieval_or_submap_used",
        "whole_image_summary_or_global_used",
        "render",
    ):
        if contract.get(field) is not False:
            raise ValueError(f"context-attention contract violates target-free protocol: {field}")
    if tuple(contract.get("families", ())) != CONTEXT_ATTENTION_FAMILIES:
        raise ValueError("context-attention contract family protocol differs")
    layout, geometry, _sources = _contract_paths(contract)
    if str(contract.get("frozen_layout_features_sha256", "")) != file_sha256_short(layout):
        raise ValueError("context-attention contract frozen layout is stale")
    if str(contract.get("support_geometry_index_sha256", "")) != file_sha256_short(geometry):
        raise ValueError("context-attention contract support geometry is stale")
    candidate_input_kind = str(
        contract.get("candidate_input_kind", _PROPOSAL_OVERLAY_CANDIDATE_INPUT)
    )
    if candidate_input_kind not in {
        _PROPOSAL_OVERLAY_CANDIDATE_INPUT,
        _MIXED_POINTS_CANDIDATE_INPUT,
    }:
        raise ValueError("context-attention contract candidate-input kind is unsupported")
    if candidate_input_kind == _MIXED_POINTS_CANDIDATE_INPUT and (
        not str(contract.get("candidate_input_lineage_path", ""))
        or not str(contract.get("candidate_input_lineage_sha256", ""))
    ):
        raise ValueError("mixed context-attention contract lacks candidate-input lineage")
    return contract


def _runtime_hash(runtime: object) -> str:
    digest = hashlib.sha256()
    for value in (
        runtime.query_image_indices,
        runtime.support_image_indices,
        runtime.support_xy,
        runtime.view_valid,
    ):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()[:16]


def _load_runtime(
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[object, ...], object]:
    layout_path, geometry_path, paths = _contract_paths(contract)
    layout, layout_metadata = load_context_attention_frozen_layout(layout_path)
    if str(layout_metadata.get("proposals_sha256", "")) != str(contract.get("proposals_sha256", "")):
        raise ValueError("context-attention contract proposal lineage differs from frozen layout")
    if str(contract.get("full_frozen_source_rows_sha256", "")) != _array_sha256_short(
        np.asarray(layout["source_row_indices"], dtype=np.int64)
    ) or str(contract.get("frozen_candidate_tracks_sha256", "")) != _array_sha256_short(
        np.asarray(layout["candidate_track_ids"], dtype=np.int64)
    ) or str(contract.get("frozen_support_view_mask_sha256", "")) != _array_sha256_short(
        np.asarray(layout["candidate_view_valid"], dtype=bool)
    ):
        raise ValueError("context-attention frozen layout arrays are stale")
    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final"],
        radio_intermediate_context_cache=paths["radio_intermediate"],
        alike_spatial_context_cache=paths["alike"],
        expected_radio_checkpoint=str(contract.get("radio_checkpoint_sha256", "")),
    )
    geometry, geometry_metadata = load_support_observation_geometry_index_npz(geometry_path)
    if geometry_metadata.get("coordinate_source") != "sfm_observation_xy":
        raise ValueError("context-attention contract geometry source differs")
    runtime = build_fixed_candidate_context_runtime(
        query_ids=np.asarray(layout["query_ids"]).astype(str),
        query_xy=np.asarray(layout["xy"], dtype=np.float32),
        candidate_track_ids=np.asarray(layout["candidate_track_ids"], dtype=np.int64),
        candidate_support_image_ids=np.asarray(layout["candidate_support_image_ids"]).astype(str),
        candidate_view_valid=np.asarray(layout["candidate_view_valid"], dtype=bool),
        cache_image_ids=sources[0].image_ids,
        support_geometry=geometry,
    )
    if _runtime_hash(runtime) != str(contract.get("runtime_indices_sha256", "")):
        raise ValueError("context-attention runtime support observations are stale")
    return layout, layout_metadata, sources, runtime


def _order_dense_mixed_point_rows(
    *,
    source_point_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index mixed-point rows by their explicit immutable source IDs.

    The context layout stores ``source_row_indices`` rather than incidental
    NPZ order.  Refuse sparse IDs instead of treating an arbitrary row order as
    proposal-table indices.
    """

    point_ids = np.asarray(source_point_ids, dtype=np.int64).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    expected = np.arange(len(point_ids), dtype=np.int64)
    if (
        len(point_ids) == 0
        or tracks.shape[0] != len(point_ids)
        or probabilities.shape != tracks.shape
        or null.shape != (len(point_ids),)
        or not np.array_equal(np.sort(point_ids), expected)
    ):
        raise ValueError("mixed verification-point IDs are not a dense candidate-row domain")
    ordered_tracks = np.empty_like(tracks)
    ordered_probabilities = np.empty_like(probabilities)
    ordered_null = np.empty_like(null)
    ordered_tracks[point_ids] = tracks
    ordered_probabilities[point_ids] = probabilities
    ordered_null[point_ids] = null
    return ordered_tracks, ordered_probabilities, ordered_null


def _load_candidate_prior_input(
    *,
    contract: Mapping[str, Any],
    proposals_path: Path | None,
    base_overlay_path: Path | None,
    verification_points_path: Path | None,
) -> _CandidatePriorInput:
    """Load exactly the candidate source declared by the frozen layout."""

    kind = str(contract.get("candidate_input_kind", _PROPOSAL_OVERLAY_CANDIDATE_INPUT))
    expected_sha = str(
        contract.get("candidate_input_lineage_sha256", contract.get("proposals_sha256", ""))
    )
    if kind == _MIXED_POINTS_CANDIDATE_INPUT:
        if proposals_path is not None or base_overlay_path is not None or verification_points_path is None:
            raise ValueError(
                "mixed context-attention contracts require --verification_points only"
            )
        source = Path(verification_points_path)
        actual_sha = file_sha256_short(source)
        if actual_sha != expected_sha or actual_sha != str(contract.get("proposals_sha256", "")):
            raise ValueError("mixed verification points differ from frozen candidate lineage")
        points = load_mixed_verification_points(source)
        tracks, probabilities, null = _order_dense_mixed_point_rows(
            source_point_ids=points.source_point_ids,
            candidate_track_ids=points.candidate_track_ids,
            candidate_probabilities=points.candidate_prior_probabilities,
            null_probabilities=points.null_probabilities,
        )
        return _CandidatePriorInput(
            kind=kind,
            source_path=source,
            source_sha256=actual_sha,
            candidate_track_ids=tracks,
            candidate_probabilities=probabilities,
            null_probabilities=null,
            metadata=dict(points.metadata),
            base_prior_path=None,
            base_prior_sha256=actual_sha,
        )
    if kind != _PROPOSAL_OVERLAY_CANDIDATE_INPUT:
        raise ValueError("context-attention candidate-input kind is unsupported")
    if proposals_path is None or base_overlay_path is None or verification_points_path is not None:
        raise ValueError("proposal contracts require --proposals and --base_prior_overlay only")
    source = Path(proposals_path)
    actual_sha = file_sha256_short(source)
    if actual_sha != expected_sha or actual_sha != str(contract.get("proposals_sha256", "")):
        raise ValueError("proposal input differs from frozen candidate lineage")
    tracks = _load_proposal_tracks(source)
    base, metadata = _load_base_overlay(
        Path(base_overlay_path), proposal_tracks=tracks, proposals_path=source
    )
    return _CandidatePriorInput(
        kind=kind,
        source_path=source,
        source_sha256=actual_sha,
        candidate_track_ids=tracks,
        candidate_probabilities=np.asarray(base["candidate_probabilities"], dtype=np.float32),
        null_probabilities=np.asarray(base["null_probabilities"], dtype=np.float32),
        metadata=metadata,
        base_prior_path=Path(base_overlay_path),
        base_prior_sha256=file_sha256_short(Path(base_overlay_path)),
    )


def _train_identity_targets(
    *,
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    colmap_model_dir: Path,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if float(radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    images_path = Path(colmap_model_dir) / "images.bin"
    if not images_path.is_file():
        raise FileNotFoundError(f"registered identity model lacks {images_path}")
    images = read_colmap_images_binary(images_path)
    by_name = {str(image.image_name): image for image in images.values()}
    train = np.flatnonzero(np.asarray(split_names).astype(str) == "train")
    if train.size == 0:
        raise ValueError("context-attention layout has no train rows")
    targets = registered_query_observation_targets(
        query_ids=np.asarray(query_ids).astype(str)[train],
        query_xy=np.asarray(query_xy, dtype=np.float32)[train],
        images_by_name=by_name,
        max_distance_px=float(radius_px),
    )
    labels = registered_candidate_identity_labels(candidate_tracks[train], targets)
    membership = registered_candidate_identity_target_membership(candidate_tracks[train], targets)
    supervised = np.asarray(targets.supervised, dtype=bool)
    if not np.any(supervised):
        raise ValueError("registered identity target found no train observations")
    selected_rows = train[supervised]
    selected_membership = membership[supervised]
    if np.any(np.sum(selected_membership, axis=1) != 1):
        raise RuntimeError("registered identity target is not singleton-or-null")
    return selected_rows, np.argmax(selected_membership, axis=1).astype(np.int64), {
        "supervision": "registered_query_observation_exact_track_or_explicit_null_v1",
        "training_objective": TRAINING_OBJECTIVE,
        "registered_identity_radius_px": float(radius_px),
        "train_split_row_count": int(len(train)),
        "registered_supervised_train_row_count": int(np.sum(supervised)),
        "registered_supervised_train_row_rate": float(np.mean(supervised)),
        "unsupervised_train_row_count": int(np.sum(~supervised)),
        "exact_track_retrieved_train_row_count": int(np.sum(supervised & np.any(labels, axis=1))),
        "exact_track_retrieved_given_registered_train_rate": float(
            np.mean(np.any(labels, axis=1)[supervised])
        ),
        "explicit_null_registered_train_row_count": int(
            np.sum(supervised & ~np.any(labels, axis=1))
        ),
        "positive_candidate_train_count": int(np.sum(labels)),
        "registered_identity_target_coverage": summarize_registered_candidate_identity(
            labels, targets
        ),
        "colmap_images_sha256": file_sha256_short(images_path),
    }


def _sharded_epoch_rows(
    rows: np.ndarray, *, epoch: int, seed: int, state: _DistributedState
) -> np.ndarray:
    generator = np.random.default_rng(int(seed) + int(epoch) * 1009)
    order = np.asarray(rows, dtype=np.int64)[generator.permutation(len(rows))]
    if state.world_size == 1:
        return order
    per_rank = int(np.ceil(len(order) / float(state.world_size)))
    padded = np.resize(order, per_rank * state.world_size)
    return padded[state.rank :: state.world_size]


def _all_reduce_loss(value: float, count: int, state: _DistributedState) -> tuple[float, int]:
    if not state.enabled:
        return float(value), int(count)
    reduced = torch.tensor([float(value), float(count)], device=state.device)
    distributed.all_reduce(reduced, op=distributed.ReduceOp.SUM)
    return float(reduced[0].item()), int(round(float(reduced[1].item())))


def _fit_one_family(
    *,
    family: str,
    static_sources: Mapping[str, torch.Tensor],
    image_sizes: np.ndarray,
    runtime: object,
    query_xy: np.ndarray,
    base_candidate: np.ndarray,
    base_null: np.ndarray,
    train_rows: np.ndarray,
    train_classes: np.ndarray,
    state: _DistributedState,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    heads: int,
    dropout: float,
    seed: int,
    position_encoding: str,
) -> tuple[CandidateContextAttentionProbe, list[dict[str, float]]]:
    if len(train_rows) != len(train_classes):
        raise ValueError("context-attention train rows and targets differ")
    _set_seed(_stable_family_seed(seed, family))
    model = CandidateContextAttentionProbe(
        family=family,
        sources=static_sources,
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        runtime=runtime,
        query_xy=np.asarray(query_xy, dtype=np.float32),
        base_candidate_probabilities=np.asarray(base_candidate, dtype=np.float32),
        base_null_probabilities=np.asarray(base_null, dtype=np.float32),
        hidden_dim=int(hidden_dim),
        heads=int(heads),
        dropout=float(dropout),
        position_encoding=str(position_encoding),
    ).to(state.device)
    train_model: nn.Module
    if state.enabled:
        train_model = DistributedDataParallel(
            model,
            device_ids=[state.local_rank],
            output_device=state.local_rank,
            broadcast_buffers=False,
        )
    else:
        train_model = model
    optimizer = torch.optim.AdamW(
        train_model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=state.device.type == "cuda")
    class_lookup = {int(row): int(target) for row, target in zip(train_rows, train_classes)}
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        train_model.train()
        local_loss = 0.0
        local_count = 0
        epoch_rows = _sharded_epoch_rows(
            train_rows, epoch=epoch, seed=_stable_family_seed(seed, family), state=state
        )
        for begin in range(0, len(epoch_rows), int(batch_size)):
            batch_rows_np = epoch_rows[begin : begin + int(batch_size)]
            batch_targets_np = np.asarray(
                [class_lookup[int(row)] for row in batch_rows_np], dtype=np.int64
            )
            rows = torch.from_numpy(batch_rows_np).to(device=state.device, dtype=torch.long)
            targets = torch.from_numpy(batch_targets_np).to(device=state.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=state.device.type,
                dtype=torch.float16,
                enabled=state.device.type == "cuda",
            ):
                _candidate, _null, _view, logits = train_model(rows)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            local_loss += float(loss.detach().item()) * len(batch_rows_np)
            local_count += int(len(batch_rows_np))
        total_loss, total_count = _all_reduce_loss(local_loss, local_count, state)
        mean_loss = total_loss / max(total_count, 1)
        history.append({"epoch": float(epoch + 1), "train_nll": float(mean_loss)})
        if state.rank == 0 and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == int(epochs)):
            print(
                json.dumps(
                    {
                        "stage": "multiscale_context_attention_fit",
                        "family": family,
                        "epoch": int(epoch + 1),
                        "epochs": int(epochs),
                        "train_nll": float(mean_loss),
                        "world_size": int(state.world_size),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if state.enabled:
        distributed.barrier()
    return model, history


def _predict_sharded(
    *, model: CandidateContextAttentionProbe, state: _DistributedState, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    row_count = model.row_count
    candidate_count = int(model._support_image_indices.shape[1])
    view_count = int(model._support_image_indices.shape[2])
    candidate = torch.zeros((row_count, candidate_count), dtype=torch.float32, device=state.device)
    null = torch.zeros((row_count,), dtype=torch.float32, device=state.device)
    views = torch.zeros(
        (row_count, candidate_count, view_count), dtype=torch.float32, device=state.device
    )
    rows = np.arange(state.rank, row_count, state.world_size, dtype=np.int64)
    with torch.no_grad():
        for begin in range(0, len(rows), int(batch_size)):
            selected = torch.from_numpy(rows[begin : begin + int(batch_size)]).to(
                device=state.device, dtype=torch.long
            )
            with torch.autocast(
                device_type=state.device.type,
                dtype=torch.float16,
                enabled=state.device.type == "cuda",
            ):
                probability, null_probability, view_logits, _logits = model(selected)
            candidate[selected] = probability.to(dtype=torch.float32)
            null[selected] = null_probability.to(dtype=torch.float32)
            views[selected] = view_logits.to(dtype=torch.float32)
    if state.enabled:
        for values in (candidate, null, views):
            distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
    output = (
        candidate.detach().cpu().numpy(),
        null.detach().cpu().numpy(),
        views.detach().cpu().numpy(),
    )
    if not np.allclose(output[0].sum(axis=1) + output[1], 1.0, atol=1e-5):
        raise RuntimeError("context-attention prediction does not conserve probability mass")
    return output


def _save_model(
    *,
    path: Path,
    model: CandidateContextAttentionProbe,
    family: str,
    metadata: Mapping[str, Any],
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format": MODEL_FORMAT,
            "family": str(family),
            "state_dict": state,
            "metadata": dict(metadata),
        },
        path,
    )


def fit_multiscale_context_attention_probe(
    *,
    contract_path: Path,
    proposals_path: Path | None,
    base_overlay_path: Path | None,
    verification_points_path: Path | None,
    colmap_model_dir: Path,
    output_dir: Path,
    families: Sequence[str],
    registered_identity_radius_px: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    heads: int,
    dropout: float,
    seed: int,
    device: str,
    family_profile: str = "relative_context_v1",
) -> dict[str, Any] | None:
    if (
        float(registered_identity_radius_px) <= 0.0
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or int(hidden_dim) <= 0
        or int(heads) <= 0
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError("context-attention optimization arguments are invalid")
    expected_families, position_encoding = _family_profile(family_profile)
    if tuple(families) != expected_families:
        raise ValueError("context-attention family set differs from the frozen paired protocol")
    state = _initialize_distributed(str(device))
    try:
        if state.rank == 0:
            success = not Path(output_dir).exists()
            if success:
                Path(output_dir).mkdir(parents=True, exist_ok=False)
        else:
            success = True
        _broadcast_success_or_raise(state, success)
        contract = _load_contract(Path(contract_path))
        layout, layout_metadata, sources, runtime = _load_runtime(contract)
        candidate_input = _load_candidate_prior_input(
            contract=contract,
            proposals_path=proposals_path,
            base_overlay_path=base_overlay_path,
            verification_points_path=verification_points_path,
        )
        proposal_tracks = candidate_input.candidate_track_ids
        rows = np.asarray(layout["source_row_indices"], dtype=np.int64)
        candidate_tracks = np.asarray(layout["candidate_track_ids"], dtype=np.int64)
        if (
            np.any(rows < 0)
            or np.any(rows >= len(proposal_tracks))
            or not np.array_equal(proposal_tracks[rows], candidate_tracks)
            or str(contract.get("proposals_sha256", "")) != candidate_input.source_sha256
        ):
            raise ValueError("context-attention layout does not align with frozen candidates")
        base = {
            "candidate_track_ids": proposal_tracks,
            "candidate_probabilities": candidate_input.candidate_probabilities,
            "null_probabilities": candidate_input.null_probabilities,
        }
        base_candidate = candidate_input.candidate_probabilities[rows]
        base_null = candidate_input.null_probabilities[rows]
        train_rows, train_classes, target_audit = _train_identity_targets(
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            query_xy=np.asarray(layout["xy"], dtype=np.float32),
            candidate_tracks=candidate_tracks,
            split_names=np.asarray(layout["split_names"]).astype(str),
            colmap_model_dir=Path(colmap_model_dir),
            radius_px=float(registered_identity_radius_px),
        )
        static_sources = {
            source.name: torch.from_numpy(
                np.ascontiguousarray(np.asarray(source.grid, dtype=np.float16))
            )
            for source in sources
        }
        image_sizes = np.asarray(sources[0].image_sizes, dtype=np.int64)
        model_dir = Path(output_dir) / "models"
        overlay_dir = Path(output_dir) / "overlays"
        if state.rank == 0:
            model_dir.mkdir()
            overlay_dir.mkdir()
        if state.enabled:
            distributed.barrier()
        probabilities: list[np.ndarray] = []
        null_probabilities: list[np.ndarray] = []
        per_view_logits: list[np.ndarray] = []
        histories: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []
        architecture = (
            "low_capacity_per_view_multiscale_cross_attention_absolute_phase_v1"
            if str(position_encoding) == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING
            else "low_capacity_per_view_multiscale_cross_attention_v1"
        )
        for family in families:
            model, history = _fit_one_family(
                family=family,
                static_sources=static_sources,
                image_sizes=image_sizes,
                runtime=runtime,
                query_xy=np.asarray(layout["xy"], dtype=np.float32),
                base_candidate=base_candidate,
                base_null=base_null,
                train_rows=train_rows,
                train_classes=train_classes,
                state=state,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                hidden_dim=int(hidden_dim),
                heads=int(heads),
                dropout=float(dropout),
                seed=int(seed),
                position_encoding=str(position_encoding),
            )
            candidate, null, view = _predict_sharded(
                model=model, state=state, batch_size=max(int(batch_size), 1)
            )
            probabilities.append(candidate.astype(np.float32, copy=False))
            null_probabilities.append(null.astype(np.float32, copy=False))
            per_view_logits.append(view.astype(np.float32, copy=False))
            histories.append({"family": family, "epochs": history})
            if state.rank == 0:
                model_path = model_dir / f"{family}.pt"
                model_metadata = {
                    "contract_sha256": file_sha256_short(Path(contract_path)),
                    "frozen_layout_features_sha256": contract["frozen_layout_features_sha256"],
                    "proposals_sha256": candidate_input.source_sha256,
                    "base_prior_overlay_sha256": candidate_input.base_prior_sha256,
                    "candidate_input_kind": candidate_input.kind,
                    "candidate_input_path": str(candidate_input.source_path),
                    "candidate_input_sha256": candidate_input.source_sha256,
                    "training_supervision_split": "train",
                    "registered_identity_radius_px": float(registered_identity_radius_px),
                    "validation_or_test_labels_used_by_fit": False,
                    "test_used_for_model_selection": False,
                    "supervision_mode": SUPERVISION_MODE,
                    "training_objective": TRAINING_OBJECTIVE,
                    "probability_semantics": PROBABILITY_SEMANTICS,
                    "base_prior_residual": True,
                    "architecture": architecture,
                    "family_profile": str(family_profile),
                    "position_encoding": str(position_encoding),
                    "position_only_control": bool(family.endswith("position_only")),
                    "hidden_dim": int(hidden_dim),
                    "heads": int(heads),
                    "dropout": float(dropout),
                    "target_audit": target_audit,
                }
                _save_model(
                    path=model_path, model=model, family=family, metadata=model_metadata
                )
                model_outputs.append(
                    {"family": family, "path": str(model_path), "sha256": file_sha256_short(model_path)}
                )
            if state.enabled:
                distributed.barrier()
        if state.rank != 0:
            return None
        probability_tensor = np.stack(probabilities, axis=0)
        null_tensor = np.stack(null_probabilities, axis=0)
        view_tensor = np.stack(per_view_logits, axis=0)
        prediction_metadata = {
            "format": PREDICTION_ARTIFACT_FORMAT,
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used_for_prediction": False,
            "training_supervision_split": "train",
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "supervision_mode": SUPERVISION_MODE,
            "training_objective": TRAINING_OBJECTIVE,
            "validation_or_test_labels_used_by_fit": False,
            "test_used_for_model_selection": False,
            "features_sha256": file_sha256_short(Path(contract["frozen_layout_features"])),
            "proposals_sha256": candidate_input.source_sha256,
            "base_prior_overlay_sha256": candidate_input.base_prior_sha256,
            "candidate_input_kind": candidate_input.kind,
            "candidate_input_path": str(candidate_input.source_path),
            "candidate_input_sha256": candidate_input.source_sha256,
            "context_attention_contract": str(contract_path),
            "context_attention_contract_sha256": file_sha256_short(Path(contract_path)),
            "families": list(families),
            "probability_semantics": PROBABILITY_SEMANTICS,
            "candidate_probability_role": "exact_registered_track_identity_or_explicit_null",
            "base_prior_residual": True,
            "architecture": architecture,
            "family_profile": str(family_profile),
            "position_encoding": str(position_encoding),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "dropout": float(dropout),
            "source_feature_protocol": {
                "image_retrieval_or_submap_used": False,
                "whole_image_summary_or_global_used": False,
                "candidate_anchor_conditioned_spatial_grid_only": True,
                "absolute_phase_coordinates": bool(
                    str(position_encoding) == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING
                ),
                "support_view_selection": layout_metadata.get("support_view_selection"),
                "context_only_center_mask_radius": contract["context_only_center_mask"]["radius"],
                "source_scales": contract["source_scales"],
            },
        }
        prediction_path = Path(output_dir) / "predictions_inference_only.npz"
        np.savez_compressed(
            prediction_path,
            source_row_indices=rows,
            query_ids=np.asarray(layout["query_ids"]).astype(str),
            split_names=np.asarray(layout["split_names"]).astype(str),
            candidate_track_ids=candidate_tracks,
            candidate_view_valid=np.asarray(layout["candidate_view_valid"], dtype=bool),
            family_names=np.asarray(families, dtype=np.str_),
            candidate_probabilities=probability_tensor,
            null_probabilities=null_tensor,
            per_view_logits=view_tensor,
            metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
        )
        overlays: list[dict[str, Any]] = []
        for index, family in enumerate(families):
            candidate, null, audit = _replace_overlay_rows(
                base=base,
                source_rows=rows,
                source_tracks=candidate_tracks,
                probabilities=probability_tensor[index],
                null_probabilities=null_tensor[index],
            )
            metadata = {
                "format": OVERLAY_ARTIFACT_FORMAT,
                "contains_ground_truth": False,
                "contains_target_errors": False,
                "probability_semantics": PROBABILITY_SEMANTICS,
                "candidate_probability_role": "exact_registered_track_identity_or_explicit_null",
                "base_prior_residual": True,
                "architecture": architecture,
                "family_profile": str(family_profile),
                "position_encoding": str(position_encoding),
                "position_only_control": bool(family.endswith("position_only")),
                "proposals_sha256": candidate_input.source_sha256,
                "features_sha256": file_sha256_short(Path(contract["frozen_layout_features"])),
                "context_attention_contract_sha256": file_sha256_short(Path(contract_path)),
                "predictions_sha256": file_sha256_short(prediction_path),
                "base_prior_overlay_sha256": candidate_input.base_prior_sha256,
                "candidate_input_kind": candidate_input.kind,
                "candidate_input_path": str(candidate_input.source_path),
                "candidate_input_sha256": candidate_input.source_sha256,
                "family": family,
                "training_supervision_split": "train",
                "registered_identity_radius_px": float(registered_identity_radius_px),
                "supervision_mode": SUPERVISION_MODE,
                "training_objective": TRAINING_OBJECTIVE,
                "validation_or_test_labels_used_by_fit": False,
                "replaced_source_row_count": int(len(rows)),
                "replaced_source_rows_sha256": _array_sha256_short(rows),
                "audit": audit,
            }
            overlay_path = overlay_dir / f"{family}.npz"
            np.savez_compressed(
                overlay_path,
                candidate_track_ids=base["candidate_track_ids"],
                candidate_probabilities=candidate,
                null_probabilities=null,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            overlays.append(
                {"family": family, "path": str(overlay_path), "sha256": file_sha256_short(overlay_path)}
            )
        summary = {
            "stage": "train_only_frozen_multiscale_context_attention_candidate_probe",
            "protocol": {
                "training_supervision_split": "train",
                "train_target": target_audit["supervision"],
                "registered_identity_radius_px": float(registered_identity_radius_px),
                "supervision_mode": SUPERVISION_MODE,
                "training_objective": TRAINING_OBJECTIVE,
                "probability_semantics": PROBABILITY_SEMANTICS,
                "base_prior_residual": True,
                "validation_or_test_labels_used_by_fit": False,
                "test_used_for_model_selection": False,
                "image_retrieval": False,
                "render": False,
                "fixed_support_view_log_mixture": True,
                "context_only_masks_anchor": True,
                "family_profile": str(family_profile),
                "position_encoding": str(position_encoding),
            },
            "target_audit": target_audit,
            "families": list(families),
            "fits": histories,
            "inputs": {
                "contract": str(contract_path),
                "contract_sha256": file_sha256_short(Path(contract_path)),
                "frozen_layout_features": str(contract["frozen_layout_features"]),
                "frozen_layout_features_sha256": contract["frozen_layout_features_sha256"],
                "candidate_input_kind": candidate_input.kind,
                "candidate_input": str(candidate_input.source_path),
                "candidate_input_sha256": candidate_input.source_sha256,
                "base_prior_overlay": (
                    None
                    if candidate_input.base_prior_path is None
                    else str(candidate_input.base_prior_path)
                ),
                "base_prior_overlay_sha256": candidate_input.base_prior_sha256,
                "base_prior_format": candidate_input.metadata.get("format"),
                "base_prior_probability_semantics": candidate_input.metadata.get(
                    "probability_semantics", "embedded_fixed_top_l_coarse_posterior"
                ),
                "colmap_model_dir": str(colmap_model_dir),
                "world_size": int(state.world_size),
            },
            "outputs": {
                "predictions": str(prediction_path),
                "predictions_sha256": file_sha256_short(prediction_path),
                "models": model_outputs,
                "overlays": overlays,
            },
        }
        summary_path = Path(output_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary
    finally:
        _close_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = fit_multiscale_context_attention_probe(
        contract_path=Path(args.contract),
        proposals_path=None if args.proposals is None else Path(args.proposals),
        base_overlay_path=(
            None if args.base_prior_overlay is None else Path(args.base_prior_overlay)
        ),
        verification_points_path=(
            None if args.verification_points is None else Path(args.verification_points)
        ),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        families=_parse_families(args.families, family_profile=str(args.family_profile)),
        family_profile=str(args.family_profile),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        heads=int(args.heads),
        dropout=float(args.dropout),
        seed=int(args.seed),
        device=str(args.device),
    )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
