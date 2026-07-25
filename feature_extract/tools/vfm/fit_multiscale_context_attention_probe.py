"""Fit a train-only multiscale candidate-context cross-attention probe.

The input contract is frozen before this command starts.  It fixes query
tokens, top-L landmark candidates, support observations, and all real-image
descriptor grids. Only train-image supervision is joined for the loss: either
registered SfM observation identity or a prebuilt train-only set of
geometrically valid top-L candidates. Validation/test labels are never loaded
by this command.

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
import math
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
    ABSOLUTE_CONTEXT_LIKELIHOOD_V2_FAMILIES,
    ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES,
    ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
    ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
    ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES,
    ANCHOR_RELATIVE_POSITION_ENCODING,
    BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES,
    CandidateBidirectionalAbsoluteContextLikelihood,
    CONTEXT_ATTENTION_FAMILIES,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V2,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V3,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V4,
    CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V5,
    CandidateContextAttentionProbe,
    build_fixed_candidate_context_runtime,
    context_attention_global_region_sizes,
    context_attention_profile,
    load_context_attention_frozen_layout,
    load_context_attention_sources,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    load_support_observation_geometry_index_npz,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.mixed_multiscale_candidate_probe import (
    geometric_membership_from_residuals,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    set_membership_negative_log_likelihood,
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
REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE = "registered_track_identity"
GEOMETRIC_SET_SUPERVISION_MODE = "geometric_set"
SUPPORTED_SUPERVISION_MODES = frozenset(
    {REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE, GEOMETRIC_SET_SUPERVISION_MODE}
)
EXACT_IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_exact_registered_track_identity_probability_plus_explicit_null_equals_one"
)
GEOMETRIC_PROBABILITY_SEMANTICS = (
    "candidate_geometric_correspondence_probability_plus_explicit_null_equals_one"
)
TRAINING_OBJECTIVE = "registered_query_observation_exact_track_or_explicit_null_nll_v1"
V2_TRAINING_OBJECTIVE = (
    "registered_query_observation_exact_track_or_explicit_null_nll_plus_"
    "frozen_coarse_prior_hard_negative_ranking_v1"
)
GEOMETRIC_SET_TRAINING_OBJECTIVE = (
    "set_log_mass_nll_over_geometric_candidate_membership_plus_"
    "frozen_coarse_prior_hard_negative_ranking_v1"
)
DUAL_HEAD_GEOMETRY_PLUS_EXACT_IDENTITY_TRAINING_OBJECTIVE = (
    "set_log_mass_nll_over_geometric_candidate_membership_plus_"
    "frozen_coarse_prior_hard_negative_ranking_plus_separate_"
    "registered_exact_identity_head_auxiliary_nll_v1"
)
GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT = "context_attention_geometric_train_targets_v1"
EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT = "context_attention_exact_identity_train_targets_v1"
LEGACY_ARCHITECTURE = "legacy_oneway_v1"
BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE = "bidirectional_absolute_v2"
BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE = "bidirectional_absolute_raw_v3"
BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE = "bidirectional_absolute_raw_layout_v4"
BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE = (
    "bidirectional_absolute_dual_head_raw_layout_v5"
)
BIDIRECTIONAL_ARCHITECTURES = frozenset(
    {
        BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
    }
)
FAMILY_PROFILES: dict[str, tuple[tuple[str, ...], str]] = {
    "relative_context_v1": (
        CONTEXT_ATTENTION_FAMILIES,
        ANCHOR_RELATIVE_POSITION_ENCODING,
    ),
    "absolute_phase_v1": (
        ABSOLUTE_PHASE_CONTEXT_ATTENTION_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
    "bidirectional_absolute_v2": (
        ABSOLUTE_CONTEXT_LIKELIHOOD_V2_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
    "bidirectional_absolute_raw_v3": (
        ABSOLUTE_CONTEXT_LIKELIHOOD_V3_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
    "bidirectional_absolute_raw_layout_v4": (
        ABSOLUTE_CONTEXT_LIKELIHOOD_V4_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
    "bidirectional_absolute_dual_head_raw_layout_v5": (
        ABSOLUTE_CONTEXT_LIKELIHOOD_V5_FAMILIES,
        ABSOLUTE_DUAL_FRAME_POSITION_ENCODING,
    ),
}
_PROFILE_ARCHITECTURES = {
    "relative_context_v1": LEGACY_ARCHITECTURE,
    "absolute_phase_v1": LEGACY_ARCHITECTURE,
    "bidirectional_absolute_v2": BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE,
    "bidirectional_absolute_raw_v3": BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE,
    "bidirectional_absolute_raw_layout_v4": BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
    "bidirectional_absolute_dual_head_raw_layout_v5": (
        BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE
    ),
}

_PROPOSAL_OVERLAY_CANDIDATE_INPUT = "proposal_overlay_v1"
_MIXED_POINTS_CANDIDATE_INPUT = "mixed_verification_points_embedded_coarse_prior_v1"


def _is_position_control_family(family: str) -> bool:
    """Return whether a paired family masks all descriptor evidence."""

    return str(family).endswith(
        (
            "position_only",
            "position_control_v2",
            "position_control_v3",
            "position_control_v4",
            "position_control_v5",
        )
    )


def _null_likelihood_name(architecture: str) -> str:
    """Name the explicit-null evidence without conflating architecture versions."""

    if str(architecture) in {
        BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
    }:
        return "permutation_invariant_candidate_raw_layout_statistics_v4"
    if str(architecture) == BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE:
        return "permutation_invariant_candidate_raw_visual_statistics_v3"
    if str(architecture) == BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE:
        return "permutation_invariant_candidate_visual_statistics_v2"
    return "immutable_base_null_prior_only"


def _architecture_evidence_metadata(architecture: str) -> dict[str, Any]:
    """Describe the fixed visual evidence path used by a saved artifact.

    The returned candidate LLR is deliberately *not* advertised as an
    independent pose likelihood.  It is a target-free candidate-reranking
    residual and can become a pose-selection factor only after a separate
    held-out calibration/denominator protocol.
    """

    is_bidirectional = str(architecture) in BIDIRECTIONAL_ARCHITECTURES
    is_raw_v3 = str(architecture) == BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE
    is_raw_layout_v4 = str(architecture) == BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE
    is_dual_head_v5 = str(architecture) == BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE
    return {
        "candidate_log_likelihood_ratio_semantics": (
            "per_candidate_per_view_visual_log_residual_relative_to_"
            "immutable_coarse_prior_candidate_reranking_only"
            if is_bidirectional
            else None
        ),
        "candidate_log_likelihood_ratio_is_independent_pose_likelihood": False,
        "shared_descriptor_projection": bool(is_raw_v3),
        "raw_frozen_cost_volume_statistics": bool(
            is_raw_v3 or is_raw_layout_v4 or is_dual_head_v5
        ),
        "raw_cost_volume_layout_statistics": bool(is_raw_layout_v4 or is_dual_head_v5),
        "attention_context_encoder": not bool(is_raw_layout_v4 or is_dual_head_v5)
        if is_bidirectional
        else False,
        "separate_exact_identity_head": bool(is_dual_head_v5),
        "identity_context_scales": (
            ["radio_final", "radio_intermediate"] if is_dual_head_v5 else None
        ),
        "identity_excluded_local_spatial_scales": (
            ["alike"] if is_dual_head_v5 else None
        ),
        "global_region_size_by_scale": (
            context_attention_global_region_sizes(str(architecture))
            if is_bidirectional
            else None
        ),
    }


def _contract_architecture(contract: Mapping[str, Any]) -> str:
    """Resolve legacy contracts while rejecting profile/format ambiguity."""

    artifact_format = str(contract.get("format", ""))
    architecture = str(contract.get("architecture", LEGACY_ARCHITECTURE))
    expected_format = {
        LEGACY_ARCHITECTURE: CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT,
        BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE: CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V2,
        BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE: CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V3,
        BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE: CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V4,
        BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE: (
            CONTEXT_ATTENTION_PROBE_CONTRACT_FORMAT_V5
        ),
    }.get(architecture)
    if expected_format is None:
        raise ValueError("unsupported context-attention contract architecture")
    if artifact_format != expected_format:
        raise ValueError("context-attention contract format and architecture differ")
    # Also validate the profile here; source-receptive-field checks happen in
    # ``_contract_paths`` below.
    expected_families, _scales, uses_global_regions = context_attention_profile(architecture)
    if tuple(contract.get("families", ())) != expected_families:
        raise ValueError("context-attention contract family profile differs")
    if bool(contract.get("whole_image_summary_or_global_used", False)) != bool(
        uses_global_regions
    ):
        raise ValueError("context-attention contract global-evidence profile differs")
    if uses_global_regions and (
        contract.get("soft_global_context_factor_used") is not True
        or contract.get("global_context_hard_retrieval_or_candidate_reselection") is not False
        or contract.get("candidate_conditioned_full_image_region_tokens") is not True
    ):
        raise ValueError("bidirectional context-attention contract lacks bounded global-evidence protocol")
    region_profile = contract.get("global_region_size_by_scale")
    if architecture in {
        BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
        BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
    } and dict(
        region_profile or {}
    ) != context_attention_global_region_sizes(architecture):
        raise ValueError("context-attention contract global-region profile differs")
    if (
        architecture == BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE
        and region_profile is not None
        and dict(region_profile) != context_attention_global_region_sizes(architecture)
    ):
        raise ValueError("context-attention contract global-region profile differs")
    return architecture


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
    parser.add_argument(
        "--architecture",
        choices=(
            LEGACY_ARCHITECTURE,
            BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE,
            BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE,
            BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
            BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE,
        ),
        default=LEGACY_ARCHITECTURE,
        help="frozen paired architecture associated with --family_profile",
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
        "--supervision_mode",
        choices=tuple(sorted(SUPPORTED_SUPERVISION_MODES)),
        default=REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
        help=(
            "registered_track_identity trains singleton exact SfM track targets; "
            "geometric_set trains the total mass of all train-only candidates "
            "whose reprojection is geometrically valid"
        ),
    )
    parser.add_argument(
        "--geometric_positive_threshold_px",
        type=float,
        default=2.0,
        help="train-only positive residual threshold for --supervision_mode=geometric_set",
    )
    parser.add_argument(
        "--geometric_train_targets",
        default=None,
        help=(
            "required with --supervision_mode=geometric_set; a prebuilt cache "
            "containing only train-row geometric memberships"
        ),
    )
    parser.add_argument(
        "--exact_identity_train_targets",
        default=None,
        help=(
            "required by the V5 dual-head geometric mode; a prebuilt cache of "
            "registered train-only exact-track-or-null targets"
        ),
    )
    parser.add_argument(
        "--exact_identity_auxiliary_loss_weight",
        type=float,
        default=0.0,
        help=(
            "weight for V5's separate exact-track identity head; this never "
            "changes the set-valued geometry head"
        ),
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
    parser.add_argument("--coarse_hard_negative_loss_weight", type=float, default=0.0)
    parser.add_argument("--coarse_hard_negative_margin", type=float, default=0.10)
    parser.add_argument("--coarse_hard_negative_prior_power", type=float, default=1.0)
    parser.add_argument("--coarse_hard_negative_evidence_weight", type=float, default=1.0)
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
    _families, expected_scales, _uses_global_regions = context_attention_profile(
        _contract_architecture(contract)
    )
    sources: dict[str, Path] = {}
    for value, scale in zip(raw_sources, expected_scales):
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
    if len(raw_sources) != len(expected_scales):
        raise ValueError("context-attention contract has extra source scales")
    return layout, geometry, sources


def _load_contract(path: Path) -> dict[str, Any]:
    contract = _metadata_file(Path(path), context="context-attention contract")
    architecture = _contract_architecture(contract)
    for field in (
        "contains_ground_truth",
        "contains_target_errors",
        "pose_or_ground_truth_used",
        "image_retrieval_or_submap_used",
        "render",
    ):
        if contract.get(field) is not False:
            raise ValueError(f"context-attention contract violates target-free protocol: {field}")
    if architecture == LEGACY_ARCHITECTURE and contract.get(
        "whole_image_summary_or_global_used"
    ) is not False:
        raise ValueError("legacy context-attention contract unexpectedly uses global evidence")
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


def _train_identity_targets_with_explicit_null(
    *,
    query_ids: np.ndarray,
    query_xy: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    colmap_model_dir: Path,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build v2 identity targets without silently discarding the null class.

    A registered observation whose physical track is absent from frozen top-L is
    a real training example for the explicit null state.  The legacy probe
    predates that distinction and intentionally remains unchanged for replay.
    """

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
    membership_all = registered_candidate_identity_target_membership(
        candidate_tracks[train], targets
    )
    supervised = np.asarray(targets.supervised, dtype=bool)
    if not np.any(supervised):
        raise ValueError("registered identity target found no train observations")
    full_membership = np.asarray(membership_all[supervised], dtype=bool)
    if full_membership.shape != (int(np.sum(supervised)), candidate_tracks.shape[1] + 1):
        raise RuntimeError("registered identity membership has an unexpected null column")
    if np.any(np.sum(full_membership, axis=1) != 1):
        raise RuntimeError("registered identity target is not singleton-or-null")
    selected_rows = train[supervised]
    candidate_count = int(candidate_tracks.shape[1])
    classes = np.where(
        np.any(full_membership[:, :-1], axis=1),
        np.argmax(full_membership[:, :-1], axis=1),
        candidate_count,
    ).astype(np.int64)
    membership = full_membership[:, :-1]
    exact = np.any(membership, axis=1)
    if np.any((classes == candidate_count) & np.any(membership, axis=1)):
        raise RuntimeError("null class and candidate hard-negative membership differ")
    return selected_rows, classes, membership, {
        "supervision": "registered_query_observation_exact_track_or_explicit_null_v2",
        "training_objective": V2_TRAINING_OBJECTIVE,
        "registered_identity_radius_px": float(radius_px),
        "train_split_row_count": int(len(train)),
        "registered_supervised_train_row_count": int(np.sum(supervised)),
        "registered_supervised_train_row_rate": float(np.mean(supervised)),
        "exact_track_retrieved_train_row_count": int(np.sum(exact)),
        "exact_track_retrieved_given_registered_train_rate": float(np.mean(exact)),
        "explicit_null_registered_train_row_count": int(np.sum(~exact)),
        "positive_candidate_train_count": int(np.sum(membership)),
        "colmap_images_sha256": file_sha256_short(images_path),
    }


def _train_geometric_targets_with_explicit_null(
    *,
    proposals_path: Path,
    source_rows: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    positive_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build train-only set-valued geometry targets without identity collapse.

    A detector anchor can legitimately be compatible with several 3-D tracks
    at the image resolution used by proposal/PnP.  For the latent top-L route,
    making one of them an arbitrary hard class is counterproductive.  This
    target maximizes the summed mass of every candidate that is geometrically
    correct under the *training* camera pose; only rows without such a
    candidate supervise the explicit null state.
    """

    if not math.isfinite(float(positive_threshold_px)) or float(positive_threshold_px) <= 0.0:
        raise ValueError("geometric positive threshold must be positive")
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    tracks = np.asarray(candidate_tracks, dtype=np.int64)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    if (
        tracks.ndim != 2
        or len(rows) != len(tracks)
        or len(splits) != len(tracks)
        or np.any(rows < 0)
    ):
        raise ValueError("geometric train target arrays are incompatible")
    train_rows = np.flatnonzero(splits == "train")
    if train_rows.size == 0:
        raise ValueError("context-attention layout has no train rows")
    with np.load(Path(proposals_path), allow_pickle=False) as data:
        if "candidate_gt_residuals_px" not in data.files:
            raise ValueError("train supervision proposals lack candidate_gt_residuals_px")
        all_residuals = np.asarray(data["candidate_gt_residuals_px"], dtype=np.float32)
    if all_residuals.ndim != 2 or np.any(rows[train_rows] >= len(all_residuals)):
        raise ValueError("geometric train residuals do not align with source rows")
    residuals = all_residuals[rows[train_rows]]
    train_tracks = tracks[train_rows]
    candidate_valid = train_tracks >= 0
    if (
        residuals.shape != train_tracks.shape
        or np.any(np.isnan(residuals))
        or np.any(residuals[candidate_valid] < 0.0)
    ):
        raise ValueError("geometric train residuals are invalid")
    membership = geometric_membership_from_residuals(
        residuals=residuals,
        candidate_valid=candidate_valid,
        threshold_px=float(positive_threshold_px),
    )
    positive = membership[:, :-1]
    positive_count = np.sum(positive, axis=1)
    if (
        membership.shape != (len(train_rows), train_tracks.shape[1] + 1)
        or np.any(np.sum(membership, axis=1) == 0)
        or np.any(membership[:, -1] & np.any(positive, axis=1))
    ):
        raise RuntimeError("geometric train target membership is invalid")
    return train_rows, membership, {
        "supervision": "query_to_projected_landmark_set_membership_or_explicit_null_v1",
        "training_objective": GEOMETRIC_SET_TRAINING_OBJECTIVE,
        "geometric_positive_threshold_px": float(positive_threshold_px),
        "train_row_count": int(len(train_rows)),
        "geometry_positive_train_row_count": int(np.sum(positive_count > 0)),
        "geometry_positive_train_row_rate": float(np.mean(positive_count > 0)),
        "multi_positive_train_row_count": int(np.sum(positive_count > 1)),
        "positive_candidate_train_count": int(np.sum(positive)),
        "explicit_null_train_row_count": int(np.sum(positive_count == 0)),
        "proposals_sha256": file_sha256_short(Path(proposals_path)),
    }


def _load_geometric_train_target_cache(
    *,
    target_cache_path: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    proposals_path: Path,
    source_rows: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    positive_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a compact geometric target table that contains train rows only.

    The original proposal archive carries residuals for every split.  It is
    valid to extract train memberships from that immutable source once, but a
    fitting process must not need to open an archive containing validation or
    test target residuals.  This loader makes the train-only boundary explicit
    and refuses stale/misaligned target caches.
    """

    cache_path = Path(target_cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"geometric train target cache is missing: {cache_path}")
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    tracks = np.asarray(candidate_tracks, dtype=np.int64)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    train_rows = np.flatnonzero(splits == "train").astype(np.int64)
    if (
        tracks.ndim != 2
        or rows.shape != splits.shape
        or len(rows) != len(tracks)
        or train_rows.size == 0
        or np.any(rows < 0)
    ):
        raise ValueError("geometric train target cache inputs are incompatible")
    required = {
        "layout_row_indices",
        "source_row_indices",
        "candidate_track_ids",
        "target_membership",
        "metadata_json",
    }
    with np.load(cache_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"geometric train target cache lacks {sorted(missing)}")
        layout_rows = np.asarray(data["layout_row_indices"], dtype=np.int64).reshape(-1)
        cache_source_rows = np.asarray(data["source_row_indices"], dtype=np.int64).reshape(-1)
        cache_tracks = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        raw_membership = np.asarray(data["target_membership"])
        try:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("geometric train target cache metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("geometric train target cache metadata is not an object")
    if raw_membership.dtype != np.dtype(bool):
        raise ValueError("geometric train target cache membership must be boolean")
    membership = np.asarray(raw_membership, dtype=bool)
    expected_source_rows = rows[train_rows]
    expected_tracks = tracks[train_rows]
    if (
        metadata.get("format") != GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT
        or metadata.get("contains_ground_truth") is not True
        or metadata.get("contains_validation_or_test_targets") is not False
        or metadata.get("training_split") != "train"
        or str(metadata.get("contract_sha256", "")) != file_sha256_short(Path(contract_path))
        or str(metadata.get("frozen_layout_features_sha256", ""))
        != str(contract.get("frozen_layout_features_sha256", ""))
        or str(metadata.get("proposals_sha256", "")) != file_sha256_short(Path(proposals_path))
        or not math.isclose(
            float(metadata.get("geometric_positive_threshold_px", float("nan"))),
            float(positive_threshold_px),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or not np.array_equal(layout_rows, train_rows)
        or not np.array_equal(cache_source_rows, expected_source_rows)
        or not np.array_equal(cache_tracks, expected_tracks)
        or membership.shape != (len(train_rows), tracks.shape[1] + 1)
        or str(metadata.get("layout_row_indices_sha256", ""))
        != _array_sha256_short(train_rows)
        or str(metadata.get("source_row_indices_sha256", ""))
        != _array_sha256_short(expected_source_rows)
        or str(metadata.get("candidate_track_ids_sha256", ""))
        != _array_sha256_short(expected_tracks)
    ):
        raise ValueError("geometric train target cache lineage or layout differs")
    positive = membership[:, :-1]
    candidate_valid = expected_tracks >= 0
    positive_count = np.sum(positive, axis=1)
    if (
        np.any(positive & ~candidate_valid)
        or np.any(np.sum(membership, axis=1) == 0)
        or np.any(membership[:, -1] & np.any(positive, axis=1))
        or np.any(membership[:, -1] != (positive_count == 0))
    ):
        raise ValueError("geometric train target cache membership is invalid")
    return train_rows, membership, {
        "supervision": "query_to_projected_landmark_set_membership_or_explicit_null_v1",
        "training_objective": GEOMETRIC_SET_TRAINING_OBJECTIVE,
        "geometric_positive_threshold_px": float(positive_threshold_px),
        "train_row_count": int(len(train_rows)),
        "geometry_positive_train_row_count": int(np.sum(positive_count > 0)),
        "geometry_positive_train_row_rate": float(np.mean(positive_count > 0)),
        "multi_positive_train_row_count": int(np.sum(positive_count > 1)),
        "positive_candidate_train_count": int(np.sum(positive)),
        "explicit_null_train_row_count": int(np.sum(positive_count == 0)),
        "proposals_sha256": file_sha256_short(Path(proposals_path)),
        "train_target_cache": str(cache_path),
        "train_target_cache_sha256": file_sha256_short(cache_path),
        "train_target_cache_format": GEOMETRIC_TRAIN_TARGET_CACHE_FORMAT,
        "validation_or_test_labels_loaded_by_fit": False,
    }


def _load_exact_identity_train_target_cache(
    *,
    target_cache_path: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    source_rows: np.ndarray,
    candidate_tracks: np.ndarray,
    split_names: np.ndarray,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only train-registered exact identity classes for V5's side head.

    Exact SfM track identity and the 2-pixel geometric set target intentionally
    have different meanings.  This compact table carries only registered train
    rows, allowing the shared visual trunk to train two *separate* output
    heads without opening a model or target archive that includes validation
    or test labels.
    """

    cache_path = Path(target_cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"exact identity train target cache is missing: {cache_path}")
    rows = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    tracks = np.asarray(candidate_tracks, dtype=np.int64)
    splits = np.asarray(split_names).astype(str).reshape(-1)
    train_rows = np.flatnonzero(splits == "train").astype(np.int64)
    if (
        tracks.ndim != 2
        or rows.shape != splits.shape
        or len(rows) != len(tracks)
        or train_rows.size == 0
        or np.any(rows < 0)
        or not math.isfinite(float(radius_px))
        or float(radius_px) <= 0.0
    ):
        raise ValueError("exact identity train target cache inputs are incompatible")
    required = {
        "layout_row_indices",
        "source_row_indices",
        "candidate_track_ids",
        "target_classes",
        "metadata_json",
    }
    with np.load(cache_path, allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"exact identity train target cache lacks {sorted(missing)}")
        layout_rows = np.asarray(data["layout_row_indices"], dtype=np.int64).reshape(-1)
        cache_source_rows = np.asarray(data["source_row_indices"], dtype=np.int64).reshape(-1)
        cache_tracks = np.asarray(data["candidate_track_ids"], dtype=np.int64)
        target_classes = np.asarray(data["target_classes"], dtype=np.int64).reshape(-1)
        try:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("exact identity train target cache metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("exact identity train target cache metadata is not an object")
    if (
        metadata.get("format") != EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT
        or metadata.get("contains_ground_truth") is not True
        or metadata.get("contains_validation_or_test_targets") is not False
        or metadata.get("training_split") != "train"
        or metadata.get("target_class_semantics")
        != "registered_exact_track_if_in_fixed_topl_else_explicit_null"
        or metadata.get("fit_must_not_open_colmap_identity_model") is not True
        or str(metadata.get("contract_sha256", "")) != file_sha256_short(Path(contract_path))
        or str(metadata.get("frozen_layout_features_sha256", ""))
        != str(contract.get("frozen_layout_features_sha256", ""))
        or not math.isclose(
            float(metadata.get("registered_identity_radius_px", float("nan"))),
            float(radius_px),
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        or layout_rows.ndim != 1
        or layout_rows.size == 0
        or np.any(layout_rows < 0)
        or np.any(layout_rows >= len(rows))
        or np.unique(layout_rows).size != len(layout_rows)
        or np.any(splits[layout_rows] != "train")
        or not np.array_equal(cache_source_rows, rows[layout_rows])
        or not np.array_equal(cache_tracks, tracks[layout_rows])
        or target_classes.shape != layout_rows.shape
        or str(metadata.get("layout_row_indices_sha256", ""))
        != _array_sha256_short(layout_rows)
        or str(metadata.get("source_row_indices_sha256", ""))
        != _array_sha256_short(rows[layout_rows])
        or str(metadata.get("candidate_track_ids_sha256", ""))
        != _array_sha256_short(tracks[layout_rows])
    ):
        raise ValueError("exact identity train target cache lineage or layout differs")
    candidate_count = int(tracks.shape[1])
    if (
        candidate_count <= 0
        or np.any(target_classes < 0)
        or np.any(target_classes > candidate_count)
        or np.any(
            (target_classes < candidate_count)
            & (cache_tracks[np.arange(len(cache_tracks)), target_classes.clip(max=candidate_count - 1)] < 0)
        )
    ):
        raise ValueError("exact identity train target cache classes are invalid")
    return layout_rows, target_classes, {
        "supervision": "registered_query_observation_exact_track_or_explicit_null_v3",
        "training_objective": "separate_exact_identity_head_auxiliary_nll_v1",
        "registered_identity_radius_px": float(radius_px),
        "registered_supervised_train_row_count": int(len(layout_rows)),
        "registered_supervised_train_row_rate": float(len(layout_rows) / len(train_rows)),
        "explicit_null_registered_train_row_count": int(
            np.sum(target_classes == candidate_count)
        ),
        "exact_track_retrieved_train_row_count": int(
            np.sum(target_classes < candidate_count)
        ),
        "train_target_cache": str(cache_path),
        "train_target_cache_sha256": file_sha256_short(cache_path),
        "train_target_cache_format": EXACT_IDENTITY_TRAIN_TARGET_CACHE_FORMAT,
        "validation_or_test_labels_loaded_by_fit": False,
    }


def _distributed_train_targets(
    *,
    architecture: str,
    state: _DistributedState,
    supervision_mode: str,
    identity_target_args: Mapping[str, Any],
    geometric_target_args: Mapping[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    """Materialize train-only targets once, then share their compact table.

    COLMAP's image observation model is large enough that every DDP rank
    independently parsing it wastes both startup time and host memory.  Only
    rank 0 sees that supervised model; the broadcast contains the resulting
    train row indices/classes/membership rather than any image observations or
    validation/test labels.
    """

    mode = str(supervision_mode)
    if mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError("unsupported context-attention supervision mode")
    if mode == GEOMETRIC_SET_SUPERVISION_MODE and str(architecture) not in BIDIRECTIONAL_ARCHITECTURES:
        raise ValueError("geometric-set supervision requires a bidirectional absolute matcher")
    payload: dict[str, Any] | None
    if state.rank == 0:
        try:
            if mode == GEOMETRIC_SET_SUPERVISION_MODE:
                rows, target_membership, audit = _load_geometric_train_target_cache(
                    **geometric_target_args
                )
                classes = None
                candidate_membership = np.asarray(target_membership[:, :-1], dtype=bool)
            elif str(architecture) in BIDIRECTIONAL_ARCHITECTURES:
                rows, classes, membership, audit = _train_identity_targets_with_explicit_null(
                    **identity_target_args
                )
                target_membership = np.zeros(
                    (len(rows), membership.shape[1] + 1), dtype=bool
                )
                target_membership[np.arange(len(rows)), classes] = True
                candidate_membership = membership
            else:
                rows, classes, audit = _train_identity_targets(**identity_target_args)
                target_membership = None
                candidate_membership = None
            payload = {
                "ok": True,
                "rows": np.asarray(rows, dtype=np.int64),
                "classes": None if classes is None else np.asarray(classes, dtype=np.int64),
                "target_membership": (
                    None
                    if target_membership is None
                    else np.asarray(target_membership, dtype=bool)
                ),
                "candidate_membership": (
                    None
                    if candidate_membership is None
                    else np.asarray(candidate_membership, dtype=bool)
                ),
                "audit": dict(audit),
            }
        except Exception as error:
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    else:
        payload = None
    if state.enabled:
        values: list[object] = [payload]
        distributed.broadcast_object_list(values, src=0)
        payload = values[0]  # type: ignore[assignment]
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        detail = "unknown rank-0 target-construction failure"
        if isinstance(payload, Mapping):
            detail = str(payload.get("error", detail))
        raise RuntimeError(f"context-attention train target construction failed: {detail}")
    rows = np.asarray(payload.get("rows"), dtype=np.int64)
    raw_classes = payload.get("classes")
    classes = None if raw_classes is None else np.asarray(raw_classes, dtype=np.int64)
    raw_target_membership = payload.get("target_membership")
    target_membership = (
        None
        if raw_target_membership is None
        else np.asarray(raw_target_membership, dtype=bool)
    )
    raw_candidate_membership = payload.get("candidate_membership")
    candidate_membership = (
        None
        if raw_candidate_membership is None
        else np.asarray(raw_candidate_membership, dtype=bool)
    )
    audit = payload.get("audit")
    if (
        rows.ndim != 1
        or not isinstance(audit, Mapping)
        or (classes is not None and classes.shape != rows.shape)
        or (target_membership is not None and target_membership.shape[0] != len(rows))
        or (candidate_membership is not None and candidate_membership.shape[0] != len(rows))
        or (classes is None and target_membership is None)
        or (
            target_membership is not None
            and candidate_membership is None
        )
    ):
        raise RuntimeError("broadcast context-attention train targets are invalid")
    return rows, classes, target_membership, candidate_membership, dict(audit)


def _distributed_exact_identity_auxiliary_targets(
    *,
    state: _DistributedState,
    target_args: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load V5's compact train-only identity side target once under DDP.

    This intentionally has a separate path from ``_distributed_train_targets``:
    V5's main geometric target covers every frozen train row, while strict SfM
    identity supervises only registered train anchors.  Keeping the payloads
    separate makes it impossible for fitting to reconstruct identity labels
    from a COLMAP model that could also contain validation/test observations.
    """

    payload: dict[str, Any] | None
    if state.rank == 0:
        try:
            rows, classes, audit = _load_exact_identity_train_target_cache(**target_args)
            payload = {
                "ok": True,
                "rows": np.asarray(rows, dtype=np.int64),
                "classes": np.asarray(classes, dtype=np.int64),
                "audit": dict(audit),
            }
        except Exception as error:
            payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    else:
        payload = None
    if state.enabled:
        values: list[object] = [payload]
        distributed.broadcast_object_list(values, src=0)
        payload = values[0]  # type: ignore[assignment]
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        detail = "unknown rank-0 identity target loading failure"
        if isinstance(payload, Mapping):
            detail = str(payload.get("error", detail))
        raise RuntimeError(f"context-attention identity target loading failed: {detail}")
    rows = np.asarray(payload.get("rows"), dtype=np.int64)
    classes = np.asarray(payload.get("classes"), dtype=np.int64)
    audit = payload.get("audit")
    if (
        rows.ndim != 1
        or classes.shape != rows.shape
        or rows.size == 0
        or np.unique(rows).size != len(rows)
        or np.any(rows < 0)
        or not isinstance(audit, Mapping)
    ):
        raise RuntimeError("broadcast context-attention identity targets are invalid")
    return rows, classes, dict(audit)


def _coarse_prior_hard_negative_ranking_loss(
    candidate_log_likelihood_ratios: torch.Tensor,
    *,
    positive_membership: torch.Tensor,
    candidate_prior: torch.Tensor,
    margin: float,
    prior_power: float,
    evidence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make visual evidence overcome high-prior frozen coarse mistakes.

    Coarse probabilities are used only as fixed comparison weights.  They are
    not passed into the visual encoder; the learned term remains a conditional
    image likelihood ratio.  Null-labelled rows are intentionally excluded
    because they contain no positive candidate identity to rank.
    """

    logits = candidate_log_likelihood_ratios.float()
    positive = positive_membership.to(device=logits.device, dtype=torch.bool)
    prior = candidate_prior.to(device=logits.device, dtype=torch.float32)
    if (
        logits.ndim != 2
        or positive.shape != logits.shape
        or prior.shape != logits.shape
        or not bool(torch.all(torch.isfinite(prior)))
        or bool(torch.any(prior < 0.0))
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
        or not math.isfinite(float(prior_power))
        or float(prior_power) <= 0.0
        or not math.isfinite(float(evidence_weight))
        or float(evidence_weight) <= 0.0
    ):
        raise ValueError("coarse-prior hard-negative ranking inputs are invalid")
    negative = ~positive
    active = torch.any(positive, dim=1) & torch.any(negative, dim=1)
    if not bool(torch.any(active)):
        return torch.zeros((), dtype=logits.dtype, device=logits.device), active
    log_prior = float(prior_power) * torch.log(prior.clamp_min(torch.finfo(torch.float32).tiny))
    positive_log_mass = torch.logsumexp(
        torch.where(
            positive,
            log_prior + float(evidence_weight) * logits,
            torch.full_like(logits, -torch.inf),
        ),
        dim=1,
    )
    negative_log_mass = torch.logsumexp(
        torch.where(
            negative,
            log_prior + float(evidence_weight) * logits,
            torch.full_like(logits, -torch.inf),
        ),
        dim=1,
    )
    loss = torch.nn.functional.softplus(
        float(margin) + negative_log_mass - positive_log_mass
    )
    return torch.mean(loss[active]), active


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
    train_classes: np.ndarray | None,
    train_target_membership: np.ndarray | None,
    train_positive_membership: np.ndarray | None,
    identity_train_rows: np.ndarray | None,
    identity_train_classes: np.ndarray | None,
    identity_auxiliary_loss_weight: float,
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
    architecture: str,
    coarse_hard_negative_loss_weight: float,
    coarse_hard_negative_margin: float,
    coarse_hard_negative_prior_power: float,
    coarse_hard_negative_evidence_weight: float,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if train_classes is None and train_target_membership is None:
        raise ValueError("context-attention fitting needs classes or target membership")
    if train_classes is not None and len(train_rows) != len(train_classes):
        raise ValueError("context-attention train classes differ from rows")
    if train_target_membership is not None and train_target_membership.shape != (
        len(train_rows),
        base_candidate.shape[1] + 1,
    ):
        raise ValueError("context-attention set targets differ from candidates plus null")
    if train_positive_membership is not None and train_positive_membership.shape != (
        len(train_rows),
        base_candidate.shape[1],
    ):
        raise ValueError("context-attention hard-negative memberships are incompatible")
    uses_dual_identity_head = (
        str(architecture) == BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE
    )
    if uses_dual_identity_head != (
        identity_train_rows is not None and identity_train_classes is not None
    ):
        raise ValueError("dual identity head and identity train targets differ")
    if uses_dual_identity_head and (
        identity_train_rows is None
        or identity_train_classes is None
        or identity_train_rows.ndim != 1
        or identity_train_classes.shape != identity_train_rows.shape
        or identity_train_rows.size == 0
        or np.unique(identity_train_rows).size != len(identity_train_rows)
        or np.any(identity_train_classes < 0)
        or np.any(identity_train_classes > base_candidate.shape[1])
        or float(identity_auxiliary_loss_weight) <= 0.0
    ):
        raise ValueError("dual identity auxiliary training inputs are invalid")
    if not uses_dual_identity_head and float(identity_auxiliary_loss_weight) != 0.0:
        raise ValueError("identity auxiliary loss requires the dual-head V5 architecture")
    _set_seed(_stable_family_seed(seed, family))
    common_model_kwargs = {
        "family": family,
        "sources": static_sources,
        "image_sizes": torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        "runtime": runtime,
        "query_xy": np.asarray(query_xy, dtype=np.float32),
        "base_candidate_probabilities": np.asarray(base_candidate, dtype=np.float32),
        "base_null_probabilities": np.asarray(base_null, dtype=np.float32),
        "hidden_dim": int(hidden_dim),
        "heads": int(heads),
        "dropout": float(dropout),
    }
    if str(architecture) == LEGACY_ARCHITECTURE:
        model: nn.Module = CandidateContextAttentionProbe(
            **common_model_kwargs, position_encoding=str(position_encoding)
        ).to(state.device)
    elif str(architecture) in BIDIRECTIONAL_ARCHITECTURES:
        model = CandidateBidirectionalAbsoluteContextLikelihood(**common_model_kwargs).to(
            state.device
        )
    else:
        raise ValueError("unsupported context-attention architecture")
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
    class_lookup = (
        None
        if train_classes is None
        else {int(row): int(target) for row, target in zip(train_rows, train_classes)}
    )
    target_membership_lookup = (
        None
        if train_target_membership is None
        else {
            int(row): np.asarray(membership, dtype=bool)
            for row, membership in zip(train_rows, train_target_membership)
        }
    )
    positive_membership_lookup = (
        None
        if train_positive_membership is None
        else {
            int(row): np.asarray(membership, dtype=bool)
            for row, membership in zip(train_rows, train_positive_membership)
        }
    )
    identity_class_lookup = (
        None
        if identity_train_rows is None or identity_train_classes is None
        else {
            int(row): int(target)
            for row, target in zip(identity_train_rows, identity_train_classes)
        }
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        train_model.train()
        local_loss = 0.0
        local_nll = 0.0
        local_hard_negative = 0.0
        local_hard_negative_groups = 0.0
        local_identity_nll = 0.0
        local_identity_count = 0
        local_count = 0
        epoch_rows = _sharded_epoch_rows(
            train_rows, epoch=epoch, seed=_stable_family_seed(seed, family), state=state
        )
        for begin in range(0, len(epoch_rows), int(batch_size)):
            batch_rows_np = epoch_rows[begin : begin + int(batch_size)]
            batch_targets_np = (
                None
                if class_lookup is None
                else np.asarray(
                    [class_lookup[int(row)] for row in batch_rows_np], dtype=np.int64
                )
            )
            batch_target_membership_np = (
                None
                if target_membership_lookup is None
                else np.stack(
                    [target_membership_lookup[int(row)] for row in batch_rows_np], axis=0
                )
            )
            batch_positive_membership_np = (
                None
                if positive_membership_lookup is None
                else np.stack(
                    [positive_membership_lookup[int(row)] for row in batch_rows_np], axis=0
                )
            )
            batch_identity_classes_np = (
                None
                if identity_class_lookup is None
                else np.asarray(
                    [identity_class_lookup.get(int(row), -1) for row in batch_rows_np],
                    dtype=np.int64,
                )
            )
            rows = torch.from_numpy(batch_rows_np).to(device=state.device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=state.device.type,
                dtype=torch.float16,
                enabled=state.device.type == "cuda",
            ):
                identity_logits: torch.Tensor | None = None
                if uses_dual_identity_head:
                    details = train_model(rows, return_details=True)
                    if not isinstance(details, Mapping) or "logits" not in details:
                        raise RuntimeError("dual identity model omitted geometry details")
                    logits = details["logits"]
                    identity_logits = details.get("identity_logits")
                    if not isinstance(identity_logits, torch.Tensor):
                        raise RuntimeError("dual identity model omitted identity logits")
                else:
                    _candidate, _null, _view, logits = train_model(rows)
                if batch_target_membership_np is None:
                    if batch_targets_np is None:
                        raise RuntimeError("context-attention batch has no target")
                    targets = torch.from_numpy(batch_targets_np).to(
                        device=state.device, dtype=torch.long
                    )
                    nll = F.cross_entropy(logits, targets)
                else:
                    target_membership = torch.from_numpy(batch_target_membership_np).to(
                        device=state.device, dtype=torch.bool
                    )
                    nll = set_membership_negative_log_likelihood(
                        F.log_softmax(logits.float(), dim=1), target_membership
                    )
                hard_negative = torch.zeros((), dtype=nll.dtype, device=nll.device)
                hard_negative_groups = torch.zeros((), dtype=torch.bool, device=nll.device)
                if (
                    batch_positive_membership_np is not None
                    and float(coarse_hard_negative_loss_weight) > 0.0
                ):
                    prior = torch.from_numpy(
                        np.asarray(base_candidate[batch_rows_np], dtype=np.float32)
                    ).to(device=state.device)
                    residual = logits[:, :-1] - torch.log(prior.clamp_min(1e-12))
                    hard_negative, hard_negative_groups = _coarse_prior_hard_negative_ranking_loss(
                        residual,
                        positive_membership=torch.from_numpy(batch_positive_membership_np).to(
                            device=state.device
                        ),
                        candidate_prior=prior,
                        margin=float(coarse_hard_negative_margin),
                        prior_power=float(coarse_hard_negative_prior_power),
                        evidence_weight=float(coarse_hard_negative_evidence_weight),
                    )
                identity_nll = torch.zeros((), dtype=nll.dtype, device=nll.device)
                identity_active = torch.zeros((), dtype=torch.bool, device=nll.device)
                if uses_dual_identity_head:
                    if batch_identity_classes_np is None or identity_logits is None:
                        raise RuntimeError("dual identity batch omitted train targets")
                    identity_targets = torch.from_numpy(batch_identity_classes_np).to(
                        device=state.device, dtype=torch.long
                    )
                    identity_active = identity_targets >= 0
                    # Keep every V5 parameter in DDP's graph even on a rare
                    # shard/batch with no registered SfM anchors.
                    identity_nll = identity_logits.sum() * 0.0
                    if bool(torch.any(identity_active)):
                        identity_nll = F.cross_entropy(
                            identity_logits[identity_active], identity_targets[identity_active]
                        )
                loss = (
                    nll
                    + float(coarse_hard_negative_loss_weight) * hard_negative
                    + float(identity_auxiliary_loss_weight) * identity_nll
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            local_loss += float(loss.detach().item()) * len(batch_rows_np)
            local_nll += float(nll.detach().item()) * len(batch_rows_np)
            local_hard_negative += float(hard_negative.detach().item()) * len(batch_rows_np)
            local_hard_negative_groups += float(hard_negative_groups.float().sum().detach().item())
            identity_active_count = int(identity_active.long().sum().detach().item())
            local_identity_nll += float(identity_nll.detach().item()) * identity_active_count
            local_identity_count += identity_active_count
            local_count += int(len(batch_rows_np))
        total_loss, total_count = _all_reduce_loss(local_loss, local_count, state)
        total_nll, _ = _all_reduce_loss(local_nll, local_count, state)
        total_hard_negative, _ = _all_reduce_loss(local_hard_negative, local_count, state)
        total_hard_negative_groups, _ = _all_reduce_loss(
            local_hard_negative_groups, local_count, state
        )
        total_identity_nll, total_identity_count = _all_reduce_loss(
            local_identity_nll, local_identity_count, state
        )
        mean_loss = total_loss / max(total_count, 1)
        mean_nll = total_nll / max(total_count, 1)
        mean_hard_negative = total_hard_negative / max(total_count, 1)
        mean_identity_nll = total_identity_nll / max(total_identity_count, 1)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(mean_loss),
                "train_nll": float(mean_nll),
                "train_coarse_hard_negative": float(mean_hard_negative),
                "train_coarse_hard_negative_group_rate": float(
                    total_hard_negative_groups / max(total_count, 1)
                ),
                "train_identity_auxiliary_nll": float(mean_identity_nll),
                "train_identity_auxiliary_row_count": float(total_identity_count),
            }
        )
        if state.rank == 0 and ((epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == int(epochs)):
            print(
                json.dumps(
                    {
                        "stage": "multiscale_context_attention_fit",
                        "family": family,
                        "epoch": int(epoch + 1),
                        "epochs": int(epochs),
                        "train_loss": float(mean_loss),
                        "train_nll": float(mean_nll),
                        "train_coarse_hard_negative": float(mean_hard_negative),
                        "train_coarse_hard_negative_group_rate": float(
                            total_hard_negative_groups / max(total_count, 1)
                        ),
                        "train_identity_auxiliary_nll": float(mean_identity_nll),
                        "train_identity_auxiliary_row_count": int(total_identity_count),
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
    *, model: nn.Module, state: _DistributedState, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    model.eval()
    row_count = model.row_count
    candidate_count = int(model._support_image_indices.shape[1])
    view_count = int(model._support_image_indices.shape[2])
    candidate = torch.zeros((row_count, candidate_count), dtype=torch.float32, device=state.device)
    null = torch.zeros((row_count,), dtype=torch.float32, device=state.device)
    views = torch.zeros(
        (row_count, candidate_count, view_count), dtype=torch.float32, device=state.device
    )
    is_bidirectional = isinstance(model, CandidateBidirectionalAbsoluteContextLikelihood)
    scale_views: torch.Tensor | None = None
    view_log_probabilities: torch.Tensor | None = None
    candidate_log_likelihood_ratios: torch.Tensor | None = None
    null_log_likelihood_ratios: torch.Tensor | None = None
    identity_candidate: torch.Tensor | None = None
    identity_null: torch.Tensor | None = None
    identity_views: torch.Tensor | None = None
    identity_scale_views: torch.Tensor | None = None
    identity_view_log_probabilities: torch.Tensor | None = None
    identity_candidate_log_likelihood_ratios: torch.Tensor | None = None
    identity_null_log_likelihood_ratios: torch.Tensor | None = None
    if is_bidirectional:
        scale_views = torch.zeros(
            (
                row_count,
                candidate_count,
                view_count,
                # V4 intentionally has no attention encoders, but its
                # target-free per-scale raw-layout evidence has one immutable
                # entry for every declared bidirectional source scale.
                len(BIDIRECTIONAL_ABSOLUTE_CONTEXT_SCALES),
            ),
            dtype=torch.float32,
            device=state.device,
        )
        view_log_probabilities = torch.zeros_like(views)
        candidate_log_likelihood_ratios = torch.zeros(
            (row_count, candidate_count), dtype=torch.float32, device=state.device
        )
        null_log_likelihood_ratios = torch.zeros(
            (row_count,), dtype=torch.float32, device=state.device
        )
        if bool(model._uses_dual_identity_head):
            identity_candidate = torch.zeros_like(candidate)
            identity_null = torch.zeros_like(null)
            identity_views = torch.zeros_like(views)
            identity_scale_views = torch.zeros_like(scale_views)
            identity_view_log_probabilities = torch.zeros_like(views)
            identity_candidate_log_likelihood_ratios = torch.zeros_like(candidate)
            identity_null_log_likelihood_ratios = torch.zeros_like(null)
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
                if is_bidirectional:
                    details = model.forward_with_details(selected)
                    probability = details["candidate_probabilities"]
                    null_probability = details["null_probabilities"]
                    view_logits = details["view_logits"]
                    assert (
                        scale_views is not None
                        and view_log_probabilities is not None
                        and candidate_log_likelihood_ratios is not None
                        and null_log_likelihood_ratios is not None
                    )
                    scale_views[selected] = details["per_scale_view_logits"].to(
                        dtype=torch.float32
                    )
                    view_log_probabilities[selected] = details["view_log_probabilities"].to(
                        dtype=torch.float32
                    )
                    candidate_log_likelihood_ratios[selected] = details[
                        "candidate_log_likelihood_ratios"
                    ].to(dtype=torch.float32)
                    null_log_likelihood_ratios[selected] = details[
                        "null_log_likelihood_ratio"
                    ].to(dtype=torch.float32)
                    if identity_candidate is not None:
                        if not (
                            identity_null is not None
                            and identity_views is not None
                            and identity_scale_views is not None
                            and identity_view_log_probabilities is not None
                            and identity_candidate_log_likelihood_ratios is not None
                            and identity_null_log_likelihood_ratios is not None
                        ):
                            raise RuntimeError("dual identity prediction buffers are incomplete")
                        identity_candidate[selected] = details[
                            "identity_candidate_probabilities"
                        ].to(dtype=torch.float32)
                        identity_null[selected] = details["identity_null_probabilities"].to(
                            dtype=torch.float32
                        )
                        identity_views[selected] = details["identity_view_logits"].to(
                            dtype=torch.float32
                        )
                        identity_scale_views[selected] = details[
                            "identity_per_scale_view_logits"
                        ].to(dtype=torch.float32)
                        identity_view_log_probabilities[selected] = details[
                            "identity_view_log_probabilities"
                        ].to(dtype=torch.float32)
                        identity_candidate_log_likelihood_ratios[selected] = details[
                            "identity_candidate_log_likelihood_ratios"
                        ].to(dtype=torch.float32)
                        identity_null_log_likelihood_ratios[selected] = details[
                            "identity_null_log_likelihood_ratio"
                        ].to(dtype=torch.float32)
                else:
                    probability, null_probability, view_logits, _logits = model(selected)
            candidate[selected] = probability.to(dtype=torch.float32)
            null[selected] = null_probability.to(dtype=torch.float32)
            views[selected] = view_logits.to(dtype=torch.float32)
    if state.enabled:
        values_to_reduce = [candidate, null, views]
        if scale_views is not None:
            values_to_reduce.append(scale_views)
        if view_log_probabilities is not None:
            values_to_reduce.append(view_log_probabilities)
        if candidate_log_likelihood_ratios is not None:
            values_to_reduce.append(candidate_log_likelihood_ratios)
        if null_log_likelihood_ratios is not None:
            values_to_reduce.append(null_log_likelihood_ratios)
        for values in (
            identity_candidate,
            identity_null,
            identity_views,
            identity_scale_views,
            identity_view_log_probabilities,
            identity_candidate_log_likelihood_ratios,
            identity_null_log_likelihood_ratios,
        ):
            if values is not None:
                values_to_reduce.append(values)
        for values in values_to_reduce:
            distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
    output = (
        candidate.detach().cpu().numpy(),
        null.detach().cpu().numpy(),
        views.detach().cpu().numpy(),
    )
    if not np.allclose(output[0].sum(axis=1) + output[1], 1.0, atol=1e-5):
        raise RuntimeError("context-attention prediction does not conserve probability mass")
    details_output = None
    if (
        scale_views is not None
        and view_log_probabilities is not None
        and candidate_log_likelihood_ratios is not None
        and null_log_likelihood_ratios is not None
    ):
        details_output = {
            "per_scale_view_logits": scale_views.detach().cpu().numpy(),
            "view_log_probabilities": view_log_probabilities.detach().cpu().numpy(),
            "candidate_log_likelihood_ratios": candidate_log_likelihood_ratios.detach()
            .cpu()
            .numpy(),
            "null_log_likelihood_ratios": null_log_likelihood_ratios.detach()
            .cpu()
            .numpy(),
        }
        if identity_candidate is not None:
            if not (
                identity_null is not None
                and identity_views is not None
                and identity_scale_views is not None
                and identity_view_log_probabilities is not None
                and identity_candidate_log_likelihood_ratios is not None
                and identity_null_log_likelihood_ratios is not None
            ):
                raise RuntimeError("dual identity prediction details are incomplete")
            details_output.update(
                {
                    "identity_candidate_probabilities": identity_candidate.detach()
                    .cpu()
                    .numpy(),
                    "identity_null_probabilities": identity_null.detach().cpu().numpy(),
                    "identity_view_logits": identity_views.detach().cpu().numpy(),
                    "identity_per_scale_view_logits": identity_scale_views.detach()
                    .cpu()
                    .numpy(),
                    "identity_view_log_probabilities": identity_view_log_probabilities.detach()
                    .cpu()
                    .numpy(),
                    "identity_candidate_log_likelihood_ratios": (
                        identity_candidate_log_likelihood_ratios.detach().cpu().numpy()
                    ),
                    "identity_null_log_likelihood_ratios": (
                        identity_null_log_likelihood_ratios.detach().cpu().numpy()
                    ),
                }
            )
    return (*output, details_output)


def _save_model(
    *,
    path: Path,
    model: nn.Module,
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
    architecture: str = LEGACY_ARCHITECTURE,
    supervision_mode: str = REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE,
    geometric_positive_threshold_px: float = 2.0,
    geometric_train_targets_path: Path | None = None,
    exact_identity_train_targets_path: Path | None = None,
    exact_identity_auxiliary_loss_weight: float = 0.0,
    coarse_hard_negative_loss_weight: float = 0.0,
    coarse_hard_negative_margin: float = 0.10,
    coarse_hard_negative_prior_power: float = 1.0,
    coarse_hard_negative_evidence_weight: float = 1.0,
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
        or not math.isfinite(float(geometric_positive_threshold_px))
        or float(geometric_positive_threshold_px) <= 0.0
        or not math.isfinite(float(exact_identity_auxiliary_loss_weight))
        or float(exact_identity_auxiliary_loss_weight) < 0.0
        or float(coarse_hard_negative_loss_weight) < 0.0
        or float(coarse_hard_negative_margin) < 0.0
        or float(coarse_hard_negative_prior_power) <= 0.0
        or float(coarse_hard_negative_evidence_weight) <= 0.0
    ):
        raise ValueError("context-attention optimization arguments are invalid")
    expected_families, position_encoding = _family_profile(family_profile)
    if tuple(families) != expected_families:
        raise ValueError("context-attention family set differs from the frozen paired protocol")
    if str(architecture) != _PROFILE_ARCHITECTURES.get(str(family_profile)):
        raise ValueError("context-attention architecture differs from its frozen family profile")
    mode = str(supervision_mode)
    if mode not in SUPPORTED_SUPERVISION_MODES:
        raise ValueError("unsupported context-attention supervision mode")
    if mode == GEOMETRIC_SET_SUPERVISION_MODE and str(architecture) not in BIDIRECTIONAL_ARCHITECTURES:
        raise ValueError("geometric-set supervision requires a bidirectional absolute matcher")
    if mode == GEOMETRIC_SET_SUPERVISION_MODE and geometric_train_targets_path is None:
        raise ValueError(
            "geometric-set supervision requires a prebuilt --geometric_train_targets cache"
        )
    uses_dual_identity_head = (
        str(architecture) == BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE
    )
    if uses_dual_identity_head and mode != GEOMETRIC_SET_SUPERVISION_MODE:
        raise ValueError("V5 dual-head fitting requires geometric-set main supervision")
    if uses_dual_identity_head and exact_identity_train_targets_path is None:
        raise ValueError(
            "V5 dual-head fitting requires a prebuilt --exact_identity_train_targets cache"
        )
    if uses_dual_identity_head and float(exact_identity_auxiliary_loss_weight) <= 0.0:
        raise ValueError("V5 dual-head fitting requires a positive identity auxiliary loss weight")
    if not uses_dual_identity_head and (
        exact_identity_train_targets_path is not None
        or float(exact_identity_auxiliary_loss_weight) != 0.0
    ):
        raise ValueError("exact identity auxiliary targets are reserved for V5 dual-head fitting")
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
        if _contract_architecture(contract) != str(architecture):
            raise ValueError("context-attention fit architecture differs from frozen contract")
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
        if mode == GEOMETRIC_SET_SUPERVISION_MODE and (
            candidate_input.kind != _PROPOSAL_OVERLAY_CANDIDATE_INPUT
            or proposals_path is None
        ):
            raise ValueError("geometric-set supervision requires the frozen proposal artifact")
        identity_target_args = {
            "query_ids": np.asarray(layout["query_ids"]).astype(str),
            "query_xy": np.asarray(layout["xy"], dtype=np.float32),
            "candidate_tracks": candidate_tracks,
            "split_names": np.asarray(layout["split_names"]).astype(str),
            "colmap_model_dir": Path(colmap_model_dir),
            "radius_px": float(registered_identity_radius_px),
        }
        geometric_target_args = {
            "target_cache_path": (
                Path(geometric_train_targets_path)
                if geometric_train_targets_path is not None
                else Path()
            ),
            "contract_path": Path(contract_path),
            "contract": contract,
            "proposals_path": Path(proposals_path) if proposals_path is not None else Path(),
            "source_rows": rows,
            "candidate_tracks": candidate_tracks,
            "split_names": np.asarray(layout["split_names"]).astype(str),
            "positive_threshold_px": float(geometric_positive_threshold_px),
        }
        (
            train_rows,
            train_classes,
            train_target_membership,
            train_positive_membership,
            target_audit,
        ) = _distributed_train_targets(
            architecture=str(architecture),
            state=state,
            supervision_mode=mode,
            identity_target_args=identity_target_args,
            geometric_target_args=geometric_target_args,
        )
        identity_train_rows: np.ndarray | None = None
        identity_train_classes: np.ndarray | None = None
        identity_auxiliary_audit: dict[str, Any] | None = None
        if uses_dual_identity_head:
            exact_identity_target_args = {
                "target_cache_path": Path(exact_identity_train_targets_path),
                "contract_path": Path(contract_path),
                "contract": contract,
                "source_rows": rows,
                "candidate_tracks": candidate_tracks,
                "split_names": np.asarray(layout["split_names"]).astype(str),
                "radius_px": float(registered_identity_radius_px),
            }
            (
                identity_train_rows,
                identity_train_classes,
                identity_auxiliary_audit,
            ) = _distributed_exact_identity_auxiliary_targets(
                state=state,
                target_args=exact_identity_target_args,
            )
            if not np.all(np.isin(identity_train_rows, train_rows)):
                raise RuntimeError("V5 exact identity targets are outside geometric train rows")
            target_audit = {
                **target_audit,
                "training_objective": (
                    DUAL_HEAD_GEOMETRY_PLUS_EXACT_IDENTITY_TRAINING_OBJECTIVE
                ),
                "identity_auxiliary": {
                    "enabled": True,
                    "loss_weight": float(exact_identity_auxiliary_loss_weight),
                    "probability_semantics": EXACT_IDENTITY_PROBABILITY_SEMANTICS,
                    "candidate_probability_role": (
                        "strict_registered_track_identity_or_explicit_null_side_head"
                    ),
                    "candidate_log_likelihood_ratio_is_independent_pose_likelihood": False,
                    "allowed_for_pnp_overlay": False,
                    **identity_auxiliary_audit,
                },
            }
        if mode == REGISTERED_TRACK_IDENTITY_SUPERVISION_MODE:
            probability_semantics = EXACT_IDENTITY_PROBABILITY_SEMANTICS
            probability_role = "exact_registered_track_identity_or_explicit_null"
        else:
            probability_semantics = GEOMETRIC_PROBABILITY_SEMANTICS
            probability_role = "geometric_candidate_set_or_explicit_null"
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
        per_scale_view_logits: list[np.ndarray] = []
        view_log_probabilities: list[np.ndarray] = []
        candidate_log_likelihood_ratios: list[np.ndarray] = []
        null_log_likelihood_ratios: list[np.ndarray] = []
        identity_probabilities: list[np.ndarray] = []
        identity_null_probabilities: list[np.ndarray] = []
        identity_per_view_logits: list[np.ndarray] = []
        identity_per_scale_view_logits: list[np.ndarray] = []
        identity_view_log_probabilities: list[np.ndarray] = []
        identity_candidate_log_likelihood_ratios: list[np.ndarray] = []
        identity_null_log_likelihood_ratios: list[np.ndarray] = []
        histories: list[dict[str, Any]] = []
        model_outputs: list[dict[str, Any]] = []
        architecture_name = {
            BIDIRECTIONAL_ABSOLUTE_ARCHITECTURE: (
                "bidirectional_per_view_multiscale_absolute_context_correlation_v2"
            ),
            BIDIRECTIONAL_RAW_ABSOLUTE_ARCHITECTURE: (
                "shared_projection_raw_cost_volume_absolute_likelihood_v3"
            ),
            BIDIRECTIONAL_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE: (
                "low_capacity_raw_layout_cost_volume_absolute_likelihood_v4"
            ),
            BIDIRECTIONAL_DUAL_HEAD_RAW_LAYOUT_ABSOLUTE_ARCHITECTURE: (
                "dual_head_raw_layout_geometry_and_exact_identity_likelihood_v5"
            ),
        }.get(
            str(architecture),
            (
                "low_capacity_per_view_multiscale_cross_attention_absolute_phase_v1"
                if str(position_encoding) == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING
                else "low_capacity_per_view_multiscale_cross_attention_v1"
            ),
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
                train_target_membership=train_target_membership,
                train_positive_membership=train_positive_membership,
                identity_train_rows=identity_train_rows,
                identity_train_classes=identity_train_classes,
                identity_auxiliary_loss_weight=float(exact_identity_auxiliary_loss_weight),
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
                architecture=str(architecture),
                coarse_hard_negative_loss_weight=float(coarse_hard_negative_loss_weight),
                coarse_hard_negative_margin=float(coarse_hard_negative_margin),
                coarse_hard_negative_prior_power=float(coarse_hard_negative_prior_power),
                coarse_hard_negative_evidence_weight=float(coarse_hard_negative_evidence_weight),
            )
            candidate, null, view, details = _predict_sharded(
                model=model, state=state, batch_size=max(int(batch_size), 1)
            )
            probabilities.append(candidate.astype(np.float32, copy=False))
            null_probabilities.append(null.astype(np.float32, copy=False))
            per_view_logits.append(view.astype(np.float32, copy=False))
            if details is not None:
                per_scale_view_logits.append(
                    details["per_scale_view_logits"].astype(np.float32, copy=False)
                )
                view_log_probabilities.append(
                    details["view_log_probabilities"].astype(np.float32, copy=False)
                )
                candidate_log_likelihood_ratios.append(
                    details["candidate_log_likelihood_ratios"].astype(
                        np.float32, copy=False
                    )
                )
                null_log_likelihood_ratios.append(
                    details["null_log_likelihood_ratios"].astype(np.float32, copy=False)
                )
                if "identity_candidate_probabilities" in details:
                    identity_probabilities.append(
                        details["identity_candidate_probabilities"].astype(
                            np.float32, copy=False
                        )
                    )
                    identity_null_probabilities.append(
                        details["identity_null_probabilities"].astype(
                            np.float32, copy=False
                        )
                    )
                    identity_per_view_logits.append(
                        details["identity_view_logits"].astype(np.float32, copy=False)
                    )
                    identity_per_scale_view_logits.append(
                        details["identity_per_scale_view_logits"].astype(
                            np.float32, copy=False
                        )
                    )
                    identity_view_log_probabilities.append(
                        details["identity_view_log_probabilities"].astype(
                            np.float32, copy=False
                        )
                    )
                    identity_candidate_log_likelihood_ratios.append(
                        details["identity_candidate_log_likelihood_ratios"].astype(
                            np.float32, copy=False
                        )
                    )
                    identity_null_log_likelihood_ratios.append(
                        details["identity_null_log_likelihood_ratios"].astype(
                            np.float32, copy=False
                        )
                    )
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
                    "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
                    "validation_or_test_labels_used_by_fit": False,
                    "test_used_for_model_selection": False,
                    "supervision_mode": mode,
                    "training_objective": target_audit["training_objective"],
                    "probability_semantics": probability_semantics,
                    "identity_probability_semantics": (
                        EXACT_IDENTITY_PROBABILITY_SEMANTICS
                        if uses_dual_identity_head
                        else None
                    ),
                    "identity_candidate_probability_role": (
                        "strict_registered_track_identity_or_explicit_null_side_head"
                        if uses_dual_identity_head
                        else None
                    ),
                    "identity_candidate_probability_allowed_for_pnp_overlay": False,
                    "base_prior_residual": True,
                    "architecture": architecture_name,
                    "architecture_id": str(architecture),
                    "family_profile": str(family_profile),
                    "position_encoding": str(position_encoding),
                    "position_only_control": _is_position_control_family(family),
                    "explicit_null_train_supervision": bool(
                        str(architecture) in BIDIRECTIONAL_ARCHITECTURES
                    ),
                    "null_likelihood": _null_likelihood_name(str(architecture)),
                    "coarse_hard_negative": {
                        "enabled": bool(float(coarse_hard_negative_loss_weight) > 0.0),
                        "loss_weight": float(coarse_hard_negative_loss_weight),
                        "margin": float(coarse_hard_negative_margin),
                        "prior_power": float(coarse_hard_negative_prior_power),
                        "evidence_weight": float(coarse_hard_negative_evidence_weight),
                        "coarse_prior_is_visual_model_input": False,
                    },
                    "hidden_dim": int(hidden_dim),
                    "heads": int(heads),
                    "dropout": float(dropout),
                    "target_audit": target_audit,
                    **_architecture_evidence_metadata(str(architecture)),
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
        if (
            bool(per_scale_view_logits) != bool(view_log_probabilities)
            or bool(per_scale_view_logits) != bool(candidate_log_likelihood_ratios)
            or bool(per_scale_view_logits) != bool(null_log_likelihood_ratios)
        ):
            raise RuntimeError("bidirectional context-attention detail tensors are incomplete")
        if per_scale_view_logits and (
            len(per_scale_view_logits) != len(families)
            or len(view_log_probabilities) != len(families)
            or len(candidate_log_likelihood_ratios) != len(families)
            or len(null_log_likelihood_ratios) != len(families)
        ):
            raise RuntimeError("bidirectional context-attention detail tensors differ from family count")
        is_bidirectional = str(architecture) in BIDIRECTIONAL_ARCHITECTURES
        if is_bidirectional != bool(per_scale_view_logits):
            raise RuntimeError("context-attention architecture and prediction details differ")
        identity_detail_lists = (
            identity_probabilities,
            identity_null_probabilities,
            identity_per_view_logits,
            identity_per_scale_view_logits,
            identity_view_log_probabilities,
            identity_candidate_log_likelihood_ratios,
            identity_null_log_likelihood_ratios,
        )
        if any(identity_detail_lists) and not all(identity_detail_lists):
            raise RuntimeError("dual identity prediction detail tensors are incomplete")
        if uses_dual_identity_head != bool(identity_probabilities):
            raise RuntimeError("V5 architecture and identity prediction details differ")
        if identity_probabilities and any(
            len(values) != len(families) for values in identity_detail_lists
        ):
            raise RuntimeError("dual identity prediction detail tensors differ from family count")
        prediction_metadata = {
            "format": PREDICTION_ARTIFACT_FORMAT,
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used_for_prediction": False,
            "training_supervision_split": "train",
            "registered_identity_radius_px": float(registered_identity_radius_px),
            "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
            "supervision_mode": mode,
            "training_objective": target_audit["training_objective"],
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
            "probability_semantics": probability_semantics,
            "identity_probability_semantics": (
                EXACT_IDENTITY_PROBABILITY_SEMANTICS if uses_dual_identity_head else None
            ),
            "identity_candidate_probability_role": (
                "strict_registered_track_identity_or_explicit_null_side_head"
                if uses_dual_identity_head
                else None
            ),
            "identity_candidate_probability_allowed_for_pnp_overlay": False,
            "candidate_probability_role": probability_role,
            "base_prior_residual": True,
            "architecture": architecture_name,
            "architecture_id": str(architecture),
            "family_profile": str(family_profile),
            "position_encoding": str(position_encoding),
            "hidden_dim": int(hidden_dim),
            "heads": int(heads),
            "dropout": float(dropout),
            "explicit_null_train_supervision": bool(is_bidirectional),
            "null_likelihood": _null_likelihood_name(str(architecture)),
            "coarse_hard_negative": {
                "enabled": bool(float(coarse_hard_negative_loss_weight) > 0.0),
                "loss_weight": float(coarse_hard_negative_loss_weight),
                "margin": float(coarse_hard_negative_margin),
                "prior_power": float(coarse_hard_negative_prior_power),
                "evidence_weight": float(coarse_hard_negative_evidence_weight),
                "coarse_prior_is_visual_model_input": False,
            },
            "source_feature_protocol": {
                "image_retrieval_or_submap_used": False,
                "whole_image_summary_or_global_used": bool(is_bidirectional),
                "soft_global_context_factor_used": bool(is_bidirectional),
                "global_context_hard_retrieval_or_candidate_reselection": False,
                "candidate_anchor_conditioned_spatial_grid_only": not bool(is_bidirectional),
                "candidate_conditioned_full_image_region_tokens": bool(is_bidirectional),
                "absolute_phase_coordinates": bool(
                    str(position_encoding) == ABSOLUTE_DUAL_FRAME_POSITION_ENCODING
                ),
                "support_view_selection": layout_metadata.get("support_view_selection"),
                "context_only_center_mask_radius": contract["context_only_center_mask"]["radius"],
                "source_scales": contract["source_scales"],
            },
            **_architecture_evidence_metadata(str(architecture)),
        }
        prediction_path = Path(output_dir) / "predictions_inference_only.npz"
        prediction_arrays: dict[str, Any] = {
            "source_row_indices": rows,
            "query_ids": np.asarray(layout["query_ids"]).astype(str),
            "split_names": np.asarray(layout["split_names"]).astype(str),
            "candidate_track_ids": candidate_tracks,
            "candidate_view_valid": np.asarray(layout["candidate_view_valid"], dtype=bool),
            "family_names": np.asarray(families, dtype=np.str_),
            "candidate_probabilities": probability_tensor,
            "null_probabilities": null_tensor,
            "per_view_logits": view_tensor,
            "metadata_json": np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
        }
        if per_scale_view_logits:
            prediction_arrays["per_scale_view_logits"] = np.stack(
                per_scale_view_logits, axis=0
            )
            prediction_arrays["view_log_probabilities"] = np.stack(
                view_log_probabilities, axis=0
            )
            prediction_arrays["candidate_log_likelihood_ratios"] = np.stack(
                candidate_log_likelihood_ratios, axis=0
            )
            prediction_arrays["null_log_likelihood_ratios"] = np.stack(
                null_log_likelihood_ratios, axis=0
            )
        if identity_probabilities:
            prediction_arrays["identity_candidate_probabilities"] = np.stack(
                identity_probabilities, axis=0
            )
            prediction_arrays["identity_null_probabilities"] = np.stack(
                identity_null_probabilities, axis=0
            )
            prediction_arrays["identity_view_logits"] = np.stack(
                identity_per_view_logits, axis=0
            )
            prediction_arrays["identity_per_scale_view_logits"] = np.stack(
                identity_per_scale_view_logits, axis=0
            )
            prediction_arrays["identity_view_log_probabilities"] = np.stack(
                identity_view_log_probabilities, axis=0
            )
            prediction_arrays["identity_candidate_log_likelihood_ratios"] = np.stack(
                identity_candidate_log_likelihood_ratios, axis=0
            )
            prediction_arrays["identity_null_log_likelihood_ratios"] = np.stack(
                identity_null_log_likelihood_ratios, axis=0
            )
        np.savez_compressed(prediction_path, **prediction_arrays)
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
                "probability_semantics": probability_semantics,
                "identity_candidate_probability_allowed_for_pnp_overlay": False,
                "candidate_probability_role": probability_role,
                "base_prior_residual": True,
                "architecture": architecture_name,
                "architecture_id": str(architecture),
                "family_profile": str(family_profile),
                "position_encoding": str(position_encoding),
                "position_only_control": _is_position_control_family(family),
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
                "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
                "supervision_mode": mode,
                "training_objective": target_audit["training_objective"],
                "explicit_null_train_supervision": bool(is_bidirectional),
                "null_likelihood": _null_likelihood_name(str(architecture)),
                "coarse_hard_negative": {
                    "enabled": bool(float(coarse_hard_negative_loss_weight) > 0.0),
                    "loss_weight": float(coarse_hard_negative_loss_weight),
                    "margin": float(coarse_hard_negative_margin),
                    "prior_power": float(coarse_hard_negative_prior_power),
                    "evidence_weight": float(coarse_hard_negative_evidence_weight),
                    "coarse_prior_is_visual_model_input": False,
                },
                "validation_or_test_labels_used_by_fit": False,
                "replaced_source_row_count": int(len(rows)),
                "replaced_source_rows_sha256": _array_sha256_short(rows),
                "audit": audit,
                **_architecture_evidence_metadata(str(architecture)),
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
                "geometric_positive_threshold_px": float(geometric_positive_threshold_px),
                "supervision_mode": mode,
                "training_objective": target_audit["training_objective"],
                "probability_semantics": probability_semantics,
                "identity_probability_semantics": (
                    EXACT_IDENTITY_PROBABILITY_SEMANTICS
                    if uses_dual_identity_head
                    else None
                ),
                "identity_candidate_probability_allowed_for_pnp_overlay": False,
                "base_prior_residual": True,
                "validation_or_test_labels_used_by_fit": False,
                "test_used_for_model_selection": False,
                "image_retrieval": False,
                "render": False,
                "fixed_support_view_log_mixture": True,
                "context_only_masks_anchor": not bool(is_bidirectional),
                "candidate_conditioned_full_image_region_tokens": bool(is_bidirectional),
                "architecture": architecture_name,
                "architecture_id": str(architecture),
                "explicit_null_train_supervision": bool(is_bidirectional),
                "null_likelihood": _null_likelihood_name(str(architecture)),
                "coarse_hard_negative": {
                    "enabled": bool(float(coarse_hard_negative_loss_weight) > 0.0),
                    "loss_weight": float(coarse_hard_negative_loss_weight),
                    "margin": float(coarse_hard_negative_margin),
                    "prior_power": float(coarse_hard_negative_prior_power),
                    "evidence_weight": float(coarse_hard_negative_evidence_weight),
                    "coarse_prior_is_visual_model_input": False,
                },
                "family_profile": str(family_profile),
                "position_encoding": str(position_encoding),
                **_architecture_evidence_metadata(str(architecture)),
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
        architecture=str(args.architecture),
        supervision_mode=str(args.supervision_mode),
        geometric_positive_threshold_px=float(args.geometric_positive_threshold_px),
        geometric_train_targets_path=(
            None
            if args.geometric_train_targets is None
            else Path(args.geometric_train_targets)
        ),
        exact_identity_train_targets_path=(
            None
            if args.exact_identity_train_targets is None
            else Path(args.exact_identity_train_targets)
        ),
        exact_identity_auxiliary_loss_weight=float(
            args.exact_identity_auxiliary_loss_weight
        ),
        coarse_hard_negative_loss_weight=float(args.coarse_hard_negative_loss_weight),
        coarse_hard_negative_margin=float(args.coarse_hard_negative_margin),
        coarse_hard_negative_prior_power=float(args.coarse_hard_negative_prior_power),
        coarse_hard_negative_evidence_weight=float(args.coarse_hard_negative_evidence_weight),
    )
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
