"""Freeze and audit wider RGB phase evidence on current P1 hard repeats.

The visual forward consumes only the immutable target-free P1 layout, real RGB
patches, a frozen FPN, and fixed candidate/support slots.  It exports a local
and a wider physical per-view spatial density before any train-only target is
loaded.  Rank zero subsequently joins coherent-repeat projection offsets and
compares correct versus strongest coherent-wrong candidates with support
appearance and all-visual-content-zero controls.

This is a train-only frozen-probe gate.  It never trains a scalar head, runs
PnP, scores validation/test, or promotes a pose.  A pass only authorizes a
later minimal correct-vs-coherent-wrong LLR calibration for the surviving
physical context scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as distributed


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.audit_p1_candidate_edge_representation_crossfit import (
    _probe_gate,
    _profile_summary,
    aggregate_hard_pose_group_gaps,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _write_json_atomically,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    HardRepeatQueryTargets,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _slice_runtime,
    build_hard_repeat_query_targets,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_multiscale_rgb_phase import (
    CANDIDATE_MULTISCALE_RGB_PHASE_FORMAT,
    RGBPhaseDensity,
    RGBPhaseScale,
    extract_rgb_phase_density,
    resolve_rgb_phase_scales,
    selected_candidate_rgb_phase_interaction_log_likelihood_ratio,
    selected_candidate_rgb_phase_log_likelihood_ratio,
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
    permute_runtime_support_appearance,
    runtime_from_target_free_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
)
from feature_extract.vfm.measurement_v1.rgb_patch_measurement_branch import (
    TexturePatchEncoder,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


AUDIT_FORMAT = "p1_multiscale_rgb_phase_frozen_probe_audit_v4"
FEATURE_FORMAT = "p1_multiscale_rgb_phase_target_free_feature_v4"
BRANCHES = (
    "normal",
    "support_permuted",
    "query_zero_support_real",
    "query_zero_support_permuted",
    "query_real_support_zero",
    "position_only",
)


@dataclass(frozen=True)
class RGBPhaseQueryFeatures:
    """One target-free visual-forward shard for one immutable P1 query."""

    query_id: str
    source_point_ids: np.ndarray
    candidate_view_weights: np.ndarray
    log_probabilities: Mapping[str, Mapping[str, np.ndarray]]
    edge_usable: Mapping[str, Mapping[str, np.ndarray]]

    def __post_init__(self) -> None:
        query_id = str(self.query_id)
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        weights = np.asarray(self.candidate_view_weights, dtype=np.float32)
        probabilities = {
            str(branch): {
                str(scale): np.asarray(values, dtype=np.float32)
                for scale, values in collection.items()
            }
            for branch, collection in self.log_probabilities.items()
        }
        usable = {
            str(branch): {
                str(scale): np.asarray(values, dtype=bool)
                for scale, values in collection.items()
            }
            for branch, collection in self.edge_usable.items()
        }
        if (
            not query_id
            or len(source_ids) == 0
            or len(np.unique(source_ids)) != len(source_ids)
            or weights.ndim != 3
            or weights.shape[0] != len(source_ids)
            or weights.shape[1] == 0
            or weights.shape[2] == 0
            or np.any(weights < 0.0)
            or not np.isfinite(weights).all()
            or set(probabilities) != set(BRANCHES)
            or set(usable) != set(BRANCHES)
        ):
            raise ValueError("RGB phase query feature arrays are invalid")
        scale_names: set[str] | None = None
        for branch in BRANCHES:
            branch_scales = set(probabilities[branch])
            if scale_names is None:
                scale_names = branch_scales
            if (
                not branch_scales
                or branch_scales != scale_names
                or set(usable[branch]) != branch_scales
            ):
                raise ValueError("RGB phase query feature scale set is inconsistent")
            for scale in branch_scales:
                values = probabilities[branch][scale]
                mask = usable[branch][scale]
                if (
                    values.ndim != 4
                    or values.shape[:3] != weights.shape
                    or values.shape[3] == 0
                    or mask.shape != weights.shape
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("RGB phase query feature density is invalid")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "candidate_view_weights", weights)
        object.__setattr__(self, "log_probabilities", probabilities)
        object.__setattr__(self, "edge_usable", usable)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texture-checkpoint", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument(
        "--edge-encoder-reference-layout",
        default="",
        help=(
            "Optional target-free layout on which the frozen per-edge texture encoder was "
            "trained. It is accepted only for a strict prefix-preserving support-view "
            "sweep; the encoder never receives a view-set aggregate."
        ),
    )
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--hard-repeat-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-chunk-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--max-abs-log-likelihood-ratio", type=float, default=6.0)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument("--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8")
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--minimum-points-per-pose", type=int, default=4)
    parser.add_argument("--minimum-eligible-query-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--catastrophic-gap-threshold", type=float, default=-0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--reuse-training-features", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _safe_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _feature_filename(query_id: str) -> str:
    value = str(query_id)
    if not value:
        raise ValueError("RGB phase query ID is empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20] + ".npz"


def _save_npz_atomically(path: Path, **arrays: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)


def _load_json_scalar(data: Mapping[str, np.ndarray], field: str) -> Mapping[str, object]:
    try:
        value = json.loads(str(np.asarray(data[field]).item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("RGB phase feature metadata is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError("RGB phase feature metadata is not an object")
    return dict(value)


def _scale_metadata(scales: Sequence[RGBPhaseScale]) -> list[dict[str, object]]:
    return [
        {
            "name": scale.name,
            "search_radius_px": float(scale.search_radius_px),
            "context_radius_px": float(scale.context_radius_px),
            "step_px": float(scale.step_px),
            "patch_side": int(scale.patch_side),
            "offset_count": int(scale.offset_count),
        }
        for scale in scales
    ]


def _feature_lineage(
    *,
    layout_path: Path,
    texture_checkpoint: Path,
    source_image_manifest_sha256: str,
    rgb_coordinate_bridge: Mapping[str, object],
    layout: CandidatePoseRGBSpatialLayout,
    scales: Sequence[RGBPhaseScale],
    temperature: float,
    max_abs_log_likelihood_ratio: float,
) -> dict[str, object]:
    return {
        "layout_sha256": file_sha256_short(layout_path),
        "texture_checkpoint_sha256": file_sha256_short(texture_checkpoint),
        "source_image_manifest_sha256": str(source_image_manifest_sha256),
        "rgb_coordinate_bridge": dict(rgb_coordinate_bridge),
        "candidate_count": int(layout.candidate_count),
        "support_view_count": int(layout.support_view_count),
        "scales": _scale_metadata(scales),
        "temperature": float(temperature),
        "density_normalization": (
            "per_view_conditional_log_softmax_over_fixed_offset_grid_with_"
            "fixed_nondustbin_half_mass_equivalent_llr_v2"
        ),
        "max_abs_log_likelihood_ratio": float(max_abs_log_likelihood_ratio),
        "view_marginalization": "fixed_maplet_support_mass_with_neutral_missing_view_v1",
        "visual_content_controls": {
            "normal": "real_query_and_fixed_support_rgb_v1",
            "support_permuted": "fixed_support_slot_derangement_v1",
            "query_zero_support_real": "zero_query_rgb_with_fixed_real_support_rgb_v1",
            "query_zero_support_permuted": "zero_query_rgb_with_deranged_support_rgb_v1",
            "query_real_support_zero": "real_query_rgb_with_zero_support_rgb_v1",
            "position_only": "all_rgb_patch_values_zero_with_same_geometry_v1",
        },
    }


def _feature_metadata(*, query_id: str, lineage: Mapping[str, object]) -> dict[str, object]:
    return {
        "format": FEATURE_FORMAT,
        "query_id": str(query_id),
        "contains_target_fields": False,
        "runtime_scorer_must_not_load_this_artifact": True,
        "target_join_after_visual_inference": True,
        "pose_or_residual_serialized": False,
        "no_render": True,
        "no_image_retrieval_or_submap": True,
        "lineage": dict(lineage),
    }


def _write_training_feature(
    *, path: Path, features: RGBPhaseQueryFeatures, lineage: Mapping[str, object]
) -> None:
    arrays: dict[str, object] = {
        "source_point_ids": np.asarray(features.source_point_ids, dtype=np.int64),
        "candidate_view_weights": np.asarray(features.candidate_view_weights, dtype=np.float32),
        "metadata_json": np.asarray(_safe_json(_feature_metadata(query_id=features.query_id, lineage=lineage))),
    }
    for branch in BRANCHES:
        for scale, values in features.log_probabilities[branch].items():
            arrays[f"{branch}_{scale}_log_probabilities"] = np.asarray(values, dtype=np.float16)
            arrays[f"{branch}_{scale}_edge_usable"] = np.asarray(
                features.edge_usable[branch][scale], dtype=bool
            )
    _save_npz_atomically(path, **arrays)


def _load_training_feature(
    *,
    path: Path,
    expected_query_id: str,
    expected_lineage: Mapping[str, object],
    scales: Sequence[RGBPhaseScale],
) -> RGBPhaseQueryFeatures:
    names = tuple(scale.name for scale in scales)
    required = {
        "source_point_ids",
        "candidate_view_weights",
        "metadata_json",
        *(f"{branch}_{name}_log_probabilities" for branch in BRANCHES for name in names),
        *(f"{branch}_{name}_edge_usable" for branch in BRANCHES for name in names),
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"RGB phase feature lacks {sorted(missing)}")
        metadata = _load_json_scalar(data, "metadata_json")
        if (
            metadata.get("format") != FEATURE_FORMAT
            or metadata.get("query_id") != str(expected_query_id)
            or metadata.get("contains_target_fields") is not False
            or metadata.get("runtime_scorer_must_not_load_this_artifact") is not True
            or metadata.get("target_join_after_visual_inference") is not True
            or metadata.get("pose_or_residual_serialized") is not False
            or metadata.get("no_render") is not True
            or metadata.get("no_image_retrieval_or_submap") is not True
            or metadata.get("lineage") != dict(expected_lineage)
        ):
            raise ValueError("RGB phase feature contract differs from this audit")
        return RGBPhaseQueryFeatures(
            query_id=str(expected_query_id),
            source_point_ids=np.asarray(data["source_point_ids"], dtype=np.int64),
            candidate_view_weights=np.asarray(data["candidate_view_weights"], dtype=np.float32),
            log_probabilities={
                branch: {
                    name: np.asarray(data[f"{branch}_{name}_log_probabilities"], dtype=np.float32)
                    for name in names
                }
                for branch in BRANCHES
            },
            edge_usable={
                branch: {
                    name: np.asarray(data[f"{branch}_{name}_edge_usable"], dtype=bool)
                    for name in names
                }
                for branch in BRANCHES
            },
        )


def _load_frozen_texture_encoder(
    *,
    path: Path,
    layout_path: Path,
    layout: CandidatePoseRGBSpatialLayout,
    device: torch.device,
    runtime_layout: CandidatePoseRGBSpatialLayout | None = None,
) -> tuple[TexturePatchEncoder, dict[str, object]]:
    """Load a target-free per-edge FPN state from the gate-checked RGB branch.

    ``layout`` is deliberately the exact checkpoint layout.  A caller may
    provide a wider ``runtime_layout`` only after the support-view prefix
    contract below has established that each existing candidate/support edge
    is identical.  ``TexturePatchEncoder`` is evaluated independently per
    edge, so this transfers no learned cross-view aggregation.
    """

    try:
        checkpoint = torch.load(Path(path), map_location="cpu")
    except (OSError, RuntimeError) as error:
        raise ValueError("RGB phase texture checkpoint is unreadable") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("RGB phase texture checkpoint is invalid")
    metadata = checkpoint.get("metadata")
    state = checkpoint.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise ValueError("RGB phase texture checkpoint lacks metadata or state")
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    training = metadata.get("training")
    if (
        metadata.get("format") != "candidate_pose_rgb_spatial_likelihood_checkpoint_v1"
        or metadata.get("model_format") != "candidate_pose_rgb_spatial_likelihood_v2"
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("fixed_global_topl") is not True
        or metadata.get("projection_after_network_only") is not True
        or not isinstance(config, Mapping)
        or not isinstance(lineage, Mapping)
        or not isinstance(training, Mapping)
        or config.get("rgb_cost_volume_only") is not True
        or int(metadata.get("fixed_candidate_top_k", 0)) != int(layout.candidate_count)
        or int(metadata.get("fixed_support_view_count", 0)) != int(layout.support_view_count)
        or str(lineage.get("layout_sha256", "")) != file_sha256_short(layout_path)
    ):
        raise ValueError("RGB phase texture checkpoint contract is incompatible")
    checkpoint_layout_sha = str(lineage.get("layout_sha256", ""))
    if not checkpoint_layout_sha:
        raise ValueError("RGB phase texture checkpoint lacks layout lineage")
    feature_dim = int(config.get("texture_feature_dim", 0))
    hidden_dim = int(config.get("hidden_dim", 0))
    if feature_dim <= 0 or hidden_dim <= 0:
        raise ValueError("RGB phase texture checkpoint architecture is invalid")
    hard_repeat = training.get("coherent_hard_repeat_edge")
    inner = training.get("inner_validation")
    inner_gate = inner.get("gate") if isinstance(inner, Mapping) else None
    try:
        inner_fold_count = int(inner.get("fold_count", 0)) if isinstance(inner, Mapping) else 0
        inner_fold_index = int(inner.get("fold_index", -1)) if isinstance(inner, Mapping) else -1
        selected_epoch = int(inner.get("selected_epoch", 0)) if isinstance(inner, Mapping) else 0
    except (TypeError, ValueError) as error:
        raise ValueError("RGB phase texture checkpoint inner-fold metadata is invalid") from error
    if (
        not isinstance(hard_repeat, Mapping)
        or hard_repeat.get("enabled") is not True
        or not isinstance(inner_gate, Mapping)
        or inner_gate.get("hard_repeat_passed") is not True
        or inner_fold_count < 2
        or not 0 <= inner_fold_index < inner_fold_count
        or selected_epoch < 1
    ):
        raise ValueError("RGB phase texture checkpoint lacks its required local hard-repeat gate")
    encoder = TexturePatchEncoder(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        input_mode="rgb_graygrad",
        encoder_arch="fpn",
    )
    texture_state = {
        str(name)[len("texture_encoder.") :]: value
        for name, value in state.items()
        if str(name).startswith("texture_encoder.")
    }
    if set(texture_state) != set(encoder.state_dict()):
        raise ValueError("RGB phase texture checkpoint FPN state is incomplete")
    encoder.load_state_dict(texture_state, strict=True)
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    active_layout = layout if runtime_layout is None else runtime_layout
    support_view_sweep = active_layout is not layout
    if support_view_sweep:
        _validate_edge_encoder_support_view_sweep(
            checkpoint_layout=layout,
            runtime_layout=active_layout,
        )
    return encoder, {
        "checkpoint_layout_sha256": checkpoint_layout_sha,
        "checkpoint_support_view_count": int(layout.support_view_count),
        "runtime_support_view_count": int(active_layout.support_view_count),
        "support_view_sweep_edge_encoder_only": bool(support_view_sweep),
        "texture_feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "local_hard_repeat_gate_passed": True,
        "checkpoint_inner_fold_count": inner_fold_count,
        "checkpoint_inner_fold_index": inner_fold_index,
        "checkpoint_selected_epoch": selected_epoch,
    }


_SUPPORT_VIEW_SWEEP_INVARIANT_ARRAYS = (
    "source_point_ids",
    "query_ids",
    "split_names",
    "xy",
    "point_sources",
    "candidate_track_ids",
    "candidate_bank_rows",
    "candidate_coarse_similarities",
    "candidate_prior_probabilities",
    "null_probabilities",
)
_SUPPORT_VIEW_SWEEP_METADATA_KEYS = (
    "candidate_set",
    "candidate_top_k",
    "candidate_prior_semantics",
    "projection_space_id",
    "descriptor_space_id",
    "support_view_selection",
    "support_coordinate_source",
    "image_retrieval_or_submap_used",
    "render",
    "candidate_reselection",
)


def _validate_edge_encoder_support_view_sweep(
    *,
    checkpoint_layout: CandidatePoseRGBSpatialLayout,
    runtime_layout: CandidatePoseRGBSpatialLayout,
) -> None:
    """Allow only a prefix-preserving, target-free support-view expansion.

    The frozen FPN predicts one query/support edge density at a time.  It can
    therefore be reused to *measure* extra fixed support observations without
    silently treating a different candidate layout as the checkpoint's
    training distribution.  All query/candidate fields must be byte-identical
    and the checkpoint support slots must be the ordered prefix of the wider
    coverage-ranked slots.  The new view weights are intentionally not
    compared: they are renormalized over the already-fixed wider view set.
    """

    if (
        int(runtime_layout.candidate_count) != int(checkpoint_layout.candidate_count)
        or int(runtime_layout.support_view_count) <= int(checkpoint_layout.support_view_count)
    ):
        raise ValueError("RGB phase support-view sweep does not widen the checkpoint layout")
    for field in _SUPPORT_VIEW_SWEEP_INVARIANT_ARRAYS:
        if not np.array_equal(
            np.asarray(getattr(runtime_layout, field)),
            np.asarray(getattr(checkpoint_layout, field)),
        ):
            raise ValueError(f"RGB phase support-view sweep changes {field}")
    for key in _SUPPORT_VIEW_SWEEP_METADATA_KEYS:
        if runtime_layout.metadata.get(key) != checkpoint_layout.metadata.get(key):
            raise ValueError(f"RGB phase support-view sweep changes metadata {key}")
    prefix = int(checkpoint_layout.support_view_count)
    for field in (
        "support_image_ids",
        "support_xy",
        "support_view_valid",
        "support_coverage_counts",
    ):
        expected = np.asarray(getattr(checkpoint_layout, field))
        # The support-view axis is always axis 2.  ``support_xy`` retains a
        # trailing coordinate axis, so ellipsis slicing would incorrectly
        # truncate x/y instead of the coverage-ranked view prefix.
        observed = np.asarray(getattr(runtime_layout, field))[:, :, :prefix]
        if not np.array_equal(observed, expected):
            raise ValueError(f"RGB phase support-view sweep changes ordered {field} prefix")


def _train_query_rows_from_target_free_layout(
    layout: CandidatePoseRGBSpatialLayout,
) -> dict[str, np.ndarray]:
    query_ids = np.asarray(layout.query_ids).astype(str)
    splits = np.asarray(layout.split_names).astype(str)
    output: dict[str, np.ndarray] = {}
    for query_id in sorted(set(query_ids[splits == "train"].tolist())):
        rows = np.flatnonzero((query_ids == str(query_id)) & (splits == "train")).astype(np.int64)
        if len(rows) != 32:
            raise ValueError("RGB phase frozen P1 layout must expose exactly 32 target-free train rows")
        output[str(query_id)] = rows
    if not output:
        raise ValueError("RGB phase frozen P1 layout has no train query")
    return output


def _rgb_edge_usable(
    *, runtime: CandidatePoseRGBSpatialRuntime, image_sizes: np.ndarray, scale: RGBPhaseScale
) -> torch.Tensor:
    active = runtime.to("cpu")
    sizes = torch.as_tensor(image_sizes, dtype=torch.float32)
    radius = float(scale.search_radius_px + scale.context_radius_px)
    query_sizes = sizes.index_select(0, active.query_image_indices)
    query_valid = (
        (active.query_xy[:, 0] >= radius)
        & (active.query_xy[:, 1] >= radius)
        & (active.query_xy[:, 0] <= query_sizes[:, 0] - 1.0 - radius)
        & (active.query_xy[:, 1] <= query_sizes[:, 1] - 1.0 - radius)
    )
    support_sizes = sizes.index_select(0, active.support_image_indices.reshape(-1)).reshape(
        *active.support_image_indices.shape, 2
    )
    support_valid = (
        (active.support_xy[..., 0] >= radius)
        & (active.support_xy[..., 1] >= radius)
        & (active.support_xy[..., 0] <= support_sizes[..., 0] - 1.0 - radius)
        & (active.support_xy[..., 1] <= support_sizes[..., 1] - 1.0 - radius)
    )
    return (
        active.support_view_valid
        & query_valid[:, None, None]
        & support_valid
    )


@torch.inference_mode()
def _forward_query_target_free(
    *,
    query_id: str,
    layout_rows: np.ndarray,
    complete_runtime: CandidatePoseRGBSpatialRuntime,
    layout: CandidatePoseRGBSpatialLayout,
    image_ids: np.ndarray,
    image_sizes: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    texture_encoder: TexturePatchEncoder,
    cache: TensorImageLRUCache,
    device: torch.device,
    scales: Sequence[RGBPhaseScale],
    edge_chunk_size: int,
    temperature: float,
    amp_enabled: bool,
    permutation_shift: int,
) -> RGBPhaseQueryFeatures:
    """Emit all visual controls before any target artifact is loaded."""

    rows = np.asarray(layout_rows, dtype=np.int64).reshape(-1)
    if len(rows) != 32:
        raise ValueError("RGB phase query needs exactly 32 frozen P1 rows")
    runtime = _slice_runtime(complete_runtime, rows)
    permuted_runtime = permute_runtime_support_appearance(
        runtime, shift=int(permutation_shift)
    )
    densities: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in BRANCHES}
    usability: dict[str, dict[str, np.ndarray]] = {branch: {} for branch in BRANCHES}
    for scale in scales:
        radius = float(scale.search_radius_px + scale.context_radius_px)
        normal_query, normal_support = _crop_runtime_rgb_patches(
            runtime=runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=radius,
            step_px=float(scale.step_px),
            cache=cache,
            device=device,
        )
        _permuted_query, permuted_support = _crop_runtime_rgb_patches(
            runtime=permuted_runtime,
            image_ids=image_ids,
            image_root=image_root,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=radius,
            step_px=float(scale.step_px),
            cache=cache,
            device=device,
        )
        normal_usable = _rgb_edge_usable(
            runtime=runtime, image_sizes=image_sizes, scale=scale
        ).to(device)
        permuted_usable = _rgb_edge_usable(
            runtime=permuted_runtime, image_sizes=image_sizes, scale=scale
        ).to(device)
        branch_inputs = {
            "normal": (normal_query, normal_support, normal_usable),
            "support_permuted": (normal_query, permuted_support, permuted_usable),
            "query_zero_support_real": (
                torch.zeros_like(normal_query),
                normal_support,
                normal_usable,
            ),
            "query_zero_support_permuted": (
                torch.zeros_like(normal_query),
                permuted_support,
                permuted_usable,
            ),
            "query_real_support_zero": (
                normal_query,
                torch.zeros_like(normal_support),
                normal_usable,
            ),
            "position_only": (
                torch.zeros_like(normal_query),
                torch.zeros_like(normal_support),
                normal_usable,
            ),
        }
        for branch, (query_patches, support_patches, edge_usable) in branch_inputs.items():
            density = extract_rgb_phase_density(
                texture_encoder=texture_encoder,
                scale=scale,
                query_patches=query_patches,
                support_patches=support_patches,
                edge_usable=edge_usable,
                edge_chunk_size=int(edge_chunk_size),
                temperature=float(temperature),
                amp_enabled=bool(amp_enabled),
            )
            densities[branch][scale.name] = density.log_probabilities.cpu().numpy()
            usability[branch][scale.name] = density.edge_usable.cpu().numpy()
    return RGBPhaseQueryFeatures(
        query_id=str(query_id),
        source_point_ids=np.asarray(layout.source_point_ids[rows], dtype=np.int64),
        candidate_view_weights=runtime.candidate_view_weights.cpu().numpy(),
        log_probabilities=densities,
        edge_usable=usability,
    )


def _density_from_feature(
    *, feature: RGBPhaseQueryFeatures, branch: str, scale: RGBPhaseScale
) -> RGBPhaseDensity:
    values = torch.from_numpy(
        np.asarray(feature.log_probabilities[str(branch)][scale.name], dtype=np.float32)
    )
    # Float16 shard storage is deliberately compact.  Re-normalize the
    # target-free density after loading to eliminate quantization drift before
    # comparing any train-only projection offsets.
    values = values - torch.logsumexp(values, dim=-1, keepdim=True)
    return RGBPhaseDensity(
        scale=scale,
        log_probabilities=values,
        edge_usable=torch.from_numpy(
            np.asarray(feature.edge_usable[str(branch)][scale.name], dtype=bool)
        ),
    )


def _selected_pair_scores(
    *,
    feature: RGBPhaseQueryFeatures,
    targets: HardRepeatQueryTargets,
    scale: RGBPhaseScale,
    branch: str,
    max_abs_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if feature.query_id != str(targets.query_id):
        raise ValueError("RGB phase query and hard-repeat target IDs differ")
    lookup = {int(source_id): index for index, source_id in enumerate(feature.source_point_ids)}
    try:
        points = torch.tensor(
            [lookup[int(source_id)] for source_id in targets.source_point_ids], dtype=torch.long
        )
    except KeyError as error:
        raise ValueError("RGB phase target source is absent from target-free feature") from error
    positive = torch.from_numpy(np.asarray(targets.positive_candidate_indices, dtype=np.int64))
    negative = torch.from_numpy(np.asarray(targets.negative_candidate_indices, dtype=np.int64))
    density = _density_from_feature(feature=feature, branch=branch, scale=scale)
    weights = torch.from_numpy(np.asarray(feature.candidate_view_weights, dtype=np.float32))
    positive_score, positive_usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density,
        candidate_view_weights=weights,
        point_indices=points,
        candidate_indices=positive,
        offsets_xy=torch.from_numpy(np.asarray(targets.positive_offsets_xy, dtype=np.float32)),
        max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
    )
    negative_score, negative_usable = selected_candidate_rgb_phase_log_likelihood_ratio(
        density=density,
        candidate_view_weights=weights,
        point_indices=points,
        candidate_indices=negative,
        offsets_xy=torch.from_numpy(np.asarray(targets.negative_offsets_xy, dtype=np.float32)),
        max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
    )
    return positive_score, positive_usable, negative_score, negative_usable


def _selected_interaction_pair_scores(
    *,
    feature: RGBPhaseQueryFeatures,
    targets: HardRepeatQueryTargets,
    scale: RGBPhaseScale,
    pair_branch: str,
    query_zero_support_branch: str,
    query_support_zero_branch: str,
    zero_branch: str,
    max_abs_log_likelihood_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a target-free query/support interaction residual after forward."""

    if feature.query_id != str(targets.query_id):
        raise ValueError("RGB phase interaction query and hard-repeat target IDs differ")
    lookup = {int(source_id): index for index, source_id in enumerate(feature.source_point_ids)}
    try:
        points = torch.tensor(
            [lookup[int(source_id)] for source_id in targets.source_point_ids], dtype=torch.long
        )
    except KeyError as error:
        raise ValueError("RGB phase interaction target source is absent from feature") from error
    positive = torch.from_numpy(np.asarray(targets.positive_candidate_indices, dtype=np.int64))
    negative = torch.from_numpy(np.asarray(targets.negative_candidate_indices, dtype=np.int64))
    densities = {
        "pair": _density_from_feature(feature=feature, branch=pair_branch, scale=scale),
        "query_zero_support": _density_from_feature(
            feature=feature, branch=query_zero_support_branch, scale=scale
        ),
        "query_support_zero": _density_from_feature(
            feature=feature, branch=query_support_zero_branch, scale=scale
        ),
        "zero": _density_from_feature(feature=feature, branch=zero_branch, scale=scale),
    }
    weights = torch.from_numpy(np.asarray(feature.candidate_view_weights, dtype=np.float32))

    def score(candidate_indices: torch.Tensor, offsets_xy: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return selected_candidate_rgb_phase_interaction_log_likelihood_ratio(
            pair_density=densities["pair"],
            query_zero_support_density=densities["query_zero_support"],
            query_support_zero_density=densities["query_support_zero"],
            zero_density=densities["zero"],
            candidate_view_weights=weights,
            point_indices=points,
            candidate_indices=candidate_indices,
            offsets_xy=torch.from_numpy(np.asarray(offsets_xy, dtype=np.float32)),
            max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
        )

    positive_score, positive_usable = score(positive, targets.positive_offsets_xy)
    negative_score, negative_usable = score(negative, targets.negative_offsets_xy)
    return positive_score, positive_usable, negative_score, negative_usable


def _evaluate_scale(
    *,
    queries: Mapping[str, RGBPhaseQueryFeatures],
    hard_targets: Mapping[str, HardRepeatQueryTargets],
    scale: RGBPhaseScale,
    minimum_points: int,
    max_abs_log_likelihood_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []
    if set(queries) != set(hard_targets):
        raise ValueError("RGB phase target join query set is inconsistent")
    for query_id in sorted(queries):
        feature = queries[query_id]
        targets = hard_targets[query_id]
        branch_values: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {
            branch: _selected_pair_scores(
                feature=feature,
                targets=targets,
                scale=scale,
                branch=branch,
                max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
            )
            for branch in BRANCHES
        }
        normal_positive, normal_positive_usable, normal_negative, normal_negative_usable = branch_values["normal"]
        permuted_positive, permuted_positive_usable, permuted_negative, permuted_negative_usable = branch_values["support_permuted"]
        query_zero_positive, query_zero_positive_usable, query_zero_negative, query_zero_negative_usable = branch_values["query_zero_support_real"]
        query_zero_permuted_positive, query_zero_permuted_positive_usable, query_zero_permuted_negative, query_zero_permuted_negative_usable = branch_values["query_zero_support_permuted"]
        support_zero_positive, support_zero_positive_usable, support_zero_negative, support_zero_negative_usable = branch_values["query_real_support_zero"]
        position_positive, position_positive_usable, position_negative, position_negative_usable = branch_values["position_only"]
        common = (
            normal_positive_usable
            & normal_negative_usable
            & permuted_positive_usable
            & permuted_negative_usable
            & query_zero_positive_usable
            & query_zero_negative_usable
            & query_zero_permuted_positive_usable
            & query_zero_permuted_negative_usable
            & support_zero_positive_usable
            & support_zero_negative_usable
            & position_positive_usable
            & position_negative_usable
        )
        interaction_normal = _selected_interaction_pair_scores(
            feature=feature,
            targets=targets,
            scale=scale,
            pair_branch="normal",
            query_zero_support_branch="query_zero_support_real",
            query_support_zero_branch="query_real_support_zero",
            zero_branch="position_only",
            max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
        )
        interaction_permuted = _selected_interaction_pair_scores(
            feature=feature,
            targets=targets,
            scale=scale,
            pair_branch="support_permuted",
            query_zero_support_branch="query_zero_support_permuted",
            query_support_zero_branch="query_real_support_zero",
            zero_branch="position_only",
            max_abs_log_likelihood_ratio=float(max_abs_log_likelihood_ratio),
        )
        interaction_positive, interaction_positive_usable, interaction_negative, interaction_negative_usable = interaction_normal
        interaction_permuted_positive, interaction_permuted_positive_usable, interaction_permuted_negative, interaction_permuted_negative_usable = interaction_permuted
        interaction_common = (
            common
            & interaction_positive_usable
            & interaction_negative_usable
            & interaction_permuted_positive_usable
            & interaction_permuted_negative_usable
        )
        normal_groups, permuted_groups, position_groups, active_points = (
            aggregate_hard_pose_group_gaps(
                normal_margins=(normal_positive - normal_negative).numpy(),
                permuted_margins=(permuted_positive - permuted_negative).numpy(),
                position_margins=(position_positive - position_negative).numpy(),
                common_active=common.numpy(),
                targets=targets,
                minimum_points=int(minimum_points),
            )
        )
        normal_check, query_zero_groups, support_zero_groups, source_active_points = (
            aggregate_hard_pose_group_gaps(
                normal_margins=(normal_positive - normal_negative).numpy(),
                permuted_margins=(query_zero_positive - query_zero_negative).numpy(),
                position_margins=(support_zero_positive - support_zero_negative).numpy(),
                common_active=common.numpy(),
                targets=targets,
                minimum_points=int(minimum_points),
            )
        )
        if (
            int(source_active_points) != int(active_points)
            or not np.allclose(normal_check, normal_groups, atol=1e-7, rtol=0.0)
        ):
            raise RuntimeError("RGB phase source ablation changed normal hard-pose grouping")
        interaction_groups, interaction_permuted_groups, interaction_zero_groups, interaction_active_points = (
            aggregate_hard_pose_group_gaps(
                normal_margins=(interaction_positive - interaction_negative).numpy(),
                permuted_margins=(
                    interaction_permuted_positive - interaction_permuted_negative
                ).numpy(),
                position_margins=np.zeros(
                    len(interaction_positive), dtype=np.float64
                ),
                common_active=interaction_common.numpy(),
                targets=targets,
                minimum_points=int(minimum_points),
            )
        )
        if len(normal_groups) == 0:
            rows.append(
                {
                    "query_id": query_id,
                    "eligible": False,
                    "common_active_edge_count": int(common.sum().item()),
                    "pose_group_count": 0,
                    "active_point_count": 0,
                }
            )
            interaction_rows.append(
                {
                    "query_id": query_id,
                    "eligible": False,
                    "common_active_edge_count": int(interaction_common.sum().item()),
                    "pose_group_count": 0,
                    "active_point_count": 0,
                }
            )
            continue
        rows.append(
            {
                "query_id": query_id,
                "eligible": True,
                "common_active_edge_count": int(common.sum().item()),
                "pose_group_count": int(len(normal_groups)),
                "active_point_count": int(active_points),
                "normal_gap": float(np.mean(normal_groups)),
                "permuted_gap": float(np.mean(permuted_groups)),
                "query_zero_support_real_gap": float(np.mean(query_zero_groups)),
                "query_real_support_zero_gap": float(np.mean(support_zero_groups)),
                "position_gap": float(np.mean(position_groups)),
            }
        )
        if len(interaction_groups) == 0:
            interaction_rows.append(
                {
                    "query_id": query_id,
                    "eligible": False,
                    "common_active_edge_count": int(interaction_common.sum().item()),
                    "pose_group_count": 0,
                    "active_point_count": 0,
                }
            )
        else:
            interaction_rows.append(
                {
                    "query_id": query_id,
                    "eligible": True,
                    "common_active_edge_count": int(interaction_common.sum().item()),
                    "pose_group_count": int(len(interaction_groups)),
                    "active_point_count": int(interaction_active_points),
                    "normal_gap": float(np.mean(interaction_groups)),
                    "permuted_gap": float(np.mean(interaction_permuted_groups)),
                    "position_gap": float(np.mean(interaction_zero_groups)),
                }
            )
    return rows, interaction_rows


def _phase_profile_summary(
    *, rows: Sequence[Mapping[str, object]], catastrophic_threshold: float
) -> dict[str, object]:
    """Add source-isolation controls to the shared paired hard-pose summary."""

    summary = dict(
        _profile_summary(rows=rows, catastrophic_threshold=float(catastrophic_threshold))
    )
    eligible = [row for row in rows if bool(row.get("eligible", False))]
    if not eligible:
        summary.update(
            {
                "query_zero_support_real_control": None,
                "query_real_support_zero_control": None,
                "normal_minus_query_zero_support_real_mean_gap": None,
                "normal_minus_query_real_support_zero_mean_gap": None,
            }
        )
        return summary

    normal = np.asarray([float(row["normal_gap"]) for row in eligible], dtype=np.float64)
    query_zero = np.asarray(
        [float(row["query_zero_support_real_gap"]) for row in eligible], dtype=np.float64
    )
    support_zero = np.asarray(
        [float(row["query_real_support_zero_gap"]) for row in eligible], dtype=np.float64
    )

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_gap": float(values.mean()),
            "median_gap": float(np.median(values)),
            "p10_gap": float(np.quantile(values, 0.1)),
            "win_fraction": float(np.mean(values > 0.0)),
            "catastrophic_count": float(np.sum(values <= float(catastrophic_threshold))),
        }

    summary.update(
        {
            "query_zero_support_real_control": summarize(query_zero),
            "query_real_support_zero_control": summarize(support_zero),
            "normal_minus_query_zero_support_real_mean_gap": float((normal - query_zero).mean()),
            "normal_minus_query_real_support_zero_mean_gap": float(
                (normal - support_zero).mean()
            ),
        }
    )
    return summary


def _phase_probe_gate(
    *, summary: Mapping[str, object], total_query_count: int, args: argparse.Namespace
) -> dict[str, object]:
    """Require real query and real support content in addition to paired gates."""

    decision = dict(_probe_gate(summary=summary, total_query_count=total_query_count, args=args))
    checks = dict(decision.get("checks", {}))
    query_delta = summary.get("normal_minus_query_zero_support_real_mean_gap")
    support_delta = summary.get("normal_minus_query_real_support_zero_mean_gap")
    if query_delta is None or support_delta is None:
        checks["query_content_visual_gap"] = False
        checks["support_content_visual_gap"] = False
    else:
        checks["query_content_visual_gap"] = float(query_delta) >= float(
            args.minimum_visual_gap_delta
        )
        checks["support_content_visual_gap"] = float(support_delta) >= float(
            args.minimum_visual_gap_delta
        )
    decision["checks"] = checks
    decision["passed"] = bool(all(bool(value) for value in checks.values()))
    decision["policy"] = (
        "checkpoint-held frozen RGB phase probe; a pass authorizes only a later "
        "correct-vs-coherent-wrong LLR fit, never PnP promotion"
    )
    return decision


def _interaction_probe_gate(
    *, summary: Mapping[str, object], total_query_count: int, args: argparse.Namespace
) -> dict[str, object]:
    """Gate the source-cancelled interaction residual on held queries only."""

    decision = dict(_probe_gate(summary=summary, total_query_count=total_query_count, args=args))
    decision["policy"] = (
        "checkpoint-held frozen RGB query/support interaction probe; a pass authorizes only "
        "a later correct-vs-coherent-wrong LLR fit, never PnP promotion"
    )
    return decision


def _require_complete_checkpoint_held_subset(
    *,
    gate: Mapping[str, object],
    complete_checkpoint_subset: bool,
    held_query_ids: Sequence[str],
    held_all_query_ids: Sequence[str],
) -> dict[str, object]:
    """Prevent a partial smoke subset from ever satisfying a checkpoint gate."""

    decision = dict(gate)
    checks = dict(decision.get("checks", {}))
    checks["complete_checkpoint_held_subset"] = bool(
        complete_checkpoint_subset and set(held_query_ids) == set(held_all_query_ids)
    )
    decision["checks"] = checks
    decision["passed"] = bool(all(bool(value) for value in checks.values()))
    return decision


def _checkpoint_inner_query_split(
    *,
    all_train_query_ids: Sequence[str],
    selected_query_ids: Sequence[str],
    fold_count: int,
    fold_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]:
    """Reconstruct the source checkpoint's deterministic train/held split."""

    all_ids = tuple(sorted(set(str(query_id) for query_id in all_train_query_ids)))
    selected = tuple(sorted(set(str(query_id) for query_id in selected_query_ids)))
    count = int(fold_count)
    index = int(fold_index)
    if (
        len(all_ids) < 2
        or count < 2
        or count > len(all_ids)
        or not 0 <= index < count
        or not set(selected).issubset(set(all_ids))
    ):
        raise ValueError("RGB phase checkpoint inner-fold split is invalid")
    held_all = tuple(query_id for position, query_id in enumerate(all_ids) if position % count == index)
    held = tuple(query_id for query_id in selected if query_id in set(held_all))
    train = tuple(query_id for query_id in selected if query_id not in set(held_all))
    return held_all, held, train, set(selected) == set(all_ids)


def _validate_args(args: argparse.Namespace) -> None:
    values = (
        float(args.temperature),
        float(args.max_abs_log_likelihood_ratio),
        float(args.rgb_cache_gb),
        float(args.minimum_eligible_query_fraction),
        float(args.minimum_normal_gap),
        float(args.minimum_win_fraction),
        float(args.minimum_visual_gap_delta),
        float(args.catastrophic_gap_threshold),
    )
    if (
        int(args.edge_chunk_size) <= 0
        or int(args.permutation_control_shift) == 0
        or int(args.max_train_queries) < 0
        or int(args.minimum_points_per_pose) < 2
        or not all(math.isfinite(value) for value in values)
        or float(args.temperature) < 1.0
        or float(args.max_abs_log_likelihood_ratio) <= 0.0
        or float(args.rgb_cache_gb) <= 0.0
        or not 0.0 < float(args.minimum_eligible_query_fraction) <= 1.0
        or float(args.minimum_normal_gap) < 0.0
        or not 0.0 < float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_visual_gap_delta) < 0.0
    ):
        raise ValueError("RGB phase probe arguments are invalid")


def audit_p1_multiscale_rgb_phase(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    paths = {
        "texture_checkpoint": Path(args.texture_checkpoint),
        "layout": Path(args.rgb_spatial_layout),
        "targets": Path(args.training_targets),
        "hard_repeat": Path(args.hard_repeat_targets),
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
        "image_root": Path(args.image_root),
    }
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError("RGB phase probe input is absent")
    checkpoint_layout_path = (
        Path(args.edge_encoder_reference_layout)
        if str(args.edge_encoder_reference_layout).strip()
        else paths["layout"]
    )
    if not checkpoint_layout_path.exists():
        raise FileNotFoundError("RGB phase edge-encoder reference layout is absent")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not bool(args.force) and not bool(args.reuse_training_features):
        raise FileExistsError(f"refusing to overwrite RGB phase output: {output_dir}")
    scales = resolve_rgb_phase_scales()
    state = _initialize_distributed(str(args.device))
    try:
        if state.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "training_features").mkdir(parents=True, exist_ok=True)
        if state.enabled:
            distributed.barrier()
        # The only data read before the target-free visual forward is the
        # frozen runtime layout and RGB/context ownership table.
        layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
        if checkpoint_layout_path.resolve() == paths["layout"].resolve():
            checkpoint_layout = layout
        else:
            checkpoint_layout = load_candidate_pose_rgb_spatial_layout(checkpoint_layout_path)
            _validate_edge_encoder_support_view_sweep(
                checkpoint_layout=checkpoint_layout,
                runtime_layout=layout,
            )
        query_rows = _train_query_rows_from_target_free_layout(layout)
        all_train_query_ids = tuple(sorted(query_rows))
        query_ids = tuple(sorted(query_rows))
        if int(args.max_train_queries) > 0:
            query_ids = query_ids[: int(args.max_train_queries)]
            query_rows = {query_id: query_rows[query_id] for query_id in query_ids}
        headers = load_context_attention_source_headers(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            expected_radio_checkpoint="",
        )
        image_ids = np.asarray(headers.image_ids).astype(str)
        image_sizes = np.asarray(headers.image_sizes, dtype=np.int64)
        if image_sizes.shape != (len(image_ids), 2) or len(image_ids) == 0:
            raise ValueError("RGB phase source image table is invalid")
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("RGB phase probe requires common aligned image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=paths["image_root"], image_id=str(image_ids[0])
        )
        source_metadata = headers.metadata_by_name["radio_final"]
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=source_metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        manifest_hash = str(source_metadata.get("source_image_manifest_sha256", ""))
        if not manifest_hash:
            raise ValueError("RGB phase source cache lacks image manifest lineage")
        texture_encoder, texture_contract = _load_frozen_texture_encoder(
            path=paths["texture_checkpoint"],
            layout_path=checkpoint_layout_path,
            layout=checkpoint_layout,
            device=state.device,
            runtime_layout=layout,
        )
        layout_sha256 = file_sha256_short(paths["layout"])
        checkpoint_layout_sha256 = file_sha256_short(checkpoint_layout_path)
        if str(texture_contract["checkpoint_layout_sha256"]) != checkpoint_layout_sha256:
            raise ValueError("RGB phase texture checkpoint layout lineage is stale")
        held_all_query_ids, held_query_ids, checkpoint_train_query_ids, complete_checkpoint_subset = (
            _checkpoint_inner_query_split(
                all_train_query_ids=all_train_query_ids,
                selected_query_ids=query_ids,
                fold_count=int(texture_contract["checkpoint_inner_fold_count"]),
                fold_index=int(texture_contract["checkpoint_inner_fold_index"]),
            )
        )
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        feature_lineage = _feature_lineage(
            layout_path=paths["layout"],
            texture_checkpoint=paths["texture_checkpoint"],
            source_image_manifest_sha256=manifest_hash,
            rgb_coordinate_bridge=rgb_bridge,
            layout=layout,
            scales=scales,
            temperature=float(args.temperature),
            max_abs_log_likelihood_ratio=float(args.max_abs_log_likelihood_ratio),
        )
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)
        local_query_count = sum(
            index % state.world_size == state.rank for index in range(len(query_ids))
        )
        local_completed = 0
        if not bool(args.reuse_training_features):
            for index, query_id in enumerate(query_ids):
                if index % state.world_size != state.rank:
                    continue
                feature = _forward_query_target_free(
                    query_id=query_id,
                    layout_rows=query_rows[query_id],
                    complete_runtime=complete_runtime,
                    layout=layout,
                    image_ids=image_ids,
                    image_sizes=image_sizes,
                    image_root=paths["image_root"],
                    coordinate_image_size=coordinate_image_size,
                    rgb_image_size=rgb_image_size,
                    texture_encoder=texture_encoder,
                    cache=cache,
                    device=state.device,
                    scales=scales,
                    edge_chunk_size=int(args.edge_chunk_size),
                    temperature=float(args.temperature),
                    amp_enabled=amp_enabled,
                    permutation_shift=int(args.permutation_control_shift),
                )
                _write_training_feature(
                    path=output_dir / "training_features" / _feature_filename(query_id),
                    features=feature,
                    lineage=feature_lineage,
                )
                local_completed += 1
                if local_completed == 1 or local_completed % 2 == 0 or local_completed == local_query_count:
                    print(
                        json.dumps(
                            {
                                "stage": "frozen_multiscale_rgb_phase_feature_shard",
                                "rank": int(state.rank),
                                "completed_queries": int(local_completed),
                                "local_query_count": int(local_query_count),
                                "query_id": query_id,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        if state.enabled:
            distributed.barrier()
        result: dict[str, object] = {
            "rank": int(state.rank),
            "output_dir": str(output_dir),
            "world_size": int(state.world_size),
        }
        if state.rank == 0:
            # Targets are deliberately loaded only now, after every normal and
            # control visual forward has been serialized as target-free data.
            targets = load_candidate_pose_rgb_spatial_training_targets(paths["targets"])
            targets_sha256 = file_sha256_short(paths["targets"])
            validate_training_layout_and_targets(
                layout=checkpoint_layout,
                targets=targets,
                layout_sha256=checkpoint_layout_sha256,
            )
            hard_repeat = load_candidate_pose_rgb_spatial_hard_repeat_targets(paths["hard_repeat"])
            hard_targets_all = build_hard_repeat_query_targets(
                layout=checkpoint_layout,
                targets=targets,
                hard_repeat_targets=hard_repeat,
                layout_sha256=checkpoint_layout_sha256,
                targets_sha256=targets_sha256,
            )
            train_groups = build_train_query_groups(layout=checkpoint_layout, targets=targets)
            if tuple(sorted(train_groups)) != tuple(sorted(_train_query_rows_from_target_free_layout(layout))):
                raise ValueError("RGB phase train target query table differs from target-free P1 layout")
            hard_targets = {query_id: hard_targets_all[query_id] for query_id in query_ids}
            if set(hard_targets) != set(query_ids):
                raise ValueError("RGB phase hard-repeat target query table is incomplete")
            queries = {
                query_id: _load_training_feature(
                    path=output_dir / "training_features" / _feature_filename(query_id),
                    expected_query_id=query_id,
                    expected_lineage=feature_lineage,
                    scales=scales,
                )
                for query_id in query_ids
            }
            profiles: dict[str, object] = {}
            for scale in scales:
                rows, interaction_rows = _evaluate_scale(
                    queries=queries,
                    hard_targets=hard_targets,
                    scale=scale,
                    minimum_points=int(args.minimum_points_per_pose),
                    max_abs_log_likelihood_ratio=float(args.max_abs_log_likelihood_ratio),
                )
                summary = _phase_profile_summary(
                    rows=rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
                )
                held_rows = [row for row in rows if str(row["query_id"]) in set(held_query_ids)]
                held_summary = _phase_profile_summary(
                    rows=held_rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
                )
                held_gate = _phase_probe_gate(
                    summary=held_summary,
                    total_query_count=len(held_all_query_ids),
                    args=args,
                )
                held_gate = _require_complete_checkpoint_held_subset(
                    gate=held_gate,
                    complete_checkpoint_subset=complete_checkpoint_subset,
                    held_query_ids=held_query_ids,
                    held_all_query_ids=held_all_query_ids,
                )
                train_rows = [
                    row for row in rows if str(row["query_id"]) in set(checkpoint_train_query_ids)
                ]
                profiles[scale.name] = {
                    "scale": _scale_metadata((scale,))[0],
                    "query_rows": rows,
                    "all_train_only_diagnostic": {"summary": summary},
                    "checkpoint_heldout": {
                        "fold_count": int(texture_contract["checkpoint_inner_fold_count"]),
                        "fold_index": int(texture_contract["checkpoint_inner_fold_index"]),
                        "expected_query_ids": list(held_all_query_ids),
                        "query_ids": list(held_query_ids),
                        "summary": held_summary,
                        "gate": held_gate,
                    },
                    "checkpoint_train_diagnostic": {
                        "query_ids": list(checkpoint_train_query_ids),
                        "summary": _phase_profile_summary(
                            rows=train_rows,
                            catastrophic_threshold=float(args.catastrophic_gap_threshold),
                        ),
                    },
                    "gate": held_gate,
                }
                interaction_summary = _profile_summary(
                    rows=interaction_rows,
                    catastrophic_threshold=float(args.catastrophic_gap_threshold),
                )
                interaction_held_rows = [
                    row for row in interaction_rows if str(row["query_id"]) in set(held_query_ids)
                ]
                interaction_held_summary = _profile_summary(
                    rows=interaction_held_rows,
                    catastrophic_threshold=float(args.catastrophic_gap_threshold),
                )
                interaction_held_gate = _interaction_probe_gate(
                    summary=interaction_held_summary,
                    total_query_count=len(held_all_query_ids),
                    args=args,
                )
                interaction_held_gate = _require_complete_checkpoint_held_subset(
                    gate=interaction_held_gate,
                    complete_checkpoint_subset=complete_checkpoint_subset,
                    held_query_ids=held_query_ids,
                    held_all_query_ids=held_all_query_ids,
                )
                interaction_train_rows = [
                    row
                    for row in interaction_rows
                    if str(row["query_id"]) in set(checkpoint_train_query_ids)
                ]
                interaction_name = f"{scale.name}_interaction_residual"
                profiles[interaction_name] = {
                    "scale": _scale_metadata((scale,))[0],
                    "evidence": "frozen_query_support_pmi_style_cost_volume_interaction_residual_v1",
                    "query_rows": interaction_rows,
                    "all_train_only_diagnostic": {"summary": interaction_summary},
                    "checkpoint_heldout": {
                        "fold_count": int(texture_contract["checkpoint_inner_fold_count"]),
                        "fold_index": int(texture_contract["checkpoint_inner_fold_index"]),
                        "expected_query_ids": list(held_all_query_ids),
                        "query_ids": list(held_query_ids),
                        "summary": interaction_held_summary,
                        "gate": interaction_held_gate,
                    },
                    "checkpoint_train_diagnostic": {
                        "query_ids": list(checkpoint_train_query_ids),
                        "summary": _profile_summary(
                            rows=interaction_train_rows,
                            catastrophic_threshold=float(args.catastrophic_gap_threshold),
                        ),
                    },
                    "gate": interaction_held_gate,
                }
            passed = [name for name, value in profiles.items() if bool(value["gate"]["passed"])]
            payload: dict[str, object] = {
                "format": AUDIT_FORMAT,
                "stage": "frozen_current_p1_multiscale_rgb_phase_correct_vs_coherent_wrong_probe",
                "output_dir": str(output_dir),
                "world_size": int(state.world_size),
                "query_count": int(len(query_ids)),
                "query_ids": list(query_ids),
                "feature_lineage": feature_lineage,
                "texture_checkpoint": {
                    "path": str(paths["texture_checkpoint"]),
                    "sha256": file_sha256_short(paths["texture_checkpoint"]),
                    **texture_contract,
                },
                "target_lineage": {
                    "target_reference_layout_sha256": checkpoint_layout_sha256,
                    "runtime_layout_sha256": layout_sha256,
                    "training_targets_sha256": targets_sha256,
                    "hard_repeat_targets_sha256": file_sha256_short(paths["hard_repeat"]),
                    "target_join_after_visual_inference": True,
                },
                "profiles": profiles,
                "passed_phase_profiles": passed,
                "rgb_cache_rank0": cache.summary(),
                "protocol": {
                    "diagnostic_only": True,
                    "model_weights_updated": False,
                    "runtime_layout_target_free": True,
                    "raw_feature_artifacts_contain_targets": False,
                    "target_join_after_visual_inference": True,
                    "train_query_only": True,
                    "heldout_validation_or_test_not_run": True,
                    "pnp_or_pose_estimation_run": False,
                    "runtime_scorer_must_not_load_feature_artifacts": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "fixed_global_topl": True,
                    "fixed_support_view_mass": True,
                    "edge_encoder_cross_view_count_transfer": bool(
                        texture_contract["support_view_sweep_edge_encoder_only"]
                    ),
                    "edge_encoder_checkpoint_support_view_count": int(
                        texture_contract["checkpoint_support_view_count"]
                    ),
                    "runtime_support_view_count": int(
                        texture_contract["runtime_support_view_count"]
                    ),
                    "no_candidate_or_view_reselection_per_pose": True,
                    "out_of_window_projection": "fixed_neutral_missing_evidence_v1",
                },
            }
            _write_json_atomically(output_dir / "audit.json", payload)
            result = {
                "rank": 0,
                "output_dir": str(output_dir),
                "passed_phase_profiles": passed,
            }
        return result
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_p1_multiscale_rgb_phase(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
