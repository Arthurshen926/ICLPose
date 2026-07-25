"""Score frozen held-out hypotheses with the high-resolution RGB likelihood.

This is an inference-only bridge for
``CandidatePoseRGBSpatialLikelihood``.  It deliberately reads no target,
registered identity, residual, or query pose.  Frozen hypothesis matrices are
used only *after* the visual network has emitted candidate-specific edge
densities, where they project the fixed global top-L landmarks into the query
image.  The output remains diagnostic-only and cannot be consumed by PnP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


# ``python path/to/script.py`` does not consistently add the repository root
# to ``sys.path``.  Keep direct held-out scoring invocations self-contained,
# matching the DDP trainer's documented command form.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.eval_grouped_hypothesis_artifact import (
    load_inference_artifact_fields,
)
from feature_extract.tools.vfm.score_frozen_multiscale_candidate_pose_evidence import (
    _load_bank_xyz,
    _load_exact_hypotheses,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    FIXED_FINAL_EPOCH_SELECTION_POLICY,
    require_current_inner_gate_evaluator_manifest,
    TensorImageLRUCache,
    _assert_geometry_fixed_support_image_control,
    _crop_geometry_fixed_permuted_support_patches,
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _source_table,
    resolve_rgb_image_cache_storage_dtype,
    train_query_partition_manifest,
    validate_rgb_coordinate_bridge,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    read_colmap_cameras_binary,
    read_colmap_image_camera_ids_binary,
)
from feature_extract.vfm.localization.candidate_pose_llr import (
    CANDIDATE_POSE_LLR_SCORE_FORMAT,
    validate_target_free_pose_llr_score_metadata,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    CandidatePoseRGBSpatialRuntime,
    candidate_pose_rgb_spatial_score_component_prediction,
    permute_runtime_support_image_appearance_only,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_selector import (
    select_target_free_spatial_quota,
    selector_input_from_target_free_layout,
    target_free_selector_scores,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_sources,
)
from feature_extract.vfm.localization.mixed_verification_points import (
    MIXED_VERIFICATION_POINTS_FORMAT,
    MixedVerificationPoints,
    load_mixed_verification_points,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    project_simple_radial_torch,
)


CHECKPOINT_FORMAT = "candidate_pose_rgb_spatial_likelihood_checkpoint_v2"
SCORE_VERSION = (
    "candidate_specific_full_rgb_context_spatial_likelihood_target_free_v2"
)
_CONTEXT_INPUT_NAMES = (
    "radio_final_context_cache",
    "radio_intermediate_context_cache",
    "alike_spatial_context_cache",
)
_BASELINE_REFERENCE_FIELDS = ("poses_w2c", "chosen_for_optional_pose")
_FIXED_POINT_ARRAY_NAMES = frozenset(
    {
        "source_names",
        "verification_source_point_ids",
        "verification_point_sources",
        "verification_source_detector_rows",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
        # The frozen layout describes support ownership, while likelihood rows
        # describe frozen pose hypotheses.  Keep both normal and deranged
        # ownership arrays point-aligned rather than forcing a false row join.
        "candidate_support_image_ids",
        "scored_support_image_ids",
    }
)


@dataclass(frozen=True)
class _QueryGeometry:
    candidate_xyz: torch.Tensor
    focal_length: float
    principal_x: float
    principal_y: float
    radial_k: float
    image_width: int
    image_height: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypothesis-artifact", required=True)
    parser.add_argument("--baseline-score-artifact", required=True)
    parser.add_argument(
        "--baseline-reference-hypothesis-artifact",
        default="",
        help="optional exact-equivalent source artifact used by the baseline",
    )
    parser.add_argument("--detector-query-cache", required=True)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--fixed-candidate-prior-overlay", required=True)
    parser.add_argument("--mixed-verification-points-artifact", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hypothesis-batch-size", type=int, default=4)
    parser.add_argument("--edge-chunk-size", type=int, default=0)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument(
        "--rgb-cache-dtype", choices=("uint8", "float16", "float32"), default="uint8"
    )
    parser.add_argument(
        "--evidence-variant",
        choices=("visual", "support_descriptor_permutation_control"),
        default="visual",
        help=(
            "visual uses the fixed query/support layout; the compatibility-named "
            "control deranges only support image IDs and re-crops real RGB at the "
            "unchanged support coordinates"
        ),
    )
    parser.add_argument(
        "--score-component",
        choices=(
            "combined",
            "rgb_cost_volume",
            "rgb_cost_volume_with_dustbin",
            "learned_spatial_no_context",
            "spatial_residual",
            "context_only",
        ),
        default="combined",
        help=(
            "target-free diagnostic component selected after the shared visual "
            "forward; it never changes frozen candidates, support views, or poses"
        ),
    )
    parser.add_argument(
        "--point-selector",
        choices=("all", "checkpoint_validation"),
        default="all",
        help=(
            "fixed target-free verifier-point denominator: all retains the 192-point "
            "P1 pool, checkpoint_validation reproduces the checkpoint's declared "
            "static support-coverage/grid-quota selector"
        ),
    )
    parser.add_argument(
        "--oof-train-query",
        action="store_true",
        help=(
            "allow exactly one train split query only when the checkpoint's serialized "
            "P1 partition proves that query was its inner-validation fold; used solely "
            "to create train-only OOF calibration evidence"
        ),
    )
    parser.add_argument(
        "--query-id",
        default="",
        help=(
            "optional explicit query group selected from an immutable multi-query "
            "S0/hypothesis shard; intended for train-only OOF scoring"
        ),
    )
    parser.add_argument(
        "--hypothesis-limit",
        type=int,
        default=0,
        help="development-only frozen hypothesis prefix; zero scores all rows",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _uses_support_image_appearance_control(evidence_variant: str) -> bool:
    """Keep the legacy CLI label while making its image-only semantics explicit."""

    value = str(evidence_variant)
    if value not in {"visual", "support_descriptor_permutation_control"}:
        raise ValueError("RGB spatial evidence variant is invalid")
    return value == "support_descriptor_permutation_control"


def _input_manifest(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": file_sha256_short(path)}
        for name, path in paths.items()
    }


def _slice_layout(
    layout: CandidatePoseRGBSpatialLayout, rows: np.ndarray
) -> CandidatePoseRGBSpatialLayout:
    """Retain one held-out query without changing any frozen denominator."""

    indices = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or np.any(indices < 0)
        or np.any(indices >= layout.row_count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("RGB spatial layout query rows are invalid")
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=layout.source_point_ids[indices],
        query_ids=layout.query_ids[indices],
        split_names=layout.split_names[indices],
        xy=layout.xy[indices],
        point_sources=layout.point_sources[indices],
        candidate_track_ids=layout.candidate_track_ids[indices],
        candidate_bank_rows=layout.candidate_bank_rows[indices],
        candidate_coarse_similarities=layout.candidate_coarse_similarities[indices],
        candidate_prior_probabilities=layout.candidate_prior_probabilities[indices],
        null_probabilities=layout.null_probabilities[indices],
        support_image_ids=layout.support_image_ids[indices],
        support_xy=layout.support_xy[indices],
        support_view_valid=layout.support_view_valid[indices],
        support_view_weights=layout.support_view_weights[indices],
        support_coverage_counts=layout.support_coverage_counts[indices],
        metadata=layout.metadata,
    )


def _frozen_rgb_subset_manifest(layout: CandidatePoseRGBSpatialLayout) -> Mapping[str, object] | None:
    """Return a strictly validated target-free RGB subset manifest, if present."""

    selector = layout.metadata.get("frozen_rgb_selector")
    required_exclusions = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if not isinstance(selector, Mapping):
        return None
    exclusions = set(str(value) for value in selector.get("selection_excludes", ()))
    if (
        selector.get("format") != "frozen_rgb_peakiness_p1_subset_layout_v1"
        or selector.get("runtime_layout_target_free") is not True
        or selector.get("selection_before_train_target_join") is not True
        or not required_exclusions.issubset(exclusions)
    ):
        return None
    return selector


def _assert_layout_matches_mixed_points(
    *, layout: CandidatePoseRGBSpatialLayout, points: MixedVerificationPoints, points_path: Path
) -> tuple[np.ndarray, dict[str, object]]:
    """Prove the RGB layout is an exact cache or frozen target-free subset.

    The high-resolution RGB selector may reduce the formal P1 point cache to a
    fixed subset before any train target is materialized.  A held-out scorer
    must retain that denominator, but may not silently accept an arbitrary
    subset.  Each selected row is therefore joined back to a unique parent
    source-point ID and all target-free candidate fields are compared before
    model inference.
    """

    if (
        points.metadata.get("format") != MIXED_VERIFICATION_POINTS_FORMAT
        or points.metadata.get("contains_ground_truth") is not False
        or points.metadata.get("contains_target_errors") is not False
        or points.metadata.get("pose_or_ground_truth_used") is not False
        or points.metadata.get("image_retrieval_or_submap_used") is not False
        or points.metadata.get("render") is not False
        or str(layout.metadata.get("verification_points_sha256", ""))
        != file_sha256_short(Path(points_path))
    ):
        raise ValueError("RGB spatial layout does not reference target-free frozen points")
    parent_source_ids = np.asarray(points.source_point_ids, dtype=np.int64).reshape(-1)
    layout_source_ids = np.asarray(layout.source_point_ids, dtype=np.int64).reshape(-1)
    if (
        len(parent_source_ids) == 0
        or len(layout_source_ids) == 0
        or len(np.unique(parent_source_ids)) != len(parent_source_ids)
        or len(np.unique(layout_source_ids)) != len(layout_source_ids)
    ):
        raise ValueError("RGB spatial layout/source-point IDs are not unique")
    parent_positions = {int(source_id): position for position, source_id in enumerate(parent_source_ids)}
    try:
        point_rows = np.asarray(
            [parent_positions[int(source_id)] for source_id in layout_source_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("RGB spatial layout has a row absent from frozen points") from error
    exact_parent_cache = bool(
        len(layout_source_ids) == len(parent_source_ids)
        and np.array_equal(point_rows, np.arange(len(parent_source_ids), dtype=np.int64))
    )
    if len(layout_source_ids) == len(parent_source_ids) and not exact_parent_cache:
        raise ValueError("RGB spatial full-cache layout does not preserve parent point order")
    subset_manifest = None if exact_parent_cache else _frozen_rgb_subset_manifest(layout)
    if not exact_parent_cache and subset_manifest is None:
        raise ValueError(
            "RGB spatial layout is neither the exact frozen point cache nor a valid "
            "target-free frozen RGB subset"
        )
    for query_id in np.unique(layout.query_ids.astype(str)):
        query_positions = np.flatnonzero(layout.query_ids.astype(str) == str(query_id))
        if len(query_positions) and not np.all(np.diff(point_rows[query_positions]) > 0):
            raise ValueError("RGB spatial frozen subset does not preserve parent query order")
    exact = {
        "source_point_ids": (layout.source_point_ids, points.source_point_ids[point_rows]),
        "query_ids": (layout.query_ids, points.query_ids[point_rows]),
        "split_names": (layout.split_names, points.split_names[point_rows]),
        "point_sources": (layout.point_sources, points.point_sources[point_rows]),
        "candidate_track_ids": (
            layout.candidate_track_ids,
            points.candidate_track_ids[point_rows],
        ),
        "candidate_bank_rows": (layout.candidate_bank_rows, points.candidate_bank_rows[point_rows]),
    }
    for name, (left, right) in exact.items():
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError(f"RGB spatial layout differs from frozen points: {name}")
    close = {
        "xy": (layout.xy, points.xy[point_rows]),
        "candidate_coarse_similarities": (
            layout.candidate_coarse_similarities,
            points.candidate_coarse_similarities[point_rows],
        ),
        "candidate_prior_probabilities": (
            layout.candidate_prior_probabilities,
            points.candidate_prior_probabilities[point_rows],
        ),
        "null_probabilities": (layout.null_probabilities, points.null_probabilities[point_rows]),
    }
    for name, (left, right) in close.items():
        if not np.allclose(np.asarray(left), np.asarray(right), atol=1e-6, rtol=0.0):
            raise ValueError(f"RGB spatial layout differs from frozen points: {name}")
    for key in ("descriptor_space_id", "projection_space_id", "projected_landmark_bank_sha256"):
        if str(layout.metadata.get(key, "")) != str(points.metadata.get(key, "")):
            raise ValueError(f"RGB spatial layout differs from frozen points: {key}")
    return point_rows, {
        "mode": (
            "exact_formal_p1_mixed_multiscale_verification_points_v1"
            if exact_parent_cache
            else "frozen_target_free_rgb_peakiness_subset_of_formal_p1_points_v1"
        ),
        "parent_point_count": int(len(parent_source_ids)),
        "selected_point_count": int(len(layout_source_ids)),
        "frozen_rgb_selector": None if subset_manifest is None else dict(subset_manifest),
    }


def _validate_checkpoint_for_target_free_scoring(
    *,
    metadata: Mapping[str, object],
    layout: CandidatePoseRGBSpatialLayout,
    cache_hashes: Mapping[str, str],
    source_image_manifest_sha256: str,
) -> dict[str, object]:
    """Reject stale, promotable, or target-bearing checkpoints before scoring."""

    required = {
        "format": CHECKPOINT_FORMAT,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "checkpoint_contains_train_targets": False,
        "diagnostic_only": True,
        "holdout_evaluation_allowed": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "pose_or_ground_truth_used_by_runtime_scorer": False,
        "runtime_layout_is_target_free": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "fixed_support_view_count": 2,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "explicit_null": True,
        "projection_after_network_only": True,
        "out_of_window_projection_semantics": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
        "image_retrieval_or_submap_used": False,
        "render": False,
        "appearance_control_geometry_fixed": True,
        "checkpoint_selection_policy": FIXED_FINAL_EPOCH_SELECTION_POLICY,
        "inner_validation_used_for_model_selection": False,
        "train_only_inner_gate_passed": True,
    }
    if not isinstance(metadata, Mapping) or any(
        metadata.get(key) != value for key, value in required.items()
    ):
        raise ValueError("RGB spatial likelihood checkpoint violates the held-out contract")
    require_current_inner_gate_evaluator_manifest(
        metadata.get("inner_gate_evaluator_manifest"),
        subject="RGB spatial likelihood checkpoint",
    )
    excludes = set(str(value) for value in metadata.get("encoder_excludes", ()))
    required_excludes = {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }
    if not required_excludes.issubset(excludes):
        raise ValueError("RGB spatial likelihood checkpoint encoder contract is incomplete")
    source_availability = metadata.get("edge_source_availability")
    if (
        not isinstance(source_availability, Mapping)
        or source_availability.get("format") != "independent_rgb_context_union_v1"
        or source_availability.get("rgb_cost_volume")
        != "in_bounds_real_rgb_query_and_support_windows_v1"
        or source_availability.get("context_identity")
        != "all_full_map_radio_alike_crops_valid_v1"
        or source_availability.get("learned_spatial_residual_and_dustbin")
        != "rgb_and_context_intersection_only_v1"
        or source_availability.get("combined_fallback")
        != "rgb_raw_cost_volume_or_context_scalar_with_neutral_missing_peer_v1"
    ):
        raise ValueError("RGB spatial likelihood checkpoint source-availability semantics differ")
    inputs = metadata.get("inputs")
    lineage = metadata.get("lineage")
    config = metadata.get("config")
    if not isinstance(inputs, Mapping) or not isinstance(lineage, Mapping) or not isinstance(config, Mapping):
        raise ValueError("RGB spatial likelihood checkpoint metadata is incomplete")
    for name, actual_hash in cache_hashes.items():
        observed = inputs.get(name)
        if (
            not isinstance(observed, Mapping)
            or str(observed.get("sha256", "")) != str(actual_hash)
        ):
            raise ValueError(f"RGB spatial likelihood checkpoint is stale for {name}")
    if (
        str(lineage.get("descriptor_space_id", ""))
        != str(layout.metadata.get("descriptor_space_id", ""))
        or str(lineage.get("projection_space_id", ""))
        != str(layout.metadata.get("projection_space_id", ""))
        or str(lineage.get("source_image_manifest_sha256", ""))
        != str(source_image_manifest_sha256)
    ):
        raise ValueError("RGB spatial likelihood checkpoint descriptor/image lineage differs")
    if int(config.get("search_radius_px", 0)) <= 0 or int(config.get("edge_chunk_size", 0)) <= 0:
        raise ValueError("RGB spatial likelihood checkpoint geometry is invalid")
    if bool(config.get("rgb_cost_volume_only", False)):
        raise ValueError("this scorer requires the full RGB-plus-context checkpoint")
    training = metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("RGB spatial likelihood checkpoint lacks strict training lineage")
    inner_validation = training.get("inner_validation")
    support_permutation = training.get("support_permutation_contrastive")
    if not isinstance(inner_validation, Mapping) or not isinstance(
        support_permutation, Mapping
    ):
        raise ValueError("RGB spatial likelihood checkpoint lacks strict control lineage")
    selection = inner_validation.get("checkpoint_selection")
    control = inner_validation.get("support_permutation_control")
    if (
        not isinstance(selection, Mapping)
        or selection.get("policy") != FIXED_FINAL_EPOCH_SELECTION_POLICY
        or selection.get("inner_validation_used_for_model_selection") is not False
        or not isinstance(control, Mapping)
        or control.get("geometry_fixed_image_only") is not True
        or control.get("rgb_support_patch")
        != "recrop_deranged_image_at_fixed_coordinate_v2"
        or support_permutation.get("rgb_patch_derangement")
        != "recrop_deranged_support_image_at_fixed_support_coordinate_v2"
    ):
        raise ValueError("RGB spatial likelihood checkpoint control lineage is not strict")
    return dict(config)


def _load_checkpoint_model(
    *,
    path: Path,
    layout: CandidatePoseRGBSpatialLayout,
    source_tensors: Mapping[str, torch.Tensor],
    image_sizes: np.ndarray,
    cache_hashes: Mapping[str, str],
    source_image_manifest_sha256: str,
    device: torch.device,
) -> tuple[CandidatePoseRGBSpatialLikelihood, dict[str, object]]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
        raise ValueError("RGB spatial likelihood checkpoint is malformed")
    metadata = dict(payload["metadata"])
    config = _validate_checkpoint_for_target_free_scoring(
        metadata=metadata,
        layout=layout,
        cache_hashes=cache_hashes,
        source_image_manifest_sha256=source_image_manifest_sha256,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("RGB spatial likelihood checkpoint has no state dict")
    model = CandidatePoseRGBSpatialLikelihood(
        sources=source_tensors,
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        search_radius_px=float(config["search_radius_px"]),
        context_radius_px=float(config["context_radius_px"]),
        step_px=float(config["step_px"]),
        texture_feature_dim=int(config["texture_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        max_abs_context_log_ratio=float(config["max_abs_context_log_ratio"]),
        edge_chunk_size=int(config["edge_chunk_size"]),
        context_windows=resolve_candidate_pose_rgb_spatial_context_windows(
            config.get("context_windows")
        ),
        context_encoder_arch=resolve_candidate_pose_rgb_spatial_context_encoder_arch(
            str(config.get("context_encoder_arch", "conv_v1"))
        ),
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("RGB spatial likelihood checkpoint state dict is incompatible") from error
    return model.to(device).eval(), metadata


def _validate_oof_train_query_checkpoint(
    *, checkpoint_metadata: Mapping[str, object], query_id: str
) -> dict[str, object]:
    """Require an explicit serialized exclusion proof for OOF train scoring."""

    training = checkpoint_metadata.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("OOF score checkpoint lacks training lineage")
    inner_validation = training.get("inner_validation")
    if not isinstance(inner_validation, Mapping):
        raise ValueError("OOF score checkpoint lacks inner-validation lineage")
    partition = inner_validation.get("query_partition")
    if not isinstance(partition, Mapping):
        raise ValueError("OOF score checkpoint lacks serialized query partition")
    if (
        partition.get("format")
        != "candidate_pose_rgb_spatial_train_query_partition_v1"
        or partition.get("assignment")
        != "sorted_unique_query_index_modulo_fold_count_v1"
    ):
        raise ValueError("OOF score checkpoint query partition is unsupported")
    try:
        reconstructed = train_query_partition_manifest(
            all_query_ids=partition["all_train"]["query_ids"],  # type: ignore[index]
            inner_train_query_ids=partition["inner_train"]["query_ids"],  # type: ignore[index]
            inner_validation_query_ids=partition["inner_validation"]["query_ids"],  # type: ignore[index]
            fold_count=int(partition["fold_count"]),
            fold_index=int(partition["fold_index"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("OOF score checkpoint query partition is malformed") from error
    if dict(partition) != reconstructed:
        raise ValueError("OOF score checkpoint query partition digest is stale")
    inner_train = set(str(value) for value in reconstructed["inner_train"]["query_ids"])  # type: ignore[index]
    inner_validation_ids = set(
        str(value) for value in reconstructed["inner_validation"]["query_ids"]  # type: ignore[index]
    )
    if str(query_id) in inner_train or str(query_id) not in inner_validation_ids:
        raise ValueError("OOF train score query was not excluded from checkpoint fitting")
    return reconstructed


def _validate_baseline_reference_hypothesis_equivalence(
    *, current_hypothesis_path: Path, reference_hypothesis_path: Path
) -> dict[str, object]:
    """Allow a baseline bridge only for byte-identical frozen pose rows."""

    current, current_metadata = load_inference_artifact_fields(
        Path(current_hypothesis_path), _BASELINE_REFERENCE_FIELDS
    )
    reference, reference_metadata = load_inference_artifact_fields(
        Path(reference_hypothesis_path), _BASELINE_REFERENCE_FIELDS
    )
    if (
        current_metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
        or reference_metadata.get("format") != "grouped_pose_hypotheses_inference_only_v1"
        or current_metadata.get("contains_target_fields") is not False
        or reference_metadata.get("contains_target_fields") is not False
        or current_metadata.get("pose_or_ground_truth_used_for_generation") is not False
        or reference_metadata.get("pose_or_ground_truth_used_for_generation") is not False
        or current_metadata.get("inputs") != reference_metadata.get("inputs")
    ):
        raise ValueError("baseline reference artifact has different target-free inputs")

    def keyed_rows(arrays: Mapping[str, np.ndarray]) -> dict[tuple[str, str, str, int], int]:
        keys = list(
            zip(
                np.asarray(arrays["query_ids"]).astype(str).tolist(),
                np.asarray(arrays["split_names"]).astype(str).tolist(),
                np.asarray(arrays["evaluation_labels"]).astype(str).tolist(),
                np.asarray(arrays["hypothesis_indices"], dtype=np.int64).tolist(),
            )
        )
        if len(keys) == 0 or len(keys) != len(set(keys)):
            raise ValueError("baseline reference has invalid frozen row identities")
        return {key: index for index, key in enumerate(keys)}

    current_rows = keyed_rows(current)
    reference_rows = keyed_rows(reference)
    if set(current_rows) != set(reference_rows):
        raise ValueError("baseline reference has different frozen row identities")
    for key, current_row in current_rows.items():
        reference_row = reference_rows[key]
        if not np.array_equal(
            np.asarray(current["poses_w2c"])[current_row],
            np.asarray(reference["poses_w2c"])[reference_row],
        ) or bool(np.asarray(current["chosen_for_optional_pose"])[current_row]) != bool(
            np.asarray(reference["chosen_for_optional_pose"])[reference_row]
        ):
            raise ValueError("baseline reference has a different frozen pose or selector")
    return {
        "rule": "exact_target_free_row_pose_selector_equivalence_v1",
        "current_hypothesis_artifact": {
            "path": str(current_hypothesis_path),
            "sha256": file_sha256_short(Path(current_hypothesis_path)),
        },
        "reference_hypothesis_artifact": {
            "path": str(reference_hypothesis_path),
            "sha256": file_sha256_short(Path(reference_hypothesis_path)),
        },
        "validated_row_count": int(len(current_rows)),
    }


def _query_rows(
    *, layout: CandidatePoseRGBSpatialLayout, query_id: str, split_name: str
) -> np.ndarray:
    rows = np.flatnonzero(
        (layout.query_ids == str(query_id)) & (layout.split_names == str(split_name))
    ).astype(np.int64)
    if len(rows) == 0:
        raise ValueError("held-out query is absent from RGB spatial layout")
    if np.any(layout.query_ids[rows] != str(query_id)) or np.any(
        layout.split_names[rows] != str(split_name)
    ):
        raise RuntimeError("RGB spatial held-out query selection is inconsistent")
    return rows


def _checkpoint_validation_selector_positions(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    config: Mapping[str, object],
    coordinate_image_size: tuple[int, int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Reproduce the checkpoint's static target-free validation denominator.

    This is intentionally separate from learned RGB selector policies.  The
    selected checkpoint used only frozen support coverage and spatial quota,
    so the held-out path can reproduce it before crops, visual prediction, or
    pose hypotheses exist.
    """

    selector = config.get("validation_selector")
    if not isinstance(selector, Mapping) or selector.get("target_free_static_only") is not True:
        raise ValueError("RGB spatial checkpoint lacks a static validation selector")
    try:
        policy = str(selector["policy"])
        budget = int(selector["point_budget"])
        rows = int(selector["grid_rows"])
        columns = int(selector["grid_columns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RGB spatial checkpoint validation selector is invalid") from error
    if budget < 4 or budget > layout.row_count or rows <= 0 or columns <= 0:
        raise ValueError("RGB spatial checkpoint validation selector is out of range")
    selector_input = selector_input_from_target_free_layout(
        layout=layout,
        rows=np.arange(layout.row_count, dtype=np.int64),
        rgb_context_radius_px=float(config["search_radius_px"])
        + float(config["context_radius_px"]),
        coordinate_image_size=coordinate_image_size,
    )
    scores = target_free_selector_scores(selector_input=selector_input, policy=policy)
    positions = select_target_free_spatial_quota(
        selector_input=selector_input,
        quality_scores=scores,
        point_budget=budget,
        grid_rows=rows,
        grid_columns=columns,
        image_size=coordinate_image_size,
    )
    if len(positions) != budget or len(np.unique(positions)) != budget:
        raise RuntimeError("RGB spatial checkpoint selector did not retain its point budget")
    return positions, {
        "mode": "checkpoint_validation_static_target_free_selector_v1",
        "policy": policy,
        "point_budget": budget,
        "grid_rows": rows,
        "grid_columns": columns,
        "target_free_static_only": True,
    }


def _validate_query_layout_against_points(
    *,
    query_layout: CandidatePoseRGBSpatialLayout,
    points: MixedVerificationPoints,
    points_rows: np.ndarray,
) -> None:
    """Require the exact same point order as the frozen-hypothesis scorer."""

    rows = np.asarray(points_rows, dtype=np.int64).reshape(-1)
    exact = {
        "source_point_ids": (query_layout.source_point_ids, points.source_point_ids[rows]),
        "query_ids": (query_layout.query_ids, points.query_ids[rows]),
        "split_names": (query_layout.split_names, points.split_names[rows]),
        "point_sources": (query_layout.point_sources, points.point_sources[rows]),
        "candidate_track_ids": (
            query_layout.candidate_track_ids,
            points.candidate_track_ids[rows],
        ),
        "candidate_bank_rows": (
            query_layout.candidate_bank_rows,
            points.candidate_bank_rows[rows],
        ),
    }
    for name, (left, right) in exact.items():
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError(f"RGB spatial held-out rows differ from frozen hypotheses: {name}")
    for name, left, right in (
        ("xy", query_layout.xy, points.xy[rows]),
        (
            "candidate_prior_probabilities",
            query_layout.candidate_prior_probabilities,
            points.candidate_prior_probabilities[rows],
        ),
        ("null_probabilities", query_layout.null_probabilities, points.null_probabilities[rows]),
    ):
        if not np.allclose(np.asarray(left), np.asarray(right), rtol=0.0, atol=1e-6):
            raise ValueError(f"RGB spatial held-out rows differ from frozen hypotheses: {name}")


def _query_geometry(
    *,
    query_id: str,
    query_layout: CandidatePoseRGBSpatialLayout,
    bank_track_ids: np.ndarray,
    bank_xyz: np.ndarray,
    colmap_model_dir: Path,
) -> _QueryGeometry:
    rows = np.asarray(query_layout.candidate_bank_rows, dtype=np.int64)
    tracks = np.asarray(query_layout.candidate_track_ids, dtype=np.int64)
    if (
        rows.ndim != 2
        or rows.shape != tracks.shape
        or np.any(rows < 0)
        or np.any(rows >= len(bank_track_ids))
        or not np.array_equal(np.asarray(bank_track_ids, dtype=np.int64)[rows], tracks)
    ):
        raise ValueError("RGB spatial layout candidate bank rows are stale")
    cameras = read_colmap_cameras_binary(Path(colmap_model_dir) / "cameras.bin")
    camera_ids = read_colmap_image_camera_ids_binary(Path(colmap_model_dir) / "images.bin")
    camera_id = camera_ids.get(str(query_id))
    if camera_id is None:
        raise ValueError("held-out query is absent from COLMAP camera ownership")
    camera = cameras.get(int(camera_id))
    if camera is None or int(camera.model_id) != 2 or len(camera.params) != 4:
        raise ValueError("RGB spatial scoring requires SIMPLE_RADIAL query cameras")
    return _QueryGeometry(
        candidate_xyz=torch.from_numpy(np.asarray(bank_xyz, dtype=np.float32)[rows]),
        focal_length=float(camera.params[0]),
        principal_x=float(camera.params[1]),
        principal_y=float(camera.params[2]),
        radial_k=float(camera.params[3]),
        image_width=int(camera.width),
        image_height=int(camera.height),
    )


def _project_candidate_offsets(
    *,
    geometry: _QueryGeometry,
    runtime: CandidatePoseRGBSpatialRuntime,
    poses_w2c: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project frozen track xyz and express positions in query-anchor offsets."""

    poses = torch.as_tensor(poses_w2c, dtype=torch.float32, device=device)
    xyz = geometry.candidate_xyz.to(device=device, dtype=torch.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError("frozen RGB spatial hypothesis poses are invalid")
    projected, valid = project_simple_radial_torch(
        xyz.reshape(-1, 3),
        poses,
        focal_length=geometry.focal_length,
        principal_x=geometry.principal_x,
        principal_y=geometry.principal_y,
        radial_k=geometry.radial_k,
        image_width=geometry.image_width,
        image_height=geometry.image_height,
    )
    projected = projected.reshape(len(poses), *xyz.shape[:2], 2)
    valid = valid.reshape(len(poses), *xyz.shape[:2])
    anchors = runtime.query_xy.to(device=device, dtype=torch.float32)[None, :, None, :]
    return projected - anchors, valid


def _score_hypotheses(
    *,
    model: CandidatePoseRGBSpatialLikelihood,
    runtime: CandidatePoseRGBSpatialRuntime,
    prediction: object,
    geometry: _QueryGeometry,
    poses_w2c: np.ndarray,
    point_sources: np.ndarray,
    hypothesis_batch_size: int,
    edge_chunk_size: int,
    max_abs_pose_log_ratio: float,
) -> dict[str, np.ndarray]:
    """Apply arbitrary frozen poses only after one target-free visual forward."""

    poses = np.asarray(poses_w2c, dtype=np.float64)
    sources = np.asarray(point_sources).astype(str).reshape(-1)
    if (
        poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) == 0
        or len(sources) != runtime.point_count
        or int(hypothesis_batch_size) <= 0
        or int(edge_chunk_size) <= 0
        or not math.isfinite(float(max_abs_pose_log_ratio))
        or float(max_abs_pose_log_ratio) <= 0.0
    ):
        raise ValueError("RGB spatial hypothesis scoring inputs are invalid")
    source_names = np.asarray(sorted(set(sources.tolist())), dtype=np.str_)
    outputs: dict[str, list[np.ndarray]] = {
        "pose_log_likelihood_ratios": [],
        "point_log_likelihood_ratios": [],
        "point_effective_candidate_counts": [],
        "point_effective_view_masses": [],
        "source_log_likelihood_means": [],
        "source_effective_point_counts": [],
    }
    active = runtime.to(model.device)
    candidate_probability = active.candidate_probabilities.unsqueeze(0).unsqueeze(3)
    view_weights = active.candidate_view_weights.unsqueeze(0)
    for begin in range(0, len(poses), int(hypothesis_batch_size)):
        end = min(begin + int(hypothesis_batch_size), len(poses))
        offsets, valid = _project_candidate_offsets(
            geometry=geometry,
            runtime=runtime,
            poses_w2c=poses[begin:end],
            device=model.device,
        )
        with torch.autocast(device_type=model.device.type, enabled=model.device.type == "cuda"):
            score = score_candidate_pose_rgb_spatial_batch(
                runtime=runtime,
                prediction=prediction,  # type: ignore[arg-type]
                candidate_projection_offsets_xy=offsets,
                candidate_projection_valid=valid,
                missing_edge_log_likelihood_ratio=0.0,
                max_abs_log_likelihood_ratio=float(max_abs_pose_log_ratio),
            )
        usable_candidate = score.edge_usable.any(dim=3) & (candidate_probability[..., 0] > 0.0)
        effective_mass = (
            score.edge_usable.to(dtype=torch.float32) * candidate_probability * view_weights
        ).sum(dim=(2, 3))
        point = score.point_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
        effective = usable_candidate.sum(dim=2).detach().cpu().numpy().astype(np.int16)
        mass = effective_mass.detach().cpu().numpy().astype(np.float32)
        outputs["pose_log_likelihood_ratios"].append(
            score.pose_log_likelihood_ratios.detach().cpu().numpy().astype(np.float32)
        )
        outputs["point_log_likelihood_ratios"].append(point)
        outputs["point_effective_candidate_counts"].append(effective)
        outputs["point_effective_view_masses"].append(mass)
        outputs["source_log_likelihood_means"].append(
            np.stack([point[:, sources == name].mean(axis=1) for name in source_names], axis=1).astype(
                np.float32
            )
        )
        outputs["source_effective_point_counts"].append(
            np.stack(
                [(effective[:, sources == name] > 0).sum(axis=1) for name in source_names],
                axis=1,
            ).astype(np.int16)
        )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    result = {name: np.concatenate(values, axis=0) for name, values in outputs.items()}
    result["source_names"] = source_names
    return result


def _frozen_query_evidence_sha(
    *, layout: CandidatePoseRGBSpatialLayout, runtime: CandidatePoseRGBSpatialRuntime
) -> str:
    digest = hashlib.sha256()
    arrays = {
        "source_point_ids": layout.source_point_ids,
        "xy": layout.xy,
        "point_sources": layout.point_sources,
        "candidate_track_ids": layout.candidate_track_ids,
        "candidate_probabilities": layout.candidate_prior_probabilities,
        "null_probabilities": layout.null_probabilities,
        "support_image_indices": runtime.support_image_indices.numpy(),
        "support_xy": runtime.support_xy.numpy(),
        "support_view_valid": runtime.support_view_valid.numpy(),
        "candidate_view_weights": runtime.candidate_view_weights.numpy(),
    }
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()[:16]


def _assert_score_row_alignment(arrays: Mapping[str, np.ndarray]) -> None:
    """Require hypothesis outputs to align without misclassifying point evidence."""

    query_ids = np.asarray(arrays.get("query_ids", ())).astype(str).reshape(-1)
    if len(query_ids) == 0:
        raise RuntimeError("RGB spatial score has no hypothesis rows")
    for name, value in arrays.items():
        if name in _FIXED_POINT_ARRAY_NAMES:
            continue
        array = np.asarray(value)
        if array.ndim == 0 or array.shape[0] != len(query_ids):
            raise RuntimeError("RGB spatial score row arrays are not aligned")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    if (
        int(args.hypothesis_batch_size) <= 0
        or int(args.edge_chunk_size) < 0
        or int(args.hypothesis_limit) < 0
        or not math.isfinite(float(args.rgb_cache_gb))
        or float(args.rgb_cache_gb) <= 0.0
    ):
        raise ValueError("RGB spatial scorer arguments are invalid")
    output_path = Path(args.output)
    if output_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite score artifact: {output_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RGB spatial scorer requested CUDA but CUDA is unavailable")
    paths = {
        "hypothesis_artifact": Path(args.hypothesis_artifact),
        "baseline_score_artifact": Path(args.baseline_score_artifact),
        "detector_query_cache": Path(args.detector_query_cache),
        "proposals": Path(args.proposals),
        "candidate_artifact": Path(args.candidate_artifact),
        "fixed_candidate_prior_overlay": Path(args.fixed_candidate_prior_overlay),
        "mixed_verification_points_artifact": Path(args.mixed_verification_points_artifact),
        "rgb_spatial_layout": Path(args.rgb_spatial_layout),
        "projected_landmark_bank": Path(args.projected_landmark_bank),
        "radio_final_context_cache": Path(args.radio_final_context_cache),
        "radio_intermediate_context_cache": Path(args.radio_intermediate_context_cache),
        "alike_spatial_context_cache": Path(args.alike_spatial_context_cache),
        "checkpoint": Path(args.checkpoint),
        "colmap_cameras_bin": Path(args.colmap_model_dir) / "cameras.bin",
        "colmap_images_bin": Path(args.colmap_model_dir) / "images.bin",
    }
    reference_path = (
        None
        if not str(args.baseline_reference_hypothesis_artifact).strip()
        else Path(args.baseline_reference_hypothesis_artifact)
    )
    if reference_path is not None:
        paths["baseline_reference_hypothesis_artifact"] = reference_path
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"RGB spatial scorer input is missing: {name} ({path})")
    if not Path(args.image_root).is_dir():
        raise FileNotFoundError(f"RGB image root is missing: {args.image_root}")

    baseline_hypothesis_path = (
        paths["hypothesis_artifact"] if reference_path is None else reference_path
    )
    exact, baseline_hypothesis_metadata, baseline_metadata = _load_exact_hypotheses(
        hypothesis_path=baseline_hypothesis_path,
        baseline_path=paths["baseline_score_artifact"],
        detector_path=paths["detector_query_cache"],
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate_artifact"],
        prior_path=paths["fixed_candidate_prior_overlay"],
        fixed_candidate_top_k=20,
        query_id=(None if not str(args.query_id).strip() else str(args.query_id)),
    )
    baseline_reference_equivalence = None
    if reference_path is None:
        hypothesis_metadata = baseline_hypothesis_metadata
    else:
        baseline_reference_equivalence = _validate_baseline_reference_hypothesis_equivalence(
            current_hypothesis_path=paths["hypothesis_artifact"],
            reference_hypothesis_path=reference_path,
        )
        _fields, hypothesis_metadata = load_inference_artifact_fields(
            paths["hypothesis_artifact"], _BASELINE_REFERENCE_FIELDS
        )
    if int(args.hypothesis_limit) > 0:
        exact = {name: np.asarray(value)[: int(args.hypothesis_limit)] for name, value in exact.items()}
    query_ids = np.unique(np.asarray(exact["query_ids"]).astype(str))
    split_names = np.unique(np.asarray(exact["split_names"]).astype(str))
    allowed_split_names = {"validation", "test"}
    if bool(args.oof_train_query):
        allowed_split_names.add("train")
    if (
        len(query_ids) != 1
        or len(split_names) != 1
        or str(split_names[0]) not in allowed_split_names
    ):
        raise ValueError("RGB spatial scorer accepts one held-out or explicitly OOF train query shard")
    query_id = str(query_ids[0])
    split_name = str(split_names[0])
    if str(split_name) == "train" and not bool(args.oof_train_query):
        raise ValueError("train query scoring requires --oof-train-query")

    layout = load_candidate_pose_rgb_spatial_layout(paths["rgb_spatial_layout"])
    points = load_mixed_verification_points(paths["mixed_verification_points_artifact"])
    layout_point_rows, layout_point_lineage = _assert_layout_matches_mixed_points(
        layout=layout,
        points=points,
        points_path=paths["mixed_verification_points_artifact"],
    )
    layout_rows = _query_rows(layout=layout, query_id=query_id, split_name=split_name)
    point_rows = np.asarray(layout_point_rows[layout_rows], dtype=np.int64)
    if len(point_rows) == 0 or np.any(points.split_names[point_rows] != split_name):
        raise ValueError("frozen hypothesis query is absent from mixed verification points")
    query_layout = _slice_layout(layout, layout_rows)
    _validate_query_layout_against_points(
        query_layout=query_layout, points=points, points_rows=point_rows
    )

    sources = load_context_attention_sources(
        radio_final_context_cache=paths["radio_final_context_cache"],
        radio_intermediate_context_cache=paths["radio_intermediate_context_cache"],
        alike_spatial_context_cache=paths["alike_spatial_context_cache"],
        expected_radio_checkpoint="",
        require_equal_descriptor_dimensions=False,
    )
    image_ids, image_sizes, source_tensors = _source_table(sources)
    source_metadata = sources[0].metadata
    unique_sizes = np.unique(image_sizes, axis=0)
    if unique_sizes.shape != (1, 2):
        raise ValueError("RGB spatial scorer requires a common processed coordinate size")
    coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
    rgb_image_size = _discover_rgb_image_size(
        image_root=Path(args.image_root), image_id=str(image_ids[0])
    )
    rgb_bridge = validate_rgb_coordinate_bridge(
        source_metadata=source_metadata,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
    )
    cache_hashes = {
        name: file_sha256_short(paths[name]) for name in _CONTEXT_INPUT_NAMES
    }
    model, checkpoint_metadata = _load_checkpoint_model(
        path=paths["checkpoint"],
        layout=query_layout,
        source_tensors=source_tensors,
        image_sizes=image_sizes,
        cache_hashes=cache_hashes,
        source_image_manifest_sha256=str(
            source_metadata.get("source_image_manifest_sha256", "")
        ),
        device=device,
    )
    oof_train_partition = (
        None
        if not bool(args.oof_train_query)
        else _validate_oof_train_query_checkpoint(
            checkpoint_metadata=checkpoint_metadata, query_id=query_id
        )
    )
    requested_chunk = int(args.edge_chunk_size)
    edge_chunk_size = int(model.edge_chunk_size) if requested_chunk == 0 else requested_chunk
    if edge_chunk_size <= 0:
        raise ValueError("RGB spatial scorer edge chunk size is invalid")
    model.edge_chunk_size = edge_chunk_size
    if str(args.point_selector) == "checkpoint_validation":
        selected_positions, point_selection = _checkpoint_validation_selector_positions(
            layout=query_layout,
            config=dict(checkpoint_metadata["config"]),
            coordinate_image_size=coordinate_image_size,
        )
        query_layout = _slice_layout(query_layout, selected_positions)
        point_rows = np.asarray(point_rows, dtype=np.int64)[selected_positions]
        _validate_query_layout_against_points(
            query_layout=query_layout, points=points, points_rows=point_rows
        )
    else:
        point_selection = {
            "mode": "all_fixed_p1_verification_points_v1",
            "point_budget": int(query_layout.row_count),
            "target_free_static_only": True,
        }
    runtime = runtime_from_target_free_layout(query_layout, image_ids=image_ids)
    bank_track_ids, bank_xyz, bank_metadata = _load_bank_xyz(paths["projected_landmark_bank"])
    if str(bank_metadata.get("descriptor_space_manifest", {}).get("projection_space_id", "")) != str(
        query_layout.metadata.get("projection_space_id", "")
    ):
        raise ValueError("projected landmark bank projection space differs from RGB layout")
    geometry = _query_geometry(
        query_id=query_id,
        query_layout=query_layout,
        bank_track_ids=bank_track_ids,
        bank_xyz=bank_xyz,
        colmap_model_dir=Path(args.colmap_model_dir),
    )
    cache = TensorImageLRUCache(
        max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
        storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
    )
    with torch.no_grad():
        query_patches, support_patches = _crop_runtime_rgb_patches(
            runtime=runtime,
            image_ids=image_ids,
            image_root=Path(args.image_root),
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
            radius_px=float(model.search_radius_px + model.context_radius_px),
            step_px=float(model.step_px),
            cache=cache,
            device=device,
        )
        active_runtime = runtime
        active_support_patches = support_patches
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            normal_prediction = model(
                runtime=runtime,
                query_rgb_patches=query_patches,
                support_rgb_patches=support_patches,
            )
        if _uses_support_image_appearance_control(str(args.evidence_variant)):
            active_runtime = permute_runtime_support_image_appearance_only(runtime, shift=1)
            active_support_patches = _crop_geometry_fixed_permuted_support_patches(
                normal_query_patches=query_patches,
                permuted_runtime=active_runtime,
                image_ids=image_ids,
                image_root=Path(args.image_root),
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                radius_px=float(model.search_radius_px + model.context_radius_px),
                step_px=float(model.step_px),
                cache=cache,
                device=device,
            )
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(
                    runtime=active_runtime,
                    query_rgb_patches=query_patches,
                    support_rgb_patches=active_support_patches,
                )
            _assert_geometry_fixed_support_image_control(
                runtime=runtime,
                permuted_runtime=active_runtime,
                normal_prediction=normal_prediction,
                permuted_prediction=prediction,
            )
        else:
            prediction = normal_prediction
        prediction = candidate_pose_rgb_spatial_score_component_prediction(
            prediction=prediction, component=str(args.score_component)
        )
        statistics = _score_hypotheses(
            model=model,
            runtime=active_runtime,
            prediction=prediction,
            geometry=geometry,
            poses_w2c=np.asarray(exact["poses_w2c"], dtype=np.float64),
            point_sources=query_layout.point_sources,
            hypothesis_batch_size=int(args.hypothesis_batch_size),
            edge_chunk_size=edge_chunk_size,
            max_abs_pose_log_ratio=float(checkpoint_metadata["config"]["max_abs_pose_log_ratio"]),
        )
    frozen_support_image_ids = image_ids[runtime.support_image_indices.numpy()]
    scored_support_image_ids = image_ids[active_runtime.support_image_indices.numpy()]
    arrays: dict[str, np.ndarray] = {
        "query_ids": np.asarray(exact["query_ids"]).astype(str),
        "split_names": np.asarray(exact["split_names"]).astype(str),
        "evaluation_labels": np.asarray(exact["evaluation_labels"]).astype(str),
        "hypothesis_indices": np.asarray(exact["hypothesis_indices"], dtype=np.int64),
        "source_chosen_for_optional_pose": np.asarray(
            exact["source_chosen_for_optional_pose"], dtype=bool
        ),
        "baseline_score_top1": np.asarray(exact["independent_score_top1"], dtype=bool),
        "baseline_selection_scores": np.asarray(
            exact["independent_selection_scores"], dtype=np.float64
        ),
        "pose_log_likelihood_ratios": statistics["pose_log_likelihood_ratios"],
        "point_log_likelihood_ratios": statistics["point_log_likelihood_ratios"],
        "point_effective_candidate_counts": statistics["point_effective_candidate_counts"],
        "point_effective_view_masses": statistics["point_effective_view_masses"],
        "source_names": statistics["source_names"],
        "source_log_likelihood_means": statistics["source_log_likelihood_means"],
        "source_effective_point_counts": statistics["source_effective_point_counts"],
        "verification_source_point_ids": query_layout.source_point_ids,
        "verification_point_sources": query_layout.point_sources,
        "verification_source_detector_rows": points.source_detector_rows[point_rows],
        "verification_xy": query_layout.xy,
        "candidate_track_ids": query_layout.candidate_track_ids,
        "candidate_probabilities": query_layout.candidate_prior_probabilities,
        "null_probabilities": query_layout.null_probabilities,
        "candidate_view_weights": runtime.candidate_view_weights.numpy(),
        "candidate_support_image_ids": frozen_support_image_ids,
        "scored_support_image_ids": scored_support_image_ids,
    }
    _assert_score_row_alignment(arrays)
    metadata: dict[str, Any] = {
        "format": CANDIDATE_POSE_LLR_SCORE_FORMAT,
        "version": SCORE_VERSION,
        "model_format": CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
        "contains_target_fields": False,
        "pose_or_ground_truth_used_for_scoring": False,
        "supervision_arrays_loaded": False,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "raw_scores_must_not_feed_pnp": True,
        "render": False,
        "image_retrieval_or_submap_used": False,
        "row_count": int(len(arrays["query_ids"])),
        "query_count": 1,
        "query_id": query_id,
        "explicit_query_id": (
            None if not str(args.query_id).strip() else str(args.query_id)
        ),
        "split_name": split_name,
        "evidence_variant": str(args.evidence_variant),
        "score_component": str(args.score_component),
        "model_checkpoint": {
            "path": str(paths["checkpoint"]),
            "sha256": file_sha256_short(paths["checkpoint"]),
        },
        "model_checkpoint_contract": {
            "architecture": checkpoint_metadata.get("architecture"),
            "config": checkpoint_metadata.get("config"),
            "holdout_evaluation_allowed": checkpoint_metadata.get(
                "holdout_evaluation_allowed"
            ),
            "train_only_inner_gate_passed": checkpoint_metadata.get(
                "train_only_inner_gate_passed"
            ),
            "oof_train_query_partition": oof_train_partition,
        },
        "inference_edge_chunk_size": edge_chunk_size,
        "requested_edge_chunk_size": requested_chunk,
        "strict_candidate_pose_llr_contract": {
            "heldout_query_rows": True,
            "formal_p1_mixed_multiscale_points": True,
            "fixed_global_topl": True,
            "fixed_candidate_top_k": 20,
            "candidate_identity_fixed_across_hypotheses": True,
            "candidate_reselection_per_pose": False,
            "support_reselection_per_pose": False,
            "fixed_support_view_count": 2,
            "explicit_null": True,
            "candidate_projection_is_only_pose_dependent_encoder_input": True,
            "candidate_pose_matrix_excluded_from_encoder": True,
            "residual_and_target_excluded_from_encoder": True,
            # Preserve the historic field for the paired-audit loader while
            # recording the stricter implementation contract beside it.
            "support_descriptor_permutation_control": _uses_support_image_appearance_control(
                str(args.evidence_variant)
            ),
            "support_image_appearance_derangement_control": _uses_support_image_appearance_control(
                str(args.evidence_variant)
            ),
            "appearance_control_geometry_fixed": True,
            "no_pnp": True,
            "oof_train_query_scoring": bool(args.oof_train_query),
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "hypothesis_scope": {
            "all_frozen_hypotheses": int(args.hypothesis_limit) == 0,
            "development_prefix_limit": int(args.hypothesis_limit),
            "scored_hypothesis_count": int(len(arrays["query_ids"])),
        },
        "raw_score_semantics": (
            "frozen_candidate_null_mixture_of_high_resolution_real_rgb_local_density_"
            "and_full_2d_radio_final_intermediate_alike_context_llr_v2;"
            f"component={str(args.score_component)}"
        ),
        "raw_score_is_calibrated_independent_pose_likelihood": False,
        "frozen_query_evidence_sha256": _frozen_query_evidence_sha(
            layout=query_layout, runtime=runtime
        ),
        "verification_point_selection": {
            "source": str(layout_point_lineage["mode"]),
            "point_count": int(query_layout.row_count),
            "parent_point_count": int(layout_point_lineage["parent_point_count"]),
            "frozen_layout_selected_point_count": int(
                layout_point_lineage["selected_point_count"]
            ),
            "frozen_rgb_selector": layout_point_lineage["frozen_rgb_selector"],
            "source_point_ids_sha256": _canonical_hash(
                {"source_point_ids": query_layout.source_point_ids.tolist()}
            ),
            **point_selection,
        },
        "support_descriptor_derangement": (
            None
            if not _uses_support_image_appearance_control(str(args.evidence_variant))
            else "fixed_coordinate_support_image_derangement_recrop_v2"
        ),
        "baseline_reference_hypothesis_equivalence": baseline_reference_equivalence,
        "rgb_coordinate_bridge": rgb_bridge,
        "inputs": _input_manifest(paths),
        "source_metadata_hashes": {
            "hypothesis": _canonical_hash(hypothesis_metadata),
            "baseline_s0": _canonical_hash(baseline_metadata),
            "mixed_verification_points": _canonical_hash(points.metadata),
            "rgb_spatial_layout": _canonical_hash(query_layout.metadata),
            "projected_landmark_bank": _canonical_hash(bank_metadata),
        },
        "elapsed_seconds": float(time.time() - start),
    }
    validate_target_free_pose_llr_score_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, output_path)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "stage": "score_candidate_pose_rgb_spatial_likelihood",
                "output": str(output_path),
                "metadata": metadata,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output_path), "metadata": metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
