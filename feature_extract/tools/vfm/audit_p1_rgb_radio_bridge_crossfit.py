"""Cross-fit a low-capacity RGB-spatial plus RADIO-final pose-evidence bridge.

This command is intentionally a train-only diagnostic.  It first runs two
frozen target-free visual encoders over the immutable current-P1 layout:

* a high-resolution real-RGB local-density expert; and
* a broad, candidate-specific RADIO-final context identity expert.

Only after both visual forwards have completed does it join the correct-first
and coherent-wrong projection stack from the training target artifact.  A
small non-negative three-parameter bridge is then selected using five-fold
query-grouped cross-fitting.  It is not a checkpoint trainer and never runs
PnP, validation/test scoring, rendering, retrieval, or submaps.
"""

from __future__ import annotations

import argparse
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

from feature_extract.tools.vfm.audit_candidate_pose_rgb_spatial_identity_llr_sources import (
    _load_checkpoint as _load_identity_checkpoint,
    _load_model_state as _load_identity_model_state,
    _validate_checkpoint_source_lineage,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_identity_llr import (
    _write_json_atomically,
)
from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import (
    _crop_runtime_rgb_patches,
    _discover_rgb_image_size,
    _finalize_distributed,
    _initialize_distributed,
    _source_table,
    build_train_query_groups,
    validate_rgb_coordinate_bridge,
    validate_training_layout_and_targets,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_bridge import (
    CandidatePoseRGBSpatialBridgeQueryEvidence,
    CandidatePoseRGBSpatialBridgeWeights,
    bridge_query_gap,
    crossfit_bridge_profiles,
    summarize_bridge_gaps,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_identity_llr import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT,
    CandidatePoseRGBSpatialIdentityLLR,
    marginalize_candidate_pose_rgb_spatial_identity_llr,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT,
    CandidatePoseRGBSpatialLikelihood,
    candidate_pose_rgb_spatial_score_component_prediction,
    permute_runtime_support_appearance,
    permute_support_patch_appearance,
    resolve_candidate_pose_rgb_spatial_context_encoder_arch,
    resolve_candidate_pose_rgb_spatial_context_windows,
    runtime_from_target_free_layout,
    score_candidate_pose_rgb_spatial_batch,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
)
from feature_extract.vfm.localization.context_attention_candidate_probe import (
    load_context_attention_source_headers,
    load_context_attention_sources,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    resolve_rgb_image_cache_storage_dtype,
)


AUDIT_FORMAT = "p1_rgb_radio_final_bridge_crossfit_audit_v1"
TRAINING_FEATURE_FORMAT = "p1_rgb_radio_final_bridge_training_feature_v1"
RGB_SPATIAL_CHECKPOINT_FORMAT = "candidate_pose_rgb_spatial_likelihood_checkpoint_v1"
_RADIO_FINAL_ONLY_SCALES = {
    "rgb": 0.0,
    "radio_final": 1.0,
    "radio_intermediate": 0.0,
    "alike": 0.0,
}
_BASELINE_WEIGHTS = CandidatePoseRGBSpatialBridgeWeights(
    spatial_weight=1.0,
    identity_weight=0.0,
    prior_exponent=1.0,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-checkpoint", required=True)
    parser.add_argument("--identity-checkpoint", required=True)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--training-targets", required=True)
    parser.add_argument("--radio-final-context-cache", required=True)
    parser.add_argument("--radio-intermediate-context-cache", required=True)
    parser.add_argument("--alike-spatial-context-cache", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crossfit-fold-count", type=int, default=5)
    parser.add_argument("--permutation-control-shift", type=int, default=1)
    parser.add_argument("--rgb-cache-gb", type=float, default=8.0)
    parser.add_argument(
        "--rgb-cache-dtype", choices=("float16", "uint8"), default="uint8"
    )
    parser.add_argument("--catastrophic-gap-threshold", type=float, default=-0.5)
    parser.add_argument("--minimum-normal-gap", type=float, default=0.05)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.55)
    parser.add_argument("--minimum-visual-gap-delta", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _safe_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _feature_filename(query_id: str) -> str:
    value = str(query_id)
    if not value:
        raise ValueError("bridge training feature query ID is empty")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20] + ".npz"


def _save_npz_atomically(path: Path, **arrays: object) -> None:
    """Write a per-query feature shard without exposing partial ZIP output."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(destination)


def _load_json_scalar(data: Mapping[str, np.ndarray], field: str) -> Mapping[str, object]:
    try:
        raw = str(np.asarray(data[field]).item())
        value = json.loads(raw)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("bridge training feature metadata is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError("bridge training feature metadata is not an object")
    return dict(value)


def _feature_metadata(
    *,
    query_id: str,
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Mark serialized score tensors as train-only and runtime-ineligible."""

    return {
        "format": TRAINING_FEATURE_FORMAT,
        "query_id": str(query_id),
        "contains_target_fields": True,
        "runtime_scorer_must_not_load_this_artifact": True,
        "target_join_after_visual_inference": True,
        "pose_or_residual_serialized": False,
        "no_render": True,
        "no_image_retrieval_or_submap": True,
        "lineage": dict(lineage),
    }


def _write_training_feature(
    *,
    path: Path,
    evidence: CandidatePoseRGBSpatialBridgeQueryEvidence,
    source_point_ids: np.ndarray,
    lineage: Mapping[str, object],
) -> None:
    ids = np.asarray(source_point_ids, dtype=np.int64).reshape(-1)
    if ids.shape != (evidence.spatial_candidate_llrs.shape[1],):
        raise ValueError("bridge training feature source-point rows are invalid")
    _save_npz_atomically(
        path,
        source_point_ids=ids,
        spatial_candidate_llrs=np.asarray(evidence.spatial_candidate_llrs, dtype=np.float32),
        control_spatial_candidate_llrs=np.asarray(
            evidence.control_spatial_candidate_llrs, dtype=np.float32
        ),
        identity_candidate_llrs=np.asarray(evidence.identity_candidate_llrs, dtype=np.float32),
        control_identity_candidate_llrs=np.asarray(
            evidence.control_identity_candidate_llrs, dtype=np.float32
        ),
        candidate_probabilities=np.asarray(evidence.candidate_probabilities, dtype=np.float32),
        null_probabilities=np.asarray(evidence.null_probabilities, dtype=np.float32),
        metadata_json=np.asarray(_safe_json(_feature_metadata(
            query_id=evidence.query_id, lineage=lineage
        ))),
    )


def _load_training_feature(
    *,
    path: Path,
    expected_query_id: str,
    expected_lineage: Mapping[str, object],
) -> CandidatePoseRGBSpatialBridgeQueryEvidence:
    """Load only train-side output from a frozen visual-forward shard."""

    required = {
        "source_point_ids",
        "spatial_candidate_llrs",
        "control_spatial_candidate_llrs",
        "identity_candidate_llrs",
        "control_identity_candidate_llrs",
        "candidate_probabilities",
        "null_probabilities",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as data:
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"bridge training feature lacks {sorted(missing)}")
        metadata = _load_json_scalar(data, "metadata_json")
        if (
            metadata.get("format") != TRAINING_FEATURE_FORMAT
            or metadata.get("query_id") != str(expected_query_id)
            or metadata.get("contains_target_fields") is not True
            or metadata.get("runtime_scorer_must_not_load_this_artifact") is not True
            or metadata.get("target_join_after_visual_inference") is not True
            or metadata.get("pose_or_residual_serialized") is not False
            or metadata.get("no_render") is not True
            or metadata.get("no_image_retrieval_or_submap") is not True
            or metadata.get("lineage") != dict(expected_lineage)
        ):
            raise ValueError("bridge training feature contract differs from this audit")
        source_ids = np.asarray(data["source_point_ids"], dtype=np.int64).reshape(-1)
        evidence = CandidatePoseRGBSpatialBridgeQueryEvidence(
            query_id=str(expected_query_id),
            spatial_candidate_llrs=np.asarray(data["spatial_candidate_llrs"], dtype=np.float32),
            control_spatial_candidate_llrs=np.asarray(
                data["control_spatial_candidate_llrs"], dtype=np.float32
            ),
            identity_candidate_llrs=np.asarray(
                data["identity_candidate_llrs"], dtype=np.float32
            ),
            control_identity_candidate_llrs=np.asarray(
                data["control_identity_candidate_llrs"], dtype=np.float32
            ),
            candidate_probabilities=np.asarray(data["candidate_probabilities"], dtype=np.float32),
            null_probabilities=np.asarray(data["null_probabilities"], dtype=np.float32),
        )
    if (
        source_ids.shape != (evidence.spatial_candidate_llrs.shape[1],)
        or len(np.unique(source_ids)) != len(source_ids)
    ):
        raise ValueError("bridge training feature source-point IDs are invalid")
    return evidence


def _candidate_weight_grid() -> tuple[CandidatePoseRGBSpatialBridgeWeights, ...]:
    """Return the frozen, low-capacity bridge profiles selected by OOF only."""

    values: set[CandidatePoseRGBSpatialBridgeWeights] = {_BASELINE_WEIGHTS}
    for spatial_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        for identity_weight in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
            for prior_exponent in (0.5, 1.0, 2.0):
                values.add(
                    CandidatePoseRGBSpatialBridgeWeights(
                        spatial_weight=float(spatial_weight),
                        identity_weight=float(identity_weight),
                        prior_exponent=float(prior_exponent),
                    )
                )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                float(item.spatial_weight),
                float(item.identity_weight),
                float(item.prior_exponent),
            ),
        )
    )


def _score_rows(
    *,
    evidences: Sequence[CandidatePoseRGBSpatialBridgeQueryEvidence],
    weights: CandidatePoseRGBSpatialBridgeWeights,
) -> list[dict[str, object]]:
    rows = [
        {
            "query_id": evidence.query_id,
            "normal_gap": bridge_query_gap(evidence=evidence, weights=weights, control=False),
            "control_gap": bridge_query_gap(evidence=evidence, weights=weights, control=True),
        }
        for evidence in evidences
    ]
    rows.sort(key=lambda row: str(row["query_id"]))
    return rows


def _profile_summary(
    rows: Sequence[Mapping[str, object]], *, catastrophic_threshold: float
) -> dict[str, object]:
    normal = [float(row["normal_gap"]) for row in rows]
    control = [float(row["control_gap"]) for row in rows]
    normal_summary = summarize_bridge_gaps(
        normal, catastrophic_threshold=float(catastrophic_threshold)
    )
    control_summary = summarize_bridge_gaps(
        control, catastrophic_threshold=float(catastrophic_threshold)
    )
    deltas = np.asarray(normal, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    return {
        "normal": normal_summary,
        "support_permuted_control": control_summary,
        "normal_minus_control_mean_gap": float(deltas.mean()),
        "normal_minus_control_median_gap": float(np.median(deltas)),
        "normal_minus_control_win_fraction": float(np.mean(deltas > 0.0)),
    }


def _paired_bridge_comparison(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    candidate = {str(row["query_id"]): float(row["normal_gap"]) for row in candidate_rows}
    baseline = {str(row["query_id"]): float(row["normal_gap"]) for row in baseline_rows}
    if not candidate or set(candidate) != set(baseline):
        raise ValueError("bridge paired comparison query IDs differ")
    delta = np.asarray(
        [candidate[query_id] - baseline[query_id] for query_id in sorted(candidate)],
        dtype=np.float64,
    )
    return {
        "query_count": float(len(delta)),
        "mean_gap_improvement": float(delta.mean()),
        "median_gap_improvement": float(np.median(delta)),
        "win_count": float(np.sum(delta > 1e-8)),
        "loss_count": float(np.sum(delta < -1e-8)),
        "tie_count": float(np.sum(np.abs(delta) <= 1e-8)),
    }


def _bridge_gate(
    *,
    bridge_summary: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    paired: Mapping[str, float],
    minimum_normal_gap: float,
    minimum_win_fraction: float,
    minimum_visual_gap_delta: float,
) -> dict[str, object]:
    normal = bridge_summary.get("normal")
    baseline_normal = baseline_summary.get("normal")
    if not isinstance(normal, Mapping) or not isinstance(baseline_normal, Mapping):
        raise ValueError("bridge gate summaries are invalid")
    checks = {
        "normal_mean_gap": float(normal["mean_correct_minus_hardest_wrong"])
        >= float(minimum_normal_gap),
        "normal_win_fraction": float(normal["correct_win_fraction"])
        >= float(minimum_win_fraction),
        "support_permutation_visual_gap": float(
            bridge_summary["normal_minus_control_mean_gap"]
        )
        >= float(minimum_visual_gap_delta),
        "catastrophic_tail_not_worse_than_rgb_baseline": float(
            normal["catastrophic_gap_count"]
        )
        <= float(baseline_normal["catastrophic_gap_count"]),
        "paired_rgb_baseline_wins_exceed_losses": float(paired["win_count"])
        > float(paired["loss_count"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "minimum_normal_gap": float(minimum_normal_gap),
        "minimum_win_fraction": float(minimum_win_fraction),
        "minimum_visual_gap_delta": float(minimum_visual_gap_delta),
        "paired_rgb_baseline": dict(paired),
    }


def _load_rgb_spatial_checkpoint(
    *,
    path: Path,
    layout: CandidatePoseRGBSpatialLayout,
    targets: CandidatePoseRGBSpatialTrainingTargets,
    context_source_dimensions: Mapping[str, int],
    image_sizes: np.ndarray,
    source_lineage: Mapping[str, object],
    layout_sha256: str,
    targets_sha256: str,
    device: torch.device,
) -> tuple[CandidatePoseRGBSpatialLikelihood, Mapping[str, object]]:
    """Load only the source-safe RGB-only hard-repeat checkpoint contract."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("RGB bridge checkpoint is malformed")
    metadata = payload.get("metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("RGB bridge checkpoint lacks metadata or state")
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    inputs = metadata.get("inputs")
    training = metadata.get("training")
    if (
        payload.get("format") != RGB_SPATIAL_CHECKPOINT_FORMAT
        or metadata.get("format") != RGB_SPATIAL_CHECKPOINT_FORMAT
        or metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_LIKELIHOOD_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("fixed_global_topl") is not True
        or int(metadata.get("fixed_candidate_top_k", -1)) != int(layout.candidate_count)
        or int(metadata.get("fixed_support_view_count", -1)) != int(layout.support_view_count)
        or metadata.get("explicit_null") is not True
        or metadata.get("projection_after_network_only") is not True
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("diagnostic_only") is not True
        or not isinstance(config, Mapping)
        or not isinstance(lineage, Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(training, Mapping)
    ):
        raise ValueError("RGB bridge checkpoint violates the frozen diagnostic contract")
    excludes = set(str(value) for value in metadata.get("encoder_excludes", ()))
    if not {
        "pose_matrix",
        "projection_offset",
        "reprojection_residual",
        "ground_truth_label",
        "track_id",
        "candidate_rank",
        "coarse_score",
    }.issubset(excludes):
        raise ValueError("RGB bridge checkpoint encoder exclusions are incomplete")
    if (
        config.get("rgb_cost_volume_only") is not True
        or config.get("trainable_parameter_scope") != "texture_encoder_only"
        or int(config.get("search_radius_px", 0)) <= 0
        or int(config.get("context_radius_px", 0)) <= 0
        or float(config.get("step_px", 0.0)) <= 0.0
        or int(config.get("edge_chunk_size", 0)) <= 0
    ):
        raise ValueError("RGB bridge checkpoint is not the source-safe RGB-only expert")
    if (
        str(lineage.get("layout_sha256", "")) != str(layout_sha256)
        or str(lineage.get("training_targets_sha256", "")) != str(targets_sha256)
        or str(lineage.get("descriptor_space_id", ""))
        != str(layout.metadata.get("descriptor_space_id", ""))
        or str(lineage.get("projection_space_id", ""))
        != str(layout.metadata.get("projection_space_id", ""))
        or lineage.get("rgb_coordinate_bridge") != source_lineage["rgb_coordinate_bridge"]
        or str(lineage.get("source_image_manifest_sha256", ""))
        != str(source_lineage["source_image_manifest_sha256"])
    ):
        raise ValueError("RGB bridge checkpoint lineage differs from current frozen inputs")
    expected_inputs = {
        "rgb_spatial_layout": str(layout_sha256),
        "training_targets": str(targets_sha256),
        "radio_final_context_cache": str(source_lineage["radio_final_context_cache_sha256"]),
        "radio_intermediate_context_cache": str(
            source_lineage["radio_intermediate_context_cache_sha256"]
        ),
        "alike_spatial_context_cache": str(source_lineage["alike_spatial_context_cache_sha256"]),
    }
    for name, expected_hash in expected_inputs.items():
        record = inputs.get(name)
        if not isinstance(record, Mapping) or str(record.get("sha256", "")) != expected_hash:
            raise ValueError(f"RGB bridge checkpoint is stale for {name}")
    inner = training.get("inner_validation")
    if (
        not isinstance(inner, Mapping)
        or not isinstance(inner.get("gate"), Mapping)
        or inner["gate"].get("hard_repeat_passed") is not True
    ):
        raise ValueError("RGB bridge checkpoint did not pass its direct hard-repeat gate")
    model = CandidatePoseRGBSpatialLikelihood(
        sources=None,
        context_source_dimensions={str(name): int(value) for name, value in context_source_dimensions.items()},
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
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError("RGB bridge checkpoint state is incompatible") from error
    return model.to(device).eval(), dict(metadata)


def _load_radio_final_identity_checkpoint(
    *,
    path: Path,
    source_tensors: Mapping[str, torch.Tensor],
    image_sizes: np.ndarray,
    layout: CandidatePoseRGBSpatialLayout,
    source_lineage: Mapping[str, object],
    device: torch.device,
) -> tuple[CandidatePoseRGBSpatialIdentityLLR, Mapping[str, object], str]:
    """Load a gate-approved V3 broad encoder, not a P1-calibrated scalar head."""

    state_dict, metadata = _load_identity_checkpoint(Path(path))
    config = metadata.get("config")
    lineage = metadata.get("lineage")
    training = metadata.get("training")
    if (
        metadata.get("model_format") != CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_LLR_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("checkpoint_contains_train_targets") is not False
        or metadata.get("runtime_layout_is_target_free") is not True
        or metadata.get("pose_or_ground_truth_used_by_runtime_scorer") is not False
        or metadata.get("render") is not False
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("p1_initialization_allowed") is not True
        or metadata.get("raw_scores_must_not_feed_pnp") is not True
        or int(metadata.get("fixed_candidate_count", -1)) != int(layout.candidate_count)
        or not isinstance(config, Mapping)
        or not isinstance(lineage, Mapping)
        or not isinstance(training, Mapping)
    ):
        raise ValueError("RADIO identity checkpoint violates the frozen broad-pretrain contract")
    broad_inner = training.get("inner_validation")
    if not isinstance(broad_inner, Mapping) or not isinstance(broad_inner.get("gate"), Mapping):
        raise ValueError("RADIO identity checkpoint lacks a broad-pretrain gate")
    if broad_inner["gate"].get("passed") is not True:
        raise ValueError("RADIO identity checkpoint broad-pretrain gate did not pass")
    lineage_contract = _validate_checkpoint_source_lineage(
        checkpoint_lineage=lineage, source_lineage=source_lineage
    )
    model = CandidatePoseRGBSpatialIdentityLLR(
        sources=source_tensors,
        image_sizes=torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)),
        rgb_context_radius_px=float(config["rgb_context_radius_px"]),
        rgb_step_px=float(config["rgb_step_px"]),
        texture_feature_dim=int(config["texture_feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        max_abs_log_ratio=float(config["max_abs_log_ratio"]),
        edge_chunk_size=int(config["edge_chunk_size"]),
        context_windows=dict(config["context_windows"]),
    )
    _load_identity_model_state(model=model, state_dict=state_dict)
    return model.to(device).eval(), dict(metadata), str(lineage_contract)


def _train_only_projection_stack(group: object) -> tuple[np.ndarray, np.ndarray]:
    """Join correct-first pose projections only after visual inference."""

    correct_offsets = np.asarray(
        getattr(group, "correct_projection_offsets_xy"), dtype=np.float32
    )
    correct_valid = np.asarray(getattr(group, "correct_projection_valid"), dtype=bool)
    wrong_offsets = np.asarray(
        getattr(group, "wrong_projection_offsets_xy"), dtype=np.float32
    )
    wrong_valid = np.asarray(getattr(group, "wrong_projection_valid"), dtype=bool)
    if (
        correct_offsets.ndim != 3
        or correct_offsets.shape[-1] != 2
        or correct_valid.shape != correct_offsets.shape[:2]
        or wrong_offsets.ndim != 4
        or wrong_offsets.shape[0] == 0
        or wrong_offsets.shape[1:] != correct_offsets.shape
        or wrong_valid.shape != wrong_offsets.shape[:-1]
        or not np.isfinite(correct_offsets).all()
        or not np.isfinite(wrong_offsets).all()
    ):
        raise ValueError("bridge train-only pose projections are invalid")
    return (
        np.concatenate((correct_offsets[None], wrong_offsets), axis=0),
        np.concatenate((correct_valid[None], wrong_valid), axis=0),
    )


@torch.inference_mode()
def _forward_query_target_free(
    *,
    group: object,
    complete_runtime: object,
    image_ids: np.ndarray,
    image_root: Path,
    coordinate_image_size: tuple[int, int],
    rgb_image_size: tuple[int, int],
    spatial_model: CandidatePoseRGBSpatialLikelihood,
    spatial_config: Mapping[str, object],
    identity_model: CandidatePoseRGBSpatialIdentityLLR,
    cache: TensorImageLRUCache,
    device: torch.device,
    amp_enabled: bool,
    permutation_shift: int,
) -> CandidatePoseRGBSpatialBridgeQueryEvidence:
    """Emit frozen visual scores, then attach the train-only pose stack."""

    query_id = str(getattr(group, "query_id"))
    layout_rows = np.asarray(getattr(group, "layout_rows"), dtype=np.int64).reshape(-1)
    source_point_ids = np.asarray(getattr(group, "source_point_ids"), dtype=np.int64).reshape(-1)
    if len(layout_rows) == 0 or source_point_ids.shape != layout_rows.shape:
        raise ValueError("bridge query has invalid frozen P1 rows")
    # Import locally to make the target-free slicing boundary explicit.
    from feature_extract.tools.vfm.train_candidate_pose_rgb_spatial_likelihood import _slice_runtime

    runtime = _slice_runtime(complete_runtime, layout_rows)
    permuted_runtime = permute_runtime_support_appearance(
        runtime, shift=int(permutation_shift)
    )
    if (
        torch.equal(runtime.support_image_indices, permuted_runtime.support_image_indices)
        and torch.equal(runtime.support_xy, permuted_runtime.support_xy)
    ):
        raise ValueError("bridge support-permutation control did not change visual ownership")

    # 1. Target-free high-resolution real-RGB local-density forward.
    crop_radius = float(spatial_config["search_radius_px"]) + float(
        spatial_config["context_radius_px"]
    )
    query_rgb, support_rgb = _crop_runtime_rgb_patches(
        runtime=runtime,
        image_ids=image_ids,
        image_root=image_root,
        coordinate_image_size=coordinate_image_size,
        rgb_image_size=rgb_image_size,
        radius_px=crop_radius,
        step_px=float(spatial_config["step_px"]),
        cache=cache,
        device=device,
    )
    permuted_support_rgb = permute_support_patch_appearance(
        runtime=permuted_runtime,
        support_patches=support_rgb,
        shift=int(permutation_shift),
    )
    with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
        spatial_prediction = spatial_model(
            runtime=runtime,
            query_rgb_patches=query_rgb,
            support_rgb_patches=support_rgb,
            rgb_cost_volume_only=True,
        )
        control_spatial_prediction = spatial_model(
            runtime=permuted_runtime,
            query_rgb_patches=query_rgb,
            support_rgb_patches=permuted_support_rgb,
            rgb_cost_volume_only=True,
        )
    spatial_prediction = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=spatial_prediction, component="rgb_cost_volume"
    )
    control_spatial_prediction = candidate_pose_rgb_spatial_score_component_prediction(
        prediction=control_spatial_prediction, component="rgb_cost_volume"
    )

    # 2. Target-free full-map RADIO-final context forward. RGB, intermediate,
    # and ALIKE content are deterministically zeroed by the frozen model's
    # audited source-scale interface; no raw RGB crop is required here.
    point_count = int(runtime.point_count)
    candidate_count = int(runtime.candidate_count)
    support_count = int(runtime.support_view_count)
    side = int(identity_model.patch_side)
    zero_query = torch.zeros((point_count, 3, side, side), dtype=torch.float32, device=device)
    zero_support = torch.zeros(
        (point_count, candidate_count, support_count, 3, side, side),
        dtype=torch.float32,
        device=device,
    )
    with torch.cuda.amp.autocast(enabled=bool(amp_enabled)):
        identity_prediction = identity_model(
            runtime=runtime,
            query_rgb_patches=zero_query,
            support_rgb_patches=zero_support,
            visual_source_scales=_RADIO_FINAL_ONLY_SCALES,
        )
        control_identity_prediction = identity_model(
            runtime=permuted_runtime,
            query_rgb_patches=zero_query,
            support_rgb_patches=zero_support,
            visual_source_scales=_RADIO_FINAL_ONLY_SCALES,
        )
    identity_candidate = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=identity_prediction, runtime=runtime
    )
    control_identity_candidate = marginalize_candidate_pose_rgb_spatial_identity_llr(
        prediction=control_identity_prediction, runtime=permuted_runtime
    )

    # 3. Only now attach train-only correct/wrong projected offsets. The visual
    # networks above never received a pose, a residual, or an identity label.
    projection_offsets, projection_valid = _train_only_projection_stack(group)
    spatial_score = score_candidate_pose_rgb_spatial_batch(
        runtime=runtime,
        prediction=spatial_prediction,
        candidate_projection_offsets_xy=torch.from_numpy(projection_offsets).to(device=device),
        candidate_projection_valid=torch.from_numpy(projection_valid).to(device=device),
        max_abs_log_likelihood_ratio=float(spatial_config["max_abs_pose_log_ratio"]),
    )
    control_spatial_score = score_candidate_pose_rgb_spatial_batch(
        runtime=permuted_runtime,
        prediction=control_spatial_prediction,
        candidate_projection_offsets_xy=torch.from_numpy(projection_offsets).to(device=device),
        candidate_projection_valid=torch.from_numpy(projection_valid).to(device=device),
        max_abs_log_likelihood_ratio=float(spatial_config["max_abs_pose_log_ratio"]),
    )
    evidence = CandidatePoseRGBSpatialBridgeQueryEvidence(
        query_id=query_id,
        spatial_candidate_llrs=spatial_score.candidate_log_likelihood_ratios.detach()
        .cpu()
        .numpy(),
        control_spatial_candidate_llrs=control_spatial_score.candidate_log_likelihood_ratios.detach()
        .cpu()
        .numpy(),
        identity_candidate_llrs=identity_candidate.detach().cpu().numpy(),
        control_identity_candidate_llrs=control_identity_candidate.detach().cpu().numpy(),
        candidate_probabilities=runtime.candidate_probabilities.detach().cpu().numpy(),
        null_probabilities=runtime.null_probabilities.detach().cpu().numpy(),
    )
    if evidence.spatial_candidate_llrs.shape[1] != len(source_point_ids):
        raise RuntimeError("bridge visual output rows drifted from frozen P1 points")
    return evidence


def _source_lineage(
    *,
    radio_final_context_cache: Path,
    radio_intermediate_context_cache: Path,
    alike_spatial_context_cache: Path,
    source_metadata: Mapping[str, object],
    rgb_coordinate_bridge: Mapping[str, object],
) -> dict[str, object]:
    return {
        "radio_final_context_cache_sha256": file_sha256_short(radio_final_context_cache),
        "radio_intermediate_context_cache_sha256": file_sha256_short(
            radio_intermediate_context_cache
        ),
        "alike_spatial_context_cache_sha256": file_sha256_short(alike_spatial_context_cache),
        "source_image_manifest_sha256": str(
            source_metadata.get("source_image_manifest_sha256", "")
        ),
        "rgb_coordinate_bridge": dict(rgb_coordinate_bridge),
    }


def _bridge_feature_lineage(
    *,
    layout_path: Path,
    targets_path: Path,
    rgb_spatial_checkpoint: Path,
    identity_checkpoint: Path,
    source_lineage: Mapping[str, object],
    candidate_count: int,
    support_view_count: int,
) -> dict[str, object]:
    return {
        "layout_sha256": file_sha256_short(layout_path),
        "training_targets_sha256": file_sha256_short(targets_path),
        "rgb_spatial_checkpoint_sha256": file_sha256_short(rgb_spatial_checkpoint),
        "identity_checkpoint_sha256": file_sha256_short(identity_checkpoint),
        "source_lineage": dict(source_lineage),
        "candidate_count": int(candidate_count),
        "support_view_count": int(support_view_count),
        "identity_source_scales": dict(_RADIO_FINAL_ONLY_SCALES),
    }


def audit_p1_rgb_radio_bridge_crossfit(args: argparse.Namespace) -> dict[str, object]:
    """Generate frozen train-only feature shards and cross-fit bridge weights."""

    paths = {
        "rgb_spatial_checkpoint": Path(args.rgb_spatial_checkpoint),
        "identity_checkpoint": Path(args.identity_checkpoint),
        "layout": Path(args.rgb_spatial_layout),
        "targets": Path(args.training_targets),
        "radio_final": Path(args.radio_final_context_cache),
        "radio_intermediate": Path(args.radio_intermediate_context_cache),
        "alike": Path(args.alike_spatial_context_cache),
        "image_root": Path(args.image_root),
    }
    output_dir = Path(args.output_dir)
    if (
        any(not path.exists() for path in paths.values())
        or int(args.crossfit_fold_count) < 2
        or int(args.permutation_control_shift) == 0
        or float(args.rgb_cache_gb) <= 0.0
        or not math.isfinite(float(args.catastrophic_gap_threshold))
        or float(args.minimum_normal_gap) < 0.0
        or not 0.0 < float(args.minimum_win_fraction) <= 1.0
        or float(args.minimum_visual_gap_delta) < 0.0
    ):
        raise ValueError("RGB/RADIO bridge audit arguments are invalid")
    if output_dir.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite bridge audit: {output_dir}")

    state = _initialize_distributed(str(args.device))
    try:
        if state.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "training_features").mkdir(parents=True, exist_ok=True)
        if state.enabled:
            distributed.barrier()

        layout = load_candidate_pose_rgb_spatial_layout(paths["layout"])
        targets = load_candidate_pose_rgb_spatial_training_targets(paths["targets"])
        layout_sha256 = file_sha256_short(paths["layout"])
        targets_sha256 = file_sha256_short(paths["targets"])
        validate_training_layout_and_targets(
            layout=layout, targets=targets, layout_sha256=layout_sha256
        )
        groups = build_train_query_groups(layout=layout, targets=targets)
        query_ids = tuple(sorted(groups))
        if int(args.crossfit_fold_count) > len(query_ids):
            raise ValueError("bridge cross-fit has more folds than train queries")
        if any(int(group.point_count) != 32 for group in groups.values()):
            raise ValueError("bridge requires the frozen 32-point target-free P1 layout")

        headers = load_context_attention_source_headers(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            expected_radio_checkpoint="",
        )
        sources = load_context_attention_sources(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            expected_radio_checkpoint="",
            require_equal_descriptor_dimensions=False,
        )
        image_ids, image_sizes, source_tensors = _source_table(sources)
        if not (
            np.array_equal(np.asarray(headers.image_ids).astype(str), image_ids)
            and np.array_equal(np.asarray(headers.image_sizes, dtype=np.int64), image_sizes)
        ):
            raise ValueError("bridge source headers differ from full context maps")
        unique_sizes = np.unique(image_sizes, axis=0)
        if unique_sizes.shape != (1, 2):
            raise ValueError("bridge requires common aligned source image dimensions")
        coordinate_image_size = (int(unique_sizes[0, 0]), int(unique_sizes[0, 1]))
        rgb_image_size = _discover_rgb_image_size(
            image_root=paths["image_root"], image_id=str(image_ids[0])
        )
        rgb_bridge = validate_rgb_coordinate_bridge(
            source_metadata=sources[0].metadata,
            coordinate_image_size=coordinate_image_size,
            rgb_image_size=rgb_image_size,
        )
        source_lineage = _source_lineage(
            radio_final_context_cache=paths["radio_final"],
            radio_intermediate_context_cache=paths["radio_intermediate"],
            alike_spatial_context_cache=paths["alike"],
            source_metadata=sources[0].metadata,
            rgb_coordinate_bridge=rgb_bridge,
        )
        if not str(source_lineage["source_image_manifest_sha256"]):
            raise ValueError("bridge source caches lack an image-manifest hash")
        spatial_model, spatial_metadata = _load_rgb_spatial_checkpoint(
            path=paths["rgb_spatial_checkpoint"],
            layout=layout,
            targets=targets,
            context_source_dimensions=headers.descriptor_dimensions,
            image_sizes=image_sizes,
            source_lineage=source_lineage,
            layout_sha256=layout_sha256,
            targets_sha256=targets_sha256,
            device=state.device,
        )
        identity_model, identity_metadata, identity_lineage_contract = (
            _load_radio_final_identity_checkpoint(
                path=paths["identity_checkpoint"],
                source_tensors=source_tensors,
                image_sizes=image_sizes,
                layout=layout,
                source_lineage=source_lineage,
                device=state.device,
            )
        )
        complete_runtime = runtime_from_target_free_layout(layout, image_ids=image_ids)
        feature_lineage = _bridge_feature_lineage(
            layout_path=paths["layout"],
            targets_path=paths["targets"],
            rgb_spatial_checkpoint=paths["rgb_spatial_checkpoint"],
            identity_checkpoint=paths["identity_checkpoint"],
            source_lineage=source_lineage,
            candidate_count=layout.candidate_count,
            support_view_count=layout.support_view_count,
        )
        cache = TensorImageLRUCache(
            max_bytes=int(float(args.rgb_cache_gb) * 1024**3),
            storage_dtype=resolve_rgb_image_cache_storage_dtype(args.rgb_cache_dtype),
        )
        amp_enabled = state.device.type == "cuda" and not bool(args.no_amp)

        # Partition immutable queries across the two GPUs.  Each process emits
        # a self-validating feature shard, avoiding variable-length all-gather.
        local_query_count = sum(
            position % state.world_size == state.rank
            for position in range(len(query_ids))
        )
        local_completed = 0
        for query_position, query_id in enumerate(query_ids):
            if query_position % state.world_size != state.rank:
                continue
            group = groups[query_id]
            evidence = _forward_query_target_free(
                group=group,
                complete_runtime=complete_runtime,
                image_ids=image_ids,
                image_root=paths["image_root"],
                coordinate_image_size=coordinate_image_size,
                rgb_image_size=rgb_image_size,
                spatial_model=spatial_model,
                spatial_config=spatial_metadata["config"],
                identity_model=identity_model,
                cache=cache,
                device=state.device,
                amp_enabled=amp_enabled,
                permutation_shift=int(args.permutation_control_shift),
            )
            _write_training_feature(
                path=output_dir / "training_features" / _feature_filename(query_id),
                evidence=evidence,
                source_point_ids=np.asarray(group.source_point_ids, dtype=np.int64),
                lineage=feature_lineage,
            )
            local_completed += 1
            if local_completed == 1 or local_completed % 4 == 0 or local_completed == local_query_count:
                print(
                    json.dumps(
                        {
                            "stage": "frozen_visual_feature_shard",
                            "rank": int(state.rank),
                            "completed_queries": int(local_completed),
                            "local_query_count": int(local_query_count),
                            "query_id": str(query_id),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if state.enabled:
            distributed.barrier()
        output: dict[str, object] = {
            "rank": int(state.rank),
            "output_dir": str(output_dir),
            "world_size": int(state.world_size),
        }
        if state.rank == 0:
            evidences = [
                _load_training_feature(
                    path=output_dir / "training_features" / _feature_filename(query_id),
                    expected_query_id=query_id,
                    expected_lineage=feature_lineage,
                )
                for query_id in query_ids
            ]
            baseline_rows = _score_rows(evidences=evidences, weights=_BASELINE_WEIGHTS)
            baseline_summary = _profile_summary(
                baseline_rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
            )
            candidates = _candidate_weight_grid()
            oof_rows, selections = crossfit_bridge_profiles(
                evidences=evidences,
                candidates=candidates,
                fold_count=int(args.crossfit_fold_count),
            )
            bridge_summary = _profile_summary(
                oof_rows, catastrophic_threshold=float(args.catastrophic_gap_threshold)
            )
            paired = _paired_bridge_comparison(
                candidate_rows=oof_rows, baseline_rows=baseline_rows
            )
            gate = _bridge_gate(
                bridge_summary=bridge_summary,
                baseline_summary=baseline_summary,
                paired=paired,
                minimum_normal_gap=float(args.minimum_normal_gap),
                minimum_win_fraction=float(args.minimum_win_fraction),
                minimum_visual_gap_delta=float(args.minimum_visual_gap_delta),
            )
            result: dict[str, object] = {
                "format": AUDIT_FORMAT,
                "bridge_format": "candidate_pose_rgb_spatial_bridge_v1",
                "stage": "frozen_p1_rgb_spatial_plus_radio_final_bridge_query_grouped_crossfit",
                "output_dir": str(output_dir),
                "world_size": int(state.world_size),
                "query_count": int(len(query_ids)),
                "query_ids": list(query_ids),
                "feature_lineage": feature_lineage,
                "checkpoints": {
                    "rgb_spatial": {
                        "path": str(paths["rgb_spatial_checkpoint"]),
                        "sha256": file_sha256_short(paths["rgb_spatial_checkpoint"]),
                        "hard_repeat_component_gate_passed": True,
                        "full_pose_gate_passed": False,
                    },
                    "radio_final_identity": {
                        "path": str(paths["identity_checkpoint"]),
                        "sha256": file_sha256_short(paths["identity_checkpoint"]),
                        "broad_pretrain_gate_passed": True,
                        "source_lineage_contract": identity_lineage_contract,
                        "p1_zero_shot_full_fusion_promoted": False,
                    },
                },
                "frozen_expert_contract": {
                    "spatial": "real_rgb_high_resolution_raw_cost_volume_only_v1",
                    "identity": "full_2d_radio_final_candidate_specific_context_only_v1",
                    "identity_source_scales": dict(_RADIO_FINAL_ONLY_SCALES),
                    "candidate_mixture": "fixed_global_top20_plus_immutable_explicit_null_v1",
                    "support_views": "fixed_two_real_sfm_observations_v1",
                },
                "crossfit": {
                    "fold_count": int(args.crossfit_fold_count),
                    "candidate_profile_count": int(len(candidates)),
                    "selection_uses_normal_train_fold_only": True,
                    "fold_selections": selections,
                    "oof_rows": oof_rows,
                },
                "rgb_spatial_baseline": {
                    "weights": _BASELINE_WEIGHTS.as_dict(),
                    "rows": baseline_rows,
                    "summary": baseline_summary,
                },
                "oof_bridge": {"summary": bridge_summary, "gate": gate},
                "rgb_cache_rank0": cache.summary(),
                "protocol": {
                    "diagnostic_only": True,
                    "model_weights_updated": False,
                    "runtime_layout_target_free": True,
                    "target_join_after_visual_inference": True,
                    "train_query_only": True,
                    "heldout_validation_or_test_not_run": True,
                    "pnp_or_pose_estimation_run": False,
                    "bridge_is_not_a_runtime_checkpoint": True,
                    "no_render": True,
                    "no_image_retrieval_or_submap": True,
                    "out_of_window_projection": "fixed_neutral_missing_edge_not_learned_dustbin_v1",
                },
            }
            _write_json_atomically(output_dir / "audit.json", result)
            output = {"rank": int(state.rank), "output_dir": str(output_dir), "gate": gate}
        return output
    finally:
        _finalize_distributed(state)


def main(argv: Sequence[str] | None = None) -> None:
    result = audit_p1_rgb_radio_bridge_crossfit(parse_args(argv))
    if int(result["rank"]) == 0:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
