"""Evaluate zero-fit hierarchy-footprint and GT-visible geometry controls.

This diagnostic asks which observation is missing from the bounded local
transport objective.  The deployable control compares the RADIO retrieval
mass and rendered target mass after aggregation by parent (coarse), connected
support (medium), or exact child (fine).  The oracle control replaces RADIO by
candidate-zero's GT render and additionally tests absolute metric log depth.

No parameter is fitted.  Score conjunctions are the predeclared product and
minimum of bounded evidence terms, so development labels cannot tune a weight.
The candidate stencil and the oracle itself remain GT-relative diagnostics and
can reject an objective family but cannot authorize a production optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_controlled_6dof_basin_gate import (
    _direction_failures,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_symmetric_coverage_control import (
    _propagated_reliability,
    _symmetric_coverage_scores,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
    _dense_fixed_identity_query_component_statistics,
    _load_dataset,
    _metrics,
    _release_streaming_cuda_cache,
    _validate_scientific_dataset_contract,
)
from feature_extract.vfm.localization.surface_maplet_mapper import (
    load_surface_maplet_mapper,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
)
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.dense_fixed_identity_transport import (
    DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
    DensePoseTransportHierarchyGPU,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    GoalMapletPhysicalMap,
)
from feature_extract.vfm.localization_goal_maplet.structured_token_child_posterior import (
    STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS,
    structured_token_child_probabilities,
)
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    load_minimal_pose_transport_readout,
    pose_transport_model_content_sha256,
)


SCHEMA = "goal_maplet_hierarchy_footprint_observability_control_v1"
DEPTH_SCALE = 0.25


def _stage_child_group_ids(hierarchy: PoseTransportHierarchy, stage: str) -> np.ndarray:
    """Return the physical identity used by one predeclared hierarchy stage."""

    parent = np.asarray(hierarchy.child_parent_ids, dtype=np.int64).reshape(-1)
    support = np.asarray(hierarchy.child_support_ids, dtype=np.int64).reshape(-1)
    if parent.size == 0 or support.shape != parent.shape or np.any(parent < 0):
        raise ValueError("hierarchy identity arrays differ")
    if stage == "coarse":
        result = parent
    elif stage == "medium":
        result = support
    elif stage == "fine":
        result = np.arange(parent.size, dtype=np.int64)
    else:
        raise ValueError("footprint stage must be coarse, medium, or fine")
    if np.any(result < 0):
        raise ValueError(f"{stage} hierarchy does not cover every physical child")
    return result


def _identity_mass_histogram(
    child_rows: np.ndarray,
    child_mass: np.ndarray,
    token_reliability: np.ndarray,
    child_group_ids: np.ndarray,
) -> np.ndarray:
    """Aggregate reliable projected mass without candidate normalization."""

    rows = np.asarray(child_rows, dtype=np.int64)
    mass = np.asarray(child_mass, dtype=np.float64)
    reliability = np.asarray(token_reliability, dtype=np.float64).reshape(-1)
    groups = np.asarray(child_group_ids, dtype=np.int64).reshape(-1)
    if (
        rows.ndim != 2 or mass.shape != rows.shape
        or reliability.shape != (rows.shape[0],) or groups.size == 0
        or np.any(~np.isfinite(mass)) or np.any(mass < 0.0)
        or np.any(~np.isfinite(reliability)) or np.any(reliability < 0.0)
        or np.any(groups < 0)
    ):
        raise ValueError("identity footprint inputs differ")
    valid = (rows >= 0) & (mass > 0.0)
    if np.any(rows[valid] >= groups.size):
        raise ValueError("identity footprint child row is out of range")
    safe = np.maximum(rows, 0)
    weighted = mass * reliability[:, None]
    return np.bincount(
        groups[safe][valid], weights=weighted[valid],
        minlength=int(groups.max()) + 1,
    ).astype(np.float64, copy=False)


def _dice_from_histograms(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Bounded mass Dice; missing and surplus mass remain in the denominator."""

    left = np.asarray(reference, dtype=np.float64).reshape(-1)
    right = np.asarray(candidate, dtype=np.float64).reshape(-1)
    length = max(left.size, right.size)
    if length == 0 or np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("footprint histograms differ")
    left = np.pad(left, (0, length - left.size))
    right = np.pad(right, (0, length - right.size))
    denominator = float(left.sum() + right.sum())
    if denominator <= 1.0e-12:
        return 0.0
    value = float(2.0 * np.minimum(left, right).sum() / denominator)
    if not np.isfinite(value) or value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise AssertionError("footprint Dice is outside its capacity bound")
    return float(np.clip(value, 0.0, 1.0))


def _deployable_hierarchy_footprint_scores(
    source_rows: np.ndarray,
    source_probability: np.ndarray,
    query_reliability: np.ndarray,
    target_rows: np.ndarray,
    target_weight: np.ndarray,
    candidate_valid: np.ndarray,
    hierarchy: PoseTransportHierarchy,
    *,
    stage: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compare RADIO q_ret and target projection at the stage identity scale."""

    groups = _stage_child_group_ids(hierarchy, stage)
    query_histogram = _identity_mass_histogram(
        source_rows, source_probability, query_reliability, groups,
    )
    propagated = _propagated_reliability(query_reliability, stage=stage)
    rows = np.asarray(target_rows, dtype=np.int64)
    weights = np.asarray(target_weight, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    if rows.ndim != 3 or weights.shape != rows.shape or valid.shape != (rows.shape[0],):
        raise ValueError("candidate footprint inputs differ")
    values, target_totals = [], []
    for candidate in range(rows.shape[0]):
        if not valid[candidate]:
            values.append(0.0); target_totals.append(0.0)
            continue
        target_histogram = _identity_mass_histogram(
            rows[candidate], weights[candidate], propagated, groups,
        )
        values.append(_dice_from_histograms(query_histogram, target_histogram))
        target_totals.append(float(target_histogram.sum()))
    f_score = np.asarray(values, dtype=np.float64)
    score = -1.0 + 2.0 * f_score
    return score.astype(np.float32), {
        "stage_identity": {
            "coarse": "parent_maplet",
            "medium": "connected_surface_support",
            "fine": "exact_surface_child",
        }[stage],
        "identity_count": int(np.unique(groups).size),
        "query_reliable_mass": float(query_histogram.sum()),
        "minimum_candidate_reliable_mass": float(np.min(np.asarray(target_totals)[valid])),
        "maximum_candidate_reliable_mass": float(np.max(np.asarray(target_totals)[valid])),
        "minimum_footprint_f": float(np.min(f_score)),
        "maximum_footprint_f": float(np.max(f_score)),
    }


def _deployable_spatial_hierarchy_footprint_scores(
    source_rows: np.ndarray,
    source_probability: np.ndarray,
    query_reliability: np.ndarray,
    target_rows: np.ndarray,
    target_weight: np.ndarray,
    candidate_valid: np.ndarray,
    hierarchy: PoseTransportHierarchy,
    *,
    stage: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compare deployable q_ret and render in token-by-stage-identity space."""

    groups = _stage_child_group_ids(hierarchy, stage)
    source_depth = np.zeros_like(source_probability, dtype=np.float64)
    source_profile = _spatial_group_profile(
        source_rows, source_probability, query_reliability, groups,
        source_depth, np.zeros_like(source_probability, dtype=bool),
    )
    rows = np.asarray(target_rows, dtype=np.int64)
    weights = np.asarray(target_weight, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    if rows.ndim != 3 or weights.shape != rows.shape or valid.shape != (rows.shape[0],):
        raise ValueError("deployable spatial footprint candidate arrays differ")
    values, target_totals = [], []
    for candidate in range(rows.shape[0]):
        if not valid[candidate]:
            values.append(0.0); target_totals.append(0.0)
            continue
        target_profile = _spatial_group_profile(
            rows[candidate], weights[candidate], query_reliability, groups,
            np.zeros_like(weights[candidate]), np.zeros_like(weights[candidate], dtype=bool),
        )
        footprint, _ = _oracle_profile_agreement(source_profile, target_profile)
        values.append(footprint); target_totals.append(float(np.sum(target_profile[1])))
    f_score = np.asarray(values, dtype=np.float64)
    return (-1.0 + 2.0 * f_score).astype(np.float32), {
        "stage_identity": {
            "coarse": "parent_maplet",
            "medium": "connected_surface_support",
            "fine": "exact_surface_child",
        }[stage],
        "spatial_identity": "exact_RADIO_token_x_stage_physical_identity",
        "query_source": "existing_pose_free_RADIO_q_ret",
        "uses_GT_query_observation": False,
        "shared_query_only_reliability_on_query_and_target_tokens": True,
        "candidate_normalization": False,
        "query_reliable_mass": float(np.sum(source_profile[1])),
        "minimum_candidate_reliable_mass": float(np.min(np.asarray(target_totals)[valid])),
        "maximum_candidate_reliable_mass": float(np.max(np.asarray(target_totals)[valid])),
        "minimum_footprint_f": float(np.min(f_score)),
        "maximum_footprint_f": float(np.max(f_score)),
    }


def _metric_log_depth(
    child_rows: np.ndarray,
    pose_w2c: np.ndarray,
    child_centers_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct absolute log depth from immutable map geometry and pose."""

    rows = np.asarray(child_rows, dtype=np.int64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    centers = np.asarray(child_centers_world, dtype=np.float64)
    if rows.ndim != 2 or pose.shape != (4, 4) or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("metric depth inputs differ")
    valid_row = rows >= 0
    if np.any(rows[valid_row] >= centers.shape[0]):
        raise ValueError("metric depth child row is out of range")
    safe = np.maximum(rows, 0)
    camera = np.einsum("ij,tsj->tsi", pose[:3, :3], centers[safe]) + pose[:3, 3]
    valid = valid_row & np.isfinite(camera[..., 2]) & (camera[..., 2] > 1.0e-6)
    result = np.zeros(rows.shape, dtype=np.float64)
    result[valid] = np.log(camera[..., 2][valid])
    return result, valid


def _spatial_group_profile(
    child_rows: np.ndarray,
    child_mass: np.ndarray,
    token_reliability: np.ndarray,
    child_group_ids: np.ndarray,
    metric_log_depth: np.ndarray,
    metric_depth_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse token-by-identity mass and depth moments."""

    rows = np.asarray(child_rows, dtype=np.int64)
    mass = np.asarray(child_mass, dtype=np.float64)
    reliability = np.asarray(token_reliability, dtype=np.float64).reshape(-1)
    groups = np.asarray(child_group_ids, dtype=np.int64).reshape(-1)
    depth = np.asarray(metric_log_depth, dtype=np.float64)
    depth_valid = np.asarray(metric_depth_valid, dtype=bool)
    if (
        rows.ndim != 2 or mass.shape != rows.shape or depth.shape != rows.shape
        or depth_valid.shape != rows.shape or reliability.shape != (rows.shape[0],)
    ):
        raise ValueError("spatial footprint inputs differ")
    row_valid = (rows >= 0) & (mass > 0.0)
    if np.any(rows[row_valid] >= groups.size):
        raise ValueError("spatial footprint child row is out of range")
    safe = np.maximum(rows, 0)
    group_count = int(groups.max()) + 1
    token = np.broadcast_to(np.arange(rows.shape[0])[:, None], rows.shape)
    keys = token * group_count + groups[safe]
    weighted = mass * reliability[:, None]
    flat_keys = keys[row_valid]
    unique, inverse = np.unique(flat_keys, return_inverse=True)
    total_mass = np.bincount(inverse, weights=weighted[row_valid], minlength=unique.size)
    valid_depth = row_valid & depth_valid
    depth_mass = np.zeros(unique.size, dtype=np.float64)
    depth_mean = np.zeros(unique.size, dtype=np.float64)
    if np.any(valid_depth):
        positions = np.searchsorted(unique, keys[valid_depth])
        depth_mass = np.bincount(
            positions, weights=weighted[valid_depth], minlength=unique.size,
        )
        depth_sum = np.bincount(
            positions, weights=weighted[valid_depth] * depth[valid_depth],
            minlength=unique.size,
        )
        np.divide(depth_sum, depth_mass, out=depth_mean, where=depth_mass > 1.0e-12)
    return unique, total_mass, depth_mass, depth_mean


def _oracle_profile_agreement(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    depth_scale: float = DEPTH_SCALE,
) -> tuple[float, float]:
    """Return projected-footprint Dice and its metric-depth-compatible subset."""

    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("oracle depth scale must be positive")
    ref_key, ref_mass, ref_depth_mass, ref_depth = reference
    can_key, can_mass, can_depth_mass, can_depth = candidate
    _, ref_index, can_index = np.intersect1d(
        ref_key, can_key, assume_unique=True, return_indices=True,
    )
    denominator = float(np.sum(ref_mass) + np.sum(can_mass))
    if denominator <= 1.0e-12:
        return 0.0, 0.0
    overlap = np.minimum(ref_mass[ref_index], can_mass[can_index])
    footprint = float(2.0 * np.sum(overlap) / denominator)
    valid_depth_overlap = np.minimum(
        ref_depth_mass[ref_index], can_depth_mass[can_index]
    )
    compatibility = np.exp(
        -np.abs(ref_depth[ref_index] - can_depth[can_index]) / float(depth_scale)
    )
    depth_footprint = float(
        2.0 * np.sum(valid_depth_overlap * compatibility) / denominator
    )
    if (
        footprint < -1.0e-12 or footprint > 1.0 + 1.0e-12
        or depth_footprint < -1.0e-12 or depth_footprint > footprint + 1.0e-12
    ):
        raise AssertionError("oracle footprint violates its mass bound")
    return float(np.clip(footprint, 0.0, 1.0)), float(np.clip(depth_footprint, 0.0, 1.0))


def _gt_visible_oracle_scores(
    target_rows: np.ndarray,
    target_weight: np.ndarray,
    candidate_poses_w2c: np.ndarray,
    candidate_valid: np.ndarray,
    query_reliability: np.ndarray,
    hierarchy: PoseTransportHierarchy,
    child_centers_world: np.ndarray,
    *,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Compare every render to candidate-zero's exact visible footprint/depth."""

    rows = np.asarray(target_rows, dtype=np.int64)
    weights = np.asarray(target_weight, dtype=np.float64)
    poses = np.asarray(candidate_poses_w2c, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    if (
        rows.ndim != 3 or weights.shape != rows.shape
        or poses.shape != (rows.shape[0], 4, 4) or valid.shape != (rows.shape[0],)
        or not valid[0]
    ):
        raise ValueError("GT-visible oracle candidate arrays differ")
    groups = _stage_child_group_ids(hierarchy, stage)
    propagated = _propagated_reliability(query_reliability, stage=stage)
    reference_depth, reference_valid = _metric_log_depth(
        rows[0], poses[0], child_centers_world,
    )
    reference = _spatial_group_profile(
        rows[0], weights[0], propagated, groups,
        reference_depth, reference_valid,
    )
    footprint_values, depth_values = [], []
    for candidate in range(rows.shape[0]):
        if not valid[candidate]:
            footprint_values.append(0.0); depth_values.append(0.0)
            continue
        log_depth, depth_valid = _metric_log_depth(
            rows[candidate], poses[candidate], child_centers_world,
        )
        profile = _spatial_group_profile(
            rows[candidate], weights[candidate], propagated, groups,
            log_depth, depth_valid,
        )
        footprint, depth_footprint = _oracle_profile_agreement(reference, profile)
        footprint_values.append(footprint); depth_values.append(depth_footprint)
    footprint_f = np.asarray(footprint_values, dtype=np.float64)
    depth_f = np.asarray(depth_values, dtype=np.float64)
    if abs(footprint_f[0] - 1.0) > 1.0e-10 or abs(depth_f[0] - 1.0) > 1.0e-10:
        raise AssertionError("candidate-zero oracle does not reproduce itself")
    return (
        (-1.0 + 2.0 * footprint_f).astype(np.float32),
        (-1.0 + 2.0 * depth_f).astype(np.float32),
        {
            "reference_candidate_index": 0,
            "reference_is_GT_relative_oracle": True,
            "spatial_identity": "token_x_stage_physical_identity",
            "absolute_metric_log_depth_reconstructed_from_map_geometry": True,
            "metric_log_depth_scale": float(DEPTH_SCALE),
            "minimum_projected_footprint_f": float(np.min(footprint_f)),
            "minimum_metric_depth_footprint_f": float(np.min(depth_f)),
        },
    )


def _bounded_conjunction(left_score: np.ndarray, right_score: np.ndarray, rule: str) -> np.ndarray:
    """Combine two [-1,1] scores without a fitted or selected weight."""

    left = np.asarray(left_score, dtype=np.float64)
    right = np.asarray(right_score, dtype=np.float64)
    if left.shape != right.shape or np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("bounded conjunction inputs differ")
    left_f = 0.5 * (left + 1.0)
    right_f = 0.5 * (right + 1.0)
    if (
        np.min(left_f) < -2.0e-4 or np.max(left_f) > 1.0 + 2.0e-4
        or np.min(right_f) < -2.0e-4 or np.max(right_f) > 1.0 + 2.0e-4
    ):
        raise AssertionError("bounded conjunction input is outside [-1,1]")
    left_f = np.clip(left_f, 0.0, 1.0); right_f = np.clip(right_f, 0.0, 1.0)
    if rule == "product":
        combined = left_f * right_f
    elif rule == "minimum":
        combined = np.minimum(left_f, right_f)
    else:
        raise ValueError("conjunction rule must be product or minimum")
    return (-1.0 + 2.0 * combined).astype(np.float32)


def _shape_gate(metrics: dict[str, object]) -> bool:
    failures = _direction_failures(metrics)
    return bool(
        metrics["gt_anchor_top1_rate"] == 1.0
        and metrics["mean_gt_anchor_margin_over_best_nonanchor"] > 0.0
        and failures["all_42_signed_directions_strictly_monotonic"]
    )


def _flatten_42_signed_ray_metrics(metrics: dict[str, object]) -> list[dict[str, object]]:
    """Flatten the authoritative 21 directions into 42 explicit signed rays."""

    capture = metrics.get("full_6dof_directional_capture")
    if not isinstance(capture, dict) or not isinstance(capture.get("per_direction"), list):
        raise ValueError("full 6-DoF directional capture is absent")
    result = []
    for direction in capture["per_direction"]:
        for sign, source in (("negative", direction["negative_sign"]),
                             ("positive", direction["positive_sign"])):
            result.append({
                "direction_id": int(direction["direction_id"]),
                "kind": str(direction["kind"]),
                "label": str(direction["label"]),
                "sign": sign,
                "radial_pair_correct": int(source["radial_pair_correct"]),
                "radial_pair_count": int(source["radial_pair_count"]),
                "radial_pair_accuracy": float(source["radial_pair_accuracy"]),
                "complete_path_count": int(source["complete_path_count"]),
                "signed_path_count": int(source["signed_path_count"]),
                "complete_path_rate": float(source["complete_path_rate"]),
                "strictly_monotonic_for_every_query": bool(
                    source["complete_path_count"] == source["signed_path_count"]
                ),
            })
    if len(result) != 42:
        raise ValueError("controlled stencil does not expose exactly 42 signed rays")
    return result


def _payload_content_sha256(payload: dict[str, object]) -> str:
    """Hash a report payload before adding its non-self-referential hash field."""

    if "report_payload_content_sha256" in payload:
        raise ValueError("payload hash field must be absent while hashing")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signed_direction_per_query_audit(
    scores: np.ndarray,
    image_ids: np.ndarray,
    radial_paths: np.ndarray,
    candidate_direction_ids: np.ndarray,
    candidate_signs: np.ndarray,
    *,
    direction_id: int,
) -> dict[str, object]:
    """Expose every query's two strict comparisons on one signed direction."""

    values = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(image_ids).reshape(-1)
    paths = np.asarray(radial_paths, dtype=np.int64)
    directions = np.asarray(candidate_direction_ids, dtype=np.int64).reshape(-1)
    signs = np.asarray(candidate_signs, dtype=np.int64).reshape(-1)
    if (
        values.ndim != 2 or identifiers.shape != (values.shape[0],)
        or paths.ndim != 2 or paths.shape[1] < 2
        or directions.shape != (values.shape[1],) or signs.shape != directions.shape
    ):
        raise ValueError("signed direction audit arrays differ")
    result: dict[str, object] = {"direction_id": int(direction_id)}
    for sign, label in ((-1, "negative"), (1, "positive")):
        matches = []
        for path in paths:
            tail = path[1:]
            if np.all(directions[tail] == int(direction_id)) and np.all(signs[tail] == sign):
                matches.append(np.asarray(path, dtype=np.int64))
        if len(matches) != 1:
            raise ValueError("signed direction audit does not have exactly one radial path")
        path = matches[0]
        query_rows = []
        for image_id, query_scores in zip(identifiers.tolist(), values):
            path_scores = query_scores[path]
            pair_margins = path_scores[:-1] - path_scores[1:]
            query_rows.append({
                "image_id": str(image_id),
                "candidate_indices": [int(value) for value in path.tolist()],
                "scores_GT_half_full": [float(value) for value in path_scores.tolist()],
                "strict_inward_pair_margins": [float(value) for value in pair_margins.tolist()],
                "pair_correct": [bool(value > 0.0) for value in pair_margins.tolist()],
                "complete_path": bool(np.all(pair_margins > 0.0)),
            })
        result[label] = query_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense_candidate_batch_size", type=int, default=8)
    args = parser.parse_args()
    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite hierarchy footprint control")
    if int(args.dense_candidate_batch_size) <= 0:
        raise ValueError("dense candidate batch size must be positive")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    arrays, metadata = _load_dataset(Path(args.dataset))
    scientific_audit = _validate_scientific_dataset_contract(arrays, metadata)
    if scientific_audit["full_6dof_observability_stencil"] is not True:
        raise ValueError("hierarchy footprint control requires complete 6-DoF stencil")
    model, model_metadata = load_minimal_pose_transport_readout(
        Path(args.model), device=str(device),
    )
    if (
        model_metadata.get("dataset_content_sha256") != metadata["content_sha256"]
        or model_metadata.get("transport_semantics")
        != FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
        or model_metadata.get("readout_training_semantics")
        != IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS
        or model_metadata.get("effective_trainable_scalar_count") != 2
    ):
        raise ValueError("footprint model and controlled dataset differ")
    mapper_path = Path(args.surface_mapper)
    if file_sha256(mapper_path) != model_metadata.get("surface_mapper_file_sha256"):
        raise ValueError("footprint surface mapper differs from trained model")
    physical_path = Path(args.physical_map)
    if file_sha256(physical_path) != metadata.get("physical_map_file_sha256"):
        raise ValueError("footprint physical map file lineage differs")
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    if physical.content_sha256 != metadata.get("physical_map_sha256"):
        raise ValueError("footprint physical map content lineage differs")

    mapper, _ = load_surface_maplet_mapper(mapper_path, device=str(device))
    mapper.model.to(device).eval()
    mapped_rows = []
    with torch.no_grad():
        for query in range(int(arrays["image_ids"].size)):
            mapped_rows.append(mapper.model(torch.as_tensor(
                arrays["radio_final"][query], device=device, dtype=torch.float32,
            )[None])[0].detach().cpu().numpy().astype(np.float32, copy=False))
    arrays["pose_query_features"] = np.stack(mapped_rows)
    hierarchy = PoseTransportHierarchy(
        child_parent_ids=np.asarray(arrays["hierarchy_child_parent_ids"], dtype=np.int64),
        child_support_ids=np.asarray(arrays["hierarchy_child_support_ids"], dtype=np.int64),
        adjacency_offsets=np.asarray(arrays["hierarchy_adjacency_offsets"], dtype=np.int64),
        adjacency_child_rows=np.asarray(
            arrays["hierarchy_adjacency_child_rows"], dtype=np.int64,
        ),
        content_sha256=str(metadata["hierarchy_content_sha256"]),
    )
    if physical.child_centers.shape[0] != hierarchy.child_parent_ids.size:
        raise ValueError("physical map and embedded hierarchy child count differ")
    dense_hierarchy = DensePoseTransportHierarchyGPU(hierarchy, device=device)
    query_count = int(arrays["image_ids"].size)
    structured_source_probabilities = np.stack([
        structured_token_child_probabilities(
            arrays["source_child_rows"][query],
            arrays["source_child_probabilities"][query],
            hierarchy.child_parent_ids, hierarchy.child_support_ids,
        )
        for query in range(query_count)
    ])
    model.eval()
    edge_weights = model.edge_weights().detach().cpu().numpy().astype(np.float64)
    train_count = len(model_metadata["train_image_ids"])
    candidate_valid = np.asarray(arrays["candidate_valid"], dtype=bool)
    score_rows: dict[str, tuple[int, dict[str, np.ndarray]]] = {}
    audits, resources = [], []
    started = time.monotonic()
    for stage in ("medium", "coarse", "fine"):
        begin = 0 if stage == "medium" else train_count
        rows_by_signal: dict[str, list[np.ndarray]] = {}
        for query in range(begin, query_count):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            statistics, resource_audit = _dense_fixed_identity_query_component_statistics(
                model, arrays, query, dense_hierarchy, stage=stage, device=device,
                candidate_batch_size=int(args.dense_candidate_batch_size),
            )
            _release_streaming_cuda_cache(device, resource_audit)
            _, local, mass_audit = _symmetric_coverage_scores(
                statistics, edge_weights,
                arrays["source_child_probabilities"][query],
                arrays["query_reliability"][query],
                arrays["target_child_weights"][query],
                candidate_valid[query], stage=stage,
            )
            deployable, deployable_audit = _deployable_hierarchy_footprint_scores(
                arrays["source_child_rows"][query],
                arrays["source_child_probabilities"][query],
                arrays["query_reliability"][query],
                arrays["target_child_rows"][query],
                arrays["target_child_weights"][query],
                candidate_valid[query], hierarchy, stage=stage,
            )
            deployable_spatial, deployable_spatial_audit = (
                _deployable_spatial_hierarchy_footprint_scores(
                    arrays["source_child_rows"][query],
                    arrays["source_child_probabilities"][query],
                    arrays["query_reliability"][query],
                    arrays["target_child_rows"][query],
                    arrays["target_child_weights"][query],
                    candidate_valid[query], hierarchy, stage=stage,
                )
            )
            structured_spatial, structured_spatial_audit = (
                _deployable_spatial_hierarchy_footprint_scores(
                    arrays["source_child_rows"][query],
                    structured_source_probabilities[query],
                    arrays["query_reliability"][query],
                    arrays["target_child_rows"][query],
                    arrays["target_child_weights"][query],
                    candidate_valid[query], hierarchy, stage=stage,
                )
            )
            oracle_footprint, oracle_depth, oracle_audit = _gt_visible_oracle_scores(
                arrays["target_child_rows"][query],
                arrays["target_child_weights"][query],
                arrays["candidate_poses_w2c"][query],
                candidate_valid[query], arrays["query_reliability"][query],
                hierarchy, physical.child_centers, stage=stage,
            )
            signals = {
                "local_symmetric": local,
                "deployable_hierarchy_footprint": deployable,
                "local_x_deployable_product": _bounded_conjunction(local, deployable, "product"),
                "local_x_deployable_minimum": _bounded_conjunction(local, deployable, "minimum"),
                "deployable_qret_spatial_hierarchy_footprint": deployable_spatial,
                "local_x_deployable_qret_spatial_product": _bounded_conjunction(
                    local, deployable_spatial, "product",
                ),
                "local_x_deployable_qret_spatial_minimum": _bounded_conjunction(
                    local, deployable_spatial, "minimum",
                ),
                "deployable_structured_qret_spatial_hierarchy_footprint": structured_spatial,
                "GT_oracle_projected_footprint": oracle_footprint,
                "local_x_GT_oracle_projected_product": _bounded_conjunction(
                    local, oracle_footprint, "product",
                ),
                "GT_oracle_metric_depth_footprint": oracle_depth,
                "local_x_GT_oracle_metric_product": _bounded_conjunction(
                    local, oracle_depth, "product",
                ),
                "local_x_GT_oracle_metric_minimum": _bounded_conjunction(
                    local, oracle_depth, "minimum",
                ),
            }
            for name, score in signals.items():
                rows_by_signal.setdefault(name, []).append(score)
            audits.append({
                "query_index": int(query), "image_id": str(arrays["image_ids"][query]),
                "stage": stage, "local_mass": mass_audit,
                "deployable_footprint": deployable_audit,
                "deployable_qret_spatial_footprint": deployable_spatial_audit,
                "deployable_structured_qret_spatial_footprint": structured_spatial_audit,
                "GT_visible_oracle": oracle_audit,
            })
            resource_audit["query_index"] = int(query); resource_audit["stage"] = stage
            resources.append(resource_audit)
            print(json.dumps({
                "query": str(arrays["image_ids"][query]), "stage": stage,
                "resource": resource_audit,
                "anchor_scores": {name: float(value[0]) for name, value in signals.items()},
            }), flush=True)
        score_rows[stage] = (
            begin, {name: np.stack(values) for name, values in rows_by_signal.items()},
        )

    radial_paths = np.asarray(arrays["controlled_radial_paths"], dtype=np.int64)
    metric_kwargs = {
        "candidate_semantics": str(metadata["candidate_semantics"]),
        "radial_paths": radial_paths,
        "candidate_direction_ids": np.asarray(
            arrays["controlled_candidate_direction_ids"], dtype=np.int64,
        ),
        "candidate_signs": np.asarray(
            arrays["controlled_candidate_signs"], dtype=np.int64,
        ),
        "direction_axis_pairs": np.asarray(
            arrays["controlled_direction_axis_pairs"], dtype=np.int64,
        ),
        "twist_order": tuple(metadata["controlled_pose_stencil_audit"]["twist_order"]),
    }

    def metrics(scores: np.ndarray, begin: int, *, stage: str) -> dict[str, object]:
        end = begin + int(scores.shape[0])
        return _metrics(
            scores, arrays["translation_m"][begin:end], arrays["rotation_deg"][begin:end],
            candidate_valid[begin:end], arrays["image_ids"][begin:end],
            stage=stage, **metric_kwargs,
        )

    dev_metrics: dict[str, dict[str, object]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for stage, (begin, signals) in score_rows.items():
        stage_metrics, stage_gates = {}, {}
        for name, scores in signals.items():
            selected = scores[train_count:] if stage == "medium" else scores
            selected_begin = train_count
            value = metrics(selected, selected_begin, stage=stage)
            stage_metrics[name] = {
                "metrics": value,
                "signed_direction_failures": _direction_failures(value),
            }
            stage_gates[name] = _shape_gate(value)
        dev_metrics[stage] = stage_metrics; gates[stage] = stage_gates
    medium_begin, medium_signals = score_rows["medium"]
    if medium_begin != 0:
        raise AssertionError("medium control must include train and dev queries")
    train_medium_metrics = {
        name: {
            "metrics": metrics(scores[:train_count], 0, stage="medium"),
        }
        for name, scores in medium_signals.items()
    }
    t_z_axis = int(tuple(
        metadata["controlled_pose_stencil_audit"]["twist_order"]
    ).index("t_z"))
    t_z_matches = np.flatnonzero(np.all(
        metric_kwargs["direction_axis_pairs"] == np.asarray([t_z_axis, -1]), axis=1,
    ))
    if t_z_matches.size != 1:
        raise ValueError("controlled stencil does not have one coordinate t_z direction")
    t_z_direction_id = int(t_z_matches[0])
    t_z_per_query: dict[str, dict[str, object]] = {}
    for stage, (begin, signals) in score_rows.items():
        stage_rows: dict[str, object] = {}
        for name, scores in signals.items():
            selected = scores[train_count:] if stage == "medium" else scores
            stage_rows[name] = _signed_direction_per_query_audit(
                selected, arrays["image_ids"][train_count:], radial_paths,
                metric_kwargs["candidate_direction_ids"], metric_kwargs["candidate_signs"],
                direction_id=t_z_direction_id,
            )
        t_z_per_query[stage] = stage_rows
    signed_ray_summaries = {
        stage: {
            name: _flatten_42_signed_ray_metrics(value["metrics"])
            for name, value in signals.items()
        }
        for stage, signals in dev_metrics.items()
    }

    medium_gate = gates["medium"]
    structured_signal = "deployable_structured_qret_spatial_hierarchy_footprint"
    global_metrics = dev_metrics["medium"]["deployable_hierarchy_footprint"]["metrics"]
    raw_spatial_metrics = dev_metrics["medium"][
        "deployable_qret_spatial_hierarchy_footprint"
    ]["metrics"]
    structured_metrics = dev_metrics["medium"][structured_signal]["metrics"]
    diagnosis = {
        "global_identity_histogram": (
            "PASS" if medium_gate["deployable_hierarchy_footprint"]
            else "FAIL_TOKEN_AXIS_WAS_DESTROYED"
        ),
        "raw_qret_token_x_connected_support": (
            "PASS" if medium_gate["deployable_qret_spatial_hierarchy_footprint"]
            else (
                "NEAR_PASS_503_OF_504_RADIAL_PAIRS_SINGLE_FRAME00012_NEGATIVE_TZ_FIRST_STEP"
                if raw_spatial_metrics["controlled_radial_pair_accuracy"] == 503.0 / 504.0
                else "FAIL"
            )
        ),
        "default_structured_qret_token_x_connected_support": (
            "PASS" if medium_gate[structured_signal]
            else (
                "NO_GAIN_OVER_RAW_SAME_503_OF_504_AND_WORSE_FRAME00012_NEGATIVE_TZ_MARGIN"
                if (
                    structured_metrics["controlled_radial_pair_accuracy"]
                    == raw_spatial_metrics["controlled_radial_pair_accuracy"]
                ) else "FAIL"
            )
        ),
        "GT_projected_footprint_diagnostic": (
            "PASS_WHEN_COMBINED_WITH_LOCAL_SYMMETRIC"
            if medium_gate["local_x_GT_oracle_projected_product"] else "FAIL"
        ),
        "metric_depth_necessity": (
            "NOT_DEMONSTRATED_GT_PROJECTED_FOOTPRINT_ALREADY_SUFFICIENT"
            if medium_gate["local_x_GT_oracle_projected_product"]
            else "UNRESOLVED"
        ),
        "conclusion": (
            "FREEZE_MEDIUM_STRUCTURED_SPATIAL_QRET"
            if medium_gate[structured_signal]
            else "ENTER_QUERY_FOOTPRINT_TEACHER_PREDICTABILITY_GATE_WITHOUT_POSE_OPTIMIZER"
        ),
        "global_histogram_medium_anchor_top1_rate": float(
            global_metrics["gt_anchor_top1_rate"]
        ),
        "raw_spatial_medium_anchor_top1_rate": float(
            raw_spatial_metrics["gt_anchor_top1_rate"]
        ),
        "structured_spatial_medium_anchor_top1_rate": float(
            structured_metrics["gt_anchor_top1_rate"]
        ),
    }

    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "dataset_content_sha256": metadata["content_sha256"],
        "model_file_sha256": file_sha256(Path(args.model)),
        "model_content_sha256": pose_transport_model_content_sha256(model),
        "surface_mapper_file_sha256": file_sha256(mapper_path),
        "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_content_sha256": physical.content_sha256,
        "scientific_dataset_contract_audit": scientific_audit,
        "candidate_semantics": "GT_relative_oracle_local_stencil_not_natural_candidates",
        "statistics_semantics": DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
        "dense_candidate_batch_size": int(args.dense_candidate_batch_size),
        "zero_fit_observability_contract": {
            "optimizer_steps": 0,
            "fit_parameters": 0,
            "development_labels_used_for_selection": False,
            "stage_identity_fixed_before_scoring": {
                "coarse": "parent_maplet", "medium": "connected_surface_support",
                "fine": "exact_surface_child",
            },
            "deployable_footprint": (
                "mass_Dice_between_reliability_weighted_RADIO_q_ret_and_same_kernel_"
                "reliability_weighted_candidate_projection_after_stage_identity_aggregation"
            ),
            "deployable_qret_spatial_footprint": (
                "mass_Dice_in_exact_RADIO_token_by_stage_identity_space_between_existing_"
                "q_ret_and_candidate_projection;both_use_same_query_only_reliability;no_GT_"
                "query_observation_and_no_candidate_normalization"
            ),
            "deployable_structured_qret_spatial_footprint": {
                "source": "same_existing_pose_free_q_ret_after_default_structured_reweight",
                "semantics": STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS,
                "module_default_parameters_fixed_before_c85": {
                    "local_radius_tokens": 1,
                    "connected_support_weight": 2.0,
                    "parent_weight": 0.5,
                },
                "support_and_per_token_mass_are_preserved": True,
                "candidate_pose_or_GT_consumed": False,
            },
            "GT_oracle_projected_footprint": (
                "mass_Dice_on_token_by_stage_identity_against_candidate0_GT_render"
            ),
            "GT_oracle_metric_depth_footprint": (
                "same_projected_overlap_times_exp_minus_absolute_metric_log_depth_"
                "difference_over_fixed_0.25"
            ),
            "bounded_combination": (
                "score_terms_are_mapped_to_[0,1],combined_by_predeclared_product_or_"
                "minimum_without_weight,and_mapped_back_to_[-1,1]"
            ),
        },
        "train_medium_controls": train_medium_metrics,
        "dev_controls_by_stage": dev_metrics,
        "controlled_shape_gate_by_stage_and_signal": gates,
        "dev_42_signed_ray_summary_by_stage_and_signal": signed_ray_summaries,
        "dev_t_z_signed_paths_per_query": t_z_per_query,
        "observability_diagnosis": diagnosis,
        "observability_diagnosis_summary": str(diagnosis["conclusion"]),
        "deployable_medium_controlled_shape_gate_passed": bool(
            gates["medium"][structured_signal]
        ),
        "deployable_multiscale_diagnostic_gate_passed": bool(
            all(gates[stage][structured_signal] for stage in gates)
        ),
        "GT_oracle_metric_controlled_shape_gate_passed": bool(
            all(gates[stage]["local_x_GT_oracle_metric_product"] for stage in gates)
        ),
        "go_kill_boundary": {
            "controlled_shape_signal_pass": (
                "GT_anchor_Top1_equals_1 AND mean_anchor_margin_gt_0 AND each_of_42_"
                "signed_rays_is_strictly_monotonic_for_every_dev_query"
            ),
            "deployable_objective_GO_requires": [
                "structured_qret_spatial_medium_controlled_shape_gate",
                "strict_route_disjoint_query_representation",
                "natural_candidate_basin_gate",
                "optimizer_trajectory_drift_gate",
            ],
            "coarse_and_fine_are_multiscale_diagnostics_not_medium_research_blockers": True,
            "GT_oracle_pass_is_diagnostic_only": True,
            "current_deployable_decision": (
                "FREEZE_MEDIUM_SPATIAL_CONTROL_SKIP_LEARNED_FOOTPRINT_HEAD"
                if gates["medium"][structured_signal]
                else "KILL_EXISTING_QRET_SPATIAL_PROFILE_ENTER_TEACHER_PREDICTABILITY_GATE"
            ),
            "next_observation_hypothesis_if_killed": (
                "predict_a_better_query_side_token_by_hierarchy_projected_footprint;metric_"
                "depth_is_a_margin_control_not_yet_a_demonstrated_necessity"
            ),
        },
        "local_optimizer_authorized": False,
        "local_optimizer_authorization_blockers": [
            "GT_relative_controlled_candidates_not_natural_candidates",
            "candidate0_GT_visible_oracle_is_non_deployable",
            "surface_mapper_supervision_overlaps_query_route",
            "no_optimizer_trajectory_drift_evaluation",
        ],
        "per_query_audits": audits,
        "resource_audits": resources,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_process_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "natural_candidate_performance_claim_supported": False,
        "strict_route_disjoint_representation_claim_supported": False,
        "optimizer_trajectory_drift_claim_supported": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    report["report_payload_content_sha256"] = _payload_content_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
