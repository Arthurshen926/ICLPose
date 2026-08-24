"""Evaluate a target-precision-aware symmetric coverage pose score.

The control keeps the existing fixed-kernel capacity coupling unchanged.  It
only replaces its source-recall readout with a bounded Dice/F score over query
mass, propagated target mass, and matched mass.  It is post-hoc, has no fitted
parameters, and is evaluated only on a GT-relative diagnostic stencil.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_controlled_6dof_basin_gate import (
    _direction_failures,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS,
    _dense_fixed_identity_query_component_statistics,
    _fixed_identity_scores_from_component_statistics,
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
    _STAGE,
)
from feature_extract.vfm.localization_goal_maplet.differentiable_pose_transport import (
    FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS,
)
from feature_extract.vfm.localization_goal_maplet.dense_fixed_identity_transport import (
    DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
    DensePoseTransportHierarchyGPU,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.trainable_pose_transport import (
    load_minimal_pose_transport_readout,
    pose_transport_model_content_sha256,
)


SCHEMA = "goal_maplet_symmetric_coverage_control_v1"


def _propagated_reliability(
    reliability: np.ndarray, *, stage: str, height: int = 36, width: int = 64,
) -> np.ndarray:
    """Propagate fixed query reliability through the same bounded kernel."""

    value = np.asarray(reliability, dtype=np.float64).reshape(-1)
    if str(stage) not in _STAGE or value.shape != (height * width,):
        raise ValueError("symmetric coverage reliability/grid differs")
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("symmetric coverage reliability must be finite/nonnegative")
    source = value.reshape(height, width)
    target = np.zeros_like(source)
    # This is the same authority used to construct M; there is intentionally
    # no second radius table that could silently drift from the transport.
    radius = int(_STAGE[str(stage)]["radius"])
    kernel = 1.0 / float((2 * radius + 1) ** 2)
    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            sy0, sy1 = max(0, -shift_y), min(height, height - shift_y)
            sx0, sx1 = max(0, -shift_x), min(width, width - shift_x)
            ty0, ty1 = sy0 + shift_y, sy1 + shift_y
            tx0, tx1 = sx0 + shift_x, sx1 + shift_x
            target[ty0:ty1, tx0:tx1] += source[sy0:sy1, sx0:sx1] * kernel
    return target.reshape(-1)


def _symmetric_coverage_scores(
    component_statistics: np.ndarray,
    edge_weights: np.ndarray,
    source_probability: np.ndarray,
    reliability: np.ndarray,
    target_weight: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    stage: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return recall and symmetric scores with capacity-bound diagnostics."""

    statistics = np.asarray(component_statistics, dtype=np.float64)
    weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
    source = np.asarray(source_probability, dtype=np.float64)
    query_weight = np.asarray(reliability, dtype=np.float64).reshape(-1)
    target = np.asarray(target_weight, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    if (
        statistics.ndim != 2 or statistics.shape[1] != 6
        or weights.shape != (6,) or target.ndim != 3
        or target.shape[0] != statistics.shape[0] or valid.shape != (target.shape[0],)
        or source.ndim != 2 or source.shape[0] != query_weight.size
        or target.shape[1] != query_weight.size
        or np.any(~np.isfinite(statistics)) or np.any(~np.isfinite(weights))
        or np.any(~np.isfinite(source)) or np.any(~np.isfinite(target))
        or np.any(source < 0.0) or np.any(target < 0.0) or np.any(weights <= 0.0)
    ):
        raise ValueError("symmetric coverage score inputs differ")
    weight_sum = float(weights.sum())
    recall_component = statistics @ weights / weight_sum
    recall_score = -1.0 + recall_component
    reliability_sum = float(query_weight.sum())
    matched_mass = 0.5 * recall_component * reliability_sum
    query_mass = float(np.sum(query_weight * source.sum(axis=1)))
    propagated = _propagated_reliability(query_weight, stage=str(stage))
    target_mass = np.sum(
        target.sum(axis=2) * propagated[None], axis=1, dtype=np.float64
    )
    target_mass = np.where(valid, target_mass, 0.0)
    matched_mass = np.where(valid, matched_mass, 0.0)
    denominator = query_mass + target_mass
    f_score = np.divide(
        2.0 * matched_mass,
        denominator,
        out=np.zeros_like(matched_mass),
        where=denominator > 1.0e-12,
    )
    symmetric_score = -1.0 + 2.0 * f_score
    tolerance = 2.0e-4
    maximum_query_excess = float(np.max(matched_mass - query_mass))
    maximum_target_excess = float(np.max(matched_mass - target_mass))
    if (
        np.any(~np.isfinite(symmetric_score))
        or np.min(f_score) < -tolerance or np.max(f_score) > 1.0 + tolerance
        or maximum_query_excess > tolerance or maximum_target_excess > tolerance
    ):
        raise AssertionError("symmetric coverage violates a capacity bound")
    return recall_score.astype(np.float32), symmetric_score.astype(np.float32), {
        "query_mass": query_mass,
        "minimum_target_mass": float(np.min(target_mass[valid])),
        "maximum_target_mass": float(np.max(target_mass[valid])),
        "minimum_matched_mass": float(np.min(matched_mass[valid])),
        "maximum_matched_mass": float(np.max(matched_mass[valid])),
        "maximum_matched_minus_query_mass": maximum_query_excess,
        "maximum_matched_minus_target_mass": maximum_target_excess,
        "minimum_f_score": float(np.min(f_score)),
        "maximum_f_score": float(np.max(f_score)),
        "all_missing_target_score": -1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--surface_mapper", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite symmetric coverage control")
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    arrays, metadata = _load_dataset(Path(args.dataset))
    scientific_audit = _validate_scientific_dataset_contract(arrays, metadata)
    if scientific_audit["full_6dof_observability_stencil"] is not True:
        raise ValueError("symmetric coverage control requires the complete 6-DoF stencil")
    model, model_metadata = load_minimal_pose_transport_readout(
        Path(args.model), device=str(device)
    )
    if (
        model_metadata.get("dataset_content_sha256") != metadata["content_sha256"]
        or model_metadata.get("transport_semantics")
        != FIXED_KERNEL_CAPACITY_TRANSPORT_SEMANTICS
        or model_metadata.get("readout_training_semantics")
        != IDENTITY_FEATURE_ONLY_READOUT_SEMANTICS
        or model_metadata.get("effective_trainable_scalar_count") != 2
    ):
        raise ValueError("symmetric coverage model and controlled dataset differ")
    mapper_path = Path(args.surface_mapper)
    if file_sha256(mapper_path) != model_metadata.get("surface_mapper_file_sha256"):
        raise ValueError("symmetric coverage surface mapper differs from trained model")
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
            arrays["hierarchy_adjacency_child_rows"], dtype=np.int64
        ),
        content_sha256=str(metadata["hierarchy_content_sha256"]),
    )
    dense_hierarchy = DensePoseTransportHierarchyGPU(hierarchy, device=device)
    model.eval()
    edge_weights = model.edge_weights().detach().cpu().numpy().astype(np.float64)
    train_count = len(model_metadata["train_image_ids"])
    query_count = int(arrays["image_ids"].size)
    score_rows: dict[str, dict[str, tuple[int, np.ndarray]]] = {}
    mass_audits, resource_audits = [], []
    started = time.monotonic()
    for stage in ("medium", "coarse", "fine"):
        begin = 0 if stage == "medium" else train_count
        stage_recall, stage_symmetric = [], []
        for query in range(begin, query_count):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            statistics, resource_audit = _dense_fixed_identity_query_component_statistics(
                model, arrays, query, dense_hierarchy, stage=stage, device=device,
                candidate_batch_size=8,
            )
            _release_streaming_cuda_cache(device, resource_audit)
            recall, symmetric, mass_audit = _symmetric_coverage_scores(
                statistics, edge_weights,
                arrays["source_child_probabilities"][query],
                arrays["query_reliability"][query],
                arrays["target_child_weights"][query],
                arrays["candidate_valid"][query],
                stage=stage,
            )
            # Independent reconstruction must be identical to the trainer readout.
            reconstructed = _fixed_identity_scores_from_component_statistics(
                model, torch.as_tensor(statistics, device=device, dtype=torch.float32)
            ).detach().cpu().numpy()
            max_difference = float(np.max(np.abs(recall - reconstructed)))
            if max_difference > 2.0e-7:
                raise AssertionError("symmetric control recall reconstruction differs")
            resource_audit["recall_reconstruction_max_abs_difference"] = max_difference
            resource_audit["stage"] = stage
            mass_audit["query_index"] = query
            mass_audit["stage"] = stage
            stage_recall.append(recall)
            stage_symmetric.append(symmetric)
            mass_audits.append(mass_audit)
            resource_audits.append(resource_audit)
            print(json.dumps({
                "query": str(arrays["image_ids"][query]),
                "stage": stage,
                "resource": resource_audit,
                "mass": mass_audit,
            }), flush=True)
        score_rows[stage] = {
            "recall": (begin, np.stack(stage_recall)),
            "symmetric": (begin, np.stack(stage_symmetric)),
        }
    radial_paths = np.asarray(arrays["controlled_radial_paths"], dtype=np.int64)
    metric_kwargs = {
        "candidate_semantics": str(metadata["candidate_semantics"]),
        "radial_paths": radial_paths,
        "candidate_direction_ids": np.asarray(
            arrays["controlled_candidate_direction_ids"], dtype=np.int64
        ),
        "candidate_signs": np.asarray(
            arrays["controlled_candidate_signs"], dtype=np.int64
        ),
        "direction_axis_pairs": np.asarray(
            arrays["controlled_direction_axis_pairs"], dtype=np.int64
        ),
        "twist_order": tuple(metadata["controlled_pose_stencil_audit"]["twist_order"]),
    }

    def metrics(scores: np.ndarray, begin: int, *, stage: str):
        end = begin + int(scores.shape[0])
        return _metrics(
            scores, arrays["translation_m"][begin:end],
            arrays["rotation_deg"][begin:end], arrays["candidate_valid"][begin:end],
            arrays["image_ids"][begin:end], stage=stage, **metric_kwargs,
        )

    stage_metrics: dict[str, dict[str, object]] = {}
    for stage, values in score_rows.items():
        begin, recall_score = values["recall"]
        _, symmetric_score = values["symmetric"]
        recall_metrics = metrics(recall_score, begin, stage=stage)
        symmetric_metrics = metrics(symmetric_score, begin, stage=stage)
        stage_metrics[stage] = {
            "recall": recall_metrics,
            "symmetric": symmetric_metrics,
            "signed_direction_failures": _direction_failures(symmetric_metrics),
        }
    recall_train = metrics(
        score_rows["medium"]["recall"][1][:train_count], 0, stage="medium"
    )
    symmetric_train = metrics(
        score_rows["medium"]["symmetric"][1][:train_count], 0, stage="medium"
    )
    recall_dev = metrics(
        score_rows["medium"]["recall"][1][train_count:], train_count,
        stage="medium",
    )
    symmetric_dev = metrics(
        score_rows["medium"]["symmetric"][1][train_count:], train_count,
        stage="medium",
    )
    signed_failures = _direction_failures(symmetric_dev)
    stage_metrics["medium"] = {
        "recall": recall_dev,
        "symmetric": symmetric_dev,
        "signed_direction_failures": signed_failures,
    }

    def stage_gate(value: dict[str, object]) -> bool:
        failures = _direction_failures(value)
        return bool(
            value["gt_anchor_top1_rate"] == 1.0
            and value["mean_gt_anchor_margin_over_best_nonanchor"] > 0.0
            and failures["all_42_signed_directions_strictly_monotonic"]
        )

    gate_by_stage = {
        stage: stage_gate(values["symmetric"])
        for stage, values in stage_metrics.items()
    }
    gate = bool(all(gate_by_stage.values()))
    core = (
        "gt_anchor_top1_rate", "mean_gt_anchor_margin_over_best_nonanchor",
        "mean_score_error_spearman", "pairwise_order_accuracy",
        "controlled_radial_pair_accuracy", "controlled_radial_complete_path_rate",
    )
    report = {
        "artifact_type": SCHEMA,
        "dataset_file_sha256": file_sha256(Path(args.dataset)),
        "dataset_content_sha256": metadata["content_sha256"],
        "model_file_sha256": file_sha256(Path(args.model)),
        "model_content_sha256": pose_transport_model_content_sha256(model),
        "surface_mapper_file_sha256": file_sha256(mapper_path),
        "scientific_dataset_contract_audit": scientific_audit,
        "candidate_semantics": "GT_relative_oracle_local_stencil_not_natural_candidates",
        "statistics_semantics": DENSE_FIXED_IDENTITY_TRANSPORT_SEMANTICS,
        "dense_candidate_batch_size": 8,
        "score_semantics": "bounded_symmetric_query_target_capacity_f_score_v1",
        "spatial_kernel_contract_source": (
            "candidate_conditioned_pose_attribution._STAGE_shared_with_transport"
        ),
        "spatial_kernel_radius_by_stage": {
            stage: int(_STAGE[stage]["radius"])
            for stage in ("coarse", "medium", "fine")
        },
        "proof_contract": {
            "query_mass": "Q=sum_query reliability_times_source_mass;candidate_independent",
            "target_mass": "R=sum_target propagated_query_reliability_times_target_mass",
            "matched_mass": "M=existing_fixed_kernel_compatibility_weighted_capacity_mass",
            "capacity_bounds": "zero_le_M_le_min_Q_R",
            "score": "F=2M/(Q+R);score=-1+2F",
            "bounded": "capacity_bounds_imply_zero_le_F_le_one_and_minus1_le_score_le1",
            "matched_evidence_deletion": (
                "holding_Q_R_fixed,a_decrease_in_M_cannot_increase_score"
            ),
            "unmatched_true_geometry_addition": (
                "holding_M_Q_fixed,an_increase_in_R_cannot_increase_score"
            ),
            "true_geometry_disappearance": (
                "a_real_decrease_in_unmatched_R_can_increase_precision;this_is_physical_"
                "F_score_semantics_not_a_missing_data_invariance_claim"
            ),
            "invalid_or_unresolved_field": (
                "planned_target_weight_is_retained_in_R_while_invalid_evidence_"
                "contributes_zero_M_so_missing_field_evidence_cannot_improve_score"
            ),
            "all_missing_target": "R=0_and_M=0_implies_score=-1",
        },
        "train_recall_control": recall_train,
        "dev_recall_control": recall_dev,
        "train_symmetric_coverage": symmetric_train,
        "dev_symmetric_coverage": symmetric_dev,
        "dev_stage_controls": stage_metrics,
        "symmetric_minus_recall_dev": {
            key: float(symmetric_dev[key]) - float(recall_dev[key]) for key in core
        },
        "dev_symmetric_signed_direction_failures": signed_failures,
        "local_optimizer_gate_by_stage": gate_by_stage,
        "local_optimizer_gate_passed": gate,
        "controlled_local_shape_gate_passed": gate,
        # A GT-relative old-mapper control can reject an objective, but can
        # never authorize production/local-optimizer use on its own.
        "local_optimizer_authorized": False,
        "local_optimizer_authorization_requires": [
            "controlled_local_shape_gate_passed",
            "strict_route_disjoint_representation",
            "natural_candidate_basin_gate",
            "optimizer_trajectory_drift_gate",
        ],
        "local_optimizer_authorization_blockers": [
            "GT_relative_oracle_candidates_not_natural_candidates",
            "surface_mapper_supervision_overlaps_query_route",
            "no_optimizer_trajectory_drift_evaluation",
        ],
        "decision": (
            "PASS_SYMMETRIC_COVERAGE_CONTROL"
            if gate else "KILL_SYMMETRIC_COVERAGE_CONTROL"
        ),
        "mass_capacity_audits": mass_audits,
        "resource_audits": resource_audits,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_process_rss_mib": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
        "natural_candidate_performance_claim_supported": False,
        "strict_route_disjoint_representation_claim_supported": bool(
            model_metadata.get("strict_query_representation_route_disjoint", False)
        ),
        "optimizer_trajectory_drift_claim_supported": False,
        "uses_alike": False,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "uses_absolute_pose_regression": False,
        "production_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
