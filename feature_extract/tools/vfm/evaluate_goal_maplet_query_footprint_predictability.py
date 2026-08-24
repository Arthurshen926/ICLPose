"""Fit and gate a small pose-free query-footprint proxy on a held route.

The evaluator opens only fit-route GT-render labels while fitting a fixed
closed-form ridge occupancy/depth proxy.  It freezes predictions for every
source query before opening the physically separate held-route teacher file.
The primary score uses projected token-by-connected-support mass only; metric
depth is reported as an optional predictability diagnostic and never enters
the candidate score.  No pose optimizer is constructed or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_query_footprint_teacher_dataset import (
    SOURCE_SCHEMA,
    SOURCE_SCHEMA_V1,
    TEACHER_SCHEMA,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_controlled_6dof_basin_gate import (
    _direction_failures,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_hierarchy_footprint_observability import (
    _deployable_spatial_hierarchy_footprint_scores,
    _flatten_42_signed_ray_metrics,
    _payload_content_sha256,
    _shape_gate,
    _signed_direction_per_query_audit,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    _load_dataset,
    _metrics,
    _validate_scientific_dataset_contract,
)
from feature_extract.vfm.localization_goal_maplet.candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.structured_token_child_posterior import (
    STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS,
    structured_token_child_probabilities,
)


SCHEMA = "goal_maplet_query_footprint_held_route_predictability_gate_v1"
RIDGE_LAMBDA = 10.0
RADIO_GROUP_COUNT = 32
MIN_FIT_QUERY_COUNT = 16
MIN_HELD_QUERY_COUNT = 16
OCCUPANCY_PROXY_NAMES = (
    "constant_fit_occupancy",
    "qret_statistics_ridge",
    "RADIO_only_ridge",
    "qret_plus_RADIO_ridge",
)


def _load_artifact(
    path: Path, schema: str | tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as source:
        if "metadata_json" not in source.files:
            raise ValueError("predictability artifact lacks metadata")
        metadata = json.loads(str(np.asarray(source["metadata_json"]).item()))
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata_json"}
    schemas = (schema,) if isinstance(schema, str) else tuple(schema)
    if (
        metadata.get("artifact_type") not in schemas
        or metadata.get("content_sha256") != arrays_sha256(arrays)
    ):
        raise ValueError("predictability artifact schema/content differs")
    return arrays, metadata


def _assert_deployable_source_contract(
    arrays: dict[str, np.ndarray], metadata: dict[str, object],
) -> None:
    forbidden = {
        "pose_w2c", "candidate_poses_w2c", "teacher_child_rows",
        "teacher_child_weights", "teacher_metric_log_depth", "translation_m",
        "rotation_deg",
    }
    required = {
        "image_ids", "route_roles", "radio_final", "source_child_rows",
        "source_child_probabilities", "query_reliability",
        "hierarchy_child_parent_ids", "hierarchy_child_support_ids",
        "retrieval_content_sha256", "retrieval_file_sha256", "radio_file_sha256",
    }
    source_schema = str(metadata.get("artifact_type", ""))
    required.discard("radio_final")
    if source_schema == SOURCE_SCHEMA:
        required.add("radio_group_means")
    elif source_schema == SOURCE_SCHEMA_V1:
        required.add("radio_final")
    else:
        raise ValueError("deployable source schema differs")
    label_markers = ("pose", "teacher", "translation", "rotation")
    if (
        forbidden & set(arrays)
        or any(any(marker in name.lower() for marker in label_markers) for name in arrays)
        or not required.issubset(arrays)
    ):
        raise ValueError("deployable source contains labels or lacks inference inputs")
    query_count = int(np.asarray(arrays["image_ids"]).size)
    if (
        (
            source_schema == SOURCE_SCHEMA
            and (
                np.asarray(arrays["radio_group_means"]).shape != (query_count, 2304, 32)
                or np.asarray(arrays["radio_group_means"]).dtype != np.dtype(np.float32)
                or "radio_final" in arrays
                or metadata.get("raw_RADIO_persisted") is not False
            )
        )
        or (
            source_schema == SOURCE_SCHEMA_V1
            and np.asarray(arrays["radio_final"]).shape != (query_count, 1280, 36, 64)
        )
        or np.asarray(arrays["source_child_rows"]).shape[:2] != (query_count, 2304)
        or np.asarray(arrays["source_child_probabilities"]).shape
        != np.asarray(arrays["source_child_rows"]).shape
        or np.asarray(arrays["query_reliability"]).shape != (query_count, 2304)
        or np.asarray(arrays["retrieval_content_sha256"]).shape != (query_count,)
        or np.asarray(arrays["retrieval_file_sha256"]).shape != (query_count,)
        or np.asarray(arrays["radio_file_sha256"]).shape != (query_count,)
        or metadata.get("contains_pose_or_GT_teacher_labels") is not False
        or not isinstance(metadata.get("retrieval_lineage"), dict)
    ):
        raise ValueError("deployable source tensor contract differs")


def _token_features(
    radio_or_group_means: np.ndarray,
    child_probabilities: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    """Return 36 fixed query-only features per RADIO token."""

    radio = np.asarray(radio_or_group_means, dtype=np.float32)
    probability = np.asarray(child_probabilities, dtype=np.float64)
    query_reliability = np.asarray(reliability, dtype=np.float64)
    if (
        probability.ndim != 3
        or query_reliability.shape != probability.shape[:2]
        or probability.shape[1] != 2304
    ):
        raise ValueError("query footprint token features differ")
    if radio.ndim == 4 and radio.shape[1:] == (1280, 36, 64):
        radio_group = radio.reshape(
            radio.shape[0], RADIO_GROUP_COUNT, 1280 // RADIO_GROUP_COUNT, 2304,
        ).mean(axis=2).transpose(0, 2, 1)
    elif radio.ndim == 3 and radio.shape[1:] == (2304, RADIO_GROUP_COUNT):
        radio_group = radio
    else:
        raise ValueError("query footprint RADIO representation differs")
    if radio_group.shape[0] != probability.shape[0]:
        raise ValueError("query footprint RADIO/query count differs")
    mass = probability.sum(axis=2)
    normalized = np.divide(
        probability, mass[:, :, None], out=np.zeros_like(probability),
        where=mass[:, :, None] > 1.0e-12,
    )
    entropy = -np.sum(
        np.where(normalized > 0.0, normalized * np.log(np.maximum(normalized, 1.0e-12)), 0.0),
        axis=2,
    ) / np.log(float(probability.shape[2]))
    top = np.max(normalized, axis=2)
    # Fixed channel grouping is architecture-defined and label-independent.  V2
    # stores these sufficient statistics so route-scale artifacts never persist
    # the 1280-D tensor or materialize it for every query at once.
    return np.concatenate([
        mass[:, :, None], query_reliability[:, :, None], top[:, :, None],
        entropy[:, :, None], radio_group.astype(np.float64),
    ], axis=2)


def _teacher_token_targets(
    weights: np.ndarray, metric_log_depth: np.ndarray, depth_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.asarray(weights, dtype=np.float64)
    depth = np.asarray(metric_log_depth, dtype=np.float64)
    valid = np.asarray(depth_valid, dtype=bool)
    if mass.shape != depth.shape or mass.shape != valid.shape or mass.ndim != 3:
        raise ValueError("teacher token target arrays differ")
    occupancy = np.sum(mass, axis=2)
    valid_weight = mass * valid
    denominator = np.sum(valid_weight, axis=2)
    mean_depth = np.divide(
        np.sum(valid_weight * depth, axis=2), denominator,
        out=np.zeros_like(denominator), where=denominator > 1.0e-8,
    )
    return occupancy, mean_depth, denominator > 1.0e-8


def _fit_fixed_ridge(
    features: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    selected = np.ones(labels.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if values.shape[0] != labels.size or np.sum(selected) <= values.shape[1]:
        raise ValueError("ridge fit does not have enough valid supervision")
    x = values[selected]; y = labels[selected]
    mean = x.mean(axis=0); scale = x.std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    design = np.concatenate([np.ones((x.shape[0], 1)), (x - mean) / scale], axis=1)
    gram = design.T @ design
    regularizer = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    coefficient = np.linalg.solve(gram + regularizer, design.T @ y)
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def _apply_ridge(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    normalized = (values - model["mean"]) / model["scale"]
    return model["coefficient"][0] + np.einsum(
        "...d,d->...", normalized, model["coefficient"][1:],
    )


def _redistribute_token_mass(probability: np.ndarray, predicted_mass: np.ndarray) -> np.ndarray:
    source = np.asarray(probability, dtype=np.float64)
    mass = np.asarray(predicted_mass, dtype=np.float64)
    if source.ndim != 3 or mass.shape != source.shape[:2]:
        raise ValueError("predicted footprint redistribution arrays differ")
    denominator = source.sum(axis=2, keepdims=True)
    normalized = np.divide(
        source, denominator, out=np.zeros_like(source), where=denominator > 1.0e-12,
    )
    result = normalized * np.clip(mass, 0.0, 1.0)[:, :, None]
    return result.astype(np.float32)


def _profile_dice_rows(
    source_rows: np.ndarray,
    probabilities: np.ndarray,
    reliability: np.ndarray,
    teacher_rows: np.ndarray,
    teacher_weights: np.ndarray,
    hierarchy: PoseTransportHierarchy,
) -> np.ndarray:
    values = []
    for query in range(source_rows.shape[0]):
        score, _ = _deployable_spatial_hierarchy_footprint_scores(
            source_rows[query], probabilities[query], reliability[query],
            teacher_rows[query][None], teacher_weights[query][None],
            np.asarray([True]), hierarchy, stage="medium",
        )
        values.append(float(0.5 * (score[0] + 1.0)))
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "query_count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "minimum": float(np.min(array)),
        "maximum": float(np.max(array)), "per_query": [float(value) for value in array],
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1); y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fit_teacher", required=True)
    parser.add_argument("--held_teacher", required=True)
    parser.add_argument("--c85_dataset", default="")
    parser.add_argument("--include_metric_depth_diagnostic", action="store_true")
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()
    output = Path(args.output_report)
    if output.exists():
        raise FileExistsError("refusing to overwrite query footprint predictability report")
    started = time.monotonic()

    source, source_metadata = _load_artifact(
        Path(args.source), (SOURCE_SCHEMA, SOURCE_SCHEMA_V1),
    )
    _assert_deployable_source_contract(source, source_metadata)
    image_ids = np.asarray(source["image_ids"]); roles = np.asarray(source["route_roles"])
    fit_indices = np.flatnonzero(roles == "fit"); held_indices = np.flatnonzero(roles == "held")
    if fit_indices.size != int(source_metadata["fit_count"]) or held_indices.size != int(source_metadata["held_count"]):
        raise ValueError("source role counts differ")
    hierarchy = PoseTransportHierarchy(
        child_parent_ids=np.asarray(source["hierarchy_child_parent_ids"], dtype=np.int64),
        child_support_ids=np.asarray(source["hierarchy_child_support_ids"], dtype=np.int64),
        adjacency_offsets=np.zeros(
            np.asarray(source["hierarchy_child_parent_ids"]).size + 1, dtype=np.int64,
        ),
        adjacency_child_rows=np.empty((0,), dtype=np.int64),
        content_sha256=str(source_metadata["hierarchy_content_sha256"]),
    )

    # Phase 1: only fit labels are opened.
    fit_teacher, fit_metadata = _load_artifact(Path(args.fit_teacher), TEACHER_SCHEMA)
    if (
        fit_metadata.get("role") != "fit"
        or fit_metadata.get("source_content_sha256") != source_metadata["content_sha256"]
        or not np.array_equal(fit_teacher["image_ids"], image_ids[fit_indices])
    ):
        raise ValueError("fit teacher/source split differs")
    structured = np.stack([
        structured_token_child_probabilities(
            source["source_child_rows"][query], source["source_child_probabilities"][query],
            hierarchy.child_parent_ids, hierarchy.child_support_ids,
        )
        for query in range(image_ids.size)
    ])
    radio_input = (
        source["radio_group_means"] if "radio_group_means" in source
        else source["radio_final"]
    )
    features = _token_features(radio_input, structured, source["query_reliability"])
    fit_occupancy, fit_depth, fit_depth_valid = _teacher_token_targets(
        fit_teacher["teacher_child_weights"], fit_teacher["teacher_metric_log_depth"],
        fit_teacher["teacher_metric_depth_valid"],
    )
    occupancy_models = {
        "qret_statistics_ridge": _fit_fixed_ridge(
            features[fit_indices, :, :4], fit_occupancy,
        ),
        "RADIO_only_ridge": _fit_fixed_ridge(
            features[fit_indices, :, 4:], fit_occupancy,
        ),
        "qret_plus_RADIO_ridge": _fit_fixed_ridge(
            features[fit_indices], fit_occupancy,
        ),
    }
    depth_model = (
        _fit_fixed_ridge(features[fit_indices], fit_depth, fit_depth_valid)
        if bool(args.include_metric_depth_diagnostic) else None
    )
    predicted_occupancy = {
        "constant_fit_occupancy": np.full(
            features.shape[:2], float(np.mean(fit_occupancy)), dtype=np.float64,
        ),
        "qret_statistics_ridge": np.clip(
            _apply_ridge(occupancy_models["qret_statistics_ridge"], features[..., :4]),
            0.0, 1.0,
        ),
        "RADIO_only_ridge": np.clip(
            _apply_ridge(occupancy_models["RADIO_only_ridge"], features[..., 4:]),
            0.0, 1.0,
        ),
        "qret_plus_RADIO_ridge": np.clip(
            _apply_ridge(occupancy_models["qret_plus_RADIO_ridge"], features),
            0.0, 1.0,
        ),
    }
    predicted_depth = (
        _apply_ridge(depth_model, features) if depth_model is not None else None
    )
    calibrated = {
        name: _redistribute_token_mass(structured, predicted_occupancy[name])
        for name in OCCUPANCY_PROXY_NAMES
    }
    prediction_arrays = {
        "structured_probabilities": structured.astype(np.float32),
        **{
            f"predicted_occupancy_{name}": predicted_occupancy[name].astype(np.float32)
            for name in OCCUPANCY_PROXY_NAMES
        },
        **{
            f"calibrated_probabilities_{name}": calibrated[name]
            for name in OCCUPANCY_PROXY_NAMES
        },
    }
    if predicted_depth is not None:
        prediction_arrays["predicted_metric_log_depth"] = predicted_depth.astype(np.float32)
    prediction_content_sha256 = arrays_sha256(prediction_arrays)
    model_arrays = {
        "constant_fit_occupancy": np.asarray(float(np.mean(fit_occupancy))),
        **{
            f"occupancy_{name}_{key}": value
            for name, model in occupancy_models.items() for key, value in model.items()
        },
    }
    if depth_model is not None:
        model_arrays.update({f"depth_{key}": value for key, value in depth_model.items()})
    model_content_sha256 = arrays_sha256(model_arrays)
    predictions_frozen_before_held_labels = True

    # Phase 2: held labels are opened only after every source prediction froze.
    held_teacher, held_metadata = _load_artifact(Path(args.held_teacher), TEACHER_SCHEMA)
    if (
        not predictions_frozen_before_held_labels or held_metadata.get("role") != "held"
        or held_metadata.get("source_content_sha256") != source_metadata["content_sha256"]
        or not np.array_equal(held_teacher["image_ids"], image_ids[held_indices])
    ):
        raise ValueError("held teacher was opened outside the frozen phase contract")
    held_occupancy, held_depth, held_depth_valid = _teacher_token_targets(
        held_teacher["teacher_child_weights"], held_teacher["teacher_metric_log_depth"],
        held_teacher["teacher_metric_depth_valid"],
    )

    def split_dice(indices: np.ndarray, teacher: dict[str, np.ndarray], probability: np.ndarray):
        return _profile_dice_rows(
            source["source_child_rows"][indices], probability[indices],
            source["query_reliability"][indices], teacher["teacher_child_rows"],
            teacher["teacher_child_weights"], hierarchy,
        )

    fit_dice = {
        "raw_qret": split_dice(fit_indices, fit_teacher, source["source_child_probabilities"]),
        "structured_qret": split_dice(fit_indices, fit_teacher, structured),
        **{
            name: split_dice(fit_indices, fit_teacher, calibrated[name])
            for name in OCCUPANCY_PROXY_NAMES
        },
    }
    held_dice = {
        "raw_qret": split_dice(held_indices, held_teacher, source["source_child_probabilities"]),
        "structured_qret": split_dice(held_indices, held_teacher, structured),
        **{
            name: split_dice(held_indices, held_teacher, calibrated[name])
            for name in OCCUPANCY_PROXY_NAMES
        },
    }

    held_occ_prediction = {
        name: predicted_occupancy[name][held_indices] for name in OCCUPANCY_PROXY_NAMES
    }
    depth_error = (
        np.abs(predicted_depth[held_indices][held_depth_valid] - held_depth[held_depth_valid])
        if predicted_depth is not None else None
    )
    proxy_mean = float(np.mean(held_dice["qret_plus_RADIO_ridge"]))
    qret_only_mean = float(np.mean(held_dice["qret_statistics_ridge"]))
    raw_mean = float(np.mean(held_dice["raw_qret"]))
    structured_mean = float(np.mean(held_dice["structured_qret"]))
    footprint_calibration_improvement_gate = bool(
        proxy_mean > raw_mean and proxy_mean > structured_mean
        and float(np.min(held_dice["qret_plus_RADIO_ridge"]))
        >= float(np.min(held_dice["raw_qret"]))
    )
    radio_incremental_predictability_gate = bool(
        footprint_calibration_improvement_gate and proxy_mean > qret_only_mean
    )
    query_count_gate = bool(
        fit_indices.size >= MIN_FIT_QUERY_COUNT and held_indices.size >= MIN_HELD_QUERY_COUNT
    )
    if not args.c85_dataset:
        report = {
            "artifact_type": SCHEMA,
            "source_file_sha256": file_sha256(Path(args.source)),
            "source_content_sha256": source_metadata["content_sha256"],
            "source_schema": source_metadata["artifact_type"],
            "source_resource_contract": {
                "feature_representation": source_metadata.get(
                    "source_feature_representation", "legacy_full_RADIO_v1"
                ),
                "raw_RADIO_persisted": bool(
                    source_metadata.get("raw_RADIO_persisted", True)
                ),
                "actual_source_uncompressed_bytes": source_metadata.get(
                    "actual_source_uncompressed_bytes"
                ),
                "max_source_uncompressed_mib": source_metadata.get(
                    "max_source_uncompressed_mib"
                ),
            },
            "fit_teacher_file_sha256": file_sha256(Path(args.fit_teacher)),
            "fit_teacher_content_sha256": fit_metadata["content_sha256"],
            "held_teacher_file_sha256": file_sha256(Path(args.held_teacher)),
            "held_teacher_content_sha256": held_metadata["content_sha256"],
        "phase_separation": {
                "fit_route": source_metadata["fit_route"],
                "held_route": source_metadata["held_route"],
                "all_source_predictions_frozen_before_held_teacher_open": True,
                "held_labels_used_for_fit_or_selection": False,
                "inference_consumes_pose_or_GT": False,
                "strict_route_disjoint_predictability_claim_supported": bool(
                    source_metadata.get("strict_route_disjoint_predictability_claim_supported", False)
                ),
                "mapping_route_overlap_control": bool(
                    source_metadata.get("mapping_route_overlap_control", False)
                ),
                "retrieval_lineage": source_metadata.get("retrieval_lineage"),
                "all_retrieval_artifacts_promotion_eligible": bool(
                    source_metadata.get("all_retrieval_artifacts_promotion_eligible", False)
                ),
                "any_retrieval_artifact_control_only": bool(
                    source_metadata.get("any_retrieval_artifact_control_only", True)
                ),
            },
            "proxy_contract": {
                "primary_target": (
                    "projected_token_occupancy_mass_with_default_structured_qret_"
                    "connected_support_identity_distribution_frozen"
                ),
                "identity_distribution": "default_structured_pose_free_q_ret_on_frozen_support",
                "occupancy_features": (
                    "qret_mass,reliability,top_share,entropy,32_fixed_RADIO_channel_group_means"
                ),
                "occupancy_fit": "closed_form_ridge_no_gradient_optimizer",
                "ridge_lambda_fixed_before_held_labels": RIDGE_LAMBDA,
                "occupancy_parameter_count_by_proxy": {
                    "constant_fit_occupancy": 1,
                    **{
                        name: int(model["coefficient"].size)
                        for name, model in occupancy_models.items()
                    },
                },
                "metric_depth_parameter_count": (
                    int(depth_model["coefficient"].size) if depth_model is not None else 0
                ),
                "metric_depth_diagnostic_enabled": bool(depth_model is not None),
                "metric_depth_optional_diagnostic_not_candidate_score": True,
                "structured_semantics": STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS,
                "model_content_sha256": model_content_sha256,
                "prediction_content_sha256": prediction_content_sha256,
            },
            "fit_route_footprint_dice": {
                name: _summary(value) for name, value in fit_dice.items()
            },
            "held_route_footprint_dice": {
                name: _summary(value) for name, value in held_dice.items()
            },
            "held_route_occupancy_by_proxy": {
                name: {
                    "mean_absolute_error": float(np.mean(np.abs(value - held_occupancy))),
                    "pearson_correlation": _correlation(value, held_occupancy),
                    "prediction_mean": float(np.mean(value)),
                    "teacher_mean": float(np.mean(held_occupancy)),
                }
                for name, value in held_occ_prediction.items()
            },
            "held_route_metric_depth_optional_diagnostic": (
                {
                    "enabled": True,
                    "valid_token_count": int(np.sum(held_depth_valid)),
                    "mean_absolute_log_depth_error": float(np.mean(depth_error)),
                    "median_absolute_log_depth_error": float(np.median(depth_error)),
                    "pearson_correlation": _correlation(
                        predicted_depth[held_indices][held_depth_valid], held_depth[held_depth_valid],
                    ),
                    "used_in_candidate_score": False,
                }
                if predicted_depth is not None else {
                    "enabled": False,
                    "reason": "disabled_by_default_footprint_is_primary_target",
                    "used_in_candidate_score": False,
                }
            ),
            "held_route_relative_improvement_gates": {
                "contract": (
                    "full_mean_Dice_gt_raw_and_structured_AND_full_min_Dice_ge_raw_min;"
                    "RADIO_increment_requires_full_mean_Dice_gt_qret_statistics_ridge_mean"
                ),
                "footprint_calibration_improvement_passed": (
                    footprint_calibration_improvement_gate
                ),
                "RADIO_incremental_predictability_passed": (
                    radio_incremental_predictability_gate
                ),
                "query_count_gate_passed": query_count_gate,
                "minimum_fit_query_count": MIN_FIT_QUERY_COUNT,
                "minimum_held_query_count": MIN_HELD_QUERY_COUNT,
                "full_minus_qret_only_mean_Dice": float(proxy_mean - qret_only_mean),
            },
            "decision": (
                "PASS_RADIO_INCREMENTAL_QUERY_FOOTPRINT_PREDICTABILITY_CONTROL"
                if radio_incremental_predictability_gate and query_count_gate
                else (
                    "FUNCTIONAL_SMOKE_ONLY_INSUFFICIENT_QUERY_COUNT"
                    if not query_count_gate
                    else (
                        "PASS_ONLY_NON_RADIO_FOOTPRINT_CALIBRATION_"
                        "KILL_RADIO_INCREMENTAL_CLAIM"
                        if footprint_calibration_improvement_gate
                        else "KILL_SMALL_QUERY_FOOTPRINT_PROXY"
                    )
                )
            ),
            "pose_backend_gate_deferred_until_strict_mapper_c85_rebuild": True,
            "local_optimizer_authorized": False,
            "optimizer_steps": 0,
            "uses_alike": False, "uses_point_correspondences": False,
            "uses_pnp": False, "uses_absolute_pose_regression": False,
            "production_eligible": False,
            "elapsed_seconds": float(time.monotonic() - started),
            "peak_process_rss_mib": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            ),
        }
        report["report_payload_content_sha256"] = _payload_content_sha256(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    # Phase 3: c85 labels/candidates are opened only after held predictions froze.
    c85, c85_metadata = _load_dataset(Path(args.c85_dataset))
    c85_audit = _validate_scientific_dataset_contract(c85, c85_metadata)
    c85_ids = np.asarray(c85["image_ids"])
    lookup = {str(value): index for index, value in enumerate(image_ids.tolist())}
    try:
        c85_source_indices = np.asarray([lookup[str(value)] for value in c85_ids.tolist()])
    except KeyError as error:
        raise ValueError("c85 query is absent from frozen source predictions") from error
    if (
        not np.all(roles[c85_source_indices] == "held")
        or not np.array_equal(c85["source_child_rows"], source["source_child_rows"][c85_source_indices])
        or not np.allclose(
            c85["source_child_probabilities"],
            source["source_child_probabilities"][c85_source_indices], atol=0.0, rtol=0.0,
        )
    ):
        raise ValueError("c85 source is not an exact held-route replay")
    c85_probability = {
        "raw_qret": source["source_child_probabilities"][c85_source_indices],
        "structured_qret": structured[c85_source_indices],
        **{
            name: calibrated[name][c85_source_indices] for name in OCCUPANCY_PROXY_NAMES
        },
    }
    c85_scores: dict[str, np.ndarray] = {}
    for name, probability in c85_probability.items():
        rows = []
        for query in range(c85_ids.size):
            score, _ = _deployable_spatial_hierarchy_footprint_scores(
                c85["source_child_rows"][query], probability[query],
                c85["query_reliability"][query], c85["target_child_rows"][query],
                c85["target_child_weights"][query], c85["candidate_valid"][query],
                hierarchy, stage="medium",
            )
            rows.append(score)
        c85_scores[name] = np.stack(rows)
    radial_paths = np.asarray(c85["controlled_radial_paths"], dtype=np.int64)
    direction_ids = np.asarray(c85["controlled_candidate_direction_ids"], dtype=np.int64)
    signs = np.asarray(c85["controlled_candidate_signs"], dtype=np.int64)
    axis_pairs = np.asarray(c85["controlled_direction_axis_pairs"], dtype=np.int64)
    twist_order = tuple(c85_metadata["controlled_pose_stencil_audit"]["twist_order"])
    metric_values, gates, rays = {}, {}, {}
    for name, scores in c85_scores.items():
        value = _metrics(
            scores, c85["translation_m"], c85["rotation_deg"], c85["candidate_valid"],
            c85_ids, stage="medium", candidate_semantics=str(c85_metadata["candidate_semantics"]),
            radial_paths=radial_paths, candidate_direction_ids=direction_ids,
            candidate_signs=signs, direction_axis_pairs=axis_pairs, twist_order=twist_order,
        )
        metric_values[name] = value; gates[name] = _shape_gate(value)
        rays[name] = _flatten_42_signed_ray_metrics(value)
    t_z_axis = twist_order.index("t_z")
    t_z_direction = int(np.flatnonzero(np.all(axis_pairs == [t_z_axis, -1], axis=1))[0])
    t_z_rows = {
        name: _signed_direction_per_query_audit(
            scores, c85_ids, radial_paths, direction_ids, signs,
            direction_id=t_z_direction,
        ) for name, scores in c85_scores.items()
    }

    primary_gate = bool(gates["qret_plus_RADIO_ridge"])
    report = {
        "artifact_type": SCHEMA,
        "source_file_sha256": file_sha256(Path(args.source)),
        "source_content_sha256": source_metadata["content_sha256"],
        "source_schema": source_metadata["artifact_type"],
        "source_resource_contract": {
            "feature_representation": source_metadata.get(
                "source_feature_representation", "legacy_full_RADIO_v1"
            ),
            "raw_RADIO_persisted": bool(source_metadata.get("raw_RADIO_persisted", True)),
            "actual_source_uncompressed_bytes": source_metadata.get(
                "actual_source_uncompressed_bytes"
            ),
            "max_source_uncompressed_mib": source_metadata.get(
                "max_source_uncompressed_mib"
            ),
        },
        "fit_teacher_file_sha256": file_sha256(Path(args.fit_teacher)),
        "fit_teacher_content_sha256": fit_metadata["content_sha256"],
        "held_teacher_file_sha256": file_sha256(Path(args.held_teacher)),
        "held_teacher_content_sha256": held_metadata["content_sha256"],
        "c85_dataset_file_sha256": file_sha256(Path(args.c85_dataset)),
        "c85_dataset_content_sha256": c85_metadata["content_sha256"],
        "c85_scientific_contract_audit": c85_audit,
            "phase_separation": {
            "fit_route": source_metadata["fit_route"],
            "held_route": source_metadata["held_route"],
            "fit_teacher_opened_before_fit": True,
            "all_source_predictions_frozen_before_held_teacher_open": True,
            "c85_queries_are_held_route": True,
            "held_labels_used_for_fit_or_selection": False,
            "c85_labels_or_candidates_used_for_fit_or_selection": False,
            "inference_consumes_pose_or_GT": False,
            "strict_route_disjoint_predictability_claim_supported": bool(
                source_metadata.get("strict_route_disjoint_predictability_claim_supported", False)
            ),
            "retrieval_lineage": source_metadata.get("retrieval_lineage"),
            "all_retrieval_artifacts_promotion_eligible": bool(
                source_metadata.get("all_retrieval_artifacts_promotion_eligible", False)
            ),
            "any_retrieval_artifact_control_only": bool(
                source_metadata.get("any_retrieval_artifact_control_only", True)
            ),
        },
        "proxy_contract": {
            "primary_target": (
                "projected_token_occupancy_mass_with_default_structured_qret_"
                "connected_support_identity_distribution_frozen"
            ),
            "identity_distribution": "default_structured_pose_free_q_ret_on_frozen_support",
            "occupancy_features": (
                "qret_mass,reliability,top_share,entropy,32_fixed_RADIO_channel_group_means"
            ),
            "occupancy_fit": "closed_form_ridge_no_gradient_optimizer",
            "ridge_lambda_fixed_before_held_labels": RIDGE_LAMBDA,
            "occupancy_parameter_count_by_proxy": {
                "constant_fit_occupancy": 1,
                **{
                    name: int(model["coefficient"].size)
                    for name, model in occupancy_models.items()
                },
            },
            "metric_depth_parameter_count": (
                int(depth_model["coefficient"].size) if depth_model is not None else 0
            ),
            "metric_depth_diagnostic_enabled": bool(depth_model is not None),
            "metric_depth_optional_diagnostic_not_candidate_score": True,
            "structured_semantics": STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS,
            "model_content_sha256": model_content_sha256,
            "prediction_content_sha256": prediction_content_sha256,
        },
        "fit_route_footprint_dice": {name: _summary(value) for name, value in fit_dice.items()},
        "held_route_footprint_dice": {name: _summary(value) for name, value in held_dice.items()},
        "held_route_occupancy_by_proxy": {
            name: {
                "mean_absolute_error": float(np.mean(np.abs(value - held_occupancy))),
                "pearson_correlation": _correlation(value, held_occupancy),
                "prediction_mean": float(np.mean(value)),
                "teacher_mean": float(np.mean(held_occupancy)),
            }
            for name, value in held_occ_prediction.items()
        },
        "held_route_metric_depth_optional_diagnostic": (
            {
                "enabled": True,
                "valid_token_count": int(np.sum(held_depth_valid)),
                "mean_absolute_log_depth_error": float(np.mean(depth_error)),
                "median_absolute_log_depth_error": float(np.median(depth_error)),
                "pearson_correlation": _correlation(
                    predicted_depth[held_indices][held_depth_valid], held_depth[held_depth_valid],
                ),
                "used_in_candidate_score": False,
            }
            if predicted_depth is not None else {
                "enabled": False,
                "reason": "disabled_by_default_footprint_is_primary_target",
                "used_in_candidate_score": False,
            }
        ),
        "c85_medium_metrics": metric_values,
        "c85_medium_shape_gate_by_proxy": gates,
        "c85_medium_42_signed_rays_by_proxy": rays,
        "c85_medium_t_z_per_query_by_proxy": t_z_rows,
        "pose_landscape_interpretation": {
            "raw_qret_strict_anchor_shape_gate_passed": bool(gates["raw_qret"]),
            "raw_qret_gt_anchor_top1_rate": float(
                metric_values["raw_qret"]["gt_anchor_top1_rate"]
            ),
            "raw_qret_selected_best_candidate_strict_0_5m_5deg": float(
                metric_values["raw_qret"]["selected_strict_0_5m_5deg"]
            ),
            "qret_plus_RADIO_strict_anchor_shape_gate_passed": bool(primary_gate),
            "qret_plus_RADIO_selected_best_candidate_strict_0_5m_5deg": float(
                metric_values["qret_plus_RADIO_ridge"]["selected_strict_0_5m_5deg"]
            ),
            "discrete_best_candidate_success_does_not_authorize_continuous_optimizer": True,
            "conclusion": (
                "KILL_SCALAR_OCCUPANCY_TEACHER_FOR_POSE_LANDSCAPE;"
                "RAW_QRET_REMAINS_HIGH_TOLERANCE_BASIN_SCORE_ONLY"
            ),
        },
        "held_route_relative_improvement_gates": {
            "footprint_calibration_improvement_passed": (
                footprint_calibration_improvement_gate
            ),
            "RADIO_incremental_predictability_passed": (
                radio_incremental_predictability_gate
            ),
            "query_count_gate_passed": query_count_gate,
            "minimum_fit_query_count": MIN_FIT_QUERY_COUNT,
            "minimum_held_query_count": MIN_HELD_QUERY_COUNT,
            "full_minus_qret_only_mean_Dice": float(proxy_mean - qret_only_mean),
        },
        "held_route_predictability_gate_passed": bool(
            primary_gate and radio_incremental_predictability_gate and query_count_gate
        ),
        "decision": (
            "PASS_SMALL_QUERY_FOOTPRINT_PROXY_ON_HELD_C85_CONTROL"
            if primary_gate and radio_incremental_predictability_gate and query_count_gate
            else "KILL_SMALL_QUERY_FOOTPRINT_PROXY"
        ),
        "local_optimizer_authorized": False,
        "local_optimizer_authorization_blockers": [
            "controlled_GT_relative_candidates_not_natural_candidates",
            "no_natural_candidate_basin_gate_for_this_proxy",
            "no_optimizer_trajectory_drift_gate",
        ],
        "optimizer_steps": 0,
        "uses_alike": False, "uses_point_correspondences": False,
        "uses_pnp": False, "uses_absolute_pose_regression": False,
        "production_eligible": False,
        "elapsed_seconds": float(time.monotonic() - started),
        "peak_process_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    report["report_payload_content_sha256"] = _payload_content_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
