"""Fit and cross-validate the low-capacity G20 conditional pose energy.

The input is a frozen factor-2 surface-verification report containing the
orientation-equivariant phase components.  Candidate generation and rendering
are never rerun here.  Every reported development prediction is produced by
an outer leave-one-acquisition-out fold.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from feature_extract.vfm.localization_goal_maplet.conditional_pose_energy import (
    COMPONENT_NAMES,
    DEFAULT_SCALE_FLOORS,
    candidate_measurements,
    normalize_candidate_measurements,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


@dataclass(frozen=True)
class QueryEvidence:
    image_id: str
    trajectory: str
    normalized: np.ndarray
    phase_peak: float
    translation_m: np.ndarray
    rotation_deg: np.ndarray
    target_index: int


def _fractional_mass_audit(
    report: dict[str, object], *, mode_name: str,
) -> dict[str, object]:
    feature_closure = []
    visibility_closure = []
    mixed_fraction = []
    values = []
    for row in report.get("rows", []):
        diagnostics = dict(row.get("ranking_diagnostics", {}).get(mode_name, {}))
        for evidence in diagnostics.get("surface_phase_components_preorder") or ():
            required = (
                "feature_fraction_mean", "visibility_fraction_mean",
                "missing_fraction_mean", "background_fraction_mean",
                "dominant_surface_fraction_mean", "mixed_surface_fraction",
            )
            if any(key not in evidence for key in required):
                raise ValueError("G20 evidence lacks fractional observation mass")
            feature = float(evidence["feature_fraction_mean"])
            visibility = float(evidence["visibility_fraction_mean"])
            missing = float(evidence["missing_fraction_mean"])
            background = float(evidence["background_fraction_mean"])
            mixed = float(evidence["mixed_surface_fraction"])
            dominant = float(evidence["dominant_surface_fraction_mean"])
            values.extend((feature, visibility, missing, background, dominant, mixed))
            feature_closure.append(abs(feature + missing + background - 1.0))
            visibility_closure.append(abs(feature + missing - visibility))
            mixed_fraction.append(mixed)
    if not feature_closure or any(not np.isfinite(value) for value in values):
        raise ValueError("G20 fractional observation audit has no finite candidates")
    if min(values) < -1.0e-6 or max(values) > 1.0 + 1.0e-6:
        raise ValueError("fractional observation mass lies outside [0,1]")
    maximum_feature_residual = float(max(feature_closure))
    maximum_visibility_residual = float(max(visibility_closure))
    if maximum_feature_residual > 1.0e-5 or maximum_visibility_residual > 1.0e-5:
        raise ValueError("fractional observation mass does not close")
    return {
        "candidate_count": len(feature_closure),
        "maximum_feature_missing_background_residual": maximum_feature_residual,
        "maximum_feature_missing_visibility_residual": maximum_visibility_residual,
        "mixed_surface_fraction_median": float(np.median(mixed_fraction)),
        "mixed_surface_fraction_p90": float(np.percentile(mixed_fraction, 90.0)),
    }


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _map_query_overlap_audit(
    report: dict[str, object],
    *,
    canonical_field_summary: Path,
    mapping_contributors: Path,
) -> dict[str, object]:
    """Recover offline map-build identities and reject silent self-map claims.

    Mapping image identities deliberately do not live in the deployment map.
    Evaluation must therefore join the offline field-build summary with its
    contributor cache explicitly rather than infer independence from a query
    calibration split.
    """

    summary = json.loads(Path(canonical_field_summary).read_text())
    summary_field_sha256 = str(summary.get("canonical_field_sha256", ""))
    if not summary_field_sha256 or summary_field_sha256 != str(
        report.get("canonical_field_sha256", "")
    ):
        raise ValueError("canonical-field summary and evidence report differ")
    mapping_trajectories = set(str(value) for value in summary.get("mapping_trajectory_ids", ()))
    expected_count = int(summary.get("mapping_image_count", 0))
    if not mapping_trajectories or expected_count <= 0:
        raise ValueError("canonical-field summary lacks mapping acquisition lineage")
    mapping_ids: set[str] = set()
    for path in sorted(Path(mapping_contributors).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if str(metadata.get("trajectory_id", "")) in mapping_trajectories:
            mapping_ids.add(str(metadata["image_id"]))
    if len(mapping_ids) != expected_count:
        raise ValueError(
            "mapping contributor identities do not reproduce canonical-field image count"
        )
    query_ids = {str(row["image_id"]) for row in report.get("rows", ())}
    overlap = sorted(query_ids.intersection(mapping_ids))
    mapping_id_sha256 = hashlib.sha256(
        "\n".join(sorted(mapping_ids)).encode("utf8")
    ).hexdigest()
    return {
        "canonical_field_summary": str(canonical_field_summary),
        "canonical_field_summary_sha256": file_sha256(canonical_field_summary),
        "mapping_contributors": str(mapping_contributors),
        "mapping_image_id_sha256": mapping_id_sha256,
        "mapping_image_count": len(mapping_ids),
        "query_image_count": len(query_ids),
        "exact_image_overlap_count": len(overlap),
        "exact_image_overlap_fraction": len(overlap) / max(float(len(query_ids)), 1.0),
        "map_query_image_disjoint": not overlap,
        "overlap_trajectory_ids": sorted({_trajectory(value) for value in overlap}),
        "deployment_map_stores_image_identity": False,
    }


def _load_queries(
    report: dict[str, object],
    *,
    mode_name: str,
    success_translation_m: float,
    success_rotation_deg: float,
) -> list[QueryEvidence]:
    rows = []
    for row in report.get("rows", []):
        image_id = str(row["image_id"])
        details = list(row.get("mode_details", {}).get(mode_name, ()))
        diagnostics = dict(row.get("ranking_diagnostics", {}).get(mode_name, {}))
        phase = list(diagnostics.get("surface_phase_components_preorder") or ())
        order = [int(value) for value in diagnostics.get("surface_alignment_original_indices", ())]
        take = int(diagnostics.get("surface_alignment_evaluated_count", len(phase)))
        if not details or take <= 0 or len(phase) != take or len(order) < take:
            raise ValueError(f"incomplete conditional-energy evidence: {image_id}")
        source_details: list[dict[str, object] | None] = [None] * take
        for ranked_index, source_index in enumerate(order):
            if source_index < take and ranked_index < len(details):
                source_details[source_index] = details[ranked_index]
        if any(value is None for value in source_details):
            raise ValueError(f"cannot invert candidate phase ordering: {image_id}")
        typed_details = [value for value in source_details if value is not None]
        identity = np.asarray(
            [value["pre_surface_score"] for value in typed_details], dtype=np.float64,
        )
        measurements = candidate_measurements(identity, phase)
        normalized = normalize_candidate_measurements(measurements)
        translation = np.asarray(
            [value["translation_m"] for value in typed_details], dtype=np.float64,
        )
        rotation = np.asarray(
            [value["rotation_deg"] for value in typed_details], dtype=np.float64,
        )
        eligible = np.flatnonzero(
            (translation <= float(success_translation_m))
            & (rotation <= float(success_rotation_deg))
        )
        if eligible.size:
            quality = translation[eligible] + 0.02 * rotation[eligible]
            target = int(eligible[int(np.argmin(quality))])
        else:
            target = int(len(typed_details))
        rows.append(QueryEvidence(
            image_id=image_id,
            trajectory=_trajectory(image_id),
            normalized=normalized,
            phase_peak=float(np.max(measurements[:, 1])),
            translation_m=translation,
            rotation_deg=rotation,
            target_index=target,
        ))
    if not rows:
        raise ValueError("conditional-energy report contains no queries")
    return rows


def _loss_and_gradient(
    parameter: np.ndarray,
    rows: list[QueryEvidence],
    active: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    active = np.asarray(active, dtype=bool)
    weight = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    weight[active] = parameter
    loss = float(regularization) * float(np.sum(weight * weight))
    gradient_weight = 2.0 * float(regularization) * weight
    trained_count = 0
    for row in rows:
        if row.target_index >= row.normalized.shape[0]:
            continue
        candidate = row.normalized @ weight
        probability = np.exp(candidate - float(np.max(candidate)))
        probability /= float(np.sum(probability))
        loss -= float(np.log(max(probability[row.target_index], 1.0e-12)))
        residual = probability
        residual[row.target_index] -= 1.0
        gradient_weight += row.normalized.T @ residual
        trained_count += 1
    denominator = max(float(trained_count), 1.0)
    gradient = gradient_weight[active] / denominator
    return loss / denominator, gradient


def _fit(
    rows: list[QueryEvidence],
    *,
    active: tuple[bool, bool, bool],
    regularization: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    active_array = np.asarray(active, dtype=bool)
    initial = np.ones(int(np.sum(active_array)), dtype=np.float64)
    bounds = [(0.0, 20.0)] * int(np.sum(active_array))
    result = minimize(
        lambda value: _loss_and_gradient(value, rows, active_array, regularization),
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not bool(result.success):
        raise RuntimeError(f"conditional-energy calibration failed: {result.message}")
    weight = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    weight[active_array] = np.asarray(result.x, dtype=np.float64)
    return weight, 0.0, {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }


def _calibrate_null_phase_support(rows: list[QueryEvidence]) -> tuple[float, float]:
    positive = np.asarray([
        row.phase_peak for row in rows if row.target_index < row.normalized.shape[0]
    ], dtype=np.float64)
    if not positive.size:
        raise ValueError("null-support calibration requires a valid candidate set")
    q25, q75 = np.percentile(positive, (25.0, 75.0))
    scale = max(float((q75 - q25) / 1.349), 0.05)
    # A one-feature-floor margin below every observed valid candidate set.
    # This gives a conservative open-set boundary even when a training fold
    # contains no supervised null example.
    threshold = float(np.min(positive) - DEFAULT_SCALE_FLOORS[1])
    return threshold, scale


def _predict(
    row: QueryEvidence,
    weight: np.ndarray,
    null_phase_threshold: float,
    null_phase_scale: float,
) -> dict[str, object]:
    candidate_energy = row.normalized @ weight
    candidate_energy -= float(np.max(candidate_energy))
    phase_presence = (
        float(row.phase_peak) - float(null_phase_threshold)
    ) / float(null_phase_scale)
    candidate_energy += phase_presence
    logits = np.r_[candidate_energy, 0.0]
    probability = np.exp(logits - float(np.max(logits)))
    probability /= float(np.sum(probability))
    selected = int(np.argmax(candidate_energy))
    null_probability = float(probability[-1])
    candidate_probability = probability[:-1]
    abstain = bool(null_probability >= float(np.max(candidate_probability)))
    return {
        "image_id": row.image_id,
        "trajectory": row.trajectory,
        "selected_index": selected,
        "translation_m": float(row.translation_m[selected]),
        "rotation_deg": float(row.rotation_deg[selected]),
        "candidate_probability": candidate_probability.tolist(),
        "null_probability": null_probability,
        "phase_peak": float(row.phase_peak),
        "null_phase_threshold": float(null_phase_threshold),
        "abstain": abstain,
        "target_is_null": bool(row.target_index == len(candidate_energy)),
        "target_index": int(row.target_index),
    }


def _metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    translation = np.asarray([item["translation_m"] for item in predictions], dtype=np.float64)
    rotation = np.asarray([item["rotation_deg"] for item in predictions], dtype=np.float64)
    strict = (translation <= 0.5) & (rotation <= 5.0)
    one_m = (translation <= 1.0) & (rotation <= 10.0)
    catastrophic = (translation > 5.0) | (rotation > 30.0)
    accepted = np.asarray([not bool(item["abstain"]) for item in predictions], dtype=bool)
    target_null = np.asarray([bool(item["target_is_null"]) for item in predictions], dtype=bool)
    predicted_null = ~accepted
    return {
        "query_count": int(len(predictions)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_0.5m_5deg": float(np.mean(strict)),
        "within_1m_10deg": float(np.mean(one_m)),
        "catastrophic_rate": float(np.mean(catastrophic)),
        "abstain_rate": float(np.mean(predicted_null)),
        "null_classification_accuracy": float(np.mean(predicted_null == target_null)),
        "accepted_strict_rate": (
            float(np.mean(strict[accepted])) if np.any(accepted) else None
        ),
        "accepted_catastrophic_rate": (
            float(np.mean(catastrophic[accepted])) if np.any(accepted) else None
        ),
        "strict_success_yield": float(np.mean(strict & accepted)),
    }


def _paired_comparison(
    reference: list[dict[str, object]],
    candidate: list[dict[str, object]],
) -> dict[str, object]:
    left = {str(item["image_id"]): item for item in reference}
    right = {str(item["image_id"]): item for item in candidate}
    if left.keys() != right.keys():
        raise ValueError("paired conditional-energy predictions differ")
    quality_delta = []
    strict_gained = 0
    strict_lost = 0
    for image_id in sorted(left):
        old, new = left[image_id], right[image_id]
        old_quality = float(old["translation_m"]) + 0.02 * float(old["rotation_deg"])
        new_quality = float(new["translation_m"]) + 0.02 * float(new["rotation_deg"])
        quality_delta.append(new_quality - old_quality)
        old_strict = float(old["translation_m"]) <= 0.5 and float(old["rotation_deg"]) <= 5.0
        new_strict = float(new["translation_m"]) <= 0.5 and float(new["rotation_deg"]) <= 5.0
        strict_gained += int(new_strict and not old_strict)
        strict_lost += int(old_strict and not new_strict)
    delta = np.asarray(quality_delta, dtype=np.float64)
    return {
        "query_count": int(delta.size),
        "candidate_better_fraction": float(np.mean(delta < -1.0e-12)),
        "tie_fraction": float(np.mean(np.abs(delta) <= 1.0e-12)),
        "candidate_worse_fraction": float(np.mean(delta > 1.0e-12)),
        "pose_quality_delta_median": float(np.median(delta)),
        "strict_successes_gained": int(strict_gained),
        "strict_successes_lost": int(strict_lost),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_report", required=True)
    parser.add_argument("--canonical_field_summary", required=True)
    parser.add_argument("--mapping_contributors", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_policy", required=True)
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--regularization", type=float, default=1.0e-3)
    parser.add_argument("--success_translation_m", type=float, default=1.0)
    parser.add_argument("--success_rotation_deg", type=float, default=10.0)
    parser.add_argument(
        "--closed_trajectories", nargs="*", default=("seq11", "seq3", "seq5", "seq13"),
    )
    parser.add_argument(
        "--map_trajectories", nargs="*", default=("seq1", "seq2", "seq4"),
        help="Mapping acquisitions excluded from query-side calibration and validation.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    policy_output = Path(args.output_policy)
    if (output.exists() or policy_output.exists()) and not args.force:
        raise FileExistsError("refusing to overwrite conditional-energy artifacts")
    evidence_path = Path(args.evidence_report)
    report = json.loads(evidence_path.read_text())
    overlap_audit = _map_query_overlap_audit(
        report,
        canonical_field_summary=Path(args.canonical_field_summary),
        mapping_contributors=Path(args.mapping_contributors),
    )
    contract = dict(report.get("surface_verification_contract") or {})
    if int(contract.get("render_supersample_factor", 0)) != 2:
        raise ValueError("G20 conditional energy requires factor-2 rendering")
    if not contract.get("phase_readout_policy_sha256"):
        raise ValueError("G20 conditional energy requires serialized phase evidence")
    fractional_mass_audit = _fractional_mass_audit(
        report, mode_name=str(args.mode_name),
    )
    all_queries = _load_queries(
        report,
        mode_name=str(args.mode_name),
        success_translation_m=float(args.success_translation_m),
        success_rotation_deg=float(args.success_rotation_deg),
    )
    map_trajectories = set(str(value) for value in args.map_trajectories)
    queries = [row for row in all_queries if row.trajectory not in map_trajectories]
    if not queries:
        raise ValueError("no cross-acquisition queries remain after excluding map trajectories")
    trajectories = sorted({row.trajectory for row in queries})
    closed = sorted(set(trajectories).intersection(str(value) for value in args.closed_trajectories))
    if closed:
        raise ValueError(f"closed trajectories cannot calibrate G20: {closed}")
    variants = {
        "identity_only": (True, False, False),
        "phase_only": (False, True, False),
        "identity_phase": (True, True, False),
        "identity_phase_observation": (True, True, True),
    }
    variant_reports = {}
    for name, active in variants.items():
        predictions = []
        folds = []
        for held_out in trajectories:
            train = [row for row in queries if row.trajectory != held_out]
            validation = [row for row in queries if row.trajectory == held_out]
            weight, bias, optimization = _fit(
                train, active=active, regularization=float(args.regularization),
            )
            null_threshold, null_scale = _calibrate_null_phase_support(train)
            fold_predictions = [
                _predict(row, weight, null_threshold, null_scale)
                for row in validation
            ]
            predictions.extend(fold_predictions)
            folds.append({
                "held_out_trajectory": held_out,
                "train_query_count": len(train),
                "validation_query_count": len(validation),
                "weights": weight.tolist(),
                "candidate_bias": bias,
                "null_phase_threshold": null_threshold,
                "null_phase_scale": null_scale,
                "optimization": optimization,
                "metrics": _metrics(fold_predictions),
            })
        predictions.sort(key=lambda item: str(item["image_id"]))
        variant_reports[name] = {
            "active_components": [
                component for component, enabled in zip(COMPONENT_NAMES, active) if enabled
            ],
            "outer_loto_metrics": _metrics(predictions),
            "folds": folds,
            "predictions": predictions,
        }
    selected = "identity_phase_observation"
    selected_active = variants[selected]
    final_weight, final_bias, final_optimization = _fit(
        queries, active=selected_active, regularization=float(args.regularization),
    )
    final_null_threshold, final_null_scale = _calibrate_null_phase_support(queries)
    phase_policy_sha256 = str(contract["phase_readout_policy_sha256"])
    policy = {
        "artifact_type": "goal_maplet_conditional_pose_energy_v1",
        "role": "candidate_softmax_with_typed_null",
        "component_names": list(COMPONENT_NAMES),
        "weights": final_weight.tolist(),
        "candidate_bias": final_bias,
        "scale_floors": DEFAULT_SCALE_FLOORS.tolist(),
        "null_phase_threshold": final_null_threshold,
        "null_phase_scale": final_null_scale,
        "null_phase_slope": 1.0,
        "normalization": "per_query_median_iqr_with_fixed_floors_v1",
        "null_hypothesis": "candidate_set_phase_support_null_logit_zero_v2",
        "null_calibration": (
            "minimum_valid_training_phase_peak_minus_one_phase_scale_floor_v1"
        ),
        "observation_definition": (
            "jacobian_observability-minus-missing-minus-half-mixed-boundary_v1"
        ),
        "monotonic_nonnegative_weights": True,
        "deep_ranker": False,
        "candidate_pool_frozen": True,
        "success_target": {
            "translation_m": float(args.success_translation_m),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "calibration_protocol": (
            "outer_leave_one_query_trajectory_out_on_fixed_overlapping_field_v1"
        ),
        "calibration_trajectories": trajectories,
        "map_trajectories_excluded": sorted(map_trajectories),
        "query_calibration_trajectories_excluded": sorted(map_trajectories),
        "map_query_image_disjoint": overlap_audit["map_query_image_disjoint"],
        "map_query_exact_image_overlap_count": overlap_audit[
            "exact_image_overlap_count"
        ],
        "deployment_allowed": bool(overlap_audit["map_query_image_disjoint"]),
        "validation_protocol_valid_for_promotion": bool(
            overlap_audit["map_query_image_disjoint"]
        ),
        "closed_trajectories_excluded": list(args.closed_trajectories),
        "evidence_report_sha256": file_sha256(evidence_path),
        "phase_readout_policy_sha256": phase_policy_sha256,
        "candidate_pool_sha256": contract.get("candidate_pool_sha256"),
        "maximum_modes": int(contract.get("maximum_modes", 0)),
        "physical_map_sha256": report.get("physical_map_sha256"),
        "canonical_field_sha256": report.get("canonical_field_sha256"),
        "physical_instance_readout_sha256": report.get("physical_instance_readout_sha256"),
        "render_supersample_factor": 2,
        "stores_mapping_rgb": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "fit_optimization": final_optimization,
    }
    result = {
        "stage": "goal_maplet_g20_conditional_energy_query_trajectory_loto",
        "evidence_report": str(evidence_path),
        "evidence_report_sha256": file_sha256(evidence_path),
        "input_query_count": len(all_queries),
        "query_count": len(queries),
        "trajectory_ids": trajectories,
        "map_trajectories_excluded": sorted(map_trajectories),
        "query_calibration_trajectories_excluded": sorted(map_trajectories),
        "closed_trajectories_reopened": False,
        "strictly_new_acquisition_available": False,
        "map_query_overlap_audit": overlap_audit,
        "validation_limitation": (
            "all evaluated images contributed to the fixed canonical field; outer query-"
            "trajectory LOTO isolates energy calibration only and is a self-map development "
            "diagnostic, not map-disjoint localization evidence"
        ),
        "fractional_observation_audit": fractional_mass_audit,
        "selected_variant": selected,
        "variants": variant_reports,
        "paired_comparisons_to_identity_only": {
            name: _paired_comparison(
                variant_reports["identity_only"]["predictions"],
                value["predictions"],
            )
            for name, value in variant_reports.items()
            if name != "identity_only"
        },
        "deployed_policy": policy,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    policy_output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage": result["stage"],
        "query_count": len(queries),
        "trajectory_ids": trajectories,
        "selected_variant": selected,
        "metrics": variant_reports[selected]["outer_loto_metrics"],
        "weights": final_weight.tolist(),
        "candidate_bias": final_bias,
        "null_phase_threshold": final_null_threshold,
        "null_phase_scale": final_null_scale,
        "map_query_image_disjoint": overlap_audit["map_query_image_disjoint"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
