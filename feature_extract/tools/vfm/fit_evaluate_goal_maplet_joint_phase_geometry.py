"""Fit/evaluate a trajectory-LOTO Stage-C phase + dense-geometry likelihood.

Candidate generation, canonical fields, rendered measurements and geometry
heads are frozen before this tool runs.  Calibration uses only query-local
normalized measurements and leaves one complete acquisition out at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch

from feature_extract.tools.vfm.fit_evaluate_goal_maplet_conditional_energy import (
    _map_query_overlap_audit,
)
from feature_extract.vfm.localization_goal_maplet.joint_phase_geometry_likelihood import (
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
    translation_m: np.ndarray
    rotation_deg: np.ndarray
    target_index: int
    baseline_index: int


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _validate_report_contract(report: dict[str, object]) -> dict[str, object]:
    contract = dict(report.get("surface_verification_contract") or {})
    if int(contract.get("render_supersample_factor", 0)) != 2:
        raise ValueError("joint phase-geometry likelihood requires factor-2 rendering")
    if not contract.get("phase_readout_policy_sha256"):
        raise ValueError("joint phase-geometry likelihood requires frozen phase evidence")
    if not contract.get("dense_geometry_head_sha256"):
        raise ValueError("joint phase-geometry likelihood requires dense geometry evidence")
    if bool(contract.get("dense_geometry_changes_ranking")):
        raise ValueError("calibration report must keep dense geometry audit-only")
    if int(contract.get("stored_map_feature_type_count", 0)) != 1:
        raise ValueError("joint likelihood requires one canonical stored map feature")
    if int(contract.get("stored_downstream_embedding_count", -1)) != 0:
        raise ValueError("joint likelihood cannot consume stored downstream embeddings")
    if bool(contract.get("stores_mapping_rgb")):
        raise ValueError("joint likelihood cannot consume stored mapping RGB")
    return contract


def _load_queries(
    report: dict[str, object],
    *,
    mode_name: str,
    success_translation_m: float,
    success_rotation_deg: float,
) -> list[QueryEvidence]:
    rows = []
    for row in report.get("rows", ()): 
        image_id = str(row["image_id"])
        details = list(row.get("mode_details", {}).get(mode_name, ()))
        diagnostics = dict(row.get("ranking_diagnostics", {}).get(mode_name, {}))
        take = int(diagnostics.get("surface_alignment_evaluated_count", 0))
        order = [int(value) for value in diagnostics.get("surface_alignment_original_indices", ())]
        phase = list(diagnostics.get("surface_phase_components_preorder") or ())
        geometry = list(diagnostics.get("dense_geometry_components_preorder") or ())
        phase_score = np.asarray(
            diagnostics.get("surface_phase_policy_scores_preorder")
            or diagnostics.get("surface_alignment_scores_preorder")
            or (),
            dtype=np.float64,
        )
        if (
            take <= 0 or len(details) < take or len(order) < take
            or len(phase) != take or len(geometry) != take or phase_score.shape != (take,)
        ):
            raise ValueError(f"incomplete joint evidence: {image_id}")
        source_details: list[dict[str, object] | None] = [None] * take
        for ranked_index, source_index in enumerate(order[:take]):
            if 0 <= source_index < take:
                source_details[source_index] = details[ranked_index]
        if any(value is None for value in source_details):
            raise ValueError(f"cannot invert frozen candidate ordering: {image_id}")
        typed = [value for value in source_details if value is not None]
        identity = np.asarray([value["pre_surface_score"] for value in typed], dtype=np.float64)
        measurements = candidate_measurements(identity, phase_score, phase, geometry)
        translation = np.asarray([value["translation_m"] for value in typed], dtype=np.float64)
        rotation = np.asarray([value["rotation_deg"] for value in typed], dtype=np.float64)
        eligible = np.flatnonzero(
            (translation <= float(success_translation_m))
            & (rotation <= float(success_rotation_deg))
        )
        if eligible.size:
            quality = translation[eligible] + 0.02 * rotation[eligible]
            target_index = int(eligible[int(np.argmin(quality))])
        else:
            target_index = int(take)
        baseline_index = int(order[0])
        if not 0 <= baseline_index < take:
            raise ValueError(f"phase baseline lies outside evaluated candidates: {image_id}")
        rows.append(QueryEvidence(
            image_id=image_id,
            trajectory=_trajectory(image_id),
            normalized=normalize_candidate_measurements(measurements),
            translation_m=translation,
            rotation_deg=rotation,
            target_index=target_index,
            baseline_index=baseline_index,
        ))
    if not rows:
        raise ValueError("joint phase-geometry report contains no queries")
    return rows


def _loss_and_gradient(
    parameter: np.ndarray,
    rows: list[QueryEvidence],
    active: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    weight = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    weight[mask] = parameter[:-1]
    null_logit = float(parameter[-1])
    loss = float(regularization) * float(np.sum(weight * weight) + 0.1 * null_logit**2)
    gradient = 2.0 * float(regularization) * weight
    null_gradient = 0.2 * float(regularization) * null_logit
    for row in rows:
        candidate_logits = row.normalized @ weight
        logits = np.concatenate(
            (candidate_logits, np.asarray([null_logit], dtype=np.float64))
        )
        probability = np.exp(logits - float(np.max(logits)))
        probability /= max(float(np.sum(probability)), 1.0e-12)
        loss -= float(np.log(max(probability[row.target_index], 1.0e-12)))
        residual = probability
        residual[row.target_index] -= 1.0
        gradient += row.normalized.T @ residual[:-1]
        null_gradient += float(residual[-1])
    if not rows:
        raise ValueError("joint likelihood has no training queries")
    combined = np.concatenate((gradient[mask], np.asarray([null_gradient])))
    return loss / float(len(rows)), combined / float(len(rows))


def _fit(
    rows: list[QueryEvidence],
    *,
    active: tuple[bool, bool, bool, bool],
    regularization: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    mask = np.asarray(active, dtype=bool)
    result = minimize(
        lambda value: _loss_and_gradient(value, rows, mask, regularization),
        np.concatenate((
            np.ones(int(np.sum(mask)), dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 20.0)] * int(np.sum(mask)) + [(-20.0, 20.0)],
        options={"maxiter": 500, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not bool(result.success):
        raise RuntimeError(f"joint likelihood calibration failed: {result.message}")
    weight = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    weight[mask] = np.asarray(result.x[:-1], dtype=np.float64)
    return weight, float(result.x[-1]), {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }


def _prediction(
    row: QueryEvidence,
    weight: np.ndarray,
    null_logit: float | None = None,
    *,
    baseline: bool = False,
) -> dict[str, object]:
    candidate_logits = row.normalized @ np.asarray(weight, dtype=np.float64)
    selected = int(row.baseline_index) if baseline else int(np.argmax(candidate_logits))
    abstained = bool(
        not baseline
        and null_logit is not None
        and float(null_logit) >= float(candidate_logits[selected])
    )
    return {
        "image_id": row.image_id,
        "trajectory": row.trajectory,
        "selected_index": selected,
        "translation_m": float(row.translation_m[selected]),
        "rotation_deg": float(row.rotation_deg[selected]),
        "target_is_null": bool(row.target_index >= row.normalized.shape[0]),
        "target_index": int(row.target_index),
        "abstained": abstained,
    }


def _metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    translation = np.asarray([item["translation_m"] for item in predictions], dtype=np.float64)
    rotation = np.asarray([item["rotation_deg"] for item in predictions], dtype=np.float64)
    abstained = np.asarray([item.get("abstained", False) for item in predictions], dtype=bool)
    strict = (~abstained) & (translation <= 0.5) & (rotation <= 5.0)
    loose = (~abstained) & (translation <= 1.0) & (rotation <= 10.0)
    catastrophic = (~abstained) & ((translation > 5.0) | (rotation > 30.0))
    return {
        "query_count": int(len(predictions)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_0.5m_5deg": float(np.mean(strict)),
        "within_1m_10deg": float(np.mean(loose)),
        "catastrophic_rate": float(np.mean(catastrophic)),
        "abstention_rate": float(np.mean(abstained)),
    }


def _paired(reference: list[dict[str, object]], candidate: list[dict[str, object]]) -> dict[str, object]:
    left = {str(item["image_id"]): item for item in reference}
    right = {str(item["image_id"]): item for item in candidate}
    if left.keys() != right.keys():
        raise ValueError("paired joint-likelihood predictions differ")
    delta = []
    gained = lost = 0
    for image_id in sorted(left):
        old, new = left[image_id], right[image_id]
        delta.append(
            float(new["translation_m"]) + 0.02 * float(new["rotation_deg"])
            - float(old["translation_m"]) - 0.02 * float(old["rotation_deg"])
        )
        old_ok = (
            not bool(old.get("abstained", False))
            and float(old["translation_m"]) <= 0.5
            and float(old["rotation_deg"]) <= 5.0
        )
        new_ok = (
            not bool(new.get("abstained", False))
            and float(new["translation_m"]) <= 0.5
            and float(new["rotation_deg"]) <= 5.0
        )
        gained += int(new_ok and not old_ok)
        lost += int(old_ok and not new_ok)
    value = np.asarray(delta, dtype=np.float64)
    return {
        "candidate_better_fraction": float(np.mean(value < -1.0e-12)),
        "candidate_worse_fraction": float(np.mean(value > 1.0e-12)),
        "pose_quality_delta_median": float(np.median(value)),
        "strict_successes_gained": int(gained),
        "strict_successes_lost": int(lost),
    }


def _geometry_head_overlap_audit(
    checkpoint_path: Path,
    report: dict[str, object],
) -> dict[str, object]:
    expected = str(_validate_report_contract(report)["dense_geometry_head_sha256"])
    actual = file_sha256(checkpoint_path)
    if actual != expected:
        raise ValueError("dense geometry checkpoint and evidence report differ")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_path = Path(str(payload.get("args", {}).get("train_geometry_manifest", "")))
    if not manifest_path.is_file():
        raise ValueError("dense geometry checkpoint lacks its training manifest")
    manifest = json.loads(manifest_path.read_text())
    train_ids = {str(item["image_id"]) for item in manifest.get("records", ())}
    query_ids = {str(item["image_id"]) for item in report.get("rows", ())}
    train_trajectories = {_trajectory(value) for value in train_ids}
    query_trajectories = {_trajectory(value) for value in query_ids}
    exact_overlap = sorted(train_ids.intersection(query_ids))
    trajectory_overlap = sorted(train_trajectories.intersection(query_trajectories))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual,
        "train_manifest": str(manifest_path),
        "train_manifest_sha256": file_sha256(manifest_path),
        "training_image_count": len(train_ids),
        "query_image_count": len(query_ids),
        "exact_image_overlap_count": len(exact_overlap),
        "trajectory_overlap": trajectory_overlap,
        "query_head_image_disjoint": not exact_overlap,
        "query_head_trajectory_disjoint": not trajectory_overlap,
    }


def _promotion(baseline: dict[str, object], joint: dict[str, object]) -> dict[str, object]:
    checks = {
        "strict_non_decreasing": float(joint["strict_0.5m_5deg"]) >= float(baseline["strict_0.5m_5deg"]),
        "loose_non_decreasing": float(joint["within_1m_10deg"]) >= float(baseline["within_1m_10deg"]),
        "catastrophic_non_increasing": float(joint["catastrophic_rate"]) <= float(baseline["catastrophic_rate"]),
        "translation_p90_not_regressed_5pct": float(joint["translation_p90_m"]) <= 1.05 * float(baseline["translation_p90_m"]),
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration_report", required=True)
    parser.add_argument("--dense_geometry_head", required=True)
    parser.add_argument("--canonical_field_summary", required=True)
    parser.add_argument("--mapping_contributors", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_policy", required=True)
    parser.add_argument("--transfer_reports", nargs="*", default=())
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--success_translation_m", type=float, default=0.5)
    parser.add_argument("--success_rotation_deg", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_json)
    policy_path = Path(args.output_policy)
    if (output_path.exists() or policy_path.exists()) and not bool(args.force):
        raise FileExistsError("refusing to overwrite joint-likelihood artifacts")
    calibration_path = Path(args.calibration_report)
    calibration = json.loads(calibration_path.read_text())
    contract = _validate_report_contract(calibration)
    overlap_audit = _map_query_overlap_audit(
        calibration,
        canonical_field_summary=Path(args.canonical_field_summary),
        mapping_contributors=Path(args.mapping_contributors),
    )
    head_audit = _geometry_head_overlap_audit(Path(args.dense_geometry_head), calibration)
    queries = _load_queries(
        calibration,
        mode_name=str(args.mode_name),
        success_translation_m=float(args.success_translation_m),
        success_rotation_deg=float(args.success_rotation_deg),
    )
    trajectories = sorted({row.trajectory for row in queries})
    if len(trajectories) < 3:
        raise ValueError("joint likelihood requires at least three query trajectories")
    baseline_predictions = sorted(
        [_prediction(row, np.zeros(4), baseline=True) for row in queries],
        key=lambda item: str(item["image_id"]),
    )
    variants = {
        "proposal_only": (True, False, False, False),
        "phase_only": (False, True, False, False),
        "phase_geometry": (False, True, False, True),
        "identity_phase_observation": (True, True, True, False),
        "joint_all": (True, True, True, True),
    }
    variant_reports = {}
    for name, active in variants.items():
        predictions = []
        folds = []
        for held_out in trajectories:
            train = [row for row in queries if row.trajectory != held_out]
            validation = [row for row in queries if row.trajectory == held_out]
            weights, null_logit, optimization = _fit(
                train, active=active, regularization=float(args.regularization),
            )
            fold_predictions = [
                _prediction(row, weights, null_logit) for row in validation
            ]
            predictions.extend(fold_predictions)
            folds.append({
                "held_out_trajectory": held_out,
                "train_query_count": len(train),
                "validation_query_count": len(validation),
                "weights": weights.tolist(),
                "null_logit": float(null_logit),
                "optimization": optimization,
                "metrics": _metrics(fold_predictions),
            })
        predictions.sort(key=lambda item: str(item["image_id"]))
        variant_reports[name] = {
            "active_components": [
                component for component, enabled in zip(COMPONENT_NAMES, active) if enabled
            ],
            "outer_loto_metrics": _metrics(predictions),
            "paired_to_frozen_phase_baseline": _paired(baseline_predictions, predictions),
            "folds": folds,
            "predictions": predictions,
        }
    final_weights, final_null_logit, final_optimization = _fit(
        queries, active=variants["joint_all"], regularization=float(args.regularization),
    )
    baseline_metrics = _metrics(baseline_predictions)
    joint_metrics = variant_reports["joint_all"]["outer_loto_metrics"]
    promotion = _promotion(baseline_metrics, joint_metrics)
    protocol_valid = bool(
        overlap_audit["map_query_image_disjoint"]
        and head_audit["query_head_image_disjoint"]
        and head_audit["query_head_trajectory_disjoint"]
    )
    policy = {
        "artifact_type": "goal_maplet_joint_phase_geometry_likelihood_v2",
        "role": "query_local_candidate_posterior",
        "component_names": list(COMPONENT_NAMES),
        "weights": final_weights.tolist(),
        "null_logit": float(final_null_logit),
        "typed_null_hypothesis": True,
        "scale_floors": DEFAULT_SCALE_FLOORS.tolist(),
        "normalization": "per_query_median_iqr_with_fixed_floors_v1",
        "monotonic_nonnegative_weights": True,
        "candidate_pool_frozen": True,
        "dense_geometry_definition": "scale_marginalized_2dgs_depth_normal_fixed_denominator_v1",
        "phase_definition": "frozen_surface_phase_policy_score_v1",
        "calibration_protocol": "outer_leave_one_query_trajectory_out_map_and_head_disjoint_v1",
        "calibration_trajectories": trajectories,
        "success_target": {
            "translation_m": float(args.success_translation_m),
            "rotation_deg": float(args.success_rotation_deg),
        },
        "deployment_allowed": bool(protocol_valid and promotion["passed"]),
        "production_deployment_allowed": False,
        "untouched_test_evaluated": False,
        "protocol_valid_for_promotion": protocol_valid,
        "loto_promotion_passed": bool(promotion["passed"]),
        "calibration_report_sha256": file_sha256(calibration_path),
        "calibration_physical_map_sha256": calibration.get("physical_map_sha256"),
        "calibration_canonical_field_sha256": calibration.get("canonical_field_sha256"),
        "calibration_phase_readout_policy_sha256": contract.get("phase_readout_policy_sha256"),
        "dense_geometry_head_sha256": contract.get("dense_geometry_head_sha256"),
        "dense_geometry_selected": bool(final_weights[3] > 1.0e-8),
        "required_render_supersample_factor": 2,
        "map_fold_transfer_uses_query_local_normalization": True,
        "stores_mapping_rgb": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
        "fit_optimization": final_optimization,
    }
    transfer = []
    for value in args.transfer_reports:
        path = Path(value)
        report = json.loads(path.read_text())
        transfer_contract = _validate_report_contract(report)
        head_differs = str(transfer_contract["dense_geometry_head_sha256"]) != str(
            contract["dense_geometry_head_sha256"]
        )
        if head_differs and final_weights[3] > 1.0e-8:
            raise ValueError("transfer report uses a different dense geometry head")
        rows = _load_queries(
            report,
            mode_name=str(args.mode_name),
            success_translation_m=float(args.success_translation_m),
            success_rotation_deg=float(args.success_rotation_deg),
        )
        reference = [_prediction(row, np.zeros(4), baseline=True) for row in rows]
        candidate = [
            _prediction(row, final_weights, final_null_logit) for row in rows
        ]
        transfer.append({
            "report": str(path),
            "report_sha256": file_sha256(path),
            "trajectory_ids": sorted({row.trajectory for row in rows}),
            "geometry_head_differs_but_component_inactive": bool(head_differs),
            "frozen_phase_baseline": _metrics(reference),
            "frozen_phase_predictions": reference,
            "joint_likelihood": _metrics(candidate),
            "paired": _paired(reference, candidate),
            "predictions": candidate,
        })
    result = {
        "stage": "goal_maplet_joint_phase_geometry_trajectory_loto",
        "calibration_report": str(calibration_path),
        "query_count": len(queries),
        "trajectory_ids": trajectories,
        "map_query_overlap_audit": overlap_audit,
        "geometry_head_overlap_audit": head_audit,
        "protocol_valid_for_promotion": protocol_valid,
        "frozen_phase_baseline": {
            "metrics": baseline_metrics,
            "predictions": baseline_predictions,
        },
        "variants": variant_reports,
        "promotion": promotion,
        "deployed_policy": policy,
        "transfer_evaluation": transfer,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "query_count": len(queries),
        "trajectory_ids": trajectories,
        "protocol_valid_for_promotion": protocol_valid,
        "baseline": baseline_metrics,
        "joint_loto": joint_metrics,
        "weights": final_weights.tolist(),
        "promotion": promotion,
        "transfer_evaluation": transfer,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
