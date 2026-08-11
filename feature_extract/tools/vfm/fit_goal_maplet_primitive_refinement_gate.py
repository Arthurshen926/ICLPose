"""Fit and LOTO-validate the low-capacity primitive-VFM refinement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.joint_phase_geometry_likelihood import (
    candidate_measurements,
    load_joint_phase_geometry_likelihood,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.primitive_refinement_gate import (
    PrimitiveRefinementGate,
    PrimitiveRefinementGateSample,
    fit_primitive_refinement_gate,
    primitive_refinement_gate_metrics,
)


def _trajectory(image_id: str) -> str:
    return str(image_id).replace("\\", "/").split("/", 1)[0]


def _evidence(row: dict[str, object], likelihood) -> tuple[float, float]:
    name = "actual_parent_actual_child"
    details = row["mode_details"][name]
    diagnostics = row["ranking_diagnostics"][name]
    count = int(diagnostics["surface_alignment_evaluated_count"])
    if count < 2:
        raise ValueError("refinement gate requires at least two phase candidates")
    original = diagnostics["surface_alignment_original_indices"]
    components = diagnostics["surface_phase_components_preorder"]
    candidates = [
        (details[index], components[int(original[index])]) for index in range(count)
    ]
    phase = np.asarray(
        [candidate[0]["surface_alignment_score"] for candidate in candidates],
        dtype=np.float64,
    )
    measurements = candidate_measurements(
        np.asarray([candidate[0]["pre_surface_score"] for candidate in candidates]),
        phase,
        [candidate[1] for candidate in candidates],
        [{"score": 0.0} for _ in candidates],
    )
    posterior_max = float(np.max(likelihood.posterior(measurements)))
    phase_margin = float(np.max(phase) - np.partition(phase, -2)[-2])
    return posterior_max, phase_margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration_report", required=True)
    parser.add_argument("--refinement_report", required=True)
    parser.add_argument("--joint_likelihood", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite primitive refinement gate")
    calibration_path = Path(args.calibration_report)
    refinement_path = Path(args.refinement_report)
    likelihood_path = Path(args.joint_likelihood)
    calibration = json.loads(calibration_path.read_text())
    refinement = json.loads(refinement_path.read_text())
    if refinement.get("candidate_report_sha256") != file_sha256(calibration_path):
        raise ValueError("refinement and calibration candidate report differ")
    likelihood = load_joint_phase_geometry_likelihood(likelihood_path)
    refined_by_image = {
        str(row["image_id"]): row for row in refinement.get("rows", [])
    }
    samples: list[PrimitiveRefinementGateSample] = []
    trajectories: list[str] = []
    image_ids: list[str] = []
    for row in calibration.get("rows", []):
        image_id = str(row["image_id"])
        refined = refined_by_image.get(image_id)
        if refined is None:
            raise ValueError(f"missing refinement calibration query: {image_id}")
        posterior_max, phase_margin = _evidence(row, likelihood)
        samples.append(
            PrimitiveRefinementGateSample(
                posterior_max=posterior_max,
                phase_margin=phase_margin,
                initial_translation_m=float(refined["initial_translation_m"]),
                initial_rotation_deg=float(refined["initial_rotation_deg"]),
                refined_translation_m=float(refined["final_translation_m"]),
                refined_rotation_deg=float(refined["final_rotation_deg"]),
            )
        )
        trajectories.append(_trajectory(image_id))
        image_ids.append(image_id)
    deployed = fit_primitive_refinement_gate(samples)
    never = PrimitiveRefinementGate(-1.0, float("inf"))
    folds = []
    held_out_samples: list[tuple[PrimitiveRefinementGateSample, PrimitiveRefinementGate]] = []
    for trajectory in sorted(set(trajectories)):
        train = [sample for sample, value in zip(samples, trajectories) if value != trajectory]
        validation = [sample for sample, value in zip(samples, trajectories) if value == trajectory]
        gate = fit_primitive_refinement_gate(train)
        held_out_samples.extend((sample, gate) for sample in validation)
        folds.append(
            {
                "held_out_trajectory": trajectory,
                "maximum_posterior": float(gate.maximum_posterior),
                "minimum_phase_margin": float(gate.minimum_phase_margin),
                "train_metrics": primitive_refinement_gate_metrics(train, gate),
                "validation_metrics": primitive_refinement_gate_metrics(validation, gate),
                "validation_baseline_metrics": primitive_refinement_gate_metrics(
                    validation, never
                ),
            }
        )
    loto_selected = []
    loto_refinement_count = 0
    for sample, gate in held_out_samples:
        selected = gate.select(sample.posterior_max, sample.phase_margin)
        loto_refinement_count += int(selected)
        loto_selected.append(
            PrimitiveRefinementGateSample(
                posterior_max=0.0,
                phase_margin=1.0,
                initial_translation_m=(
                    sample.refined_translation_m if selected else sample.initial_translation_m
                ),
                initial_rotation_deg=(
                    sample.refined_rotation_deg if selected else sample.initial_rotation_deg
                ),
                refined_translation_m=(
                    sample.refined_translation_m if selected else sample.initial_translation_m
                ),
                refined_rotation_deg=(
                    sample.refined_rotation_deg if selected else sample.initial_rotation_deg
                ),
            )
        )
    loto_metrics = primitive_refinement_gate_metrics(loto_selected, never)
    loto_metrics["selected_refinement_count"] = int(loto_refinement_count)
    baseline_metrics = primitive_refinement_gate_metrics(samples, never)
    fold_success_non_regression = all(
        int(fold["validation_metrics"]["strict_count"])
        >= int(fold["validation_baseline_metrics"]["strict_count"])
        and int(fold["validation_metrics"]["within_1m_10deg_count"])
        >= int(fold["validation_baseline_metrics"]["within_1m_10deg_count"])
        and int(fold["validation_metrics"]["catastrophic_count"])
        <= int(fold["validation_baseline_metrics"]["catastrophic_count"])
        for fold in folds
    )
    payload = {
        "artifact_type": "goal_maplet_primitive_refinement_gate_v1",
        "role": "selective_pose_conditioned_primitive_vfm_trust_region",
        "maximum_posterior": float(deployed.maximum_posterior),
        "minimum_phase_margin": float(deployed.minimum_phase_margin),
        "normalization": "same_query_joint_phase_posterior_and_top2_phase_margin",
        "calibration_trajectories": sorted(set(trajectories)),
        "calibration_image_count": int(len(image_ids)),
        "outer_loto_baseline": baseline_metrics,
        "outer_loto_selected": loto_metrics,
        "outer_loto_refinement_count": int(loto_refinement_count),
        "full_fit_selected": primitive_refinement_gate_metrics(samples, deployed),
        "promotion": {
            "fold_success_non_regression": bool(fold_success_non_regression),
            "aggregate_strict_non_decreasing": bool(
                int(loto_metrics["strict_count"]) >= int(baseline_metrics["strict_count"])
            ),
            "aggregate_loose_non_decreasing": bool(
                int(loto_metrics["within_1m_10deg_count"])
                >= int(baseline_metrics["within_1m_10deg_count"])
            ),
            "aggregate_catastrophic_non_increasing": bool(
                int(loto_metrics["catastrophic_count"])
                <= int(baseline_metrics["catastrophic_count"])
            ),
            "aggregate_risk_improved": bool(
                float(loto_metrics["clipped_normalized_risk_sum"])
                < float(baseline_metrics["clipped_normalized_risk_sum"])
            ),
            "passed": bool(
                fold_success_non_regression
                and int(loto_metrics["strict_count"]) >= int(baseline_metrics["strict_count"])
                and int(loto_metrics["within_1m_10deg_count"])
                >= int(baseline_metrics["within_1m_10deg_count"])
                and int(loto_metrics["catastrophic_count"])
                <= int(baseline_metrics["catastrophic_count"])
                and float(loto_metrics["clipped_normalized_risk_sum"])
                < float(baseline_metrics["clipped_normalized_risk_sum"])
            ),
        },
        "folds": folds,
        "calibration_report_sha256": file_sha256(calibration_path),
        "refinement_report_sha256": file_sha256(refinement_path),
        "joint_likelihood_sha256": file_sha256(likelihood_path),
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
