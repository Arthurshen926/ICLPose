"""Replay a frozen primitive-VFM refinement gate on held-out trajectories."""

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
)


def _refinement_uses_candidate_report(
    refinement_payload: dict[str, object], candidate_sha256: str,
) -> bool:
    """Accept legacy single-source and current multi-source lineage."""

    lineage = refinement_payload.get("candidate_report_sha256")
    if isinstance(lineage, str):
        return lineage == candidate_sha256
    if isinstance(lineage, list):
        return candidate_sha256 in {str(value) for value in lineage}
    return False


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
    return (
        float(np.max(likelihood.posterior(measurements))),
        float(np.max(phase) - np.partition(phase, -2)[-2]),
    )


def _metrics(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int]:
    translation = np.asarray([row[f"{prefix}_translation_m"] for row in rows])
    rotation = np.asarray([row[f"{prefix}_rotation_deg"] for row in rows])
    return {
        "query_count": int(len(rows)),
        "translation_median_m": float(np.median(translation)),
        "translation_p90_m": float(np.percentile(translation, 90.0)),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90.0)),
        "strict_count": int(np.sum((translation <= 0.5) & (rotation <= 5.0))),
        "within_1m_10deg_count": int(np.sum((translation <= 1.0) & (rotation <= 10.0))),
        "catastrophic_count": int(np.sum((translation > 2.0) | (rotation > 20.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--joint_likelihood", required=True)
    parser.add_argument("--baseline_evaluation", required=True)
    parser.add_argument("--candidate_reports", nargs="+", required=True)
    parser.add_argument("--refinement_reports", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if len(args.candidate_reports) != len(args.refinement_reports):
        raise ValueError("candidate and refinement report counts differ")
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite refinement-gate evaluation")
    gate_path = Path(args.gate)
    gate_payload = json.loads(gate_path.read_text())
    if gate_payload.get("artifact_type") != "goal_maplet_primitive_refinement_gate_v1":
        raise ValueError("not a primitive refinement gate")
    if not bool((gate_payload.get("promotion") or {}).get("passed")):
        raise ValueError("primitive refinement gate did not pass LOTO promotion")
    gate = PrimitiveRefinementGate(
        maximum_posterior=float(gate_payload["maximum_posterior"]),
        minimum_phase_margin=float(gate_payload["minimum_phase_margin"]),
    )
    likelihood_path = Path(args.joint_likelihood)
    if gate_payload.get("joint_likelihood_sha256") != file_sha256(likelihood_path):
        raise ValueError("refinement gate and joint likelihood differ")
    likelihood = load_joint_phase_geometry_likelihood(likelihood_path)
    baseline_path = Path(args.baseline_evaluation)
    baseline_payload = json.loads(baseline_path.read_text())
    baseline = {
        str(row["image_id"]): row
        for fold in baseline_payload.get("transfer_evaluation", [])
        for row in fold.get("predictions", [])
    }
    rows: list[dict[str, object]] = []
    for candidate_name, refinement_name in zip(
        args.candidate_reports, args.refinement_reports
    ):
        candidate_path, refinement_path = Path(candidate_name), Path(refinement_name)
        candidates = json.loads(candidate_path.read_text())
        refinements = json.loads(refinement_path.read_text())
        if not _refinement_uses_candidate_report(
            refinements, file_sha256(candidate_path)
        ):
            raise ValueError("refinement and candidate report differ")
        refined_by_image = {
            str(row["image_id"]): row for row in refinements.get("rows", [])
        }
        for candidate in candidates.get("rows", []):
            image_id = str(candidate["image_id"])
            base = baseline.get(image_id)
            refined = refined_by_image.get(image_id)
            if base is None or refined is None:
                raise ValueError(f"missing frozen baseline/refinement query: {image_id}")
            posterior_max, phase_margin = _evidence(candidate, likelihood)
            selected = gate.select(posterior_max, phase_margin)
            rows.append(
                {
                    "image_id": image_id,
                    "refinement_selected": bool(selected),
                    "posterior_max": float(posterior_max),
                    "phase_margin": float(phase_margin),
                    "baseline_translation_m": float(base["translation_m"]),
                    "baseline_rotation_deg": float(base["rotation_deg"]),
                    "selected_translation_m": float(
                        refined["final_translation_m"] if selected else base["translation_m"]
                    ),
                    "selected_rotation_deg": float(
                        refined["final_rotation_deg"] if selected else base["rotation_deg"]
                    ),
                }
            )
    baseline_metrics = _metrics(rows, "baseline")
    selected_metrics = _metrics(rows, "selected")
    payload = {
        "stage": "goal_maplet_selective_primitive_vfm_refinement_evaluation",
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected_metrics,
        "paired": {
            "strict_delta": int(selected_metrics["strict_count"] - baseline_metrics["strict_count"]),
            "within_1m_10deg_delta": int(
                selected_metrics["within_1m_10deg_count"]
                - baseline_metrics["within_1m_10deg_count"]
            ),
            "catastrophic_delta": int(
                selected_metrics["catastrophic_count"]
                - baseline_metrics["catastrophic_count"]
            ),
            "refinement_selected_count": int(
                sum(bool(row["refinement_selected"]) for row in rows)
            ),
        },
        "gate_sha256": file_sha256(gate_path),
        "joint_likelihood_sha256": file_sha256(likelihood_path),
        "baseline_evaluation_sha256": file_sha256(baseline_path),
        "candidate_report_sha256": [file_sha256(Path(path)) for path in args.candidate_reports],
        "refinement_report_sha256": [file_sha256(Path(path)) for path in args.refinement_reports],
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "stores_mapping_rgb": False,
        "uses_pnp": False,
        "uses_point_correspondences": False,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
