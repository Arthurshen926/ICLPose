"""Evaluate one frozen G20 energy on a map/query-disjoint phase report.

This tool never fits a weight or rerenders a pose.  It reconstructs the exact
candidate preorder stored by the runtime verifier, checks that the evaluated
candidate generator matches the calibration generator, audits exact map/query
image overlap from offline build lineage, and applies the serialized energy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.fit_evaluate_goal_maplet_conditional_energy import (
    _load_queries,
    _map_query_overlap_audit,
    _metrics,
    _predict,
)
from feature_extract.vfm.localization_goal_maplet.conditional_pose_energy import (
    load_conditional_pose_energy,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _candidate_generator_contract(report: dict[str, object]) -> dict[str, object]:
    surface = dict(report.get("surface_verification_contract") or {})
    return {
        "proposal_method": report.get("proposal_method"),
        "proposal_seed_policy": report.get("proposal_seed_policy"),
        "graph_seed_parent_pair_count": report.get("graph_seed_parent_pair_count"),
        "identity_render_mode": report.get("identity_render_mode"),
        "render_identity_rerank": report.get("render_identity_rerank"),
        "cascade_contract": report.get("cascade_contract"),
        "base_maximum_modes": report.get("maximum_modes"),
        "maximum_modes": surface.get("maximum_modes"),
        "physical_map_sha256": report.get("physical_map_sha256"),
        "canonical_field_sha256": report.get("canonical_field_sha256"),
        "field_feature_contract_sha256": report.get("field_feature_contract_sha256"),
        "physical_instance_readout_sha256": report.get(
            "physical_instance_readout_sha256"
        ),
        "typed_graph_sha256": report.get("typed_graph_sha256"),
        "validity_calibration_sha256": report.get("validity_calibration_sha256"),
    }


def _predictions(rows, weight: np.ndarray, policy) -> list[dict[str, object]]:
    result = [
        _predict(
            row,
            np.asarray(weight, dtype=np.float64),
            float(policy.null_phase_threshold),
            float(policy.null_phase_scale),
        )
        for row in rows
    ]
    result.sort(key=lambda item: str(item["image_id"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_report", required=True)
    parser.add_argument("--conditional_policy", required=True)
    parser.add_argument("--calibration_evidence", required=True)
    parser.add_argument("--canonical_field_summary", required=True)
    parser.add_argument("--mapping_contributors", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--mode_name", default="actual_parent_actual_child")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite frozen-energy evaluation")

    phase_path = Path(args.phase_report)
    calibration_path = Path(args.calibration_evidence)
    policy_path = Path(args.conditional_policy)
    report = json.loads(phase_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    policy = load_conditional_pose_energy(policy_path)
    if policy.metadata.get("evidence_report_sha256") != file_sha256(calibration_path):
        raise ValueError("conditional policy and calibration evidence differ")
    if _candidate_generator_contract(report) != _candidate_generator_contract(calibration):
        raise ValueError("evaluation and calibration candidate generators differ")
    surface = dict(report.get("surface_verification_contract") or {})
    if surface.get("phase_readout_policy_sha256") != policy.metadata.get(
        "phase_readout_policy_sha256"
    ):
        raise ValueError("evaluation and conditional policy phase readouts differ")
    if int(surface.get("render_supersample_factor", 0)) != int(
        policy.metadata.get("render_supersample_factor", 0)
    ):
        raise ValueError("evaluation and conditional policy render protocols differ")

    overlap = _map_query_overlap_audit(
        report,
        canonical_field_summary=Path(args.canonical_field_summary),
        mapping_contributors=Path(args.mapping_contributors),
    )
    if not bool(overlap["map_query_image_disjoint"]):
        raise ValueError("frozen transfer evaluation is not map/query image disjoint")
    rows = _load_queries(
        report,
        mode_name=str(args.mode_name),
        success_translation_m=float(policy.metadata["success_target"]["translation_m"]),
        success_rotation_deg=float(policy.metadata["success_target"]["rotation_deg"]),
    )
    variants = {
        "identity_only": np.asarray((1.0, 0.0, 0.0)),
        "jacobian_phase_only": np.asarray((0.0, 1.0, 0.0)),
        "frozen_full_energy": np.asarray(policy.weights, dtype=np.float64),
    }
    evaluations = {}
    for name, weight in variants.items():
        predictions = _predictions(rows, weight, policy)
        evaluations[name] = {
            "weights": weight.tolist(),
            "metrics": _metrics(predictions),
            "predictions": predictions,
        }
    result = {
        "stage": "goal_maplet_g20_frozen_map_disjoint_transfer_evaluation",
        "phase_report": str(phase_path),
        "phase_report_sha256": file_sha256(phase_path),
        "conditional_policy": str(policy_path),
        "conditional_policy_sha256": file_sha256(policy_path),
        "calibration_evidence": str(calibration_path),
        "calibration_evidence_sha256": file_sha256(calibration_path),
        "candidate_generator_contract": _candidate_generator_contract(report),
        "map_query_overlap_audit": overlap,
        "query_count": len(rows),
        "trajectory_ids": sorted({row.trajectory for row in rows}),
        "weights_refit_on_evaluation": False,
        "closed_set_used_for_tuning": False,
        "calibration_field_was_self_map": bool(
            not policy.metadata.get("map_query_image_disjoint", False)
        ),
        "production_promotion_allowed": False,
        "production_blocker": (
            "policy weights were calibrated on self-map evidence and the audit sets "
            "were used historically"
        ),
        "evaluations": evaluations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage": result["stage"],
        "query_count": result["query_count"],
        "trajectory_ids": result["trajectory_ids"],
        "map_query_image_disjoint": overlap["map_query_image_disjoint"],
        "evaluations": {
            name: value["metrics"] for name, value in evaluations.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
