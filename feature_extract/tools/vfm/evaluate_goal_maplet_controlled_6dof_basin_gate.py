"""Issue a fail-closed local-optimizer gate from the controlled 6-DoF report.

This is a parameter-free post-hoc audit.  It never treats the GT-relative
stencil as a natural-candidate or end-to-end localization evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.train_evaluate_goal_maplet_sparse_pose_transport import (
    REPORT_SCHEMA,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


SCHEMA = "goal_maplet_controlled_6dof_local_basin_gate_v1"


def _direction_failures(metrics: dict[str, object]) -> dict[str, object]:
    directional = metrics.get("full_6dof_directional_capture")
    if not isinstance(directional, dict) or int(directional.get("direction_count", 0)) != 21:
        raise ValueError("transport report lacks the complete 21-direction 6-DoF audit")
    failures = []
    stable = []
    for direction in directional.get("per_direction", ()):
        for sign_name, sign_value in (
            ("negative", direction["negative_sign"]),
            ("positive", direction["positive_sign"]),
        ):
            pair_accuracy = float(sign_value["radial_pair_accuracy"])
            complete_rate = float(sign_value["complete_path_rate"])
            if pair_accuracy < 1.0 or complete_rate < 1.0:
                row = {
                    "direction_id": int(direction["direction_id"]),
                    "label": str(direction["label"]),
                    "kind": str(direction["kind"]),
                    "sign": sign_name,
                    "radial_pair_correct": int(sign_value["radial_pair_correct"]),
                    "radial_pair_count": int(sign_value["radial_pair_count"]),
                    "radial_pair_accuracy": pair_accuracy,
                    "complete_path_count": int(sign_value["complete_path_count"]),
                    "signed_path_count": int(sign_value["signed_path_count"]),
                    "complete_path_rate": complete_rate,
                }
                failures.append(row)
                if complete_rate == 0.0:
                    stable.append(row)
    return {
        "imperfect_signed_direction_count": len(failures),
        "stable_zero_complete_signed_direction_count": len(stable),
        "imperfect_signed_directions": failures,
        "stable_zero_complete_signed_directions": stable,
        "all_42_signed_directions_strictly_monotonic": not failures,
    }


def _metric_summary(metrics: dict[str, object]) -> dict[str, object]:
    directional = metrics["full_6dof_directional_capture"]
    return {
        "gt_anchor_top1_rate": float(metrics["gt_anchor_top1_rate"]),
        "mean_gt_anchor_margin_over_best_nonanchor": float(
            metrics["mean_gt_anchor_margin_over_best_nonanchor"]
        ),
        "mean_score_error_spearman": float(metrics["mean_score_error_spearman"]),
        "pairwise_order_accuracy": float(metrics["pairwise_order_accuracy"]),
        "radial_pair_correct": int(directional["aggregate"]["radial_pair_correct"]),
        "radial_pair_count": int(directional["aggregate"]["radial_pair_count"]),
        "radial_pair_accuracy": float(directional["aggregate"]["radial_pair_accuracy"]),
        "complete_path_count": int(directional["aggregate"]["complete_path_count"]),
        "signed_path_count": int(directional["aggregate"]["signed_path_count"]),
        "complete_path_rate": float(directional["aggregate"]["complete_path_rate"]),
        "coordinate_axes": directional["coordinate_axes"],
        "pair_couplings": directional["pair_couplings"],
        "minimum_direction_radial_pair_accuracy": float(
            directional["minimum_direction_radial_pair_accuracy"]
        ),
        "minimum_direction_complete_path_rate": float(
            directional["minimum_direction_complete_path_rate"]
        ),
        "direction_failures": _direction_failures(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport_report", required=True)
    parser.add_argument("--output_report", required=True)
    args = parser.parse_args()
    source_path = Path(args.transport_report)
    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError("refusing to overwrite controlled 6-DoF basin gate")
    source = json.loads(source_path.read_text())
    if (
        source.get("artifact_type") != REPORT_SCHEMA
        or source.get("controlled_candidates_are_gt_relative_oracle_diagnostic") is not True
        or source.get("scientific_dataset_contract_audit", {}).get(
            "full_6dof_observability_stencil"
        ) is not True
        or source.get("production_eligible") is not False
    ):
        raise ValueError("transport report is not the controlled 6-DoF diagnostic")
    dev = source["dev_medium_sparse_transport"]
    initial = source["preoptimizer_equal_weight_frozen_baseline"]["dev_medium"]
    dev_failures = _direction_failures(dev)
    anchor_gate = (
        float(dev["gt_anchor_top1_rate"]) == 1.0
        and float(dev["mean_gt_anchor_margin_over_best_nonanchor"]) > 0.0
    )
    directional_gate = bool(dev_failures["all_42_signed_directions_strictly_monotonic"])
    local_optimizer_gate = bool(anchor_gate and directional_gate)
    report = {
        "artifact_type": SCHEMA,
        "transport_report_file_sha256": file_sha256(source_path),
        "dataset_content_sha256": source["dataset_content_sha256"],
        "model_content_sha256": source["model_content_sha256"],
        "candidate_semantics": "GT_relative_oracle_local_stencil_not_natural_candidates",
        "natural_candidate_performance_claim_supported": False,
        "strict_route_disjoint_representation_claim_supported": bool(
            source["strict_route_disjoint_backend_claim_supported"]
        ),
        "dev_medium": _metric_summary(dev),
        "preoptimizer_equal_weight_dev_medium": _metric_summary(initial),
        "training_gain_dev_medium": source[
            "training_gain_over_preoptimizer_baseline"
        ]["dev_medium"],
        "dev_coarse_structural_control": _metric_summary(
            source["dev_coarse_sparse_transport"]
        ),
        "dev_fine_exact_child_structural_control": _metric_summary(
            source["dev_fine_exact_child_edge_control"]
        ),
        "gt_anchor_gate_passed": anchor_gate,
        "all_signed_direction_monotonic_gate_passed": directional_gate,
        "local_optimizer_gate_passed": local_optimizer_gate,
        "local_optimizer_authorized": local_optimizer_gate,
        "decision": (
            "PASS_CONTROLLED_LOCAL_BASIN"
            if local_optimizer_gate
            else "KILL_CURRENT_OBJECTIVE_FOR_LOCAL_OPTIMIZATION"
        ),
        "decision_reason": (
            "GT anchor must be a strict maximum and all 42 signed rays must "
            "decrease monotonically before local optimization is safe"
        ),
        "objective_trajectory_drift_claim_supported": bool(
            source["objective_drift_claim_supported"]
        ),
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
