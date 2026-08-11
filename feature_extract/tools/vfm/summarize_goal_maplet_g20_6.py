"""Summarize G20.6 cross-trajectory Stage-C and mapping-readout decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.fit_evaluate_goal_maplet_joint_phase_geometry import (
    _metrics,
    _paired,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint_reports", nargs="+", required=True)
    parser.add_argument("--fallback_report", required=True)
    parser.add_argument("--teacher_compression_audit", required=True)
    parser.add_argument("--baseline_readout_summary", required=True)
    parser.add_argument("--signed_readout_summaries", nargs="+", required=True)
    parser.add_argument("--runtime_smoke", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {output}")
    reports = []
    baseline_predictions = []
    joint_predictions = []
    for value in args.joint_reports:
        path = Path(value)
        payload = json.loads(path.read_text())
        transfer = list(payload.get("transfer_evaluation", ()))
        if len(transfer) != 1:
            raise ValueError("each G20.6 fold report must contain one outer transfer")
        baseline_predictions.extend(transfer[0]["frozen_phase_predictions"])
        joint_predictions.extend(transfer[0]["predictions"])
        reports.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "trajectory_ids": transfer[0]["trajectory_ids"],
            "loto_promotion": payload["promotion"],
            "weights": payload["deployed_policy"]["weights"],
            "dense_geometry_selected": payload["deployed_policy"]["dense_geometry_selected"],
            "transfer_baseline": transfer[0]["frozen_phase_baseline"],
            "transfer_joint": transfer[0]["joint_likelihood"],
        })
    image_ids = [str(item["image_id"]) for item in joint_predictions]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("outer G20.6 transfers overlap")
    baseline_metrics = _metrics(baseline_predictions)
    joint_metrics = _metrics(joint_predictions)
    stage_c_checks = {
        "all_calibration_loto_gates_pass": all(item["loto_promotion"]["passed"] for item in reports),
        "strict_non_decreasing": float(joint_metrics["strict_0.5m_5deg"]) >= float(baseline_metrics["strict_0.5m_5deg"]),
        "loose_non_decreasing": float(joint_metrics["within_1m_10deg"]) >= float(baseline_metrics["within_1m_10deg"]),
        "catastrophic_non_increasing": float(joint_metrics["catastrophic_rate"]) <= float(baseline_metrics["catastrophic_rate"]),
        "translation_p90_non_increasing": float(joint_metrics["translation_p90_m"]) <= float(baseline_metrics["translation_p90_m"]),
    }
    stage_c_passed = bool(all(stage_c_checks.values()))
    geometry_selected = bool(all(item["dense_geometry_selected"] for item in reports))
    fallback_path = Path(args.fallback_report)
    fallback_payload = json.loads(fallback_path.read_text())
    fallback_baseline = []
    fallback_predictions = []
    for transfer in fallback_payload.get("transfer_evaluation", ()):
        fallback_baseline.extend(transfer["frozen_phase_predictions"])
        fallback_predictions.extend(transfer["predictions"])
    if {str(item["image_id"]) for item in fallback_predictions} != set(image_ids):
        raise ValueError("fallback and joint outer transfers differ")
    fallback_baseline_metrics = _metrics(fallback_baseline)
    fallback_metrics = _metrics(fallback_predictions)
    fallback_checks = {
        "calibration_loto_gate_pass": bool(fallback_payload["promotion"]["passed"]),
        "strict_non_decreasing": float(fallback_metrics["strict_0.5m_5deg"]) >= float(fallback_baseline_metrics["strict_0.5m_5deg"]),
        "loose_non_decreasing": float(fallback_metrics["within_1m_10deg"]) >= float(fallback_baseline_metrics["within_1m_10deg"]),
        "catastrophic_non_increasing": float(fallback_metrics["catastrophic_rate"]) <= float(fallback_baseline_metrics["catastrophic_rate"]),
        "translation_p90_non_increasing": float(fallback_metrics["translation_p90_m"]) <= float(fallback_baseline_metrics["translation_p90_m"]),
    }
    fallback_passed = bool(all(fallback_checks.values()))
    baseline_readout_path = Path(args.baseline_readout_summary)
    baseline_readout = json.loads(baseline_readout_path.read_text())
    signed = []
    for value in args.signed_readout_summaries:
        path = Path(value)
        payload = json.loads(path.read_text())
        signed.append({
            "path": str(path),
            "sha256": file_sha256(path),
            "selection_best_score": payload["selection_best_score"],
            "selected_validation": payload["selected_validation"],
        })
    best_signed = max(
        signed,
        key=lambda item: float(item["selected_validation"]["joint_parent32_child16"]),
    )
    baseline_joint = float(baseline_readout["selected_validation"]["joint_parent32_child16"])
    signed_joint = float(best_signed["selected_validation"]["joint_parent32_child16"])
    teacher_path = Path(args.teacher_compression_audit)
    runtime_smoke = None
    if args.runtime_smoke:
        runtime_path = Path(args.runtime_smoke)
        runtime_payload = json.loads(runtime_path.read_text())
        contract = runtime_payload.get("surface_verification_contract", {})
        runtime_smoke = {
            "path": str(runtime_path),
            "sha256": file_sha256(runtime_path),
            "query_count": int(runtime_payload.get("query_count", 0)),
            "score": contract.get("score"),
            "joint_dense_geometry_selected": bool(
                contract.get("joint_dense_geometry_selected", False)
            ),
            "stored_map_feature_type_count": contract.get(
                "stored_map_feature_type_count"
            ),
            "stored_downstream_embedding_count": contract.get(
                "stored_downstream_embedding_count"
            ),
            "stores_mapping_rgb": contract.get("stores_mapping_rgb"),
            "uses_point_correspondences": contract.get("uses_point_correspondences"),
            "uses_pnp": contract.get("uses_pnp"),
        }
        expected_runtime = {
            "joint_dense_geometry_selected": False,
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
        }
        for key, expected in expected_runtime.items():
            if runtime_smoke[key] != expected:
                raise ValueError(f"G20.6 runtime smoke contract differs: {key}")
    result = {
        "stage": "goal_maplet_g20_6_decision",
        "stage_c": {
            "fold_reports": reports,
            "frozen_phase_baseline": baseline_metrics,
            "evaluated_dense_joint_likelihood": joint_metrics,
            "paired_dense_joint": _paired(baseline_predictions, joint_predictions),
            "checks": stage_c_checks,
            "development_promotion_passed": stage_c_passed,
            "dense_geometry_promoted": bool(stage_c_passed and geometry_selected),
            "reliability_fallback": {
                "path": str(fallback_path),
                "sha256": file_sha256(fallback_path),
                "baseline": fallback_baseline_metrics,
                "metrics": fallback_metrics,
                "paired": _paired(fallback_baseline, fallback_predictions),
                "checks": fallback_checks,
                "development_promotion_passed": fallback_passed,
                "weights": fallback_payload["deployed_policy"]["weights"],
                "dense_geometry_selected": fallback_payload["deployed_policy"]["dense_geometry_selected"],
            },
            "selected_policy_family": (
                "joint_phase_dense_geometry"
                if stage_c_passed and geometry_selected
                else (
                    "reliability_calibrated_phase_observation"
                    if fallback_passed else "frozen_parameter_free_phase"
                )
            ),
            "selected_policy_metrics": (
                joint_metrics
                if stage_c_passed and geometry_selected
                else (fallback_metrics if fallback_passed else baseline_metrics)
            ),
            "runtime_smoke": runtime_smoke,
            "production_promotion_passed": False,
            "production_blocker": "untouched_test_not_evaluated",
        },
        "mapping_readout": {
            "teacher_compression_audit": str(teacher_path),
            "teacher_compression_audit_sha256": file_sha256(teacher_path),
            "baseline_summary": str(baseline_readout_path),
            "baseline_summary_sha256": file_sha256(baseline_readout_path),
            "baseline_validation_joint_parent32_child16": baseline_joint,
            "signed_candidates": signed,
            "best_signed_validation_joint_parent32_child16": signed_joint,
            "signed_readout_promoted": bool(signed_joint > baseline_joint),
            "selected_readout": (
                best_signed["path"] if signed_joint > baseline_joint else str(baseline_readout_path)
            ),
        },
        "method_contract": {
            "stored_map_feature_type_count": 1,
            "stored_downstream_embedding_count": 0,
            "stores_mapping_rgb": False,
            "uses_point_correspondences": False,
            "uses_pnp": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
