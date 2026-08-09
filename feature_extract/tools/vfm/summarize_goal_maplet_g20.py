"""Create the reproducible G20 method and continuous-refinement decision record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256


def _discrete_gate(report: dict[str, object]) -> dict[str, object]:
    variants = report["variants"]
    baseline = variants["identity_only"]["outer_loto_metrics"]
    selected = variants[report["selected_variant"]]["outer_loto_metrics"]
    checks = {
        "translation_p90_non_regression": (
            selected["translation_p90_m"] <= baseline["translation_p90_m"]
        ),
        "strict_non_regression": (
            selected["strict_0.5m_5deg"] >= baseline["strict_0.5m_5deg"]
        ),
        "one_meter_non_regression": (
            selected["within_1m_10deg"] >= baseline["within_1m_10deg"]
        ),
        "catastrophic_non_regression": (
            selected["catastrophic_rate"] <= baseline["catastrophic_rate"]
        ),
    }
    diagnostic_pass = bool(all(checks.values()))
    map_disjoint = bool(
        report.get("map_query_overlap_audit", {}).get("map_query_image_disjoint", False)
    )
    return {
        "checks": checks,
        "self_map_query_trajectory_loto_diagnostic_pass": diagnostic_pass,
        "map_disjoint_cross_acquisition_pass": bool(diagnostic_pass and map_disjoint),
        "map_query_image_disjoint": map_disjoint,
        "production_promotion_allowed": False,
        "production_blocker": (
            "evaluated query images contributed to the canonical field"
            if not map_disjoint else "no historically untouched acquisition remains"
        ),
    }


def _continuous_gate(
    basin: dict[str, object], *, map_query_image_disjoint: bool = False,
) -> dict[str, object]:
    summary = basin["cross_acquisition_summary"]
    checks = {
        "translation_0.25m": summary["gate"]["translation_0.25m_local_max_fraction"] >= 0.8,
        "translation_0.5m": summary["gate"]["translation_0.5m_local_max_fraction"] >= 0.9,
        "rotation_3deg": summary["gate"]["rotation_3deg_local_max_fraction"] >= 0.8,
        "tangent1_local_max": summary["tangent1"]["gt_strict_local_max_fraction"] >= 0.5,
        "tangent2_local_max": summary["tangent2"]["gt_strict_local_max_fraction"] >= 0.5,
        "normal_local_max": summary["normal"]["gt_strict_local_max_fraction"] >= 0.5,
        "roll_local_max": summary["roll"]["gt_strict_local_max_fraction"] >= 0.5,
        "pitch_local_max": summary["pitch"]["gt_strict_local_max_fraction"] >= 0.5,
        "yaw_local_max": summary["yaw"]["gt_strict_local_max_fraction"] >= 0.5,
    }
    component = basin["cross_acquisition_component_summary"]
    phase_normal = component["jacobian_phase_visible"]["normal"][
        "gt_strict_local_max_fraction"
    ]
    scale_normal = component["jacobian_log_scale_agreement"]["normal"][
        "gt_strict_local_max_fraction"
    ]
    basin_pass = bool(all(checks.values()))
    return {
        "checks": checks,
        "self_map_basin_gate_pass": basin_pass,
        "g21_continuous_refinement_open": bool(
            basin_pass and map_query_image_disjoint
        ),
        "g21_blocker": (
            None
            if basin_pass and map_query_image_disjoint
            else (
                "basin queries contributed to the canonical field"
                if not map_query_image_disjoint else "six-DoF basin gate failed"
            )
        ),
        "phase_normal_local_max_fraction": phase_normal,
        "scale_normal_local_max_fraction": scale_normal,
        "normal_evidence_diagnosis": (
            "vfm_gradient_scale_is_complementary"
            if phase_normal < 0.5 <= scale_normal
            else (
                "phase_already_constrains_normal"
                if phase_normal >= 0.5
                else "neither_phase_nor_scale_has_a_reliable_normal_basin"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional_report", required=True)
    parser.add_argument("--basin_report", required=True)
    parser.add_argument("--phase_policy", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite G20 decision record")
    conditional_path = Path(args.conditional_report)
    basin_path = Path(args.basin_report)
    phase_path = Path(args.phase_policy)
    conditional = json.loads(conditional_path.read_text())
    basin = json.loads(basin_path.read_text())
    phase = json.loads(phase_path.read_text())
    if phase.get("artifact_type") != "goal_maplet_phase_readout_policy_v2":
        raise ValueError("G20 decision requires phase policy v2")
    if basin.get("phase_readout_artifact_type") != phase.get("artifact_type"):
        raise ValueError("basin and phase-policy contracts differ")
    result = {
        "stage": "goal_maplet_g20_decision_record_v1",
        "phase_policy_sha256": file_sha256(phase_path),
        "conditional_report_sha256": file_sha256(conditional_path),
        "basin_report_sha256": file_sha256(basin_path),
        "g20_a_fractional_observation_semantics": True,
        "g20_b_orientation_equivariant_phase": True,
        "g20_c_conditional_energy": _discrete_gate(conditional),
        "g20_d_continuous_basin": _continuous_gate(
            basin,
            map_query_image_disjoint=bool(
                conditional.get("map_query_overlap_audit", {}).get(
                    "map_query_image_disjoint", False,
                )
            ),
        ),
        "seq11_reopened": False,
        "strict12_reopened": False,
        "stores_mapping_rgb": False,
        "stored_map_feature_type_count": 1,
        "stored_downstream_embedding_count": 0,
        "uses_point_correspondences": False,
        "uses_pnp": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
