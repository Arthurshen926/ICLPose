"""Build the lineaged decision record for the G19-C correction round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import load_phase_readout_policy


def _pose_metrics(report: dict[str, object]) -> dict[str, object]:
    value = report["summary"]["actual_parent_actual_child"]
    return {
        "translation_median_m": float(value["current_top1_translation_m"]["median"]),
        "translation_p90_m": float(value["current_top1_translation_m"]["p90"]),
        "rotation_median_deg": float(value["current_top1_rotation_deg"]["median"]),
        "rotation_p90_deg": float(value["current_top1_rotation_deg"]["p90"]),
        "strict_0.5m_5deg": float(value["coverage_top1_0.5m_5deg"]),
        "success_1m_10deg": float(value["coverage_top1_1m_10deg"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("basin", "correctness", "seq11_1x", "seq11_2x", "strict_baseline", "strict_1x", "strict_2x", "selected_policy"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not args.force:
        raise FileExistsError("refusing to overwrite G19-C decision record")
    paths = {name: Path(getattr(args, name)) for name in ("basin", "correctness", "seq11_1x", "seq11_2x", "strict_baseline", "strict_1x", "strict_2x", "selected_policy")}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items() if name != "selected_policy"}
    physical_hashes = {str(value.get("physical_map_sha256", "")) for value in reports.values()}
    canonical_hashes = {str(value.get("canonical_field_sha256", "")) for value in reports.values()}
    if len(physical_hashes) != 1 or len(canonical_hashes) != 1:
        raise ValueError("G19-C report lineage differs")
    policy = load_phase_readout_policy(paths["selected_policy"])
    factor = int(policy.metadata.get("required_render_supersample_factor", 0))
    if factor != 2:
        raise ValueError("G19-C selected policy is not bound to 2x rendering")
    basin_gate = reports["basin"]["summary"]["gate"]
    stability = reports["correctness"]["summary"]["candidate_order_stability"]
    result = {
        "stage": "goal_maplet_g19_c_decision_record",
        "physical_map_sha256": next(iter(physical_hashes)),
        "canonical_field_sha256": next(iter(canonical_hashes)),
        "selected_policy_sha256": file_sha256(paths["selected_policy"]),
        "required_render_supersample_factor": factor,
        "metrics": {
            "seq11_1x": _pose_metrics(reports["seq11_1x"]),
            "seq11_2x": _pose_metrics(reports["seq11_2x"]),
            "strict12_graph_baseline": _pose_metrics(reports["strict_baseline"]),
            "strict12_1x": _pose_metrics(reports["strict_1x"]),
            "strict12_2x": _pose_metrics(reports["strict_2x"]),
        },
        "basin_gate": basin_gate,
        "raster_stability": stability,
        "decisions": {
            "phase_ranker_research_protocol": "mask_aware_2x_supersample",
            "continuous_refiner_enabled": False,
            "continuous_refiner_blocker": "no reliable six_dof local maximum at 0.25m",
            "legacy_implicit_mixture_enabled": False,
            "strict12_reopened_for_future_tuning": False,
            "centimetre_target_reached": False,
            "next_method_gate": "orientation_equivariant_spatial_phase plus conditional identity_phase_observability_energy trained outside seq11_and_strict12",
        },
        "lineage": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "lineage"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
