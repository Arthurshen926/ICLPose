"""Seal strict baseline versus seq10-frozen layout-allocator raw support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.tools.vfm.summarize_goal_maplet_allocator_downstream_raw_support import (
    _branch,
    _delta,
)
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256


SCHEMA = "goal_maplet_layout_allocator_downstream_raw_support_comparison_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for route in ("seq12", "seq14"):
        for branch in ("baseline", "layout"):
            parser.add_argument(f"--{route}_{branch}_coverage", required=True)
            parser.add_argument(f"--{route}_{branch}_direct", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite layout allocator comparison")
    routes: dict[str, object] = {}
    route_gate: dict[str, bool] = {}
    for route in ("seq12", "seq14"):
        baseline = _branch(
            Path(getattr(args, f"{route}_baseline_coverage")),
            Path(getattr(args, f"{route}_baseline_direct")),
            expected_route=route, branch="baseline",
        )
        layout = _branch(
            Path(getattr(args, f"{route}_layout_coverage")),
            Path(getattr(args, f"{route}_layout_direct")),
            expected_route=route, branch="layout",
        )
        if baseline["query_count"] != layout["query_count"]:
            raise ValueError("baseline/layout query counts differ")
        comparison = _delta(layout, baseline)
        route_gate[route] = bool(comparison["strict_pareto_improvement"])
        routes[route] = {
            "baseline": baseline,
            "layout": layout,
            "comparison": comparison,
        }
    gate = bool(all(route_gate.values()))
    report: dict[str, object] = {
        "artifact_type": SCHEMA,
        "routes": routes,
        "pre_registered_robust_raw_improvement_gate": {
            "semantics": (
                "strict_non_degradation_on_all_pool_and_factor_basin_counts_"
                "on_each_held_route_and_at_least_one_strict_gain_v1"
            ),
            "per_route": route_gate,
            "layout_allocator_passed": gate,
        },
        "parent_layout_guide_rerun_permitted": gate,
        "decision": (
            "continue_parent_layout_guide"
            if gate else "stop_after_raw_support_due_to_non_pareto_pose_support"
        ),
        "held_labels_used_during_pool_or_factor_generation": False,
        "final_pose_pool_budget": 64,
        "final_position_seed_budget": 4,
        "final_orientation_budget": 64,
        "raw_factor_support_is_implicit_cartesian_upper_bound": True,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(output.resolve()),
        "content_sha256": report["content_sha256"],
        "gate": report["pre_registered_robust_raw_improvement_gate"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
