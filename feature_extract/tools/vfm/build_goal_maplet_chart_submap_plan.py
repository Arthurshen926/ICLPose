"""Build a route-clean, overlap-aware explicit-chart submap plan on CPU."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSelectionConfig,
    build_chart_submap_plan,
    load_disjoint_mast3r_source_views,
    load_route_clean_moge3_views,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection_source",
        choices=("mast3r_disjoint_source", "moge3_system_control"),
        default="mast3r_disjoint_source",
    )
    parser.add_argument("--source_root", type=Path)
    parser.add_argument("--disjoint_authority", type=Path)
    parser.add_argument("--cameras", type=Path)
    parser.add_argument("--moge3_initializers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping_routes",
        nargs="+",
        default=["seq1", "seq2", "seq4", "seq6", "seq7", "seq8", "seq9", "seq11"],
    )
    parser.add_argument("--sample_stride", type=int, default=4)
    parser.add_argument("--absolute_depth_tolerance_m", type=float, default=0.30)
    parser.add_argument("--relative_depth_tolerance", type=float, default=0.025)
    parser.add_argument("--maximum_unsigned_normal_angle_deg", type=float, default=35.0)
    parser.add_argument("--minimum_same_surface_side_fraction", type=float, default=0.80)
    parser.add_argument("--minimum_symmetric_surface_overlap", type=float, default=0.08)
    parser.add_argument("--minimum_alignment_baseline_m", type=float, default=0.50)
    parser.add_argument("--maximum_alignment_baseline_m", type=float, default=20.0)
    parser.add_argument("--minimum_baseline_to_depth_ratio", type=float, default=0.025)
    parser.add_argument("--maximum_baseline_to_depth_ratio", type=float, default=0.70)
    parser.add_argument("--minimum_median_triangulation_angle_deg", type=float, default=1.5)
    parser.add_argument("--maximum_median_triangulation_angle_deg", type=float, default=60.0)
    parser.add_argument("--maximum_camera_forward_angle_deg", type=float, default=70.0)
    parser.add_argument("--minimum_submap_views", type=int, default=3)
    parser.add_argument("--target_supported_view_fraction", type=float, default=0.90)
    parser.add_argument("--target_per_view_surface_support", type=float, default=0.25)
    parser.add_argument("--minimum_selected_charts_per_submap", type=int, default=2)
    parser.add_argument("--maximum_selected_charts_per_submap", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("refusing to reuse chart-submap plan output")

    config = ChartSelectionConfig(
        sample_stride=args.sample_stride,
        absolute_depth_tolerance_m=args.absolute_depth_tolerance_m,
        relative_depth_tolerance=args.relative_depth_tolerance,
        maximum_unsigned_normal_angle_deg=args.maximum_unsigned_normal_angle_deg,
        minimum_same_surface_side_fraction=args.minimum_same_surface_side_fraction,
        minimum_symmetric_surface_overlap=args.minimum_symmetric_surface_overlap,
        minimum_alignment_baseline_m=args.minimum_alignment_baseline_m,
        maximum_alignment_baseline_m=args.maximum_alignment_baseline_m,
        minimum_baseline_to_depth_ratio=args.minimum_baseline_to_depth_ratio,
        maximum_baseline_to_depth_ratio=args.maximum_baseline_to_depth_ratio,
        minimum_median_triangulation_angle_deg=args.minimum_median_triangulation_angle_deg,
        maximum_median_triangulation_angle_deg=args.maximum_median_triangulation_angle_deg,
        maximum_camera_forward_angle_deg=args.maximum_camera_forward_angle_deg,
        minimum_submap_views=args.minimum_submap_views,
        target_supported_view_fraction=args.target_supported_view_fraction,
        target_per_view_surface_support=args.target_per_view_surface_support,
        minimum_selected_charts_per_submap=(
            args.minimum_selected_charts_per_submap
        ),
        maximum_selected_charts_per_submap=args.maximum_selected_charts_per_submap,
    ).validated()
    if args.selection_source == "mast3r_disjoint_source":
        if args.source_root is None or args.disjoint_authority is None:
            raise ValueError("MASt3R source selection requires --source_root and --disjoint_authority")
        views, lineage = load_disjoint_mast3r_source_views(
            args.source_root,
            args.disjoint_authority,
        )
    else:
        if args.cameras is None or args.moge3_initializers is None:
            raise ValueError("MoGe-3 system control requires --cameras and --moge3_initializers")
        views, lineage = load_route_clean_moge3_views(
            args.cameras,
            args.moge3_initializers,
            mapping_routes=args.mapping_routes,
        )
    plan = build_chart_submap_plan(views, config=config, lineage=lineage)
    metadata = plan.save_npz(args.output)
    upper = np.triu_indices(plan.chart_count, 1)
    overlap = plan.symmetric_surface_overlap[upper]
    selected_rows = np.flatnonzero(plan.selected_mask)
    selected_rows = selected_rows[np.argsort(plan.selection_rank[selected_rows])]
    strict_config = replace(
        config,
        absolute_depth_tolerance_m=0.75 * config.absolute_depth_tolerance_m,
        relative_depth_tolerance=0.75 * config.relative_depth_tolerance,
        maximum_unsigned_normal_angle_deg=max(5.0, config.maximum_unsigned_normal_angle_deg - 5.0),
        minimum_same_surface_side_fraction=min(0.95, config.minimum_same_surface_side_fraction + 0.10),
        minimum_symmetric_surface_overlap=min(1.0, 1.25 * config.minimum_symmetric_surface_overlap),
        minimum_baseline_to_depth_ratio=1.20 * config.minimum_baseline_to_depth_ratio,
        minimum_median_triangulation_angle_deg=1.25 * config.minimum_median_triangulation_angle_deg,
        maximum_camera_forward_angle_deg=max(10.0, config.maximum_camera_forward_angle_deg - 10.0),
    ).validated()
    loose_config = replace(
        config,
        absolute_depth_tolerance_m=1.25 * config.absolute_depth_tolerance_m,
        relative_depth_tolerance=1.25 * config.relative_depth_tolerance,
        maximum_unsigned_normal_angle_deg=min(90.0, config.maximum_unsigned_normal_angle_deg + 5.0),
        minimum_same_surface_side_fraction=max(0.0, config.minimum_same_surface_side_fraction - 0.10),
        minimum_symmetric_surface_overlap=0.75 * config.minimum_symmetric_surface_overlap,
        minimum_baseline_to_depth_ratio=0.80 * config.minimum_baseline_to_depth_ratio,
        minimum_median_triangulation_angle_deg=0.80 * config.minimum_median_triangulation_angle_deg,
        maximum_camera_forward_angle_deg=min(179.0, config.maximum_camera_forward_angle_deg + 10.0),
    ).validated()
    strict_plan = build_chart_submap_plan(views, config=strict_config, lineage=lineage)
    loose_plan = build_chart_submap_plan(views, config=loose_config, lineage=lineage)

    def jaccard(first: np.ndarray, second: np.ndarray) -> float:
        union = np.logical_or(first, second).sum()
        return float(np.logical_and(first, second).sum() / union) if union else 1.0

    stability = {
        "strict": {
            "config": strict_config.to_dict(),
            "coverage_edge_count": int(np.triu(strict_plan.coverage_edges, 1).sum()),
            "alignment_edge_count": int(np.triu(strict_plan.alignment_edges, 1).sum()),
            "selected_chart_count": int(strict_plan.selected_mask.sum()),
            "operational_submap_count": strict_plan.metadata["operational_submap_count"],
            "base_edge_jaccard": jaccard(strict_plan.coverage_edges, plan.coverage_edges),
            "base_selected_jaccard": jaccard(strict_plan.selected_mask, plan.selected_mask),
        },
        "loose": {
            "config": loose_config.to_dict(),
            "coverage_edge_count": int(np.triu(loose_plan.coverage_edges, 1).sum()),
            "alignment_edge_count": int(np.triu(loose_plan.alignment_edges, 1).sum()),
            "selected_chart_count": int(loose_plan.selected_mask.sum()),
            "operational_submap_count": loose_plan.metadata["operational_submap_count"],
            "base_edge_jaccard": jaccard(loose_plan.coverage_edges, plan.coverage_edges),
            "base_selected_jaccard": jaccard(loose_plan.selected_mask, plan.selected_mask),
        },
    }
    report = {
        "artifact_type": (
            "goal_maplet_overlap_aware_chart_submap_plan_audit_v3"
            if metadata["artifact_type"].endswith("_v3")
            else "goal_maplet_overlap_aware_chart_submap_plan_audit_v2"
        ),
        "plan_path": str(args.output),
        "plan_file_sha256": file_sha256(args.output),
        "plan_content_sha256": metadata["content_sha256"],
        "chart_count": plan.chart_count,
        "selected_chart_count": int(plan.selected_mask.sum()),
        "coverage_edge_count": int(np.triu(plan.coverage_edges, 1).sum()),
        "alignment_edge_count": int(np.triu(plan.alignment_edges, 1).sum()),
        "nonisolated_coverage_chart_count": int((plan.coverage_edges.sum(1) > 0).sum()),
        "surface_overlap_nonzero_pair_fraction": float(np.mean(overlap > 0)),
        "surface_overlap_pair_p90": float(np.quantile(overlap, 0.9)),
        "operational_submap_count": metadata["operational_submap_count"],
        "comparison_inventory_eligible": metadata["comparison_inventory_eligible"],
        "system_control_only": metadata["system_control_only"],
        "threshold_stability": stability,
        "components": metadata["components"],
        "selected_chart_names_in_order": plan.chart_names[selected_rows].tolist(),
        "selected_chart_names_in_order_sha256": metadata[
            "selected_chart_names_in_order_sha256"
        ],
        "alignment_runner_contract": metadata["alignment_runner_contract"],
        "alignment_runner_must_supply_expected_plan_content_sha256": True,
        "alignment_runner_must_not_resample_by_route_or_count": True,
        "decision": (
            "GO_model_neutral_selected_components_to_bounded_alignment_gate"
            if metadata["operational_submap_count"] and metadata["comparison_inventory_eligible"]
            else (
                "SYSTEM_CONTROL_ONLY_not_a_primary_comparison_inventory"
                if metadata["system_control_only"]
                else "KILL_no_component_passed_operational_coverage_rule"
            )
        ),
        "limitations": [
            (
                "source-only MASt3R projective consistency is model-neutral for M1/M2 but is "
                "still a mapping diagnostic, not sensor depth GT"
                if metadata["comparison_inventory_eligible"]
                else "MoGe-3 projective consistency is a system-control proxy and cannot define M1/M2"
            ),
            "operational component coverage is not a semantic complete-facade claim",
            "isolated charts remain explicit coverage holes and must not be silently discarded",
        ],
    }
    report["content_sha256"] = canonical_json_sha256(report)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
