"""Build a source-only physical-seam-filtered chart submap plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.physical_seam_chart_selection import (
    PhysicalSeamSelectionConfig,
    build_physical_seam_chart_submap_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream_plan", type=Path, required=True)
    parser.add_argument("--expected_upstream_plan_content_sha256", required=True)
    parser.add_argument("--reference_safe_domain_v3", type=Path, required=True)
    parser.add_argument(
        "--expected_reference_safe_domain_content_sha256", required=True
    )
    parser.add_argument("--source_seam_authority", type=Path, required=True)
    parser.add_argument(
        "--expected_source_seam_authority_content_sha256", required=True
    )
    parser.add_argument("--minimum_selected_charts", type=int, default=12)
    parser.add_argument("--maximum_selected_charts", type=int, default=16)
    parser.add_argument("--target_per_view_surface_support", type=float)
    parser.add_argument(
        "--diagnostic_allow_legacy_closest_surface",
        action="store_true",
        help=(
            "permit a non-promotable dry-run with the legacy closest-world-"
            "triangle authority; production use requires a sealed projective "
            "exact-face edge_formal_valid mask"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit_output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit_output.exists():
        raise FileExistsError("refusing to overwrite physical-seam selection output")

    result = build_physical_seam_chart_submap_plan(
        args.upstream_plan,
        expected_upstream_plan_content_sha256=(
            args.expected_upstream_plan_content_sha256
        ),
        reference_safe_domain_v3_path=args.reference_safe_domain_v3,
        expected_reference_safe_domain_content_sha256=(
            args.expected_reference_safe_domain_content_sha256
        ),
        source_seam_authority_path=args.source_seam_authority,
        expected_source_seam_authority_content_sha256=(
            args.expected_source_seam_authority_content_sha256
        ),
        config=PhysicalSeamSelectionConfig(
            minimum_selected_charts=args.minimum_selected_charts,
            maximum_selected_charts=args.maximum_selected_charts,
            target_per_view_surface_support=(
                args.target_per_view_surface_support
            ),
            diagnostic_allow_legacy_closest_surface=(
                args.diagnostic_allow_legacy_closest_surface
            ),
        ),
    )
    audit = dict(result.audit)
    if result.plan is None:
        audit.update(
            {
                "output_plan_path": None,
                "requested_output_plan_path": str(args.output.resolve()),
                "output_plan_created": False,
            }
        )
        audit["content_sha256"] = canonical_json_sha256(audit)
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True))
        print(
            json.dumps(
                {
                    "formal_decision": audit.get("formal_decision", "KILL"),
                    "failure_reason": audit.get("failure_reason"),
                    "output_plan_created": False,
                    "audit_output": str(args.audit_output),
                    "audit_file_sha256": file_sha256(args.audit_output),
                    "audit_content_sha256": audit["content_sha256"],
                    "maximum_usable_component_chart_count": audit.get(
                        "maximum_usable_component_chart_count"
                    ),
                    "required_minimum_selected_charts": audit.get(
                        "required_minimum_selected_charts"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    plan_metadata = result.plan.save_npz(args.output)
    audit.update(
        {
            "output_plan_path": str(args.output.resolve()),
            "output_plan_file_sha256": file_sha256(args.output),
            "output_plan_content_sha256": plan_metadata["content_sha256"],
        }
    )
    audit["content_sha256"] = canonical_json_sha256(audit)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_file_sha256": file_sha256(args.output),
                "content_sha256": plan_metadata["content_sha256"],
                "audit_output": str(args.audit_output),
                "audit_file_sha256": file_sha256(args.audit_output),
                "audit_content_sha256": audit["content_sha256"],
                "selected_chart_count": plan_metadata["selected_chart_count"],
                "selected_chart_names_in_order": plan_metadata[
                    "selected_chart_names_in_order"
                ],
                "promotion_eligible": plan_metadata["promotion_eligible"],
                "source_geometry_selection_eligible": plan_metadata[
                    "source_geometry_selection_eligible"
                ],
                "comparison_inventory_eligible": plan_metadata[
                    "comparison_inventory_eligible"
                ],
                "system_control_only": plan_metadata["system_control_only"],
                "full_gate_or_exporter_consumption_eligible": plan_metadata[
                    "full_gate_or_exporter_consumption_eligible"
                ],
                "source_seam_authority_production_eligible": plan_metadata[
                    "source_seam_authority_production_eligible"
                ],
                "source_seam_authority_formal_selector_handoff_eligible": (
                    plan_metadata[
                        "source_seam_authority_formal_selector_handoff_eligible"
                    ]
                ),
                "topology_stride": plan_metadata["topology_stride"],
                "paired_stride2_densification_diagnostic": plan_metadata[
                    "paired_stride2_densification_diagnostic"
                ],
                "topology_caveat": plan_metadata["topology_caveat"],
                "final_model_neutral_map_topology_eligible": plan_metadata[
                    "final_model_neutral_map_topology_eligible"
                ],
                "source_seam_authority_semantics_version": plan_metadata[
                    "source_seam_authority_semantics_version"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
