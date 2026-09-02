"""Seal the exact16 plan from the MoGe3/reference-only physical graph."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import ChartSubmapPlan
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_exact21_plan", type=Path, required=True)
    parser.add_argument("--expected_source_plan_content_sha256", required=True)
    parser.add_argument("--moge_reference_authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--selected_plan_template", type=Path, required=True)
    parser.add_argument("--expected_template_content_sha256", required=True)
    parser.add_argument("--optimizer_domain", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MoGe/reference exact16 plan")
    source = ChartSubmapPlan.load_npz(args.source_exact21_plan)
    if source.metadata.get("content_sha256") != args.expected_source_plan_content_sha256:
        raise ValueError("source exact21 plan differs from pin")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.moge_reference_authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("MoGe/reference authority differs from pin")
    if authority.metadata.get("moge_reference_only_topology") is not True or authority.metadata.get("dav2_geometry_consumed") is not False:
        raise ValueError("physical authority is not MoGe/reference-only")
    if authority.chart_names.astype(str).tolist() != list(source.selected_chart_names_in_order):
        raise ValueError("exact21 authority and source plan order differ")
    template = ChartSubmapPlan.load_npz(args.selected_plan_template)
    if template.metadata.get("content_sha256") != args.expected_template_content_sha256:
        raise ValueError("selected plan template differs from pin")

    names21 = authority.chart_names.astype(str).tolist()
    row21 = {name: row for row, name in enumerate(names21)}
    physical = np.zeros((len(names21), len(names21)), bool)
    for edge, valid in zip(authority.edge_chart_indices, authority.edge_formal_valid):
        if valid:
            first, second = map(int, edge)
            physical[first, second] = physical[second, first] = True
    # Frozen deterministic selector: source-plan priority, connected expansion.
    priority = list(source.selected_chart_names_in_order)
    selected = [priority[0]]
    while len(selected) < template.chart_count:
        candidates = [
            name for name in priority
            if name not in selected
            and any(physical[row21[name], row21[other]] for other in selected)
        ]
        if not candidates:
            raise ValueError("MoGe/reference physical graph cannot reach exact16")
        selected.append(candidates[0])
    if selected != list(template.selected_chart_names_in_order):
        raise ValueError("MoGe/reference selector does not reproduce frozen exact16 order")

    template_names = template.chart_names.astype(str).tolist()
    for matrix_name in ("coverage_edges", "alignment_edges"):
        matrix = np.asarray(getattr(template, matrix_name), bool)
        for first, second in np.argwhere(np.triu(matrix, 1)):
            if not physical[row21[template_names[first]], row21[template_names[second]]]:
                raise ValueError(f"template {matrix_name} contains a nonphysical edge")

    metadata = dict(template.metadata)
    metadata.pop("arrays_sha256", None)
    metadata.pop("content_sha256", None)
    metadata.update(
        {
            "representation": "offline_source_moge_reference_physical_seam_filtered_chart_selection",
            "paired_stride2_densification_diagnostic": False,
            "moge_reference_only_topology": True,
            "dav2_geometry_consumed": False,
            "paired_initializer_geometry_encoded_upstream": False,
            "source_geometry_selection_scope": "moge_reference_only_stride2_selector_control",
            "selection_geometry_source": ["MoGe3_and_source_MASt3R_reference_on_exact_stride2_topology"],
            "topology_caveat": "model_specific_MoGe3_plus_MASt3R_source_topology_control_not_model_neutral",
            "candidate_official_ordered_names": priority,
            "dropped_candidate_names": [name for name in priority if name not in selected],
            "isolated_candidate_names": [name for name in priority if not physical[row21[name]].any()],
            "physically_isolated_candidate_names": [name for name in priority if not physical[row21[name]].any()],
            "greedy_addition_order_diagnostic_only": selected,
            "diagnostic_alignment_adapter_required": True,
            "moge_reference_alignment_adapter_required": True,
            "system_control_only": True,
            "comparison_inventory_eligible": False,
            "promotion_eligible": False,
            "final_model_neutral_map_topology_eligible": False,
            "full_gate_or_exporter_consumption_eligible": False,
        }
    )
    lineage = dict(metadata.get("lineage", {}))
    for key in tuple(lineage):
        if key.startswith("paired_stride2_") or key.startswith("reference_safe_v3_"):
            lineage.pop(key)
    lineage.update(
        {
            "comparison_inventory_eligible": False,
            "system_control_only": True,
            "moge_reference_only_topology": True,
            "dav2_geometry_consumed": False,
            "paired_initializer_geometry_encoded_upstream": False,
            "projective_topology_stride": 2,
            "topology_caveat": metadata["topology_caveat"],
            "source_seam_authority_file_sha256": file_sha256(args.moge_reference_authority),
            "source_seam_authority_content_sha256": args.expected_authority_content_sha256,
            "source_seam_authority_arrays_sha256": authority.metadata.get("arrays_sha256"),
            "edge_formal_valid_sha256": authority.metadata.get("edge_formal_valid_sha256"),
            "optimizer_domain_file_sha256": file_sha256(args.optimizer_domain),
            "source_exact21_plan_file_sha256": file_sha256(args.source_exact21_plan),
            "source_exact21_plan_content_sha256": args.expected_source_plan_content_sha256,
            "query_or_ground_truth_consumed": False,
            "held_root_opened_by_selector": False,
        }
    )
    metadata["lineage"] = lineage
    output = replace(template, metadata=metadata)
    saved = output.save_npz(args.output)
    replay = ChartSubmapPlan.load_npz(args.output)
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": saved["content_sha256"],
        "selected_chart_names_in_order": list(replay.selected_chart_names_in_order),
        "coverage_edge_count": int(np.triu(replay.coverage_edges, 1).sum()),
        "alignment_edge_count": int(np.triu(replay.alignment_edges, 1).sum()),
        "system_control_only": True,
        "promotion_eligible": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
