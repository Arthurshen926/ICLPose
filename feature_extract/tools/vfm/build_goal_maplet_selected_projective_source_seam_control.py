"""Subset a sealed projective authority to an exact source-only control plan.

This tool never promotes the paired DAV2/MoGe/MASt3R topology.  It only
recomputes projective correspondences on chart blocks already sealed in a
parent stride-2 authority, using the exact ordered inventory and coverage
edges of a hash-pinned diagnostic selector plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
    load_paired_stride2_diagnostic_alignment_selection,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    ProjectiveSeamConfig,
    _common_valid_normal_stencil,
    _dense_world_normals,
    freeze_projective_exact_face_correspondences,
)


def _subset_topology(
    parent: ProjectiveExactFaceSeamAuthority,
    selected_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    parent_names = parent.chart_names.astype(str).tolist()
    row = {name: index for index, name in enumerate(parent_names)}
    if len(row) != len(parent_names) or any(name not in row for name in selected_names):
        raise ValueError("selected projective chart inventory is absent or ambiguous")
    selected_rows = np.asarray([row[name] for name in selected_names], np.int64)

    vertex_offsets = [0]
    face_offsets = [0]
    pixels: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    for chart in selected_rows.tolist():
        old_vlo, old_vhi = map(
            int, parent.chart_vertex_offsets[chart : chart + 2]
        )
        old_flo, old_fhi = map(int, parent.chart_face_offsets[chart : chart + 2])
        new_vlo = vertex_offsets[-1]
        pixels.append(parent.sampled_vertex_pixel_indices[old_vlo:old_vhi])
        faces.append(
            parent.faces[old_flo:old_fhi].astype(np.int64)
            - old_vlo
            + new_vlo
        )
        vertex_offsets.append(new_vlo + old_vhi - old_vlo)
        face_offsets.append(face_offsets[-1] + old_fhi - old_flo)
    return {
        "selected_rows": selected_rows,
        "chart_vertex_offsets": np.asarray(vertex_offsets, np.int64),
        "sampled_vertex_pixel_indices": np.concatenate(pixels).astype(
            np.int64, copy=False
        ),
        "chart_face_offsets": np.asarray(face_offsets, np.int64),
        "faces": np.concatenate(faces).astype(np.int64, copy=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent_authority", type=Path, required=True)
    parser.add_argument("--expected_parent_content_sha256", required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite selected projective authority")

    parent = ProjectiveExactFaceSeamAuthority.load_npz(args.parent_authority)
    if parent.metadata.get("content_sha256") != args.expected_parent_content_sha256:
        raise ValueError("parent projective authority content hash differs")
    if (
        parent.metadata.get("paired_stride2_densification_diagnostic") is not True
        or parent.metadata.get("projective_correspondence_production_candidate")
        is not False
        or parent.metadata.get("final_model_neutral_map_topology_eligible")
        is not False
    ):
        raise ValueError("parent is not the sealed paired stride-2 control authority")

    selection = load_paired_stride2_diagnostic_alignment_selection(
        args.frozen_submap_plan,
        expected_plan_content_sha256=args.expected_plan_content_sha256,
        diagnostic_alignment_adapter_opt_in=True,
    )
    if len(selection.operational_submaps) != 1:
        raise ValueError("selected projective control requires one submap")
    selected_names = selection.ordered_names
    plan = ChartSubmapPlan.load_npz(args.frozen_submap_plan)
    topology = _subset_topology(parent, selected_names)
    selected_rows = topology.pop("selected_rows")
    common_valid = parent.common_valid[selected_rows]
    dense_points = parent.reference_points_world_dense[selected_rows]
    dense_normals = np.stack(
        [
            _dense_world_normals(
                dense_points[row],
                _common_valid_normal_stencil(common_valid[row]),
            )
            for row in range(len(selected_rows))
        ]
    )
    pointmaps = parent.metadata.get("source_selected_pointmap_inventory")
    if not isinstance(pointmaps, dict):
        raise ValueError("parent projective authority lacks pointmap inventory")
    selected_pointmaps = {name: pointmaps[name] for name in selected_names}
    metadata = dict(parent.metadata)
    for key in (
        "arrays_sha256",
        "content_sha256",
        "chart_count",
        "frozen_edge_count",
        "frozen_correspondence_count",
        "edge_with_any_correspondence_count",
        "edge_with_both_directions_count",
        "direction_formal_valid_count",
        "edge_formal_valid_count",
        "edge_formal_valid_sha256",
        "selected_chart_self_reprojection",
        "selected_chart_self_reprojection_all_p90_pass",
        "selected_chart_self_reprojection_global_p50_px",
        "selected_chart_self_reprojection_global_p90_px",
        "selected_chart_self_reprojection_global_maximum_px",
        "stride2_graph_summary",
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "parent_projective_authority_file_sha256": file_sha256(
                args.parent_authority
            ),
            "parent_projective_authority_content_sha256": (
                args.expected_parent_content_sha256
            ),
            "parent_projective_authority_arrays_sha256": parent.metadata[
                "arrays_sha256"
            ],
            "frozen_submap_plan_file_sha256": file_sha256(
                args.frozen_submap_plan
            ),
            "frozen_submap_plan_content_sha256": (
                args.expected_plan_content_sha256
            ),
            "selected_chart_names_in_order_sha256": plan.metadata[
                "selected_chart_names_in_order_sha256"
            ],
            "selected_topology_derived_by_exact_chart_block_subset": True,
            "source_selected_pointmap_inventory": selected_pointmaps,
            "source_selected_pointmap_inventory_sha256": canonical_json_sha256(
                selected_pointmaps
            ),
            "projective_correspondence_production_candidate": False,
            "final_model_neutral_map_topology_eligible": False,
            "formal_selector_handoff_eligible": True,
            "system_control_only": True,
            "promotion_eligible": False,
            "full_gate_or_exporter_consumption_eligible": False,
        }
    )
    config = ProjectiveSeamConfig(**parent.metadata["config"]).validated()
    authority = freeze_projective_exact_face_correspondences(
        chart_names=np.asarray(selected_names),
        common_valid=common_valid,
        chart_vertex_offsets=topology["chart_vertex_offsets"],
        sampled_vertex_pixel_indices=topology[
            "sampled_vertex_pixel_indices"
        ],
        chart_face_offsets=topology["chart_face_offsets"],
        faces=topology["faces"],
        reference_points_world=dense_points,
        reference_dense_normals_world=dense_normals,
        camera_to_world=parent.camera_to_world[selected_rows],
        focal_px=parent.focal_px[selected_rows],
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=config,
        metadata=metadata,
    )
    output_metadata = authority.save_npz(args.output)
    # Force a full semantic replay before reporting success.
    replay = ProjectiveExactFaceSeamAuthority.load_npz(args.output)
    report = {
        "output": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output),
        "content_sha256": output_metadata["content_sha256"],
        "arrays_sha256": output_metadata["arrays_sha256"],
        "chart_count": len(replay.chart_names),
        "edge_count": len(replay.edge_chart_indices),
        "formal_edge_count": int(np.sum(replay.edge_formal_valid)),
        "correspondence_count": len(replay.source_vertex_indices),
        "system_control_only": True,
        "promotion_eligible": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
