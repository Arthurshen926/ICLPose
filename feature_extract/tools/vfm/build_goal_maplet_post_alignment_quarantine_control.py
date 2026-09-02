"""Build a non-promotable topology after removing folded chart quads.

The mask is the union of deterministic failures in every supplied alignment
arm.  A single-arm run can test whether a selected system sanitizes its own
map.  It remains diagnostic: evaluated aligned geometry defines the mask and
therefore cannot serve as independent integrity or held-data evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import (
    _load_aligned_vertices,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
)
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    ProjectiveSeamConfig,
    _common_valid_normal_stencil,
    _dense_world_normals,
    freeze_projective_exact_face_correspondences,
)


def _arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must have NAME=ALIGNMENT_DIR form")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--arm", action="append", type=_arm, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite quarantine authority")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("quarantine parent authority hash differs")
    plan = ChartSubmapPlan.load_npz(args.frozen_submap_plan)
    if plan.metadata.get("content_sha256") != args.expected_plan_content_sha256:
        raise ValueError("quarantine plan hash differs")
    if not np.array_equal(plan.chart_names.astype(str), authority.chart_names.astype(str)):
        raise ValueError("quarantine plan and authority chart order differ")
    arms = {name: _load_aligned_vertices(root, authority) for name, root in args.arm}
    if len(arms) != len(args.arm) or not arms:
        raise ValueError("quarantine requires at least one uniquely named arm")

    reference = authority.reference_vertices_world
    faces = authority.faces
    triangle0 = reference[faces]
    cross0 = np.cross(
        triangle0[:, 1] - triangle0[:, 0],
        triangle0[:, 2] - triangle0[:, 0],
    )
    area0 = np.linalg.norm(cross0, axis=1)
    if np.any(area0 <= 1e-12):
        raise ValueError("quarantine parent contains collapsed source faces")
    unsafe = np.zeros(len(faces), bool)
    arm_audit = {}
    for name, vertices in arms.items():
        triangle = vertices[faces]
        cross = np.cross(
            triangle[:, 1] - triangle[:, 0],
            triangle[:, 2] - triangle[:, 0],
        )
        area = np.linalg.norm(cross, axis=1)
        raw_ratio = area / area0
        global_scale = float(np.median(raw_ratio))
        ratio = raw_ratio / global_scale
        flip = np.sum(cross0 * cross, axis=1) <= 0
        failed = flip | (ratio < 0.25) | (ratio > 4.0) | ~np.isfinite(ratio)
        unsafe |= failed
        arm_audit[name] = {
            "alignment_manifest_file_sha256": file_sha256(
                dict(args.arm)[name] / "manifest.json"
            ),
            "face_flip_count": int(np.sum(flip)),
            "collapsed_or_expanded_count": int(
                np.sum((ratio < 0.25) | (ratio > 4.0))
            ),
            "unsafe_face_count": int(np.sum(failed)),
        }
    # Faces are emitted as two triangles per sampled image-grid quad.  Remove
    # the complete quad whenever either triangle is unsafe in any supplied arm.
    if len(unsafe) % 2 or any(int(value) % 2 for value in authority.chart_face_offsets):
        raise ValueError("quarantine topology does not preserve triangle pairs")
    unsafe_quad = unsafe.reshape(-1, 2).any(axis=1)
    keep_face = np.repeat(~unsafe_quad, 2)

    vertex_offsets = [0]
    face_offsets = [0]
    pixel_blocks = []
    face_blocks = []
    removed_by_chart = []
    for chart in range(len(authority.chart_names)):
        vlo, vhi = map(int, authority.chart_vertex_offsets[chart : chart + 2])
        flo, fhi = map(int, authority.chart_face_offsets[chart : chart + 2])
        chart_faces_old = faces[flo:fhi][keep_face[flo:fhi]]
        used_old = np.unique(chart_faces_old)
        if not len(chart_faces_old) or not len(used_old):
            raise ValueError("quarantine emptied a selected chart")
        lookup = np.full(vhi - vlo, -1, np.int64)
        new_vlo = vertex_offsets[-1]
        lookup[used_old - vlo] = np.arange(len(used_old), dtype=np.int64) + new_vlo
        remapped = lookup[chart_faces_old - vlo]
        if np.any(remapped < 0):
            raise AssertionError("quarantine face remap failed")
        pixel_blocks.append(authority.sampled_vertex_pixel_indices[used_old])
        face_blocks.append(remapped)
        vertex_offsets.append(new_vlo + len(used_old))
        face_offsets.append(face_offsets[-1] + len(remapped))
        removed_by_chart.append(int((fhi - flo) - len(remapped)))

    common_valid = authority.common_valid
    dense_points = authority.reference_points_world_dense
    dense_normals = np.stack(
        [
            _dense_world_normals(
                dense_points[row],
                _common_valid_normal_stencil(common_valid[row]),
            )
            for row in range(len(common_valid))
        ]
    )
    metadata = dict(authority.metadata)
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
    ):
        metadata.pop(key, None)
    metadata.update(
        {
            "parent_projective_authority_file_sha256": file_sha256(args.authority),
            "parent_projective_authority_content_sha256": args.expected_authority_content_sha256,
            "frozen_submap_plan_file_sha256": file_sha256(args.frozen_submap_plan),
            "frozen_submap_plan_content_sha256": args.expected_plan_content_sha256,
            "post_alignment_common_quarantine_control": True,
            "quarantine_is_independent_integrity_evidence": False,
            "quarantine_rule": "remove_whole_quad_if_either_triangle_flipped_or_area_ratio_outside_[0.25,4]_in_any_supplied_arm",
            "quarantine_arm_count": int(len(arms)),
            "quarantine_arm_audit": arm_audit,
            "quarantine_removed_face_count": int(np.sum(~keep_face)),
            "quarantine_removed_face_fraction": float(np.mean(~keep_face)),
            "quarantine_removed_face_count_by_chart": removed_by_chart,
            "system_control_only": True,
            "promotion_eligible": False,
            "full_gate_or_exporter_consumption_eligible": False,
        }
    )
    output = freeze_projective_exact_face_correspondences(
        chart_names=authority.chart_names,
        common_valid=common_valid,
        chart_vertex_offsets=np.asarray(vertex_offsets, np.int64),
        sampled_vertex_pixel_indices=np.concatenate(pixel_blocks),
        chart_face_offsets=np.asarray(face_offsets, np.int64),
        faces=np.concatenate(face_blocks),
        reference_points_world=dense_points,
        reference_dense_normals_world=dense_normals,
        camera_to_world=authority.camera_to_world,
        focal_px=authority.focal_px,
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=ProjectiveSeamConfig(**authority.metadata["config"]),
        metadata=metadata,
    )
    saved = output.save_npz(args.output)
    replay = ProjectiveExactFaceSeamAuthority.load_npz(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_sha256": file_sha256(args.output),
                "content_sha256": saved["content_sha256"],
                "arrays_sha256": saved["arrays_sha256"],
                "removed_face_count": int(np.sum(~keep_face)),
                "removed_face_fraction": float(np.mean(~keep_face)),
                "remaining_face_count": len(replay.faces),
                "formal_edge_count": int(np.sum(replay.edge_formal_valid)),
                "edge_count": len(replay.edge_formal_valid),
                "system_control_only": True,
                "promotion_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
