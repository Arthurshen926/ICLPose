"""Render a MoGe/reference chart control on an explicitly historical held set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_projective_aligned_charts_control import _load_aligned_vertices
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    FullSubmapGeometryGateConfig,
    StrictHeldRayInventory,
    _SurfaceMesh,
    _render_and_measure,
)
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import ProjectiveExactFaceSeamAuthority


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    triangle = vertices[faces]
    face = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-15)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    return normals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected_authority_content_sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite held render control")
    authority = ProjectiveExactFaceSeamAuthority.load_npz(args.authority)
    if authority.metadata.get("content_sha256") != args.expected_authority_content_sha256:
        raise ValueError("held render authority differs from pin")
    if (
        authority.metadata.get("moge_reference_only_topology") is not True
        or authority.metadata.get("dav2_geometry_consumed") is not False
        or authority.metadata.get("post_alignment_common_quarantine_control") is not True
    ):
        raise ValueError("held render requires quarantined MoGe/reference topology")
    vertices = _load_aligned_vertices(args.alignment, authority)
    mesh = _SurfaceMesh(
        names=authority.chart_names,
        vertex_offsets=authority.chart_vertex_offsets,
        vertices=vertices,
        normals=_normals(vertices, authority.faces),
        face_offsets=authority.chart_face_offsets,
        faces=authority.faces,
    )
    rays = StrictHeldRayInventory.load_npz(args.held_rays)
    config = FullSubmapGeometryGateConfig().validated()
    report, _ = _render_and_measure(mesh, rays, config)
    macro = report["macro_view"]
    good = macro["good_ray_recall"]["mean"]
    joint = macro["joint_depth_normal_recall_20"]["mean"]
    payload = {
        "artifact_type": "goal_maplet_moge_reference_held_render_historical_control_v1",
        "authority_file_sha256": file_sha256(args.authority),
        "authority_content_sha256": args.expected_authority_content_sha256,
        "alignment_manifest_file_sha256": file_sha256(args.alignment / "manifest.json"),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "held_inventory_previously_opened": True,
        "blind_or_preregistered_claim": False,
        "uses_query_or_ground_truth": True,
        "pose_source": "held_mapping_camera_pose_for_renderer_only_not_localization_prediction",
        "production_eligible": False,
        "promotion_eligible": False,
        "report": report,
        "absolute_point_estimate": {
            "good_ray_recall": good,
            "joint_depth_normal_recall_20": joint,
            "minimum_good_ray_recall": config.minimum_absolute_good_ray_recall,
            "minimum_joint_depth_normal_recall_20": config.minimum_absolute_joint_depth_normal_recall_20,
            "decision": "GO" if good >= config.minimum_absolute_good_ray_recall and joint >= config.minimum_absolute_joint_depth_normal_recall_20 else "KILL",
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        **payload["absolute_point_estimate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
