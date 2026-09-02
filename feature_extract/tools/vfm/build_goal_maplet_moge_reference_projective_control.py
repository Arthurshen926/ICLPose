"""Build a MoGe3 + source-reference-only stride-2 projective control.

The selected chart inventory may still come from a paired diagnostic selector,
so the output is explicitly non-promotable.  Within that frozen inventory,
however, DAV2 files, validity, and geometry are not consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_chart_comparison_domain import (
    _conservative_face_domain,
)
from feature_extract.vfm.localization_goal_maplet.chart_comparison_domain import (
    _one_stride_topology,
)
from feature_extract.vfm.localization_goal_maplet.chart_submap_selection import (
    ChartSubmapPlan,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.projective_source_seam_authority import (
    ProjectiveExactFaceSeamAuthority,
    ProjectiveSeamConfig,
    _common_valid_normal_stencil,
    _dense_world_normals,
    _load_source_reference_dense,
    freeze_projective_exact_face_correspondences,
)


def _metadata(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {
            name: np.asarray(data[name])
            for name in ("chart_names", "valid", "face_valid_stride4", "face_valid_stride8")
        }
        metadata = json.loads(str(data["metadata_json"].item()))
    replay = dict(metadata)
    claimed = replay.pop("content_sha256", None)
    if claimed != canonical_json_sha256(replay) or metadata.get("arrays_sha256") != arrays_sha256(arrays):
        raise ValueError("MoGe/reference optimizer domain hash differs")
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer_domain", type=Path, required=True)
    parser.add_argument("--expected_optimizer_domain_content_sha256", required=True)
    parser.add_argument("--frozen_submap_plan", type=Path, required=True)
    parser.add_argument("--expected_plan_content_sha256", required=True)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--moge3_initializer_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite MoGe/reference authority")

    base, domain = _metadata(args.optimizer_domain)
    if domain.get("content_sha256") != args.expected_optimizer_domain_content_sha256:
        raise ValueError("MoGe/reference optimizer domain differs from external pin")
    if domain.get("moge_reference_only_control") is not True or domain.get("dav2_geometry_consumed") is not False:
        raise ValueError("optimizer domain is not MoGe/reference-only")
    names = base["chart_names"].astype(str).tolist()
    plan = ChartSubmapPlan.load_npz(args.frozen_submap_plan)
    if plan.metadata.get("content_sha256") != args.expected_plan_content_sha256 or names != list(plan.selected_chart_names_in_order):
        raise ValueError("MoGe/reference plan binding differs")

    source_root = args.source_root.resolve()
    cameras_path = source_root / "cameras.json"
    if file_sha256(cameras_path) != domain.get("cameras_file_sha256"):
        raise ValueError("MoGe/reference cameras differ")
    cameras = json.loads(cameras_path.read_text())
    camera_rows = {Path(value).name: row for row, value in enumerate(cameras["filepaths"])}
    manifest_path = args.moge3_initializer_root / "manifest.json"
    if file_sha256(manifest_path) != domain.get("moge_initializer_manifest_file_sha256"):
        raise ValueError("MoGe/reference manifest differs")

    height, width = base["valid"].shape[1:]
    reference, moge, camera_to_world, focals, shapes = [], [], [], [], []
    pointmap_hashes, moge_hashes = {}, {}
    for name in names:
        pointmap = source_root / "pointmaps" / f"{Path(name).stem}.json"
        pointmap_hashes[name] = file_sha256(pointmap)
        if pointmap_hashes[name] != domain["pointmap_inventory"][name]:
            raise ValueError(f"source pointmap differs for {name}")
        points, shape = _load_source_reference_dense(pointmap, output_height=height, output_width=width)
        reference.append(points)
        shapes.append(shape)
        initializer = args.moge3_initializer_root / f"{name}.npz"
        moge_hashes[name] = file_sha256(initializer)
        if moge_hashes[name] != domain["moge_initializer_file_sha256"][name]:
            raise ValueError(f"MoGe initializer differs for {name}")
        with np.load(initializer, allow_pickle=False) as data:
            camera_points = np.asarray(data["points_camera"], np.float64)
            valid = np.asarray(data["valid"], bool)
        row = camera_rows[name]
        c2w = np.asarray(cameras["cams2world"][row], np.float64)
        world = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
        if np.any(base["valid"][len(moge)] & ~(valid & np.isfinite(world).all(2))):
            raise ValueError(f"optimizer valid exceeds MoGe validity for {name}")
        moge.append(world)
        camera_to_world.append(c2w)
        focals.append(float(cameras["focals"][row]) * width / shape[1])

    reference = np.stack(reference)
    moge = np.stack(moge)
    camera_to_world = np.stack(camera_to_world)
    centers = camera_to_world[:, :3, 3]
    face2 = _conservative_face_domain(base["valid"], (moge, reference), centers, 2)
    topology = _one_stride_topology(base["valid"], face2, stride=2)
    if np.any(np.diff(topology["sampled_vertex_offsets_stride2"]) <= 0) or np.any(np.diff(topology["face_offsets_stride2"]) <= 0):
        raise ValueError("MoGe/reference topology contains an empty chart")
    eligible = np.stack([_common_valid_normal_stencil(mask) for mask in base["valid"]])
    normals = np.stack([_dense_world_normals(reference[i], eligible[i]) for i in range(len(names))])
    metadata = {
        "optimizer_domain_file_sha256": file_sha256(args.optimizer_domain),
        "optimizer_domain_content_sha256": args.expected_optimizer_domain_content_sha256,
        "frozen_submap_plan_file_sha256": file_sha256(args.frozen_submap_plan),
        "frozen_submap_plan_content_sha256": args.expected_plan_content_sha256,
        "source_root": str(source_root),
        "source_tree_sha256": domain.get("source_tree_sha256"),
        "source_cameras_file_sha256": file_sha256(cameras_path),
        "source_selected_pointmap_inventory": pointmap_hashes,
        "moge3_initializer_manifest_file_sha256": file_sha256(manifest_path),
        "moge3_initializer_manifest_content_sha256": domain.get("moge_initializer_manifest_content_sha256"),
        "moge3_initializer_file_sha256": moge_hashes,
        "moge_reference_only_topology": True,
        "dav2_geometry_consumed": False,
        "paired_comparison_topology": False,
        "paired_initializer_geometry_encoded_upstream": False,
        "upstream_plan_paired_geometry_encoded": bool(
            plan.metadata.get("paired_stride2_densification_diagnostic", False)
        ),
        "model_neutral_topology_claimed": False,
        "projective_correspondence_production_candidate": False,
        "final_model_neutral_map_topology_eligible": False,
        "system_control_only": True,
        "promotion_eligible": False,
        "full_gate_or_exporter_consumption_eligible": False,
        "uses_query_or_ground_truth": False,
        "held_geometry_consumed": False,
        "aligned_arm_geometry_consumed": False,
        "full_submap_gate_primary_stride": 2,
        "topology_caveat": "MoGe3+MASt3R topology on an upstream paired-selected chart inventory",
        "source_eligible_denominator": int(eligible.sum()),
        "stride2_face_quad_count": int(face2.sum()),
        "stride2_triangle_count": int(len(topology["faces_stride2"])),
        "stride2_packed_vertex_count": int(len(topology["sampled_vertex_pixel_indices_stride2"])),
    }
    authority = freeze_projective_exact_face_correspondences(
        chart_names=base["chart_names"],
        common_valid=base["valid"],
        chart_vertex_offsets=topology["sampled_vertex_offsets_stride2"],
        sampled_vertex_pixel_indices=topology["sampled_vertex_pixel_indices_stride2"],
        chart_face_offsets=topology["face_offsets_stride2"],
        faces=topology["faces_stride2"],
        reference_points_world=reference,
        reference_dense_normals_world=normals,
        camera_to_world=camera_to_world,
        focal_px=np.asarray(focals),
        plan_chart_names=plan.chart_names,
        coverage_edges=plan.coverage_edges,
        symmetric_surface_overlap=plan.symmetric_surface_overlap,
        config=ProjectiveSeamConfig(topology_stride=2),
        metadata=metadata,
    )
    saved = authority.save_npz(args.output)
    ProjectiveExactFaceSeamAuthority.load_npz(args.output)
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": saved["content_sha256"],
        "chart_count": len(names),
        "face_count": len(authority.faces),
        "vertex_count": len(authority.sampled_vertex_pixel_indices),
        "edge_count": len(authority.edge_chart_indices),
        "formal_edge_count": int(authority.edge_formal_valid.sum()),
        "correspondence_count": len(authority.source_vertex_indices),
        "promotion_eligible": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
