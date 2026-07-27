"""G0 audit for rasterized primitive IDs and exact ray/disk geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.build_2dgs_surface_feature_field import (
    _clean_source_indices,
)
from feature_extract.tools.vfm.build_detector_weighted_2dgs_surface_field import (
    _camera_rays,
)
from feature_extract.tools.vfm.localize_2dgs_surface_queries import (
    _load_query_camera_manifest,
)
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFeatureView,
    load_gaussian_vfm_source_from_ply,
)
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    clean_primitive_surface_elements,
    render_primitive_contributors,
)
from feature_extract.vfm.tokens import TokenBankManifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", required=True)
    parser.add_argument("--clean_gaussian_ply", required=True)
    parser.add_argument("--mapping_manifest", required=True)
    parser.add_argument("--mapping_pose_file", required=True)
    parser.add_argument("--mapping_camera_manifest", required=True)
    parser.add_argument("--mapping_depth_bank", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--minimum_weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sample_depth(depth: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, depth.shape[1] - 1)
    y = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, depth.shape[0] - 1)
    return np.asarray(depth[y, x], dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError("refusing to overwrite contributor audit")
    source = load_gaussian_vfm_source_from_ply(Path(args.gaussian_ply))
    clean = _clean_source_indices(Path(args.clean_gaussian_ply))
    # Contributor identity and occlusion must use the same declared clean 2DGS
    # prior as canonical atlas geometry.  The full PLY supplies stable source
    # indices; removed primitives must not re-enter as invisible occluders.
    elements = clean_primitive_surface_elements(
        source, clean
    )
    clean_lookup = np.zeros((source.xyz.shape[0],), dtype=bool)
    clean_lookup[clean] = True
    manifest = TokenBankManifest.from_json(Path(args.mapping_manifest))
    pose_by_image = {
        record.image_id: record.pose_w2c
        for record in parse_cambridge_pose_file(Path(args.mapping_pose_file))
    }
    camera_by_image, camera_audit = _load_query_camera_manifest(
        Path(args.mapping_camera_manifest)
    )
    depth_payload = json.loads(Path(args.mapping_depth_bank).read_text())
    depth_by_image = {
        str(record["image_id"]): Path(record["path"])
        for record in depth_payload["records"]
    }
    records = [
        record
        for record in manifest.records
        if record.image_id in pose_by_image
        and record.image_id in camera_by_image
        and record.image_id in depth_by_image
    ]
    if len(records) > int(args.max_views) > 0:
        indices = np.linspace(
            0, len(records) - 1, int(args.max_views), dtype=np.int64
        )
        records = [records[int(index)] for index in indices]
    depth_residuals: list[np.ndarray] = []
    reprojection_residuals: list[np.ndarray] = []
    footprint_radius: list[np.ndarray] = []
    accepted_counts = []
    contributor_counts = []
    for record in records:
        camera = camera_by_image[record.image_id]
        pose = pose_by_image[record.image_id]
        view = GaussianVFMFeatureView(
            image_id=record.image_id,
            feature_map=np.zeros(
                (1, int(args.height), int(args.width)), dtype=np.float32
            ),
            pose_w2c=pose,
            camera=camera,
        )
        contributors = render_primitive_contributors(
            elements,
            view,
            width=int(args.width),
            height=int(args.height),
            top_k=4,
            device=str(args.device),
        )
        mask = (
            (contributors.dominant_ids >= 0)
            & (contributors.dominant_weights >= float(args.minimum_weight))
        )
        mask &= clean_lookup[np.maximum(contributors.dominant_ids, 0)]
        yy, xx = np.nonzero(mask)
        contributor_counts.append(int(yy.size))
        ids = contributors.dominant_ids[yy, xx]
        image_xy = np.stack(
            [
                (xx.astype(np.float64) + 0.5) * camera.width / int(args.width)
                - 0.5,
                (yy.astype(np.float64) + 0.5) * camera.height / int(args.height)
                - 0.5,
            ],
            axis=1,
        )
        origin, rays = _camera_rays(image_xy, camera=camera, pose_w2c=pose)
        centers = np.asarray(source.xyz, dtype=np.float64)[ids]
        normals = np.asarray(source.normal, dtype=np.float64)[ids]
        denominator = np.sum(normals * rays, axis=1)
        ray_t = np.sum(normals * (centers - origin[None]), axis=1) / np.where(
            np.abs(denominator) > 1e-8,
            denominator,
            np.where(denominator < 0.0, -1e-8, 1e-8),
        )
        intersection = origin[None] + ray_t[:, None] * rays
        projected, camera_depth = project_world_points(intersection, pose, camera)
        depth_map = np.load(depth_by_image[record.image_id], mmap_mode="r")
        rendered_depth = _sample_depth(depth_map, image_xy)
        local_rows = np.searchsorted(elements.element_ids, ids)
        delta = intersection - centers
        local_u = np.sum(delta * elements.tangent1[local_rows], axis=1) / np.maximum(
            elements.scale1[local_rows], 1e-6
        )
        local_v = np.sum(delta * elements.tangent2[local_rows], axis=1) / np.maximum(
            elements.scale2[local_rows], 1e-6
        )
        radius = np.sqrt(local_u * local_u + local_v * local_v)
        exact_valid = (
            np.isfinite(rendered_depth)
            & np.isfinite(camera_depth)
            & np.isfinite(radius)
            & (ray_t > 0.0)
            & (radius <= 3.0)
        )
        depth_residuals.append(
            np.abs(camera_depth[exact_valid] - rendered_depth[exact_valid])
        )
        reprojection_residuals.append(
            np.linalg.norm(projected[exact_valid] - image_xy[exact_valid], axis=1)
        )
        footprint_radius.append(radius[exact_valid])
        accepted_counts.append(int(np.sum(exact_valid)))
    depth_values = np.concatenate(depth_residuals) if depth_residuals else np.zeros(0)
    reprojection_values = (
        np.concatenate(reprojection_residuals)
        if reprojection_residuals
        else np.zeros(0)
    )
    footprint_values = (
        np.concatenate(footprint_radius) if footprint_radius else np.zeros(0)
    )
    report = {
        "stage": "v6_g0_raster_primitive_contributor_audit",
        "view_count": len(records),
        "sample_count": int(depth_values.size),
        "raster_contributor_per_view": contributor_counts,
        "accepted_per_view": accepted_counts,
        "depth_residual_m": {
            "median": float(np.median(depth_values)),
            "p90": float(np.quantile(depth_values, 0.90)),
            "role": "diagnostic_expected_depth_blend_difference_not_gate",
        },
        "canonical_reprojection_rms_px": float(
            np.sqrt(np.mean(reprojection_values**2))
        ),
        "disk_sigma_radius": {
            "median": float(np.median(footprint_values)),
            "p99": float(np.quantile(footprint_values, 0.99)),
        },
        "camera_audit": camera_audit,
        "production_assignment": "gsplat_topk_contributor_source_index",
        "uses_kdtree_fallback": False,
        "g0_pass": bool(
            reprojection_values.size
            and np.sqrt(np.mean(reprojection_values**2)) < 0.5
            and np.quantile(footprint_values, 0.99) <= 3.0
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
