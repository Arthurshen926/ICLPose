"""Measure complete full-train 2DGS geometry on a frozen held-ray inventory.

This is deliberately an unequal-budget, post-hoc diagnostic.  The complete
2DGS may have consumed mapping images outside the chart source inventory, so
the result answers only whether the *map support* missing from a finite chart
atlas exists in the trained 2DGS.  It is never a promotion-eligible comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.full_submap_chart_geometry_gate import (
    FullSubmapGeometryGateConfig,
    StrictHeldRayInventory,
    _finite_summary,
    _view_metrics,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.official_2dgs_renderer import (
    Official2DGSSource,
    load_official_2dgs_source_from_ply,
    render_official_2dgs_rgb_depth,
)


def _depth_normals_world(
    depth: np.ndarray,
    valid: np.ndarray,
    focal_xy: np.ndarray,
    principal_xy: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    """Finite-difference normals from camera-z depth on the pixel-center grid."""

    depth = np.asarray(depth, np.float64)
    valid = np.asarray(valid, bool)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("depth and valid must be equal-size images")
    height, width = depth.shape
    fx, fy = np.asarray(focal_xy, np.float64).reshape(2)
    cx, cy = np.asarray(principal_xy, np.float64).reshape(2)
    if min(float(fx), float(fy)) <= 0.0:
        raise ValueError("focal lengths must be positive")
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    finite_depth = np.where(valid & np.isfinite(depth), depth, 0.0)
    points = np.stack(
        (
            (xx - cx) * finite_depth / fx,
            (yy - cy) * finite_depth / fy,
            finite_depth,
        ),
        axis=-1,
    )
    output = np.zeros((height, width, 3), np.float64)
    if height < 3 or width < 3:
        return output
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    camera_normal = np.cross(dx, dy)
    length = np.linalg.norm(camera_normal, axis=2)
    stencil_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & (length > 1e-10)
    )
    camera_normal /= np.maximum(length[..., None], 1e-15)
    rotation = np.asarray(camera_to_world, np.float64).reshape(4, 4)[:3, :3]
    world_normal = camera_normal @ rotation.T
    world_normal /= np.maximum(np.linalg.norm(world_normal, axis=2, keepdims=True), 1e-15)
    output[1:-1, 1:-1][stencil_valid] = world_normal[stencil_valid]
    return output


def _camera(focal_xy: np.ndarray, principal_xy: np.ndarray, width: int, height: int) -> ColmapCamera:
    fx, fy = np.asarray(focal_xy, np.float64).reshape(2)
    cx, cy = np.asarray(principal_xy, np.float64).reshape(2)
    return ColmapCamera(
        camera_id=1,
        model_id=1,  # PINHOLE: fx, fy, cx, cy
        width=int(width),
        height=int(height),
        params=(float(fx), float(fy), float(cx), float(cy)),
    )


def _planar_source_subset(
    source: Official2DGSSource,
    physical: GoalMapletPhysicalMap,
    planar: GeometryNativePlanarMap,
) -> tuple[Official2DGSSource, np.ndarray]:
    """Select original 2DGS rows owned by the finite planar map."""

    planar = planar.validated(physical.primitive_ids.size)
    member_rows = np.asarray(planar.member_primitive_rows, np.int64)
    source_rows = np.unique(np.asarray(physical.primitive_ids, np.int64)[member_rows])
    if source_rows.size == 0:
        raise ValueError("planar map owns no 2DGS primitive")
    if np.any((source_rows < 0) | (source_rows >= source.gaussian_count)):
        raise ValueError("planar primitive lineage is outside the original 2DGS PLY")
    return Official2DGSSource(
        xyz=source.xyz[source_rows],
        sh_features=source.sh_features[source_rows],
        opacity_logits=source.opacity_logits[source_rows],
        log_scales_2d=source.log_scales_2d[source_rows],
        rotations=source.rotations[source_rows],
        loc_features=None if source.loc_features is None else source.loc_features[source_rows],
        sh_degree=source.sh_degree,
        path=f"{source.path}#geometry_native_planar_subset",
    ), source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian_ply", type=Path, required=True)
    parser.add_argument("--held_rays", type=Path, required=True)
    parser.add_argument("--physical_map", type=Path)
    parser.add_argument("--planar_map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum_alpha", type=float, default=0.05)
    parser.add_argument(
        "--acknowledge_full_train_unequal_budget",
        action="store_true",
        help="Required acknowledgement that this control is not a fair atlas comparison.",
    )
    args = parser.parse_args()
    if not args.acknowledge_full_train_unequal_budget:
        raise ValueError("full-train 2DGS control requires explicit unequal-budget acknowledgement")
    if not 0.0 <= float(args.minimum_alpha) <= 1.0:
        raise ValueError("minimum_alpha must be in [0, 1]")
    if (args.physical_map is None) != (args.planar_map is None):
        raise ValueError("physical_map and planar_map must be supplied together")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite full-2DGS held diagnostic")

    started = time.perf_counter()
    held = StrictHeldRayInventory.load_npz(args.held_rays)
    full_source = load_official_2dgs_source_from_ply(args.gaussian_ply)
    source = full_source
    planar_source_rows = None
    planar_lineage: dict[str, object] = {}
    if args.physical_map is not None:
        physical = GoalMapletPhysicalMap.load_npz(args.physical_map)
        planar = GeometryNativePlanarMap.load_npz(
            args.planar_map, primitive_count=physical.primitive_ids.size
        )
        source, planar_source_rows = _planar_source_subset(full_source, physical, planar)
        planar_lineage = {
            "physical_map": str(args.physical_map.resolve()),
            "physical_map_file_sha256": file_sha256(args.physical_map),
            "planar_map": str(args.planar_map.resolve()),
            "planar_map_file_sha256": file_sha256(args.planar_map),
            "planar_map_content_sha256": planar.metadata.get("content_sha256"),
            "planar_plane_count": int(planar.plane_ids.size),
            "planar_member_primitive_count": int(planar.member_primitive_rows.size),
            "selected_original_2dgs_row_count": int(planar_source_rows.size),
        }
    config = FullSubmapGeometryGateConfig().validated()
    views, height, width = held.shape
    rows: list[dict[str, float]] = []
    alpha_recall: list[float] = []
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=False)
    for view in range(views):
        pose_w2c = np.linalg.inv(held.camera_to_world[view])
        _rgb, depth, alpha = render_official_2dgs_rgb_depth(
            source,
            pose_w2c=pose_w2c,
            camera=_camera(held.focal_xy[view], held.principal_xy[view], width, height),
            width=width,
            height=height,
            device=str(args.device),
        )
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & np.isfinite(alpha)
            & (alpha >= float(args.minimum_alpha))
        )
        render_depth = np.where(valid, np.asarray(depth, np.float64), np.inf)
        normal = _depth_normals_world(
            render_depth,
            valid,
            held.focal_xy[view],
            held.principal_xy[view],
            held.camera_to_world[view],
        )
        rows.append(
            _view_metrics(
                render_depth,
                normal,
                held.reference_depth_m[view],
                held.reference_normal_world[view],
                held.reference_valid[view],
                held.reference_boundary[view],
                config,
            )
        )
        reference_count = max(int(held.reference_valid[view].sum()), 1)
        alpha_recall.append(float(np.sum(valid & held.reference_valid[view]) / reference_count))
        if args.cache_dir is not None:
            np.savez_compressed(
                args.cache_dir / f"{str(held.view_names[view])}.npz",
                depth=np.asarray(depth, np.float32),
                alpha=np.asarray(alpha, np.float32),
                normal_world=np.asarray(normal, np.float32),
                valid=valid,
            )

    keys = tuple(rows[0])
    vectors = {key: np.asarray([row[key] for row in rows], np.float64) for key in keys}
    report = {
        "view_count": int(views),
        "macro_view": {
            key: _finite_summary(vectors[key])
            for key in keys
            if key != "reference_ray_count"
        },
        "alpha_rendered_ray_recall": _finite_summary(np.asarray(alpha_recall, np.float64)),
        "per_view": [
            {
                "name": str(held.view_names[index]),
                "block_id": str(held.block_ids[index]),
                "alpha_rendered_ray_recall": float(alpha_recall[index]),
                **{
                    key: (float(value) if np.isfinite(value) else None)
                    for key, value in rows[index].items()
                },
            }
            for index in range(views)
        ],
    }
    payload = {
        "artifact_type": (
            "goal_maplet_full_train_2dgs_planar_subset_held_geometry_unequal_budget_control_v1"
            if planar_source_rows is not None
            else "goal_maplet_full_train_2dgs_held_geometry_unequal_budget_control_v1"
        ),
        "gaussian_ply": str(args.gaussian_ply.resolve()),
        "gaussian_ply_file_sha256": file_sha256(args.gaussian_ply),
        "full_gaussian_count": int(full_source.gaussian_count),
        "rendered_gaussian_count": int(source.gaussian_count),
        **planar_lineage,
        "held_rays": str(args.held_rays.resolve()),
        "held_rays_file_sha256": file_sha256(args.held_rays),
        "minimum_alpha": float(args.minimum_alpha),
        "uses_held_mapping_pose_and_reference_geometry": True,
        "full_train_map_may_have_consumed_held_mapping_images": True,
        "unequal_mapping_budget_diagnostic_only": True,
        "promotion_eligible": False,
        "production_eligible": False,
        "claim": (
            "finite_planar_support_existence_diagnostic_not_representation_comparison"
            if planar_source_rows is not None
            else "map_support_existence_diagnostic_not_representation_comparison"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "report": report,
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": payload["content_sha256"],
        "full_gaussian_count": int(full_source.gaussian_count),
        "rendered_gaussian_count": int(source.gaussian_count),
        "runtime_seconds": payload["runtime_seconds"],
        "macro_view": report["macro_view"],
        "alpha_rendered_ray_recall": report["alpha_rendered_ray_recall"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
