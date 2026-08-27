"""Evaluate a strict planar side map with GT pixel-level 2DGS surface observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.oracle_pose import intersect_rays_with_primitive_planes
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    fit_child_planar_maplets,
    fit_weighted_plane,
    select_normal_diverse_planes,
    solve_metric_translation,
    solve_robust_metric_translation,
    solve_rotation_from_plane_normals,
    solve_scaled_translation,
    strict_planar_side_map_from_child_seeds,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import load_pose_candidate_dataset
from feature_extract.vfm.localization_goal_maplet.visibility import render_exact_maplet_visibility


SCHEMA = "goal_maplet_planar_pixel_surface_oracle_v2"
RENDER_WIDTH = 256
RENDER_HEIGHT = 144
MINIMUM_REGION_PIXELS = 20
MAXIMUM_PLANES = 16


def _rotation_error_deg(estimate_c2w: np.ndarray, pose_w2c: np.ndarray) -> float:
    relative = np.asarray(estimate_c2w).T @ np.asarray(pose_w2c)[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def _primitive_regions(physical: GoalMapletPhysicalMap, child_groups: list[np.ndarray]) -> np.ndarray:
    result = np.full((physical.primitive_ids.size,), -1, np.int64)
    for region, children in enumerate(child_groups):
        members = np.unique(np.concatenate([
            np.asarray(physical.child_member_primitive_rows[
                int(physical.child_member_offsets[child]):int(physical.child_member_offsets[child + 1])
            ], np.int64) for child in children.tolist()
        ]))
        if np.any(result[members] >= 0):
            raise ValueError("strict planar regions overlap in primitive membership")
        result[members] = int(region)
    return result


def _summary(rows: list[dict]) -> dict:
    total = len(rows)
    usable = [row for row in rows if row["solver_usable"]]
    def distribution(key: str) -> dict:
        values = np.asarray([row[key] for row in usable], np.float64)
        if values.size == 0:
            return {"count": 0}
        return {
            "count": int(values.size), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "maximum": float(np.max(values)),
        }
    def correlation(left: str, right: str) -> float | None:
        if len(usable) < 2:
            return None
        a = np.asarray([row[left] for row in usable], np.float64)
        b = np.asarray([row[right] for row in usable], np.float64)
        if float(np.std(a)) <= 1.0e-15 or float(np.std(b)) <= 1.0e-15:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    return {
        "query_count": total,
        "solver_usable_query_count": len(usable),
        "solver_usable_rate": len(usable) / max(total, 1),
        "metric_1m10_recall_all_queries": sum(
            row["solver_usable"] and row["translation_error_m"] <= 1.0
            and row["rotation_error_deg"] <= 10.0 for row in rows
        ) / max(total, 1),
        "metric_2m45_recall_all_queries": sum(
            row["solver_usable"] and row["translation_error_m"] <= 2.0
            and row["rotation_error_deg"] <= 45.0 for row in rows
        ) / max(total, 1),
        "scale_aware_1m10_recall_all_queries": sum(
            row["solver_usable"] and row["scale_translation_error_m"] <= 1.0
            and row["rotation_error_deg"] <= 10.0 for row in rows
        ) / max(total, 1),
        "rotation_error_deg": distribution("rotation_error_deg"),
        "translation_error_m": distribution("translation_error_m"),
        "scale_translation_error_m": distribution("scale_translation_error_m"),
        "normal_condition": distribution("normal_condition"),
        "scale_design_condition": distribution("scale_design_condition"),
        "absolute_plane_offset_error_median_m": distribution("absolute_plane_offset_error_median_m"),
        "absolute_plane_offset_error_p90_m": distribution("absolute_plane_offset_error_p90_m"),
        "selected_rendered_mass_fraction": distribution("selected_rendered_mass_fraction"),
        "selected_view_incidence_median": distribution("selected_view_incidence_median"),
        "selected_plane_distance_median_m": distribution("selected_plane_distance_median_m"),
        "translation_error_correlations": {
            "absolute_plane_offset_error_median": correlation(
                "absolute_plane_offset_error_median_m", "translation_error_m",
            ),
            "normal_condition": correlation("normal_condition", "translation_error_m"),
            "view_incidence_median": correlation(
                "selected_view_incidence_median", "translation_error_m",
            ),
            "plane_distance_median": correlation(
                "selected_plane_distance_median_m", "translation_error_m",
            ),
            "selected_rendered_mass_fraction": correlation(
                "selected_rendered_mass_fraction", "translation_error_m",
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum_queries", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite pixel plane oracle")
    physical_path = Path(args.physical_map).resolve()
    direct_path = Path(args.direct_dataset).resolve()
    camera_path = Path(args.camera_manifest).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    direct, direct_metadata = load_pose_candidate_dataset(direct_path, require_rendered_targets=False)
    image_ids = np.asarray(direct["image_ids"]).astype(str)
    poses = np.asarray(direct["candidate_poses_w2c"], np.float64)[:, 0]
    if int(args.maximum_queries) > 0:
        image_ids = image_ids[:int(args.maximum_queries)]
        poses = poses[:int(args.maximum_queries)]
    cameras = json.loads(camera_path.read_text())["cameras"]
    if any(image_id not in cameras for image_id in image_ids.tolist()):
        raise ValueError("camera manifest misses a query")

    child = fit_child_planar_maplets(physical)
    strict = strict_planar_side_map_from_child_seeds(physical, child)
    balanced = strict_planar_side_map_from_child_seeds(
        physical, child, distance_rms_m=.05, distance_p95_m=.10,
        normal_cosine_p10=float(np.cos(np.deg2rad(15.0))),
    )
    variants = {"strict_3cm_5cm_10deg": strict, "balanced_5cm_10cm_15deg": balanced}
    prepared = {}
    total_surface = float(np.sum(child.member_area_m2))
    for name, (side_map, parent_rows, groups, audit) in variants.items():
        prepared[name] = {
            "side_map": side_map, "parent_rows": parent_rows, "groups": groups,
            "audit": audit, "primitive_region": _primitive_regions(physical, groups),
            "weighted_surface_fraction": float(np.sum(side_map.member_area_m2) / total_surface),
            "rows": {key: [] for key in ("mass_wls", "diverse_gls", "diverse_robust_irls")},
        }
    primitive_by_id = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}

    for query_index, (image_id, pose) in enumerate(zip(image_ids.tolist(), poses)):
        value = cameras[image_id]
        camera = ColmapCamera(
            0, int(value["model_id"]), int(value["width"]), int(value["height"]),
            tuple(float(item) for item in value["params"]),
        )
        rendered = render_exact_maplet_visibility(
            physical, pose, camera, width=RENDER_WIDTH, height=RENDER_HEIGHT,
            device=str(args.device),
        )
        valid_pixel = rendered.dominant_primitive_ids.reshape(-1) >= 0
        flat_ids = rendered.dominant_primitive_ids.reshape(-1)[valid_pixel]
        primitive_rows = np.asarray([primitive_by_id[int(value)] for value in flat_ids], np.int64)
        flat_weight = rendered.dominant_weights.reshape(-1)[valid_pixel].astype(np.float64)
        yy, xx = np.indices((RENDER_HEIGHT, RENDER_WIDTH))
        xy = np.column_stack((xx.reshape(-1)[valid_pixel], yy.reshape(-1)[valid_pixel])).astype(np.float64)
        xy[:, 0] = (xy[:, 0] + .5) * camera.width / RENDER_WIDTH
        xy[:, 1] = (xy[:, 1] + .5) * camera.height / RENDER_HEIGHT
        world, ray_valid = intersect_rays_with_primitive_planes(
            xy, primitive_rows, physical, pose, camera,
        )
        primitive_rows, flat_weight, world = (
            primitive_rows[ray_valid], flat_weight[ray_valid], world[ray_valid]
        )
        center = camera_center_from_pose_w2c(pose)
        for name, item in prepared.items():
            side_map = item["side_map"]
            region = item["primitive_region"][primitive_rows]
            represented = region >= 0
            region_count = side_map.normals_world.shape[0]
            pixel_count = np.bincount(region[represented], minlength=region_count)
            region_mass = np.bincount(
                region[represented], weights=flat_weight[represented], minlength=region_count,
            )
            candidates = np.flatnonzero(pixel_count >= MINIMUM_REGION_PIXELS)
            map_normal, map_offset, query_normal, query_offset = [], [], [], []
            visible_world_normal, visible_world_offset = [], []
            query_rms, usable_region, masses = [], [], []
            for region_row in candidates.tolist():
                selected = region == region_row
                normal, offset, eigenvalue = fit_weighted_plane(
                    world[selected], np.maximum(flat_weight[selected], 1.0e-8),
                    side_map.normals_world[region_row],
                )
                map_normal.append(side_map.normals_world[region_row])
                map_offset.append(side_map.offsets_world[region_row])
                query_normal.append(pose[:3, :3] @ normal)
                query_offset.append(offset - float(normal @ center))
                visible_world_normal.append(normal)
                visible_world_offset.append(offset)
                query_rms.append(float(np.sqrt(max(eigenvalue[0], 0.0))))
                usable_region.append(region_row)
                masses.append(region_mass[region_row])
            map_normal = np.asarray(map_normal, np.float64).reshape(-1, 3)
            map_offset = np.asarray(map_offset, np.float64)
            query_normal = np.asarray(query_normal, np.float64).reshape(-1, 3)
            query_offset = np.asarray(query_offset, np.float64)
            visible_world_normal = np.asarray(visible_world_normal, np.float64).reshape(-1, 3)
            visible_world_offset = np.asarray(visible_world_offset, np.float64)
            query_rms = np.asarray(query_rms, np.float64)
            masses = np.asarray(masses, np.float64)
            usable_region = np.asarray(usable_region, np.int64)
            variance = (
                side_map.center_residual_rms_m[usable_region] ** 2
                + query_rms ** 2 + .01 ** 2
            ) if usable_region.size else np.zeros(0)
            inverse_variance = np.maximum(masses, 1.0e-8) / np.maximum(variance, 1.0e-8)
            mass_order = np.lexsort((usable_region, -masses))[:MAXIMUM_PLANES]
            diverse_order = select_normal_diverse_planes(
                map_normal, inverse_variance, MAXIMUM_PLANES,
            ) if usable_region.size else np.zeros(0, np.int64)
            for method, selected in (
                ("mass_wls", mass_order),
                ("diverse_gls", diverse_order),
                ("diverse_robust_irls", diverse_order),
            ):
                row = {
                    "query_index": query_index, "image_id": image_id,
                    "visible_plane_count": int(usable_region.size),
                    "selected_plane_count": int(selected.size),
                    "solver_usable": False,
                }
                if selected.size >= 4:
                    weight = masses[selected] if method == "mass_wls" else inverse_variance[selected]
                    rotation, _ = solve_rotation_from_plane_normals(
                        query_normal[selected], map_normal[selected], weight,
                    )
                    if method == "diverse_robust_irls":
                        estimated, rank, singular, residual = solve_robust_metric_translation(
                            map_normal[selected], map_offset[selected], query_offset[selected], weight,
                        )
                    else:
                        estimated, rank, singular = solve_metric_translation(
                            map_normal[selected], map_offset[selected], query_offset[selected], weight,
                        )
                        residual = map_offset[selected] - query_offset[selected] - map_normal[selected] @ estimated
                    scaled, scale, scale_rank, scale_singular = solve_scaled_translation(
                        map_normal[selected], map_offset[selected], query_offset[selected], weight,
                    )
                    # Plane rho values are origin-dependent and cannot be
                    # compared when normals differ.  Use the signed distance
                    # from the frozen map-plane center to the visible fitted
                    # plane, which is a coordinate-invariant geometric error.
                    absolute_offset = np.abs(
                        np.sum(
                            visible_world_normal[selected]
                            * side_map.centers_world[usable_region[selected]], axis=1,
                        ) - visible_world_offset[selected]
                    )
                    selected_center = side_map.centers_world[usable_region[selected]]
                    view = center[None] - selected_center
                    distance = np.linalg.norm(view, axis=1)
                    incidence = np.abs(np.sum(
                        map_normal[selected] * view / np.maximum(distance[:, None], 1.0e-12),
                        axis=1,
                    ))
                    row.update({
                        "solver_usable": bool(rank == 3), "normal_rank": int(rank),
                        "scale_rank": int(scale_rank),
                        "rotation_error_deg": _rotation_error_deg(rotation, pose),
                        "translation_error_m": float(np.linalg.norm(estimated - center)),
                        "scale_translation_error_m": float(np.linalg.norm(scaled - center)),
                        "estimated_scale": float(scale),
                        "normal_condition": float(singular[0] / max(singular[-1], 1.0e-15)),
                        "scale_design_condition": float(scale_singular[0] / max(scale_singular[-1], 1.0e-15)),
                        "absolute_plane_offset_error_median_m": float(np.median(absolute_offset)),
                        "absolute_plane_offset_error_p90_m": float(np.quantile(absolute_offset, .9)),
                        "solver_residual_rms_m": float(np.sqrt(np.mean(residual ** 2))),
                        "selected_rendered_mass_fraction": float(
                            np.sum(masses[selected]) / max(float(np.sum(flat_weight)), 1.0e-12)
                        ),
                        "selected_view_incidence_median": float(np.median(incidence)),
                        "selected_plane_distance_median_m": float(np.median(distance)),
                    })
                item["rows"][method].append(row)
        print(f"{query_index + 1}/{len(image_ids)} {image_id}", flush=True)

    result = {}
    for name, item in prepared.items():
        result[name] = {
            "side_map_audit": item["audit"],
            "weighted_surface_fraction": item["weighted_surface_fraction"],
            "methods": {
                method: {"summary": _summary(rows), "rows": rows}
                for method, rows in item["rows"].items()
            },
        }
    best = max(
        result[name]["methods"][method]["summary"]["metric_1m10_recall_all_queries"]
        for name in result for method in result[name]["methods"]
    )
    report = {
        "artifact_type": SCHEMA, "query_route": "seq10", "query_count": len(image_ids),
        "physical_map": str(physical_path), "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_content_sha256": physical.content_sha256,
        "direct_dataset": str(direct_path), "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": direct_metadata["content_sha256"],
        "camera_manifest": str(camera_path), "camera_manifest_file_sha256": file_sha256(camera_path),
        "source_files": {
            "evaluator": {"path": str(Path(__file__).resolve()), "file_sha256": file_sha256(Path(__file__).resolve())},
            "planar_core": {"path": str(Path(__file__).resolve().parents[2] / "vfm/localization_goal_maplet/planar_maplet_oracle.py"), "file_sha256": file_sha256(Path(__file__).resolve().parents[2] / "vfm/localization_goal_maplet/planar_maplet_oracle.py")},
            "visibility": {"path": str(Path(__file__).resolve().parents[2] / "vfm/localization_goal_maplet/visibility.py"), "file_sha256": file_sha256(Path(__file__).resolve().parents[2] / "vfm/localization_goal_maplet/visibility.py")},
        },
        "pixel_oracle_contract": {
            "render_width": RENDER_WIDTH, "render_height": RENDER_HEIGHT,
            "minimum_region_pixels": MINIMUM_REGION_PIXELS, "maximum_planes": MAXIMUM_PLANES,
            "surface_point": "GT camera ray intersected with dominant visible exact 2DGS primitive plane",
            "region_correspondence": "oracle strict planar side-map region ID",
            "uses_pixel_xyz": True, "uses_primitive_center_as_query_measurement": False,
        },
        "variants": result, "best_metric_1m10_recall": best,
        "feasibility_target_metric_1m10_recall": .80,
        "decision": "GO_TO_REAL_QUERY_PLANE_RECOVERY_GATE" if best >= .80 else "KILL_CURRENT_STRONG_PLANAR_ORACLE_BELOW_TARGET",
        "actual_query_plane_recovery_or_matching_evaluated": False,
        "production_eligible": False, "uses_alike": False, "uses_pnp": False,
        "uses_gaussian_retraining": False, "uses_pose_lattice": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
