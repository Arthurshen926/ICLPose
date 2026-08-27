"""Evaluate MoGe-2 query geometry with GT plane masks and correspondences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    fit_child_planar_maplets, fit_weighted_plane, select_normal_diverse_planes,
    solve_metric_translation, solve_robust_metric_translation,
    solve_rotation_from_plane_normals, solve_scaled_translation,
    strict_planar_side_map_from_child_seeds,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import load_pose_candidate_dataset
from feature_extract.vfm.localization_goal_maplet.visibility import render_exact_maplet_visibility


SCHEMA = "goal_maplet_moge_gt_mask_planar_oracle_v2"
MINIMUM_REGION_PIXELS = 20
MAXIMUM_PLANES = 16


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _regions(physical: GoalMapletPhysicalMap, groups: list[np.ndarray]) -> np.ndarray:
    result = np.full(physical.primitive_ids.size, -1, np.int64)
    for region, children in enumerate(groups):
        members = np.unique(np.concatenate([
            physical.child_member_primitive_rows[
                int(physical.child_member_offsets[child]):int(physical.child_member_offsets[child + 1])
            ] for child in children.tolist()
        ]))
        if np.any(result[members] >= 0):
            raise ValueError("planar regions overlap")
        result[members] = region
    return result


def _rotation_error(rotation_c2w: np.ndarray, pose: np.ndarray) -> float:
    relative = rotation_c2w.T @ pose[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))


def _summarize(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["usable"]]
    def dist(key: str) -> dict:
        value = np.asarray([row[key] for row in usable], np.float64)
        return {"count": int(value.size)} if not value.size else {
            "count": int(value.size), "mean": float(np.mean(value)),
            "median": float(np.median(value)), "p90": float(np.quantile(value, .9)),
            "maximum": float(np.max(value)),
        }
    count = max(len(rows), 1)
    return {
        "query_count": len(rows), "usable_count": len(usable), "usable_rate": len(usable) / count,
        "metric_1m10_recall": sum(row["usable"] and row["translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0 for row in rows) / count,
        "metric_2m45_recall": sum(row["usable"] and row["translation_error_m"] <= 2.0 and row["rotation_error_deg"] <= 45.0 for row in rows) / count,
        "scale_aware_1m10_recall": sum(row["usable"] and row["scale_translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0 for row in rows) / count,
        "rotation_error_deg": dist("rotation_error_deg"),
        "translation_error_m": dist("translation_error_m"),
        "scale_translation_error_m": dist("scale_translation_error_m"),
        "query_normal_error_deg_median": dist("query_normal_error_deg_median"),
        "query_offset_error_median_m": dist("query_offset_error_median_m"),
        "normal_condition": dist("normal_condition"),
        "scale_condition": dist("scale_condition"),
        "selected_mask_mass_fraction": dist("selected_mask_mass_fraction"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--camera_manifest", required=True)
    parser.add_argument("--moge_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plane_parameter_source", choices=("point_pca", "normal_head_robust_offset"), default="point_pca")
    parser.add_argument("--include_oracle_parameter_ablations", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite MoGe plane evaluation")
    physical_path, direct_path = Path(args.physical_map).resolve(), Path(args.direct_dataset).resolve()
    camera_path, moge_path = Path(args.camera_manifest).resolve(), Path(args.moge_manifest).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    direct, direct_meta = load_pose_candidate_dataset(direct_path, require_rendered_targets=False)
    image_ids = np.asarray(direct["image_ids"]).astype(str)
    poses = np.asarray(direct["candidate_poses_w2c"], np.float64)[:, 0]
    cameras = json.loads(camera_path.read_text())["cameras"]
    moge = json.loads(moge_path.read_text())
    if moge.get("uses_pose") or moge.get("uses_ground_truth") or int(moge["query_count"]) != len(image_ids):
        raise ValueError("MoGe manifest is not complete pose-free query geometry")
    moge_rows = {row["image_id"]: row for row in moge["rows"]}
    if set(moge_rows) != set(image_ids.tolist()):
        raise ValueError("MoGe and direct query inventories differ")

    child = fit_child_planar_maplets(physical)
    variants = {
        "strict": strict_planar_side_map_from_child_seeds(physical, child),
        "balanced": strict_planar_side_map_from_child_seeds(
            physical, child, distance_rms_m=.05, distance_p95_m=.10,
            normal_cosine_p10=float(np.cos(np.deg2rad(15.0))),
        ),
    }
    prepared = {}
    for name, (side, parents, groups, audit) in variants.items():
        prepared[name] = {
            "side": side, "groups": groups, "audit": audit,
            "primitive_region": _regions(physical, groups),
            "rows": {
                ablation: {"mass_wls": [], "diverse_gls": [], "diverse_robust_irls": []}
                for ablation in (
                    ("predicted", "oracle_normal", "oracle_offset", "oracle_both")
                    if args.include_oracle_parameter_ablations else ("predicted",)
                )
            },
        }
    primitive_by_id = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}

    for query_index, (image_id, pose) in enumerate(zip(image_ids.tolist(), poses)):
        geometry_path = Path(moge_rows[image_id]["path"])
        if _sha(geometry_path) != moge_rows[image_id]["file_sha256"]:
            raise ValueError("MoGe geometry file hash differs")
        with np.load(geometry_path, allow_pickle=False) as data:
            points_camera = np.asarray(data["points_camera"], np.float64)
            predicted_normal = np.asarray(data["normal_camera"], np.float64)
            predicted_valid = np.asarray(data["valid"], bool)
        height, width = points_camera.shape[:2]
        camera_value = cameras[image_id]
        camera = ColmapCamera(0, int(camera_value["model_id"]), int(camera_value["width"]), int(camera_value["height"]), tuple(camera_value["params"]))
        visibility = render_exact_maplet_visibility(
            physical, pose, camera, width=width, height=height, device=args.device,
        )
        primitive_id = visibility.dominant_primitive_ids.reshape(-1)
        mask = (primitive_id >= 0) & predicted_valid.reshape(-1) & np.isfinite(points_camera.reshape(-1, 3)).all(axis=1)
        primitive_row = np.full(primitive_id.shape, -1, np.int64)
        primitive_row[mask] = np.asarray([primitive_by_id[int(value)] for value in primitive_id[mask]], np.int64)
        point = points_camera.reshape(-1, 3)
        pixel_weight = visibility.dominant_weights.reshape(-1).astype(np.float64)
        center = camera_center_from_pose_w2c(pose)
        for name, item in prepared.items():
            side = item["side"]
            region = np.full(primitive_row.shape, -1, np.int64)
            region[mask] = item["primitive_region"][primitive_row[mask]]
            represented = region >= 0
            count = np.bincount(region[represented], minlength=len(item["groups"]))
            mass = np.bincount(region[represented], weights=pixel_weight[represented], minlength=len(item["groups"]))
            candidates = np.flatnonzero(count >= MINIMUM_REGION_PIXELS)
            map_normal, map_offset, query_normal, query_offset, ideal_query_normal, ideal_query_offset = [], [], [], [], [], []
            query_rms, region_rows, masses, normal_error, offset_error = [], [], [], [], []
            for region_row in candidates.tolist():
                selected = region == region_row
                reference = pose[:3, :3] @ side.normals_world[region_row]
                selected_weight = np.maximum(pixel_weight[selected], 1.0e-8)
                if args.plane_parameter_source == "point_pca":
                    normal, offset, eigen = fit_weighted_plane(
                        point[selected], selected_weight, reference,
                    )
                    plane_rms = float(np.sqrt(max(eigen[0], 0.0)))
                else:
                    normals = predicted_normal.reshape(-1, 3)[selected]
                    normals = normals * np.where(
                        normals @ reference >= 0.0, 1.0, -1.0,
                    )[:, None]
                    normal = np.sum(normals * selected_weight[:, None], axis=0)
                    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
                    signed = point[selected] @ normal
                    order = np.argsort(signed, kind="mergesort")
                    cumulative = np.cumsum(selected_weight[order])
                    offset = float(signed[order[np.searchsorted(cumulative, cumulative[-1] * .5)]])
                    plane_rms = float(np.sqrt(np.average((signed - offset) ** 2, weights=selected_weight)))
                ideal_offset = side.offsets_world[region_row] - float(side.normals_world[region_row] @ center)
                map_normal.append(side.normals_world[region_row]); map_offset.append(side.offsets_world[region_row])
                query_normal.append(normal); query_offset.append(offset)
                ideal_query_normal.append(reference); ideal_query_offset.append(ideal_offset)
                query_rms.append(plane_rms)
                region_rows.append(region_row); masses.append(mass[region_row])
                normal_error.append(float(np.degrees(np.arccos(np.clip(abs(float(normal @ reference)), -1.0, 1.0)))))
                offset_error.append(abs(float(offset - ideal_offset)))
            map_normal = np.asarray(map_normal, np.float64).reshape(-1, 3)
            map_offset, query_offset = np.asarray(map_offset), np.asarray(query_offset)
            query_normal = np.asarray(query_normal, np.float64).reshape(-1, 3)
            ideal_query_normal = np.asarray(ideal_query_normal, np.float64).reshape(-1, 3)
            ideal_query_offset = np.asarray(ideal_query_offset, np.float64)
            query_rms, masses = np.asarray(query_rms), np.asarray(masses)
            region_rows = np.asarray(region_rows, np.int64)
            normal_error, offset_error = np.asarray(normal_error), np.asarray(offset_error)
            variance = side.center_residual_rms_m[region_rows] ** 2 + query_rms ** 2 + .01 ** 2 if region_rows.size else np.zeros(0)
            inverse = np.maximum(masses, 1e-8) / np.maximum(variance, 1e-8)
            mass_order = np.lexsort((region_rows, -masses))[:MAXIMUM_PLANES]
            diverse = select_normal_diverse_planes(map_normal, inverse, MAXIMUM_PLANES) if region_rows.size else np.zeros(0, np.int64)
            ablation_parameters = {"predicted": (query_normal, query_offset)}
            if args.include_oracle_parameter_ablations:
                ablation_parameters.update({
                    "oracle_normal": (ideal_query_normal, query_offset),
                    "oracle_offset": (query_normal, ideal_query_offset),
                    "oracle_both": (ideal_query_normal, ideal_query_offset),
                })
            for ablation, (used_normal, used_offset) in ablation_parameters.items():
                for method, selected in (("mass_wls", mass_order), ("diverse_gls", diverse), ("diverse_robust_irls", diverse)):
                    row = {"query_index": query_index, "image_id": image_id, "visible_plane_count": int(region_rows.size), "selected_plane_count": int(selected.size), "usable": False}
                    if selected.size >= 4:
                        weights = masses[selected] if method == "mass_wls" else inverse[selected]
                        rotation, _ = solve_rotation_from_plane_normals(used_normal[selected], map_normal[selected], weights)
                        if method == "diverse_robust_irls":
                            estimate, rank, singular, _ = solve_robust_metric_translation(map_normal[selected], map_offset[selected], used_offset[selected], weights)
                        else:
                            estimate, rank, singular = solve_metric_translation(map_normal[selected], map_offset[selected], used_offset[selected], weights)
                        scaled, scale, scale_rank, scale_singular = solve_scaled_translation(map_normal[selected], map_offset[selected], used_offset[selected], weights)
                        used_normal_error = normal_error[selected] if ablation in ("predicted", "oracle_offset") else np.zeros(selected.size)
                        used_offset_error = offset_error[selected] if ablation in ("predicted", "oracle_normal") else np.zeros(selected.size)
                        row.update({
                            "usable": rank == 3, "rotation_error_deg": _rotation_error(rotation, pose),
                            "translation_error_m": float(np.linalg.norm(estimate - center)),
                            "scale_translation_error_m": float(np.linalg.norm(scaled - center)),
                            "estimated_scale": float(scale),
                            "normal_condition": float(singular[0] / max(singular[-1], 1e-15)),
                            "scale_condition": float(scale_singular[0] / max(scale_singular[-1], 1e-15)),
                            "query_normal_error_deg_median": float(np.median(used_normal_error)),
                            "query_offset_error_median_m": float(np.median(used_offset_error)),
                            "selected_mask_mass_fraction": float(np.sum(masses[selected]) / max(float(np.sum(pixel_weight)), 1e-12)),
                        })
                    item["rows"][ablation][method].append(row)
        print(f"{query_index + 1}/{len(image_ids)} {image_id}", flush=True)

    result = {
        name: {
            "side_map_audit": item["audit"],
            "ablations": {
                ablation: {"methods": {method: {"summary": _summarize(rows), "rows": rows} for method, rows in methods.items()}}
                for ablation, methods in item["rows"].items()
            },
        }
        for name, item in prepared.items()
    }
    best = max(
        result[name]["ablations"]["predicted"]["methods"][method]["summary"]["metric_1m10_recall"]
        for name in result for method in result[name]["ablations"]["predicted"]["methods"]
    )
    report = {
        "artifact_type": SCHEMA, "query_count": len(image_ids), "query_route": "seq10",
        "physical_map": str(physical_path), "physical_map_file_sha256": _sha(physical_path),
        "direct_dataset": str(direct_path), "direct_dataset_file_sha256": _sha(direct_path), "direct_dataset_content_sha256": direct_meta["content_sha256"],
        "camera_manifest": str(camera_path), "camera_manifest_file_sha256": _sha(camera_path),
        "moge_manifest": str(moge_path), "moge_manifest_file_sha256": _sha(moge_path), "moge_manifest_content_sha256": moge["content_sha256"],
        "evaluation_contract": {"query_geometry": f"pose-free {moge['model_id']} metric point map", "plane_parameter_source": args.plane_parameter_source, "plane_mask": "GT 2DGS visibility oracle", "region_correspondence": "GT planar side-map oracle", "maximum_planes": MAXIMUM_PLANES, "minimum_region_pixels": MINIMUM_REGION_PIXELS, "oracle_parameter_ablations": args.include_oracle_parameter_ablations},
        "variants": result, "best_metric_1m10_recall": best, "target": .80,
        "decision": "GO_TO_REAL_PLANE_MASK_AND_MATCHING_GATE" if best >= .80 else "KILL_ZERO_SHOT_MOGE_QUERY_GEOMETRY_FOR_PLANAR_POSE",
        "uses_alike": False, "uses_pnp": False, "uses_pose_lattice": False,
        "production_eligible": False,
    }
    report["content_sha256"] = _canonical(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decision": report["decision"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
