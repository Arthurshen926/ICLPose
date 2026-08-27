"""Evaluate PlanaReLoc-style oracle plane pose on the seq10 development route."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.localization_goal_maplet.lineage import canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.physical_map import GoalMapletPhysicalMap
from feature_extract.vfm.localization_goal_maplet.planar_maplet_oracle import (
    MERGE_BOUNDARY_GAP_M, MERGE_NORMAL_COSINE, MERGE_PLANE_DISTANCE_M,
    MINIMUM_BOUNDARY_AREA_M2, PLANARITY_DISTANCE_RMS_M,
    fit_child_planar_maplets, fit_weighted_primitive_plane,
    merge_coplanar_child_regions, solve_metric_translation,
    solve_rotation_from_plane_normals, solve_scaled_translation,
)
from feature_extract.vfm.localization_goal_maplet.pose_candidate_dataset import load_pose_candidate_dataset


SCHEMA = "goal_maplet_planar_pose_oracle_seq10_v1"
MAXIMUM_VISIBLE_REGIONS = 16
FEASIBILITY_TARGET_1M10 = 0.80


def _rotation_error_deg(estimate_c2w: np.ndarray, target_w2c: np.ndarray) -> float:
    estimate_c2w = np.asarray(estimate_c2w, np.float64)
    target_w2c = np.asarray(target_w2c, np.float64)
    if estimate_c2w.shape != (3, 3) or target_w2c.shape not in ((3, 4), (4, 4)):
        raise ValueError("rotation error requires 3x3 c2w and 3x4/4x4 w2c")
    target_c2w = target_w2c[:3, :3].T
    relative = estimate_c2w.T @ target_c2w
    value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _summary(rows: list[dict]) -> dict:
    def quantile(key: str):
        value = np.asarray([row[key] for row in rows], np.float64)
        return {
            "mean": float(np.mean(value)), "minimum": float(np.min(value)),
            "p10": float(np.quantile(value, .1)), "median": float(np.median(value)),
            "p90": float(np.quantile(value, .9)), "maximum": float(np.max(value)),
        }
    return {
        "query_count": len(rows),
        "normal_rank3_rate": float(np.mean([row["normal_rank"] >= 3 for row in rows])),
        "scale_design_rank4_rate": float(np.mean([row["scale_design_rank"] >= 4 for row in rows])),
        "metric_1m10_recall": float(np.mean([
            row["metric_translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0
            for row in rows
        ])),
        "scale_aware_1m10_recall": float(np.mean([
            row["scale_translation_error_m"] <= 1.0 and row["rotation_error_deg"] <= 10.0
            for row in rows
        ])),
        "rotation_error_deg": quantile("rotation_error_deg"),
        "metric_translation_error_m": quantile("metric_translation_error_m"),
        "scale_translation_error_m": quantile("scale_translation_error_m"),
        "estimated_scale_absolute_error": quantile("estimated_scale_absolute_error"),
        "selected_visible_mass_fraction": quantile("selected_visible_mass_fraction"),
        "map_normal_condition": quantile("map_normal_condition"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_map", required=True)
    parser.add_argument("--direct_dataset", required=True)
    parser.add_argument("--contributor_glob", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite planar oracle report")
    physical_path = Path(args.physical_map).resolve()
    direct_path = Path(args.direct_dataset).resolve()
    physical = GoalMapletPhysicalMap.load_npz(physical_path)
    direct, direct_metadata = load_pose_candidate_dataset(
        direct_path, require_rendered_targets=False,
    )
    image_ids = np.asarray(direct["image_ids"]).astype(str).tolist()
    poses = np.asarray(direct["candidate_poses_w2c"], np.float64)[:, 0]
    contributor_paths = [Path(value).resolve() for value in glob.glob(args.contributor_glob)]
    contributor_by_image = {
        path.stem.replace("__", "/", 1): path for path in contributor_paths
    }
    if set(contributor_by_image) != set(image_ids) or len(contributor_by_image) != len(image_ids):
        raise ValueError("seq10 contributor inventory differs")

    child = fit_child_planar_maplets(physical)
    parent_merged, parent_ids, parent_groups = merge_coplanar_child_regions(
        physical, child, restrict_to_parent=True,
    )
    global_merged, global_parent_ids, global_groups = merge_coplanar_child_regions(
        physical, child, restrict_to_parent=False,
    )
    atomic_seed = (
        (child.boundary_area_m2 >= MINIMUM_BOUNDARY_AREA_M2)
        & (child.center_residual_rms_m <= PLANARITY_DISTANCE_RMS_M)
        & (child.normal_cosine_p10 >= .90)
    )
    atomic_groups = [np.asarray([row], np.int64) for row in np.flatnonzero(atomic_seed)]
    atomic_rows = np.flatnonzero(atomic_seed)
    variants = {
        "atomic_child": (child, atomic_groups, atomic_rows),
        "parent_restricted_bounded_merge": (parent_merged, parent_groups, np.arange(len(parent_groups))),
        "cross_parent_bounded_merge": (global_merged, global_groups, np.arange(len(global_groups))),
    }
    primitive_id_to_row = {int(value): row for row, value in enumerate(physical.primitive_ids.tolist())}
    total_surface_mass = float(np.sum(child.member_area_m2))
    result = {}
    for variant_name, (planes, child_groups, plane_rows) in variants.items():
        eligible = (
            (planes.boundary_area_m2 >= MINIMUM_BOUNDARY_AREA_M2)
            & (planes.center_residual_rms_m <= PLANARITY_DISTANCE_RMS_M)
            & (planes.normal_cosine_p10 >= .90)
        )
        if variant_name == "atomic_child":
            eligible = np.ones(len(child_groups), dtype=bool)
            plane_normal = planes.normals_world[plane_rows]
            plane_offset = planes.offsets_world[plane_rows]
            plane_area = planes.member_area_m2[plane_rows]
        else:
            plane_normal, plane_offset, plane_area = (
                planes.normals_world, planes.offsets_world, planes.member_area_m2,
            )
        region_members = []
        primitive_to_region = np.full((physical.primitive_ids.size,), -1, np.int64)
        for region, children in enumerate(child_groups):
            members = np.unique(np.concatenate([
                np.asarray(physical.child_member_primitive_rows[
                    int(physical.child_member_offsets[child_row]):
                    int(physical.child_member_offsets[child_row + 1])
                ], np.int64)
                for child_row in children.tolist()
            ]))
            region_members.append(members)
            primitive_to_region[members] = region
        ideal_rows, visible_rows = [], []
        for query_index, (image_id, target_w2c) in enumerate(zip(image_ids, poses)):
            path = contributor_by_image[image_id]
            with np.load(path, allow_pickle=False) as data:
                primitive_id = np.asarray(data["element_ids"], np.int64)
                contribution = np.asarray(data["element_weights"], np.float64)
            primitive_row = np.asarray([
                primitive_id_to_row.get(int(value), -1) for value in primitive_id
            ], np.int64)
            valid = primitive_row >= 0
            primitive_mass = np.bincount(
                primitive_row[valid], weights=contribution[valid],
                minlength=physical.primitive_ids.size,
            )
            visible_primitive = np.flatnonzero(primitive_mass > 0.0)
            region = primitive_to_region[visible_primitive]
            represented = region >= 0
            region_mass = np.bincount(
                region[represented], weights=primitive_mass[visible_primitive[represented]],
                minlength=len(child_groups),
            )
            candidate = np.flatnonzero(eligible & (region_mass > 0.0))
            candidate = candidate[
                np.lexsort((candidate, -region_mass[candidate]))[:MAXIMUM_VISIBLE_REGIONS]
            ]
            map_normal = plane_normal[candidate]
            map_offset = plane_offset[candidate]
            weight = region_mass[candidate]
            center = camera_center_from_pose_w2c(target_w2c)
            query_normal_ideal = (target_w2c[:3, :3] @ map_normal.T).T
            query_offset_ideal = map_offset - map_normal @ center
            query_normal_visible, query_offset_visible = [], []
            for region_row in candidate.tolist():
                members = region_members[region_row]
                members = members[primitive_mass[members] > 0.0]
                normal, offset, _ = fit_weighted_primitive_plane(
                    physical, members, primitive_mass[members], plane_normal[region_row],
                )
                query_normal_visible.append(target_w2c[:3, :3] @ normal)
                query_offset_visible.append(offset - float(normal @ center))
            for mode, query_normal, query_offset, destination in (
                ("ideal_parameter", query_normal_ideal, query_offset_ideal, ideal_rows),
                ("visible_subset_refit", np.asarray(query_normal_visible), np.asarray(query_offset_visible), visible_rows),
            ):
                rotation, _ = solve_rotation_from_plane_normals(
                    query_normal, map_normal, weight,
                )
                metric_center, normal_rank, normal_singular = solve_metric_translation(
                    map_normal, map_offset, query_offset, weight,
                )
                scaled_center, scale, scale_rank, scale_singular = solve_scaled_translation(
                    map_normal, map_offset, query_offset, weight,
                )
                condition = (
                    float(normal_singular[0] / normal_singular[-1])
                    if normal_rank == 3 and normal_singular[-1] > 0 else float("inf")
                )
                destination.append({
                    "query_index": query_index, "image_id": image_id,
                    "selected_region_count": int(candidate.size),
                    "normal_rank": int(normal_rank), "scale_design_rank": int(scale_rank),
                    "map_normal_condition": condition,
                    "rotation_error_deg": _rotation_error_deg(rotation, target_w2c),
                    "metric_translation_error_m": float(np.linalg.norm(metric_center - center)),
                    "scale_translation_error_m": float(np.linalg.norm(scaled_center - center)),
                    "estimated_scale": float(scale),
                    "estimated_scale_absolute_error": float(abs(scale - 1.0)),
                    "selected_visible_mass_fraction": float(
                        np.sum(weight) / max(float(np.sum(region_mass)), 1.0e-12)
                    ),
                })
        result[variant_name] = {
            "region_count": len(child_groups), "eligible_region_count": int(np.sum(eligible)),
            "eligible_weighted_surface_fraction": float(np.sum(plane_area[eligible]) / total_surface_mass),
            "ideal_parameter": {"summary": _summary(ideal_rows), "rows": ideal_rows},
            "visible_subset_refit": {"summary": _summary(visible_rows), "rows": visible_rows},
        }
    best = result["parent_restricted_bounded_merge"]["visible_subset_refit"]["summary"]
    report = {
        "artifact_type": SCHEMA, "query_route": "seq10", "query_count": len(image_ids),
        "source_files": {
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "file_sha256": file_sha256(Path(__file__).resolve()),
            },
            "planar_core": {
                "path": str(
                    Path(__file__).resolve().parents[2]
                    / "vfm/localization_goal_maplet/planar_maplet_oracle.py"
                ),
                "file_sha256": file_sha256(
                    Path(__file__).resolve().parents[2]
                    / "vfm/localization_goal_maplet/planar_maplet_oracle.py"
                ),
            },
        },
        "physical_map": str(physical_path), "physical_map_file_sha256": file_sha256(physical_path),
        "physical_map_content_sha256": physical.content_sha256,
        "direct_dataset": str(direct_path), "direct_dataset_file_sha256": file_sha256(direct_path),
        "direct_dataset_content_sha256": direct_metadata["content_sha256"],
        "contributor_inventory": [
            {"image_id": image_id, "path": str(contributor_by_image[image_id]),
             "file_sha256": file_sha256(contributor_by_image[image_id])}
            for image_id in image_ids
        ],
        "map_extraction_contract": {
            "atomic_seed": "physical_child_exact_primitive_membership",
            "plane_equation": "unit_n_world_dot_x_equals_rho_world",
            "maximum_visible_regions": MAXIMUM_VISIBLE_REGIONS,
            "planarity_distance_rms_m": PLANARITY_DISTANCE_RMS_M,
            "minimum_boundary_area_m2": MINIMUM_BOUNDARY_AREA_M2,
            "merge_normal_cosine": MERGE_NORMAL_COSINE,
            "merge_plane_distance_m": MERGE_PLANE_DISTANCE_M,
            "merge_boundary_gap_m": MERGE_BOUNDARY_GAP_M,
        },
        "solver_contract": {
            "rotation": "weighted_Kabsch_query_normal_to_world_normal",
            "metric_translation": "rank3_weighted_linear_plane_offset_solver_fixed_scale1",
            "scale_aware_translation": "rank4_weighted_linear_plane_offset_and_global_scale_solver",
            "paper_three_plane_scale_claim_treated_as_insufficient_without_rank4": True,
        },
        "variants": result,
        "feasibility_target_metric_1m10_recall": FEASIBILITY_TARGET_1M10,
        "decision": (
            "GO_TO_QUERY_PLANE_RECOVERY"
            if best["metric_1m10_recall"] >= FEASIBILITY_TARGET_1M10
            else "KILL_PLANE_ONLY_TRANSLATION_BACKEND"
        ),
        "recommended_residual_role": "orientation_and_multihypothesis_sidecar",
        "uses_alike": False, "uses_pnp": False, "uses_pose_lattice": False,
        "uses_gaussian_retraining": False, "oracle_correspondences": True,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "decision": report["decision"],
        "summaries": {
            name: value["visible_subset_refit"]["summary"]
            for name, value in result.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
