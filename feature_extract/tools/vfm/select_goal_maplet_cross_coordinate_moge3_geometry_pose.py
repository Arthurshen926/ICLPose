"""Select point or continuous-surface pose using common MoGe3 plane geometry.

Both candidates arise from the same anonymous RADIO token/prototype matches.
The incumbent point-coordinate pose defines one common, frozen set of
query-region/map-plane associations.  Each candidate is then scored on that
same set using MoGe3 plane normals, plane offsets, and one independently fitted
global depth scale with the already frozen weak metric prior.  This avoids
candidate-specific correspondence selection and gives MoGe3 a verifier role
that can reject motion along weakly constrained image-plane directions.

The surface-coordinate pose is selected only when its robust plane-geometry
objective is strictly lower.  The selected pose is sealed before query pose or
ground truth is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_moge3 import (
    _greedy_plane_associations,
    _many_to_one_plane_associations,
    _map_plane_balanced_association_weights,
    _reprojection_rows,
)
from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_surface_pose import (
    _paired_contract,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)
from feature_extract.vfm.localization_goal_maplet.query_plane_regions import QueryPlaneRegions


NORMAL_RESIDUAL_SCALE_DEG = 10.0
OFFSET_RESIDUAL_SCALE_M = 0.20
LOG_SCALE_PRIOR_SIGMA = float(np.log(1.5))
MINIMUM_ASSOCIATIONS = 2


def _huber_mean(residual: np.ndarray) -> float:
    absolute = np.abs(np.asarray(residual, np.float64).reshape(-1))
    loss = np.where(absolute <= 1.0, 0.5 * np.square(absolute), absolute - 0.5)
    return float(np.mean(loss)) if len(loss) else float("inf")


def _frozen_plane_signs(
    reference_pose_w2c: np.ndarray,
    map_normals_world: np.ndarray,
    query_normals_camera: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_pose_w2c, np.float64).reshape(4, 4)
    map_normal = np.asarray(map_normals_world, np.float64).reshape(-1, 3)
    query_normal = np.asarray(query_normals_camera, np.float64).reshape(-1, 3)
    if len(map_normal) != len(query_normal):
        raise ValueError("map/query plane-normal counts differ")
    camera_normal = map_normal @ reference[:3, :3].T
    return np.where(np.sum(camera_normal * query_normal, axis=1) < 0.0, -1.0, 1.0)


def _plane_geometry_objective(
    pose_w2c: np.ndarray,
    map_normals_world: np.ndarray,
    map_offsets_world: np.ndarray,
    query_normals_camera: np.ndarray,
    query_offsets_camera: np.ndarray,
    association_weights: np.ndarray,
    frozen_signs: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    """Fit only the one shared MoGe3 scale, then score frozen plane pairs."""
    pose = np.asarray(pose_w2c, np.float64).reshape(4, 4)
    sign = np.asarray(frozen_signs, np.float64).reshape(-1)
    map_normal = np.asarray(map_normals_world, np.float64).reshape(-1, 3) * sign[:, None]
    map_offset = np.asarray(map_offsets_world, np.float64).reshape(-1) * sign
    query_normal = np.asarray(query_normals_camera, np.float64).reshape(-1, 3)
    query_offset = np.asarray(query_offsets_camera, np.float64).reshape(-1)
    weight = np.asarray(association_weights, np.float64).reshape(-1)
    if not (
        len(map_normal) == len(map_offset) == len(query_normal)
        == len(query_offset) == len(weight) == len(sign)
    ):
        raise ValueError("plane geometry score inputs differ")
    if len(map_normal) < MINIMUM_ASSOCIATIONS:
        return float("inf"), 1.0, {
            "normal_median_deg": float("inf"),
            "offset_median_m": float("inf"),
        }
    camera_normal = map_normal @ pose[:3, :3].T
    camera_offset = map_offset + np.sum(camera_normal * pose[:3, 3], axis=1)
    normal_cross = np.cross(camera_normal, query_normal)
    normal_residual = (
        normal_cross * weight[:, None] / np.sin(np.deg2rad(NORMAL_RESIDUAL_SCALE_DEG))
    ).reshape(-1)

    def residual(log_scale: np.ndarray) -> np.ndarray:
        offset = (
            camera_offset - float(np.exp(log_scale[0])) * query_offset
        ) * weight / OFFSET_RESIDUAL_SCALE_M
        return np.r_[normal_residual, offset, log_scale[0] / LOG_SCALE_PRIOR_SIGMA]

    solution = least_squares(
        residual,
        np.zeros(1, np.float64),
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=80,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not solution.success or not np.all(np.isfinite(solution.x)):
        return float("inf"), 1.0, {
            "normal_median_deg": float("inf"),
            "offset_median_m": float("inf"),
        }
    scale = float(np.exp(solution.x[0]))
    offset_error = np.abs(camera_offset - scale * query_offset)
    normal_angle = np.rad2deg(np.arcsin(np.clip(
        np.linalg.norm(normal_cross, axis=1), 0.0, 1.0,
    )))
    return _huber_mean(residual(solution.x)), scale, {
        "normal_median_deg": float(np.median(normal_angle)),
        "offset_median_m": float(np.median(offset_error)),
    }


def _select_surface(point_objective: np.ndarray, surface_objective: np.ndarray) -> np.ndarray:
    point = np.asarray(point_objective, np.float64).reshape(-1)
    surface = np.asarray(surface_objective, np.float64).reshape(-1)
    if len(point) != len(surface) or np.any(np.isnan(point)) or np.any(np.isnan(surface)):
        raise ValueError("candidate plane-geometry objectives differ")
    return (surface < point).astype(np.int8)


def _pose_error(pose_w2c: np.ndarray, target_w2c: np.ndarray) -> tuple[float, float]:
    pose = np.asarray(pose_w2c, np.float64)
    target = np.asarray(target_w2c, np.float64)
    center = -pose[:3, :3].T @ pose[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    return (
        float(np.linalg.norm(center - target_center)),
        float(Rotation.from_matrix(
            pose[:3, :3] @ target[:3, :3].T,
        ).magnitude() * 180.0 / np.pi),
    )


def _threshold_counts(translation: np.ndarray, rotation: np.ndarray) -> dict[str, int]:
    return {
        "0.1m_1deg": int(np.sum((translation <= 0.1) & (rotation <= 1.0))),
        "0.25m_2deg": int(np.sum((translation <= 0.25) & (rotation <= 2.0))),
        "0.5m_5deg": int(np.sum((translation <= 0.5) & (rotation <= 5.0))),
        "1m_10deg": int(np.sum((translation <= 1.0) & (rotation <= 10.0))),
        "2m_45deg": int(np.sum((translation <= 2.0) & (rotation <= 45.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point_pose", type=Path, required=True)
    parser.add_argument("--surface_pose", type=Path, required=True)
    parser.add_argument("--point_correspondences", type=Path, required=True)
    parser.add_argument("--surface_correspondences", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument(
        "--plane_association_policy",
        choices=("one_to_one", "many_query_fragments_per_map_plane"),
        default="one_to_one",
    )
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite cross-coordinate MoGe3 selection")

    point_pose, point_pose_meta = _load_pose_candidate(args.point_pose)
    surface_pose, surface_pose_meta = _load_pose_candidate(args.surface_pose)
    point, point_meta = _load_correspondences(args.point_correspondences)
    surface, surface_meta = _load_correspondences(args.surface_correspondences)
    _paired_contract(point, point_meta, surface, surface_meta)
    names = point["names"].astype(str)
    if not (
        np.array_equal(names, surface["names"].astype(str))
        and np.array_equal(names, point_pose["names"].astype(str))
        and np.array_equal(names, surface_pose["names"].astype(str))
    ):
        raise ValueError("cross-coordinate MoGe3 query order differs")
    planar_map = GeometryNativePlanarMap.load_npz(args.planar_map)

    objectives = np.full((len(names), 2), np.inf, np.float64)
    scales = np.ones((len(names), 2), np.float64)
    normal_median = np.full((len(names), 2), np.inf, np.float64)
    offset_median = np.full((len(names), 2), np.inf, np.float64)
    association_count = np.zeros(len(names), np.int64)
    query_plane_hashes: list[str] = []
    for query, name in enumerate(names.tolist()):
        query_plane, query_meta = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        query_plane_hashes.append(file_sha256(args.query_plane_dir / name))
        if query_meta.get("uses_pose_or_ground_truth") is not False:
            raise ValueError("MoGe3 query plane inventory opened pose or ground truth")
        lo, hi = map(int, point["correspondence_offsets"][query:query + 2])
        initial_rows, _ = _reprojection_rows(
            point_pose["pose_w2c"][query],
            point["world_points"][lo:hi],
            point["query_measurements_xy"][lo:hi],
            point["camera_matrices"][query],
            float(point["radial_k1"][query]),
        )
        association_builder = (
            _many_to_one_plane_associations
            if args.plane_association_policy == "many_query_fragments_per_map_plane"
            else _greedy_plane_associations
        )
        association = association_builder(
            point["provenance_region_plane_atlas_row"][lo:hi], initial_rows,
        )
        association_count[query] = len(association)
        if len(association) < MINIMUM_ASSOCIATIONS:
            continue
        region = association[:, 0]
        plane = association[:, 1]
        if np.any(region >= len(query_plane.normals_camera)) or np.any(plane >= len(planar_map.normals_world)):
            raise ValueError("frozen plane association is out of range")
        weight = _map_plane_balanced_association_weights(association)
        signs = _frozen_plane_signs(
            point_pose["pose_w2c"][query],
            planar_map.normals_world[plane],
            query_plane.normals_camera[region],
        )
        for candidate, poses in enumerate((point_pose, surface_pose)):
            objective, scale, detail = _plane_geometry_objective(
                poses["pose_w2c"][query],
                planar_map.normals_world[plane],
                planar_map.offsets_world[plane],
                query_plane.normals_camera[region],
                query_plane.offsets_camera[region],
                weight,
                signs,
            )
            objectives[query, candidate] = objective
            scales[query, candidate] = scale
            normal_median[query, candidate] = detail["normal_median_deg"]
            offset_median[query, candidate] = detail["offset_median_m"]

    selected = _select_surface(objectives[:, 0], objectives[:, 1])
    candidate_pose = np.stack([point_pose["pose_w2c"], surface_pose["pose_w2c"]], axis=1)
    candidate_usable = np.stack([point_pose["usable"], surface_pose["usable"]], axis=1)
    selected = np.where(candidate_usable[:, 1], selected, 0).astype(np.int8)
    row = np.arange(len(names))
    arrays = {
        "names": point_pose["names"],
        "pose_w2c": candidate_pose[row, selected],
        "usable": candidate_usable[row, selected],
        "selected_branch": selected,
        "moge3_plane_geometry_objective": objectives,
        "moge3_fitted_depth_scale": scales,
        "moge3_plane_normal_median_deg": normal_median,
        "moge3_plane_offset_median_m": offset_median,
        "frozen_plane_association_count": association_count,
    }
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_cross_coordinate_moge3_geometry_pose_selection_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": True,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "selection_rule": "surface_only_if_strictly_lower_robust_MoGe3_plane_normal_offset_plus_refit_single_scale_objective_on_pointV5_frozen_associations",
        "association_policy": (
            "many_query_fragments_per_map_plane_from_pointV5_4px_reprojection_support_min3_with_unit_total_map_plane_weight"
            if args.plane_association_policy == "many_query_fragments_per_map_plane"
            else "one_to_one_region_map_plane_from_pointV5_4px_reprojection_support_min3"
        ),
        "candidate_specific_association_selection": False,
        "plane_signs_frozen_at_point_candidate": True,
        "normal_residual_scale_deg": NORMAL_RESIDUAL_SCALE_DEG,
        "offset_residual_scale_m": OFFSET_RESIDUAL_SCALE_M,
        "log_scale_prior_sigma": LOG_SCALE_PRIOR_SIGMA,
        "minimum_associations": MINIMUM_ASSOCIATIONS,
        "point_pose_file_sha256": file_sha256(args.point_pose),
        "point_pose_content_sha256": point_pose_meta.get("content_sha256"),
        "surface_pose_file_sha256": file_sha256(args.surface_pose),
        "surface_pose_content_sha256": surface_pose_meta.get("content_sha256"),
        "point_correspondence_file_sha256": file_sha256(args.point_correspondences),
        "point_correspondence_content_sha256": point_meta.get("content_sha256"),
        "surface_correspondence_file_sha256": file_sha256(args.surface_correspondences),
        "surface_correspondence_content_sha256": surface_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "query_plane_file_sha256_in_order": query_plane_hashes,
        "selected_surface_count": int(np.sum(selected)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    # Phase 2: the selected pose is immutable before opening pose-bearing files.
    point_translation = np.full(len(names), np.inf, np.float64)
    point_rotation = np.full(len(names), np.inf, np.float64)
    final_translation = np.full(len(names), np.inf, np.float64)
    final_rotation = np.full(len(names), np.inf, np.float64)
    rows: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        point_translation[query], point_rotation[query] = _pose_error(
            point_pose["pose_w2c"][query], target,
        )
        final_translation[query], final_rotation[query] = _pose_error(
            arrays["pose_w2c"][query], target,
        )
        rows.append({
            "name": name,
            "selected_branch": int(selected[query]),
            "frozen_plane_association_count": int(association_count[query]),
            "candidate_plane_geometry_objective": objectives[query].tolist(),
            "candidate_moge3_scale": scales[query].tolist(),
            "point_translation_error_m": float(point_translation[query]),
            "point_rotation_error_deg": float(point_rotation[query]),
            "translation_error_m": float(final_translation[query]),
            "rotation_error_deg": float(final_rotation[query]),
        })
    usable = np.asarray(arrays["usable"], bool)
    finite = usable & np.isfinite(final_translation) & np.isfinite(final_rotation)
    point_hits = _threshold_counts(point_translation[finite], point_rotation[finite])
    final_hits = _threshold_counts(final_translation[finite], final_rotation[finite])
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_cross_coordinate_moge3_geometry_pose_selection_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)),
        "usable_count": int(np.sum(finite)),
        "selected_surface_count": int(np.sum(selected)),
        "point_median_translation_m": float(np.median(point_translation[finite])),
        "point_median_rotation_deg": float(np.median(point_rotation[finite])),
        "point_p90_translation_m": float(np.quantile(point_translation[finite], 0.9)),
        "point_p90_rotation_deg": float(np.quantile(point_rotation[finite], 0.9)),
        "median_translation_m": float(np.median(final_translation[finite])),
        "median_rotation_deg": float(np.median(final_rotation[finite])),
        "p90_translation_m": float(np.quantile(final_translation[finite], 0.9)),
        "p90_rotation_deg": float(np.quantile(final_rotation[finite], 0.9)),
        "point_threshold_hit_counts": point_hits,
        "threshold_hit_counts": final_hits,
        "threshold_hit_count_delta": {
            key: int(final_hits[key] - point_hits[key]) for key in final_hits
        },
        "rows": rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
