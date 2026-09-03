"""Refine frozen RADIO surface poses with MoGe3 plane normals and one scale.

The input PnP pose and all region-to-map-plane associations are frozen before
any query pose/ground-truth member is opened.  MoGe3 contributes finite-plane
normal and offset measurements; its metric depth scale is an explicit latent
variable with a weak metric prior, not an assumed ground truth depth map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_frozen_correspondences,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _plane_balance_weights,
    _token_pixels,
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


def _reprojection_rows(
    pose_w2c: np.ndarray,
    world: np.ndarray,
    pixel: np.ndarray,
    K: np.ndarray,
    k1: float,
    maximum_error_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, np.float64)
    world = np.asarray(world, np.float64).reshape(-1, 3)
    pixel = np.asarray(pixel, np.float64).reshape(-1, 2)
    if len(world) != len(pixel):
        raise ValueError("world/pixel correspondence count differs")
    error = np.full(len(world), np.inf, np.float64)
    # An upstream solver may deliberately seal an unusable candidate with a
    # non-finite pose.  It must contribute no verifier evidence rather than
    # reaching OpenCV with NaNs (which can return ``projected=None``).
    if (
        pose.shape != (4, 4)
        or not np.all(np.isfinite(pose))
        or not np.all(np.isfinite(K))
        or not np.isfinite(k1)
    ):
        return np.zeros(0, np.int64), error
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
        np.asarray(K, np.float64), np.asarray([k1, 0.0, 0.0, 0.0, 0.0]),
    )
    if projected is None:
        return np.zeros(0, np.int64), error
    error = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    rows = np.flatnonzero(
        (camera[:, 2] > 0.0) & np.isfinite(error) & (error <= float(maximum_error_px))
    )
    return rows, error


def _greedy_plane_associations(
    provenance: np.ndarray,
    inlier_rows: np.ndarray,
    *,
    minimum_support: int = 3,
) -> np.ndarray:
    """Select distinct query-region/map-plane pairs by frozen inlier support."""
    values = np.asarray(provenance, np.int64).reshape(-1, 3)[np.asarray(inlier_rows, np.int64)]
    if not len(values):
        return np.zeros((0, 3), np.int64)
    pair, count = np.unique(values[:, :2], axis=0, return_counts=True)
    order = np.lexsort((pair[:, 1], pair[:, 0], -count))
    output: list[tuple[int, int, int]] = []
    used_region: set[int] = set()
    used_plane: set[int] = set()
    for row in order.tolist():
        region, plane = map(int, pair[row])
        support = int(count[row])
        if support < int(minimum_support):
            continue
        if region in used_region or plane in used_plane:
            continue
        used_region.add(region); used_plane.add(plane)
        output.append((region, plane, support))
    return np.asarray(output, np.int64).reshape(-1, 3)


def _many_to_one_plane_associations(
    provenance: np.ndarray,
    inlier_rows: np.ndarray,
    *,
    minimum_support: int = 3,
) -> np.ndarray:
    """Keep split query fragments while assigning each region only once.

    Sparse foreground occlusion can split one facade into several disconnected
    query regions.  A physical map plane may therefore repeat, but a query
    region is assigned only to its strongest supported map plane.
    """

    values = np.asarray(provenance, np.int64).reshape(-1, 3)[np.asarray(inlier_rows, np.int64)]
    if not len(values):
        return np.zeros((0, 3), np.int64)
    pair, count = np.unique(values[:, :2], axis=0, return_counts=True)
    order = np.lexsort((pair[:, 1], pair[:, 0], -count))
    output: list[tuple[int, int, int]] = []
    used_region: set[int] = set()
    for row in order.tolist():
        region, plane = map(int, pair[row])
        support = int(count[row])
        if support < int(minimum_support) or region in used_region:
            continue
        used_region.add(region)
        output.append((region, plane, support))
    return np.asarray(output, np.int64).reshape(-1, 3)


def _map_plane_balanced_association_weights(associations: np.ndarray) -> np.ndarray:
    """Give every physical map plane unit total squared residual weight."""

    pair = np.asarray(associations, np.int64).reshape(-1, 3)
    if not len(pair):
        return np.zeros(0, np.float64)
    support = np.maximum(pair[:, 2].astype(np.float64), 1.0)
    total = np.zeros(len(pair), np.float64)
    for plane in np.unique(pair[:, 1]):
        rows = np.flatnonzero(pair[:, 1] == plane)
        total[rows] = float(np.sum(support[rows]))
    return np.sqrt(support / np.maximum(total, 1e-12))


def _conditional_scalar_information(jacobian: np.ndarray, scalar_column: int = -1) -> float:
    """Schur-complement information for one scalar after nuisance removal."""
    jac = np.asarray(jacobian, np.float64)
    if jac.ndim != 2 or jac.shape[0] == 0 or jac.shape[1] < 2:
        raise ValueError("observability Jacobian differs")
    column = int(scalar_column) % jac.shape[1]
    nuisance = np.delete(jac, column, axis=1)
    scalar = jac[:, column]
    h_nn = nuisance.T @ nuisance
    h_ns = nuisance.T @ scalar
    value = float(scalar @ scalar - h_ns @ np.linalg.pinv(h_nn, rcond=1e-10) @ h_ns)
    return max(value, 0.0)


MINIMUM_SCALE_DATA_TO_PRIOR_INFORMATION_RATIO = 1.0


def _refine_pose_scale(
    pose_w2c: np.ndarray,
    world: np.ndarray,
    pixel: np.ndarray,
    provenance: np.ndarray,
    inlier_rows: np.ndarray,
    associations: np.ndarray,
    map_normals_world: np.ndarray,
    map_offsets_world: np.ndarray,
    query_normals_camera: np.ndarray,
    query_offsets_camera: np.ndarray,
    K: np.ndarray,
    k1: float,
    query_visible_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray, float, bool, dict[str, float]]:
    """Joint Huber surface-reprojection/plane-normal/plane-offset solve."""
    pose = np.asarray(pose_w2c, np.float64)
    selected = np.asarray(inlier_rows, np.int64)
    pair = np.asarray(associations, np.int64).reshape(-1, 3)
    if len(selected) < 6 or len(pair) < 2:
        return pose.copy(), 1.0, False, {"initial_reprojection_median_px": float("inf")}
    map_plane = pair[:, 1]
    query_region = pair[:, 0]
    map_n = np.asarray(map_normals_world[map_plane], np.float64).copy()
    map_d = np.asarray(map_offsets_world[map_plane], np.float64).copy()
    query_n = np.asarray(query_normals_camera[query_region], np.float64)
    query_d = np.asarray(query_offsets_camera[query_region], np.float64)
    association_weight = _map_plane_balanced_association_weights(pair)
    initial_nc = map_n @ pose[:3, :3].T
    flip = np.sum(initial_nc * query_n, axis=1) < 0.0
    map_n[flip] *= -1.0; map_d[flip] *= -1.0
    point_weight = _plane_balance_weights(np.asarray(provenance)[selected, 1])
    if query_visible_fraction is not None:
        visible = np.asarray(query_visible_fraction, np.float64).reshape(-1)
        if len(visible) != len(provenance) or not np.all(np.isfinite(visible)):
            raise ValueError("query visible-fraction weights differ")
        point_weight *= np.sqrt(np.clip(visible[selected], 0.0, 1.0))
        point_weight /= max(float(np.sqrt(np.mean(point_weight * point_weight))), 1e-12)
    distortion = np.asarray([k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    initial = np.r_[cv2.Rodrigues(pose[:3, :3])[0].reshape(3), pose[:3, 3], 0.0]

    def components(parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        rotation = cv2.Rodrigues(parameter[:3])[0]
        projected, _ = cv2.projectPoints(
            world[selected], parameter[:3], parameter[3:6], K, distortion,
        )
        reprojection = projected.reshape(-1, 2) - pixel[selected]
        normal_camera = map_n @ rotation.T
        normal = np.cross(normal_camera, query_n)
        camera_offset = map_d + np.sum(normal_camera * parameter[3:6], axis=1)
        scale = float(np.exp(parameter[6]))
        offset = camera_offset - scale * query_d
        return reprojection, normal, offset, scale

    def residual(parameter: np.ndarray) -> np.ndarray:
        reprojection, normal, offset, _ = components(parameter)
        return np.r_[
            (reprojection * point_weight[:, None] / 2.0).reshape(-1),
            (normal * association_weight[:, None] / np.sin(np.deg2rad(10.0))).reshape(-1),
            offset * association_weight / 0.20,
            parameter[6] / np.log(1.5),
        ]

    initial_reprojection, initial_normal, initial_offset, _ = components(initial)
    solution = least_squares(
        residual, initial, method="trf", loss="huber", f_scale=1.0,
        max_nfev=120, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    diagnostics = {
        "initial_reprojection_median_px": float(np.median(np.linalg.norm(initial_reprojection, axis=1))),
        "initial_plane_normal_median_deg": float(np.median(np.rad2deg(np.arcsin(np.clip(
            np.linalg.norm(initial_normal, axis=1), 0.0, 1.0,
        ))))),
        "initial_plane_offset_median_m": float(np.median(np.abs(initial_offset))),
    }
    if not solution.success or not np.all(np.isfinite(solution.x)):
        return pose.copy(), 1.0, False, diagnostics
    # The last residual is the weak log-scale prior.  Excluding it makes this
    # diagnostic measure whether image/plane data themselves observe scale
    # after marginalizing the six pose nuisance variables.
    data_scale_information = _conditional_scalar_information(solution.jac[:-1], -1)
    prior_scale_information = float(1.0 / np.square(np.log(1.5)))
    final_reprojection, final_normal, final_offset, scale = components(solution.x)
    diagnostics.update({
        "final_reprojection_median_px": float(np.median(np.linalg.norm(final_reprojection, axis=1))),
        "final_plane_normal_median_deg": float(np.median(np.rad2deg(np.arcsin(np.clip(
            np.linalg.norm(final_normal, axis=1), 0.0, 1.0,
        ))))),
        "final_plane_offset_median_m": float(np.median(np.abs(final_offset))),
        "fitted_moge3_depth_scale": scale,
        "conditional_log_scale_data_information": data_scale_information,
        "log_scale_prior_information": prior_scale_information,
        "conditional_scale_data_to_prior_information_ratio": float(
            data_scale_information / prior_scale_information
        ),
        "scale_data_observable_at_least_as_strong_as_prior": bool(
            data_scale_information >= prior_scale_information
        ),
    })
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = cv2.Rodrigues(solution.x[:3])[0]
    output[:3, 3] = solution.x[3:6]
    final_rows, final_error = _reprojection_rows(output, world, pixel, K, k1)
    initial_rows, initial_error = _reprojection_rows(pose, world, pixel, K, k1)
    accepted = bool(
        data_scale_information
        >= MINIMUM_SCALE_DATA_TO_PRIOR_INFORMATION_RATIO * prior_scale_information
        and
        len(final_rows) >= max(6, int(np.floor(0.9 * len(initial_rows))))
        and float(np.median(final_error[final_rows]))
        <= float(np.median(initial_error[initial_rows])) + 0.25
        and diagnostics["final_plane_normal_median_deg"]
        <= diagnostics["initial_plane_normal_median_deg"] + 1e-9
        and diagnostics["final_plane_offset_median_m"]
        <= diagnostics["initial_plane_offset_median_m"] + 1e-9
    )
    return (output if accepted else pose.copy()), (scale if accepted else 1.0), accepted, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--query_plane_dir", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument(
        "--query_support_weighting", choices=("uniform", "sqrt_visible_fraction"),
        default="uniform",
    )
    parser.add_argument(
        "--plane_association_policy",
        choices=("one_to_one", "many_query_fragments_per_map_plane"),
        default="one_to_one",
    )
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite MoGe3 plane refinement")
    poses, pose_meta = _load_frozen_poses(args.frozen_pose_inventory)
    correspondence, correspondence_meta = _load_frozen_correspondences(args.frozen_correspondences)
    if (
        args.query_support_weighting == "sqrt_visible_fraction"
        and "query_plane_visible_fraction" not in correspondence
    ):
        raise ValueError("soft core/halo weighting requires correspondence inventory v2")
    if not np.array_equal(poses["names"].astype(str), correspondence["names"].astype(str)):
        raise ValueError("pose and correspondence query inventory differs")
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    output_pose = np.asarray(poses["pose_w2c"], np.float64).copy()
    output_scale = np.ones(len(output_pose), np.float64)
    accepted = np.zeros(len(output_pose), bool)
    association_count = np.zeros(len(output_pose), np.int64)
    output_inliers = np.asarray(poses["pnp_inlier_count"], np.int64).copy()
    diagnostics = []
    for index, name in enumerate(poses["names"].astype(str).tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(poses["usable"][index])}
        if not row["usable"]:
            diagnostics.append(row); continue
        lo, hi = map(int, correspondence["correspondence_offsets"][index:index + 2])
        world = np.asarray(correspondence["world_points"][lo:hi], np.float64)
        token = np.asarray(correspondence["query_tokens"][lo:hi], np.int64)
        provenance = np.asarray(
            correspondence["provenance_region_plane_atlas_row"][lo:hi], np.int64,
        )
        pixel = (
            np.asarray(correspondence["query_measurements_xy"][lo:hi], np.float64)
            if "query_measurements_xy" in correspondence
            else _token_pixels(token, tuple(correspondence_meta.get("token_grid", (36, 64))))
        )
        K = np.asarray(correspondence["camera_matrices"][index], np.float64)
        k1 = float(correspondence["radial_k1"][index])
        initial_rows, _ = _reprojection_rows(output_pose[index], world, pixel, K, k1)
        association = (
            _many_to_one_plane_associations(provenance, initial_rows)
            if args.plane_association_policy == "many_query_fragments_per_map_plane"
            else _greedy_plane_associations(provenance, initial_rows)
        )
        query_plane, query_meta = QueryPlaneRegions.load_npz(args.query_plane_dir / name)
        if np.any(association[:, 0] >= len(query_plane.normals_camera)):
            raise ValueError("frozen region association lies outside query planes")
        refined, scale, use, detail = _refine_pose_scale(
            output_pose[index], world, pixel, provenance, initial_rows, association,
            planes.normals_world, planes.offsets_world,
            query_plane.normals_camera, query_plane.offsets_camera, K, k1,
            query_visible_fraction=(
                correspondence["query_plane_visible_fraction"][lo:hi]
                if args.query_support_weighting == "sqrt_visible_fraction" else None
            ),
        )
        output_pose[index] = refined; output_scale[index] = scale; accepted[index] = use
        association_count[index] = len(association)
        final_rows, _ = _reprojection_rows(refined, world, pixel, K, k1)
        output_inliers[index] = len(final_rows)
        row.update({
            "refinement_accepted": bool(use),
            "plane_association_count": int(len(association)),
            "query_plane_file_sha256": file_sha256(args.query_plane_dir / name),
            "query_plane_content_sha256": query_meta.get("content_sha256"),
            **detail,
        })
        diagnostics.append(row)

    arrays = {
        "names": poses["names"], "pose_w2c": output_pose,
        "usable": np.asarray(poses["usable"], bool),
        "candidate_correspondence_count": np.asarray(poses["candidate_correspondence_count"], np.int64),
        "pnp_inlier_count": output_inliers,
        "moge3_depth_scale": output_scale,
        "plane_refinement_accepted": accepted,
        "plane_association_count": association_count,
    }
    metadata = {
        "artifact_type": "goal_maplet_moge3_plane_scale_surface_refinement_v2",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(output_pose)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": True,
        "moge3_role": "plane_normal_and_offset_with_one_weakly_metric_scale_latent",
        "reprojection_scale_px": 2.0,
        "plane_normal_scale_deg": 10.0,
        "plane_offset_scale_m": 0.20,
        "metric_scale_prior_log_sigma": float(np.log(1.5)),
        "scale_observability": (
            "Schur complement of robust data Jacobian after six pose nuisance variables; "
            "prior row excluded; weakly observed scale rejects the joint update and falls "
            "back to the input pose with unit scale"
        ),
        "minimum_scale_data_to_prior_information_ratio": (
            MINIMUM_SCALE_DATA_TO_PRIOR_INFORMATION_RATIO
        ),
        "minimum_pair_support": 3,
        "minimum_distinct_plane_pairs": 2,
        "query_support_weighting": str(args.query_support_weighting),
        "plane_association_policy": str(args.plane_association_policy),
        "map_plane_fragment_weighting": (
            "support_proportional_unit_total_squared_weight_per_physical_map_plane"
        ),
        "frozen_pose_inventory_file_sha256": file_sha256(args.frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": pose_meta.get("content_sha256"),
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": correspondence_meta.get("content_sha256"),
        "planar_map_file_sha256": file_sha256(args.planar_map),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    # Phase 2: labels are opened only after the refined inventory is sealed.
    rows = []
    for index, name in enumerate(arrays["names"].astype(str).tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        pose = arrays["pose_w2c"][index]
        center = -pose[:3, :3].T @ pose[:3, 3]
        gt_center = -gt[:3, :3].T @ gt[:3, 3]
        rows.append({
            **diagnostics[index],
            "translation_error_m": float(np.linalg.norm(center - gt_center)),
            "rotation_error_deg": float(Rotation.from_matrix(
                pose[:3, :3] @ gt[:3, :3].T,
            ).magnitude() * 180.0 / np.pi),
        })
    translation = np.asarray([row["translation_error_m"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    valid = np.asarray(arrays["usable"], bool) & np.isfinite(translation) & np.isfinite(rotation)
    if not np.any(valid):
        raise ValueError("refined pose inventory has no finite usable pose")
    report = {
        "artifact_type": "goal_maplet_moge3_plane_scale_surface_refinement_evaluation_v2",
        "query_count": int(len(rows)),
        "accepted_count": int(np.sum(accepted)),
        "scale_observable_count": int(sum(
            row.get("scale_data_observable_at_least_as_strong_as_prior") is True
            for row in diagnostics
        )),
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "usable_count": int(np.sum(valid)),
        "median_translation_m": float(np.median(translation[valid])),
        "median_rotation_deg": float(np.median(rotation[valid])),
        "p90_translation_m": float(np.quantile(translation[valid], 0.9)),
        "p90_rotation_deg": float(np.quantile(rotation[valid], 0.9)),
        "recall_2m45": float(np.mean(valid & (translation <= 2.0) & (rotation <= 45.0))),
        "recall_1m10": float(np.mean(valid & (translation <= 1.0) & (rotation <= 10.0))),
        "recall_0p5m5": float(np.mean(valid & (translation <= 0.5) & (rotation <= 5.0))),
        "recall_0p25m2": float(np.mean(valid & (translation <= 0.25) & (rotation <= 2.0))),
        "recall_0p1m1": float(np.mean(valid & (translation <= 0.1) & (rotation <= 1.0))),
        "rows": rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
