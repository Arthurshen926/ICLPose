"""Refine frozen plane poses with propagated anonymous-map uncertainty.

The query image contributes only the already frozen RADIO token coordinates.
Every token selects at most one mutually exclusive 3D hypothesis at the input
pose.  The selected rows and their uncertainty are then frozen while a bounded
six-DoF reprojection solve is performed.  No query pose, ground truth, source
RGB, source path, or source-view identity enters phase 1.
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
    _load as _load_correspondences,
    _score,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _plane_balance_weights,
    _token_pixels,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _projection_variance_px2,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


TOKEN_QUANTIZATION_VARIANCE_PX2 = 4.0 ** 2 / 12.0
MINIMUM_FIXED_ROWS = 12
MINIMUM_PHYSICAL_PLANES = 2
MAXIMUM_REPROJECTION_ERROR_PX = 4.0
MINIMUM_GLOBAL_SUPPORT_FRACTION = 0.95
MAXIMUM_ROTATION_STEP_DEG = 5.0
MAXIMUM_CAMERA_CENTER_STEP_M = 0.5
MINIMUM_PURITY = 0.25


def _project(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, np.float64)
    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3],
        np.asarray(camera_matrix, np.float64),
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    return projected.reshape(-1, 2), camera


def _fixed_token_hypotheses(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    query_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    *,
    maximum_error_px: float = MAXIMUM_REPROJECTION_ERROR_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze the best valid 3D hypothesis for every distinct query token."""

    token = np.asarray(query_tokens, np.int64).reshape(-1)
    pixel = np.asarray(query_pixels, np.float64).reshape(-1, 2)
    if len(token) != len(pixel) or not np.all(np.isfinite(pixel)):
        raise ValueError("query token measurements differ")
    projected, camera = _project(pose_w2c, world_points, camera_matrix, radial_k1)
    error = np.linalg.norm(projected - pixel, axis=1)
    valid = (camera[:, 2] > 0.0) & np.isfinite(error) & (error <= float(maximum_error_px))
    selected: list[int] = []
    for token_id in np.unique(token):
        rows = np.flatnonzero((token == token_id) & valid)
        if len(rows):
            # argmin is stable and therefore breaks exact ties by source row.
            selected.append(int(rows[np.argmin(error[rows])]))
    return np.asarray(selected, np.int64), error


def _fixed_reprojection_sigma_px(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    covariance_world_m2: np.ndarray,
    plane_purity: np.ndarray,
    plane_depth_dispersion_m: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Propagate atlas geometry uncertainty without double-counting depth.

    The world covariance and plane depth dispersion are two estimates of the
    same map-point uncertainty.  Their maximum, rather than their sum, is used
    conservatively.  Four-pixel RADIO cells contribute their exact uniform-cell
    quantization variance per image coordinate.
    """

    pose = np.asarray(pose_w2c, np.float64)
    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    covariance = np.asarray(covariance_world_m2, np.float64).reshape(-1, 3, 3)
    purity = np.asarray(plane_purity, np.float64).reshape(-1)
    dispersion = np.asarray(plane_depth_dispersion_m, np.float64).reshape(-1)
    if not (len(world) == len(covariance) == len(purity) == len(dispersion)):
        raise ValueError("uncertainty arrays differ in length")
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    covariance_variance = _projection_variance_px2(
        camera, covariance, pose[:3, :3], np.asarray(camera_matrix, np.float64),
    )
    focal = 0.5 * (float(camera_matrix[0, 0]) + float(camera_matrix[1, 1]))
    depth_variance = np.square(
        focal * np.maximum(dispersion, 0.0) / np.maximum(camera[:, 2], 1e-9)
    )
    variance = (
        TOKEN_QUANTIZATION_VARIANCE_PX2
        + np.maximum(covariance_variance, depth_variance)
    ) / np.clip(purity, MINIMUM_PURITY, 1.0)
    sigma = np.sqrt(np.maximum(variance, 1e-12))
    if not np.all(np.isfinite(sigma)):
        raise ValueError("propagated reprojection uncertainty is nonfinite")
    return sigma


def _huber_mean(residual: np.ndarray) -> float:
    absolute = np.abs(np.asarray(residual, np.float64).reshape(-1))
    loss = np.where(absolute <= 1.0, 0.5 * np.square(absolute), absolute - 0.5)
    return float(np.mean(loss)) if len(loss) else float("inf")


def _pose_step(initial: np.ndarray, final: np.ndarray) -> tuple[float, float]:
    rotation = float(Rotation.from_matrix(
        final[:3, :3] @ initial[:3, :3].T,
    ).magnitude() * 180.0 / np.pi)
    initial_center = -initial[:3, :3].T @ initial[:3, 3]
    final_center = -final[:3, :3].T @ final[:3, 3]
    return rotation, float(np.linalg.norm(final_center - initial_center))


def _refine_pose(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    query_pixels: np.ndarray,
    physical_plane_ids: np.ndarray,
    reprojection_sigma_px: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, bool, dict[str, float | int | bool]]:
    """Bounded left-SE(3) robust solve on already frozen correspondences."""

    pose = np.asarray(pose_w2c, np.float64)
    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    pixel = np.asarray(query_pixels, np.float64).reshape(-1, 2)
    plane = np.asarray(physical_plane_ids, np.int64).reshape(-1)
    sigma = np.asarray(reprojection_sigma_px, np.float64).reshape(-1)
    diagnostics: dict[str, float | int | bool] = {
        "fixed_row_count": int(len(world)),
        "fixed_physical_plane_count": int(len(np.unique(plane))),
    }
    if (
        len(world) < MINIMUM_FIXED_ROWS
        or len(np.unique(plane)) < MINIMUM_PHYSICAL_PLANES
        or not (len(world) == len(pixel) == len(plane) == len(sigma))
    ):
        diagnostics["solver_attempted"] = False
        return pose.copy(), False, diagnostics
    balance = _plane_balance_weights(plane)
    distortion = np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64)

    def candidate(parameter: np.ndarray) -> np.ndarray:
        delta_rotation = Rotation.from_rotvec(parameter[:3]).as_matrix()
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = delta_rotation @ pose[:3, :3]
        output[:3, 3] = delta_rotation @ pose[:3, 3] + parameter[3:6]
        return output

    def residual(parameter: np.ndarray) -> np.ndarray:
        output = candidate(parameter)
        projected, _ = cv2.projectPoints(
            world, cv2.Rodrigues(output[:3, :3])[0], output[:3, 3],
            camera_matrix, distortion,
        )
        normalized = (projected.reshape(-1, 2) - pixel) / sigma[:, None]
        return (normalized * balance[:, None]).reshape(-1)

    initial_parameter = np.zeros(6, np.float64)
    initial_residual = residual(initial_parameter)
    rotation_bound = np.deg2rad(MAXIMUM_ROTATION_STEP_DEG)
    lower = np.asarray([-rotation_bound] * 3 + [-MAXIMUM_CAMERA_CENTER_STEP_M] * 3)
    upper = -lower
    solution = least_squares(
        residual, initial_parameter, bounds=(lower, upper), method="trf",
        loss="huber", f_scale=1.0, max_nfev=100,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    output = candidate(solution.x)
    final_residual = residual(solution.x)
    rotation_step, center_step = _pose_step(pose, output)
    initial_objective = _huber_mean(initial_residual)
    final_objective = _huber_mean(final_residual)
    diagnostics.update({
        "solver_attempted": True,
        "solver_success": bool(solution.success),
        "initial_weighted_huber_objective": initial_objective,
        "final_weighted_huber_objective": final_objective,
        "rotation_step_deg": rotation_step,
        "camera_center_step_m": center_step,
        "sigma_px_median": float(np.median(sigma)),
        "sigma_px_p90": float(np.quantile(sigma, 0.9)),
    })
    accepted = bool(
        solution.success
        and np.all(np.isfinite(output))
        and final_objective < initial_objective - 1e-9
        and rotation_step <= MAXIMUM_ROTATION_STEP_DEG + 1e-8
        and center_step <= MAXIMUM_CAMERA_CENTER_STEP_M + 1e-8
    )
    return (output if accepted else pose.copy()), accepted, diagnostics


def _evaluate(
    names: np.ndarray,
    poses_w2c: np.ndarray,
    usable: np.ndarray,
    diagnostics: list[dict[str, object]],
    query_contributors: Path,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    translation = np.full(len(names), np.inf, np.float64)
    rotation = np.full(len(names), np.inf, np.float64)
    for index, name in enumerate(names.astype(str).tolist()):
        with np.load(query_contributors / name, allow_pickle=False) as data:
            gt = np.asarray(data["pose_w2c"], np.float64)
        pose = np.asarray(poses_w2c[index], np.float64)
        if bool(usable[index]) and np.all(np.isfinite(pose)):
            center = -pose[:3, :3].T @ pose[:3, 3]
            gt_center = -gt[:3, :3].T @ gt[:3, 3]
            translation[index] = np.linalg.norm(center - gt_center)
            rotation[index] = Rotation.from_matrix(
                pose[:3, :3] @ gt[:3, :3].T,
            ).magnitude() * 180.0 / np.pi
        rows.append({
            **diagnostics[index], "name": name,
            "translation_error_m": float(translation[index]),
            "rotation_error_deg": float(rotation[index]),
        })
    finite = np.asarray(usable, bool) & np.isfinite(translation) & np.isfinite(rotation)
    return rows, translation, rotation, finite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite uncertainty refinement")
    poses, pose_metadata = _load_frozen_poses(args.frozen_pose_inventory)
    corr, corr_metadata = _load_correspondences(args.frozen_correspondences)
    if corr_metadata.get("artifact_type") not in (
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
    ):
        raise ValueError("uncertainty refinement requires V3/V4 correspondences")
    if not np.array_equal(poses["names"].astype(str), corr["names"].astype(str)):
        raise ValueError("pose and correspondence query order differs")

    output_pose = np.asarray(poses["pose_w2c"], np.float64).copy()
    usable = np.asarray(poses["usable"], bool).copy()
    accepted = np.zeros(len(output_pose), bool)
    fixed_count = np.zeros(len(output_pose), np.int64)
    output_inliers = np.asarray(poses["pnp_inlier_count"], np.int64).copy()
    diagnostics: list[dict[str, object]] = []
    for index, name in enumerate(poses["names"].astype(str).tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(usable[index])}
        if not usable[index]:
            diagnostics.append(row)
            continue
        lo, hi = map(int, corr["correspondence_offsets"][index:index + 2])
        world = np.asarray(corr["world_points"][lo:hi], np.float64)
        token = np.asarray(corr["query_tokens"][lo:hi], np.int64)
        provenance = np.asarray(corr["provenance_region_plane_atlas_row"][lo:hi], np.int64)
        pixel = (
            np.asarray(corr["query_measurements_xy"][lo:hi], np.float64)
            if "query_measurements_xy" in corr
            else _token_pixels(token, tuple(corr_metadata.get("token_grid", (36, 64))))
        )
        K = np.asarray(corr["camera_matrices"][index], np.float64)
        k1 = float(corr["radial_k1"][index])
        initial_score = _score(output_pose[index], world, token, provenance, K, k1,
                               pixel if "query_measurements_xy" in corr else None)
        selected, _ = _fixed_token_hypotheses(
            output_pose[index], world, token, pixel, K, k1,
        )
        fixed_count[index] = len(selected)
        sigma = _fixed_reprojection_sigma_px(
            output_pose[index], world[selected],
            corr["prototype_world_covariance_m2"][lo:hi][selected],
            corr["prototype_plane_pixel_purity"][lo:hi][selected],
            corr["prototype_plane_depth_dispersion_m"][lo:hi][selected], K,
        )
        refined, use, detail = _refine_pose(
            output_pose[index], world[selected], pixel[selected], provenance[selected, 1],
            sigma, K, k1,
        )
        final_score = _score(refined, world, token, provenance, K, k1,
                             pixel if "query_measurements_xy" in corr else None)
        initial_median = initial_score["reprojection_median_px"]
        final_median = final_score["reprojection_median_px"]
        raw_support_pass = bool(
            int(final_score["inlier_count"]) >= max(
                6, int(np.floor(MINIMUM_GLOBAL_SUPPORT_FRACTION * int(initial_score["inlier_count"])))
            )
            and final_median is not None and initial_median is not None
            and float(final_median) <= float(initial_median) + 0.25
        )
        use = bool(use and raw_support_pass)
        if use:
            output_pose[index] = refined
            output_inliers[index] = int(final_score["inlier_count"])
            accepted[index] = True
        row.update({
            **detail, "refinement_accepted": bool(use),
            "raw_support_acceptance_pass": raw_support_pass,
            "initial_global_inlier_count": int(initial_score["inlier_count"]),
            "final_global_inlier_count": int(final_score["inlier_count"]),
            "initial_global_reprojection_median_px": initial_score["reprojection_median_px"],
            "final_global_reprojection_median_px": final_score["reprojection_median_px"],
        })
        diagnostics.append(row)

    arrays = {
        "names": poses["names"], "pose_w2c": output_pose, "usable": usable,
        "candidate_correspondence_count": np.asarray(poses["candidate_correspondence_count"], np.int64),
        "pnp_inlier_count": output_inliers,
        "uncertainty_refinement_accepted": accepted,
        "fixed_token_hypothesis_count": fixed_count,
    }
    metadata = {
        "artifact_type": "goal_maplet_uncertainty_weighted_plane_pose_refinement_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(output_pose)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "uncertainty_model": {
            "token_uniform_cell_variance_px2": TOKEN_QUANTIZATION_VARIANCE_PX2,
            "world_covariance": "first_order_pinhole_J_R_Cworld_RT_JT",
            "depth_dispersion": "focal_over_depth_times_metric_dispersion",
            "combination": "token_variance_plus_max_covariance_or_depth_variance_divided_by_clipped_purity",
            "minimum_purity": MINIMUM_PURITY,
        },
        "fixed_hypothesis_rule": "one_minimum_residual_valid_3D_hypothesis_per_query_token",
        "minimum_fixed_rows": MINIMUM_FIXED_ROWS,
        "minimum_physical_planes": MINIMUM_PHYSICAL_PLANES,
        "maximum_initial_reprojection_error_px": MAXIMUM_REPROJECTION_ERROR_PX,
        "minimum_global_support_fraction": MINIMUM_GLOBAL_SUPPORT_FRACTION,
        "maximum_rotation_step_deg": MAXIMUM_ROTATION_STEP_DEG,
        "maximum_camera_center_step_m": MAXIMUM_CAMERA_CENTER_STEP_M,
        "frozen_pose_inventory_file_sha256": file_sha256(args.frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": pose_metadata.get("content_sha256"),
        "frozen_correspondence_file_sha256": file_sha256(args.frozen_correspondences),
        "frozen_correspondence_content_sha256": corr_metadata.get("content_sha256"),
        "accepted_count": int(np.sum(accepted)), "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    # Phase 2 starts only after the refined pose artifact is sealed.
    rows, translation, rotation, finite = _evaluate(
        arrays["names"], arrays["pose_w2c"], arrays["usable"], diagnostics,
        args.query_contributors,
    )
    report = {
        "artifact_type": "goal_maplet_uncertainty_weighted_plane_pose_refinement_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(rows)), "usable_count": int(np.sum(finite)),
        "accepted_count": int(np.sum(accepted)),
        "median_translation_m": float(np.median(translation[finite])),
        "median_rotation_deg": float(np.median(rotation[finite])),
        "p90_translation_m": float(np.quantile(translation[finite], 0.9)),
        "p90_rotation_deg": float(np.quantile(rotation[finite], 0.9)),
        "recall_2m45": float(np.mean(finite & (translation <= 2.0) & (rotation <= 45.0))),
        "recall_1m10": float(np.mean(finite & (translation <= 1.0) & (rotation <= 10.0))),
        "recall_0p5m5": float(np.mean(finite & (translation <= 0.5) & (rotation <= 5.0))),
        "recall_0p25m2": float(np.mean(finite & (translation <= 0.25) & (rotation <= 2.0))),
        "recall_0p1m1": float(np.mean(finite & (translation <= 0.1) & (rotation <= 1.0))),
        "rows": rows, "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
