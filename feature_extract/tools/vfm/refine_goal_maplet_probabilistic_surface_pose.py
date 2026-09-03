"""Unify strict/loose chart hypotheses in a null-aware surface pose solve.

PnP/MoGe3 supplies only the initial pose.  Each RADIO token retains all unique
retrieved chart prototypes from the two frozen homography scales.  Duplicate
strict/loose rows use the stricter metric-UV covariance.  Three deterministic
EM rounds alternate between per-token hypothesis/null responsibilities and a
bounded, plane-balanced, uncertainty-weighted reprojection solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_correspondences,
    _score,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _plane_balance_weights,
    _token_pixels,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    MAXIMUM_CAMERA_CENTER_STEP_M,
    MAXIMUM_ROTATION_STEP_DEG,
    MINIMUM_GLOBAL_SUPPORT_FRACTION,
    TOKEN_QUANTIZATION_VARIANCE_PX2,
    _pose_step,
    _projection_variance_px2,
)
from feature_extract.tools.vfm.select_goal_maplet_cross_atlas_geometry_consensus import (
    _load_selected,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import (
    GeometryNativePlanarMap,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


EM_ROUNDS = 3
NULL_PRIOR = 0.10
MINIMUM_ROWS = 12
MINIMUM_PLANES = 2
MINIMUM_PURITY = 0.25
NULL_REPROJECTION_BOUNDARY_PX = 4.0


def _load_initial(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    if (
        metadata.get("artifact_type") not in (
            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
        )
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("source_rgb_stored_or_consumed_at_runtime") is not False
        or not {"names", "pose_w2c", "usable"}.issubset(arrays)
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("probabilistic surface initial pose contract differs")
    return arrays, metadata


def _merge_query_hypotheses(
    correlation: list[dict[str, np.ndarray]],
    metadata: list[dict[str, object]],
    query: int,
) -> dict[str, np.ndarray]:
    """Union rows and let the stricter scale own duplicate token/prototype pairs."""

    rows: list[dict[str, object]] = []
    for branch, (corr, meta) in enumerate(zip(correlation, metadata)):
        lo, hi = map(int, corr["correspondence_offsets"][query:query + 2])
        threshold = float(meta["homography_threshold_m"])
        measurement = (
            np.asarray(corr["query_measurements_xy"][lo:hi], np.float64)
            if "query_measurements_xy" in corr
            else _token_pixels(corr["query_tokens"][lo:hi], tuple(meta.get("token_grid", (36, 64))))
        )
        for local in range(hi - lo):
            index = lo + local
            rows.append({
                "token": int(corr["query_tokens"][index]),
                "prototype": int(corr["prototype_atlas_row"][index]),
                "world": np.asarray(corr["world_points"][index], np.float64),
                "pixel": np.asarray(measurement[local], np.float64),
                "provenance": np.asarray(corr["provenance_region_plane_atlas_row"][index], np.int64),
                "covariance": np.asarray(corr["prototype_world_covariance_m2"][index], np.float64),
                "purity": float(corr["prototype_plane_pixel_purity"][index]),
                "dispersion": float(corr["prototype_plane_depth_dispersion_m"][index]),
                "radio_score": float(corr["radio_match_score"][index]),
                "threshold": threshold,
                "branch": branch,
            })
    # Minimum threshold first; score and source row make the choice deterministic.
    rows.sort(key=lambda row: (
        row["token"], row["prototype"], row["threshold"], -row["radio_score"], row["branch"],
    ))
    unique: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["token"]), int(row["prototype"]))
        if key not in seen:
            seen.add(key); unique.append(row)
    if not unique:
        return {
            "token": np.zeros(0, np.int64), "prototype": np.zeros(0, np.int64),
            "world": np.zeros((0, 3)), "pixel": np.zeros((0, 2)),
            "provenance": np.zeros((0, 3), np.int64), "covariance": np.zeros((0, 3, 3)),
            "purity": np.zeros(0), "dispersion": np.zeros(0), "radio_score": np.zeros(0),
            "threshold": np.zeros(0), "source_branch": np.zeros(0, np.int8),
        }
    return {
        "token": np.asarray([row["token"] for row in unique], np.int64),
        "prototype": np.asarray([row["prototype"] for row in unique], np.int64),
        "world": np.asarray([row["world"] for row in unique], np.float64),
        "pixel": np.asarray([row["pixel"] for row in unique], np.float64),
        "provenance": np.asarray([row["provenance"] for row in unique], np.int64),
        "covariance": np.asarray([row["covariance"] for row in unique], np.float64),
        "purity": np.asarray([row["purity"] for row in unique], np.float64),
        "dispersion": np.asarray([row["dispersion"] for row in unique], np.float64),
        "radio_score": np.asarray([row["radio_score"] for row in unique], np.float64),
        "threshold": np.asarray([row["threshold"] for row in unique], np.float64),
        "source_branch": np.asarray([row["branch"] for row in unique], np.int8),
    }


def _surface_covariance(
    base_covariance: np.ndarray,
    plane_ids: np.ndarray,
    homography_threshold_m: np.ndarray,
    planar_map: GeometryNativePlanarMap,
) -> np.ndarray:
    covariance = np.asarray(base_covariance, np.float64).copy()
    plane = np.asarray(plane_ids, np.int64)
    scale2 = np.square(np.asarray(homography_threshold_m, np.float64)) / 12.0
    u = planar_map.frames_world[plane, 0]
    v = planar_map.frames_world[plane, 1]
    covariance += scale2[:, None, None] * (
        u[:, :, None] * u[:, None, :] + v[:, :, None] * v[:, None, :]
    )
    return covariance


def _candidate_pose(initial_pose: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    delta_rotation = Rotation.from_rotvec(np.asarray(parameter[:3], np.float64)).as_matrix()
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = delta_rotation @ initial_pose[:3, :3]
    output[:3, 3] = delta_rotation @ initial_pose[:3, 3] + parameter[3:6]
    return output


def _likelihood_and_residual(
    pose_w2c: np.ndarray,
    hypotheses: dict[str, np.ndarray],
    covariance_world: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = hypotheses["world"]
    pose = np.asarray(pose_w2c, np.float64)
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], camera_matrix,
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = projected.reshape(-1, 2) - hypotheses["pixel"]
    covariance_variance = _projection_variance_px2(
        camera, covariance_world, pose[:3, :3], camera_matrix,
    )
    focal = 0.5 * (float(camera_matrix[0, 0]) + float(camera_matrix[1, 1]))
    depth_variance = np.square(
        focal * np.maximum(hypotheses["dispersion"], 0.0)
        / np.maximum(camera[:, 2], 1e-9)
    )
    variance = (
        TOKEN_QUANTIZATION_VARIANCE_PX2
        + np.maximum(covariance_variance, depth_variance)
    ) / np.clip(hypotheses["purity"], MINIMUM_PURITY, 1.0)
    residual2 = np.sum(np.square(residual), axis=1)
    compatibility = (
        TOKEN_QUANTIZATION_VARIANCE_PX2 / np.maximum(variance, 1e-12)
        * np.exp(-0.5 * residual2 / np.maximum(variance, 1e-12))
    )
    compatibility[camera[:, 2] <= 0.0] = 0.0
    return compatibility, residual, np.sqrt(np.maximum(variance, 1e-12))


def _responsibilities(
    token: np.ndarray,
    compatibility: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize match mass per token, so extra hypotheses gain no free prior."""

    token = np.asarray(token, np.int64)
    value = np.asarray(compatibility, np.float64)
    unique, inverse, count = np.unique(token, return_inverse=True, return_counts=True)
    candidate_mass = (1.0 - NULL_PRIOR) * value / count[inverse]
    null_likelihood = float(np.exp(
        -0.5 * NULL_REPROJECTION_BOUNDARY_PX ** 2 / TOKEN_QUANTIZATION_VARIANCE_PX2
    ))
    denominator = np.full(len(unique), NULL_PRIOR * null_likelihood, np.float64)
    np.add.at(denominator, inverse, candidate_mass)
    responsibility = candidate_mass / np.maximum(denominator[inverse], 1e-300)
    null_responsibility = NULL_PRIOR * null_likelihood / np.maximum(denominator, 1e-300)
    nll = float(-np.mean(np.log(np.maximum(denominator, 1e-300))))
    return responsibility, null_responsibility, nll


def _probabilistic_refine(
    initial_pose: np.ndarray,
    hypotheses: dict[str, np.ndarray],
    covariance_world: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, bool, dict[str, object]]:
    world = hypotheses["world"]
    plane = hypotheses["provenance"][:, 1]
    diagnostics: dict[str, object] = {
        "unique_hypothesis_count": int(len(world)),
        "unique_token_count": int(len(np.unique(hypotheses["token"]))),
        "physical_plane_count": int(len(np.unique(plane))),
    }
    if len(world) < MINIMUM_ROWS or len(np.unique(plane)) < MINIMUM_PLANES:
        diagnostics["solver_attempted"] = False
        return np.asarray(initial_pose, np.float64).copy(), False, diagnostics
    initial = np.asarray(initial_pose, np.float64)
    parameter = np.zeros(6, np.float64)
    rotation_bound = np.deg2rad(MAXIMUM_ROTATION_STEP_DEG)
    lower = np.asarray([-rotation_bound] * 3 + [-MAXIMUM_CAMERA_CENTER_STEP_M] * 3)
    upper = -lower
    balance = _plane_balance_weights(plane)
    initial_nll: float | None = None
    effective_history: list[float] = []
    null_history: list[float] = []
    for _ in range(EM_ROUNDS):
        pose = _candidate_pose(initial, parameter)
        compatibility, _, sigma = _likelihood_and_residual(
            pose, hypotheses, covariance_world, camera_matrix, radial_k1,
        )
        responsibility, null, nll = _responsibilities(hypotheses["token"], compatibility)
        if initial_nll is None:
            initial_nll = nll
        effective_history.append(float(np.sum(responsibility)))
        null_history.append(float(np.mean(null)))
        fixed_weight = np.sqrt(np.maximum(responsibility, 0.0)) * balance / sigma

        def residual(candidate_parameter: np.ndarray) -> np.ndarray:
            candidate = _candidate_pose(initial, candidate_parameter)
            projected, _ = cv2.projectPoints(
                world, cv2.Rodrigues(candidate[:3, :3])[0], candidate[:3, 3], camera_matrix,
                np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
            )
            return ((projected.reshape(-1, 2) - hypotheses["pixel"])
                    * fixed_weight[:, None]).reshape(-1)

        solution = least_squares(
            residual, parameter, bounds=(lower, upper), method="trf",
            loss="huber", f_scale=1.0, max_nfev=80,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            diagnostics["solver_attempted"] = True
            diagnostics["solver_success"] = False
            return initial.copy(), False, diagnostics
        parameter = solution.x
    output = _candidate_pose(initial, parameter)
    final_compatibility, _, _ = _likelihood_and_residual(
        output, hypotheses, covariance_world, camera_matrix, radial_k1,
    )
    final_responsibility, final_null, final_nll = _responsibilities(
        hypotheses["token"], final_compatibility,
    )
    rotation_step, center_step = _pose_step(initial, output)
    diagnostics.update({
        "solver_attempted": True, "solver_success": True,
        "initial_negative_log_mixture": float(initial_nll),
        "final_negative_log_mixture": final_nll,
        "effective_match_mass_history": effective_history,
        "mean_null_responsibility_history": null_history,
        "final_effective_match_mass": float(np.sum(final_responsibility)),
        "final_mean_null_responsibility": float(np.mean(final_null)),
        "rotation_step_deg": rotation_step, "camera_center_step_m": center_step,
    })
    accepted = bool(
        final_nll < float(initial_nll) - 1e-9
        and rotation_step <= MAXIMUM_ROTATION_STEP_DEG + 1e-8
        and center_step <= MAXIMUM_CAMERA_CENTER_STEP_M + 1e-8
    )
    return (output if accepted else initial.copy()), accepted, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial_pose_inventory", type=Path, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite probabilistic surface pose")
    initial, initial_meta = _load_initial(args.initial_pose_inventory)
    corr_items = [_load_correspondences(path) for path in args.frozen_correspondences]
    corr = [item[0] for item in corr_items]; corr_meta = [item[1] for item in corr_items]
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    names = initial["names"].astype(str)
    if any(not np.array_equal(names, item["names"].astype(str)) for item in corr):
        raise ValueError("probabilistic surface query inventories differ")
    atlas = [meta.get("plane_uv_atlas_content_sha256") for meta in corr_meta]
    threshold = [float(meta.get("homography_threshold_m", -1.0)) for meta in corr_meta]
    if atlas[0] != atlas[1] or len(set(threshold)) != 2 or min(threshold) <= 0:
        raise ValueError("probabilistic solver requires two scales of one atlas")

    output_pose = np.asarray(initial["pose_w2c"], np.float64).copy()
    usable = np.asarray(initial["usable"], bool).copy()
    accepted = np.zeros(len(names), bool)
    hypothesis_count = np.zeros(len(names), np.int64)
    null_probability = np.ones(len(names), np.float64)
    selected_ratio = np.asarray(initial.get("selected_inlier_ratio", np.zeros(len(names))), np.float64).copy()
    selected_inlier = np.asarray(initial.get("selected_pnp_inlier_count", np.zeros(len(names))), np.int64).copy()
    diagnostics: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        row: dict[str, object] = {"name": name, "usable": bool(usable[query])}
        if not usable[query]:
            diagnostics.append(row); continue
        merged = _merge_query_hypotheses(corr, corr_meta, query)
        hypothesis_count[query] = len(merged["world"])
        covariance = _surface_covariance(
            merged["covariance"], merged["provenance"][:, 1], merged["threshold"], planes,
        )
        K = np.asarray(corr[0]["camera_matrices"][query], np.float64)
        k1 = float(corr[0]["radial_k1"][query])
        candidate, use, detail = _probabilistic_refine(
            output_pose[query], merged, covariance, K, k1,
        )
        initial_scores=[]; final_scores=[]
        for branch in range(2):
            lo, hi = map(int, corr[branch]["correspondence_offsets"][query:query + 2])
            measurement = corr[branch].get("query_measurements_xy")
            measurement = None if measurement is None else measurement[lo:hi]
            initial_scores.append(_score(
                output_pose[query], corr[branch]["world_points"][lo:hi],
                corr[branch]["query_tokens"][lo:hi], corr[branch]["provenance_region_plane_atlas_row"][lo:hi],
                corr[branch]["camera_matrices"][query], float(corr[branch]["radial_k1"][query]), measurement,
            ))
            final_scores.append(_score(
                candidate, corr[branch]["world_points"][lo:hi],
                corr[branch]["query_tokens"][lo:hi], corr[branch]["provenance_region_plane_atlas_row"][lo:hi],
                corr[branch]["camera_matrices"][query], float(corr[branch]["radial_k1"][query]), measurement,
            ))
        support_pass = all(
            int(final_score["inlier_count"]) >= max(
                6, int(np.floor(MINIMUM_GLOBAL_SUPPORT_FRACTION * int(initial_score["inlier_count"])))
            )
            for initial_score, final_score in zip(initial_scores, final_scores)
        )
        use = bool(use and support_pass)
        if use:
            output_pose[query] = candidate; accepted[query] = True
            selected_inlier[query] = int(round(np.mean([score["inlier_count"] for score in final_scores])))
            selected_ratio[query] = float(np.mean([score["inlier_ratio"] for score in final_scores]))
        if "final_mean_null_responsibility" in detail:
            null_probability[query] = float(detail["final_mean_null_responsibility"])
        row.update({
            **detail, "refinement_accepted": bool(use), "raw_support_acceptance_pass": support_pass,
            "initial_branch_inlier_count": [int(score["inlier_count"]) for score in initial_scores],
            "final_branch_inlier_count": [int(score["inlier_count"]) for score in final_scores],
        })
        diagnostics.append(row)

    arrays = {
        "names": initial["names"], "pose_w2c": output_pose, "usable": usable,
        "selected_branch": np.ones(len(names), np.int8),
        "selected_inlier_ratio": selected_ratio,
        "selected_candidate_correspondence_count": hypothesis_count,
        "selected_pnp_inlier_count": selected_inlier,
        "probabilistic_refinement_accepted": accepted,
        "unique_hypothesis_count": hypothesis_count,
        "mean_null_responsibility": null_probability,
    }
    metadata = {
        "artifact_type": "goal_maplet_probabilistic_surface_pose_refinement_v1",
        "arrays_sha256": arrays_sha256(arrays), "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "strict_loose_unification": "unique_token_prototype_union_stricter_duplicate_covariance",
        "homography_metric_uv_covariance": "uniform_tangent_square_threshold_squared_over_12",
        "hypothesis_prior": "fixed_total_match_mass_uniform_over_each_token_candidate_count",
        "null_prior": NULL_PRIOR,
        "null_likelihood": "base_token_gaussian_compatibility_at_frozen_4px_inlier_boundary",
        "em_rounds": EM_ROUNDS,
        "minimum_rows": MINIMUM_ROWS, "minimum_planes": MINIMUM_PLANES,
        "minimum_global_support_fraction_each_branch": MINIMUM_GLOBAL_SUPPORT_FRACTION,
        "maximum_rotation_step_deg": MAXIMUM_ROTATION_STEP_DEG,
        "maximum_camera_center_step_m": MAXIMUM_CAMERA_CENTER_STEP_M,
        "initial_pose_inventory_file_sha256": file_sha256(args.initial_pose_inventory),
        "initial_pose_inventory_content_sha256": initial_meta.get("content_sha256"),
        "frozen_correspondence_file_sha256": [file_sha256(path) for path in args.frozen_correspondences],
        "frozen_correspondence_content_sha256": [meta.get("content_sha256") for meta in corr_meta],
        "planar_map_file_sha256": file_sha256(args.planar_map),
        "accepted_count": int(np.sum(accepted)), "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    # Phase 2: only now open pose-bearing query contributor files.
    rows=[]; translation=[]; rotation=[]
    for query, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        pose = output_pose[query]
        center = -pose[:3, :3].T @ pose[:3, 3]
        target_center = -target[:3, :3].T @ target[:3, 3]
        t = float(np.linalg.norm(center - target_center))
        r = float(Rotation.from_matrix(pose[:3, :3] @ target[:3, :3].T).magnitude() * 180.0 / np.pi)
        translation.append(t); rotation.append(r)
        rows.append({**diagnostics[query], "translation_error_m": t, "rotation_error_deg": r})
    translation=np.asarray(translation); rotation=np.asarray(rotation)
    finite=usable & np.isfinite(translation) & np.isfinite(rotation)
    report = {
        "artifact_type": "goal_maplet_probabilistic_surface_pose_refinement_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)), "usable_count": int(np.sum(finite)),
        "accepted_count": int(np.sum(accepted)),
        "median_translation_m": float(np.median(translation[finite])),
        "median_rotation_deg": float(np.median(rotation[finite])),
        "threshold_hit_counts": {
            "0.1m_1deg": int(np.sum(finite & (translation <= .1) & (rotation <= 1))),
            "0.25m_2deg": int(np.sum(finite & (translation <= .25) & (rotation <= 2))),
            "0.5m_5deg": int(np.sum(finite & (translation <= .5) & (rotation <= 5))),
            "1m_10deg": int(np.sum(finite & (translation <= 1) & (rotation <= 10))),
            "2m_45deg": int(np.sum(finite & (translation <= 2) & (rotation <= 45))),
        },
        "rows": rows, "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
