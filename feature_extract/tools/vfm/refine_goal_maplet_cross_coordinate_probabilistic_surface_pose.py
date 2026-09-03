"""Refine plane pose with paired point and continuous-surface coordinates.

The V5 point-coordinate and V7 continuous-surface inventories contain the
same anonymous RADIO token/prototype matches.  They differ only in the
mapping-only coordinate estimator and its calibrated uncertainty.  This tool
treats that coordinate choice as a fixed, equal-prior latent variable rather
than selecting between two already estimated poses.

For every query token, all prototype hypotheses from both coordinate arms are
marginalized together with the calibrated uniform-image null.  Three fixed EM
rounds alternate between match responsibilities and a bounded, plane-balanced
reprojection solve.  A candidate is accepted only when the joint likelihood
strictly improves, neither coordinate arm loses likelihood, and raw support is
preserved independently under both arms.  No query pose or ground truth is
opened until the frozen pose artifact has been written.
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
    _score as _raw_score,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_radio_plane_pnp import (
    _plane_balance_weights,
)
from feature_extract.tools.vfm.refine_goal_maplet_plane_pose_with_uncertainty import (
    MAXIMUM_CAMERA_CENTER_STEP_M,
    MAXIMUM_ROTATION_STEP_DEG,
    MINIMUM_GLOBAL_SUPPORT_FRACTION,
    _pose_step,
)
from feature_extract.tools.vfm.select_goal_maplet_cross_coordinate_surface_pose import (
    _paired_contract,
)
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
    _projection_variance_px2,
    _uncertainty_normalized_token_likelihood,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


EM_ROUNDS = 3
MINIMUM_ROWS = 12
MINIMUM_PLANES = 2
IMAGE_AREA_PX2 = 256.0 * 144.0
LIKELIHOOD_TOLERANCE = 1e-12


def _candidate_pose(initial_pose: np.ndarray, parameter: np.ndarray) -> np.ndarray:
    """Apply a left SE(3) increment to a frozen initial world-to-camera pose."""
    initial = np.asarray(initial_pose, np.float64).reshape(4, 4)
    value = np.asarray(parameter, np.float64).reshape(6)
    delta_rotation = Rotation.from_rotvec(value[:3]).as_matrix()
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = delta_rotation @ initial[:3, :3]
    output[:3, 3] = delta_rotation @ initial[:3, 3] + value[3:]
    return output


def _project(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, np.float64).reshape(4, 4)
    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected, _ = cv2.projectPoints(
        world,
        cv2.Rodrigues(pose[:3, :3])[0],
        pose[:3, 3],
        np.asarray(camera_matrix, np.float64),
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    return projected.reshape(-1, 2), camera


def _paired_query_hypotheses(
    point: dict[str, np.ndarray],
    surface: dict[str, np.ndarray],
    query: int,
) -> dict[str, np.ndarray]:
    """Stack equal-prior coordinate arms without changing match multiplicity."""
    point_lo, point_hi = map(int, point["correspondence_offsets"][query:query + 2])
    surface_lo, surface_hi = map(int, surface["correspondence_offsets"][query:query + 2])
    if point_hi - point_lo != surface_hi - surface_lo:
        raise ValueError("paired coordinate query row counts differ")
    slices = (slice(point_lo, point_hi), slice(surface_lo, surface_hi))
    items = (point, surface)
    world = np.concatenate([
        np.asarray(item["world_points"][slc], np.float64)
        for item, slc in zip(items, slices)
    ])
    pixel = np.concatenate([
        np.asarray(item["query_measurements_xy"][slc], np.float64)
        for item, slc in zip(items, slices)
    ])
    token = np.concatenate([
        np.asarray(item["query_tokens"][slc], np.int64)
        for item, slc in zip(items, slices)
    ])
    plane = np.concatenate([
        np.asarray(item["provenance_region_plane_atlas_row"][slc, 1], np.int64)
        for item, slc in zip(items, slices)
    ])
    existence = np.concatenate([
        np.asarray(item["correspondence_match_probability"][slc], np.float64)
        * np.clip(np.asarray(item["prototype_plane_pixel_purity"][slc], np.float64), 0.0, 1.0)
        for item, slc in zip(items, slices)
    ])
    query_variance = np.concatenate([
        np.asarray(item["query_measurement_variance_px2"][slc], np.float64)
        for item, slc in zip(items, slices)
    ])
    centroid_covariance = np.zeros((len(world), 3, 3), np.float64)
    arm_count = point_hi - point_lo
    centroid_covariance[arm_count:] = np.asarray(
        surface["prototype_centroid_covariance_world_m2"][surface_lo:surface_hi],
        np.float64,
    )
    if (
        np.any(~np.isfinite(world))
        or np.any(~np.isfinite(pixel))
        or np.any(~np.isfinite(query_variance))
        or np.any(query_variance <= 0.0)
        or np.any(~np.isfinite(centroid_covariance))
        or np.any(~np.isfinite(existence))
        or np.any((existence < 0.0) | (existence > 1.0))
    ):
        raise ValueError("paired coordinate hypothesis values differ")
    return {
        "world": world,
        "pixel": pixel,
        "token": token,
        "plane": plane,
        "existence_probability": existence,
        "query_variance_px2": query_variance,
        "centroid_covariance_world_m2": centroid_covariance,
        "coordinate_arm": np.concatenate([
            np.zeros(arm_count, np.int8), np.ones(arm_count, np.int8),
        ]),
    }


def _mixture_statistics(
    pose_w2c: np.ndarray,
    hypotheses: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    radial_k1: float,
    *,
    image_area_px2: float = IMAGE_AREA_PX2,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return proper token likelihood and matched-component responsibilities.

    Every token owns unit prior mass, split uniformly over its prototype and
    coordinate-arm components.  Each component then marginalizes its mapping-
    calibrated correspondence-existence probability against a uniform image
    null.  Thus duplicating an identical coordinate arm only splits the same
    prior mass and cannot improve the likelihood by multiplicity alone.
    """
    world = np.asarray(hypotheses["world"], np.float64).reshape(-1, 3)
    pixel = np.asarray(hypotheses["pixel"], np.float64).reshape(-1, 2)
    token = np.asarray(hypotheses["token"], np.int64).reshape(-1)
    existence = np.asarray(hypotheses["existence_probability"], np.float64).reshape(-1)
    query_variance = np.asarray(hypotheses["query_variance_px2"], np.float64).reshape(-1)
    centroid_covariance = np.asarray(
        hypotheses["centroid_covariance_world_m2"], np.float64,
    ).reshape(-1, 3, 3)
    if not (
        len(world) == len(pixel) == len(token) == len(existence)
        == len(query_variance) == len(centroid_covariance)
    ):
        raise ValueError("mixture hypothesis array lengths differ")
    if not len(world):
        return (
            float("-inf"), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
        )
    projected, camera = _project(pose_w2c, world, camera_matrix, radial_k1)
    residual = projected - pixel
    variance = query_variance + _projection_variance_px2(
        camera,
        centroid_covariance,
        np.asarray(pose_w2c, np.float64)[:3, :3],
        np.asarray(camera_matrix, np.float64),
        radial_k1=radial_k1,
    )
    variance = np.maximum(variance, 1e-12)
    residual2 = np.sum(np.square(residual), axis=1)
    gaussian = np.exp(-0.5 * residual2 / variance) / (2.0 * np.pi * variance)
    gaussian[camera[:, 2] <= 0.0] = 0.0
    unique, inverse, count = np.unique(token, return_inverse=True, return_counts=True)
    prior = 1.0 / count[inverse].astype(np.float64)
    null_density = 1.0 / float(image_area_px2)
    match_mass = prior * existence * gaussian
    null_mass = prior * (1.0 - existence) * null_density
    denominator = np.zeros(len(unique), np.float64)
    np.add.at(denominator, inverse, match_mass + null_mass)
    denominator = np.maximum(denominator, np.finfo(np.float64).tiny)
    responsibility = match_mass / denominator[inverse]
    token_null = np.zeros(len(unique), np.float64)
    np.add.at(token_null, inverse, null_mass)
    token_null /= denominator
    return (
        float(np.mean(np.log(denominator))),
        responsibility,
        token_null,
        residual,
        np.sqrt(variance),
    )


def _probabilistic_refine(
    initial_pose: np.ndarray,
    hypotheses: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    radial_k1: float,
) -> tuple[np.ndarray, bool, dict[str, object]]:
    world = np.asarray(hypotheses["world"], np.float64)
    pixel = np.asarray(hypotheses["pixel"], np.float64)
    plane = np.asarray(hypotheses["plane"], np.int64)
    token = np.asarray(hypotheses["token"], np.int64)
    diagnostics: dict[str, object] = {
        "coordinate_component_count": int(len(world)),
        "anonymous_match_count": int(len(world) // 2),
        "unique_token_count": int(len(np.unique(token))),
        "physical_plane_count": int(len(np.unique(plane))),
    }
    if len(world) < 2 * MINIMUM_ROWS or len(np.unique(plane)) < MINIMUM_PLANES:
        diagnostics["solver_attempted"] = False
        return np.asarray(initial_pose, np.float64).copy(), False, diagnostics
    initial = np.asarray(initial_pose, np.float64).reshape(4, 4)
    parameter = np.zeros(6, np.float64)
    rotation_bound = np.deg2rad(MAXIMUM_ROTATION_STEP_DEG)
    lower = np.asarray([-rotation_bound] * 3 + [-MAXIMUM_CAMERA_CENTER_STEP_M] * 3)
    upper = -lower
    balance = _plane_balance_weights(plane)
    likelihood_history: list[float] = []
    match_mass_history: list[float] = []
    null_history: list[float] = []
    for _ in range(EM_ROUNDS):
        pose = _candidate_pose(initial, parameter)
        likelihood, responsibility, null, _, sigma = _mixture_statistics(
            pose, hypotheses, camera_matrix, radial_k1,
        )
        likelihood_history.append(likelihood)
        match_mass_history.append(float(np.sum(responsibility)))
        null_history.append(float(np.mean(null)))
        fixed_weight = np.sqrt(np.maximum(responsibility, 0.0)) * balance / sigma

        def residual(candidate_parameter: np.ndarray) -> np.ndarray:
            candidate = _candidate_pose(initial, candidate_parameter)
            projected, _ = _project(candidate, world, camera_matrix, radial_k1)
            return ((projected - pixel) * fixed_weight[:, None]).reshape(-1)

        solution = least_squares(
            residual,
            parameter,
            bounds=(lower, upper),
            method="trf",
            loss="huber",
            f_scale=1.0,
            max_nfev=80,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if not solution.success or not np.all(np.isfinite(solution.x)):
            diagnostics.update({"solver_attempted": True, "solver_success": False})
            return initial.copy(), False, diagnostics
        parameter = np.asarray(solution.x, np.float64)
    output = _candidate_pose(initial, parameter)
    final_likelihood, final_responsibility, final_null, _, _ = _mixture_statistics(
        output, hypotheses, camera_matrix, radial_k1,
    )
    rotation_step, center_step = _pose_step(initial, output)
    diagnostics.update({
        "solver_attempted": True,
        "solver_success": True,
        "joint_log_likelihood_history": likelihood_history,
        "initial_joint_log_likelihood": float(likelihood_history[0]),
        "final_joint_log_likelihood": float(final_likelihood),
        "effective_match_mass_history": match_mass_history,
        "mean_null_responsibility_history": null_history,
        "final_effective_match_mass": float(np.sum(final_responsibility)),
        "final_mean_null_responsibility": float(np.mean(final_null)),
        "rotation_step_deg": rotation_step,
        "camera_center_step_m": center_step,
    })
    accepted = bool(
        final_likelihood > likelihood_history[0] + LIKELIHOOD_TOLERANCE
        and rotation_step <= MAXIMUM_ROTATION_STEP_DEG + 1e-8
        and center_step <= MAXIMUM_CAMERA_CENTER_STEP_M + 1e-8
    )
    return (output if accepted else initial.copy()), accepted, diagnostics


def _arm_likelihood(
    pose_w2c: np.ndarray,
    corr: dict[str, np.ndarray],
    query: int,
) -> float:
    lo, hi = map(int, corr["correspondence_offsets"][query:query + 2])
    value, _ = _uncertainty_normalized_token_likelihood(
        pose_w2c=pose_w2c,
        world_points=corr["world_points"][lo:hi],
        query_tokens=corr["query_tokens"][lo:hi],
        covariance_world_m2=corr["prototype_world_covariance_m2"][lo:hi],
        plane_purity=corr["prototype_plane_pixel_purity"][lo:hi],
        camera_matrix=corr["camera_matrices"][query],
        radial_k1=float(corr["radial_k1"][query]),
        query_measurements_xy=corr["query_measurements_xy"][lo:hi],
        query_measurement_variance_px2=corr["query_measurement_variance_px2"][lo:hi],
        correspondence_match_probability=corr["correspondence_match_probability"][lo:hi],
        centroid_covariance_world_m2=(
            corr["prototype_centroid_covariance_world_m2"][lo:hi]
            if "prototype_centroid_covariance_world_m2" in corr else None
        ),
        explicit_null_marginalization=True,
    )
    return float(value)


def _raw_support(
    pose_w2c: np.ndarray,
    corr: dict[str, np.ndarray],
    query: int,
) -> dict[str, object]:
    lo, hi = map(int, corr["correspondence_offsets"][query:query + 2])
    return _raw_score(
        pose_w2c,
        corr["world_points"][lo:hi],
        corr["query_tokens"][lo:hi],
        corr["provenance_region_plane_atlas_row"][lo:hi],
        corr["camera_matrices"][query],
        float(corr["radial_k1"][query]),
        corr["query_measurements_xy"][lo:hi],
    )


def _evaluate_pose(
    pose_w2c: np.ndarray,
    target_w2c: np.ndarray,
) -> tuple[float, float]:
    pose = np.asarray(pose_w2c, np.float64)
    target = np.asarray(target_w2c, np.float64)
    center = -pose[:3, :3].T @ pose[:3, 3]
    target_center = -target[:3, :3].T @ target[:3, 3]
    translation = float(np.linalg.norm(center - target_center))
    rotation = float(
        Rotation.from_matrix(pose[:3, :3] @ target[:3, :3].T).magnitude()
        * 180.0 / np.pi
    )
    return translation, rotation


def _threshold_counts(
    translation: np.ndarray,
    rotation: np.ndarray,
    finite: np.ndarray,
) -> dict[str, int]:
    return {
        "0.1m_1deg": int(np.sum(finite & (translation <= 0.1) & (rotation <= 1.0))),
        "0.25m_2deg": int(np.sum(finite & (translation <= 0.25) & (rotation <= 2.0))),
        "0.5m_5deg": int(np.sum(finite & (translation <= 0.5) & (rotation <= 5.0))),
        "1m_10deg": int(np.sum(finite & (translation <= 1.0) & (rotation <= 10.0))),
        "2m_45deg": int(np.sum(finite & (translation <= 2.0) & (rotation <= 45.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial_pose_inventory", type=Path, required=True)
    parser.add_argument("--point_correspondences", type=Path, required=True)
    parser.add_argument("--surface_correspondences", type=Path, required=True)
    parser.add_argument("--query_contributors", type=Path, required=True)
    parser.add_argument("--output_frozen_pose_inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_frozen_pose_inventory.exists():
        raise FileExistsError("refusing to overwrite cross-coordinate probabilistic pose")

    initial, initial_meta = _load_pose_candidate(args.initial_pose_inventory)
    point, point_meta = _load_correspondences(args.point_correspondences)
    surface, surface_meta = _load_correspondences(args.surface_correspondences)
    _paired_contract(point, point_meta, surface, surface_meta)
    names = point["names"].astype(str)
    if not (
        np.array_equal(names, surface["names"].astype(str))
        and np.array_equal(names, initial["names"].astype(str))
    ):
        raise ValueError("cross-coordinate probabilistic query order differs")

    initial_pose = np.asarray(initial["pose_w2c"], np.float64)
    output_pose = initial_pose.copy()
    usable = np.asarray(initial["usable"], bool).copy()
    accepted = np.zeros(len(names), bool)
    candidate_count = np.zeros(len(names), np.int64)
    diagnostics: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        detail: dict[str, object] = {"name": name, "usable": bool(usable[query])}
        if not usable[query]:
            diagnostics.append(detail)
            continue
        hypotheses = _paired_query_hypotheses(point, surface, query)
        candidate_count[query] = len(hypotheses["world"])
        K = np.asarray(point["camera_matrices"][query], np.float64)
        k1 = float(point["radial_k1"][query])
        candidate, solver_accept, solver_detail = _probabilistic_refine(
            initial_pose[query], hypotheses, K, k1,
        )
        initial_arm = np.asarray([
            _arm_likelihood(initial_pose[query], point, query),
            _arm_likelihood(initial_pose[query], surface, query),
        ])
        final_arm = np.asarray([
            _arm_likelihood(candidate, point, query),
            _arm_likelihood(candidate, surface, query),
        ])
        arm_pareto_pass = bool(np.all(final_arm >= initial_arm - LIKELIHOOD_TOLERANCE))
        initial_support = [_raw_support(initial_pose[query], item, query) for item in (point, surface)]
        final_support = [_raw_support(candidate, item, query) for item in (point, surface)]
        support_pass = all(
            int(after["inlier_count"]) >= max(
                6,
                int(np.floor(MINIMUM_GLOBAL_SUPPORT_FRACTION * int(before["inlier_count"]))),
            )
            for before, after in zip(initial_support, final_support)
        )
        use = bool(solver_accept and arm_pareto_pass and support_pass)
        if use:
            output_pose[query] = candidate
            accepted[query] = True
        detail.update({
            **solver_detail,
            "coordinate_arm_initial_log_likelihood": initial_arm.tolist(),
            "coordinate_arm_final_log_likelihood": final_arm.tolist(),
            "coordinate_arm_pareto_pass": arm_pareto_pass,
            "raw_support_acceptance_pass": support_pass,
            "initial_coordinate_arm_inlier_count": [
                int(value["inlier_count"]) for value in initial_support
            ],
            "final_coordinate_arm_inlier_count": [
                int(value["inlier_count"]) for value in final_support
            ],
            "refinement_accepted": use,
        })
        diagnostics.append(detail)

    arrays = {
        "names": initial["names"],
        "pose_w2c": output_pose,
        "usable": usable,
        "candidate_correspondence_count": candidate_count,
        "pnp_inlier_count": np.asarray(initial.get("pnp_inlier_count", np.zeros(len(names))), np.int64),
        "cross_coordinate_refinement_accepted": accepted,
    }
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_cross_coordinate_probabilistic_surface_pose_refinement_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "query_pose_or_ground_truth_read": False,
        "query_depth_or_scale_used_by_pose_solver": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "source_view_identity_retained_at_runtime": False,
        "coordinate_model": "equal_prior_latent_pointV5_or_continuous_surfaceV7_per_anonymous_RADIO_match",
        "coordinate_arm_prior": [0.5, 0.5],
        "hypothesis_prior": "unit_token_mass_uniform_over_coordinate_arms_and_matched_prototypes",
        "null_model": "mapping_calibrated_correspondence_existence_times_gaussian_plus_uniform_image_null",
        "map_uncertainty": "continuous_surface_centroid_covariance_only; atlas_footprint_scatter_and_depth_dispersion_excluded",
        "em_rounds": EM_ROUNDS,
        "acceptance": "strict_joint_likelihood_improvement_and_non_decreasing_each_coordinate_arm_likelihood_and_95pct_raw_support_each_arm",
        "minimum_rows": MINIMUM_ROWS,
        "minimum_planes": MINIMUM_PLANES,
        "minimum_global_support_fraction_each_arm": MINIMUM_GLOBAL_SUPPORT_FRACTION,
        "maximum_rotation_step_deg": MAXIMUM_ROTATION_STEP_DEG,
        "maximum_camera_center_step_m": MAXIMUM_CAMERA_CENTER_STEP_M,
        "initial_pose_inventory_file_sha256": file_sha256(args.initial_pose_inventory),
        "initial_pose_inventory_content_sha256": initial_meta.get("content_sha256"),
        "point_correspondence_file_sha256": file_sha256(args.point_correspondences),
        "point_correspondence_content_sha256": point_meta.get("content_sha256"),
        "surface_correspondence_file_sha256": file_sha256(args.surface_correspondences),
        "surface_correspondence_content_sha256": surface_meta.get("content_sha256"),
        "accepted_count": int(np.sum(accepted)),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output_frozen_pose_inventory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_frozen_pose_inventory,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    # Phase 2 begins only after the pose bytes and their content hash are sealed.
    initial_translation = np.full(len(names), np.inf, np.float64)
    initial_rotation = np.full(len(names), np.inf, np.float64)
    final_translation = np.full(len(names), np.inf, np.float64)
    final_rotation = np.full(len(names), np.inf, np.float64)
    rows: list[dict[str, object]] = []
    for query, name in enumerate(names.tolist()):
        with np.load(args.query_contributors / name, allow_pickle=False) as data:
            target = np.asarray(data["pose_w2c"], np.float64)
        initial_translation[query], initial_rotation[query] = _evaluate_pose(initial_pose[query], target)
        final_translation[query], final_rotation[query] = _evaluate_pose(output_pose[query], target)
        rows.append({
            **diagnostics[query],
            "initial_translation_error_m": float(initial_translation[query]),
            "initial_rotation_error_deg": float(initial_rotation[query]),
            "translation_error_m": float(final_translation[query]),
            "rotation_error_deg": float(final_rotation[query]),
        })
    finite = usable & np.isfinite(final_translation) & np.isfinite(final_rotation)
    initial_hits = _threshold_counts(initial_translation, initial_rotation, finite)
    final_hits = _threshold_counts(final_translation, final_rotation, finite)
    paired = {}
    thresholds = ((0.1, 1.0), (0.25, 2.0), (0.5, 5.0), (1.0, 10.0), (2.0, 45.0))
    for label, (translation_limit, rotation_limit) in zip(final_hits, thresholds):
        before = finite & (initial_translation <= translation_limit) & (initial_rotation <= rotation_limit)
        after = finite & (final_translation <= translation_limit) & (final_rotation <= rotation_limit)
        paired[label] = {
            "gains": int(np.sum(~before & after)),
            "losses": int(np.sum(before & ~after)),
            "net": int(np.sum(after) - np.sum(before)),
        }
    report: dict[str, object] = {
        "artifact_type": "goal_maplet_cross_coordinate_probabilistic_surface_pose_refinement_evaluation_v1",
        "pose_frozen_before_query_pose_or_ground_truth_open": True,
        "frozen_pose_inventory_file_sha256": file_sha256(args.output_frozen_pose_inventory),
        "frozen_pose_inventory_content_sha256": metadata["content_sha256"],
        "query_count": int(len(names)),
        "usable_count": int(np.sum(finite)),
        "accepted_count": int(np.sum(accepted)),
        "initial_median_translation_m": float(np.median(initial_translation[finite])),
        "initial_median_rotation_deg": float(np.median(initial_rotation[finite])),
        "initial_p90_translation_m": float(np.quantile(initial_translation[finite], 0.9)),
        "initial_p90_rotation_deg": float(np.quantile(initial_rotation[finite], 0.9)),
        "median_translation_m": float(np.median(final_translation[finite])),
        "median_rotation_deg": float(np.median(final_rotation[finite])),
        "p90_translation_m": float(np.quantile(final_translation[finite], 0.9)),
        "p90_rotation_deg": float(np.quantile(final_rotation[finite], 0.9)),
        "initial_threshold_hit_counts": initial_hits,
        "threshold_hit_counts": final_hits,
        "paired_threshold_changes": paired,
        "rows": rows,
        "production_eligible": False,
    }
    report["content_sha256"] = canonical_json_sha256(report)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
