"""Select plane-pose hypotheses using propagated 3D covariance and plane purity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


BASE_REPROJECTION_SIGMA_PX = 4.0


def _load_pose_candidate(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load a frozen pose candidate without requiring selector-specific fields."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
    if (
        metadata.get("artifact_type") not in {
            "goal_maplet_dual_surface_geometry_consensus_selected_v1",
            "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
            "goal_maplet_probabilistic_surface_pose_refinement_v1",
            "goal_maplet_uncertainty_weighted_plane_pose_refinement_v1",
            "goal_maplet_cross_coordinate_probabilistic_surface_pose_refinement_v1",
            "goal_maplet_cross_coordinate_moge3_geometry_pose_selection_v1",
            "goal_maplet_coordinate_pose_geometry_consensus_v1",
            "goal_maplet_relative_multiplane_layout_selected_pose_v1",
        }
        or metadata.get("query_pose_or_ground_truth_read") is not False
        or metadata.get("source_rgb_stored_or_consumed_at_runtime") is not False
        or not {"names", "pose_w2c", "usable"}.issubset(arrays)
        or arrays_sha256(arrays) != metadata.get("arrays_sha256")
    ):
        raise ValueError("frozen pose candidate contract differs")
    return arrays, metadata


def _projection_variance_px2(
    camera_points: np.ndarray,
    covariance_world_m2: np.ndarray,
    rotation_w2c: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    radial_k1: float = 0.0,
) -> np.ndarray:
    """First-order SIMPLE_RADIAL propagation of world covariance to image variance."""
    xyz = np.asarray(camera_points, np.float64).reshape(-1, 3)
    covariance = np.asarray(covariance_world_m2, np.float64).reshape(-1, 3, 3)
    if len(xyz) != len(covariance):
        raise ValueError("point/covariance lengths differ")
    z = np.maximum(xyz[:, 2], 1e-9)
    x = xyz[:, 0] / z
    y = xyz[:, 1] / z
    k1 = float(radial_k1)
    radial = 1.0 + k1 * (np.square(x) + np.square(y))
    normalized_jacobian = np.zeros((len(xyz), 2, 3), np.float64)
    normalized_jacobian[:, 0, 0] = 1.0 / z
    normalized_jacobian[:, 0, 2] = -x / z
    normalized_jacobian[:, 1, 1] = 1.0 / z
    normalized_jacobian[:, 1, 2] = -y / z
    distortion_jacobian = np.empty((len(xyz), 2, 2), np.float64)
    distortion_jacobian[:, 0, 0] = float(camera_matrix[0, 0]) * (
        radial + 2.0 * k1 * np.square(x)
    )
    distortion_jacobian[:, 0, 1] = float(camera_matrix[0, 0]) * (2.0 * k1 * x * y)
    distortion_jacobian[:, 1, 0] = float(camera_matrix[1, 1]) * (2.0 * k1 * x * y)
    distortion_jacobian[:, 1, 1] = float(camera_matrix[1, 1]) * (
        radial + 2.0 * k1 * np.square(y)
    )
    J = distortion_jacobian @ normalized_jacobian
    camera_covariance = rotation_w2c[None] @ covariance @ rotation_w2c.T[None]
    projected_covariance = J @ camera_covariance @ np.swapaxes(J, 1, 2)
    return np.maximum(0.5 * np.trace(projected_covariance, axis1=1, axis2=2), 0.0)


def _uncertainty_normalized_token_likelihood(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    covariance_world_m2: np.ndarray,
    plane_purity: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    query_measurements_xy: np.ndarray | None = None,
    query_measurement_variance_px2: np.ndarray | None = None,
    correspondence_match_probability: np.ndarray | None = None,
    centroid_covariance_world_m2: np.ndarray | None = None,
    *,
    explicit_null_marginalization: bool = False,
    image_area_px2: float = 256.0 * 144.0,
) -> tuple[float, np.ndarray]:
    """Return a pose likelihood marginalized over hypotheses within each token.

    The legacy path is retained byte-for-byte for V3/V4 inventories.  V5 uses
    the mapping-only centroid variance and an explicit uniform-image null.  Its
    arithmetic mean within each token makes adding more hypotheses a
    redistribution of probability mass rather than an automatic score bonus.
    """
    world = np.asarray(world_points, np.float64).reshape(-1, 3)
    tokens = np.asarray(query_tokens, np.int64).reshape(-1)
    purity = np.asarray(plane_purity, np.float64).reshape(-1)
    if not (len(world) == len(tokens) == len(purity)):
        raise ValueError("uncertainty likelihood inputs differ in length")
    if not len(world):
        return 0.0, np.zeros(0, np.float64)
    camera = world @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose_w2c[:3, :3])[0], pose_w2c[:3, 3], camera_matrix,
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    pixel = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel) != len(tokens) or not np.all(np.isfinite(pixel)):
        raise ValueError("query measurements differ from tokens")
    residual2 = np.sum(np.square(projected.reshape(-1, 2) - pixel), axis=1)
    if query_measurement_variance_px2 is None:
        variance = BASE_REPROJECTION_SIGMA_PX ** 2 + _projection_variance_px2(
            camera, covariance_world_m2, pose_w2c[:3, :3], camera_matrix,
            radial_k1=radial_k1,
        )
    else:
        variance = np.asarray(query_measurement_variance_px2, np.float64).reshape(-1)
        if len(variance) != len(world) or np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
            raise ValueError("query measurement variance differs from hypotheses")
        if centroid_covariance_world_m2 is not None:
            centroid_covariance = np.asarray(
                centroid_covariance_world_m2, np.float64,
            ).reshape(-1, 3, 3)
            if len(centroid_covariance) != len(world) or not np.all(np.isfinite(centroid_covariance)):
                raise ValueError("centroid covariance differs from hypotheses")
            variance = variance + _projection_variance_px2(
                camera, centroid_covariance, pose_w2c[:3, :3], camera_matrix,
                radial_k1=radial_k1,
            )
    if explicit_null_marginalization:
        if correspondence_match_probability is None:
            raise ValueError("explicit null marginalization requires match probabilities")
        match = np.asarray(correspondence_match_probability, np.float64).reshape(-1)
        if len(match) != len(world) or np.any(~np.isfinite(match)) or np.any((match < 0.0) | (match > 1.0)):
            raise ValueError("correspondence match probability differs from hypotheses")
        # Purity is evidence that the atlas row really belongs to the selected
        # surface, so it modulates correspondence existence rather than
        # inflating an image-coordinate variance.
        match = np.clip(match * np.clip(purity, 0.0, 1.0), 0.0, 1.0)
        gaussian_density = np.exp(-0.5 * residual2 / variance) / (2.0 * np.pi * variance)
        gaussian_density[camera[:, 2] <= 0.0] = 0.0
        null_density = 1.0 / float(image_area_px2)
        row_likelihood = match * gaussian_density + (1.0 - match) * null_density
        unique, inverse = np.unique(tokens, return_inverse=True)
        token_likelihood = np.zeros(len(unique), np.float64)
        token_count = np.zeros(len(unique), np.int64)
        np.add.at(token_likelihood, inverse, row_likelihood)
        np.add.at(token_count, inverse, 1)
        token_likelihood /= np.maximum(token_count, 1)
        token_log_likelihood = np.log(np.maximum(token_likelihood, np.finfo(np.float64).tiny))
        return float(np.mean(token_log_likelihood)), token_log_likelihood
    # Relative Gaussian density: the determinant-like base/total variance term
    # prevents uncertain geometry from receiving a larger tolerance for free.
    likelihood = (
        np.clip(purity, 0.0, 1.0)
        * (BASE_REPROJECTION_SIGMA_PX ** 2 / variance)
        * np.exp(-0.5 * residual2 / variance)
    )
    likelihood[camera[:, 2] <= 0.0] = 0.0
    unique, inverse = np.unique(tokens, return_inverse=True)
    token_likelihood = np.zeros(len(unique), np.float64)
    np.maximum.at(token_likelihood, inverse, likelihood)
    return float(np.mean(token_likelihood)), token_likelihood


def _select(scores: np.ndarray) -> np.ndarray:
    value = np.asarray(scores, np.float64)
    if value.ndim != 3 or value.shape[1] < 2 or value.shape[2] != 2 or not np.all(np.isfinite(value)):
        raise ValueError("uncertainty scores must have shape (query,>=2,2)")
    return np.argmax(np.mean(value, axis=2), axis=1).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--uncertainty_correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite uncertainty-normalized selection")
    if len(args.pose_inventory) < 2:
        raise ValueError("at least two pose hypotheses are required")
    pose_items = [_load_pose_candidate(path) for path in args.pose_inventory]
    corr_items = [_load(path) for path in args.uncertainty_correspondences]
    names = pose_items[0][0]["names"].astype(str)
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in pose_items[1:]):
        raise ValueError("pose query order differs")
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in corr_items):
        raise ValueError("correspondence query order differs")
    artifact_types = [item[1].get("artifact_type") for item in corr_items]
    if any(value not in {
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
    } for value in artifact_types):
        raise ValueError("V3 through V7 uncertainty correspondences are required")
    probabilistic = all(value in {
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
    } for value in artifact_types)
    if any(value in {
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v6",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7",
    } for value in artifact_types) != probabilistic or (probabilistic and len(set(artifact_types)) != 1):
        raise ValueError("probabilistic and legacy or point/surface correspondence semantics cannot be mixed")
    atlas = [item[1].get("plane_uv_atlas_content_sha256") for item in corr_items]
    threshold = [float(item[1].get("homography_threshold_m", -1.0)) for item in corr_items]
    if atlas[0] != atlas[1] or len(set(threshold)) != 2:
        raise ValueError("uncertainty inventories must be two thresholds of one atlas")

    scores = np.full(
        (len(names), len(pose_items), 2), -1e12 if probabilistic else 0.0, np.float64,
    )
    for query in range(len(names)):
        for geometry, (corr, _) in enumerate(corr_items):
            lo, hi = map(int, corr["correspondence_offsets"][query : query + 2])
            for candidate, (pose, _) in enumerate(pose_items):
                if not bool(pose["usable"][query]):
                    continue
                scores[query, candidate, geometry], _ = _uncertainty_normalized_token_likelihood(
                    pose["pose_w2c"][query], corr["world_points"][lo:hi],
                    corr["query_tokens"][lo:hi], corr["prototype_world_covariance_m2"][lo:hi],
                    corr["prototype_plane_pixel_purity"][lo:hi], corr["camera_matrices"][query],
                    float(corr["radial_k1"][query]),
                    corr["query_measurements_xy"][lo:hi]
                    if "query_measurements_xy" in corr else None,
                    corr["query_measurement_variance_px2"][lo:hi]
                    if probabilistic else None,
                    corr["correspondence_match_probability"][lo:hi]
                    if probabilistic else None,
                    centroid_covariance_world_m2=(
                        corr["prototype_centroid_covariance_world_m2"][lo:hi]
                        if "prototype_centroid_covariance_world_m2" in corr else None
                    ),
                    explicit_null_marginalization=probabilistic,
                )
    selected = _select(scores)
    row = np.arange(len(names))
    arrays = {
        "names": names,
        "pose_w2c": np.stack([pose_items[int(branch)][0]["pose_w2c"][query] for query, branch in enumerate(selected)]),
        "usable": np.asarray([bool(pose_items[int(branch)][0]["usable"][query]) for query, branch in enumerate(selected)]),
        "selected_branch": selected,
        "selected_inlier_ratio": np.mean(scores, axis=2)[row, selected],
        "selected_candidate_correspondence_count": np.zeros(len(names), np.int64),
        "selected_pnp_inlier_count": np.zeros(len(names), np.int64),
        "cross_candidate_uncertainty_normalized_likelihood": scores,
    }
    metadata = {
        "artifact_type": (
            "goal_maplet_null_aware_marginalized_plane_pose_selected_v2"
            if probabilistic else "goal_maplet_uncertainty_normalized_plane_pose_selected_v1"
        ),
        "selection_rule": (
            "maximum_mean_log_mapping_centroid_gaussian_plus_uniform_null_across_loose_strict"
            if probabilistic else "maximum_mean_uncertainty_normalized_token_likelihood_across_loose_strict"
        ),
        "base_reprojection_sigma_px": None if probabilistic else BASE_REPROJECTION_SIGMA_PX,
        "covariance_projection": (
            "only_learned_centroid_covariance_with_full_SIMPLE_RADIAL_Jacobian_surface_footprint_excluded"
            if probabilistic else "first_order_SIMPLE_RADIAL_J_R_Cworld_RT_JT"
        ),
        "normalization": (
            "normalized_2D_Gaussian_density_using_mapping_predicted_centroid_variance"
            if probabilistic else "base_variance_over_total_variance_times_plane_purity"
        ),
        "hypothesis_marginalization": (
            "arithmetic_mean_probability_per_query_token_no_multiplicity_bonus"
            if probabilistic else "maximum_per_query_token"
        ),
        "null_model": (
            "mapping_only_match_probability_times_gaussian_plus_one_minus_match_times_uniform_image"
            if probabilistic else "implicit_zero_likelihood"
        ),
        "map_footprint_scatter_used_as_centroid_uncertainty": False if probabilistic else True,
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "selected_confidence_semantics": (
            "mean_log_normalized_probability_density_with_explicit_null"
            if probabilistic else "mean_uncertainty_normalized_token_likelihood_uncalibrated"
        ),
        "pose_inventory_file_sha256": [file_sha256(path) for path in args.pose_inventory],
        "pose_inventory_content_sha256": [item[1].get("content_sha256") for item in pose_items],
        "correspondence_file_sha256": [file_sha256(path) for path in args.uncertainty_correspondences],
        "correspondence_content_sha256": [item[1].get("content_sha256") for item in corr_items],
        "atlas_content_sha256": atlas[0],
        "homography_threshold_m": threshold,
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({**metadata, "selected_branch_counts": np.bincount(selected, minlength=len(pose_items)).tolist()}, indent=2))


if __name__ == "__main__":
    main()
