"""Select plane-pose hypotheses using propagated 3D covariance and plane purity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.select_goal_maplet_cross_atlas_geometry_consensus import _load_selected
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256


BASE_REPROJECTION_SIGMA_PX = 4.0


def _projection_variance_px2(
    camera_points: np.ndarray,
    covariance_world_m2: np.ndarray,
    rotation_w2c: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """First-order pinhole propagation of world-point covariance to image variance."""
    xyz = np.asarray(camera_points, np.float64).reshape(-1, 3)
    covariance = np.asarray(covariance_world_m2, np.float64).reshape(-1, 3, 3)
    if len(xyz) != len(covariance):
        raise ValueError("point/covariance lengths differ")
    z = np.maximum(xyz[:, 2], 1e-9)
    J = np.zeros((len(xyz), 2, 3), np.float64)
    J[:, 0, 0] = float(camera_matrix[0, 0]) / z
    J[:, 0, 2] = -float(camera_matrix[0, 0]) * xyz[:, 0] / np.square(z)
    J[:, 1, 1] = float(camera_matrix[1, 1]) / z
    J[:, 1, 2] = -float(camera_matrix[1, 1]) * xyz[:, 1] / np.square(z)
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
) -> tuple[float, np.ndarray]:
    """Return a proper normalized likelihood, marginalized per query token."""
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
    variance = BASE_REPROJECTION_SIGMA_PX ** 2 + _projection_variance_px2(
        camera, covariance_world_m2, pose_w2c[:3, :3], camera_matrix,
    )
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
    pose_items = [_load_selected(path) for path in args.pose_inventory]
    corr_items = [_load(path) for path in args.uncertainty_correspondences]
    names = pose_items[0][0]["names"].astype(str)
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in pose_items[1:]):
        raise ValueError("pose query order differs")
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in corr_items):
        raise ValueError("correspondence query order differs")
    if any(item[1].get("artifact_type") not in {
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v3",
        "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v4",
    } for item in corr_items):
        raise ValueError("V3/V4 uncertainty correspondences are required")
    atlas = [item[1].get("plane_uv_atlas_content_sha256") for item in corr_items]
    threshold = [float(item[1].get("homography_threshold_m", -1.0)) for item in corr_items]
    if atlas[0] != atlas[1] or len(set(threshold)) != 2:
        raise ValueError("uncertainty inventories must be two thresholds of one atlas")

    scores = np.zeros((len(names), len(pose_items), 2), np.float64)
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
        "artifact_type": "goal_maplet_uncertainty_normalized_plane_pose_selected_v1",
        "selection_rule": "maximum_mean_uncertainty_normalized_token_likelihood_across_loose_strict",
        "base_reprojection_sigma_px": BASE_REPROJECTION_SIGMA_PX,
        "covariance_projection": "first_order_pinhole_J_R_Cworld_RT_JT_radial_only_in_residual",
        "normalization": "base_variance_over_total_variance_times_plane_purity",
        "hypothesis_marginalization": "maximum_per_query_token",
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "selected_confidence_semantics": "mean_uncertainty_normalized_token_likelihood_uncalibrated",
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
