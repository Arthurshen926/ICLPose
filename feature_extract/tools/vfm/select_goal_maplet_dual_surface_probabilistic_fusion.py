"""Fuse loose/strict chart matches with a soft, pose-free reprojection likelihood."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.select_goal_maplet_dual_surface_geometry_consensus import _poses
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


REPROJECTION_SIGMA_PX = 4.0


def _token_marginal_likelihood(
    pose_w2c: np.ndarray,
    world: np.ndarray,
    tokens: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    query_measurements_xy: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Marginalize mutually exclusive atlas hypotheses independently per query token."""
    if not len(world):
        return 0.0, np.zeros(0, np.float64)
    pixel = (
        np.c_[(tokens % 64) * 4 + 1.5, (tokens // 64) * 4 + 1.5]
        if query_measurements_xy is None
        else np.asarray(query_measurements_xy, np.float64).reshape(-1, 2)
    )
    if len(pixel) != len(tokens) or not np.all(np.isfinite(pixel)):
        raise ValueError("query measurements differ from tokens")
    camera = world @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    projected, _ = cv2.projectPoints(
        world, cv2.Rodrigues(pose_w2c[:3, :3])[0], pose_w2c[:3, 3],
        camera_matrix, np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = np.linalg.norm(projected.reshape(-1, 2) - pixel, axis=1)
    likelihood = np.exp(-0.5 * np.square(residual / REPROJECTION_SIGMA_PX))
    likelihood[camera[:, 2] <= 0.0] = 0.0
    unique, inverse = np.unique(tokens, return_inverse=True)
    token_likelihood = np.zeros(len(unique), np.float64)
    np.maximum.at(token_likelihood, inverse, likelihood)
    return float(np.mean(token_likelihood)), token_likelihood


def _select(score: np.ndarray) -> np.ndarray:
    value = np.asarray(score, np.float64)
    if value.ndim != 3 or value.shape[1:] != (2, 2) or not np.all(np.isfinite(value)):
        raise ValueError("cross-scale likelihood must have shape (query,2,2)")
    return np.argmax(np.mean(value, axis=2), axis=1).astype(np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_inventory", type=Path, nargs=2, required=True)
    parser.add_argument("--frozen_correspondences", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite probabilistic fusion")
    poses_and_meta = [_poses(path) for path in args.pose_inventory]
    corr_and_meta = [_load(path) for path in args.frozen_correspondences]
    names = poses_and_meta[0][0]["names"].astype(str)
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in poses_and_meta[1:]):
        raise ValueError("pose query order differs")
    if any(not np.array_equal(names, item[0]["names"].astype(str)) for item in corr_and_meta):
        raise ValueError("correspondence query order differs")

    scores = np.zeros((len(names), 2, 2), np.float64)
    for query in range(len(names)):
        for geometry, (corr, _) in enumerate(corr_and_meta):
            lo, hi = map(int, corr["correspondence_offsets"][query : query + 2])
            for candidate, (pose, _) in enumerate(poses_and_meta):
                if not bool(pose["usable"][query]):
                    continue
                scores[query, candidate, geometry], _ = _token_marginal_likelihood(
                    pose["pose_w2c"][query], corr["world_points"][lo:hi],
                    corr["query_tokens"][lo:hi], corr["camera_matrices"][query],
                    float(corr["radial_k1"][query]),
                    corr["query_measurements_xy"][lo:hi]
                    if "query_measurements_xy" in corr else None,
                )
    selected = _select(scores)
    row = np.arange(len(names))
    arrays = {
        "names": names,
        "pose_w2c": np.stack([
            poses_and_meta[int(branch)][0]["pose_w2c"][query]
            for query, branch in enumerate(selected)
        ]),
        "usable": np.asarray([
            bool(poses_and_meta[int(branch)][0]["usable"][query])
            for query, branch in enumerate(selected)
        ]),
        "selected_branch": selected,
        "selected_inlier_ratio": np.mean(scores, axis=2)[row, selected],
        "selected_candidate_correspondence_count": np.asarray([
            poses_and_meta[int(branch)][0]["candidate_correspondence_count"][query]
            for query, branch in enumerate(selected)
        ], np.int64),
        "selected_pnp_inlier_count": np.asarray([
            poses_and_meta[int(branch)][0]["pnp_inlier_count"][query]
            for query, branch in enumerate(selected)
        ], np.int64),
        "cross_geometry_token_marginal_likelihood": scores,
    }
    metadata = {
        "artifact_type": "goal_maplet_dual_surface_probabilistic_fusion_selected_v1",
        "selection_rule": "maximum_mean_token_marginal_gaussian_reprojection_likelihood",
        "reprojection_sigma_px": REPROJECTION_SIGMA_PX,
        "hypothesis_marginalization": "maximum_over_mutually_exclusive_atlas_hypotheses_per_query_token",
        "scale_fusion": "equal_mean_over_frozen_loose_and_strict_correspondence_inventories",
        "stable_tie_break": "branch_zero",
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "selected_confidence_semantics": "mean_cross_scale_token_marginal_likelihood_uncalibrated",
        "source_rgb_stored_or_consumed_at_runtime": False,
        "pose_inventory_file_sha256": [file_sha256(path) for path in args.pose_inventory],
        "pose_inventory_content_sha256": [item[1].get("content_sha256") for item in poses_and_meta],
        "correspondence_file_sha256": [file_sha256(path) for path in args.frozen_correspondences],
        "correspondence_content_sha256": [item[1].get("content_sha256") for item in corr_and_meta],
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps({**metadata, "selected_branch_counts": np.bincount(selected, minlength=2).tolist()}, indent=2))


if __name__ == "__main__":
    main()
