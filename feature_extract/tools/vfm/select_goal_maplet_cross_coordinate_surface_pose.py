"""Select a frozen pose by cross-scoring it under point and surface coordinates.

The two coordinate inventories share the same anonymous RADIO matches but use
different mapping-only measurement models.  Every candidate is scored under
both inventories before either score is averaged, so a candidate cannot win
solely because it was optimized against its own coordinate realization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import _load
from feature_extract.tools.vfm.select_goal_maplet_uncertainty_normalized_plane_pose import (
    _load_pose_candidate,
    _uncertainty_normalized_token_likelihood,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


POINT_SCHEMA = "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v5"
SURFACE_SCHEMA = "goal_maplet_frozen_direct_plane_pnp_correspondence_inventory_v7"
LINE_SEARCH_ALPHAS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], np.float64)
FIXED_QUARTER_SURFACE_FRACTION = 0.25


def _interpolate_pose(point_pose: np.ndarray, surface_pose: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate camera center linearly and rotation on SO(3)."""
    first = np.asarray(point_pose, np.float64).reshape(4, 4)
    second = np.asarray(surface_pose, np.float64).reshape(4, 4)
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("pose interpolation fraction differs")
    relative = second[:3, :3] @ first[:3, :3].T
    rotation = Rotation.from_rotvec(
        value * Rotation.from_matrix(relative).as_rotvec(),
    ).as_matrix() @ first[:3, :3]
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    center = (1.0 - value) * first_center + value * second_center
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -rotation @ center
    return output


def _paired_contract(
    point: dict[str, np.ndarray], point_meta: dict[str, object],
    surface: dict[str, np.ndarray], surface_meta: dict[str, object],
) -> None:
    if point_meta.get("artifact_type") != POINT_SCHEMA or surface_meta.get("artifact_type") != SURFACE_SCHEMA:
        raise ValueError("cross-coordinate selector requires point V5 and cell-bounded surface V7 inventories")
    for key in (
        "names", "correspondence_offsets", "query_tokens",
        "provenance_region_plane_atlas_row", "prototype_atlas_row",
        "camera_matrices", "radial_k1", "radio_match_score",
    ):
        if not np.array_equal(point[key], surface[key]):
            raise ValueError(f"cross-coordinate correspondence pairing differs: {key}")
    for key in (
        "plane_uv_atlas_content_sha256", "plane_ranking_file_sha256",
        "query_camera_only_inventory_content_sha256", "homography_threshold_m",
        "hypotheses_per_query_token", "topk_planes", "query_support_policy",
    ):
        if point_meta.get(key) != surface_meta.get(key):
            raise ValueError(f"cross-coordinate metadata pairing differs: {key}")


def _select_pareto_line(scores: np.ndarray) -> np.ndarray:
    """Choose the best fixed line step that does not regress either evidence arm."""
    value = np.asarray(scores, np.float64)
    if value.ndim != 3 or value.shape[1] < 2 or value.shape[2] != 2:
        raise ValueError("cross-coordinate scores must have shape (query,>=2,2)")
    aggregate = np.mean(value, axis=2)
    selected = np.zeros(len(value), np.int8)
    for query in range(len(value)):
        baseline = value[query, 0]
        admissible = np.all(value[query] >= baseline[None, :], axis=1)
        admissible &= np.any(value[query] > baseline[None, :], axis=1)
        admissible[0] = True
        candidate = np.flatnonzero(admissible)
        selected[query] = int(candidate[np.argmax(aggregate[query, candidate])])
    return selected


def _score(
    pose: np.ndarray, corr: dict[str, np.ndarray], lo: int, hi: int, query: int,
) -> float:
    score, _ = _uncertainty_normalized_token_likelihood(
        pose_w2c=pose,
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
    return float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point_pose", type=Path, required=True)
    parser.add_argument("--surface_pose", type=Path, required=True)
    parser.add_argument("--point_correspondences", type=Path, required=True)
    parser.add_argument("--surface_correspondences", type=Path, required=True)
    parser.add_argument(
        "--selection_mode", choices=(
            "pareto_endpoints", "cross_likelihood_line_search", "pareto_line_search",
            "fixed_quarter_surface_update",
        ),
        default="pareto_endpoints",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite cross-coordinate pose selection")

    point_pose, point_pose_meta = _load_pose_candidate(args.point_pose)
    surface_pose, surface_pose_meta = _load_pose_candidate(args.surface_pose)
    point, point_meta = _load(args.point_correspondences)
    surface, surface_meta = _load(args.surface_correspondences)
    _paired_contract(point, point_meta, surface, surface_meta)
    names = point["names"].astype(str)
    if not (
        np.array_equal(names, surface["names"].astype(str))
        and np.array_equal(names, point_pose["names"].astype(str))
        and np.array_equal(names, surface_pose["names"].astype(str))
    ):
        raise ValueError("cross-coordinate pose query order differs")

    evidence = (point, surface)
    alpha = (
        np.asarray([FIXED_QUARTER_SURFACE_FRACTION], np.float64)
        if args.selection_mode == "fixed_quarter_surface_update"
        else LINE_SEARCH_ALPHAS
        if args.selection_mode in {"cross_likelihood_line_search", "pareto_line_search"}
        else np.asarray([0.0, 1.0])
    )
    candidate_poses = np.empty((len(names), len(alpha), 4, 4), np.float64)
    candidate_usable = np.zeros((len(names), len(alpha)), bool)
    scores = np.full((len(names), len(alpha), 2), -np.inf, np.float64)
    for query in range(len(names)):
        for candidate, fraction in enumerate(alpha.tolist()):
            candidate_usable[query, candidate] = bool(
                point_pose["usable"][query] and surface_pose["usable"][query]
            )
            candidate_poses[query, candidate] = _interpolate_pose(
                point_pose["pose_w2c"][query], surface_pose["pose_w2c"][query], fraction,
            )
        for coordinate, corr in enumerate(evidence):
            lo, hi = map(int, corr["correspondence_offsets"][query:query + 2])
            for candidate in range(len(alpha)):
                if candidate_usable[query, candidate]:
                    scores[query, candidate, coordinate] = _score(
                        candidate_poses[query, candidate], corr, lo, hi, query,
                    )
    aggregate = np.mean(scores, axis=2)
    # Surface refinement is optional and must improve under both the unchanged
    # point coordinate model and its own continuous coordinate model.  Exact
    # ties fall back to the stable point branch.  This Pareto rule has no
    # tunable weight and prevents one self-consistent evidence arm from
    # compensating a regression under the other arm.
    if args.selection_mode == "fixed_quarter_surface_update":
        selected = np.zeros(len(names), np.int8)
    elif args.selection_mode == "pareto_endpoints":
        selected = np.all(scores[:, 1, :] > scores[:, 0, :], axis=1).astype(np.int8)
    elif args.selection_mode == "pareto_line_search":
        selected = _select_pareto_line(scores)
    else:
        selected = np.argmax(aggregate, axis=1).astype(np.int8)
    row = np.arange(len(names))
    arrays = {
        "names": names,
        "pose_w2c": candidate_poses[row, selected],
        "usable": candidate_usable[row, selected],
        "selected_branch": selected,
        "selected_surface_step_fraction": alpha[selected],
        "cross_coordinate_log_likelihood": scores,
        "selected_mean_log_likelihood": aggregate[row, selected],
    }
    metadata: dict[str, object] = {
        "artifact_type": "goal_maplet_cross_coordinate_surface_pose_selection_v1",
        "selection_rule": (
            "fixed_one_quarter_bounded_SE3_surface_coordinate_correction_from_pointV5_toward_surfaceV7"
            if args.selection_mode == "fixed_quarter_surface_update"
            else "pareto_admissible_then_argmax_equal_mean_pointV5_cell_bounded_surfaceV7_contract_log_likelihood_on_fixed_SE3_line_search"
            if args.selection_mode == "pareto_line_search"
            else "argmax_equal_mean_pointV5_cell_bounded_surfaceV7_contract_log_likelihood_on_fixed_SE3_line_search"
            if args.selection_mode == "cross_likelihood_line_search"
            else "surface_only_if_strictly_better_under_both_pointV5_and_cell_bounded_surfaceV7_contract_log_likelihoods_else_point"
        ),
        "candidate_self_consistency_control": "every_candidate_scored_on_both_coordinate_inventories",
        "candidate_count": int(len(alpha)),
        "surface_step_fractions": alpha.tolist(),
        "coordinate_evidence_count": 2,
        "query_pose_or_ground_truth_read": False,
        "source_rgb_stored_or_consumed_at_runtime": False,
        "point_pose_file_sha256": file_sha256(args.point_pose),
        "point_pose_content_sha256": point_pose_meta.get("content_sha256"),
        "surface_pose_file_sha256": file_sha256(args.surface_pose),
        "surface_pose_content_sha256": surface_pose_meta.get("content_sha256"),
        "point_correspondence_file_sha256": file_sha256(args.point_correspondences),
        "point_correspondence_content_sha256": point_meta.get("content_sha256"),
        "surface_correspondence_file_sha256": file_sha256(args.surface_correspondences),
        "surface_correspondence_content_sha256": surface_meta.get("content_sha256"),
        "arrays_sha256": arrays_sha256(arrays),
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        **metadata,
        "selected_step_counts": np.bincount(selected, minlength=len(alpha)).tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
