"""Fuse frozen map-density PnP hypotheses using their union 2D--3D support.

Both input branches have already estimated poses without opening query pose
labels.  Each branch contributes at most one 3D hypothesis per query token.
The two poses are rescored against the union, where every image token votes at
most once, and are refined on the winning token-to-3D assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from feature_extract.tools.vfm.build_goal_maplet_direct_plane_pnp_render_consistency import (
    _load_frozen_poses,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_direct_plane_pnp_multihypothesis import (
    _load as _load_frozen_correspondences,
)
from feature_extract.vfm.localization_goal_maplet.lineage import (
    arrays_sha256,
    canonical_json_sha256,
    file_sha256,
)


def _merge_pose(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    shards: dict[str, list[np.ndarray]] = {}
    metadata = []
    for path in paths:
        arrays, meta = _load_frozen_poses(path)
        metadata.append(meta)
        for key, value in arrays.items():
            shards.setdefault(key, []).append(np.asarray(value))
    return {key: np.concatenate(value) for key, value in shards.items()}, metadata


def _merge_correspondence(
    paths: list[Path],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    queries: list[dict[str, np.ndarray]] = []
    metadata = []
    for path in paths:
        arrays, meta = _load_frozen_correspondences(path)
        metadata.append(meta)
        for index, name in enumerate(arrays["names"].astype(str).tolist()):
            lo, hi = map(int, arrays["correspondence_offsets"][index:index + 2])
            queries.append({
                "name": np.asarray(name),
                "world_points": arrays["world_points"][lo:hi],
                "query_tokens": arrays["query_tokens"][lo:hi],
                "provenance": arrays["provenance_region_plane_atlas_row"][lo:hi],
                "camera_matrix": arrays["camera_matrices"][index],
                "radial_k1": np.asarray(arrays["radial_k1"][index]),
            })
    return queries, metadata


def _token_pixels(tokens: np.ndarray, token_grid: tuple[int, int]) -> np.ndarray:
    height, width = map(int, token_grid)
    tokens = np.asarray(tokens, np.int64)
    return np.c_[
        (tokens % width + 0.5) * 256.0 / width - 0.5,
        (tokens // width + 0.5) * 144.0 / height - 0.5,
    ].astype(np.float64)


def _compose_union_support(
    correspondence_rows: list[dict[str, np.ndarray]],
    *,
    mode: str,
    consensus_radius_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if mode == "alternatives":
        return (
            np.concatenate([row["world_points"] for row in correspondence_rows], axis=0),
            np.concatenate([row["query_tokens"] for row in correspondence_rows], axis=0),
            0,
            0,
        )
    if mode != "cross_density_consensus_average":
        raise ValueError("union hypothesis mode differs")
    by_token: dict[int, list[np.ndarray]] = {}
    for row in correspondence_rows:
        tokens = np.asarray(row["query_tokens"], np.int64)
        if len(np.unique(tokens)) != len(tokens):
            raise ValueError("a map-density branch contains duplicate query-token hypotheses")
        for token, point in zip(tokens.tolist(), np.asarray(row["world_points"], np.float64)):
            by_token.setdefault(int(token), []).append(point)
    output_world = []
    output_tokens = []
    consensus_count = ambiguous_count = 0
    for token in sorted(by_token):
        points = np.asarray(by_token[token], np.float64)
        if len(points) >= 2:
            distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            if float(np.max(distance)) <= float(consensus_radius_m):
                output_world.append(np.mean(points, axis=0))
                output_tokens.append(token)
                consensus_count += 1
                continue
            ambiguous_count += 1
        output_world.extend(points)
        output_tokens.extend([token] * len(points))
    return (
        np.asarray(output_world, np.float64).reshape(-1, 3),
        np.asarray(output_tokens, np.int64),
        consensus_count,
        ambiguous_count,
    )


def _unique_token_inliers(
    pose_w2c: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    token_grid: tuple[int, int],
    maximum_reprojection_error_px: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(pose_w2c)) or not len(world_points):
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    rotation = np.asarray(pose_w2c[:3, :3], np.float64)
    translation = np.asarray(pose_w2c[:3, 3], np.float64)
    camera = np.asarray(world_points, np.float64) @ rotation.T + translation
    projected, _ = cv2.projectPoints(
        np.asarray(world_points, np.float64), cv2.Rodrigues(rotation)[0],
        translation, np.asarray(camera_matrix, np.float64),
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
    )
    residual = np.linalg.norm(
        projected.reshape(-1, 2) - _token_pixels(query_tokens, token_grid), axis=1,
    )
    valid = (camera[:, 2] > 0.0) & np.isfinite(residual) & (
        residual <= float(maximum_reprojection_error_px)
    )
    selected = []
    for token in np.unique(query_tokens):
        rows = np.flatnonzero((query_tokens == token) & valid)
        if len(rows):
            selected.append(int(rows[np.argmin(residual[rows])]))
    rows = np.asarray(selected, np.int64)
    return rows, residual[rows]


def _refine(
    pose_w2c: np.ndarray,
    rows: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    token_grid: tuple[int, int],
) -> np.ndarray:
    if len(rows) < 6:
        return np.asarray(pose_w2c, np.float64)
    rotation = np.asarray(pose_w2c[:3, :3], np.float64)
    translation = np.asarray(pose_w2c[:3, 3], np.float64)
    rvec = cv2.Rodrigues(rotation)[0]
    refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
        np.asarray(world_points[rows], np.float64),
        _token_pixels(query_tokens[rows], token_grid),
        np.asarray(camera_matrix, np.float64),
        np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64),
        rvec, translation.reshape(3, 1),
    )
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = cv2.Rodrigues(refined_rvec)[0]
    output[:3, 3] = refined_tvec.reshape(3)
    return output


def _robust_refine(
    pose_w2c: np.ndarray,
    rows: np.ndarray,
    world_points: np.ndarray,
    query_tokens: np.ndarray,
    camera_matrix: np.ndarray,
    radial_k1: float,
    token_grid: tuple[int, int],
    *,
    huber_scale_px: float = 2.0,
    spatial_balance: str = "none",
    spatial_group_labels: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Refine a fixed token assignment with a metric-free Huber reprojection loss."""
    if len(rows) < 6:
        return np.asarray(pose_w2c, np.float64)
    world = np.asarray(world_points[rows], np.float64)
    pixel = _token_pixels(query_tokens[rows], token_grid)
    K = np.asarray(camera_matrix, np.float64)
    distortion = np.asarray([radial_k1, 0.0, 0.0, 0.0, 0.0], np.float64)
    selected_groups = None
    if spatial_group_labels is not None:
        groups = np.asarray(spatial_group_labels)
        if groups.shape != (len(world_points),):
            raise ValueError("spatial group labels must align to correspondences")
        selected_groups = groups[rows]
    weights = _spatial_balance_weights(
        query_tokens[rows], token_grid, spatial_balance,
        group_labels=selected_groups,
    )
    if point_weights is not None:
        external = np.asarray(point_weights, np.float64)
        if external.shape != (len(world_points),) or not np.all(np.isfinite(external)):
            raise ValueError("point weights must be a finite vector aligned to correspondences")
        if np.any(external <= 0.0):
            raise ValueError("point weights must be positive")
        weights = weights * external[rows]
    initial = np.r_[
        cv2.Rodrigues(np.asarray(pose_w2c[:3, :3], np.float64))[0].reshape(3),
        np.asarray(pose_w2c[:3, 3], np.float64),
    ]

    def residual(parameter: np.ndarray) -> np.ndarray:
        projected, _ = cv2.projectPoints(
            world, parameter[:3], parameter[3:], K, distortion,
        )
        return ((projected.reshape(-1, 2) - pixel) * weights[:, None]).reshape(-1)

    solution = least_squares(
        residual,
        initial,
        method="trf",
        loss="huber",
        f_scale=float(huber_scale_px),
        max_nfev=100,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        return np.asarray(pose_w2c, np.float64)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = cv2.Rodrigues(solution.x[:3])[0]
    output[:3, 3] = solution.x[3:]
    return output


def _spatial_balance_weights(
    query_tokens: np.ndarray,
    token_grid: tuple[int, int],
    mode: str,
    *,
    group_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Bounded equal-mass weighting over fixed image macrocells.

    The optional half-cell phases provide deterministic staggered partitions.
    They do not add evidence or inspect pose labels; they only prevent a dense
    match island that happens to straddle one fixed grid boundary from receiving
    a materially different weight than the same island shifted by one token.
    Border cells are intentionally clipped and all final weights remain bounded.
    """
    tokens = np.asarray(query_tokens, np.int64)
    if mode == "none":
        return np.ones(len(tokens), np.float64)
    combined_modes = {
        "query_region_x_macrocell_4x6": "macrocell_equal_mass_4x6",
        "query_region_x_macrocell_6x10": "macrocell_equal_mass_6x10",
    }
    if mode in combined_modes:
        region = _spatial_balance_weights(
            tokens, token_grid, "query_region_equal_mass",
            group_labels=group_labels,
        )
        macro = _spatial_balance_weights(tokens, token_grid, combined_modes[mode])
        return np.clip(np.sqrt(region * macro), 0.5, 2.0)
    if mode == "query_region_equal_mass":
        if group_labels is None:
            raise ValueError("query-region balance requires group labels")
        labels = np.asarray(group_labels, np.int64)
        if labels.shape != tokens.shape or np.any(labels < 0):
            raise ValueError("query-region balance labels differ")
        if not len(labels):
            return np.ones(0, np.float64)
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        mean_count = float(len(labels)) / float(len(counts))
        weights = np.sqrt(mean_count / counts[inverse].astype(np.float64))
        return np.clip(weights, 0.5, 2.0)
    if mode == "local_density_radius2":
        height, width = map(int, token_grid)
        if np.any(tokens < 0) or np.any(tokens >= height * width):
            raise ValueError("query token lies outside token grid")
        if not len(tokens):
            return np.ones(0, np.float64)
        y = tokens // width
        x = tokens % width
        # Count occupied matched tokens in a fixed 5x5 Chebyshev window.  The
        # square-root inverse density is deliberately bounded: it reduces the
        # leverage of a compact foreground/texture cluster without discarding
        # it or allowing a single isolated token to dominate PnP.
        local_count = np.sum(
            (np.abs(y[:, None] - y[None, :]) <= 2)
            & (np.abs(x[:, None] - x[None, :]) <= 2),
            axis=1,
        ).astype(np.float64)
        reference = float(np.median(local_count))
        weights = np.sqrt(reference / local_count)
        return np.clip(weights, 0.5, 2.0)
    macrocell_shapes = {
        "macrocell_equal_mass_3x5": (3, 5),
        "macrocell_equal_mass_4x6": (4, 6),
        "macrocell_equal_mass_4x6_shift_x": (4, 6),
        "macrocell_equal_mass_4x6_shift_y": (4, 6),
        "macrocell_equal_mass_4x6_shift_xy": (4, 6),
        "macrocell_equal_mass_5x8": (5, 8),
        "macrocell_equal_mass_6x10": (6, 10),
        "macrocell_equal_mass_6x10_shift_xy": (6, 10),
        "macrocell_equal_mass_8x12": (8, 12),
    }
    if mode not in macrocell_shapes:
        raise ValueError("spatial balance mode differs")
    height, width = map(int, token_grid)
    if np.any(tokens < 0) or np.any(tokens >= height * width):
        raise ValueError("query token lies outside token grid")
    rows, columns = macrocell_shapes[mode]
    phase_y = 0.5 if mode.endswith(("shift_y", "shift_xy")) else 0.0
    phase_x = 0.5 if mode.endswith(("shift_x", "shift_xy")) else 0.0
    macro_y = np.clip(
        np.floor((tokens // width) * rows / height + phase_y).astype(np.int64),
        0, rows - 1,
    )
    macro_x = np.clip(
        np.floor((tokens % width) * columns / width + phase_x).astype(np.int64),
        0, columns - 1,
    )
    macro = macro_y * columns + macro_x
    counts = np.bincount(macro, minlength=rows * columns).astype(np.float64)
    occupied = counts[counts > 0.0]
    if not len(occupied):
        return np.ones(0, np.float64)
    mean_count = float(len(tokens)) / float(len(occupied))
    weights = np.sqrt(mean_count / counts[macro])
    return np.clip(weights, 0.5, 2.0)


def _candidate(
    pose: np.ndarray,
    world: np.ndarray,
    tokens: np.ndarray,
    K: np.ndarray,
    k1: float,
    token_grid: tuple[int, int],
    refinement_mode: str,
    spatial_balance: str = "none",
) -> dict[str, object]:
    rows, residual = _unique_token_inliers(pose, world, tokens, K, k1, token_grid)
    if refinement_mode == "lm":
        refine = _refine
        def apply_refine(current: np.ndarray, selected_rows: np.ndarray) -> np.ndarray:
            return refine(current, selected_rows, world, tokens, K, k1, token_grid)
    else:
        def apply_refine(current: np.ndarray, selected_rows: np.ndarray) -> np.ndarray:
            return _robust_refine(
                current, selected_rows, world, tokens, K, k1, token_grid,
                spatial_balance=spatial_balance,
            )
    refined = apply_refine(pose, rows)
    final_rows, final_residual = _unique_token_inliers(refined, world, tokens, K, k1, token_grid)
    # One more fixed-assignment LM pass is deterministic and removes the small
    # discontinuity introduced when the best 3D hypothesis changes per token.
    refined = apply_refine(refined, final_rows)
    final_rows, final_residual = _unique_token_inliers(
        refined, world, tokens, K, k1, token_grid,
    )
    return {
        "pose": refined,
        "rows": final_rows,
        "inlier_count": int(len(final_rows)),
        "median_residual": (
            float(np.median(final_residual)) if len(final_residual) else float("inf")
        ),
    }


def _midpoint_pose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_matrix(np.asarray([left[:3, :3], right[:3, :3]]))
    midpoint_rotation = Slerp([0.0, 1.0], rotations)([0.5]).as_matrix()[0]
    left_center = -left[:3, :3].T @ left[:3, 3]
    right_center = -right[:3, :3].T @ right[:3, 3]
    midpoint_center = 0.5 * (left_center + right_center)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = midpoint_rotation
    output[:3, 3] = -midpoint_rotation @ midpoint_center
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--dense_pose_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--sparse_correspondence_inventory", type=Path, nargs="+", required=True)
    parser.add_argument("--dense_correspondence_inventory", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--supplemental_correspondence_inventory", type=Path, nargs="*", default=[],
        help="Extra pose-free hypotheses used only during union-support refinement.",
    )
    parser.add_argument("--auxiliary_pose_inventory", type=Path, nargs="*", default=[])
    parser.add_argument("--auxiliary_correspondence_inventory", type=Path, nargs="*", default=[])
    parser.add_argument("--auxiliary_view_count", type=int)
    parser.add_argument("--sparse_view_count", type=int, required=True)
    parser.add_argument("--dense_view_count", type=int, required=True)
    parser.add_argument("--refinement_mode", choices=("lm", "robust_huber"), default="lm")
    parser.add_argument(
        "--robust_spatial_balance",
        choices=(
            "none", "macrocell_equal_mass_3x5", "macrocell_equal_mass_4x6",
            "macrocell_equal_mass_5x8", "macrocell_equal_mass_6x10",
            "macrocell_equal_mass_8x12", "local_density_radius2",
        ),
        default="none",
    )
    parser.add_argument(
        "--union_hypothesis_mode",
        choices=("alternatives", "cross_density_consensus_average"),
        default="alternatives",
    )
    parser.add_argument(
        "--consistent_pose_fusion", choices=("none", "se3_midpoint"), default="none",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite fused map-density poses")
    if not 0 < int(args.sparse_view_count) < int(args.dense_view_count):
        raise ValueError("map-density view counts differ")
    counts = {
        len(args.sparse_pose_inventory), len(args.dense_pose_inventory),
        len(args.sparse_correspondence_inventory), len(args.dense_correspondence_inventory),
    }
    if args.supplemental_correspondence_inventory:
        counts.add(len(args.supplemental_correspondence_inventory))
    if len(counts) != 1:
        raise ValueError("map-density shard counts differ")
    auxiliary_enabled = bool(args.auxiliary_pose_inventory or args.auxiliary_correspondence_inventory)
    if auxiliary_enabled and (
        args.auxiliary_view_count is None
        or len(args.auxiliary_pose_inventory) != next(iter(counts))
        or len(args.auxiliary_correspondence_inventory) != next(iter(counts))
        or not 0 < int(args.auxiliary_view_count) < int(args.sparse_view_count)
    ):
        raise ValueError("auxiliary map-density branch differs")

    sparse_pose, sparse_pose_meta = _merge_pose(args.sparse_pose_inventory)
    dense_pose, dense_pose_meta = _merge_pose(args.dense_pose_inventory)
    sparse_corr, sparse_corr_meta = _merge_correspondence(args.sparse_correspondence_inventory)
    dense_corr, dense_corr_meta = _merge_correspondence(args.dense_correspondence_inventory)
    supplemental_corr = supplemental_corr_meta = None
    if args.supplemental_correspondence_inventory:
        supplemental_corr, supplemental_corr_meta = _merge_correspondence(
            args.supplemental_correspondence_inventory
        )
    auxiliary_pose = auxiliary_pose_meta = auxiliary_corr = auxiliary_corr_meta = None
    if auxiliary_enabled:
        auxiliary_pose, auxiliary_pose_meta = _merge_pose(args.auxiliary_pose_inventory)
        auxiliary_corr, auxiliary_corr_meta = _merge_correspondence(
            args.auxiliary_correspondence_inventory
        )
    names = sparse_pose["names"].astype(str)
    inventories = [
        dense_pose["names"].astype(str),
        np.asarray([str(row["name"].item()) for row in sparse_corr]),
        np.asarray([str(row["name"].item()) for row in dense_corr]),
    ]
    if auxiliary_enabled:
        inventories.extend([
            auxiliary_pose["names"].astype(str),
            np.asarray([str(row["name"].item()) for row in auxiliary_corr]),
        ])
    if supplemental_corr is not None:
        inventories.append(np.asarray([
            str(row["name"].item()) for row in supplemental_corr
        ]))
    if any(not np.array_equal(names, other) for other in inventories):
        raise ValueError("map-density query inventories differ")
    if len(set(names.tolist())) != len(names):
        raise ValueError("map-density query names are duplicated")
    camera_hashes = {
        str(meta.get("query_camera_only_inventory_file_sha256"))
        for meta in (
            sparse_pose_meta + dense_pose_meta + sparse_corr_meta + dense_corr_meta
            + ([] if supplemental_corr_meta is None else supplemental_corr_meta)
            + ([] if auxiliary_pose_meta is None else auxiliary_pose_meta)
            + ([] if auxiliary_corr_meta is None else auxiliary_corr_meta)
        )
    }
    if len(camera_hashes) != 1 or "None" in camera_hashes:
        raise ValueError("map-density camera lineage differs")
    token_grids = {
        tuple(map(int, meta.get("token_grid", ())))
        for meta in (
            sparse_corr_meta + dense_corr_meta
            + ([] if supplemental_corr_meta is None else supplemental_corr_meta)
            + ([] if auxiliary_corr_meta is None else auxiliary_corr_meta)
        )
    }
    if len(token_grids) != 1:
        raise ValueError("map-density token grids differ")
    token_grid = next(iter(token_grids))

    output_pose = []
    output_usable = []
    output_branch = []
    output_ratio = []
    output_candidates = []
    output_inliers = []
    sparse_ratio = []
    dense_ratio = []
    consensus_token_count = ambiguous_multibranch_token_count = 0
    midpoint_fusion_count = 0
    for index in range(len(names)):
        left, right = sparse_corr[index], dense_corr[index]
        correspondence_rows = [left, right]
        if supplemental_corr is not None:
            correspondence_rows.append(supplemental_corr[index])
        if auxiliary_enabled:
            correspondence_rows.insert(0, auxiliary_corr[index])
        if any(
            not np.allclose(left["camera_matrix"], row["camera_matrix"], atol=0.0, rtol=0.0)
            or float(left["radial_k1"]) != float(row["radial_k1"])
            for row in correspondence_rows[1:]
        ):
            raise ValueError("map-density query cameras differ")
        world, tokens, consensus_count, ambiguous_count = _compose_union_support(
            correspondence_rows,
            mode=str(args.union_hypothesis_mode),
            consensus_radius_m=0.5,
        )
        consensus_token_count += int(consensus_count)
        ambiguous_multibranch_token_count += int(ambiguous_count)
        denominator = int(len(np.unique(tokens)))
        candidates = []
        branches = [
            (int(args.sparse_view_count), sparse_pose["pose_w2c"][index], sparse_pose["usable"][index]),
            (int(args.dense_view_count), dense_pose["pose_w2c"][index], dense_pose["usable"][index]),
        ]
        if auxiliary_enabled:
            branches.insert(0, (
                int(args.auxiliary_view_count), auxiliary_pose["pose_w2c"][index],
                auxiliary_pose["usable"][index],
            ))
        for branch, pose, usable in branches:
            candidate = (
                _candidate(
                    pose, world, tokens, left["camera_matrix"], float(left["radial_k1"]),
                    token_grid, str(args.refinement_mode), str(args.robust_spatial_balance),
                )
                if bool(usable) else
                {"pose": np.full((4, 4), np.nan), "inlier_count": 0,
                 "median_residual": float("inf")}
            )
            candidate["branch"] = branch
            candidates.append(candidate)
        candidate_by_branch = {int(row["branch"]): row for row in candidates}
        if str(args.consistent_pose_fusion) == "se3_midpoint":
            sparse_candidate = candidate_by_branch[int(args.sparse_view_count)]
            dense_candidate = candidate_by_branch[int(args.dense_view_count)]
            if (
                int(sparse_candidate["inlier_count"]) >= 6
                and int(dense_candidate["inlier_count"]) >= 6
            ):
                sparse_candidate_pose = np.asarray(sparse_candidate["pose"], np.float64)
                dense_candidate_pose = np.asarray(dense_candidate["pose"], np.float64)
                sparse_center = -sparse_candidate_pose[:3, :3].T @ sparse_candidate_pose[:3, 3]
                dense_center = -dense_candidate_pose[:3, :3].T @ dense_candidate_pose[:3, 3]
                center_distance = float(np.linalg.norm(sparse_center - dense_center))
                rotation_distance = float(
                    Rotation.from_matrix(
                        sparse_candidate_pose[:3, :3] @ dense_candidate_pose[:3, :3].T
                    ).magnitude() * 180.0 / np.pi
                )
                if center_distance <= 0.5 and rotation_distance <= 5.0:
                    midpoint = _midpoint_pose(sparse_candidate_pose, dense_candidate_pose)
                    midpoint_rows, midpoint_residual = _unique_token_inliers(
                        midpoint, world, tokens, left["camera_matrix"],
                        float(left["radial_k1"]), token_grid,
                    )
                    candidates.append({
                        "pose": midpoint,
                        "rows": midpoint_rows,
                        "inlier_count": int(len(midpoint_rows)),
                        "median_residual": (
                            float(np.median(midpoint_residual))
                            if len(midpoint_residual) else float("inf")
                        ),
                        "branch": int((args.sparse_view_count + args.dense_view_count) // 2),
                    })
                    midpoint_fusion_count += 1
        sparse_ratio.append(
            float(candidate_by_branch[int(args.sparse_view_count)]["inlier_count"])
            / max(denominator, 1)
        )
        dense_ratio.append(
            float(candidate_by_branch[int(args.dense_view_count)]["inlier_count"])
            / max(denominator, 1)
        )
        # Ties retain the lower-density branch.
        best = max(
            candidates,
            key=lambda row: (
                int(row["inlier_count"]), -float(row["median_residual"]),
                -int(row["branch"]),
            ),
        )
        usable = int(best["inlier_count"]) >= 6 and np.isfinite(best["pose"]).all()
        output_pose.append(best["pose"] if usable else np.full((4, 4), np.nan))
        output_usable.append(usable)
        output_branch.append(best["branch"])
        output_ratio.append(float(best["inlier_count"]) / max(denominator, 1))
        output_candidates.append(denominator)
        output_inliers.append(best["inlier_count"])

    arrays = {
        "names": names,
        "pose_w2c": np.asarray(output_pose, np.float64),
        "usable": np.asarray(output_usable, bool),
        "selected_branch": np.asarray(output_branch, np.int16),
        "selected_inlier_ratio": np.asarray(output_ratio, np.float64),
        "selected_candidate_correspondence_count": np.asarray(output_candidates, np.int64),
        "selected_pnp_inlier_count": np.asarray(output_inliers, np.int64),
        "sparse_inlier_ratio": np.asarray(sparse_ratio, np.float64),
        "dense_inlier_ratio": np.asarray(dense_ratio, np.float64),
    }
    metadata = {
        "artifact_type": "goal_maplet_direct_plane_pnp_map_density_inlier_selected_v1",
        "arrays_sha256": arrays_sha256(arrays),
        "query_count": int(len(names)),
        "sparse_view_count": int(args.sparse_view_count),
        "dense_view_count": int(args.dense_view_count),
        "selection_rule": "maximum_unique_token_inliers_on_union_support_then_residual_then_lower_density",
        "union_support_pose_refinement": True,
        "refinement_mode": str(args.refinement_mode),
        "robust_huber_scale_px": 2.0 if args.refinement_mode == "robust_huber" else None,
        "robust_spatial_balance": str(args.robust_spatial_balance),
        "alternating_hypothesis_assignment_rounds": 2,
        "union_hypothesis_mode": str(args.union_hypothesis_mode),
        "cross_density_consensus_radius_m": 0.5,
        "cross_density_consensus_token_count": int(consensus_token_count),
        "cross_density_ambiguous_multibranch_token_count": int(
            ambiguous_multibranch_token_count
        ),
        "consistent_pose_fusion": str(args.consistent_pose_fusion),
        "consistent_pose_translation_limit_m": 0.5,
        "consistent_pose_rotation_limit_deg": 5.0,
        "midpoint_fusion_candidate_count": int(midpoint_fusion_count),
        "query_pose_or_ground_truth_read": False,
        "strict_runtime_phase_separation_eligible": True,
        "query_depth_or_scale_used_by_pose_solver": False,
        "query_camera_only_inventory_file_sha256": next(iter(camera_hashes)),
        "token_grid": list(token_grid),
        "sparse_pose_file_sha256_in_order": [file_sha256(path) for path in args.sparse_pose_inventory],
        "dense_pose_file_sha256_in_order": [file_sha256(path) for path in args.dense_pose_inventory],
        "sparse_correspondence_file_sha256_in_order": [file_sha256(path) for path in args.sparse_correspondence_inventory],
        "dense_correspondence_file_sha256_in_order": [file_sha256(path) for path in args.dense_correspondence_inventory],
        "supplemental_correspondence_file_sha256_in_order": [
            file_sha256(path) for path in args.supplemental_correspondence_inventory
        ],
        "auxiliary_view_count": (
            None if not auxiliary_enabled else int(args.auxiliary_view_count)
        ),
        "auxiliary_pose_file_sha256_in_order": [
            file_sha256(path) for path in args.auxiliary_pose_inventory
        ],
        "auxiliary_correspondence_file_sha256_in_order": [
            file_sha256(path) for path in args.auxiliary_correspondence_inventory
        ],
        "selected_dense_count": int(np.sum(np.asarray(output_branch) == int(args.dense_view_count))),
        "production_eligible": False,
    }
    metadata["content_sha256"] = canonical_json_sha256(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".temporary.npz")
    np.savez_compressed(
        temporary, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    print(json.dumps({**metadata, "output_file_sha256": file_sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
