"""Geometry diagnostics for detector-referenced global landmark proposals."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


def _project_with_visibility(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    projected = project_world_to_image(points, pose, camera)
    camera_points = points @ pose[:3, :3].T + pose[:3, 3][None]
    visible = np.isfinite(projected).all(axis=1) & np.isfinite(camera_points[:, 2])
    visible &= camera_points[:, 2] > 1e-6
    visible &= (projected[:, 0] >= 0.0) & (projected[:, 0] <= float(camera.width - 1))
    visible &= (projected[:, 1] >= 0.0) & (projected[:, 1] <= float(camera.height - 1))
    return projected, visible


def nearest_visible_landmarks(
    query_xy: np.ndarray,
    landmark_index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the nearest projected unique SfM track for each detector point."""

    points = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    _unique_tracks, first_rows = np.unique(landmark_index.track_ids, return_index=True)
    projected, visible = _project_with_visibility(
        landmark_index.xyz[first_rows],
        pose_w2c,
        camera,
    )
    visible_rows = first_rows[visible]
    if len(visible_rows) == 0 or len(points) == 0:
        return (
            np.full((len(points),), -1, dtype=np.int64),
            np.full((len(points),), -1, dtype=np.int64),
            np.full((len(points),), np.inf, dtype=np.float32),
        )
    tree = cKDTree(projected[visible])
    distances, positions = tree.query(points, k=1, workers=-1)
    bank_rows = visible_rows[np.asarray(positions, dtype=np.int64)]
    return (
        bank_rows.astype(np.int64, copy=False),
        landmark_index.track_ids[bank_rows].astype(np.int64, copy=False),
        np.asarray(distances, dtype=np.float32),
    )


def candidate_reprojection_residuals(
    query_xy: np.ndarray,
    candidate_bank_rows: np.ndarray,
    landmark_index: LandmarkMapIndex,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> np.ndarray:
    """Compute GT geometric residual for every detector-to-landmark proposal."""

    points = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    rows = np.asarray(candidate_bank_rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[0] != points.shape[0]:
        raise ValueError("candidate rows must have shape (N, L) matching query points")
    output = np.full(rows.shape, np.inf, dtype=np.float32)
    valid = (rows >= 0) & (rows < len(landmark_index))
    if not np.any(valid):
        return output
    unique_rows = np.unique(rows[valid])
    projected, visible = _project_with_visibility(
        landmark_index.xyz[unique_rows],
        pose_w2c,
        camera,
    )
    row_to_position = {int(row): int(position) for position, row in enumerate(unique_rows.tolist())}
    for query_row in range(len(points)):
        for column in np.flatnonzero(valid[query_row]).tolist():
            position = row_to_position[int(rows[query_row, column])]
            if visible[position]:
                output[query_row, column] = float(
                    np.linalg.norm(projected[position] - points[query_row])
                )
    return output


def summarize_detector_proposal_geometry(
    *,
    nearest_landmark_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    query_ids: Sequence[str],
    thresholds_px: Sequence[float] = (1.0, 2.0, 5.0, 8.0),
    top_ls: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, object]:
    """Separate detector mappability from proposal retrieval recall."""

    nearest = np.asarray(nearest_landmark_residuals, dtype=np.float32).reshape(-1)
    candidates = np.asarray(candidate_residuals, dtype=np.float32)
    ids = np.asarray([str(value) for value in query_ids], dtype=np.str_)
    if candidates.ndim != 2 or candidates.shape[0] != nearest.shape[0] or len(ids) != len(nearest):
        raise ValueError("proposal geometry arrays have incompatible shapes")
    unique_queries = tuple(dict.fromkeys(ids.tolist()))
    output: dict[str, object] = {
        "detector_point_count": int(len(nearest)),
        "query_count": int(len(unique_queries)),
        "thresholds_px": {},
    }
    for threshold in thresholds_px:
        value = float(threshold)
        if value <= 0.0:
            raise ValueError("geometry thresholds must be positive")
        mappable = nearest <= value
        valid = candidates <= value
        first_rank = np.full((len(nearest),), -1, dtype=np.int64)
        for row in np.flatnonzero(np.any(valid, axis=1)).tolist():
            first_rank[row] = int(np.flatnonzero(valid[row])[0] + 1)
        query_positive_counts = []
        query_mappable_counts = []
        for query_id in unique_queries:
            mask = ids == query_id
            query_positive_counts.append(int(np.sum(np.any(valid[mask], axis=1))))
            query_mappable_counts.append(int(np.sum(mappable[mask])))
        threshold_summary: dict[str, object] = {
            "mappable_point_count": int(np.sum(mappable)),
            "mappable_point_rate": float(np.mean(mappable)) if len(mappable) else 0.0,
            "proposal_positive_point_count": int(np.sum(np.any(valid, axis=1))),
            "query_with_at_least_4_positive_rate": float(
                np.mean(np.asarray(query_positive_counts) >= 4)
            ) if query_positive_counts else 0.0,
            "median_positive_points_per_query": float(np.median(query_positive_counts)) if query_positive_counts else 0.0,
            "median_mappable_points_per_query": float(np.median(query_mappable_counts)) if query_mappable_counts else 0.0,
            "median_first_positive_rank_when_retrieved": (
                None if not np.any(first_rank > 0) else float(np.median(first_rank[first_rank > 0]))
            ),
        }
        for top_l in top_ls:
            limit = min(int(top_l), int(candidates.shape[1]))
            if limit <= 0:
                raise ValueError("top-L values must be positive")
            retrieved = np.any(valid[:, :limit], axis=1)
            threshold_summary[f"recall_at_{int(top_l)}_given_mappable"] = (
                0.0 if not np.any(mappable) else float(np.mean(retrieved[mappable]))
            )
            threshold_summary[f"positive_point_rate_at_{int(top_l)}_all"] = (
                float(np.mean(retrieved)) if len(retrieved) else 0.0
            )
        output["thresholds_px"][f"{value:g}"] = threshold_summary
    return output


def summarize_ranked_detector_proposal_geometry(
    *,
    nearest_landmark_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    candidate_scores: np.ndarray,
    query_ids: Sequence[str],
    thresholds_px: Sequence[float] = (1.0, 2.0, 5.0, 8.0),
    top_ls: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, object]:
    """Evaluate geometric recall after reranking one fixed proposal pool."""

    residuals = np.asarray(candidate_residuals, dtype=np.float32)
    scores = np.asarray(candidate_scores, dtype=np.float32)
    if residuals.shape != scores.shape or residuals.ndim != 2:
        raise ValueError("candidate residuals and scores must have matching shape (N, L)")
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-safe_scores, axis=1, kind="stable")
    ranked_residuals = np.take_along_axis(residuals, order, axis=1)
    ranked_scores = np.take_along_axis(safe_scores, order, axis=1)
    ranked_residuals[~np.isfinite(ranked_scores)] = np.inf
    return summarize_detector_proposal_geometry(
        nearest_landmark_residuals=nearest_landmark_residuals,
        candidate_residuals=ranked_residuals,
        query_ids=query_ids,
        thresholds_px=thresholds_px,
        top_ls=top_ls,
    )


def rank_candidate_pool(
    *,
    candidate_scores: np.ndarray,
    top_l: int,
    arrays: Sequence[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Rank aligned candidate arrays and retain one fixed top-L pool."""

    scores = np.asarray(candidate_scores, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError("candidate_scores must have shape (N, L)")
    limit = min(int(top_l), int(scores.shape[1]))
    if limit <= 0:
        raise ValueError("top_l must be positive")
    safe_scores = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.argsort(-safe_scores, axis=1, kind="stable")[:, :limit]
    ranked: list[np.ndarray] = []
    for values in arrays:
        source = np.asarray(values)
        if source.shape != scores.shape:
            raise ValueError("all candidate arrays must match candidate_scores")
        ranked.append(np.take_along_axis(source, order, axis=1))
    ranked_scores = np.take_along_axis(safe_scores, order, axis=1)
    return (ranked_scores, *ranked)


def geometry_oracle_scores(
    candidate_residuals_px: np.ndarray,
    *,
    threshold_px: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build diagnostic-only scores that select the lowest GT residual."""

    residuals = np.asarray(candidate_residuals_px, dtype=np.float32)
    if residuals.ndim != 2 or float(threshold_px) <= 0.0:
        raise ValueError("residuals must have shape (N, L) and threshold_px must be positive")
    valid = np.isfinite(residuals) & (residuals <= float(threshold_px))
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != residuals.shape:
            raise ValueError("valid_mask must match candidate residuals")
        valid &= supplied
    scores = np.full(residuals.shape, -np.inf, dtype=np.float32)
    scores[valid] = -residuals[valid]
    return scores


def summarize_query_proposal_difficulty(
    *,
    nearest_landmark_residuals: np.ndarray,
    ranked_candidate_residuals: np.ndarray,
    ranked_candidate_track_ids: np.ndarray,
    query_ids: Sequence[str],
    query_xy: np.ndarray,
    image_sizes: Mapping[str, tuple[int, int]],
    thresholds_px: Sequence[float] = (1.0, 2.0, 5.0),
    grid_shape: tuple[int, int] = (4, 4),
) -> list[dict[str, object]]:
    """Report per-image recall, unique tracks, and spatial coverage."""

    nearest = np.asarray(nearest_landmark_residuals, dtype=np.float32).reshape(-1)
    residuals = np.asarray(ranked_candidate_residuals, dtype=np.float32)
    tracks = np.asarray(ranked_candidate_track_ids, dtype=np.int64)
    ids = np.asarray([str(value) for value in query_ids], dtype=np.str_)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if (
        residuals.ndim != 2
        or tracks.shape != residuals.shape
        or len(nearest) != residuals.shape[0]
        or len(ids) != len(nearest)
        or len(xy) != len(nearest)
    ):
        raise ValueError("query proposal arrays have incompatible shapes")
    grid_rows, grid_cols = int(grid_shape[0]), int(grid_shape[1])
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_shape must be positive")

    output: list[dict[str, object]] = []
    for query_id in dict.fromkeys(ids.tolist()):
        row_mask = ids == str(query_id)
        width, height = image_sizes.get(str(query_id), (0, 0))
        query_summary: dict[str, object] = {
            "query_id": str(query_id),
            "point_count": int(np.sum(row_mask)),
            "thresholds_px": {},
        }
        for threshold in thresholds_px:
            value = float(threshold)
            if value <= 0.0:
                raise ValueError("geometry thresholds must be positive")
            local_residuals = residuals[row_mask]
            local_tracks = tracks[row_mask]
            positive_edges = np.isfinite(local_residuals) & (local_residuals <= value)
            positive_rows = np.any(positive_edges, axis=1)
            mappable_rows = nearest[row_mask] <= value
            first_ranks = np.full((len(local_residuals),), -1, dtype=np.int64)
            selected_tracks: list[int] = []
            for local_row in np.flatnonzero(positive_rows).tolist():
                columns = np.flatnonzero(positive_edges[local_row])
                first_ranks[local_row] = int(columns[0] + 1)
                best_column = int(columns[np.argmin(local_residuals[local_row, columns])])
                selected_tracks.append(int(local_tracks[local_row, best_column]))
            occupied: set[tuple[int, int]] = set()
            if int(width) > 0 and int(height) > 0:
                for point in xy[row_mask][positive_rows]:
                    col = int(np.clip(np.floor(float(point[0]) / float(width) * grid_cols), 0, grid_cols - 1))
                    row = int(np.clip(np.floor(float(point[1]) / float(height) * grid_rows), 0, grid_rows - 1))
                    occupied.add((row, col))
            retrieved_mappable = positive_rows[mappable_rows]
            query_summary["thresholds_px"][f"{value:g}"] = {
                "mappable_point_count": int(np.sum(mappable_rows)),
                "positive_point_count": int(np.sum(positive_rows)),
                "positive_unique_track_count": int(len(set(selected_tracks))),
                "recall_given_mappable": (
                    0.0 if not np.any(mappable_rows) else float(np.mean(retrieved_mappable))
                ),
                "median_first_positive_rank": (
                    None
                    if not np.any(first_ranks > 0)
                    else float(np.median(first_ranks[first_ranks > 0]))
                ),
                "grid_coverage": float(len(occupied) / float(grid_rows * grid_cols)),
                "grid_cell_count": int(len(occupied)),
            }
        output.append(query_summary)
    return output
