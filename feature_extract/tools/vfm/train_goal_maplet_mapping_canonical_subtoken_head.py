"""Train a deployment-matched mapping-only canonical-prototype subtoken head.

Fit targets pair a mapping token with a canonical metric-cell prototype made
from *other* fit observations.  Validation tokens come exclusively from the
held mapping route while their canonical prototypes come exclusively from fit
routes.  This closes the observation-pair/prototype distribution gap of the
v1 mechanism smoke without opening any query data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from feature_extract.tools.vfm.build_goal_maplet_plane_pnp_observation_bank import _load_contributor_geometry
from feature_extract.tools.vfm.train_goal_maplet_chart_local_radio_projection import _load_observation_bank
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import (
    MAXIMUM_POSITIVE_WORLD_DISTANCE_M,
    MINIMUM_VARIANCE_PX2,
    MappingSurfaceCoordinateHead,
    MappingSurfaceCoordinateContextHead,
    MappingSurfaceCoordinateDeepContextHead,
    MappingSurfaceCoordinateHomographyContextHead,
    MappingSurfaceCoordinateLocalCorrelationHead,
    MappingSurfaceCoordinateMixtureHead,
    SUBTOKEN_HALF_EXTENT_PX,
    TOKEN_GRID,
    TOKEN_SIZE_PX,
    MappingSubtokenHead,
    _load_projection,
    load_mapping_subtoken_head,
    _metrics,
    _project_world_to_pixel,
    _representative_rows,
    _state_arrays,
    _token_centres,
)
from feature_extract.vfm.localization_goal_maplet.geometry_native_planar_map import GeometryNativePlanarMap
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256, canonical_json_sha256, file_sha256
from feature_extract.vfm.localization_goal_maplet.plane_visibility_atlas import PlaneVisibilityAtlas


MAXIMUM_QUERY_ROWS_PER_IDENTITY = 8
HOMOGRAPHY_CONTEXT_THRESHOLD_M = 0.25
LOCAL_CORRELATION_OFFSETS_YX = np.asarray(
    [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], np.int64,
)


def _local_radio_correlation_volume(
    query_features: np.ndarray,
    map_features: np.ndarray,
    query_valid: np.ndarray,
    map_valid: np.ndarray,
) -> np.ndarray:
    """Return a masked 3x3-by-3x3 cosine volume plus both validity masks."""
    query = np.asarray(query_features, np.float32)
    mapping = np.asarray(map_features, np.float32)
    qvalid = np.asarray(query_valid, bool)
    mvalid = np.asarray(map_valid, bool)
    if (
        query.ndim != 3 or mapping.shape != query.shape or query.shape[1] != 9
        or qvalid.shape != query.shape[:2] or mvalid.shape != query.shape[:2]
        or not (np.all(np.isfinite(query)) and np.all(np.isfinite(mapping)))
    ):
        raise ValueError("local RADIO correlation inputs differ")
    pair_valid = qvalid[:, :, None] & mvalid[:, None, :]
    correlation = np.einsum("nif,njf->nij", query, mapping, optimize=True)
    correlation = np.where(pair_valid, np.clip(correlation, -1.0, 1.0), 0.0)
    result = np.concatenate(
        (correlation.reshape(len(query), -1), qvalid.astype(np.float32), mvalid.astype(np.float32)),
        axis=1,
    ).astype(np.float32)
    if result.shape != (
        len(query), MappingSurfaceCoordinateLocalCorrelationHead.LOCAL_CORRELATION_DIMENSION,
    ):
        raise AssertionError("local RADIO correlation dimension differs")
    return result


def _local_correlation_reference_gate(
    current: dict[str, object], reference: dict[str, object],
) -> tuple[bool, dict[str, object]]:
    """Require the local head to dominate the frozen V11 held-route metrics."""
    current_uv = current["chart_uv_offset"]
    reference_uv = reference["chart_uv_offset"]
    comparisons = {
        "image_median_nonincrease": bool(
            current["predicted_error_px"]["median"] <= reference["predicted_error_px"]["median"]
        ),
        "image_p90_nonincrease": bool(
            current["predicted_error_px"]["p90"] <= reference["predicted_error_px"]["p90"]
        ),
        "image_nll_nonincrease": bool(
            current["predicted_isotropic_gaussian_nll"]
            <= reference["predicted_isotropic_gaussian_nll"]
        ),
        "chart_uv_median_nonincrease": bool(
            current_uv["predicted_error_m"]["median"]
            <= reference_uv["predicted_error_m"]["median"]
        ),
        "chart_uv_p90_nonincrease": bool(
            current_uv["predicted_error_m"]["p90"]
            <= reference_uv["predicted_error_m"]["p90"]
        ),
        "chart_uv_nll_nonincrease": bool(
            current_uv["predicted_isotropic_gaussian_nll"]
            <= reference_uv["predicted_isotropic_gaussian_nll"]
        ),
        "positive_negative_margin_nondecrease": bool(
            current["positive_match_probability_mean"] - current["negative_match_probability_mean"]
            >= reference["positive_match_probability_mean"]
            - reference["negative_match_probability_mean"]
        ),
        "image_uncertainty_monotonic": bool(np.all(np.diff(
            current["uncertainty_quantile_mean_error_px"],
        ) >= 0.0)),
        "chart_uv_uncertainty_monotonic": bool(np.all(np.diff(
            current_uv["uncertainty_quantile_mean_error_m"],
        ) >= 0.0)),
    }
    summary = {
        "criteria": comparisons,
        "image_median_delta_px": float(
            current["predicted_error_px"]["median"] - reference["predicted_error_px"]["median"]
        ),
        "image_p90_delta_px": float(
            current["predicted_error_px"]["p90"] - reference["predicted_error_px"]["p90"]
        ),
        "image_nll_delta": float(
            current["predicted_isotropic_gaussian_nll"]
            - reference["predicted_isotropic_gaussian_nll"]
        ),
        "chart_uv_median_delta_m": float(
            current_uv["predicted_error_m"]["median"]
            - reference_uv["predicted_error_m"]["median"]
        ),
        "chart_uv_p90_delta_m": float(
            current_uv["predicted_error_m"]["p90"]
            - reference_uv["predicted_error_m"]["p90"]
        ),
        "chart_uv_nll_delta": float(
            current_uv["predicted_isotropic_gaussian_nll"]
            - reference_uv["predicted_isotropic_gaussian_nll"]
        ),
    }
    return bool(all(comparisons.values())), summary


def _project_query_local_feature_grid(
    query_rows: np.ndarray,
    *,
    observation: np.ndarray,
    token_ids: np.ndarray,
    raw_features: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the same plane-observation's 3x3 RADIO token neighborhood.

    The observation bank is plane specific, so a missing neighbor means that
    the neighboring token is not part of the same physical-plane observation.
    This mask is available at deployment from the query plane region itself.
    """
    rows = np.asarray(query_rows, np.int64).reshape(-1)
    obs = np.asarray(observation, np.int64).reshape(-1)
    token = np.asarray(token_ids, np.int64).reshape(-1)
    raw = np.asarray(raw_features, np.float32)
    weight = np.asarray(projection, np.float32)
    if (
        len(obs) != len(token) or raw.ndim != 2 or len(raw) != len(token)
        or weight.ndim != 2 or raw.shape[1] != weight.shape[1]
        or np.any((rows < 0) | (rows >= len(token)))
    ):
        raise ValueError("query local RADIO inventory differs")
    key = obs.astype(np.int64) * int(np.prod(TOKEN_GRID)) + token
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    if np.any(sorted_key[1:] == sorted_key[:-1]):
        raise ValueError("query plane observation contains duplicate RADIO tokens")
    centre = token[rows]
    cy, cx = np.divmod(centre, TOKEN_GRID[1])
    neighbor_y = cy[:, None] + LOCAL_CORRELATION_OFFSETS_YX[None, :, 0]
    neighbor_x = cx[:, None] + LOCAL_CORRELATION_OFFSETS_YX[None, :, 1]
    inside = (
        (neighbor_y >= 0) & (neighbor_y < TOKEN_GRID[0])
        & (neighbor_x >= 0) & (neighbor_x < TOKEN_GRID[1])
    )
    neighbor_token = neighbor_y * TOKEN_GRID[1] + neighbor_x
    target_key = obs[rows, None] * int(np.prod(TOKEN_GRID)) + neighbor_token
    position = np.searchsorted(sorted_key, target_key)
    clipped = np.minimum(position, max(len(sorted_key) - 1, 0))
    valid = inside & (position < len(sorted_key))
    if len(sorted_key):
        valid &= sorted_key[clipped] == target_key
    matched = np.zeros_like(position, dtype=np.int64)
    matched[valid] = order[clipped[valid]]
    unique = np.unique(matched[valid])
    projected = np.zeros((len(token), weight.shape[0]), np.float32)
    for start in range(0, len(unique), 8192):
        selected = unique[start : start + 8192]
        projected[selected] = _normalise(raw[selected] @ weight.T)
    grid = projected[matched]
    grid[~valid] = 0.0
    return grid, valid


def _view_cell_mean_features(raw_features, projection, identity, observation, representative_rows):
    """Project each token before view/cell averaging; leave center inputs untouched."""
    identity = np.asarray(identity, np.int64)
    observation = np.asarray(observation, np.int64)
    reps = np.asarray(representative_rows, np.int64)
    if len(raw_features) != len(identity) or identity.shape != observation.shape:
        raise ValueError("view-cell token inventory differs")
    projected = np.empty((len(identity), projection.shape[0]), np.float32)
    for lo in range(0, len(identity), 8192):
        projected[lo:lo + 8192] = _normalise(
            np.asarray(raw_features[lo:lo + 8192], np.float32) @ projection.T,
        )
    order = np.lexsort((observation, identity))
    starts = np.flatnonzero(np.r_[True, (identity[order][1:] != identity[order][:-1])
                                 | (observation[order][1:] != observation[order][:-1])])
    counts = np.diff(np.r_[starts, len(order)])
    means = _normalise(np.add.reduceat(projected[order], starts, axis=0) / counts[:, None])
    lookup = {(int(identity[row]), int(observation[row])): i
              for i, row in enumerate(order[starts])}
    output = np.zeros_like(projected)
    for row in reps:
        output[row] = means[lookup[(int(identity[row]), int(observation[row]))]]
    return output


def _mapping_local_feature_grid(
    query_rows: np.ndarray,
    candidate_features: np.ndarray,
    candidate_world: np.ndarray,
    *,
    leave_query_observation_out: bool,
    representative_rows: np.ndarray,
    representative_identity: np.ndarray,
    identity_keys_plane_cell: np.ndarray,
    observation: np.ndarray,
    route_per_observation: np.ndarray,
    fit_routes: set[str],
    projected_features: np.ndarray,
    plane_rows: np.ndarray,
    plane_centers_world: np.ndarray,
    plane_frames_world: np.ndarray,
    cell_size_m: float = 0.5,
    neighbor_policy: str = "canonical_mean",
    maximum_neighbor_modes: int = 4,
    minimum_neighbor_views: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a source-view-free local canonical map grid for each candidate."""
    qrows = np.asarray(query_rows, np.int64).reshape(-1)
    candidate = _normalise(np.asarray(candidate_features, np.float32))
    candidate_xyz = np.asarray(candidate_world, np.float64).reshape(-1, 3)
    reps = np.asarray(representative_rows, np.int64).reshape(-1)
    rep_identity = np.asarray(representative_identity, np.int64).reshape(-1)
    identity_keys = np.asarray(identity_keys_plane_cell, np.int64).reshape(-1, 3)
    obs = np.asarray(observation, np.int64).reshape(-1)
    route = np.asarray(route_per_observation).astype(str)
    projected = np.asarray(projected_features, np.float32)
    plane = np.asarray(plane_rows, np.int64).reshape(-1)
    if (
        len(qrows) != len(candidate) or len(qrows) != len(candidate_xyz)
        or len(reps) != len(rep_identity) or len(obs) != len(plane)
        or len(projected) != len(obs) or np.any((qrows < 0) | (qrows >= len(obs)))
        or np.any((reps < 0) | (reps >= len(obs)))
        or len(route) <= int(np.max(obs, initial=-1))
    ):
        raise ValueError("mapping local RADIO inventory differs")
    fit_rep = np.isin(route[obs[reps]], sorted(fit_routes))
    fit_rows = reps[fit_rep]
    fit_identity = rep_identity[fit_rep]
    if neighbor_policy not in ("canonical_mean", "atlas_modes_mean") or maximum_neighbor_modes < 1:
        raise ValueError("invalid local neighbor policy")
    grouped_rows = {}
    for row, identity in zip(fit_rows.tolist(), fit_identity.tolist()):
        grouped_rows.setdefault(identity, []).append(row)
    mode_cache = {}
    feature_sum = np.zeros((len(identity_keys), projected.shape[1]), np.float64)
    feature_count = np.zeros(len(identity_keys), np.int64)
    np.add.at(feature_sum, fit_identity, projected[fit_rows])
    np.add.at(feature_count, fit_identity, 1)
    per_observation: dict[tuple[int, int], int] = {
        (int(identity), int(obs[row])): int(row)
        for row, identity in zip(fit_rows.tolist(), fit_identity.tolist())
    }
    key_to_identity = {tuple(value.tolist()): index for index, value in enumerate(identity_keys)}
    qplane = plane[qrows]
    tangent = np.asarray(plane_frames_world, np.float64)[qplane, :2]
    candidate_uv = np.einsum(
        "ni,nji->nj",
        candidate_xyz - np.asarray(plane_centers_world, np.float64)[qplane], tangent,
    )
    candidate_cell = np.floor(candidate_uv / float(cell_size_m)).astype(np.int64)
    grid = np.zeros((len(qrows), 9, projected.shape[1]), np.float32)
    valid = np.zeros((len(qrows), 9), bool)
    for sample in range(len(qrows)):
        for offset_index, (dy, dx) in enumerate(LOCAL_CORRELATION_OFFSETS_YX.tolist()):
            if dy == 0 and dx == 0:
                grid[sample, offset_index] = candidate[sample]
                valid[sample, offset_index] = True
                continue
            key = (
                int(qplane[sample]), int(candidate_cell[sample, 0] + dx),
                int(candidate_cell[sample, 1] + dy),
            )
            identity = key_to_identity.get(key)
            if identity is None:
                continue
            if neighbor_policy == "atlas_modes_mean":
                excluded_obs = int(obs[qrows[sample]]) if leave_query_observation_out else -1
                cache_key = (identity, excluded_obs)
                if cache_key not in mode_cache:
                    pool = np.asarray(sorted(grouped_rows.get(identity, [])), np.int64)
                    pool = pool[obs[pool] != excluded_obs]
                    if len(np.unique(obs[pool])) >= minimum_neighbor_views:
                        modes = projected[pool][_diverse_mode_indices(
                            projected[pool], maximum_neighbor_modes,
                        )]
                        # Atlas descriptors are persisted in float16 before runtime aggregation.
                        modes = modes.astype(np.float16).astype(np.float32)
                        mean = np.mean(modes, axis=0, dtype=np.float64).astype(np.float32)
                        norm = float(np.linalg.norm(mean))
                        if not np.isfinite(norm) or norm <= 1e-8:
                            raise ValueError("atlas local RADIO descriptor is invalid")
                        mode_cache[cache_key] = mean / norm
                    else:
                        mode_cache[cache_key] = None
                value = mode_cache[cache_key]
                if value is not None:
                    grid[sample, offset_index] = value
                    valid[sample, offset_index] = True
                continue
            total = feature_sum[identity].copy()
            count = int(feature_count[identity])
            if leave_query_observation_out:
                excluded = per_observation.get((identity, int(obs[qrows[sample]])))
                if excluded is not None:
                    total -= projected[excluded]
                    count -= 1
            if count <= 0:
                continue
            grid[sample, offset_index] = _normalise(total / count)
            valid[sample, offset_index] = True
    return grid, valid


def _undistort_simple_radial_xy(
    pixels_xy: np.ndarray, camera_matrices: np.ndarray, radial_k1: np.ndarray,
) -> np.ndarray:
    """Invert SIMPLE_RADIAL in normalized coordinates with deterministic Newton steps."""
    pixel = np.asarray(pixels_xy, np.float64).reshape(-1, 2)
    matrix = np.asarray(camera_matrices, np.float64).reshape(-1, 3, 3)
    radial = np.asarray(radial_k1, np.float64).reshape(-1)
    if len(pixel) != len(matrix) or len(pixel) != len(radial):
        raise ValueError("radial undistortion arrays differ")
    distorted = (pixel - matrix[:, (0, 1), (2, 2)]) / matrix[:, (0, 1), (0, 1)]
    distorted_radius = np.linalg.norm(distorted, axis=1)
    radius = distorted_radius.copy()
    for _ in range(12):
        residual = radius + radial * radius ** 3 - distorted_radius
        derivative = 1.0 + 3.0 * radial * radius ** 2
        radius = np.maximum(radius - residual / np.maximum(np.abs(derivative), 1e-8), 0.0)
    scale = np.divide(
        radius, distorted_radius, out=np.ones_like(radius), where=distorted_radius > 1e-12,
    )
    ideal = distorted * scale[:, None]
    if not np.all(np.isfinite(ideal)):
        raise ValueError("radial undistortion is nonfinite")
    return ideal


def _surface_geometric_context(
    query_rows: np.ndarray,
    map_world: np.ndarray,
    *,
    observation: np.ndarray,
    token_ids: np.ndarray,
    plane_rows: np.ndarray,
    poses_w2c: np.ndarray,
    camera_matrices: np.ndarray,
    radial_coefficients: np.ndarray,
    plane_centers_world: np.ndarray,
    plane_frames_world: np.ndarray,
    cell_size_m: float = 0.5,
    homography_projected_uv_m: np.ndarray | None = None,
    homography_valid: np.ndarray | None = None,
) -> np.ndarray:
    """Build pose-free-at-deployment ray/incidence/prototype-phase conditioning."""
    query = np.asarray(query_rows, np.int64).reshape(-1)
    mapping = np.asarray(map_world, np.float64).reshape(-1, 3)
    if len(query) != len(mapping):
        raise ValueError("surface context rows differ")
    query_observation = np.asarray(observation, np.int64)[query]
    query_plane = np.asarray(plane_rows, np.int64)[query]
    pose = np.asarray(poses_w2c, np.float64)[query_observation]
    matrix = np.asarray(camera_matrices, np.float64)[query_observation]
    radial = np.asarray(radial_coefficients, np.float64)[query_observation]
    ideal = _undistort_simple_radial_xy(_token_centres(token_ids[query]), matrix, radial)
    ray = np.c_[ideal, np.ones(len(ideal), np.float64)]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
    normal_world = np.asarray(plane_frames_world, np.float64)[query_plane, 2]
    normal_camera = np.einsum("nij,nj->ni", pose[:, :3, :3], normal_world)
    offset_camera = (
        np.sum(normal_world * np.asarray(plane_centers_world, np.float64)[query_plane], axis=1)
        + np.sum(normal_camera * pose[:, :3, 3], axis=1)
    )
    normal_camera *= np.where(offset_camera >= 0.0, 1.0, -1.0)[:, None]
    incidence = np.abs(np.sum(normal_camera * ray, axis=1))
    tangent = np.asarray(plane_frames_world, np.float64)[query_plane, :2]
    uv = np.einsum(
        "ni,nji->nj",
        mapping - np.asarray(plane_centers_world, np.float64)[query_plane], tangent,
    )
    phase = 2.0 * (uv / float(cell_size_m) - np.floor(uv / float(cell_size_m)) - 0.5)
    context = np.c_[ideal, phase, incidence]
    expected_dimension = MappingSurfaceCoordinateContextHead.CONTEXT_DIMENSION
    if homography_projected_uv_m is not None:
        projected_uv = np.asarray(homography_projected_uv_m, np.float64).reshape(-1, 2)
        valid = np.asarray(homography_valid, bool).reshape(-1)
        if len(projected_uv) != len(query) or len(valid) != len(query):
            raise ValueError("mapping homography context arrays differ")
        residual = np.clip(
            (projected_uv - uv) / HOMOGRAPHY_CONTEXT_THRESHOLD_M, -2.0, 2.0,
        )
        residual[~valid] = 0.0
        context = np.c_[context, residual, valid.astype(np.float64)]
        expected_dimension = MappingSurfaceCoordinateHomographyContextHead.CONTEXT_DIMENSION
    elif homography_valid is not None:
        raise ValueError("mapping homography context arrays differ")
    if context.shape != (len(query), expected_dimension):
        raise AssertionError("surface context dimension differs")
    return context.astype(np.float32)


def _mapping_homography_projection(
    query_rows: np.ndarray,
    map_world: np.ndarray,
    *,
    observation: np.ndarray,
    token_ids: np.ndarray,
    plane_rows: np.ndarray,
    plane_centers_world: np.ndarray,
    plane_frames_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one mapping-only coarse image-to-chart homography per view/plane."""
    query = np.asarray(query_rows, np.int64).reshape(-1)
    mapping = np.asarray(map_world, np.float64).reshape(-1, 3)
    if len(query) != len(mapping):
        raise ValueError("mapping homography rows differ")
    obs = np.asarray(observation, np.int64)[query]
    plane = np.asarray(plane_rows, np.int64)[query]
    tangent = np.asarray(plane_frames_world, np.float64)[plane, :2]
    target_uv = np.einsum(
        "ni,nji->nj", mapping - np.asarray(plane_centers_world, np.float64)[plane], tangent,
    )
    token = np.asarray(token_ids, np.int64)[query]
    source_xy = np.c_[token % 64, token // 64].astype(np.float64)
    projected = target_uv.copy()
    valid = np.zeros(len(query), bool)
    group = np.c_[obs, plane]
    _, inverse = np.unique(group, axis=0, return_inverse=True)
    for value in range(int(np.max(inverse)) + 1 if len(inverse) else 0):
        rows = np.flatnonzero(inverse == value)
        if len(rows) < 4:
            continue
        cv2.setRNGSeed(260901)
        homography, mask = cv2.findHomography(
            source_xy[rows], target_uv[rows], cv2.RANSAC,
            HOMOGRAPHY_CONTEXT_THRESHOLD_M, maxIters=2000, confidence=0.995,
        )
        if homography is None or mask is None or int(np.sum(mask)) < 4:
            continue
        estimate = cv2.perspectiveTransform(
            source_xy[rows].reshape(-1, 1, 2), np.asarray(homography, np.float64),
        ).reshape(-1, 2)
        finite = np.all(np.isfinite(estimate), axis=1)
        projected[rows[finite]] = estimate[finite]
        valid[rows[finite]] = True
    return projected, valid


def _normalise(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _diverse_mode_indices(features: np.ndarray, maximum_modes: int) -> np.ndarray:
    """Replay the atlas medoid-then-farthest deterministic mode selector."""
    value = _normalise(np.asarray(features, np.float32))
    if value.ndim != 2 or not len(value) or int(maximum_modes) < 1:
        raise ValueError("mode-selection features differ")
    similarity = value @ value.T
    selected = [int(np.argmax(np.sum(similarity, axis=1)))]
    while len(selected) < min(int(maximum_modes), len(value)):
        remaining = np.asarray(
            [row for row in range(len(value)) if row not in selected], np.int64,
        )
        maximum_similarity = np.max(similarity[remaining][:, selected], axis=1)
        selected.append(int(remaining[np.argmin(maximum_similarity)]))
    return np.asarray(selected, np.int64)


def _closed_form_coordinate_shrinkage(mean: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(mean, np.float64).reshape(-1, 2)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    if prediction.shape != truth.shape or not len(prediction):
        raise ValueError("coordinate shrinkage inputs differ")
    return float(np.clip(
        np.sum(prediction * truth) / max(np.sum(prediction * prediction), 1e-12),
        0.0, 1.0,
    ))


def _closed_form_coordinate_affine(mean: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the six-parameter 2D affine calibration on mapping observations."""
    prediction = np.asarray(mean, np.float64).reshape(-1, 2)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    if prediction.shape != truth.shape or len(prediction) < 3:
        raise ValueError("coordinate affine inputs differ")
    design = np.c_[prediction, np.ones(len(prediction), np.float64)]
    coefficient = np.linalg.lstsq(design, truth, rcond=None)[0]
    matrix, bias = coefficient[:2], coefficient[2]
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(bias))):
        raise ValueError("coordinate affine calibration is invalid")
    return matrix, bias


def _apply_coordinate_calibration(
    mean: np.ndarray,
    *,
    shrinkage: float,
    affine_matrix: np.ndarray | None,
    affine_bias: np.ndarray | None,
) -> np.ndarray:
    value = np.asarray(mean, np.float64).reshape(-1, 2)
    if affine_matrix is None:
        result = float(shrinkage) * value
    else:
        matrix = np.asarray(affine_matrix, np.float64).reshape(2, 2)
        bias = np.asarray(affine_bias, np.float64).reshape(2)
        result = value @ matrix + bias
    # The measurement remains an offset inside its original 4x4 RADIO token.
    return np.clip(result, -SUBTOKEN_HALF_EXTENT_PX, SUBTOKEN_HALF_EXTENT_PX)


def _isotropic_variance_scale(
    mean: np.ndarray, target: np.ndarray, variance: np.ndarray,
) -> float:
    """Maximum-likelihood scalar for a predicted isotropic 2D variance."""
    prediction = np.asarray(mean, np.float64).reshape(-1, 2)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    raw = np.asarray(variance, np.float64).reshape(-1)
    if prediction.shape != truth.shape or len(raw) != len(truth) or not len(raw) or np.any(raw <= 0.0):
        raise ValueError("variance calibration inputs differ")
    scale = 0.5 * float(np.mean(np.sum(np.square(prediction - truth), axis=1) / raw))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("variance calibration scale is invalid")
    return scale


def _isotropic_gaussian_nll(mean: np.ndarray, target: np.ndarray, variance: np.ndarray) -> float:
    prediction = np.asarray(mean, np.float64).reshape(-1, 2)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    value = np.asarray(variance, np.float64).reshape(-1)
    if prediction.shape != truth.shape or len(value) != len(truth) or np.any(value <= 0.0):
        raise ValueError("Gaussian NLL inputs differ")
    return float(np.mean(0.5 * np.sum(np.square(prediction - truth), axis=1) / value + np.log(value)))


def _continuous_offset_metrics(
    target: np.ndarray, mean: np.ndarray, variance: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    prediction = np.asarray(mean, np.float64).reshape(-1, 2)
    raw_variance = np.asarray(variance, np.float64).reshape(-1)
    if len(truth) != len(prediction) or len(truth) != len(raw_variance):
        raise ValueError("continuous offset metric inputs differ")
    error = np.linalg.norm(prediction - truth, axis=1)
    baseline = np.linalg.norm(truth, axis=1)
    order = np.argsort(raw_variance, kind="stable")
    quartiles = np.array_split(order, 4)
    return {
        "pair_count": int(len(truth)),
        "zero_offset_error_m": {
            "median": float(np.median(baseline)), "p90": float(np.quantile(baseline, 0.9)),
            "mean": float(np.mean(baseline)),
        },
        "predicted_error_m": {
            "median": float(np.median(error)), "p90": float(np.quantile(error, 0.9)),
            "mean": float(np.mean(error)),
        },
        "predicted_better_fraction": float(np.mean(error < baseline)),
        "relative_median_improvement": float(1.0 - np.median(error) / max(np.median(baseline), 1e-12)),
        "uncertainty_quantile_mean_error_m": [
            float(np.mean(error[part])) for part in quartiles if len(part)
        ],
    }


def _mixture_log_probabilities(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, np.float64)
    shifted = value - np.max(value, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _isotropic_mixture_nll(
    means: np.ndarray, target: np.ndarray, variances: np.ndarray, logits: np.ndarray,
) -> float:
    prediction = np.asarray(means, np.float64)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    variance = np.asarray(variances, np.float64)
    if (
        prediction.ndim != 3 or prediction.shape[0] != len(truth)
        or prediction.shape[2] != 2 or variance.shape != prediction.shape[:2]
        or np.asarray(logits).shape != prediction.shape[:2]
        or np.any(variance <= 0.0)
    ):
        raise ValueError("mixture Gaussian inputs differ")
    error2 = np.sum(np.square(prediction - truth[:, None, :]), axis=2)
    component = _mixture_log_probabilities(logits) - np.log(variance) - 0.5 * error2 / variance
    maximum = np.max(component, axis=1)
    return float(-np.mean(maximum + np.log(np.sum(np.exp(component - maximum[:, None]), axis=1))))


def _mixture_variance_scale(
    means: np.ndarray, target: np.ndarray, variances: np.ndarray, logits: np.ndarray,
) -> float:
    prediction = np.asarray(means, np.float64)
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    variance = np.asarray(variances, np.float64)
    error2 = np.sum(np.square(prediction - truth[:, None, :]), axis=2)
    log_responsibility = (
        _mixture_log_probabilities(logits) - np.log(variance) - 0.5 * error2 / variance
    )
    log_responsibility -= np.max(log_responsibility, axis=1, keepdims=True)
    responsibility = np.exp(log_responsibility)
    responsibility /= np.sum(responsibility, axis=1, keepdims=True)
    scale = 0.5 * float(np.mean(np.sum(responsibility * error2 / variance, axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("mixture variance calibration scale is invalid")
    return scale


def _mixture_offset_metrics(
    target: np.ndarray, means: np.ndarray, variances: np.ndarray, logits: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(target, np.float64).reshape(-1, 2)
    prediction = np.asarray(means, np.float64)
    variance = np.asarray(variances, np.float64)
    log_weight = _mixture_log_probabilities(logits)
    weight = np.exp(log_weight)
    posterior_mean = np.sum(weight[:, :, None] * prediction, axis=1)
    top = np.argmax(log_weight, axis=1)
    top_mean = prediction[np.arange(len(prediction)), top]
    per_mode_error = np.linalg.norm(prediction - truth[:, None, :], axis=2)
    posterior_error = np.linalg.norm(posterior_mean - truth, axis=1)
    top_error = np.linalg.norm(top_mean - truth, axis=1)
    oracle_error = np.min(per_mode_error, axis=1)
    baseline = np.linalg.norm(truth, axis=1)
    predictive_variance = np.sum(
        weight * (variance + 0.5 * np.sum(np.square(prediction - posterior_mean[:, None, :]), axis=2)),
        axis=1,
    )
    quartiles = np.array_split(np.argsort(predictive_variance, kind="stable"), 4)
    entropy = -np.sum(weight * log_weight, axis=1)
    return {
        "pair_count": int(len(truth)),
        "zero_offset_error_m": {
            "median": float(np.median(baseline)), "p90": float(np.quantile(baseline, 0.9)),
            "mean": float(np.mean(baseline)),
        },
        "posterior_mean_error_m": {
            "median": float(np.median(posterior_error)), "p90": float(np.quantile(posterior_error, 0.9)),
            "mean": float(np.mean(posterior_error)),
        },
        "top_probability_mode_error_m": {
            "median": float(np.median(top_error)), "p90": float(np.quantile(top_error, 0.9)),
            "mean": float(np.mean(top_error)),
        },
        "best_mode_oracle_error_m": {
            "median": float(np.median(oracle_error)), "p90": float(np.quantile(oracle_error, 0.9)),
            "mean": float(np.mean(oracle_error)),
        },
        "posterior_mean_better_fraction": float(np.mean(posterior_error < baseline)),
        "posterior_mean_relative_median_improvement": float(
            1.0 - np.median(posterior_error) / max(np.median(baseline), 1e-12)
        ),
        "best_mode_relative_median_improvement": float(
            1.0 - np.median(oracle_error) / max(np.median(baseline), 1e-12)
        ),
        "mixture_isotropic_gaussian_nll": _isotropic_mixture_nll(
            prediction, truth, variance, logits,
        ),
        "predictive_uncertainty_quantile_mean_error_m": [
            float(np.mean(posterior_error[part])) for part in quartiles if len(part)
        ],
        "mean_mode_entropy_nats": float(np.mean(entropy)),
        "mean_effective_mode_count": float(np.mean(np.exp(entropy))),
    }


def _freeze_coordinate_parameters(model):
    count=0
    for name,parameter in model.named_parameters():
        trainable=name.startswith('match.')
        parameter.requires_grad_(trainable)
        count+=int(trainable)
    if not count:
        raise ValueError('model has no separate match layer')


def _source_view_mode_pool(identity, source_views, world, features):
    """Atlas-equivalent token pooling; each mode retains its own mean geometry."""
    identity=np.asarray(identity); source_views=np.asarray(source_views)
    if (identity.ndim != 1 or not len(identity) or source_views.shape != identity.shape
            or np.shape(world) != (len(identity),3) or np.ndim(features) != 2
            or len(features) != len(identity) or not np.isfinite(world).all()
            or not np.isfinite(features).all()):
        raise ValueError("source mode token inventory differs")
    order=np.lexsort((source_views,identity))
    keys=np.c_[identity[order],source_views[order]]
    starts=np.flatnonzero(np.r_[True,np.any(keys[1:]!=keys[:-1],axis=1)])
    counts=np.diff(np.r_[starts,len(order)])
    feature=_normalise(np.add.reduceat(np.asarray(features,np.float32)[order],starts,axis=0)/counts[:,None])
    point=np.add.reduceat(np.asarray(world,np.float64)[order],starts,axis=0)/counts[:,None]
    return keys[starts],feature,point


def _fit_source_view_mode_dataset(representatives, rep_identity, identity, observation,
        source_view_per_observation, routes, fit_routes, validation_route, world,
        all_projected, token_ids, poses, matrices, radial, maximum_modes):
    source=source_view_per_observation[observation]
    keys, features, points=_source_view_mode_pool(identity,source,world,all_projected)
    source_route={int(v):str(routes[o]) for o,v in enumerate(source_view_per_observation)}
    pool_by_identity={}
    for i,key in enumerate(keys):
        if source_route[int(key[1])] in fit_routes:
            pool_by_identity.setdefault(int(key[0]),[]).append(i)
    qs=[[],[]]; ms=[[],[]]; cache={}
    counts={}
    for query,ident in zip(representatives,rep_identity):
        route=str(routes[observation[query]])
        split=0 if route in fit_routes else 1 if route==validation_route else -1
        if split<0:continue
        countkey=(int(ident),split)
        counts[countkey]=counts.get(countkey,0)+1
        if counts[countkey]>MAXIMUM_QUERY_ROWS_PER_IDENTITY:continue
        excluded=int(source[query]) if split==0 else -1
        cachekey=(int(ident),excluded)
        if cachekey not in cache:
            pool=np.asarray(pool_by_identity.get(int(ident),[]),np.int64)
            pool=pool[keys[pool,1]!=excluded]
            cache[cachekey]=(pool[_diverse_mode_indices(features[pool],maximum_modes)]
                             if len(pool)>=2 else np.zeros(0,np.int64))
        selected=cache[cachekey]
        qs[split].extend([int(query)]*len(selected));ms[split].extend(selected.tolist())
    result={}
    for split,prefix in enumerate(('fit','validation')):
        q=np.asarray(qs[split],np.int64);m=np.asarray(ms[split],np.int64)
        obs=observation[q]
        pixel,depth=_project_world_to_pixel(points[m],poses[obs],matrices[obs],radial[obs])
        target=pixel-_token_centres(token_ids[q])
        keep=((depth>0)&np.isfinite(target).all(axis=1)&(np.max(np.abs(target),axis=1)<2.)
              &(np.linalg.norm(world[q]-points[m],axis=1)<=MAXIMUM_POSITIVE_WORLD_DISTANCE_M))
        result.update({prefix+'_query_rows':q[keep],prefix+'_map_features':features[m[keep]].astype(np.float32),
                       prefix+'_map_world':points[m[keep]],prefix+'_targets':target[keep].astype(np.float32),
                       prefix+'_map_source_views':keys[m[keep],1]})
        if split==0 and np.any(source[q[keep]]==keys[m[keep],1]):
            raise AssertionError('source view leaked into anonymous center')
    pool=np.asarray([i for rows in pool_by_identity.values() for i in rows],np.int64)
    result.update(negative_pool_features=features[pool].astype(np.float32),negative_pool_world=points[pool],
                  negative_pool_source=keys[pool,1],negative_pool_identity=keys[pool,0])
    return result


def _isolated_null_indices(query_rows, map_world, map_source, observation, source_views, identity, plane, world):
    """Same-plane, different-cell negatives; exclude the entire query source image."""
    query_rows=np.asarray(query_rows); result=np.zeros(len(query_rows),np.int64)
    valid=np.zeros(len(query_rows),bool)
    qplane=plane[query_rows]; qcell=identity[query_rows]; qsource=source_views[observation[query_rows]]
    for physical in np.unique(qplane):
        rows=np.flatnonzero(qplane==physical)
        for cell in np.unique(qcell[rows]):
            queries=rows[qcell[rows]==cell]
            for source in np.unique(qsource[queries]):
                selected=queries[qsource[queries]==source]
                pool=rows[(qcell[rows]!=cell)&(map_source[rows]!=source)]
                if not len(pool):continue
                center=np.mean(world[query_rows[selected]],axis=0)
                far=pool[np.argmax(np.linalg.norm(map_world[pool]-center,axis=1))]
                result[selected]=far
                valid[selected]=np.linalg.norm(world[query_rows[selected]]-map_world[far],axis=1)>MAXIMUM_POSITIVE_WORLD_DISTANCE_M
    return result,valid


def _fit_mode_prototype_dataset(
    representative_rows: np.ndarray,
    representative_identity: np.ndarray,
    observation: np.ndarray,
    route_per_observation: np.ndarray,
    fit_routes: set[str],
    validation_route: str,
    world: np.ndarray,
    projected_features: np.ndarray,
    token_ids: np.ndarray,
    poses_w2c: np.ndarray,
    camera_matrices: np.ndarray,
    radial_coefficients: np.ndarray,
    maximum_modes: int,
) -> dict[str, np.ndarray]:
    """Make labels against the anonymous diverse modes used by the atlas."""
    order = np.lexsort((representative_rows, representative_identity))
    rows = representative_rows[order]
    ids = representative_identity[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    fit_q: list[int] = []
    fit_m: list[int] = []
    val_q: list[int] = []
    val_m: list[int] = []
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        group = rows[lo:hi]
        routes = route_per_observation[observation[group]]
        fit = group[np.isin(routes, sorted(fit_routes))][:MAXIMUM_QUERY_ROWS_PER_IDENTITY]
        validation = group[routes == str(validation_route)][:MAXIMUM_QUERY_ROWS_PER_IDENTITY]
        if len(fit) < 2:
            continue
        validation_modes = fit[_diverse_mode_indices(projected_features[fit], maximum_modes)]
        for query in fit.tolist():
            pool = fit[fit != query]
            modes = pool[_diverse_mode_indices(projected_features[pool], maximum_modes)]
            fit_q.extend([int(query)] * len(modes))
            fit_m.extend(modes.tolist())
        for query in validation.tolist():
            val_q.extend([int(query)] * len(validation_modes))
            val_m.extend(validation_modes.tolist())

    def finish(q: list[int], m: list[int]) -> tuple[np.ndarray, ...]:
        qrow = np.asarray(q, np.int64)
        mrow = np.asarray(m, np.int64)
        qobs = observation[qrow]
        pixel, depth = _project_world_to_pixel(
            world[mrow], poses_w2c[qobs], camera_matrices[qobs], radial_coefficients[qobs],
        )
        target = pixel - _token_centres(token_ids[qrow])
        distance = np.linalg.norm(world[qrow] - world[mrow], axis=1)
        keep = (
            (depth > 0.0) & np.isfinite(target).all(axis=1)
            & (np.max(np.abs(target), axis=1) < 2.0)
            & (distance <= MAXIMUM_POSITIVE_WORLD_DISTANCE_M)
        )
        return (
            qrow[keep], projected_features[mrow[keep]].astype(np.float32),
            world[mrow[keep]], target[keep].astype(np.float32),
        )

    fit = finish(fit_q, fit_m)
    validation = finish(val_q, val_m)
    return {
        "fit_query_rows": fit[0], "fit_map_features": fit[1],
        "fit_map_world": fit[2], "fit_targets": fit[3],
        "validation_query_rows": validation[0], "validation_map_features": validation[1],
        "validation_map_world": validation[2], "validation_targets": validation[3],
    }


def _fit_prototype_dataset(
    representative_rows: np.ndarray,
    representative_identity: np.ndarray,
    observation: np.ndarray,
    route_per_observation: np.ndarray,
    fit_routes: set[str],
    validation_route: str,
    world: np.ndarray,
    projected_features: np.ndarray,
    token_ids: np.ndarray,
    poses_w2c: np.ndarray,
    camera_matrices: np.ndarray,
    radial_coefficients: np.ndarray,
    source_view_per_observation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    order = np.lexsort((representative_rows, representative_identity))
    rows = representative_rows[order]; ids = representative_identity[order]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    fit_q, fit_mf, fit_mw, val_q, val_mf, val_mw = [], [], [], [], [], []
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        group = rows[lo:hi]
        routes = route_per_observation[observation[group]]
        fit = group[np.isin(routes, sorted(fit_routes))]
        validation = group[routes == str(validation_route)]
        if len(fit) < 2:
            continue
        fit = fit[:MAXIMUM_QUERY_ROWS_PER_IDENTITY]
        validation = validation[:MAXIMUM_QUERY_ROWS_PER_IDENTITY]
        if source_view_per_observation is not None:
            views = np.asarray(source_view_per_observation)[observation[fit]]
            def source_mean(pool):
                source = np.asarray(source_view_per_observation)[observation[pool]]
                unique = np.unique(source)
                if len(unique) < 2:
                    return None
                features = [_normalise(np.mean(projected_features[pool[source == v]], axis=0))
                            for v in unique]
                points = [np.mean(world[pool[source == v]], axis=0) for v in unique]
                return _normalise(np.mean(features, axis=0)), np.mean(points, axis=0)
            for query in fit.tolist():
                value = source_mean(fit[views != source_view_per_observation[observation[query]]])
                if value is not None:
                    fit_q.append(query); fit_mf.append(value[0]); fit_mw.append(value[1])
            value = source_mean(fit)
            if value is not None:
                for query in validation.tolist():
                    val_q.append(query); val_mf.append(value[0]); val_mw.append(value[1])
            continue
        total_feature = np.sum(projected_features[fit], axis=0, dtype=np.float64)
        total_world = np.sum(world[fit], axis=0, dtype=np.float64)
        for query in fit.tolist():
            mapping_feature = _normalise((total_feature - projected_features[query]) / (len(fit) - 1))
            mapping_world = (total_world - world[query]) / (len(fit) - 1)
            fit_q.append(query); fit_mf.append(mapping_feature); fit_mw.append(mapping_world)
        canonical_feature = _normalise(total_feature / len(fit))
        canonical_world = total_world / len(fit)
        for query in validation.tolist():
            val_q.append(query); val_mf.append(canonical_feature); val_mw.append(canonical_world)

    def finish(q: list[int], mf: list[np.ndarray], mw: list[np.ndarray]) -> tuple[np.ndarray, ...]:
        qrow = np.asarray(q, np.int64)
        map_feature = np.asarray(mf, np.float32).reshape(-1, projected_features.shape[1])
        map_world = np.asarray(mw, np.float64).reshape(-1, 3)
        qobs = observation[qrow]
        pixel, depth = _project_world_to_pixel(
            map_world, poses_w2c[qobs], camera_matrices[qobs], radial_coefficients[qobs],
        )
        target = pixel - _token_centres(token_ids[qrow])
        distance = np.linalg.norm(world[qrow] - map_world, axis=1)
        keep = (
            (depth > 0.0) & np.isfinite(target).all(axis=1)
            & (np.max(np.abs(target), axis=1) < 2.0)
            & (distance <= MAXIMUM_POSITIVE_WORLD_DISTANCE_M)
        )
        return qrow[keep], map_feature[keep], map_world[keep], target[keep].astype(np.float32)

    fit = finish(fit_q, fit_mf, fit_mw)
    validation = finish(val_q, val_mf, val_mw)
    return {
        "fit_query_rows": fit[0], "fit_map_features": fit[1],
        "fit_map_world": fit[2], "fit_targets": fit[3],
        "validation_query_rows": validation[0], "validation_map_features": validation[1],
        "validation_map_world": validation[2], "validation_targets": validation[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation_bank", type=Path, required=True)
    parser.add_argument("--visibility_atlas", type=Path, required=True)
    parser.add_argument("--planar_map", type=Path, required=True)
    parser.add_argument("--radio_projection", type=Path, required=True)
    parser.add_argument("--mapping_contributors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay_head", type=Path, help="Replay frozen weights; no optimizer steps.")
    parser.add_argument("--recalibrate_replay", action="store_true",
                        help="Frozen-weight transfer audit on new prototype inputs; refit mapping calibration only.")
    parser.add_argument("--match_only_finetune", action="store_true")
    parser.add_argument("--output_joint_residual_bank", type=Path)
    parser.add_argument('--joint_reprojection_weight',type=float,default=0.)
    parser.add_argument("--validation_route", default="seq9")
    parser.add_argument("--hidden_dimension", type=int, default=96)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260904)
    parser.add_argument(
        "--prototype_policy", choices=("canonical_mean", "diverse_view_modes", "source_view_modes"),
        default="canonical_mean",
    )
    parser.add_argument("--maximum_prototypes_per_cell", type=int, default=4)
    parser.add_argument("--source_isolated_nulls", action="store_true")
    parser.add_argument("--null_mining_policy", choices=("legacy", "pool_far", "radio_topk"), default="legacy")
    parser.add_argument("--local_neighbor_policy", choices=("canonical_mean", "atlas_modes_mean"),
                        default="canonical_mean")
    parser.add_argument("--local_neighbor_token_pooling", choices=("representative", "view_cell_mean"),
                        default="representative")
    parser.add_argument("--source_view_training_contract", action="store_true",
                        help="Source-isolated centers, two-view support, image-disjoint calibration.")
    parser.add_argument(
        "--calibrate_coordinate_shrinkage", action="store_true",
        help="Fit one scalar on half of the held mapping route and evaluate on the other half.",
    )
    parser.add_argument(
        "--calibrate_coordinate_affine", action="store_true",
        help="Fit a 2D affine calibration on half of the held mapping route and evaluate on the other half.",
    )
    parser.add_argument(
        "--predict_chart_uv", action="store_true",
        help="Jointly predict a continuous chart-UV correction and query sub-token coordinate.",
    )
    parser.add_argument(
        "--chart_uv_mixture_modes", type=int, default=1,
        help="Use a mapping-only chart-UV mixture posterior when greater than one.",
    )
    parser.add_argument(
        "--geometric_context", action="store_true",
        help="Condition on undistorted query ray, plane incidence, and prototype cell phase.",
    )
    parser.add_argument(
        "--deep_geometric_context", action="store_true",
        help="Use one frozen two-layer geometry-context capacity check.",
    )
    parser.add_argument(
        "--homography_context", action="store_true",
        help="Add a mapping-only coarse homography UV residual and validity bit.",
    )
    parser.add_argument(
        "--local_radio_correlation_context", action="store_true",
        help="Add a candidate-conditioned 3x3 query-by-chart RADIO correlation volume.",
    )
    parser.add_argument(
        "--reference_homography_head", type=Path,
        help="Frozen V11 head whose held mapping metrics the local head must dominate.",
    )
    args = parser.parse_args()
    if not np.isfinite(args.joint_reprojection_weight) or args.joint_reprojection_weight<0:
        raise ValueError('invalid joint coordinate loss weight')
    if args.joint_reprojection_weight and (not args.predict_chart_uv or args.chart_uv_mixture_modes!=1):
        raise ValueError('joint coordinate loss requires single candidate-anchored UV head')
    if args.joint_reprojection_weight and args.match_only_finetune:
        raise ValueError('joint coordinate loss is not a match-only training task')
    if args.recalibrate_replay and args.replay_head is None:
        raise ValueError("recalibration requires frozen replay weights")
    if args.match_only_finetune and not (args.replay_head is not None and args.recalibrate_replay):
        raise ValueError("match-only training requires a source checkpoint and mapping recalibration")
    if args.replay_head is not None:
        if args.output.exists() or (args.output_joint_residual_bank is not None
                                   and args.output_joint_residual_bank.exists()):
            raise FileExistsError("refusing to overwrite frozen replay artifacts")
        if args.output.resolve() == args.replay_head.resolve():
            raise ValueError("replay must not overwrite the source checkpoint")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite canonical subtoken head")
    if args.calibrate_coordinate_shrinkage and args.calibrate_coordinate_affine:
        raise ValueError("coordinate calibration modes are mutually exclusive")
    if args.predict_chart_uv and not (
        args.calibrate_coordinate_shrinkage or args.calibrate_coordinate_affine
    ):
        raise ValueError("surface-coordinate head requires a disjoint mapping calibration split")
    if int(args.maximum_prototypes_per_cell) < 1:
        raise ValueError("maximum prototypes per cell must be positive")
    if int(args.chart_uv_mixture_modes) < 1 or (
        int(args.chart_uv_mixture_modes) > 1 and not args.predict_chart_uv
    ):
        raise ValueError("chart-UV mixture modes require the surface-coordinate head")
    mixture_surface = bool(args.predict_chart_uv and int(args.chart_uv_mixture_modes) > 1)
    if args.geometric_context and (not args.predict_chart_uv or mixture_surface):
        raise ValueError("geometric context currently requires the single surface-coordinate head")
    if args.deep_geometric_context and not args.geometric_context:
        raise ValueError("deep geometric context requires geometric context")
    if args.homography_context and (
        not args.geometric_context or args.deep_geometric_context
    ):
        raise ValueError("homography context requires the single-layer geometric context head")
    if args.local_radio_correlation_context and not args.homography_context:
        raise ValueError("local RADIO correlation requires homography context")
    if args.local_neighbor_policy != "canonical_mean" and not args.local_radio_correlation_context:
        raise ValueError("local neighbor policy requires local RADIO correlation")
    if args.local_neighbor_token_pooling == "view_cell_mean" and (
        not args.local_radio_correlation_context or args.local_neighbor_policy != "atlas_modes_mean"
    ):
        raise ValueError("view-cell pooling requires atlas-mode local correlation")
    if args.local_radio_correlation_context != (args.reference_homography_head is not None):
        raise ValueError("local RADIO correlation requires exactly one frozen V11 reference")
    reference_meta = None
    if args.reference_homography_head is not None:
        _, reference_meta = load_mapping_subtoken_head(args.reference_homography_head)
        if (
            reference_meta.get("artifact_type")
            != "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11"
            or reference_meta.get("mapping_validation_gate_pass") is not True
        ):
            raise ValueError("local RADIO reference must be a validated V11 head")
        if bool(reference_meta.get("source_view_training_contract", False)) != args.source_view_training_contract:
            raise ValueError("frozen V11 source-view training contract differs")
        if bool(reference_meta.get("source_isolated_nulls", False)) != args.source_isolated_nulls:
            raise ValueError("frozen V11 negative sampling contract differs")
        if reference_meta.get("null_mining_policy", "legacy") != args.null_mining_policy:
            raise ValueError("frozen V11 negative mining policy differs")
    context_surface = bool(args.geometric_context)
    if args.source_view_training_contract and args.prototype_policy not in ("canonical_mean", "source_view_modes"):
        raise ValueError("source-view training contract requires canonical centers")
    if args.source_view_training_contract and args.local_radio_correlation_context and (
        args.local_neighbor_token_pooling != "view_cell_mean"
    ):
        raise ValueError("source-view local head requires view-cell pooling")

    visibility, visibility_meta = PlaneVisibilityAtlas.load_npz(args.visibility_atlas)
    planes = GeometryNativePlanarMap.load_npz(args.planar_map)
    bank, bank_meta = _load_observation_bank(args.observation_bank)
    projection, projection_meta = _load_projection(args.radio_projection)
    if (
        bank_meta.get("visibility_atlas_content_sha256") != visibility_meta.get("content_sha256")
        or projection_meta.get("observation_bank_content_sha256") != bank_meta.get("content_sha256")
        or visibility_meta.get("planar_map_file_sha256") != file_sha256(args.planar_map)
    ):
        raise ValueError("canonical subtoken lineage differs")
    offsets = np.asarray(bank["observation_offsets"], np.int64)
    world = np.asarray(bank["world_points"], np.float64)
    token_ids = np.asarray(bank["token_ids"], np.int64)
    observation = np.repeat(np.arange(len(visibility.view_names)), np.diff(offsets))
    observation_plane = np.repeat(np.arange(len(planes.plane_ids)), np.diff(visibility.plane_offsets))
    plane = observation_plane[observation]
    uv = np.empty((len(world), 2), np.float64)
    for row in range(len(planes.plane_ids)):
        selected = np.flatnonzero(plane == row)
        uv[selected] = (world[selected] - planes.centers_world[row]) @ planes.frames_world[row, :2].T
    cell = np.floor(uv / 0.5).astype(np.int64)
    identity_keys, identity = np.unique(np.c_[plane, cell], axis=0, return_inverse=True)
    route_per_observation = np.asarray([str(name).split("__", 1)[0] for name in visibility.view_names.astype(str)])
    all_routes = set(route_per_observation.tolist()); fit_routes = all_routes - {str(args.validation_route)}
    if str(args.validation_route) not in all_routes or not fit_routes:
        raise ValueError("validation route does not form a mapping-only split")

    camera_matrices = np.empty((len(visibility.view_names), 3, 3), np.float64)
    radial_coefficients = np.empty(len(visibility.view_names), np.float64)
    contributor_rows = []
    names = visibility.view_names.astype(str)
    _, source_view_per_observation = np.unique(names, return_inverse=True)
    for name in sorted(set(names.tolist())):
        path = args.mapping_contributors / name
        _, pose, matrix, k1 = _load_contributor_geometry(path)
        rows = np.flatnonzero(names == name)
        if not np.allclose(visibility.poses_w2c[rows], pose, atol=1e-12, rtol=0.0):
            raise ValueError("mapping contributor pose differs")
        camera_matrices[rows] = matrix; radial_coefficients[rows] = float(k1)
        contributor_rows.append((name, file_sha256(path)))
    contributor_inventory_sha = hashlib.sha256(json.dumps(contributor_rows, separators=(",", ":")).encode()).hexdigest()

    representatives, rep_identity = _representative_rows(
        identity, observation, uv, cell, np.ones(len(visibility.view_names), bool),
    )
    projected = np.zeros((len(world), projection.shape[0]), np.float16)
    projected[representatives] = _normalise(
        np.asarray(bank["radio_features"][representatives], np.float32) @ projection.T,
    ).astype(np.float16)
    if args.prototype_policy == "source_view_modes":
        if not args.source_view_training_contract:
            raise ValueError("anonymous mode centers require source-view isolation")
        all_projected=np.empty((len(world),projection.shape[0]),np.float32)
        for lo in range(0,len(world),8192):
            all_projected[lo:lo+8192]=_normalise(np.asarray(bank['radio_features'][lo:lo+8192],np.float32)@projection.T)
        dataset=_fit_source_view_mode_dataset(representatives,rep_identity,identity,observation,
            source_view_per_observation,route_per_observation,fit_routes,str(args.validation_route),world,
            all_projected,token_ids,visibility.poses_w2c,camera_matrices,radial_coefficients,
            int(args.maximum_prototypes_per_cell))
        del all_projected
    elif args.prototype_policy == "diverse_view_modes":
        dataset = _fit_mode_prototype_dataset(
            representatives, rep_identity, observation, route_per_observation,
            fit_routes, str(args.validation_route), world,
            projected.astype(np.float32), token_ids, visibility.poses_w2c,
            camera_matrices, radial_coefficients, int(args.maximum_prototypes_per_cell),
        )
    else:
        dataset = _fit_prototype_dataset(
            representatives, rep_identity, observation, route_per_observation,
            fit_routes, str(args.validation_route), world,
            projected.astype(np.float32), token_ids, visibility.poses_w2c,
            camera_matrices, radial_coefficients,
            source_view_per_observation=(source_view_per_observation
                                         if args.source_view_training_contract else None),
        )
    fit_q = dataset["fit_query_rows"]; val_q = dataset["validation_query_rows"]
    null_indices={}; null_valid={}; mining_audit={}
    for prefix,qrows in (("fit",fit_q),("validation",val_q)):
        if args.source_isolated_nulls:
            if args.prototype_policy != "source_view_modes":
                raise ValueError("isolated nulls require source-mode provenance during training")
            if args.null_mining_policy != "legacy":
                from feature_extract.tools.vfm.mapping_retrieved_negatives import mine
                null_indices[prefix],null_valid[prefix],mining_audit[prefix]=mine(
                    qrows,projected[qrows].astype(np.float32),source_view_per_observation[observation[qrows]],
                    identity[qrows],plane[qrows],world[qrows],dataset['negative_pool_features'],
                    dataset['negative_pool_world'],dataset['negative_pool_source'],dataset['negative_pool_identity'],
                    identity_keys[dataset['negative_pool_identity'],0],int(args.maximum_prototypes_per_cell),
                    MAXIMUM_POSITIVE_WORLD_DISTANCE_M,args.null_mining_policy)
            else:
                null_indices[prefix],null_valid[prefix]=_isolated_null_indices(qrows,
                    dataset[prefix+'_map_world'],dataset[prefix+'_map_source_views'],observation,
                    source_view_per_observation,identity,plane,world)
        else:
            if args.null_mining_policy != "legacy":
                raise ValueError("retrieved mining requires source-isolated nulls")
            null_valid[prefix]=np.ones(len(qrows),bool)
        if np.sum(null_valid[prefix]) < int(args.batch_size):
            raise ValueError("insufficient valid mapping negative pairs")
    if min(len(fit_q), len(val_q)) < int(args.batch_size):
        raise ValueError("insufficient deployment-matched mapping-only pairs")

    # Local nulls are formed by rolling canonical prototypes within each physical plane.
    def null_map_features(query_rows: np.ndarray, positive_map: np.ndarray) -> np.ndarray:
        if args.source_isolated_nulls:
            prefix="fit" if query_rows is fit_q else "validation"
            if args.null_mining_policy != "legacy":
                key='negative_pool_world' if positive_map is dataset[prefix+'_map_world'] else 'negative_pool_features'
                return dataset[key][null_indices[prefix]]
            return positive_map[null_indices[prefix]]
        result = np.empty_like(positive_map)
        query_plane = plane[query_rows]
        for value in np.unique(query_plane):
            selected = np.flatnonzero(query_plane == value)
            result[selected] = positive_map[np.roll(selected, 1)]
        return result

    fit_null_map = null_map_features(fit_q, dataset["fit_map_features"])
    val_null_map = null_map_features(val_q, dataset["validation_map_features"])
    fit_null_world = null_map_features(fit_q, dataset["fit_map_world"])
    val_null_world = null_map_features(val_q, dataset["validation_map_world"])
    null_overlap_audit={}
    for prefix,qrows,negative in (("fit",fit_q,fit_null_world),("validation",val_q,val_null_world)):
        frame=planes.frames_world[plane[qrows],:2]
        positive_uv=np.einsum('ni,nji->nj',dataset[prefix+'_map_world']-planes.centers_world[plane[qrows]],frame)
        negative_uv=np.einsum('ni,nji->nj',negative-planes.centers_world[plane[qrows]],frame)
        selected=null_valid[prefix]
        null_overlap_audit[prefix]={'valid_count':int(selected.sum()),
            'same_cell_fraction':float(np.mean(np.all(np.floor(positive_uv[selected]/.5)==np.floor(negative_uv[selected]/.5),axis=1))),
            'within_positive_distance_fraction':float(np.mean(np.linalg.norm(world[qrows[selected]]-negative[selected],axis=1)<=MAXIMUM_POSITIVE_WORLD_DISTANCE_M))}
        if args.prototype_policy == "source_view_modes":
            legacy=np.arange(len(qrows))
            for physical in np.unique(plane[qrows]):
                grouped=np.flatnonzero(plane[qrows]==physical)
                legacy[grouped]=np.roll(grouped,1)
            null_overlap_audit[prefix]['legacy_rolled_same_identity_fraction']=float(np.mean(identity[qrows]==identity[qrows[legacy]]))
            null_overlap_audit[prefix]['legacy_rolled_same_source_fraction']=float(np.mean(
                dataset[prefix+'_map_source_views'][legacy]==source_view_per_observation[observation[qrows]]))

    fit_homography_uv = val_homography_uv = None
    fit_homography_valid = val_homography_valid = None
    if args.homography_context:
        fit_homography_uv, fit_homography_valid = _mapping_homography_projection(
            fit_q, dataset["fit_map_world"], observation=observation,
            token_ids=token_ids, plane_rows=plane,
            plane_centers_world=planes.centers_world,
            plane_frames_world=planes.frames_world,
        )
        val_homography_uv, val_homography_valid = _mapping_homography_projection(
            val_q, dataset["validation_map_world"], observation=observation,
            token_ids=token_ids, plane_rows=plane,
            plane_centers_world=planes.centers_world,
            plane_frames_world=planes.frames_world,
        )

    def geometric_context(
        qrows: np.ndarray, map_world: np.ndarray,
        projected_uv: np.ndarray | None = None, valid: np.ndarray | None = None,
    ) -> np.ndarray:
        return _surface_geometric_context(
            qrows, map_world, observation=observation, token_ids=token_ids,
            plane_rows=plane, poses_w2c=visibility.poses_w2c,
            camera_matrices=camera_matrices, radial_coefficients=radial_coefficients,
            plane_centers_world=planes.centers_world,
            plane_frames_world=planes.frames_world,
            homography_projected_uv_m=projected_uv, homography_valid=valid,
        )

    fit_context = geometric_context(
        fit_q, dataset["fit_map_world"], fit_homography_uv, fit_homography_valid,
    ) if context_surface else None
    val_context = geometric_context(
        val_q, dataset["validation_map_world"], val_homography_uv, val_homography_valid,
    ) if context_surface else None
    fit_null_context = geometric_context(
        fit_q, fit_null_world, fit_homography_uv, fit_homography_valid,
    ) if context_surface else None
    val_null_context = geometric_context(
        val_q, val_null_world, val_homography_uv, val_homography_valid,
    ) if context_surface else None

    if args.local_radio_correlation_context:
        neighbor_observation = observation
        neighbor_routes = route_per_observation
        neighbor_representatives = representatives
        neighbor_identity = rep_identity
        if args.local_neighbor_token_pooling == "view_cell_mean":
            source_names, source_view = np.unique(visibility.view_names.astype(str), return_inverse=True)
            neighbor_observation = source_view[observation]
            neighbor_routes = np.asarray([name.split("__", 1)[0] for name in source_names])
            # Multiple plane observations may belong to one source view.
            # Their tokens form one anonymous view/cell mode, never duplicates.
            pairs = np.c_[rep_identity, neighbor_observation[representatives]]
            _, first = np.unique(pairs, axis=0, return_index=True)
            neighbor_representatives = representatives[first]
            neighbor_identity = rep_identity[first]
        neighbor_projected = (
            _view_cell_mean_features(bank["radio_features"], projection, identity,
                                     neighbor_observation, neighbor_representatives)
            if args.local_neighbor_token_pooling == "view_cell_mean"
            else projected.astype(np.float32)
        )
        fit_query_grid, fit_query_valid = _project_query_local_feature_grid(
            fit_q, observation=observation, token_ids=token_ids,
            raw_features=bank["radio_features"], projection=projection,
        )
        val_query_grid, val_query_valid = _project_query_local_feature_grid(
            val_q, observation=observation, token_ids=token_ids,
            raw_features=bank["radio_features"], projection=projection,
        )

        def local_context(
            qrows: np.ndarray, map_features: np.ndarray, map_world: np.ndarray,
            query_grid: np.ndarray, query_valid: np.ndarray,
            *, leave_query_observation_out: bool,
        ) -> np.ndarray:
            map_grid, map_valid = _mapping_local_feature_grid(
                qrows, map_features, map_world,
                leave_query_observation_out=leave_query_observation_out,
                representative_rows=neighbor_representatives,
                representative_identity=neighbor_identity,
                identity_keys_plane_cell=identity_keys,
                observation=neighbor_observation,
                route_per_observation=neighbor_routes,
                fit_routes=fit_routes,
                projected_features=neighbor_projected,
                plane_rows=plane,
                plane_centers_world=planes.centers_world,
                plane_frames_world=planes.frames_world,
                neighbor_policy=args.local_neighbor_policy,
                maximum_neighbor_modes=args.maximum_prototypes_per_cell,
                minimum_neighbor_views=2 if args.source_view_training_contract else 1,
            )
            return _local_radio_correlation_volume(
                query_grid, map_grid, query_valid, map_valid,
            )

        fit_local = local_context(
            fit_q, dataset["fit_map_features"], dataset["fit_map_world"],
            fit_query_grid, fit_query_valid, leave_query_observation_out=True,
        )
        fit_null_local = local_context(
            fit_q, fit_null_map, fit_null_world,
            fit_query_grid, fit_query_valid, leave_query_observation_out=True,
        )
        val_local = local_context(
            val_q, dataset["validation_map_features"], dataset["validation_map_world"],
            val_query_grid, val_query_valid, leave_query_observation_out=False,
        )
        val_null_local = local_context(
            val_q, val_null_map, val_null_world,
            val_query_grid, val_query_valid, leave_query_observation_out=False,
        )
        fit_context = np.c_[fit_context, fit_local].astype(np.float32)
        fit_null_context = np.c_[fit_null_context, fit_null_local].astype(np.float32)
        val_context = np.c_[val_context, val_local].astype(np.float32)
        val_null_context = np.c_[val_null_context, val_null_local].astype(np.float32)
        expected = MappingSurfaceCoordinateLocalCorrelationHead.CONTEXT_DIMENSION
        if any(value.shape[1] != expected for value in (
            fit_context, fit_null_context, val_context, val_null_context,
        )):
            raise AssertionError("local RADIO context dimension differs")

    def surface_targets(qrows: np.ndarray, map_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qobs = observation[qrows]
        pixel, depth = _project_world_to_pixel(
            world[qrows], visibility.poses_w2c[qobs], camera_matrices[qobs], radial_coefficients[qobs],
        )
        image = pixel - _token_centres(token_ids[qrows])
        chart_frame = planes.frames_world[plane[qrows], :2]
        chart_uv = np.einsum("ni,nji->nj", world[qrows] - map_world, chart_frame)
        if (
            np.any(depth <= 0.0) or not np.all(np.isfinite(image))
            or not np.all(np.isfinite(chart_uv))
            or np.any(np.max(np.abs(image), axis=1) >= SUBTOKEN_HALF_EXTENT_PX + 1e-6)
            or np.any(np.max(np.abs(chart_uv), axis=1) > 0.5 + 1e-6)
        ):
            raise ValueError("surface-coordinate targets exceed the frozen cell/token support")
        return image.astype(np.float32), chart_uv.astype(np.float32)

    fit_uv_targets = validation_uv_targets = None
    if args.predict_chart_uv:
        dataset["fit_targets"], fit_uv_targets = surface_targets(fit_q, dataset["fit_map_world"])
        dataset["validation_targets"], validation_uv_targets = surface_targets(
            val_q, dataset["validation_map_world"],
        )

    torch.manual_seed(int(args.seed)); np.random.seed(int(args.seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        MappingSurfaceCoordinateMixtureHead(
            projection.shape[0], int(args.hidden_dimension), int(args.chart_uv_mixture_modes),
        )
        if mixture_surface else
        MappingSurfaceCoordinateDeepContextHead(projection.shape[0], int(args.hidden_dimension))
        if args.deep_geometric_context else
        MappingSurfaceCoordinateLocalCorrelationHead(
            projection.shape[0], int(args.hidden_dimension),
        )
        if args.local_radio_correlation_context else
        MappingSurfaceCoordinateHomographyContextHead(
            projection.shape[0], int(args.hidden_dimension),
        )
        if args.homography_context else
        MappingSurfaceCoordinateContextHead(projection.shape[0], int(args.hidden_dimension))
        if context_surface else
        MappingSurfaceCoordinateHead(projection.shape[0], int(args.hidden_dimension))
        if args.predict_chart_uv else
        MappingSubtokenHead(projection.shape[0], int(args.hidden_dimension))
    ).to(device)
    replay_meta = None
    if args.replay_head is not None:
        frozen_model, replay_meta = load_mapping_subtoken_head(args.replay_head)
        if float(replay_meta.get('joint_reprojection_weight',0.))!=args.joint_reprojection_weight:
            raise ValueError('replay joint coordinate loss contract differs')
        if not args.source_view_training_contract or not args.predict_chart_uv or mixture_surface:
            raise ValueError("replay requires source-isolated single-UV surface head")
        for key, expected in (
            ("source_view_training_contract", True),
            ("seed", int(args.seed)),
            ("validation_mapping_route", str(args.validation_route)),
            ("prototype_policy", str(args.prototype_policy)),
            ("source_isolated_nulls", bool(args.source_isolated_nulls)),
            ("null_mining_policy", str(args.null_mining_policy)),
            ("local_radio_neighbor_policy", str(args.local_neighbor_policy)),
            ("local_radio_neighbor_token_pooling", str(args.local_neighbor_token_pooling)),
            ("visibility_atlas_file_sha256", file_sha256(args.visibility_atlas)),
            ("observation_bank_file_sha256", file_sha256(args.observation_bank)),
            ("planar_map_file_sha256", file_sha256(args.planar_map)),
            ("radio_projection_file_sha256", file_sha256(args.radio_projection)),
            ("mapping_contributor_inventory_sha256", contributor_inventory_sha),
        ):
            if key in ("prototype_policy", "source_isolated_nulls", "null_mining_policy") and args.recalibrate_replay:
                continue
            if key == "null_mining_policy" and replay_meta.get(key,"legacy") == expected:
                continue
            if key == "source_isolated_nulls" and bool(replay_meta.get(key, False)) == expected:
                continue
            if replay_meta.get(key) != expected:
                raise ValueError(f"frozen replay contract differs: {key}")
        model.load_state_dict(frozen_model.state_dict(), strict=True)
        if args.match_only_finetune:
            _freeze_coordinate_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed)); losses = []
    qfeature = projected
    for step in range(0 if args.replay_head is not None and not args.match_only_finetune else int(args.steps)):
        rows = rng.integers(0, len(fit_q), size=int(args.batch_size))
        q = torch.from_numpy(qfeature[fit_q[rows]].astype(np.float32)).to(device)
        m = torch.from_numpy(dataset["fit_map_features"][rows]).to(device)
        n = torch.from_numpy(fit_null_map[rows]).to(device)
        token = torch.from_numpy(token_ids[fit_q[rows]]).to(device)
        target = torch.from_numpy(dataset["fit_targets"][rows]).to(device)
        context = (
            torch.from_numpy(fit_context[rows]).to(device) if context_surface else None
        )
        null_context = (
            torch.from_numpy(fit_null_context[rows]).to(device) if context_surface else None
        )
        output = model(q, m, token, context) if context_surface else model(q, m, token)
        mean, variance = output[:2]
        positive_logit = output[5] if mixture_surface else output[4] if args.predict_chart_uv else output[2]
        negative_output = (
            model(q, n, token, null_context) if context_surface else model(q, n, token)
        )
        negative_logit = (
            negative_output[5] if mixture_surface
            else negative_output[4] if args.predict_chart_uv else negative_output[2]
        )
        error2 = torch.sum((mean - target) ** 2, dim=1, keepdim=True)
        nll = (0.5 * error2 / variance + torch.log(variance)).mean()
        if args.predict_chart_uv:
            uv_target = torch.from_numpy(fit_uv_targets[rows]).to(device)
            uv_mean, uv_variance = output[2:4]
            if mixture_surface:
                uv_logits = output[4]
                uv_error2 = torch.sum((uv_mean - uv_target[:, None, :]) ** 2, dim=2)
                component = (
                    F.log_softmax(uv_logits, dim=1) - torch.log(uv_variance)
                    - 0.5 * uv_error2 / uv_variance
                )
                nll = nll - torch.logsumexp(component, dim=1).mean()
            else:
                uv_error2 = torch.sum((uv_mean - uv_target) ** 2, dim=1, keepdim=True)
                nll = nll + (0.5 * uv_error2 / uv_variance + torch.log(uv_variance)).mean()
        loss = nll + 0.5 * F.smooth_l1_loss(mean, target, beta=0.25) + 0.5 * (
            F.binary_cross_entropy_with_logits(positive_logit, torch.ones_like(positive_logit))
            + (F.binary_cross_entropy_with_logits(negative_logit, torch.zeros_like(negative_logit),reduction='none').reshape(-1)
               * torch.from_numpy(null_valid['fit'][rows].astype(np.float32)).to(device)).sum()
              / max(int(np.sum(null_valid['fit'][rows])),1)
        )
        if args.joint_reprojection_weight:
            from feature_extract.tools.vfm.mapping_joint_coordinate_loss import joint_error
            qrows=fit_q[rows];qobs=observation[qrows];qplane=plane[qrows]
            pose=visibility.poses_w2c[qobs];anchor=dataset['fit_map_world'][rows]
            frame=planes.frames_world[qplane,:2]
            anchor_camera=np.einsum('nij,nj->ni',pose[:,:3,:3],anchor)+pose[:,:3,3]
            tangent_camera=pose[:,:3,:3]@frame.transpose(0,2,1)
            anchor_uv=np.einsum('ni,nji->nj',anchor-planes.centers_world[qplane],frame)
            lower=(np.floor(anchor_uv/.5)*.5-anchor_uv).astype(np.float32)
            upper=np.nextafter(lower+np.float32(.5),lower)
            tensors=[torch.from_numpy(np.asarray(v,np.float32)).to(device) for v in
                     [anchor_camera,tangent_camera,camera_matrices[qobs],radial_coefficients[qobs],lower,upper]]
            error=joint_error(mean,target,uv_mean,uv_target,*tensors)
            loss=loss+args.joint_reprojection_weight*F.smooth_l1_loss(error,torch.zeros_like(error),beta=.25)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{args.steps} loss={np.mean(losses[-200:]):.6f}", flush=True)

    def predict(
        qrows: np.ndarray, map_features: np.ndarray, null_features: np.ndarray,
        contexts: np.ndarray | None = None, null_contexts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        means=[]; variances=[]; uv_means=[]; uv_variances=[]; uv_logits=[]; pp=[]; npv=[]; model.eval()
        with torch.no_grad():
            for start in range(0, len(qrows), 8192):
                stop=min(start+8192,len(qrows))
                q=torch.from_numpy(qfeature[qrows[start:stop]].astype(np.float32)).to(device)
                m=torch.from_numpy(map_features[start:stop]).to(device)
                n=torch.from_numpy(null_features[start:stop]).to(device)
                token=torch.from_numpy(token_ids[qrows[start:stop]]).to(device)
                if context_surface:
                    if contexts is None or null_contexts is None:
                        raise ValueError("geometric contexts are required")
                    context = torch.from_numpy(contexts[start:stop]).to(device)
                    null_context = torch.from_numpy(null_contexts[start:stop]).to(device)
                    output=model(q,m,token,context); negative_output=model(q,n,token,null_context)
                else:
                    output=model(q,m,token); negative_output=model(q,n,token)
                mean,var=output[:2]
                logit=output[5] if mixture_surface else output[4] if args.predict_chart_uv else output[2]
                nlogit=(
                    negative_output[5] if mixture_surface
                    else negative_output[4] if args.predict_chart_uv else negative_output[2]
                )
                means.append(mean.cpu().numpy()); variances.append(var.cpu().numpy())
                if args.predict_chart_uv:
                    uv_means.append(output[2].cpu().numpy())
                    uv_variances.append(output[3].cpu().numpy())
                    if mixture_surface:
                        uv_logits.append(output[4].cpu().numpy())
                pp.append(torch.sigmoid(logit).cpu().numpy()); npv.append(torch.sigmoid(nlogit).cpu().numpy())
        values = (
            (means, variances, uv_means, uv_variances, uv_logits, pp, npv)
            if mixture_surface else
            (means, variances, uv_means, uv_variances, pp, npv)
            if args.predict_chart_uv else
            (means, variances, pp, npv)
        )
        prediction=list(np.concatenate(value) for value in values)
        prefix="fit" if qrows is fit_q else "validation"
        prediction[-1]=np.where(null_valid[prefix].reshape(prediction[-1].shape),prediction[-1],np.nan)
        return tuple(prediction)

    def evaluate(
        targets: np.ndarray,
        prediction: tuple[np.ndarray, ...],
        rows: np.ndarray | None = None,
        *,
        shrinkage: float = 1.0,
        affine_matrix: np.ndarray | None = None,
        affine_bias: np.ndarray | None = None,
    ) -> dict[str, object]:
        mean, variance = prediction[:2]
        pp, npv = prediction[-2:]
        selected = np.arange(len(targets)) if rows is None else np.asarray(rows, np.int64)
        calibrated = _apply_coordinate_calibration(
            mean[selected], shrinkage=shrinkage,
            affine_matrix=affine_matrix, affine_bias=affine_bias,
        )
        result=_metrics(targets[selected],calibrated,variance[selected])
        result["positive_match_probability_mean"]=float(np.mean(pp[selected]))
        result["negative_match_probability_mean"]=float(np.nanmean(npv[selected]))
        return result

    fit_prediction = predict(
        fit_q, dataset["fit_map_features"], fit_null_map, fit_context, fit_null_context,
    )
    validation_prediction = predict(
        val_q, dataset["validation_map_features"], val_null_map, val_context, val_null_context,
    )
    shrinkage = 1.0
    affine_matrix: np.ndarray | None = None
    affine_bias: np.ndarray | None = None
    scalar_reference_shrinkage = 1.0
    variance_scale = 1.0
    uv_variance_scale = 1.0
    uv_coordinate_shrinkage = 1.0
    zero_uv_variance = 1.0
    token_center_variance = TOKEN_SIZE_PX * TOKEN_SIZE_PX / 12.0
    shrinkage_calibration_count = 0
    validation_evaluation_rows: np.ndarray | None = None
    if args.calibrate_coordinate_shrinkage or args.calibrate_coordinate_affine:
        validation_observations = observation[val_q]
        if args.source_view_training_contract:
            validation_observations = source_view_per_observation[validation_observations]
        unique_observations = np.unique(validation_observations)
        calibration_observations = unique_observations[::2]
        evaluation_observations = unique_observations[1::2]
        calibration_rows = np.flatnonzero(np.isin(validation_observations, calibration_observations))
        validation_evaluation_rows = np.flatnonzero(np.isin(validation_observations, evaluation_observations))
        if min(len(calibration_rows), len(validation_evaluation_rows)) < int(args.batch_size):
            raise ValueError("insufficient disjoint mapping shrinkage calibration/evaluation pairs")
        raw_mean = validation_prediction[0][calibration_rows].astype(np.float64)
        target = dataset["validation_targets"][calibration_rows].astype(np.float64)
        scalar_reference_shrinkage = _closed_form_coordinate_shrinkage(raw_mean, target)
        if args.calibrate_coordinate_affine:
            affine_matrix, affine_bias = _closed_form_coordinate_affine(raw_mean, target)
        else:
            shrinkage = scalar_reference_shrinkage
        calibrated_calibration_mean = _apply_coordinate_calibration(
            raw_mean, shrinkage=shrinkage,
            affine_matrix=affine_matrix, affine_bias=affine_bias,
        )
        variance_scale = _isotropic_variance_scale(
            calibrated_calibration_mean, target, validation_prediction[1][calibration_rows],
        )
        token_center_variance = 0.5 * float(np.mean(np.sum(np.square(target), axis=1)))
        if args.predict_chart_uv:
            if mixture_surface:
                calibration_weight = np.exp(_mixture_log_probabilities(
                    validation_prediction[4][calibration_rows],
                ))
                calibration_mean = np.sum(
                    calibration_weight[:, :, None]
                    * validation_prediction[2][calibration_rows], axis=1,
                )
                uv_coordinate_shrinkage = _closed_form_coordinate_shrinkage(
                    calibration_mean, validation_uv_targets[calibration_rows],
                )
                uv_variance_scale = _mixture_variance_scale(
                    uv_coordinate_shrinkage * validation_prediction[2][calibration_rows],
                    validation_uv_targets[calibration_rows],
                    validation_prediction[3][calibration_rows],
                    validation_prediction[4][calibration_rows],
                )
            else:
                uv_coordinate_shrinkage = _closed_form_coordinate_shrinkage(
                    validation_prediction[2][calibration_rows],
                    validation_uv_targets[calibration_rows],
                )
                uv_variance_scale = _isotropic_variance_scale(
                    uv_coordinate_shrinkage * validation_prediction[2][calibration_rows],
                    validation_uv_targets[calibration_rows],
                    validation_prediction[3][calibration_rows],
                )
            zero_uv_variance = 0.5 * float(np.mean(np.sum(
                np.square(validation_uv_targets[calibration_rows]), axis=1,
            )))
        shrinkage_calibration_count = int(len(calibration_rows))
    calibrated_fit_prediction = list(fit_prediction)
    calibrated_fit_prediction[1] = variance_scale * calibrated_fit_prediction[1]
    if args.predict_chart_uv:
        calibrated_fit_prediction[3] = uv_variance_scale * calibrated_fit_prediction[3]
    calibrated_fit_prediction = tuple(calibrated_fit_prediction)
    calibrated_validation_prediction = list(validation_prediction)
    calibrated_validation_prediction[1] = variance_scale * calibrated_validation_prediction[1]
    if args.predict_chart_uv:
        calibrated_validation_prediction[3] = uv_variance_scale * calibrated_validation_prediction[3]
    calibrated_validation_prediction = tuple(calibrated_validation_prediction)
    fit_metrics=evaluate(
        dataset["fit_targets"], calibrated_fit_prediction, shrinkage=shrinkage,
        affine_matrix=affine_matrix, affine_bias=affine_bias,
    )
    validation_metrics=evaluate(
        dataset["validation_targets"], calibrated_validation_prediction,
        rows=validation_evaluation_rows, shrinkage=shrinkage,
        affine_matrix=affine_matrix, affine_bias=affine_bias,
    )
    evaluation_rows = (
        np.arange(len(val_q), dtype=np.int64)
        if validation_evaluation_rows is None else validation_evaluation_rows
    )
    validation_metrics["predicted_isotropic_gaussian_nll"] = _isotropic_gaussian_nll(
        _apply_coordinate_calibration(
            validation_prediction[0][evaluation_rows], shrinkage=shrinkage,
            affine_matrix=affine_matrix, affine_bias=affine_bias,
        ),
        dataset["validation_targets"][evaluation_rows],
        variance_scale * validation_prediction[1][evaluation_rows],
    )
    validation_metrics["calibrated_token_center_isotropic_gaussian_nll"] = _isotropic_gaussian_nll(
        np.zeros_like(dataset["validation_targets"][evaluation_rows]),
        dataset["validation_targets"][evaluation_rows],
        np.full(len(evaluation_rows), token_center_variance, np.float64),
    )
    surface_coordinate_gate = True
    if args.predict_chart_uv:
        if mixture_surface:
            fit_metrics["chart_uv_offset"] = _mixture_offset_metrics(
                fit_uv_targets, uv_coordinate_shrinkage * fit_prediction[2],
                uv_variance_scale * fit_prediction[3], fit_prediction[4],
            )
            validation_metrics["chart_uv_offset"] = _mixture_offset_metrics(
                validation_uv_targets[evaluation_rows],
                uv_coordinate_shrinkage * validation_prediction[2][evaluation_rows],
                uv_variance_scale * validation_prediction[3][evaluation_rows],
                validation_prediction[4][evaluation_rows],
            )
        else:
            fit_metrics["chart_uv_offset"] = _continuous_offset_metrics(
                fit_uv_targets, uv_coordinate_shrinkage * fit_prediction[2],
                uv_variance_scale * fit_prediction[3],
            )
            validation_metrics["chart_uv_offset"] = _continuous_offset_metrics(
                validation_uv_targets[evaluation_rows],
                uv_coordinate_shrinkage * validation_prediction[2][evaluation_rows],
                uv_variance_scale * validation_prediction[3][evaluation_rows],
            )
        surface = validation_metrics["chart_uv_offset"]
        if not mixture_surface:
            surface["predicted_isotropic_gaussian_nll"] = _isotropic_gaussian_nll(
                uv_coordinate_shrinkage * validation_prediction[2][evaluation_rows],
                validation_uv_targets[evaluation_rows],
                uv_variance_scale * validation_prediction[3][evaluation_rows],
            )
        surface["zero_offset_calibrated_isotropic_gaussian_nll"] = _isotropic_gaussian_nll(
            np.zeros_like(validation_uv_targets[evaluation_rows]),
            validation_uv_targets[evaluation_rows],
            np.full(len(evaluation_rows), zero_uv_variance, np.float64),
        )
        image_uncertainty_monotonic = bool(np.all(np.diff(
            validation_metrics["uncertainty_quantile_mean_error_px"],
        ) >= 0.0))
        if mixture_surface:
            uv_uncertainty_monotonic = bool(np.all(np.diff(
                surface["predictive_uncertainty_quantile_mean_error_m"],
            ) >= 0.0))
            surface_coordinate_gate = bool(
                surface["posterior_mean_relative_median_improvement"] >= 0.05
                and surface["posterior_mean_error_m"]["p90"]
                <= surface["zero_offset_error_m"]["p90"]
                and surface["posterior_mean_better_fraction"] > 0.5
                and surface["best_mode_relative_median_improvement"] >= 0.10
                and surface["mean_effective_mode_count"] >= 1.25
                and surface["mixture_isotropic_gaussian_nll"]
                <= surface["zero_offset_calibrated_isotropic_gaussian_nll"]
                and validation_metrics["predicted_isotropic_gaussian_nll"]
                <= validation_metrics["calibrated_token_center_isotropic_gaussian_nll"]
                and image_uncertainty_monotonic and uv_uncertainty_monotonic
            )
        else:
            uv_uncertainty_monotonic = bool(np.all(np.diff(
                surface["uncertainty_quantile_mean_error_m"],
            ) >= 0.0))
            surface_coordinate_gate = bool(
                surface["relative_median_improvement"] >= 0.05
                and surface["predicted_error_m"]["p90"] <= surface["zero_offset_error_m"]["p90"]
                and surface["predicted_better_fraction"] > 0.5
                and surface["predicted_isotropic_gaussian_nll"]
                <= surface["zero_offset_calibrated_isotropic_gaussian_nll"]
                and validation_metrics["predicted_isotropic_gaussian_nll"]
                <= validation_metrics["calibrated_token_center_isotropic_gaussian_nll"]
                and image_uncertainty_monotonic and uv_uncertainty_monotonic
            )
    affine_vs_scalar_gate = True
    if args.calibrate_coordinate_affine:
        scalar_reference = _metrics(
            dataset["validation_targets"][evaluation_rows],
            _apply_coordinate_calibration(
                validation_prediction[0][evaluation_rows],
                shrinkage=scalar_reference_shrinkage,
                affine_matrix=None, affine_bias=None,
            ),
            variance_scale * validation_prediction[1][evaluation_rows],
        )
        validation_metrics["scalar_calibration_reference"] = scalar_reference
        affine_vs_scalar_gate = bool(
            validation_metrics["predicted_error_px"]["median"]
            <= scalar_reference["predicted_error_px"]["median"]
            and validation_metrics["predicted_error_px"]["p90"]
            <= scalar_reference["predicted_error_px"]["p90"]
        )
    base_coordinate_gate = bool(
        validation_metrics["predicted_error_px"]["p90"]
        <= validation_metrics["token_centre_error_px"]["p90"]
        and validation_metrics["predicted_better_fraction"] > 0.5
        and validation_metrics["predicted_isotropic_gaussian_nll"]
        <= validation_metrics["calibrated_token_center_isotropic_gaussian_nll"]
    )
    if not args.predict_chart_uv:
        base_coordinate_gate = bool(
            base_coordinate_gate and validation_metrics["relative_median_improvement"] >= 0.05
        )
    local_reference_gate = True
    local_reference_summary = None
    if args.local_radio_correlation_context:
        assert reference_meta is not None
        if bool(reference_meta.get("source_view_training_contract", False)) != args.source_view_training_contract:
            raise ValueError("frozen V11 source-view training contract differs")
        if any(reference_meta.get(key) != value for key, value in (
            ("observation_bank_content_sha256", bank_meta.get("content_sha256")),
            ("visibility_atlas_content_sha256", visibility_meta.get("content_sha256")),
            ("planar_map_file_sha256", file_sha256(args.planar_map)),
            ("radio_projection_content_sha256", projection_meta.get("content_sha256")),
            ("fit_mapping_routes", sorted(fit_routes)),
            ("validation_mapping_route", str(args.validation_route)),
            ("validation_metrics_observation_partition", "odd_sorted_validation_source_views"
             if args.source_view_training_contract else "odd_sorted_validation_mapping_observations"),
            ("prototype_policy", str(args.prototype_policy)),
        )):
            raise ValueError("frozen V11 reference split or lineage differs")
        local_reference_gate, local_reference_summary = _local_correlation_reference_gate(
            validation_metrics, reference_meta["validation_metrics"],
        )
    gate=bool(base_coordinate_gate and affine_vs_scalar_gate
              and surface_coordinate_gate and local_reference_gate)
    arrays=_state_arrays(model)
    metadata: dict[str, object]={
        "artifact_type":(
            "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_mixture_head_v8"
            if mixture_surface
            else "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_deep_context_head_v10"
            if args.deep_geometric_context
            else "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_local_correlation_head_v12"
            if args.local_radio_correlation_context
            else "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_homography_context_head_v11"
            if args.homography_context
            else "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_context_head_v9"
            if context_surface
            else "goal_maplet_mapping_only_pairwise_radio_surface_coordinate_head_v7"
            if args.predict_chart_uv
            else "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v6"
            if args.calibrate_coordinate_affine
            else "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v5"
            if args.calibrate_coordinate_shrinkage
            else (
                "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v3"
                if args.prototype_policy == "diverse_view_modes"
                else "goal_maplet_mapping_only_pairwise_radio_subtoken_head_v2"
            )
        ),
        "feature_dimension":int(projection.shape[0]),"hidden_dimension":int(args.hidden_dimension),
        "coordinate_target":(
            "fit_query_token_to_leave_observation_out_fit_anonymous_mode_projection; validation_seq9_query_to_nonseq9_anonymous_mode_projection"
            if args.prototype_policy == "diverse_view_modes"
            else "fit_query_token_to_leave_observation_out_fit_canonical_prototype_projection; validation_seq9_query_to_nonseq9_canonical_prototype_projection"
        ),
        "canonical_prototype":(
            "all_token_source_view_cell_means_then_medoid_farthest_modes_with_own_mean_world_and_full_source_exclusion"
            if args.prototype_policy == "source_view_modes" else
            "atlas_exact_medoid_then_farthest_anonymous_view_mode_RADIO64_and_mode_specific_metric_world_point_per_finite_plane_0.5m_cell"
            if args.prototype_policy == "diverse_view_modes"
            else "l2_normalized_mean_RADIO64_and_mean_metric_world_point_per_finite_plane_0.5m_cell"
        ),
        "prototype_policy":str(args.prototype_policy),
        "source_isolated_nulls":bool(args.source_isolated_nulls),
        "null_mining_policy":str(args.null_mining_policy),
        "retrieved_negative_audit":mining_audit,
        "null_overlap_audit":null_overlap_audit,
        "null_valid_pair_counts":{k:int(v.sum()) for k,v in null_valid.items()},
        "center_mode_token_pooling":("all_tokens_per_source_view_cell" if args.prototype_policy == "source_view_modes" else "legacy_representative"),
        "center_mode_geometry_binding":("selected_mode_own_mean_world" if args.prototype_policy == "source_view_modes" else "legacy"),
        "center_mode_exclusion":("entire_source_image_before_mode_selection" if args.prototype_policy == "source_view_modes" else "legacy"),
        "mapping_recalibration_transfer_audit":bool(args.recalibrate_replay),
        "source_view_training_contract":bool(args.source_view_training_contract),
        "minimum_neighbor_independent_views":2 if args.source_view_training_contract else 1,
        "maximum_prototypes_per_cell":int(args.maximum_prototypes_per_cell),
        "chart_uv_mixture_modes":int(args.chart_uv_mixture_modes),
        "chart_uv_mixture_semantics":(
            "categorical_isotropic_Gaussian_components_trained_by_exact_logsumexp_NLL"
            if mixture_surface else "single_isotropic_Gaussian"
        ),
        "geometric_context_enabled":context_surface,
        "geometric_context_network_depth":2 if args.deep_geometric_context else 1 if context_surface else 0,
        "geometric_context_semantics":(
            "SIMPLE_RADIAL_undistorted_token_center_xy_plus_prototype_metric_cell_phase_uv_plus_absolute_query_plane_ray_incidence_plus_frozen_coarse_homography_normalized_UV_residual_and_validity_plus_candidate_conditioned_3x3_query_by_3x3_chart_RADIO64_cosine_volume_and_validity"
            if args.local_radio_correlation_context else
            "SIMPLE_RADIAL_undistorted_token_center_xy_plus_prototype_metric_cell_phase_uv_plus_absolute_query_plane_ray_incidence_plus_frozen_coarse_homography_normalized_UV_residual_and_validity"
            if args.homography_context else
            "SIMPLE_RADIAL_undistorted_token_center_xy_plus_prototype_metric_cell_phase_uv_plus_absolute_query_plane_ray_incidence"
            if context_surface else "none"
        ),
        "geometric_context_dimension":(
            MappingSurfaceCoordinateLocalCorrelationHead.CONTEXT_DIMENSION
            if args.local_radio_correlation_context else
            MappingSurfaceCoordinateHomographyContextHead.CONTEXT_DIMENSION if args.homography_context else
            MappingSurfaceCoordinateContextHead.CONTEXT_DIMENSION if context_surface else 0
        ),
        "homography_context_enabled":bool(args.homography_context),
        "homography_context_threshold_m":(
            HOMOGRAPHY_CONTEXT_THRESHOLD_M if args.homography_context else None
        ),
        "local_radio_correlation_context_enabled":bool(args.local_radio_correlation_context),
        "local_radio_correlation_context_dimension":(
            MappingSurfaceCoordinateLocalCorrelationHead.LOCAL_CORRELATION_DIMENSION
            if args.local_radio_correlation_context else 0
        ),
        "local_radio_correlation_offsets_yx":(
            LOCAL_CORRELATION_OFFSETS_YX.tolist()
            if args.local_radio_correlation_context else None
        ),
        "local_radio_correlation_semantics":(
            "row_major_query_3x3_by_chart_metric_cell_3x3_RADIO64_cosines_then_query_and_map_validity;fit_map_neighborhood_excludes_entire_query_observation;validation_map_uses_fit_routes_only"
            if args.local_radio_correlation_context else "none"
        ),
        "local_radio_candidate_conditioned":bool(args.local_radio_correlation_context),
        "local_radio_neighbor_policy":str(args.local_neighbor_policy),
        "local_radio_neighbor_token_pooling":str(args.local_neighbor_token_pooling),
        "local_radio_neighbor_exclusion_unit":(
            "entire_source_view" if args.local_neighbor_token_pooling == "view_cell_mean"
            else "plane_observation"
        ),
        "local_radio_neighbor_maximum_modes":int(args.maximum_prototypes_per_cell),
        "local_radio_reference_head_file_sha256":(
            file_sha256(args.reference_homography_head)
            if args.reference_homography_head is not None else None
        ),
        "local_radio_reference_head_content_sha256":(
            reference_meta.get("content_sha256") if reference_meta is not None else None
        ),
        "local_radio_v11_relative_gate_pass":(
            bool(local_reference_gate) if args.local_radio_correlation_context else None
        ),
        "local_radio_v11_relative_gate":local_reference_summary,
        "coordinate_shrinkage":float(shrinkage),
        "coordinate_affine_matrix":None if affine_matrix is None else affine_matrix.tolist(),
        "coordinate_affine_bias_px":None if affine_bias is None else affine_bias.tolist(),
        "measurement_variance_scale":float(variance_scale),
        "chart_uv_measurement_variance_scale":float(uv_variance_scale),
        "chart_uv_coordinate_shrinkage":float(uv_coordinate_shrinkage),
        "zero_uv_calibrated_variance_m2":float(zero_uv_variance),
        "token_center_calibrated_variance_px2":float(token_center_variance),
        "coordinate_shrinkage_calibration":(
            "closed_form_2d_affine_least_squares_on_even_sorted_validation_mapping_observations_only"
            if args.calibrate_coordinate_affine else
            "closed_form_scalar_least_squares_on_even_sorted_validation_mapping_observations_only"
            if args.calibrate_coordinate_shrinkage else "none"
        ),
        "coordinate_shrinkage_calibration_pair_count":int(shrinkage_calibration_count),
        "validation_metrics_observation_partition":(
            ("odd_sorted_validation_source_views" if args.source_view_training_contract
             else "odd_sorted_validation_mapping_observations")
            if args.calibrate_coordinate_shrinkage or args.calibrate_coordinate_affine
            else "all_validation_mapping_observations"
        ),
        "coordinate_output":"continuous_raw_image_pixel_offset_bounded_to_original_4x4_RADIO_token",
        "chart_uv_output":(
            "continuous_in_plane_metric_offset_from_anonymous_atlas_prototype_bounded_per_axis_to_0.5m_cell_then_mapping_calibrated_scalar_shrinkage"
            if args.predict_chart_uv else "none"
        ),
        "uncertainty_output":(
            "isotropic_query_centroid_variance_px2_and_chart_uv_variance_m2_not_map_surface_footprint"
            if args.predict_chart_uv else
            "isotropic_centroid_measurement_variance_px2_not_map_surface_footprint"
        ),
        "null_output":("pair_score_with_same_plane_different_cell_source_excluded_far_negatives_not_deployment_calibrated_probability"
                       if args.source_isolated_nulls else "pair_match_probability_with_same_plane_rolled_canonical_prototype_negatives"),
        "fit_mapping_routes":sorted(fit_routes),"validation_mapping_route":str(args.validation_route),
        "validation_prototype_routes":sorted(fit_routes),"validation_route_absent_from_prototypes":True,
        "fit_validation_route_disjoint":True,"query_rgb_pose_depth_or_ground_truth_read":False,
        "mapping_rgb_read_or_stored":False,"source_view_identity_retained_at_runtime":False,
        "fit_positive_pair_count":int(len(fit_q)),"validation_positive_pair_count":int(len(val_q)),
        "fit_metrics":fit_metrics,"validation_metrics":validation_metrics,
        "mapping_validation_gate_definition":(
            "base_surface_coordinate_gate_AND_frozen_V11_nonincrease_for_image_median_p90_NLL_chartUV_median_p90_NLL_AND_match_margin_nondecrease_AND_both_uncertainties_monotonic"
            if args.local_radio_correlation_context else
            "subpixel_NLL_better_AND_chartUV_posterior_mean_median_improvement>=5pct_AND_p90_nonincrease_AND_better_fraction>0.5_AND_best_mode_median_improvement>=10pct_AND_effective_modes>=1.25_AND_mixture_NLL_better_than_zero_offset_AND_both_uncertainties_monotonic"
            if mixture_surface else
            "subpixel_NLL_better_than_token_center_AND_p90_nonincrease_AND_uncertainty_monotonic_AND_chartUV_median_improvement>=5pct_AND_chartUV_NLL_better_than_zero_offset_AND_chartUV_p90_nonincrease_AND_chartUV_uncertainty_monotonic"
            if args.predict_chart_uv else
            "median_relative_improvement>=5pct_AND_p90_nonincrease_AND_better_fraction>0.5_AND_predicted_Gaussian_NLL<=calibrated_token_center_NLL_AND_if_affine_median_and_p90_noninferior_to_scalar"
        ),
        "mapping_validation_gate_pass":gate,"steps":int(args.steps),"batch_size":int(args.batch_size),"seed":int(args.seed),
        "observation_bank_file_sha256":file_sha256(args.observation_bank),"observation_bank_content_sha256":bank_meta.get("content_sha256"),
        "visibility_atlas_file_sha256":file_sha256(args.visibility_atlas),"visibility_atlas_content_sha256":visibility_meta.get("content_sha256"),
        "planar_map_file_sha256":file_sha256(args.planar_map),
        "radio_projection_file_sha256":file_sha256(args.radio_projection),"radio_projection_content_sha256":projection_meta.get("content_sha256"),
        "mapping_contributor_inventory_sha256":contributor_inventory_sha,"arrays_sha256":arrays_sha256(arrays),
    }
    metadata["content_sha256"]=canonical_json_sha256(metadata)
    if replay_meta is not None:
        if args.match_only_finetune:
            for name,value in model.state_dict().items():
                if not name.startswith('match.') and not torch.equal(value.cpu(),frozen_model.state_dict()[name].cpu()):
                    raise AssertionError('match-only training changed coordinate weights: '+name)
        for key in ("arrays_sha256", "coordinate_shrinkage", "measurement_variance_scale",
                    "chart_uv_measurement_variance_scale", "chart_uv_coordinate_shrinkage"):
            if args.match_only_finetune and key == "arrays_sha256":
                continue
            if args.recalibrate_replay and key != "arrays_sha256":
                continue
            if metadata[key] != replay_meta[key]:
                raise ValueError(f"frozen replay predictions/calibration differ: {key}")
        metadata["replay_head_content_sha256"] = replay_meta["content_sha256"]
        metadata["optimizer_steps_this_run"] = int(args.steps) if args.match_only_finetune else 0
        metadata["match_only_finetune"] = bool(args.match_only_finetune)
        metadata["nonmatch_weights_asserted_identical"] = bool(args.match_only_finetune)
        metadata.pop("content_sha256")
        metadata["content_sha256"] = canonical_json_sha256(metadata)
    if args.joint_reprojection_weight:
        metadata['joint_reprojection_weight']=float(args.joint_reprojection_weight)
        metadata['joint_reprojection_loss']='smooth_l1_joint_coordinate_error_on_same_anchor_tangent_sheet_not_raw_zero_reprojection'
        metadata.pop('content_sha256',None);metadata['content_sha256']=canonical_json_sha256(metadata)
    if args.output_joint_residual_bank is not None:
        if (validation_evaluation_rows is None or not args.source_view_training_contract
                or not args.predict_chart_uv or mixture_surface):
            raise ValueError("joint residual export requires source-isolated single-UV disjoint calibration")
        from feature_extract.tools.vfm.calibrate_goal_maplet_joint_reprojection import export_bank
        export_bank(
            args.output_joint_residual_bank, validation_prediction, shrinkage,
            variance_scale, uv_coordinate_shrinkage, uv_variance_scale,
            dataset["validation_map_world"], world[val_q], plane[val_q], planes,
            _token_centres(token_ids[val_q]), observation[val_q],
            source_view_per_observation[observation[val_q]], visibility.poses_w2c,
            camera_matrices, radial_coefficients, calibration_rows,
            validation_evaluation_rows, (replay_meta["content_sha256"]
                if replay_meta is not None and not args.recalibrate_replay else metadata["content_sha256"]),
        )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    print(json.dumps(metadata,indent=2))


if __name__ == "__main__":
    main()
