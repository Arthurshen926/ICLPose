"""Descriptor-aligned landmark banks from real-image spatial context grids.

This module is intentionally small and target-free.  It samples a frozen
per-image descriptor grid at the SfM support observations, then aggregates
only observations belonging to the already-fixed physical-track universe.
It is the valid path for a raw backbone descriptor space: query descriptors
and landmark descriptors are both produced by full-image grid sampling.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)
from feature_extract.vfm.localization.pose_conditioned_support_alignment import (
    bilinear_sample_image_grid,
)
from feature_extract.vfm.localization.spatial_image_context import (
    SpatialImageContextCache,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


def sample_spatial_context_descriptors(
    cache: SpatialImageContextCache,
    *,
    image_ids: Sequence[str] | np.ndarray,
    xy: np.ndarray,
    grid_size: int,
    boundary_mode: str = "error",
) -> np.ndarray:
    """Bilinearly sample and L2-normalize real-image grid descriptors.

    The coordinate convention is the same pixel-endpoint-to-grid-endpoint
    ``align_corners=True`` convention used by the existing appearance probes.
    No coordinates are projected from a query pose in this function.
    """

    requested = np.asarray(image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if len(requested) != len(coordinates) or not len(requested):
        raise ValueError("context descriptor samples require aligned non-empty rows")
    if np.any(requested == "") or np.any(~np.isfinite(coordinates)):
        raise ValueError("context descriptor sample inputs are invalid")
    if str(boundary_mode) not in {"error", "border"}:
        raise ValueError("context descriptor boundary_mode must be 'error' or 'border'")
    output = np.empty((len(requested), cache.descriptor_dim), dtype=np.float32)
    for image_id in sorted(set(requested.tolist())):
        rows = np.flatnonzero(requested == str(image_id))
        grid, image_size = cache.image_grid_descriptors(
            str(image_id), grid_size=int(grid_size)
        )
        sampled_xy = coordinates[rows]
        width, height = (int(value) for value in image_size)
        out_of_bounds = (
            (sampled_xy[:, 0] < 0.0)
            | (sampled_xy[:, 0] > float(width - 1))
            | (sampled_xy[:, 1] < 0.0)
            | (sampled_xy[:, 1] > float(height - 1))
        )
        if np.any(out_of_bounds) and str(boundary_mode) == "error":
            raise ValueError("image-grid coordinates are outside image bounds")
        if str(boundary_mode) == "border":
            sampled_xy = np.clip(
                sampled_xy,
                np.asarray([0.0, 0.0], dtype=np.float32),
                np.asarray([float(width - 1), float(height - 1)], dtype=np.float32),
            )
        output[rows] = bilinear_sample_image_grid(
            grid,
            sampled_xy,
            image_size=image_size,
        )
    output, valid = normalize_rows(output)
    if not np.all(valid):
        raise RuntimeError("context grid sampling produced a zero descriptor")
    return output


def spatial_context_boundary_audit(
    cache: SpatialImageContextCache,
    *,
    image_ids: Sequence[str] | np.ndarray,
    xy: np.ndarray,
    grid_size: int,
) -> tuple[int, float]:
    """Count out-of-frame samples before an explicit border-clamp decision."""

    requested = np.asarray(image_ids).astype(str).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if len(requested) != len(coordinates) or not len(requested):
        raise ValueError("context boundary audit requires aligned non-empty rows")
    count = 0
    maximum = 0.0
    for image_id in sorted(set(requested.tolist())):
        rows = np.flatnonzero(requested == str(image_id))
        _grid, image_size = cache.image_grid_descriptors(
            str(image_id), grid_size=int(grid_size)
        )
        width, height = (int(value) for value in image_size)
        values = coordinates[rows]
        excursion = np.maximum.reduce(
            (
                -values[:, 0],
                values[:, 0] - float(width - 1),
                -values[:, 1],
                values[:, 1] - float(height - 1),
                np.zeros((len(values),), dtype=np.float32),
            )
        )
        count += int(np.sum(excursion > 0.0))
        maximum = max(maximum, float(np.max(excursion, initial=0.0)))
    return count, maximum


def canonical_track_rows(
    track_ids: np.ndarray,
    canonical_track_ids: np.ndarray,
) -> np.ndarray:
    """Resolve arbitrary physical track IDs into a one-row-per-track bank."""

    requested = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    canonical = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    if not len(canonical) or np.unique(canonical).size != len(canonical):
        raise ValueError("canonical track IDs must be unique and non-empty")
    order = np.argsort(canonical, kind="stable")
    sorted_tracks = canonical[order]
    output = np.full((len(requested),), -1, dtype=np.int64)
    valid = requested >= 0
    if not np.any(valid):
        return output
    positions = np.searchsorted(sorted_tracks, requested[valid])
    in_range = positions < len(sorted_tracks)
    matched = np.zeros_like(in_range, dtype=bool)
    matched[in_range] = (
        sorted_tracks[positions[in_range]] == requested[valid][in_range]
    )
    valid_positions = np.flatnonzero(valid)
    output[valid_positions[matched]] = order[positions[matched]]
    return output


def aggregate_normalized_observations(
    *,
    canonical_rows: np.ndarray,
    descriptors: np.ndarray,
    landmark_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate unit observation descriptors with a normalized track mean."""

    rows = np.asarray(canonical_rows, dtype=np.int64).reshape(-1)
    values, valid = normalize_rows(np.asarray(descriptors, dtype=np.float32))
    if (
        values.ndim != 2
        or len(rows) != len(values)
        or int(landmark_count) <= 0
        or np.any(rows < 0)
        or np.any(rows >= int(landmark_count))
        or not np.all(valid)
    ):
        raise ValueError("normalized observation aggregation inputs are invalid")
    sums = np.zeros((int(landmark_count), values.shape[1]), dtype=np.float32)
    counts = np.zeros((int(landmark_count),), dtype=np.int64)
    np.add.at(sums, rows, values)
    np.add.at(counts, rows, 1)
    if np.any(counts <= 0):
        raise ValueError("every canonical landmark must receive an observation")
    unnormalized = sums / counts[:, None].astype(np.float32)
    means, valid_means = normalize_rows(unnormalized)
    if not np.all(valid_means):
        raise RuntimeError("track aggregation produced a zero descriptor")
    resultant = np.linalg.norm(unnormalized, axis=1)
    variances = (1.0 - np.clip(resultant, 0.0, 1.0)).astype(np.float32)
    return means, counts, variances


def build_context_observation_landmark_index(
    *,
    cache: SpatialImageContextCache,
    geometry: SupportObservationGeometryIndex,
    source_landmark_index: LandmarkMapIndex,
    support_image_ids: Sequence[str],
    grid_size: int,
) -> tuple[LandmarkMapIndex, dict[str, int | float]]:
    """Build one raw-context descriptor bank over a fixed track universe."""

    support_ids = tuple(sorted({str(value) for value in support_image_ids}))
    if not support_ids or tuple(geometry.image_ids) != support_ids:
        raise ValueError("support image manifest and geometry index differ")
    if np.unique(source_landmark_index.track_ids).size != len(source_landmark_index):
        raise ValueError("source landmark index must have one row per physical track")
    if int(grid_size) not in cache.grids:
        raise ValueError("requested context grid is absent from the cache")

    count = len(source_landmark_index)
    descriptor_sums = np.zeros((count, cache.descriptor_dim), dtype=np.float32)
    observation_counts = np.zeros((count,), dtype=np.int64)
    sampled_observation_count = 0
    excluded_observation_count = 0
    border_clamped_observation_count = 0
    maximum_boundary_excursion_px = 0.0
    for image_id in support_ids:
        image_slice = geometry.image_slice(image_id)
        tracks = geometry.track_ids[image_slice]
        if not len(tracks):
            continue
        rows = canonical_track_rows(tracks, source_landmark_index.track_ids)
        included = rows >= 0
        excluded_observation_count += int(np.sum(~included))
        if not np.any(included):
            continue
        sample_xy = geometry.xy[image_slice][included]
        sample_ids = np.asarray(
            [str(image_id)] * int(np.sum(included)), dtype=np.str_
        )
        clamped_count, maximum_excursion = spatial_context_boundary_audit(
            cache,
            image_ids=sample_ids,
            xy=sample_xy,
            grid_size=int(grid_size),
        )
        if maximum_excursion > 1.0:
            raise ValueError(
                "mapping observation coordinates exceed the declared image frame by "
                f"{maximum_excursion:.3f}px"
            )
        samples = sample_spatial_context_descriptors(
            cache,
            image_ids=sample_ids,
            xy=sample_xy,
            grid_size=int(grid_size),
            boundary_mode="border",
        )
        np.add.at(descriptor_sums, rows[included], samples)
        np.add.at(observation_counts, rows[included], 1)
        sampled_observation_count += int(len(samples))
        border_clamped_observation_count += int(clamped_count)
        maximum_boundary_excursion_px = max(
            maximum_boundary_excursion_px, float(maximum_excursion)
        )

    if not np.array_equal(observation_counts, source_landmark_index.observation_counts):
        mismatched = int(np.sum(observation_counts != source_landmark_index.observation_counts))
        raise ValueError(
            "context observation counts differ from the fixed source bank "
            f"for {mismatched} tracks"
        )
    unnormalized = descriptor_sums / observation_counts[:, None].astype(np.float32)
    features, valid = normalize_rows(unnormalized)
    if not np.all(valid):
        raise RuntimeError("context observation landmark bank has a zero prototype")
    resultant = np.linalg.norm(unnormalized, axis=1)
    variances = (1.0 - np.clip(resultant, 0.0, 1.0)).astype(np.float32)
    index = LandmarkMapIndex(
        track_ids=source_landmark_index.track_ids,
        xyz=source_landmark_index.xyz,
        features=features,
        mean_variances=variances,
        observation_counts=observation_counts,
        observation_image_ids=source_landmark_index.observation_image_ids,
        reprojection_errors=source_landmark_index.reprojection_errors,
        feature_ambiguities=source_landmark_index.feature_ambiguities,
        prototype_ids=source_landmark_index.prototype_ids,
    )
    return index, {
        "sampled_observation_count": int(sampled_observation_count),
        "excluded_observation_count": int(excluded_observation_count),
        "source_landmark_count": int(count),
        "mean_track_variance": float(np.mean(variances)),
        "border_clamped_observation_count": int(border_clamped_observation_count),
        "maximum_boundary_excursion_px": float(maximum_boundary_excursion_px),
    }
