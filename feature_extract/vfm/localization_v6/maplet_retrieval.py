"""Candidate-group RADIO-final maplet retrieval with exact omitted mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    V6ProbabilityCalibration,
)


@dataclass(frozen=True)
class QueryMapletGroup:
    query_region_xy: np.ndarray
    query_region_extent: np.ndarray
    maplet_ids: np.ndarray
    probabilities: np.ndarray
    null_probability: float
    omitted_probability: float
    candidate_centers: np.ndarray | None = None
    candidate_covariances: np.ndarray | None = None
    component_offsets: np.ndarray | None = None
    component_centers: np.ndarray | None = None
    component_covariances: np.ndarray | None = None
    component_probabilities: np.ndarray | None = None
    spatial_available: np.ndarray | None = None
    spatial_null_probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class MapletRetrievalResult:
    groups: tuple[QueryMapletGroup, ...]
    ranked_maplet_ids: np.ndarray
    evidence: np.ndarray
    scene_evidence_aggregation: str = "legacy_sum"


def _restrict_group_to_maplets(
    group: QueryMapletGroup,
    retained_maplet_ids: np.ndarray,
) -> QueryMapletGroup:
    """Apply the scene-level Top-K decision to one regional posterior.

    Retrieval first needs a wider per-region pool to accumulate scene-level
    evidence.  Coarse pose, however, must not silently keep sampling that
    preliminary pool after the scene-level maplet set has been selected.
    Probability removed here is transferred to unknown/null.
    """

    retained = np.asarray(retained_maplet_ids, dtype=np.int64).reshape(-1)
    keep = np.isin(
        np.asarray(group.maplet_ids, dtype=np.int64),
        retained,
        assume_unique=False,
    )
    if np.all(keep):
        return group
    probabilities = np.asarray(group.probabilities, dtype=np.float64)
    removed_probability = float(np.sum(probabilities[~keep]))
    component_offsets = None
    component_centers = None
    component_covariances = None
    component_probabilities = None
    if group.component_offsets is not None:
        if (
            group.component_centers is None
            or group.component_covariances is None
            or group.component_probabilities is None
        ):
            raise ValueError("component geometry fields must be stored together")
        old_offsets = np.asarray(group.component_offsets, dtype=np.int64)
        selected_components = [
            np.arange(
                int(old_offsets[row]),
                int(old_offsets[row + 1]),
                dtype=np.int64,
            )
            for row in np.flatnonzero(keep).tolist()
        ]
        lengths = np.asarray(
            [value.size for value in selected_components], dtype=np.int64
        )
        component_offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        component_rows = (
            np.concatenate(selected_components)
            if selected_components
            else np.zeros((0,), dtype=np.int64)
        )
        component_centers = np.asarray(group.component_centers)[component_rows]
        component_covariances = np.asarray(group.component_covariances)[
            component_rows
        ]
        component_probabilities = np.asarray(group.component_probabilities)[
            component_rows
        ]
    return QueryMapletGroup(
        query_region_xy=group.query_region_xy,
        query_region_extent=group.query_region_extent,
        maplet_ids=np.asarray(group.maplet_ids)[keep],
        probabilities=np.asarray(group.probabilities)[keep],
        null_probability=float(
            np.clip(group.null_probability + removed_probability, 0.0, 1.0)
        ),
        omitted_probability=float(
            np.clip(group.omitted_probability + removed_probability, 0.0, 1.0)
        ),
        candidate_centers=(
            None
            if group.candidate_centers is None
            else np.asarray(group.candidate_centers)[keep]
        ),
        candidate_covariances=(
            None
            if group.candidate_covariances is None
            else np.asarray(group.candidate_covariances)[keep]
        ),
        component_offsets=component_offsets,
        component_centers=component_centers,
        component_covariances=component_covariances,
        component_probabilities=component_probabilities,
        spatial_available=(
            None
            if group.spatial_available is None
            else np.asarray(group.spatial_available, dtype=bool)[keep]
        ),
        spatial_null_probabilities=(
            None
            if group.spatial_null_probabilities is None
            else np.asarray(
                group.spatial_null_probabilities, dtype=np.float32
            )[keep]
        ),
    )


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def _compress_surface_location_modes(
    probabilities: np.ndarray,
    centers: np.ndarray,
    covariances: np.ndarray,
    *,
    maximum_modes: int,
    nms_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Merge nearby samples, then return capacity-truncated mass as null.

    NMS and capacity truncation are different probability events.  Samples
    inside one metric neighbourhood describe one wider resolved mode, so
    their mass and moments are merged.  Only complete, mutually distinct
    clusters removed by ``maximum_modes`` become spatial-null mass.
    """

    probability = np.maximum(
        np.asarray(probabilities, dtype=np.float64).reshape(-1), 0.0
    )
    center = np.asarray(centers, dtype=np.float64)
    covariance = np.asarray(covariances, dtype=np.float64)
    if (
        center.shape != (probability.size, 3)
        or covariance.shape != (probability.size, 3, 3)
        or probability.size == 0
    ):
        raise ValueError("surface-location mixture shapes differ")
    # Multiple appearance modes at one atlas texel share one physical
    # location.  Aggregate their probability before spatial NMS so they do
    # not consume several pose modes.
    _unique, inverse = np.unique(
        np.round(center, decimals=6), axis=0, return_inverse=True
    )
    mode_count = int(np.max(inverse)) + 1
    aggregate_probability = np.bincount(
        inverse, weights=probability, minlength=mode_count
    )
    aggregate_center = np.zeros((mode_count, 3), dtype=np.float64)
    np.add.at(aggregate_center, inverse, probability[:, None] * center)
    aggregate_center /= np.maximum(aggregate_probability[:, None], 1e-12)
    aggregate_covariance = np.zeros((mode_count, 3, 3), dtype=np.float64)
    for component in range(probability.size):
        mode = int(inverse[component])
        residual = center[component] - aggregate_center[mode]
        aggregate_covariance[mode] += probability[component] * (
            covariance[component] + np.outer(residual, residual)
        )
    aggregate_covariance /= np.maximum(
        aggregate_probability[:, None, None], 1e-12
    )
    separation = max(float(nms_distance_m), 0.0)
    unassigned = np.ones(mode_count, dtype=bool)
    cluster_probability = []
    cluster_center = []
    cluster_covariance = []
    for seed in np.argsort(
        -aggregate_probability, kind="mergesort"
    ).tolist():
        if not bool(unassigned[seed]):
            continue
        if separation > 0.0:
            members = np.flatnonzero(
                unassigned
                & (
                    np.linalg.norm(
                        aggregate_center - aggregate_center[seed],
                        axis=1,
                    )
                    < separation
                )
            )
        else:
            members = np.asarray([seed], dtype=np.int64)
        unassigned[members] = False
        local_probability = aggregate_probability[members]
        total_probability = float(np.sum(local_probability))
        mean = np.sum(
            local_probability[:, None] * aggregate_center[members],
            axis=0,
        ) / max(total_probability, 1e-12)
        residual = aggregate_center[members] - mean[None]
        covariance_value = np.sum(
            local_probability[:, None, None]
            * (
                aggregate_covariance[members]
                + residual[:, :, None] * residual[:, None, :]
            ),
            axis=0,
        ) / max(total_probability, 1e-12)
        cluster_probability.append(total_probability)
        cluster_center.append(mean)
        cluster_covariance.append(covariance_value)
    clustered_probability = np.asarray(
        cluster_probability, dtype=np.float64
    )
    clustered_center = np.asarray(cluster_center, dtype=np.float64)
    clustered_covariance = np.asarray(
        cluster_covariance, dtype=np.float64
    )
    limit = min(max(int(maximum_modes), 1), clustered_probability.size)
    selected_rows = np.argsort(
        -clustered_probability, kind="mergesort"
    )[:limit]
    selected_probability = clustered_probability[selected_rows]
    discarded_probability = max(
        float(np.sum(clustered_probability) - np.sum(selected_probability)),
        0.0,
    )
    return (
        clustered_center[selected_rows].astype(np.float32),
        clustered_covariance[selected_rows].astype(np.float32),
        selected_probability.astype(np.float32),
        discarded_probability,
    )


def _rectangle_iou(
    center: np.ndarray,
    extent: np.ndarray,
    selected_centers: np.ndarray,
    selected_extents: np.ndarray,
) -> np.ndarray:
    lower = center - extent
    upper = center + extent
    selected_lower = selected_centers - selected_extents
    selected_upper = selected_centers + selected_extents
    intersection = np.maximum(
        np.minimum(upper[None], selected_upper)
        - np.maximum(lower[None], selected_lower),
        0.0,
    )
    intersection_area = intersection[:, 0] * intersection[:, 1]
    area = float(np.prod(np.maximum(2.0 * extent, 0.0)))
    selected_area = np.prod(
        np.maximum(2.0 * selected_extents, 0.0), axis=1
    )
    return intersection_area / np.maximum(
        area + selected_area - intersection_area, 1e-8
    )


def aggregate_scene_maplet_evidence(
    groups: tuple[QueryMapletGroup, ...] | list[QueryMapletGroup],
    maplet_ids: np.ndarray,
    *,
    method: str = "topq_nms",
    top_q: int = 4,
    overlap_iou: float = 0.30,
    block_grid_shape: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Aggregate independent regional support for each scene maplet.

    Overlapping RADIO regions are correlated measurements.  All non-legacy
    methods first suppress overlapping rectangles independently per maplet.
    ``topq_nms`` caps the remaining support count, ``noisy_or_nms`` computes
    a bounded union probability, and ``block_balanced`` allows at most one
    contribution per fixed image block.
    """

    allowed = {
        "legacy_sum",
        "topq_nms",
        "noisy_or_nms",
        "block_balanced",
    }
    name = str(method)
    if name not in allowed:
        raise ValueError(f"unknown scene evidence aggregation: {name}")
    ids = np.asarray(maplet_ids, dtype=np.int64).reshape(-1)
    row_by_id = {
        int(value): int(row) for row, value in enumerate(ids.tolist())
    }
    observations: list[list[tuple[float, np.ndarray, np.ndarray]]] = [
        [] for _ in range(ids.size)
    ]
    for group in groups:
        center = np.asarray(group.query_region_xy, dtype=np.float64)
        extent = np.maximum(
            np.asarray(group.query_region_extent, dtype=np.float64), 1e-3
        )
        for maplet_id, probability in zip(
            np.asarray(group.maplet_ids, dtype=np.int64).tolist(),
            np.asarray(group.probabilities, dtype=np.float64).tolist(),
        ):
            row = row_by_id.get(int(maplet_id))
            if row is not None and float(probability) > 0.0:
                observations[row].append(
                    (float(probability), center, extent)
                )
    evidence = np.zeros(ids.size, dtype=np.float64)
    if name == "legacy_sum":
        for row, values in enumerate(observations):
            evidence[row] = sum(value[0] for value in values)
        return evidence
    if not groups:
        return evidence
    all_centers = np.asarray(
        [group.query_region_xy for group in groups], dtype=np.float64
    )
    all_extents = np.asarray(
        [group.query_region_extent for group in groups], dtype=np.float64
    )
    image_upper = np.max(
        all_centers + np.maximum(all_extents, 0.0), axis=0
    )
    block_width = max(int(block_grid_shape[0]), 1)
    block_height = max(int(block_grid_shape[1]), 1)
    for row, values in enumerate(observations):
        if not values:
            continue
        values.sort(key=lambda value: -value[0])
        retained: list[tuple[float, np.ndarray, np.ndarray]] = []
        for value in values:
            if retained:
                iou = _rectangle_iou(
                    value[1],
                    value[2],
                    np.asarray([item[1] for item in retained]),
                    np.asarray([item[2] for item in retained]),
                )
                if np.any(iou > float(overlap_iou)):
                    continue
            retained.append(value)
        probability = np.asarray(
            [value[0] for value in retained], dtype=np.float64
        )
        if name == "topq_nms":
            evidence[row] = float(
                np.sum(probability[: max(int(top_q), 1)])
            )
        elif name == "noisy_or_nms":
            evidence[row] = float(
                1.0 - np.prod(1.0 - np.clip(probability, 0.0, 1.0))
            )
        else:
            per_block: dict[tuple[int, int], float] = {}
            for probability_value, center, _extent in retained:
                normalized = center / np.maximum(image_upper, 1.0)
                block = (
                    max(
                        0,
                        min(
                            int(
                                np.floor(
                                    normalized[0] * block_width
                                )
                            ),
                            block_width - 1,
                        ),
                    ),
                    max(
                        0,
                        min(
                            int(
                                np.floor(
                                    normalized[1] * block_height
                                )
                            ),
                            block_height - 1,
                        ),
                    ),
                )
                per_block[block] = max(
                    per_block.get(block, 0.0), probability_value
                )
            evidence[row] = float(np.sum(list(per_block.values())))
    return evidence


def retrieve_candidate_groups(
    query_descriptors: np.ndarray,
    query_region_xy: np.ndarray,
    query_region_extent: np.ndarray,
    bank: SurfaceRetrievalMapletBank,
    *,
    preliminary_candidates: int = 64,
    maximum_maplets: int = 64,
    descriptor_temperature: float = 0.08,
    mixture_temperature: float = 0.06,
    set_rerank_strength: float = 0.0,
    set_rerank_neighbors: int = 8,
    maximum_components_per_maplet: int = 16,
    component_nms_distance_m: float = 0.02,
    spatial_query_descriptors: np.ndarray | None = None,
    spatial_bank: SurfaceRetrievalMapletBank | None = None,
    probability_calibration: V6ProbabilityCalibration | None = None,
    scene_evidence_aggregation: str = "topq_nms",
    scene_evidence_top_q: int = 4,
    scene_evidence_overlap_iou: float = 0.30,
    compute_spatial_modes: bool = True,
) -> MapletRetrievalResult:
    if probability_calibration is None:
        raise ValueError(
            "V6 retrieval requires a frozen probability calibration artifact"
        )
    query = _normalize(query_descriptors)
    spatial_query = (
        query
        if spatial_query_descriptors is None
        else _normalize(spatial_query_descriptors)
    )
    geometry_bank = bank if spatial_bank is None else spatial_bank
    if (spatial_bank is None) != (spatial_query_descriptors is None):
        raise ValueError(
            "spatial_query_descriptors and spatial_bank must be supplied together"
        )
    if spatial_query.shape[0] != query.shape[0]:
        raise ValueError("identity and spatial query region counts differ")
    if spatial_query.shape[1] != geometry_bank.descriptors.shape[1]:
        raise ValueError("spatial query/map descriptor dimensions differ")
    geometry_row_by_id = {
        int(value): int(row)
        for row, value in enumerate(geometry_bank.maplet_ids.tolist())
    }
    xy = np.asarray(query_region_xy, dtype=np.float32)
    extent = np.asarray(query_region_extent, dtype=np.float32)
    if xy.shape != (query.shape[0], 2) or extent.shape != (query.shape[0], 2):
        raise ValueError("query region geometry must have shape (R,2)")
    if query.shape[1] != bank.descriptors.shape[1]:
        raise ValueError("query/maplet descriptor dimensions differ")
    # One large GEMM is substantially faster and more deterministic than
    # hundreds of tiny per-maplet matrix multiplications.
    identity_component_scores = query @ bank.descriptors.T
    logits = np.empty((query.shape[0], len(bank)), dtype=np.float64)
    tau = max(float(mixture_temperature), 1e-4)
    for row in range(len(bank)):
        begin, end = (
            int(bank.descriptor_offsets[row]),
            int(bank.descriptor_offsets[row + 1]),
        )
        scores = identity_component_scores[:, begin:end]
        log_weights = np.log(
            np.clip(bank.descriptor_weights[begin:end], 1e-12, 1.0)
        )
        component_logits = scores / tau + log_weights[None]
        normalizer = logsumexp(component_logits, axis=1)
        logits[:, row] = tau * normalizer
    logits += 0.15 * np.log(np.clip(bank.quality_scores[None], 1e-4, 1.0))
    logits -= 0.10 * np.clip(bank.descriptor_uncertainties[None], 0.0, 10.0)
    scaled = logits / max(float(descriptor_temperature), 1e-4)
    base_null = probability_calibration.identity.probability(scaled)
    conditional_maplet_probability = np.exp(
        scaled - logsumexp(scaled, axis=1, keepdims=True)
    )
    maplet_probability = (
        (1.0 - base_null[:, None]) * conditional_maplet_probability
    )
    keep = min(max(int(preliminary_candidates), 1), len(bank))
    columns = np.argpartition(-maplet_probability, kth=keep - 1, axis=1)[
        :, :keep
    ]
    groups = []
    for region in range(query.shape[0]):
        order = np.argsort(
            -maplet_probability[region, columns[region]], kind="mergesort"
        )
        selected = columns[region, order]
        probability = maplet_probability[region, selected]
        omitted = float(1.0 - base_null[region] - np.sum(probability))
        omitted = max(omitted, 0.0)
        null = float(np.clip(base_null[region] + omitted, 0.0, 1.0))
        groups.append(
            QueryMapletGroup(
                query_region_xy=xy[region],
                query_region_extent=extent[region],
                maplet_ids=bank.maplet_ids[selected],
                probabilities=probability.astype(np.float32),
                null_probability=null,
                omitted_probability=omitted,
                candidate_centers=bank.centers[selected],
            )
        )
    if float(set_rerank_strength) > 0.0 and len(groups) >= 2:
        groups = list(
            rerank_candidate_groups(
                tuple(groups),
                bank,
                strength=float(set_rerank_strength),
                neighbors=int(set_rerank_neighbors),
            )
        )
    evidence = aggregate_scene_maplet_evidence(
        groups,
        bank.maplet_ids,
        method=str(scene_evidence_aggregation),
        top_q=int(scene_evidence_top_q),
        overlap_iou=float(scene_evidence_overlap_iou),
    )
    ranked = np.argsort(-evidence, kind="mergesort")
    ranked = ranked[evidence[ranked] > 0.0][: int(maximum_maplets)]
    ranked_maplet_ids = bank.maplet_ids[ranked]
    groups = [
        _restrict_group_to_maplets(group, ranked_maplet_ids)
        for group in groups
    ]
    if not bool(compute_spatial_modes):
        return MapletRetrievalResult(
            groups=tuple(groups),
            ranked_maplet_ids=ranked_maplet_ids,
            evidence=evidence[ranked].astype(np.float32),
            scene_evidence_aggregation=str(scene_evidence_aggregation),
        )
    # Only now evaluate within-maplet surface locations.  Computing the
    # spatial bank over the whole scene before identity Top-K both violates
    # the intended coarse-to-fine graph and is needlessly quadratic in map
    # size.
    selected_component_rows = []
    selected_component_slices: dict[int, tuple[int, int, int, int]] = {}
    cursor = 0
    for maplet_id in ranked_maplet_ids.tolist():
        geometry_row = geometry_row_by_id.get(int(maplet_id))
        if geometry_row is None:
            continue
        begin, end = (
            int(geometry_bank.descriptor_offsets[geometry_row]),
            int(geometry_bank.descriptor_offsets[geometry_row + 1]),
        )
        rows = np.arange(begin, end, dtype=np.int64)
        selected_component_rows.append(rows)
        selected_component_slices[int(maplet_id)] = (
            cursor,
            cursor + rows.size,
            begin,
            end,
        )
        cursor += rows.size
    spatial_component_scores = None
    if selected_component_rows:
        spatial_rows = np.concatenate(selected_component_rows)
        spatial_component_scores = (
            spatial_query @ geometry_bank.descriptors[spatial_rows].T
        )
    enriched_groups = []
    for region, group in enumerate(groups):
        component_offsets = [0]
        component_centers = []
        component_covariances = []
        component_probabilities = []
        candidate_centers = []
        candidate_covariances = []
        spatial_available = []
        spatial_null_probabilities = []
        for maplet_id in group.maplet_ids.tolist():
            component_slice = selected_component_slices.get(int(maplet_id))
            if component_slice is None or spatial_component_scores is None:
                spatial_available.append(False)
                spatial_null_probabilities.append(1.0)
                candidate_centers.append(
                    np.full((3,), np.nan, dtype=np.float32)
                )
                candidate_covariances.append(
                    np.full((3, 3), np.nan, dtype=np.float32)
                )
                component_offsets.append(component_offsets[-1])
                continue
            spatial_available.append(True)
            local_begin, local_end, begin, end = component_slice
            local_score = spatial_component_scores[
                region, local_begin:local_end
            ]
            local_logit = (
                local_score / tau
                + np.log(
                    np.clip(
                        geometry_bank.descriptor_weights[begin:end],
                        1e-12,
                        1.0,
                    )
                )
            )
            base_spatial_null = float(
                probability_calibration.spatial.probability(local_logit)[0]
            )
            local_probability = (
                (1.0 - base_spatial_null)
                * np.exp(local_logit - logsumexp(local_logit))
            ).astype(np.float32)
            all_centers = np.asarray(
                geometry_bank.descriptor_centers[begin:end],
                dtype=np.float64,
            )
            all_covariances = np.asarray(
                geometry_bank.descriptor_covariances[begin:end],
                dtype=np.float64,
            )
            # A candidate representative is a real MAP surface cell, never
            # the mean of distinct locations.
            map_component = int(np.argmax(local_probability))
            candidate_centers.append(
                all_centers[map_component].astype(np.float32)
            )
            candidate_covariances.append(
                all_covariances[map_component].astype(np.float32)
            )
            (
                local_centers,
                local_covariances,
                retained_probability,
                discarded_probability,
            ) = _compress_surface_location_modes(
                local_probability,
                all_centers,
                all_covariances,
                maximum_modes=int(maximum_components_per_maplet),
                nms_distance_m=float(component_nms_distance_m),
            )
            spatial_null_probabilities.append(
                float(
                    np.clip(
                        base_spatial_null + discarded_probability,
                        0.0,
                        1.0,
                    )
                )
            )
            component_centers.append(local_centers)
            component_covariances.append(local_covariances)
            component_probabilities.append(retained_probability)
            component_offsets.append(
                component_offsets[-1] + int(retained_probability.size)
            )
        enriched_groups.append(
            QueryMapletGroup(
                query_region_xy=group.query_region_xy,
                query_region_extent=group.query_region_extent,
                maplet_ids=group.maplet_ids,
                probabilities=group.probabilities,
                null_probability=group.null_probability,
                omitted_probability=group.omitted_probability,
                candidate_centers=np.asarray(
                    candidate_centers, dtype=np.float32
                ).reshape(-1, 3),
                candidate_covariances=np.asarray(
                    candidate_covariances, dtype=np.float32
                ).reshape(-1, 3, 3),
                component_offsets=np.asarray(
                    component_offsets, dtype=np.int64
                ),
                component_centers=(
                    np.concatenate(component_centers, axis=0)
                    if component_centers
                    else np.empty((0, 3), dtype=np.float32)
                ),
                component_covariances=(
                    np.concatenate(component_covariances, axis=0)
                    if component_covariances
                    else np.empty((0, 3, 3), dtype=np.float32)
                ),
                component_probabilities=(
                    np.concatenate(component_probabilities, axis=0)
                    if component_probabilities
                    else np.empty((0,), dtype=np.float32)
                ),
                spatial_available=np.asarray(
                    spatial_available, dtype=bool
                ),
                spatial_null_probabilities=np.asarray(
                    spatial_null_probabilities, dtype=np.float32
                ),
            )
        )
    groups = enriched_groups
    return MapletRetrievalResult(
        groups=tuple(groups),
        ranked_maplet_ids=ranked_maplet_ids,
        evidence=evidence[ranked].astype(np.float32),
        scene_evidence_aggregation=str(scene_evidence_aggregation),
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    row = int(np.searchsorted(cumulative, cumulative[-1] * 0.5))
    return float(values[order[min(row, order.size - 1)]])


def rerank_candidate_groups(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    *,
    strength: float = 0.35,
    neighbors: int = 8,
    log_scale_sigma: float = 1.0,
) -> tuple[QueryMapletGroup, ...]:
    """Legacy scale-ratio ablation; disabled by default.

    The unknown image-to-world scale is marginalized by a weighted median of
    pairwise 3D/2D distance ratios. Candidate support then comes from other
    query regions whose relative layout is compatible at that broad scale.
    Retained maplet mass is conserved exactly, so omitted mass remains null.
    Perspective scenes do not in general admit one global 3D/2D scale, so
    production set consistency belongs in pose-hypothesis likelihood instead.
    """

    if len(groups) < 2:
        return groups
    row_by_id = {
        int(value): int(row) for row, value in enumerate(bank.maplet_ids.tolist())
    }
    k = max(int(neighbors), 1)
    ratios = []
    ratio_weights = []
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            image_distance = float(
                np.linalg.norm(
                    groups[first].query_region_xy
                    - groups[second].query_region_xy
                )
            )
            if image_distance < 1.0:
                continue
            ids_a = groups[first].maplet_ids[:k]
            ids_b = groups[second].maplet_ids[:k]
            rows_a = np.asarray([row_by_id[int(value)] for value in ids_a])
            rows_b = np.asarray([row_by_id[int(value)] for value in ids_b])
            distance = np.linalg.norm(
                bank.centers[rows_a, None] - bank.centers[rows_b][None], axis=2
            )
            weight = (
                groups[first].probabilities[: rows_a.size, None]
                * groups[second].probabilities[None, : rows_b.size]
            )
            valid = distance > 1e-3
            ratios.append(np.log(distance[valid] / image_distance))
            ratio_weights.append(weight[valid])
    if not ratios or not sum(value.size for value in ratios):
        return groups
    ratio_values = np.concatenate(ratios)
    ratio_weight_values = np.concatenate(ratio_weights)
    positive = ratio_weight_values > 0.0
    if not np.any(positive):
        return groups
    log_scale = _weighted_median(
        ratio_values[positive], ratio_weight_values[positive]
    )
    reranked = []
    for group_index, group in enumerate(groups):
        rows = np.asarray(
            [row_by_id[int(value)] for value in group.maplet_ids], dtype=np.int64
        )
        support = np.zeros(rows.size, dtype=np.float64)
        normalizer = 0.0
        for other_index, other in enumerate(groups):
            if other_index == group_index:
                continue
            image_distance = float(
                np.linalg.norm(group.query_region_xy - other.query_region_xy)
            )
            if image_distance < 1.0:
                continue
            other_ids = other.maplet_ids[:k]
            other_rows = np.asarray(
                [row_by_id[int(value)] for value in other_ids], dtype=np.int64
            )
            distance = np.linalg.norm(
                bank.centers[rows, None] - bank.centers[other_rows][None], axis=2
            )
            residual = (
                np.log(np.maximum(distance, 1e-4) / image_distance) - log_scale
            )
            compatibility = np.exp(
                -0.5 * (residual / max(float(log_scale_sigma), 1e-3)) ** 2
            )
            other_probability = other.probabilities[: other_rows.size]
            support += compatibility @ other_probability
            normalizer += float(np.sum(other_probability))
        support /= max(normalizer, 1e-8)
        original = np.asarray(group.probabilities, dtype=np.float64)
        retained_mass = float(np.sum(original))
        logits = np.log(np.maximum(original, 1e-12)) + float(strength) * support
        probability = np.exp(logits - np.max(logits))
        probability *= retained_mass / max(float(np.sum(probability)), 1e-12)
        reranked.append(
            QueryMapletGroup(
                query_region_xy=group.query_region_xy,
                query_region_extent=group.query_region_extent,
                maplet_ids=group.maplet_ids,
                probabilities=probability.astype(np.float32),
                null_probability=group.null_probability,
                omitted_probability=group.omitted_probability,
                candidate_centers=group.candidate_centers,
                candidate_covariances=group.candidate_covariances,
                component_offsets=group.component_offsets,
                component_centers=group.component_centers,
                component_covariances=group.component_covariances,
                component_probabilities=group.component_probabilities,
                spatial_available=group.spatial_available,
                spatial_null_probabilities=group.spatial_null_probabilities,
            )
        )
    return tuple(reranked)
