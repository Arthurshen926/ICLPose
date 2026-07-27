"""Candidate-group RADIO-final maplet retrieval with exact omitted mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
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


@dataclass(frozen=True)
class MapletRetrievalResult:
    groups: tuple[QueryMapletGroup, ...]
    ranked_maplet_ids: np.ndarray
    evidence: np.ndarray


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep distinct, real 3D modes without creating a virtual mean point."""

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
    limit = max(int(maximum_modes), 1)
    separation = max(float(nms_distance_m), 0.0)
    selected: list[int] = []
    for mode in np.argsort(-aggregate_probability, kind="mergesort").tolist():
        if selected and separation > 0.0:
            distance = np.linalg.norm(
                aggregate_center[selected] - aggregate_center[mode], axis=1
            )
            if np.any(distance < separation):
                continue
        selected.append(int(mode))
        if len(selected) >= limit:
            break
    selected_rows = np.asarray(selected, dtype=np.int64)
    selected_probability = aggregate_probability[selected_rows]
    selected_probability /= max(float(np.sum(selected_probability)), 1e-12)
    return (
        aggregate_center[selected_rows].astype(np.float32),
        aggregate_covariance[selected_rows].astype(np.float32),
        selected_probability.astype(np.float32),
    )


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
    null_logit: float = 0.0,
    set_rerank_strength: float = 0.0,
    set_rerank_neighbors: int = 8,
    maximum_components_per_maplet: int = 16,
    component_nms_distance_m: float = 0.02,
    spatial_query_descriptors: np.ndarray | None = None,
    spatial_bank: SurfaceRetrievalMapletBank | None = None,
) -> MapletRetrievalResult:
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
    missing_geometry = [
        int(value)
        for value in bank.maplet_ids.tolist()
        if int(value) not in geometry_row_by_id
    ]
    if missing_geometry:
        raise ValueError(
            "identity bank contains maplets absent from spatial bank"
        )
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
    joint = np.concatenate(
        [
            scaled,
            np.full((query.shape[0], 1), float(null_logit), dtype=np.float64),
        ],
        axis=1,
    )
    posterior = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    maplet_probability = posterior[:, :-1]
    base_null = posterior[:, -1]
    keep = min(max(int(preliminary_candidates), 1), len(bank))
    columns = np.argpartition(-maplet_probability, kth=keep - 1, axis=1)[
        :, :keep
    ]
    groups = []
    evidence = np.zeros((len(bank),), dtype=np.float64)
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
        evidence[selected] += probability
    if float(set_rerank_strength) > 0.0 and len(groups) >= 2:
        groups = list(
            rerank_candidate_groups(
                tuple(groups),
                bank,
                strength=float(set_rerank_strength),
                neighbors=int(set_rerank_neighbors),
            )
        )
        evidence.fill(0.0)
        row_by_id = {
            int(value): int(row)
            for row, value in enumerate(bank.maplet_ids.tolist())
        }
        for group in groups:
            rows = np.asarray(
                [row_by_id[int(value)] for value in group.maplet_ids],
                dtype=np.int64,
            )
            evidence[rows] += group.probabilities
    ranked = np.argsort(-evidence, kind="mergesort")
    ranked = ranked[evidence[ranked] > 0.0][: int(maximum_maplets)]
    ranked_maplet_ids = bank.maplet_ids[ranked]
    groups = [
        _restrict_group_to_maplets(group, ranked_maplet_ids)
        for group in groups
    ]
    # Only now evaluate within-maplet surface locations.  Computing the
    # spatial bank over the whole scene before identity Top-K both violates
    # the intended coarse-to-fine graph and is needlessly quadratic in map
    # size.
    selected_component_rows = []
    selected_component_slices: dict[int, tuple[int, int, int, int]] = {}
    cursor = 0
    for maplet_id in ranked_maplet_ids.tolist():
        geometry_row = geometry_row_by_id[int(maplet_id)]
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
    if selected_component_rows:
        spatial_rows = np.concatenate(selected_component_rows)
        spatial_component_scores = (
            spatial_query @ geometry_bank.descriptors[spatial_rows].T
        )
        enriched_groups = []
        for region, group in enumerate(groups):
            if group.maplet_ids.size == 0:
                enriched_groups.append(group)
                continue
            component_offsets = [0]
            component_centers = []
            component_covariances = []
            component_probabilities = []
            candidate_centers = []
            candidate_covariances = []
            for maplet_id in group.maplet_ids.tolist():
                local_begin, local_end, begin, end = (
                    selected_component_slices[int(maplet_id)]
                )
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
                local_probability = np.exp(
                    local_logit - logsumexp(local_logit)
                ).astype(np.float32)
                all_centers = np.asarray(
                    geometry_bank.descriptor_centers[begin:end],
                    dtype=np.float64,
                )
                all_covariances = np.asarray(
                    geometry_bank.descriptor_covariances[begin:end],
                    dtype=np.float64,
                )
                moment_center = (
                    np.asarray(local_probability, dtype=np.float64)
                    @ all_centers
                )
                moment_residual = all_centers - moment_center[None]
                moment_covariance = np.sum(
                    np.asarray(local_probability, dtype=np.float64)[
                        :, None, None
                    ]
                    * (
                        all_covariances
                        + moment_residual[:, :, None]
                        * moment_residual[:, None, :]
                    ),
                    axis=0,
                )
                candidate_centers.append(
                    moment_center.astype(np.float32)
                )
                candidate_covariances.append(
                    moment_covariance.astype(np.float32)
                )
                (
                    local_centers,
                    local_covariances,
                    local_probability,
                ) = _compress_surface_location_modes(
                    local_probability,
                    all_centers,
                    all_covariances,
                    maximum_modes=int(maximum_components_per_maplet),
                    nms_distance_m=float(component_nms_distance_m),
                )
                component_centers.append(local_centers)
                component_covariances.append(local_covariances)
                component_probabilities.append(local_probability)
                component_offsets.append(
                    component_offsets[-1] + int(local_probability.size)
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
                    ),
                    candidate_covariances=np.asarray(
                        candidate_covariances, dtype=np.float32
                    ),
                    component_offsets=np.asarray(
                        component_offsets, dtype=np.int64
                    ),
                    component_centers=np.concatenate(
                        component_centers, axis=0
                    ),
                    component_covariances=np.concatenate(
                        component_covariances, axis=0
                    ),
                    component_probabilities=np.concatenate(
                        component_probabilities, axis=0
                    ),
                )
            )
        groups = enriched_groups
    return MapletRetrievalResult(
        groups=tuple(groups),
        ranked_maplet_ids=ranked_maplet_ids,
        evidence=evidence[ranked].astype(np.float32),
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
            )
        )
    return tuple(reranked)
