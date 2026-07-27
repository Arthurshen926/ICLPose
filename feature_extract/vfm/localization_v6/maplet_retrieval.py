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


@dataclass(frozen=True)
class MapletRetrievalResult:
    groups: tuple[QueryMapletGroup, ...]
    ranked_maplet_ids: np.ndarray
    evidence: np.ndarray


def _normalize(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def retrieve_candidate_groups(
    query_descriptors: np.ndarray,
    query_region_xy: np.ndarray,
    query_region_extent: np.ndarray,
    bank: SurfaceRetrievalMapletBank,
    *,
    preliminary_candidates: int = 64,
    maximum_maplets: int = 24,
    descriptor_temperature: float = 0.08,
    mixture_temperature: float = 0.06,
    null_logit: float = 0.0,
    set_rerank_strength: float = 0.35,
    set_rerank_neighbors: int = 8,
) -> MapletRetrievalResult:
    query = _normalize(query_descriptors)
    xy = np.asarray(query_region_xy, dtype=np.float32)
    extent = np.asarray(query_region_extent, dtype=np.float32)
    if xy.shape != (query.shape[0], 2) or extent.shape != (query.shape[0], 2):
        raise ValueError("query region geometry must have shape (R,2)")
    if query.shape[1] != bank.descriptors.shape[1]:
        raise ValueError("query/maplet descriptor dimensions differ")
    logits = np.empty((query.shape[0], len(bank)), dtype=np.float64)
    tau = max(float(mixture_temperature), 1e-4)
    for row in range(len(bank)):
        begin, end = (
            int(bank.descriptor_offsets[row]),
            int(bank.descriptor_offsets[row + 1]),
        )
        scores = query @ bank.descriptors[begin:end].T
        log_weights = np.log(
            np.clip(bank.descriptor_weights[begin:end], 1e-12, 1.0)
        )
        logits[:, row] = tau * logsumexp(
            scores / tau + log_weights[None], axis=1
        )
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
    return MapletRetrievalResult(
        groups=tuple(groups),
        ranked_maplet_ids=bank.maplet_ids[ranked],
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
    """Lightweight set consistency without collapsing candidate groups.

    The unknown image-to-world scale is marginalized by a weighted median of
    pairwise 3D/2D distance ratios. Candidate support then comes from other
    query regions whose relative layout is compatible at that broad scale.
    Retained maplet mass is conserved exactly, so omitted mass remains null.
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
            )
        )
    return tuple(reranked)
