"""Group-aware latent assignment for fixed Goal-Maplet surface modes.

The solver deliberately does not estimate a pose and does not regenerate any
feature likelihood.  It only asks whether the child/local modes proposed once
by the VFM can explain a *fixed* pose configuration without reusing the same
physical evidence arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LatentAssignment:
    """One valid mode or null for every fixed query group."""

    option_rows: np.ndarray
    child_rows: np.ndarray
    primitive_rows: np.ndarray
    factor_rows: np.ndarray
    valid_mass: np.ndarray
    scores: np.ndarray
    assigned: np.ndarray

    def __post_init__(self) -> None:
        count = np.asarray(self.option_rows).reshape(-1).size
        for name in (
            "child_rows", "primitive_rows", "factor_rows", "valid_mass",
            "scores", "assigned",
        ):
            if np.asarray(getattr(self, name)).reshape(-1).size != count:
                raise ValueError("latent assignment arrays differ")


def correlated_support_clusters(
    xy_px: np.ndarray,
    extent_px: np.ndarray,
    *,
    minimum_iou: float = 0.50,
) -> np.ndarray:
    """Cluster heavily overlapping query supports without using map identity.

    Query grouping has already merged adjacent tokens that share a retrieved
    parent.  This second, conservative pass only de-correlates supports whose
    receptive-field boxes overlap by at least 50%, including supports that
    happened to retrieve different repeated identities.
    """

    xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(extent_px, dtype=np.float64).reshape(-1, 2)
    if xy.shape != extent.shape or np.any(extent <= 0.0):
        raise ValueError("invalid query support boxes")
    count = xy.shape[0]
    low, high = xy - extent, xy + extent
    area = np.prod(np.maximum(high - low, 0.0), axis=1)
    overlap = np.eye(count, dtype=np.float64)
    for left in range(count):
        intersection_extent = np.maximum(
            np.minimum(high[left], high[left + 1 :])
            - np.maximum(low[left], low[left + 1 :]),
            0.0,
        )
        intersection = np.prod(intersection_extent, axis=1)
        union_area = area[left] + area[left + 1 :] - intersection
        value = intersection / np.maximum(union_area, 1e-12)
        overlap[left, left + 1 :] = value
        overlap[left + 1 :, left] = value
    # Complete-link grouping avoids the transitive-chain failure of connected
    # components: A overlapping B and B overlapping C does not make distant A
    # and C one observation.  Raster order is fixed upstream, so the result is
    # deterministic and candidate independent.
    members: list[list[int]] = []
    for row in range(count):
        destination = None
        for cluster_index, cluster_members in enumerate(members):
            if np.all(overlap[row, cluster_members] >= float(minimum_iou)):
                destination = cluster_index
                break
        if destination is None:
            members.append([row])
        else:
            members[destination].append(row)
    output = np.zeros((count,), dtype=np.int64)
    for cluster_index, cluster_members in enumerate(members):
        output[np.asarray(cluster_members, dtype=np.int64)] = cluster_index
    return output


def effective_group_weights(cluster_rows: np.ndarray) -> np.ndarray:
    """Give each correlated support cluster unit total evidence mass."""

    cluster = np.asarray(cluster_rows, dtype=np.int64).reshape(-1)
    if np.any(cluster < 0):
        raise ValueError("invalid correlated support cluster")
    count = np.bincount(cluster, minlength=int(np.max(cluster)) + 1) if cluster.size else np.zeros(0)
    return (1.0 / np.maximum(count[cluster], 1)).astype(np.float64)


def independent_assignment(
    *,
    group_count: int,
    option_group_rows: np.ndarray,
    option_child_rows: np.ndarray,
    option_primitive_rows: np.ndarray,
    option_factor_rows: np.ndarray,
    option_scores: np.ndarray,
    option_valid_mass: np.ndarray,
    minimum_valid_mass: float = 0.20,
) -> LatentAssignment:
    """Independent per-group MAP over a fixed option set plus explicit null."""

    group = np.asarray(option_group_rows, dtype=np.int64).reshape(-1)
    child = np.asarray(option_child_rows, dtype=np.int64).reshape(-1)
    primitive = np.asarray(option_primitive_rows, dtype=np.int64).reshape(-1)
    factor = np.asarray(option_factor_rows, dtype=np.int64).reshape(-1)
    score = np.asarray(option_scores, dtype=np.float64).reshape(-1)
    mass = np.asarray(option_valid_mass, dtype=np.float64).reshape(-1)
    size = group.size
    if not all(value.size == size for value in (child, primitive, factor, score, mass)):
        raise ValueError("latent option arrays differ")
    if np.any((group < 0) | (group >= int(group_count))) or not np.all(np.isfinite(score)):
        raise ValueError("invalid latent options")
    selected = np.full((int(group_count),), -1, dtype=np.int64)
    if size == 0:
        return LatentAssignment(
            option_rows=selected,
            child_rows=np.full((int(group_count),), -1, dtype=np.int64),
            primitive_rows=np.full((int(group_count),), -1, dtype=np.int64),
            factor_rows=np.full((int(group_count),), -1, dtype=np.int64),
            valid_mass=np.zeros((int(group_count),), dtype=np.float64),
            scores=np.zeros((int(group_count),), dtype=np.float64),
            assigned=np.zeros((int(group_count),), dtype=bool),
        )
    for row in range(int(group_count)):
        options = np.flatnonzero((group == row) & (mass >= float(minimum_valid_mass)))
        if options.size:
            selected[row] = int(options[np.argmax(score[options])])
    assigned = selected >= 0
    safe = np.maximum(selected, 0)
    return LatentAssignment(
        option_rows=selected,
        child_rows=np.where(assigned, child[safe], -1),
        primitive_rows=np.where(assigned, primitive[safe], -1),
        factor_rows=np.where(assigned, factor[safe], -1),
        valid_mass=np.where(assigned, mass[safe], 0.0),
        scores=np.where(assigned, score[safe], 0.0),
        assigned=assigned,
    )


def capacitated_assignment(
    *,
    group_count: int,
    option_group_rows: np.ndarray,
    option_child_rows: np.ndarray,
    option_primitive_rows: np.ndarray,
    option_factor_rows: np.ndarray,
    option_scores: np.ndarray,
    option_valid_mass: np.ndarray,
    group_cluster_rows: np.ndarray,
    child_capacities: Mapping[int, int],
    minimum_valid_mass: float = 0.20,
) -> LatentAssignment:
    """Deterministic capacity-feasible assignment with an explicit null.

    Groups are visited by confidence margin, not raster order.  A physical
    primitive can support only one independent support cluster.  Child
    capacity counts independent clusters as well, so overlapping/correlated
    observations do not consume capacity repeatedly.  Every rejected group
    remains in the fixed denominator as null.

    This is a low-variance greedy b-matching baseline.  It is intentionally
    kept separate from pose refinement and from any learned ranker.
    """

    group = np.asarray(option_group_rows, dtype=np.int64).reshape(-1)
    child = np.asarray(option_child_rows, dtype=np.int64).reshape(-1)
    primitive = np.asarray(option_primitive_rows, dtype=np.int64).reshape(-1)
    factor = np.asarray(option_factor_rows, dtype=np.int64).reshape(-1)
    score = np.asarray(option_scores, dtype=np.float64).reshape(-1)
    mass = np.asarray(option_valid_mass, dtype=np.float64).reshape(-1)
    cluster = np.asarray(group_cluster_rows, dtype=np.int64).reshape(-1)
    size = group.size
    if not all(value.size == size for value in (child, primitive, factor, score, mass)):
        raise ValueError("latent option arrays differ")
    if cluster.shape != (int(group_count),) or np.any(cluster < 0):
        raise ValueError("latent group clusters differ")
    if np.any((group < 0) | (group >= int(group_count))) or not np.all(np.isfinite(score)):
        raise ValueError("invalid latent options")

    if size == 0:
        return LatentAssignment(
            option_rows=np.full((int(group_count),), -1, dtype=np.int64),
            child_rows=np.full((int(group_count),), -1, dtype=np.int64),
            primitive_rows=np.full((int(group_count),), -1, dtype=np.int64),
            factor_rows=np.full((int(group_count),), -1, dtype=np.int64),
            valid_mass=np.zeros((int(group_count),), dtype=np.float64),
            scores=np.zeros((int(group_count),), dtype=np.float64),
            assigned=np.zeros((int(group_count),), dtype=bool),
        )

    options_by_group: list[np.ndarray] = []
    best_score = np.full((int(group_count),), -np.inf, dtype=np.float64)
    margin = np.full((int(group_count),), -np.inf, dtype=np.float64)
    for row in range(int(group_count)):
        options = np.flatnonzero((group == row) & (mass >= float(minimum_valid_mass)))
        options = options[np.argsort(-score[options], kind="stable")]
        options_by_group.append(options)
        if options.size:
            best_score[row] = score[options[0]]
            second = score[options[1]] if options.size > 1 else best_score[row] - 20.0
            margin[row] = best_score[row] - second
    # Strong, unambiguous groups reserve physical capacity first.  Ties are
    # broken by score and then group index for exact reproducibility.
    order = np.lexsort((np.arange(int(group_count)), -best_score, -margin))
    selected = np.full((int(group_count),), -1, dtype=np.int64)
    primitive_owner: dict[int, int] = {}
    child_owners: dict[int, set[int]] = {}
    for row in order.tolist():
        current_cluster = int(cluster[row])
        for option in options_by_group[row].tolist():
            current_primitive = int(primitive[option])
            current_child = int(child[option])
            primitive_cluster = primitive_owner.get(current_primitive)
            if primitive_cluster is not None and primitive_cluster != current_cluster:
                continue
            owners = child_owners.setdefault(current_child, set())
            capacity = max(int(child_capacities.get(current_child, 1)), 1)
            if current_cluster not in owners and len(owners) >= capacity:
                continue
            selected[row] = int(option)
            primitive_owner[current_primitive] = current_cluster
            owners.add(current_cluster)
            break
    assigned = selected >= 0
    safe = np.maximum(selected, 0)
    return LatentAssignment(
        option_rows=selected,
        child_rows=np.where(assigned, child[safe], -1),
        primitive_rows=np.where(assigned, primitive[safe], -1),
        factor_rows=np.where(assigned, factor[safe], -1),
        valid_mass=np.where(assigned, mass[safe], 0.0),
        scores=np.where(assigned, score[safe], 0.0),
        assigned=assigned,
    )


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape:
        raise ValueError("weighted statistic arrays differ")
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def weighted_quantile(value: np.ndarray, weight: np.ndarray, quantile: float) -> float:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or values.size == 0:
        raise ValueError("weighted quantile arrays differ")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    threshold = float(quantile) * max(float(cumulative[-1]), 1e-12)
    return float(values[min(int(np.searchsorted(cumulative, threshold, side="left")), values.size - 1)])
