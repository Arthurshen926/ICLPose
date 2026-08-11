"""Query-graph message passing for multi-modal physical-parent posteriors.

The operation in this module is deliberately pose-free.  It does not turn a
maplet centre into a point correspondence and it does not compare image axes
with world axes.  Instead it asks whether neighbouring query regions admit a
*jointly plausible* set of physical parents under map topology.  Retained
in-map probability is redistributed, while out-of-map and truncated-tail
mass remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap
from .query_edge_factor import LOCAL_EDGE, QueryEdgeGraph, build_query_edge_graph
from .typed_graph import TypedParentGraph


@dataclass(frozen=True)
class StructuredPosteriorDiagnostics:
    entropy_before: float
    entropy_after: float
    top1_changed_fraction: float
    edge_count: int


def _normalized_entropy(probability: np.ndarray) -> float:
    value = np.asarray(probability, dtype=np.float64)
    mass = np.sum(value, axis=1, keepdims=True)
    normalized = np.divide(value, mass, out=np.zeros_like(value), where=mass > 0.0)
    valid = normalized > 0.0
    entropy = -np.sum(np.where(valid, normalized * np.log(np.maximum(normalized, 1e-12)), 0.0), axis=1)
    denominator = np.log(np.maximum(np.sum(valid, axis=1), 2))
    return float(np.mean(entropy / denominator)) if value.shape[0] else 0.0


def refine_parent_posteriors_with_query_graph(
    candidate_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    query_xy_normalized: np.ndarray,
    query_context_descriptors: np.ndarray,
    physical: GoalMapletPhysicalMap,
    graph: TypedParentGraph,
    *,
    iterations: int = 2,
    damping: float = 0.5,
    message_strength: float = 0.75,
    query_graph: QueryEdgeGraph | None = None,
) -> tuple[np.ndarray, StructuredPosteriorDiagnostics]:
    """Redistribute retained mass using fixed query/map topology factors.

    Local query edges prefer equal or geometrically continuous parents.  Long
    edges prefer mapping co-visibility.  Direction is intentionally absent:
    signed image/world displacement is only meaningful after an SE(3) mode
    exists and is handled by :mod:`query_edge_factor` at that later stage.
    """

    ids = np.asarray(candidate_ids, dtype=np.int64)
    probability = np.asarray(candidate_probabilities, dtype=np.float64)
    xy = np.asarray(query_xy_normalized, dtype=np.float64).reshape(-1, 2)
    descriptor = np.asarray(query_context_descriptors, dtype=np.float64)
    if (
        ids.shape != probability.shape
        or ids.ndim != 2
        or xy.shape != (ids.shape[0], 2)
        or descriptor.ndim != 2
        or descriptor.shape[0] != ids.shape[0]
    ):
        raise ValueError("structured parent-posterior inputs differ")
    if graph.physical_map_sha256 != physical.content_sha256:
        raise ValueError("structured posterior graph and physical map differ")
    if int(iterations) < 0:
        raise ValueError("iterations must be non-negative")
    if not 0.0 <= float(damping) <= 1.0 or float(message_strength) < 0.0:
        raise ValueError("invalid structured posterior update parameters")
    row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    rows = np.full(ids.shape, -1, dtype=np.int64)
    for support in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            rows[support, slot] = row_by_id.get(int(ids[support, slot]), -1)
    valid = (rows >= 0) & (probability > 0.0)
    retained_mass = np.sum(probability, axis=1, keepdims=True)
    belief = np.divide(
        probability,
        retained_mass,
        out=np.zeros_like(probability),
        where=retained_mass > 0.0,
    )
    unary = belief.copy()
    edges = query_graph or build_query_edge_graph(xy, descriptor)
    covisibility = graph.covisibility_matrix().astype(np.float64)
    centers = np.asarray(physical.maplet_centers, dtype=np.float64)
    normals = np.asarray(physical.maplet_normals, dtype=np.float64)
    extents = np.asarray(physical.maplet_extents, dtype=np.float64)

    for _ in range(int(iterations)):
        message = np.zeros_like(belief)
        degree = np.zeros((belief.shape[0], 1), dtype=np.float64)
        for source, target, kind in zip(edges.source.tolist(), edges.target.tolist(), edges.kind.tolist()):
            source_rows = rows[source]
            target_rows = rows[target]
            source_valid = valid[source]
            target_valid = valid[target]
            safe_source = np.maximum(source_rows, 0)
            safe_target = np.maximum(target_rows, 0)
            left = safe_source[:, None]
            right = safe_target[None, :]
            pair_valid = source_valid[:, None] & target_valid[None, :]
            co = covisibility[left, right]
            distance = np.linalg.norm(centers[left] - centers[right], axis=2)
            scale = np.maximum(
                np.linalg.norm(extents[left], axis=2) + np.linalg.norm(extents[right], axis=2),
                1.0,
            )
            near = np.exp(-0.5 * np.square(distance / (2.0 * scale)))
            normal = np.clip(np.abs(np.sum(normals[left] * normals[right], axis=2)), 0.0, 1.0)
            same = source_rows[:, None] == target_rows[None, :]
            if int(kind) == int(LOCAL_EDGE):
                compatibility = 0.05 + 0.30 * co + 0.40 * near * normal + 0.25 * same
                weight = 1.0
            else:
                compatibility = 0.05 + 0.80 * co + 0.15 * np.exp(-distance / 15.0)
                weight = 0.75
            compatibility = np.where(pair_valid, compatibility, 0.0)
            source_message = np.log(np.maximum(compatibility @ belief[target], 1e-8))
            target_message = np.log(np.maximum(compatibility.T @ belief[source], 1e-8))
            message[source] += weight * source_message
            message[target] += weight * target_message
            degree[source] += weight
            degree[target] += weight
        averaged = np.divide(message, degree, out=np.zeros_like(message), where=degree > 0.0)
        logits = np.log(np.maximum(unary, 1e-12)) + float(message_strength) * averaged
        logits[~valid] = -np.inf
        row_max = np.max(logits, axis=1, keepdims=True)
        finite_row = np.isfinite(row_max)
        updated = np.zeros_like(belief)
        exponent = np.zeros_like(logits)
        np.exp(logits - np.where(finite_row, row_max, 0.0), out=exponent, where=np.isfinite(logits))
        normalizer = np.sum(exponent, axis=1, keepdims=True)
        np.divide(exponent, normalizer, out=updated, where=normalizer > 0.0)
        belief = (1.0 - float(damping)) * belief + float(damping) * updated
        belief[~valid] = 0.0
        belief /= np.maximum(np.sum(belief, axis=1, keepdims=True), 1e-12)

    output = belief * retained_mass
    before_top1 = np.argmax(probability, axis=1)
    after_top1 = np.argmax(output, axis=1)
    diagnostics = StructuredPosteriorDiagnostics(
        entropy_before=_normalized_entropy(probability),
        entropy_after=_normalized_entropy(output),
        top1_changed_fraction=float(np.mean(before_top1 != after_top1)) if ids.shape[0] else 0.0,
        edge_count=int(edges.source.size),
    )
    return output, diagnostics
