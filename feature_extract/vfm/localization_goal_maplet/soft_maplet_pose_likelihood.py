"""Pose-conditioned marginal likelihood over ambiguous physical maplets.

The query does not commit to one maplet before a pose exists.  For a frozen
pose, every query support retains its parent-conditioned child alternatives,
projects their complete surface tiles, and sums their probability mass plus a
typed null.  A bounded set of query edges is then marginalized over the most
probable assignment pairs.  The returned MAP assignment is explanation only;
it does not define the likelihood or generate a point correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap
from .query_edge_factor import QueryEdgeGraph, project_child_surfaces


@dataclass(frozen=True)
class SoftMapletPoseEvidence:
    score: float
    unary_log_likelihood: float
    edge_log_likelihood: float
    chosen_parent_rows: np.ndarray
    chosen_child_rows: np.ndarray
    supporting_region_count: int
    null_mass_mean: float


def _huber(value: np.ndarray, delta: float = 1.0) -> np.ndarray:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    return np.where(
        absolute <= float(delta),
        0.5 * np.square(absolute),
        float(delta) * (absolute - 0.5 * float(delta)),
    )


def _bounded_edges(graph: QueryEdgeGraph, xy: np.ndarray, maximum_edges: int) -> np.ndarray:
    if graph.source.size <= int(maximum_edges):
        return np.arange(graph.source.size, dtype=np.int64)
    displacement = np.linalg.norm(xy[graph.target] - xy[graph.source], axis=1)
    priority = (
        2.0 * graph.distinctive_context.astype(np.float64)
        + graph.high_displacement.astype(np.float64)
        + displacement
    )
    selected = np.argpartition(-priority, kth=int(maximum_edges) - 1)[: int(maximum_edges)]
    return selected[np.argsort(-priority[selected], kind="stable")]


def pose_conditioned_soft_maplet_evidence(
    pose_w2c: np.ndarray,
    query_xy_normalized: np.ndarray,
    query_extent_normalized: np.ndarray,
    query_graph: QueryEdgeGraph,
    parent_probability: np.ndarray,
    parent_out_of_map: np.ndarray,
    parent_unresolved: np.ndarray,
    child_rows_by_parent: np.ndarray,
    child_probability_by_parent: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
    *,
    maximum_parent_candidates: int,
    edge_candidate_count: int = 4,
    maximum_edges: int = 128,
    edge_weight: float = 0.35,
    missing_identity_huber_loss: float = 2.0,
) -> SoftMapletPoseEvidence:
    """Evaluate one pose without hardening parent/child identity first.

    The pose likelihood is conditional on a support being in-map.  Its
    calibrated out-of-map mass therefore weights how much that support can
    affect pose ranking; it is not shrunk into a fake floor that forces a map
    identity.  Truncated parent mass and probability outside the retained
    best child remain an explicit missing-identity branch.
    """

    count = min(int(maximum_parent_candidates), int(parent_probability.shape[1]))
    children = np.asarray(child_rows_by_parent, dtype=np.int64)
    child_mass = np.asarray(child_probability_by_parent, dtype=np.float64)
    if children.ndim == 2:
        children = children[..., None]
        child_mass = child_mass[..., None]
    children = children[:, :count]
    child_mass = child_mass[:, :count]
    parent_mass = np.asarray(parent_probability[:, :count], dtype=np.float64)
    out_of_map = np.asarray(parent_out_of_map, dtype=np.float64).reshape(-1)
    unresolved = np.asarray(parent_unresolved, dtype=np.float64).reshape(-1)
    in_map = np.clip(1.0 - out_of_map, 1.0e-8, 1.0)
    conditional_parent = parent_mass / in_map[:, None]
    prior = conditional_parent[..., None] * child_mass
    xy = np.asarray(query_xy_normalized, dtype=np.float64).reshape(-1, 2)
    extent_query = np.asarray(query_extent_normalized, dtype=np.float64).reshape(-1, 2)
    if (
        children.ndim != 3
        or children.shape != prior.shape
        or child_mass.shape != children.shape
        or children.shape[0] != xy.shape[0]
        or out_of_map.shape != (xy.shape[0],)
        or unresolved.shape != out_of_map.shape
    ):
        raise ValueError("soft maplet likelihood arrays differ")
    flat_xy, flat_extent, flat_visible = project_child_surfaces(
        children.reshape(-1), pose_w2c, physical, camera,
    )
    projected = flat_xy.reshape(children.shape + (2,))
    projected_extent = flat_extent.reshape(children.shape + (2,))
    visible = flat_visible.reshape(children.shape) & (children >= 0) & (prior > 0.0)

    # Parent x child is one latent identity axis for pose scoring.
    alternative_count = int(children.shape[1] * children.shape[2])
    children = children.reshape(children.shape[0], alternative_count)
    prior = prior.reshape(prior.shape[0], alternative_count)
    projected = projected.reshape(projected.shape[0], alternative_count, 2)
    projected_extent = projected_extent.reshape(
        projected_extent.shape[0], alternative_count, 2,
    )
    visible = visible.reshape(visible.shape[0], alternative_count)

    expanded = projected_extent + 0.10 * extent_query[:, None]
    outside = np.maximum(np.abs(xy[:, None] - projected) - expanded, 0.0)
    sigma = np.maximum(0.008, 0.20 * np.linalg.norm(extent_query, axis=1))
    residual = np.linalg.norm(outside, axis=2) / sigma[:, None]
    assignment = prior * np.exp(-0.5 * np.square(np.minimum(residual, 8.0))) * visible
    truncated_parent = np.clip(unresolved - out_of_map, 0.0, 1.0) / in_map
    retained_child_mass = np.sum(child_mass, axis=2)
    missing_child = np.sum(
        conditional_parent * np.clip(1.0 - retained_child_mass, 0.0, 1.0),
        axis=1,
    )
    null_mass = np.clip(truncated_parent + missing_child, 0.0, 1.0)
    missing_likelihood = np.exp(-float(missing_identity_huber_loss)) * null_mass
    evidence_mass = np.sum(assignment, axis=1) + missing_likelihood
    support_weight = np.clip(in_map, 1.0e-4, 1.0)
    unary = float(np.sum(
        support_weight * np.log(np.maximum(evidence_mass, 1.0e-12))
    ) / np.sum(support_weight))
    posterior = assignment / np.maximum(evidence_mass[:, None], 1.0e-12)

    chosen_slot = np.argmax(posterior, axis=1)
    chosen_child = children[np.arange(children.shape[0]), chosen_slot].copy()
    chosen_probability = posterior[np.arange(children.shape[0]), chosen_slot]
    chosen_child[
        chosen_probability
        <= missing_likelihood / np.maximum(evidence_mass, 1.0e-12)
    ] = -1
    chosen_parent = np.full(chosen_child.shape, -1, dtype=np.int64)
    valid_child = chosen_child >= 0
    chosen_parent[valid_child] = physical.child_parent_rows[chosen_child[valid_child]]

    edge_rows = _bounded_edges(query_graph, xy, int(maximum_edges))
    edge_values: list[float] = []
    pair_count = max(1, min(int(edge_candidate_count), alternative_count))
    for edge_index in edge_rows.tolist():
        source = int(query_graph.source[edge_index])
        target = int(query_graph.target[edge_index])
        source_order = np.argsort(-posterior[source], kind="stable")[:pair_count]
        target_order = np.argsort(-posterior[target], kind="stable")[:pair_count]
        source_probability = posterior[source, source_order]
        target_probability = posterior[target, target_order]
        pair_probability = source_probability[:, None] * target_probability[None, :]
        source_projected = projected[source, source_order]
        target_projected = projected[target, target_order]
        predicted = target_projected[None] - source_projected[:, None]
        observed = xy[target] - xy[source]
        edge_sigma = max(
            float(np.linalg.norm(extent_query[source]) + np.linalg.norm(extent_query[target])),
            0.02,
        )
        normalized = np.linalg.norm(predicted - observed, axis=2) / edge_sigma
        compatibility = np.exp(-_huber(normalized))
        explained = float(np.sum(pair_probability * compatibility))
        retained_pair_mass = float(np.sum(pair_probability))
        # Any truncated assignment pair is a typed missing edge, not a perfect
        # match and not a disappearing denominator.
        missing = max(0.0, 1.0 - retained_pair_mass)
        edge_values.append(np.log(max(explained + np.exp(-2.0) * missing, 1.0e-12)))
    edge_score = float(np.mean(edge_values)) if edge_values else -2.0
    score = float(unary + float(edge_weight) * edge_score)
    return SoftMapletPoseEvidence(
        score=score,
        unary_log_likelihood=unary,
        edge_log_likelihood=edge_score,
        chosen_parent_rows=chosen_parent,
        chosen_child_rows=chosen_child,
        supporting_region_count=int(np.sum(
            np.sum(assignment, axis=1) > missing_likelihood
        )),
        null_mass_mean=float(np.mean(
            missing_likelihood / np.maximum(evidence_mass, 1.0e-12)
        )),
    )
