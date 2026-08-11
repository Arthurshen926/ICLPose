"""Hierarchical child-tile posterior from one canonical surface field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap


@dataclass(frozen=True)
class ChildTilePosterior:
    candidate_child_rows: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    conditional_parent_ids: np.ndarray | None = None
    conditional_parent_log_evidence: np.ndarray | None = None
    best_child_rows_by_parent: np.ndarray | None = None
    best_child_probabilities_by_parent: np.ndarray | None = None
    child_rows_by_parent: np.ndarray | None = None
    child_probabilities_by_parent: np.ndarray | None = None

    def __post_init__(self) -> None:
        rows = np.asarray(self.candidate_child_rows, dtype=np.int64)
        probability = np.asarray(self.candidate_probabilities, dtype=np.float64)
        null = np.asarray(self.null_probabilities, dtype=np.float64).reshape(-1)
        if (
            rows.ndim != 2
            or probability.shape != rows.shape
            or null.shape != (rows.shape[0],)
            or np.any(probability < 0.0)
            or np.any((null < 0.0) | (null > 1.0))
        ):
            raise ValueError("invalid child-tile posterior")
        optional = (
            self.conditional_parent_ids,
            self.conditional_parent_log_evidence,
            self.best_child_rows_by_parent,
            self.best_child_probabilities_by_parent,
        )
        if any(value is not None for value in optional):
            if not all(value is not None for value in optional):
                raise ValueError("incomplete parent-conditioned child evidence")
            parent_ids = np.asarray(self.conditional_parent_ids, dtype=np.int64)
            log_evidence = np.asarray(self.conditional_parent_log_evidence, dtype=np.float64)
            best_rows = np.asarray(self.best_child_rows_by_parent, dtype=np.int64)
            best_probability = np.asarray(self.best_child_probabilities_by_parent, dtype=np.float64)
            if (
                parent_ids.ndim != 2
                or parent_ids.shape[0] != rows.shape[0]
                or log_evidence.shape != parent_ids.shape
                or best_rows.shape != parent_ids.shape
                or best_probability.shape != parent_ids.shape
                or np.any((best_probability < 0.0) | (best_probability > 1.0))
            ):
                raise ValueError("invalid parent-conditioned child evidence")
            object.__setattr__(self, "conditional_parent_ids", parent_ids)
            object.__setattr__(self, "conditional_parent_log_evidence", log_evidence)
            object.__setattr__(self, "best_child_rows_by_parent", best_rows)
            object.__setattr__(self, "best_child_probabilities_by_parent", best_probability)
        alternatives = (self.child_rows_by_parent, self.child_probabilities_by_parent)
        if any(value is not None for value in alternatives):
            if not all(value is not None for value in alternatives):
                raise ValueError("incomplete parent-conditioned child alternatives")
            if self.conditional_parent_ids is None:
                raise ValueError("child alternatives require parent-conditioned evidence")
            alternative_rows = np.asarray(self.child_rows_by_parent, dtype=np.int64)
            alternative_probability = np.asarray(
                self.child_probabilities_by_parent, dtype=np.float64,
            )
            expected = np.asarray(self.conditional_parent_ids).shape
            if (
                alternative_rows.ndim != 3
                or alternative_rows.shape[:2] != expected
                or alternative_probability.shape != alternative_rows.shape
                or np.any((alternative_probability < 0.0) | (alternative_probability > 1.0))
            ):
                raise ValueError("invalid parent-conditioned child alternatives")
            object.__setattr__(self, "child_rows_by_parent", alternative_rows)
            object.__setattr__(
                self, "child_probabilities_by_parent", alternative_probability,
            )
        object.__setattr__(self, "candidate_child_rows", rows)
        object.__setattr__(self, "candidate_probabilities", probability)
        object.__setattr__(self, "null_probabilities", null)


def retrieve_children_given_parents(
    local_descriptors: np.ndarray,
    parent_candidate_ids: np.ndarray,
    parent_probabilities: np.ndarray,
    parent_null_probabilities: np.ndarray,
    child_descriptors: np.ndarray,
    child_coverage: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_child_candidates: int = 64,
    temperature: float = 0.07,
) -> ChildTilePosterior:
    """Factor P(child|query) as P(parent|context)P(child|parent,local)."""

    local = np.asarray(local_descriptors, dtype=np.float32)
    parent_ids = np.asarray(parent_candidate_ids, dtype=np.int64)
    parent_probability = np.asarray(parent_probabilities, dtype=np.float64)
    parent_null = np.asarray(parent_null_probabilities, dtype=np.float64).reshape(-1)
    child_feature = np.asarray(child_descriptors, dtype=np.float32)
    coverage = np.asarray(child_coverage, dtype=np.float64).reshape(-1)
    if (
        local.ndim != 2
        or parent_ids.ndim != 2
        or parent_ids.shape[0] != local.shape[0]
        or parent_probability.shape != parent_ids.shape
        or parent_null.shape != (local.shape[0],)
        or child_feature.shape != (physical.child_parent_rows.size, local.shape[1])
        or coverage.shape != (physical.child_parent_rows.size,)
    ):
        raise ValueError("child retrieval arrays differ")
    local = local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-8)
    child_feature = child_feature / np.maximum(np.linalg.norm(child_feature, axis=1, keepdims=True), 1e-8)
    parent_row_by_id = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    keep = max(1, int(maximum_child_candidates))
    output_rows = np.full((local.shape[0], keep), -1, dtype=np.int64)
    output_probability = np.zeros((local.shape[0], keep), dtype=np.float64)
    output_null = np.ones((local.shape[0],), dtype=np.float64)
    conditional_log_evidence = np.full(parent_ids.shape, -np.inf, dtype=np.float64)
    best_rows_by_parent = np.full(parent_ids.shape, -1, dtype=np.int64)
    best_probability_by_parent = np.zeros(parent_ids.shape, dtype=np.float64)
    alternatives_per_parent = 4
    rows_by_parent = np.full(
        parent_ids.shape + (alternatives_per_parent,), -1, dtype=np.int64,
    )
    probability_by_parent = np.zeros(
        parent_ids.shape + (alternatives_per_parent,), dtype=np.float64,
    )
    for support in range(local.shape[0]):
        accumulated: dict[int, float] = {}
        for parent_slot, (parent_id, parent_value) in enumerate(
            zip(parent_ids[support].tolist(), parent_probability[support].tolist())
        ):
            parent_row = parent_row_by_id.get(int(parent_id))
            if parent_row is None or float(parent_value) <= 0.0:
                continue
            start, end = int(physical.maplet_child_offsets[parent_row]), int(physical.maplet_child_offsets[parent_row + 1])
            children = np.arange(start, end, dtype=np.int64)
            children = children[coverage[children] > 0.0]
            if children.size == 0:
                continue
            score = child_feature[children] @ local[support]
            scaled_score = score / max(float(temperature), 1e-4)
            maximum = float(np.max(scaled_score))
            # Parent-wise softmax mass is always one and therefore cannot
            # reject a contextually plausible but locally incompatible
            # repeated instance.  Retain its calibrated local evidence before
            # normalizing the child coordinate distribution.
            conditional_log_evidence[support, parent_slot] = maximum + float(
                np.log(np.mean(np.exp(scaled_score - maximum)))
            )
            conditional = np.exp(scaled_score - maximum)
            conditional /= max(float(np.sum(conditional)), 1e-12)
            best = int(np.argmax(conditional))
            best_rows_by_parent[support, parent_slot] = int(children[best])
            best_probability_by_parent[support, parent_slot] = float(conditional[best])
            alternative_count = min(int(alternatives_per_parent), int(children.size))
            alternative_order = np.argsort(-conditional, kind="stable")[:alternative_count]
            rows_by_parent[
                support, parent_slot, :alternative_count
            ] = children[alternative_order]
            probability_by_parent[
                support, parent_slot, :alternative_count
            ] = conditional[alternative_order]
            for child, value in zip(children.tolist(), conditional.tolist()):
                accumulated[int(child)] = accumulated.get(int(child), 0.0) + float(parent_value) * float(value)
        ranked = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))[:keep]
        if ranked:
            output_rows[support, : len(ranked)] = [item[0] for item in ranked]
            output_probability[support, : len(ranked)] = [item[1] for item in ranked]
        output_null[support] = float(np.clip(max(parent_null[support], 1.0 - np.sum(output_probability[support])), 0.0, 1.0))
    return ChildTilePosterior(
        output_rows,
        output_probability.astype(np.float32),
        output_null.astype(np.float32),
        parent_ids,
        conditional_log_evidence.astype(np.float32),
        best_rows_by_parent,
        best_probability_by_parent.astype(np.float32),
        rows_by_parent,
        probability_by_parent.astype(np.float32),
    )
