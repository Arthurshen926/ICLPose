"""Hierarchical child-tile posterior from one canonical surface field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap


def _rank_sparse_joint_topk(
    token_rows: np.ndarray,
    child_rows: np.ndarray,
    probabilities: np.ndarray,
    *,
    token_count: int,
    keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank sparse rows per token without globally sorting every candidate.

    Input is ordered by ``(token, child)`` and contains one value per key.
    At the kth boundary all values strictly above the threshold are retained,
    then the smallest child rows fill an exact tie.  This reproduces the
    historical stable ``(-probability, child_row)`` ordering.
    """

    token = np.asarray(token_rows, dtype=np.int64).reshape(-1)
    child = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    count = int(token_count)
    width = int(keep)
    if (
        child.shape != token.shape
        or probability.shape != token.shape
        or count <= 0
        or width <= 0
        or np.any((token < 0) | (token >= count))
        or np.any(child < 0)
        or np.any(~np.isfinite(probability))
        or np.any(probability <= 0.0)
        or (token.size > 1 and np.any(token[1:] < token[:-1]))
    ):
        raise ValueError("invalid sparse child ranking input")
    output_rows = np.full((count, width), -1, dtype=np.int64)
    output_probability = np.zeros((count, width), dtype=np.float64)
    counts = np.bincount(token, minlength=count)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    for token_row in np.flatnonzero(counts).tolist():
        start = int(starts[token_row])
        end = start + int(counts[token_row])
        local_child = child[start:end]
        local_probability = probability[start:end]
        retained_count = min(width, int(local_child.size))
        if local_child.size > retained_count:
            threshold_index = local_child.size - retained_count
            threshold = float(
                np.partition(local_probability, threshold_index)[threshold_index]
            )
            above = np.flatnonzero(local_probability > threshold)
            tied = np.flatnonzero(local_probability == threshold)
            needed = retained_count - int(above.size)
            retained = np.concatenate((above, tied[:needed]))
        else:
            retained = np.arange(local_child.size, dtype=np.int64)
        order = np.lexsort(
            (local_child[retained], -local_probability[retained])
        )
        retained = retained[order]
        output_rows[token_row, :retained_count] = local_child[retained]
        output_probability[token_row, :retained_count] = local_probability[retained]
    return output_rows, output_probability


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
    local = local / np.maximum(
        np.linalg.norm(local, axis=1, keepdims=True), 1e-8
    )
    child_feature = child_feature / np.maximum(
        np.linalg.norm(child_feature, axis=1, keepdims=True), 1e-8
    )
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    parent_rows = np.asarray(
        [
            [parent_row_by_id.get(int(value), -1) for value in values]
            for values in parent_ids.tolist()
        ],
        dtype=np.int64,
    )
    keep = max(1, int(maximum_child_candidates))
    output_rows = np.full((local.shape[0], keep), -1, dtype=np.int64)
    output_probability = np.zeros((local.shape[0], keep), dtype=np.float64)
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

    # Group equal parents, but retain only actual token×child triples.  The
    # former dense T×all-children matrix allocated ~141 MiB and then sorted all
    # 7,653 children for every token even though a token's 64 parent slots
    # expose only a small subset.  Sparse triples preserve the exact
    # P(parent|u)P(child|parent,u) factorization and deterministic child-row tie
    # break while making work proportional to the exposed hierarchy edges.
    joint_token_parts: list[np.ndarray] = []
    joint_child_parts: list[np.ndarray] = []
    joint_probability_parts: list[np.ndarray] = []
    flat_parent = parent_rows.reshape(-1)
    flat_probability = parent_probability.reshape(-1)
    valid_slots = np.flatnonzero((flat_parent >= 0) & (flat_probability > 0.0))
    if valid_slots.size:
        order = np.argsort(flat_parent[valid_slots], kind="stable")
        ordered_slots = valid_slots[order]
        ordered_parent = flat_parent[ordered_slots]
        boundaries = np.r_[
            0,
            np.flatnonzero(ordered_parent[1:] != ordered_parent[:-1]) + 1,
            ordered_parent.size,
        ]
        slot_count = parent_ids.shape[1]
        scale = max(float(temperature), 1e-4)
        for group in range(boundaries.size - 1):
            slots = ordered_slots[boundaries[group] : boundaries[group + 1]]
            parent_row = int(ordered_parent[boundaries[group]])
            start = int(physical.maplet_child_offsets[parent_row])
            end = int(physical.maplet_child_offsets[parent_row + 1])
            children = np.arange(start, end, dtype=np.int64)
            children = children[coverage[children] > 0.0]
            if children.size == 0:
                continue
            support_rows = slots // slot_count
            parent_slots = slots % slot_count
            scaled_score = (
                local[support_rows] @ child_feature[children].T
            ).astype(np.float64) / scale
            maximum = np.max(scaled_score, axis=1, keepdims=True)
            exponential = np.exp(scaled_score - maximum)
            conditional = exponential / np.maximum(
                np.sum(exponential, axis=1, keepdims=True), 1e-12
            )
            conditional_log_evidence[support_rows, parent_slots] = (
                maximum[:, 0]
                + np.log(np.mean(exponential, axis=1))
            )
            local_order = np.argsort(-conditional, axis=1, kind="stable")
            best = local_order[:, 0]
            best_rows_by_parent[support_rows, parent_slots] = children[best]
            best_probability_by_parent[support_rows, parent_slots] = conditional[
                np.arange(slots.size), best
            ]
            alternative_count = min(alternatives_per_parent, int(children.size))
            alternatives = local_order[:, :alternative_count]
            rows_by_parent[
                support_rows, parent_slots, :alternative_count
            ] = children[alternatives]
            probability_by_parent[
                support_rows, parent_slots, :alternative_count
            ] = np.take_along_axis(
                conditional, alternatives, axis=1
            )
            joint_probability = (
                flat_probability[slots, None] * conditional
            )
            joint_token_parts.append(
                np.repeat(support_rows, children.size).astype(np.int64)
            )
            joint_child_parts.append(
                np.tile(children, support_rows.size).astype(np.int64)
            )
            joint_probability_parts.append(joint_probability.reshape(-1))

    if joint_token_parts:
        joint_token = np.concatenate(joint_token_parts)
        joint_child = np.concatenate(joint_child_parts)
        joint_probability = np.concatenate(joint_probability_parts)
        key = (
            joint_token.astype(np.int64) * int(physical.child_parent_rows.size)
            + joint_child
        )
        key_order = np.argsort(key, kind="stable")
        ordered_key = key[key_order]
        ordered_probability = joint_probability[key_order]
        start = np.r_[
            0, np.flatnonzero(ordered_key[1:] != ordered_key[:-1]) + 1
        ]
        unique_key = ordered_key[start]
        unique_probability = np.add.reduceat(ordered_probability, start)
        unique_token = unique_key // int(physical.child_parent_rows.size)
        unique_child = unique_key % int(physical.child_parent_rows.size)
        positive = unique_probability > 0.0
        unique_token = unique_token[positive]
        unique_child = unique_child[positive]
        unique_probability = unique_probability[positive]
        output_rows, output_probability = _rank_sparse_joint_topk(
            unique_token,
            unique_child,
            unique_probability,
            token_count=local.shape[0],
            keep=keep,
        )
    output_null = np.clip(
        np.maximum(parent_null, 1.0 - np.sum(output_probability, axis=1)),
        0.0,
        1.0,
    )
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
