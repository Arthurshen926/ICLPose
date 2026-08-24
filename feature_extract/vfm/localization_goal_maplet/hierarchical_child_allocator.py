"""Pose-free parent-balanced allocation of a fixed scene-child budget.

The token posterior remains frozen.  This module only changes which children
from its positive evidence union occupy the bounded scene-level return set.
It uses the already-ranked physical parents, static parent/child membership,
and the original joint child evidence; it has no GT, pose, RGB, or renderer
interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap


SEMANTICS = "parent_mass_prefix_one_child_then_global_joint_fill_v1"


@dataclass(frozen=True)
class HierarchicalChildAllocation:
    child_rows: np.ndarray
    child_scores: np.ndarray
    seeded_parent_rows: np.ndarray
    represented_parent_count: int
    suppressed_duplicate_count: int


def _members(physical: GoalMapletPhysicalMap, child: int) -> np.ndarray:
    start = int(physical.child_member_offsets[int(child)])
    end = int(physical.child_member_offsets[int(child) + 1])
    return np.unique(
        np.asarray(
            physical.child_member_primitive_rows[start:end], dtype=np.int64
        )
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    return float(intersection / max(left.size + right.size - intersection, 1))


def allocate_parent_balanced_scene_children(
    scene_parent_ids: np.ndarray,
    scene_parent_scores: np.ndarray,
    child_scores: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    parent_mass_fraction: float,
    maximum_children: int = 64,
    maximum_primitive_iou: float = 0.50,
) -> HierarchicalChildAllocation:
    """Seed one child for a parent-score prefix, then globally fill the budget."""

    parent_ids = np.asarray(scene_parent_ids, dtype=np.int64).reshape(-1)
    parent_scores = np.asarray(scene_parent_scores, dtype=np.float64).reshape(-1)
    score = np.asarray(child_scores, dtype=np.float64).reshape(-1)
    fraction = float(parent_mass_fraction)
    if (
        parent_ids.shape != parent_scores.shape
        or score.shape != physical.child_parent_rows.shape
        or np.any(~np.isfinite(parent_scores))
        or np.any(parent_scores < 0.0)
        or np.any(~np.isfinite(score))
        or np.any(score < 0.0)
        or not 0.0 <= fraction <= 1.0
        or int(maximum_children) <= 0
        or not 0.0 <= float(maximum_primitive_iou) <= 1.0
    ):
        raise ValueError("invalid parent-balanced child allocation input")
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    parent_rows = np.asarray(
        [parent_row_by_id.get(int(value), -1) for value in parent_ids.tolist()],
        dtype=np.int64,
    )
    if np.any(parent_rows < 0) or np.unique(parent_rows).size != parent_rows.size:
        raise ValueError("scene parents differ from the physical hierarchy")

    positive_parent = parent_scores > 0.0
    eligible_parent_rows = parent_rows[positive_parent]
    eligible_parent_scores = parent_scores[positive_parent]
    if fraction <= 0.0 or eligible_parent_rows.size == 0:
        seeded_parent_rows = np.zeros((0,), dtype=np.int64)
    else:
        cumulative = np.cumsum(eligible_parent_scores)
        target = fraction * float(cumulative[-1])
        count = int(np.searchsorted(cumulative, target, side="left") + 1)
        seeded_parent_rows = eligible_parent_rows[:count]

    global_order = np.lexsort(
        (np.arange(score.size, dtype=np.int64), -score)
    )
    global_order = global_order[score[global_order] > 0.0]
    selected: list[int] = []
    members: list[np.ndarray] = []
    suppressed = 0

    def try_add(child: int) -> bool:
        nonlocal suppressed
        row = int(child)
        if row in selected:
            return False
        candidate_members = _members(physical, row)
        if any(
            _iou(candidate_members, prior) >= float(maximum_primitive_iou)
            for prior in members
        ):
            suppressed += 1
            return False
        selected.append(row)
        members.append(candidate_members)
        return True

    child_parent = np.asarray(physical.child_parent_rows, dtype=np.int64)
    for parent_row in seeded_parent_rows.tolist():
        for child in global_order[child_parent[global_order] == int(parent_row)].tolist():
            if try_add(int(child)):
                break
        if len(selected) >= int(maximum_children):
            break
    if len(selected) < int(maximum_children):
        for child in global_order.tolist():
            try_add(int(child))
            if len(selected) >= int(maximum_children):
                break
    rows = np.asarray(selected, dtype=np.int64)
    represented = int(np.unique(child_parent[rows]).size) if rows.size else 0
    return HierarchicalChildAllocation(
        child_rows=rows,
        child_scores=score[rows].astype(np.float32),
        seeded_parent_rows=seeded_parent_rows,
        represented_parent_count=represented,
        suppressed_duplicate_count=int(suppressed),
    )
