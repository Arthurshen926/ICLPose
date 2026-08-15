"""Budgeted set selection for pose-free fine physical support retrieval.

The child posterior is already a distribution over physical surface supports
for every RADIO token.  Consequently, the expected visible token mass covered
by a returned set is the sum of its posterior mass; it is not the sum of a
fixed number of per-region peak responses.  This module selects that set under
an explicit 2DGS surface-area budget and suppresses duplicate primitive
support.  It has no pose, image-feature matcher, PnP, renderer, or ground-truth
interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physical_map import GoalMapletPhysicalMap


SELECTION_SEMANTICS = "posterior_mass_per_physical_area_knapsack_v1"


def child_surface_area_m2(physical: GoalMapletPhysicalMap) -> np.ndarray:
    """Return unique 2DGS ellipse area represented by every child support."""

    primitive_area = (
        np.pi
        * np.asarray(physical.primitive_scale1, dtype=np.float64)
        * np.asarray(physical.primitive_scale2, dtype=np.float64)
    )
    offsets = np.asarray(physical.child_member_offsets, dtype=np.int64)
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("fine supports must contain at least one primitive")
    member_area = primitive_area[
        np.asarray(physical.child_member_primitive_rows, dtype=np.int64)
    ]
    area = np.add.reduceat(member_area, offsets[:-1]).astype(np.float64)
    if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
        raise ValueError("fine supports require positive finite physical area")
    return area


def total_map_surface_area_m2(physical: GoalMapletPhysicalMap) -> float:
    area = (
        np.pi
        * np.asarray(physical.primitive_scale1, dtype=np.float64)
        * np.asarray(physical.primitive_scale2, dtype=np.float64)
    )
    total = float(np.sum(area))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("physical map surface area must be positive and finite")
    return total


def _members(physical: GoalMapletPhysicalMap, child: int) -> np.ndarray:
    start = int(physical.child_member_offsets[int(child)])
    end = int(physical.child_member_offsets[int(child) + 1])
    return np.unique(
        np.asarray(physical.child_member_primitive_rows[start:end], dtype=np.int64)
    )


def _intersection_size(left: np.ndarray, right: np.ndarray) -> int:
    i = j = count = 0
    while i < left.size and j < right.size:
        a, b = int(left[i]), int(right[j])
        if a == b:
            count += 1
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return count


@dataclass(frozen=True)
class FineSupportSelection:
    child_rows: np.ndarray
    posterior_mass: np.ndarray
    posterior_density: np.ndarray
    selected_surface_area_m2: float
    maximum_surface_area_m2: float
    candidate_child_count: int
    suppressed_duplicate_count: int
    eligible_posterior_mass: float
    target_posterior_mass_fraction: float

    def __post_init__(self) -> None:
        rows = np.asarray(self.child_rows, dtype=np.int64).reshape(-1)
        mass = np.asarray(self.posterior_mass, dtype=np.float64).reshape(-1)
        density = np.asarray(self.posterior_density, dtype=np.float64).reshape(-1)
        if (
            rows.shape != mass.shape
            or rows.shape != density.shape
            or np.unique(rows).size != rows.size
            or np.any(rows < 0)
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
            or np.any(~np.isfinite(density))
            or np.any(density <= 0.0)
            or not np.isfinite(float(self.selected_surface_area_m2))
            or not np.isfinite(float(self.maximum_surface_area_m2))
            or float(self.selected_surface_area_m2)
            > float(self.maximum_surface_area_m2) + 1e-8
            or int(self.candidate_child_count) < rows.size
            or int(self.suppressed_duplicate_count) < 0
            or not np.isfinite(float(self.eligible_posterior_mass))
            or float(self.eligible_posterior_mass) < float(np.sum(mass)) - 1e-6
            or not 0.0 < float(self.target_posterior_mass_fraction) <= 1.0
        ):
            raise ValueError("invalid fine-support selection")
        object.__setattr__(self, "child_rows", rows)
        object.__setattr__(self, "posterior_mass", mass.astype(np.float32))
        object.__setattr__(self, "posterior_density", density.astype(np.float32))


def select_fine_supports_under_area_budget(
    token_child_rows: np.ndarray,
    token_child_probabilities: np.ndarray,
    selected_parent_ids: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_area_fraction: float,
    maximum_children: int = 2048,
    maximum_primitive_iou: float = 0.50,
    target_candidate_posterior_mass_fraction: float = 1.0,
    precomputed_child_surface_area_m2: np.ndarray | None = None,
    precomputed_total_map_surface_area_m2: float | None = None,
) -> FineSupportSelection:
    """Maximize posterior visible mass under an explicit physical-area budget.

    The deterministic density-greedy solution is the fractional-knapsack
    priority for expected visible mass.  We additionally compare it with the
    best feasible singleton, giving the standard bounded 0/1-knapsack control.
    Physical IoU suppression prevents overlapping hierarchy entries from
    buying the same primitive support twice.
    """

    rows = np.asarray(token_child_rows, dtype=np.int64)
    probability = np.asarray(token_child_probabilities, dtype=np.float64)
    parent_ids = np.asarray(selected_parent_ids, dtype=np.int64).reshape(-1)
    fraction = float(maximum_area_fraction)
    target_fraction = float(target_candidate_posterior_mass_fraction)
    if (
        rows.ndim != 2
        or probability.shape != rows.shape
        or np.any(~np.isfinite(probability))
        or np.any((probability < 0.0) | (probability > 1.0))
        or np.unique(parent_ids).size != parent_ids.size
        or not 0.0 < fraction <= 1.0
        or int(maximum_children) <= 0
        or not 0.0 <= float(maximum_primitive_iou) <= 1.0
        or not 0.0 < target_fraction <= 1.0
    ):
        raise ValueError("invalid fine-support budget input")

    child_count = int(physical.child_parent_rows.size)
    valid = (rows >= 0) & (rows < child_count) & (probability > 0.0)
    mass = np.zeros((child_count,), dtype=np.float64)
    if np.any(valid):
        np.add.at(mass, rows[valid], probability[valid])

    parent_row_by_id = {
        int(parent_id): row
        for row, parent_id in enumerate(physical.maplet_ids.tolist())
    }
    selected_parent_rows = np.asarray(
        [parent_row_by_id[value] for value in parent_ids.tolist() if value in parent_row_by_id],
        dtype=np.int64,
    )
    parent_allowed = np.zeros((physical.maplet_ids.size,), dtype=bool)
    parent_allowed[selected_parent_rows] = True
    candidate = (mass > 0.0) & parent_allowed[physical.child_parent_rows]
    candidate_rows = np.flatnonzero(candidate)
    area = (
        child_surface_area_m2(physical)
        if precomputed_child_surface_area_m2 is None
        else np.asarray(precomputed_child_surface_area_m2, dtype=np.float64).reshape(-1)
    )
    total_area = (
        total_map_surface_area_m2(physical)
        if precomputed_total_map_surface_area_m2 is None
        else float(precomputed_total_map_surface_area_m2)
    )
    if (
        area.shape != (child_count,)
        or np.any(~np.isfinite(area))
        or np.any(area <= 0.0)
        or not np.isfinite(total_area)
        or total_area <= 0.0
    ):
        raise ValueError("invalid precomputed physical-area ledger")
    maximum_area = fraction * total_area
    density = np.divide(mass, area, out=np.zeros_like(mass), where=area > 0.0)
    order = candidate_rows[
        np.lexsort((candidate_rows, -mass[candidate_rows], -density[candidate_rows]))
    ]

    selected: list[int] = []
    selected_members: list[np.ndarray] = []
    selected_by_primitive: dict[int, list[int]] = {}
    used_area = 0.0
    suppressed = 0
    eligible_mass = float(np.sum(mass[candidate_rows]))
    target_mass = target_fraction * eligible_mass
    selected_mass = 0.0
    for row in order.tolist():
        if selected_mass >= target_mass - 1e-12:
            break
        if len(selected) >= int(maximum_children):
            break
        cost = float(area[row])
        if used_area + cost > maximum_area + 1e-12:
            continue
        members = _members(physical, row)
        duplicate = False
        possible_prior = sorted(
            {
                selected_index
                for primitive in members.tolist()
                for selected_index in selected_by_primitive.get(int(primitive), ())
            }
        )
        for selected_index in possible_prior:
            prior = selected_members[selected_index]
            intersection = _intersection_size(members, prior)
            union = int(members.size + prior.size - intersection)
            if float(intersection / max(union, 1)) >= float(maximum_primitive_iou):
                duplicate = True
                break
        if duplicate:
            suppressed += 1
            continue
        selected.append(row)
        selected_members.append(members)
        selected_index = len(selected_members) - 1
        for primitive in members.tolist():
            selected_by_primitive.setdefault(int(primitive), []).append(selected_index)
        used_area += cost
        selected_mass += float(mass[row])

    # Density greedy can exclude one large, valuable support.  Comparing the
    # feasible singleton is the deterministic 0/1-knapsack safeguard.
    feasible = candidate_rows[area[candidate_rows] <= maximum_area + 1e-12]
    if feasible.size:
        singleton = int(
            feasible[np.lexsort((feasible, -mass[feasible]))[0]]
        )
        if float(mass[singleton]) > float(np.sum(mass[selected])):
            selected = [singleton]
            used_area = float(area[singleton])

    selected_rows = np.asarray(selected, dtype=np.int64)
    return FineSupportSelection(
        child_rows=selected_rows,
        posterior_mass=mass[selected_rows],
        posterior_density=density[selected_rows],
        selected_surface_area_m2=used_area,
        maximum_surface_area_m2=maximum_area,
        candidate_child_count=int(candidate_rows.size),
        suppressed_duplicate_count=int(suppressed),
        eligible_posterior_mass=eligible_mass,
        target_posterior_mass_fraction=target_fraction,
    )
