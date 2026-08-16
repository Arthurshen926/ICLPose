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


SELECTION_SEMANTICS = "joint_token_evidence_per_physical_area_knapsack_v2"
LEGACY_SELECTION_SEMANTICS = "posterior_mass_per_physical_area_knapsack_v1"
CHILD_PROBABILITY_SEMANTICS = (
    "truncated_joint_parent_child_probability_per_radio_token_v1"
)
EVIDENCE_TOKEN_SUM = "joint_token_sum_v1"
EVIDENCE_BLOCK_CAPPED_SUM = "joint_4x4_block_capped_sum_v1"
EVIDENCE_BLOCK_MAX_SUM = "joint_4x4_block_max_sum_v1"
EVIDENCE_SEMANTICS = (
    EVIDENCE_TOKEN_SUM,
    EVIDENCE_BLOCK_CAPPED_SUM,
    EVIDENCE_BLOCK_MAX_SUM,
)


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
    evidence_semantics: str = EVIDENCE_TOKEN_SUM
    connected_component_count: int = 0
    maximum_connected_components: int = 0

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
            or str(self.evidence_semantics) not in EVIDENCE_SEMANTICS
            or int(self.connected_component_count) < 0
            or int(self.maximum_connected_components) < 0
            or (
                int(self.maximum_connected_components) > 0
                and int(self.connected_component_count)
                > int(self.maximum_connected_components)
            )
        ):
            raise ValueError("invalid fine-support selection")
        object.__setattr__(self, "child_rows", rows)
        object.__setattr__(self, "posterior_mass", mass.astype(np.float32))
        object.__setattr__(self, "posterior_density", density.astype(np.float32))

    @property
    def summed_token_evidence_mass(self) -> np.ndarray:
        """Non-calibrated evidence name retained alongside the legacy field."""

        return self.posterior_mass

    @property
    def eligible_summed_token_evidence_mass(self) -> float:
        return float(self.eligible_posterior_mass)

    @property
    def target_eligible_evidence_fraction(self) -> float:
        return float(self.target_posterior_mass_fraction)


def aggregate_child_evidence(
    token_child_rows: np.ndarray,
    token_child_probabilities: np.ndarray,
    token_xy: np.ndarray,
    *,
    child_count: int,
    semantics: str = EVIDENCE_TOKEN_SUM,
    block_size: int = 4,
) -> np.ndarray:
    """Aggregate frozen joint token probabilities without probability claims.

    ``block_capped`` and ``block_max`` are explicit correlated-token controls.
    They do not alter the token posterior or renormalize candidate tails.
    """

    rows = np.asarray(token_child_rows, dtype=np.int64)
    probability = np.asarray(token_child_probabilities, dtype=np.float64)
    xy = np.asarray(token_xy, dtype=np.int64)
    count = int(child_count)
    mode = str(semantics)
    if (
        rows.ndim != 2
        or probability.shape != rows.shape
        or xy.shape != (rows.shape[0], 2)
        or count <= 0
        or int(block_size) <= 0
        or mode not in EVIDENCE_SEMANTICS
        or np.any(~np.isfinite(probability))
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ValueError("invalid child evidence aggregation input")
    valid = (rows >= 0) & (rows < count) & (probability > 0.0)
    evidence = np.zeros((count,), dtype=np.float64)
    if not np.any(valid):
        return evidence
    if mode == EVIDENCE_TOKEN_SUM:
        np.add.at(evidence, rows[valid], probability[valid])
        return evidence

    block_xy = xy // int(block_size)
    block_width = int(np.max(block_xy[:, 0])) + 1
    block = block_xy[:, 1] * block_width + block_xy[:, 0]
    flat_block = np.broadcast_to(block[:, None], rows.shape)[valid]
    flat_row = rows[valid]
    flat_probability = probability[valid]
    key = flat_block.astype(np.int64) * count + flat_row
    order = np.argsort(key, kind="stable")
    ordered_key = key[order]
    ordered_probability = flat_probability[order]
    start = np.r_[0, np.flatnonzero(ordered_key[1:] != ordered_key[:-1]) + 1]
    child = (ordered_key[start] % count).astype(np.int64)
    if mode == EVIDENCE_BLOCK_CAPPED_SUM:
        value = np.minimum(1.0, np.add.reduceat(ordered_probability, start))
    else:
        value = np.maximum.reduceat(ordered_probability, start)
    np.add.at(evidence, child, value)
    return evidence


def audit_joint_child_probability_contract(
    token_parent_ids: np.ndarray,
    token_parent_probabilities: np.ndarray,
    token_child_rows: np.ndarray,
    token_child_probabilities: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    tolerance: float = 2e-5,
) -> dict[str, object]:
    """Prove that stored child mass is truncated joint parent-child mass.

    For every token and physical parent, retained child mass cannot exceed the
    corresponding retained parent mass.  This is the observable conservation
    law of ``P(p|u) P(c|p,u)`` after child Top-K truncation.
    """

    parent_ids = np.asarray(token_parent_ids, dtype=np.int64)
    parent_probability = np.asarray(token_parent_probabilities, dtype=np.float64)
    child_rows = np.asarray(token_child_rows, dtype=np.int64)
    child_probability = np.asarray(token_child_probabilities, dtype=np.float64)
    if (
        parent_ids.ndim != 2
        or parent_probability.shape != parent_ids.shape
        or child_rows.ndim != 2
        or child_probability.shape != child_rows.shape
        or parent_ids.shape[0] != child_rows.shape[0]
        or np.any(~np.isfinite(parent_probability))
        or np.any(~np.isfinite(child_probability))
        or np.any(parent_probability < 0.0)
        or np.any(child_probability < 0.0)
    ):
        raise ValueError("invalid parent-child probability audit input")
    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    parent_rows = np.asarray(
        [[parent_row_by_id.get(int(value), -1) for value in row]
         for row in parent_ids.tolist()],
        dtype=np.int64,
    )
    child_count = int(physical.child_parent_rows.size)
    if np.any((child_rows < 0) & (child_probability > 0.0)):
        raise ValueError("invalid child row carries positive probability")
    if np.any((parent_rows < 0) & (parent_probability > 0.0)):
        raise ValueError("invalid parent row carries positive probability")
    valid_child = (
        (child_rows >= 0)
        & (child_rows < child_count)
        & (child_probability > 0.0)
    )
    maximum_excess = 0.0
    violation_count = 0
    retained_child_mass = float(np.sum(child_probability[valid_child]))
    token_child_mass = np.sum(
        np.where(valid_child, child_probability, 0.0), axis=1
    )
    parent_count = int(physical.maplet_ids.size)
    token_index_parent = np.broadcast_to(
        np.arange(parent_ids.shape[0], dtype=np.int64)[:, None], parent_rows.shape
    )
    valid_parent = (parent_rows >= 0) & (parent_probability > 0.0)
    parent_key = (
        token_index_parent[valid_parent] * parent_count + parent_rows[valid_parent]
    )
    parent_order = np.argsort(parent_key, kind="stable")
    parent_key = parent_key[parent_order]
    parent_value = parent_probability[valid_parent][parent_order]
    parent_start = np.r_[
        0, np.flatnonzero(parent_key[1:] != parent_key[:-1]) + 1
    ] if parent_key.size else np.zeros(0, dtype=np.int64)
    unique_parent_key = parent_key[parent_start]
    unique_parent_mass = (
        np.add.reduceat(parent_value, parent_start)
        if parent_start.size else np.zeros(0, dtype=np.float64)
    )
    retained_parent_mass = float(np.sum(unique_parent_mass))

    token_index_child = np.broadcast_to(
        np.arange(child_rows.shape[0], dtype=np.int64)[:, None], child_rows.shape
    )
    child_parent = physical.child_parent_rows[child_rows[valid_child]]
    child_key = token_index_child[valid_child] * parent_count + child_parent
    child_order = np.argsort(child_key, kind="stable")
    child_key = child_key[child_order]
    child_value = child_probability[valid_child][child_order]
    child_start = np.r_[
        0, np.flatnonzero(child_key[1:] != child_key[:-1]) + 1
    ] if child_key.size else np.zeros(0, dtype=np.int64)
    unique_child_key = child_key[child_start]
    unique_child_mass = (
        np.add.reduceat(child_value, child_start)
        if child_start.size else np.zeros(0, dtype=np.float64)
    )
    if unique_child_key.size:
        location = np.searchsorted(unique_parent_key, unique_child_key)
        present = location < unique_parent_key.size
        valid_location = np.flatnonzero(present)
        present[valid_location] = (
            unique_parent_key[location[valid_location]]
            == unique_child_key[valid_location]
        )
        matching_parent_mass = np.zeros_like(unique_child_mass)
        matching_parent_mass[present] = unique_parent_mass[location[present]]
        excess = unique_child_mass - matching_parent_mass
        maximum_excess = float(np.max(excess, initial=0.0))
        violation_count = int(np.sum(excess > float(tolerance)))
    if violation_count:
        raise ValueError("child joint mass exceeds its retained parent mass")
    return {
        "child_probability_semantics": CHILD_PROBABILITY_SEMANTICS,
        "joint_mass_conservation_verified": True,
        "parent_child_mass_violation_count": 0,
        "maximum_parent_child_mass_excess": float(max(maximum_excess, 0.0)),
        "retained_child_to_parent_mass_ratio": float(
            retained_child_mass / max(retained_parent_mass, 1e-12)
        ),
        "retained_child_topk_is_subprobability": True,
        "maximum_retained_child_mass_per_token": float(
            np.max(token_child_mass, initial=0.0)
        ),
        "capped_token_coverage_equals_additive_joint_mass": bool(
            np.all(token_child_mass <= 1.0 + float(tolerance))
        ),
        "reason": (
            "atomic child labels are mutually exclusive within each token; "
            "sum over any selected subset cannot exceed one"
        ),
        "calibrated_credible_set_claim": False,
    }


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
    token_xy: np.ndarray | None = None,
    evidence_semantics: str = EVIDENCE_TOKEN_SUM,
    evidence_block_size: int = 4,
    maximum_connected_components: int = 0,
    maximum_normal_angle_degrees: float = 30.0,
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
    if token_xy is None:
        if str(evidence_semantics) != EVIDENCE_TOKEN_SUM:
            raise ValueError("block evidence requires token coordinates")
        xy = np.column_stack(
            (np.arange(rows.shape[0], dtype=np.int64), np.zeros(rows.shape[0], dtype=np.int64))
        )
    else:
        xy = np.asarray(token_xy, dtype=np.int64)
    mass = aggregate_child_evidence(
        rows,
        probability,
        xy,
        child_count=child_count,
        semantics=str(evidence_semantics),
        block_size=int(evidence_block_size),
    )

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
    maximum_components = int(maximum_connected_components)
    if maximum_components < 0:
        raise ValueError("maximum connected components must be non-negative")
    child_size = float(physical.metadata.get("child_voxel_size_m", 0.0))
    normal_angle = float(maximum_normal_angle_degrees)
    if maximum_components > 0 and (
        not np.isfinite(child_size)
        or child_size <= 0.0
        or not 0.0 <= normal_angle <= 90.0
    ):
        raise ValueError("invalid component-budget geometry")
    selected_component_parent: list[int] = []
    selected_by_voxel_axis: dict[tuple[int, int, int, int], list[int]] = {}

    def component_find(value: int) -> int:
        root = int(value)
        while selected_component_parent[root] != root:
            root = selected_component_parent[root]
        while selected_component_parent[value] != value:
            next_value = selected_component_parent[value]
            selected_component_parent[value] = root
            value = next_value
        return root

    component_count = 0
    threshold = float(np.cos(np.deg2rad(normal_angle)))
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
        neighbours: set[int] = set()
        key: tuple[int, int, int, int] | None = None
        if maximum_components > 0:
            center = np.asarray(physical.child_centers[row], dtype=np.float64)
            normal = np.asarray(physical.child_normals[row], dtype=np.float64)
            voxel = np.floor(center / child_size).astype(np.int64)
            axis = int(np.argmax(np.abs(normal)))
            key = (int(voxel[0]), int(voxel[1]), int(voxel[2]), axis)
            for dx, dy, dz in (
                (-1, 0, 0), (1, 0, 0), (0, -1, 0),
                (0, 1, 0), (0, 0, -1), (0, 0, 1),
            ):
                neighbour_key = (key[0] + dx, key[1] + dy, key[2] + dz, axis)
                for selected_index in selected_by_voxel_axis.get(neighbour_key, ()):
                    prior_row = selected[selected_index]
                    prior_normal = np.asarray(
                        physical.child_normals[prior_row], dtype=np.float64
                    )
                    if float(abs(np.dot(normal, prior_normal))) >= threshold:
                        neighbours.add(component_find(selected_index))
            next_count = component_count + (1 if not neighbours else 0) - max(
                len(neighbours) - 1, 0
            )
            if next_count > maximum_components:
                continue
        selected.append(row)
        selected_members.append(members)
        selected_index = len(selected_members) - 1
        for primitive in members.tolist():
            selected_by_primitive.setdefault(int(primitive), []).append(selected_index)
        used_area += cost
        selected_mass += float(mass[row])
        if maximum_components > 0:
            selected_index = len(selected) - 1
            selected_component_parent.append(selected_index)
            if not neighbours:
                component_count += 1
            else:
                root = min(neighbours)
                selected_component_parent[selected_index] = root
                for other in sorted(neighbours):
                    other_root = component_find(other)
                    root = component_find(root)
                    if other_root != root:
                        selected_component_parent[max(root, other_root)] = min(root, other_root)
                        component_count -= 1
            assert key is not None
            selected_by_voxel_axis.setdefault(key, []).append(selected_index)

    # Density greedy can exclude one large, valuable support.  Comparing the
    # feasible singleton is the deterministic 0/1-knapsack safeguard.
    feasible = candidate_rows[area[candidate_rows] <= maximum_area + 1e-12]
    if feasible.size and maximum_components == 0:
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
        evidence_semantics=str(evidence_semantics),
        connected_component_count=int(component_count),
        maximum_connected_components=maximum_components,
    )
