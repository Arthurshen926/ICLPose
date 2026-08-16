"""Evaluator-only attribution for fine physical-support retrieval loss.

This module never changes a retrieval result.  It opens frozen token/scene
posteriors and coordinate-correct 2DGS visibility after retrieval, then asks
where visible fine-surface mass was lost: representation, parent gating,
token candidates, aggregation, or physical budget.  Ground truth is used only
for diagnostics and oracle ceilings and is never written back to a retriever.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import sparse

from .fine_support_selection import child_surface_area_m2
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval, aggregate_sparse_token_evidence


DEFAULT_AREA_BUDGETS = (0.01, 0.02, 0.05, 0.10)


def _child_surface_area(physical: GoalMapletPhysicalMap) -> np.ndarray:
    return child_surface_area_m2(physical)


def _greedy_area_selection(
    pool: np.ndarray,
    priority: np.ndarray,
    area: np.ndarray,
    *,
    maximum_area: float,
) -> np.ndarray:
    rows = np.unique(np.asarray(pool, dtype=np.int64))
    value = np.asarray(priority, dtype=np.float64).reshape(-1)
    if (
        np.any(rows < 0)
        or np.any(rows >= area.size)
        or value.shape != area.shape
        or not np.isfinite(float(maximum_area))
        or float(maximum_area) <= 0.0
    ):
        raise ValueError("invalid fine-support area selection")
    order = rows[np.lexsort((rows, -value[rows]))]
    chosen: list[int] = []
    used = 0.0
    for row in order.tolist():
        if float(value[row]) <= 0.0:
            break
        cost = float(area[row])
        if used + cost <= float(maximum_area) + 1e-12:
            chosen.append(int(row))
            used += cost
    return np.asarray(chosen, dtype=np.int64)


def _curve_row(
    selected: np.ndarray,
    truth_mass: np.ndarray,
    area: np.ndarray,
    total_visible_mass: float,
) -> dict[str, object]:
    rows = np.unique(np.asarray(selected, dtype=np.int64))
    return {
        "exact_visible_mass_recall": float(
            np.sum(truth_mass[rows]) / max(float(total_visible_mass), 1e-12)
        ),
        "selected_child_count": int(rows.size),
        "selected_surface_area_m2": float(np.sum(area[rows])),
    }


def evaluate_child_loss_decomposition(
    retrieval: PureRadioPhysicalRetrieval,
    token_primitive: sparse.csr_matrix,
    primitive_to_parent: sparse.csr_matrix,
    primitive_to_child: sparse.csr_matrix,
    physical: GoalMapletPhysicalMap,
    *,
    canonical_supported_primitive: np.ndarray,
    current_surface_metrics: Mapping[str, object] | None = None,
    area_budgets: Sequence[float] = DEFAULT_AREA_BUDGETS,
) -> dict[str, object]:
    """Attribute one frozen retrieval result without changing its ranking."""

    primitive_truth = np.asarray(token_primitive.sum(axis=0)).reshape(-1)
    parent_truth = np.asarray(
        (token_primitive @ primitive_to_parent).sum(axis=0)
    ).reshape(-1)
    child_truth = np.asarray(
        (token_primitive @ primitive_to_child).sum(axis=0)
    ).reshape(-1)
    total = max(float(np.sum(primitive_truth)), 1e-12)
    supported = np.asarray(canonical_supported_primitive, dtype=bool).reshape(-1)
    if supported.shape != primitive_truth.shape:
        raise ValueError("canonical support mask differs from physical primitives")

    parent_row_by_id = {
        int(value): row for row, value in enumerate(physical.maplet_ids.tolist())
    }
    selected_parent = np.zeros((physical.maplet_ids.size,), dtype=bool)
    for parent_id in retrieval.scene_parent_ids.tolist():
        row = parent_row_by_id.get(int(parent_id))
        if row is not None:
            selected_parent[row] = True
    oracle_parent = parent_truth > 0.0

    candidate_child = np.zeros((physical.child_parent_rows.size,), dtype=bool)
    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    valid = (rows >= 0) & (rows < candidate_child.size)
    candidate_child[rows[valid]] = True
    selected_child = np.zeros_like(candidate_child)
    selected_child[np.asarray(retrieval.scene_child_rows, dtype=np.int64)] = True
    child_score = aggregate_sparse_token_evidence(
        retrieval.token_xy,
        retrieval.token_child_rows,
        retrieval.token_child_probabilities,
        entity_count=physical.child_parent_rows.size,
        token_height=int(retrieval.metadata["token_height"]),
        token_width=int(retrieval.metadata["token_width"]),
    )
    positive_child = child_score > 0.0
    child_parent_selected = selected_parent[physical.child_parent_rows]
    child_parent_oracle = oracle_parent[physical.child_parent_rows]

    # C0 is independent of the later hierarchy.  C1--C4 form an exclusive
    # chain inside the current representation and selected-parent contract.
    c0 = float(np.sum(primitive_truth[~supported]) / total)
    c1 = float(np.sum(child_truth[~child_parent_selected]) / total)
    remaining_parent = child_parent_selected
    c2 = float(np.sum(child_truth[remaining_parent & ~candidate_child]) / total)
    c3 = float(
        np.sum(child_truth[remaining_parent & candidate_child & ~positive_child])
        / total
    )
    c4_mask = remaining_parent & candidate_child & positive_child & ~selected_child
    c4 = float(np.sum(child_truth[c4_mask]) / total)
    current_exact = float(np.sum(child_truth[selected_child]) / total)
    candidate_exact = float(np.sum(child_truth[candidate_child]) / total)
    current_parent_candidate = candidate_child & child_parent_selected
    current_parent_candidate_exact = float(
        np.sum(child_truth[current_parent_candidate]) / total
    )
    oracle_parent_candidate = candidate_child & child_parent_oracle
    oracle_parent_candidate_exact = float(
        np.sum(child_truth[oracle_parent_candidate]) / total
    )

    primitive_area = (
        np.pi
        * np.asarray(physical.primitive_scale1, dtype=np.float64)
        * np.asarray(physical.primitive_scale2, dtype=np.float64)
    )
    total_map_area = float(np.sum(primitive_area))
    child_area = _child_surface_area(physical)
    selected_rows = np.flatnonzero(selected_child)
    selected_member_rows: list[np.ndarray] = []
    for child in selected_rows.tolist():
        start = int(physical.child_member_offsets[child])
        end = int(physical.child_member_offsets[child + 1])
        selected_member_rows.append(
            np.asarray(
                physical.child_member_primitive_rows[start:end], dtype=np.int64
            )
        )
    union = (
        np.unique(np.concatenate(selected_member_rows))
        if selected_member_rows
        else np.zeros(0, dtype=np.int64)
    )
    summed_area = float(np.sum(child_area[selected_rows]))
    union_area = float(np.sum(primitive_area[union]))
    overlap_waste = max(summed_area - union_area, 0.0) / max(summed_area, 1e-12)

    surface = dict(current_surface_metrics or {})
    exact_from_surface = float(surface.get("exact_visible_mass_recall", current_exact))
    tolerant = float(
        surface.get("tolerant_visible_mass_recall_0.5m", exact_from_surface)
    )
    c5 = max(tolerant - exact_from_surface, 0.0)
    c7 = float(
        np.mean(
            np.asarray(retrieval.token_out_of_map_probabilities, dtype=np.float64)
            + np.asarray(
                retrieval.token_in_map_tail_probabilities, dtype=np.float64
            )
        )
    )

    curves: dict[str, object] = {}
    pools = {
        "current_parent_current_candidates": np.flatnonzero(current_parent_candidate),
        "oracle_parent_current_candidates": np.flatnonzero(oracle_parent_candidate),
        "current_parent_oracle_all_children": np.flatnonzero(child_parent_selected),
    }
    for fraction in area_budgets:
        budget = float(fraction)
        if not 0.0 < budget <= 1.0:
            raise ValueError("area budgets must be in (0,1]")
        maximum_area = budget * total_map_area
        key = f"area_{budget:.2f}"
        curves[key] = {}
        for name, pool in pools.items():
            if name == "current_parent_oracle_all_children":
                priority = np.divide(
                    child_truth,
                    child_area,
                    out=np.zeros_like(child_truth),
                    where=child_area > 0.0,
                )
            else:
                priority = np.divide(
                    child_score,
                    child_area,
                    out=np.zeros_like(child_score),
                    where=child_area > 0.0,
                )
            chosen = _greedy_area_selection(
                pool, priority, child_area, maximum_area=maximum_area
            )
            curves[key][name] = _curve_row(chosen, child_truth, child_area, total)
        matched_pool = np.flatnonzero(current_parent_candidate)
        matched_truth_priority = np.divide(
            child_truth,
            child_area,
            out=np.zeros_like(child_truth),
            where=child_area > 0.0,
        )
        matched_oracle = _greedy_area_selection(
            matched_pool,
            matched_truth_priority,
            child_area,
            maximum_area=maximum_area,
        )
        curves[key]["matched_candidate_gt_budget_oracle"] = _curve_row(
            matched_oracle, child_truth, child_area, total
        )
        curves[key]["matched_candidate_unconstrained_ceiling"] = _curve_row(
            matched_pool, child_truth, child_area, total
        )

    attribution = {
        "C0_visible_mass_without_canonical_feature": c0,
        "C1_visible_mass_parent_not_selected": c1,
        "C2_visible_mass_child_not_in_token_candidates_given_parent": c2,
        "C3_visible_mass_candidate_with_zero_scene_evidence": c3,
        "C4_downstream_ranking_budget_and_set_construction_residual": c4,
        "C5_boundary_or_wrong_scale_credit_within_0.5m": c5,
        "C6_selected_child_area_overlap_waste_fraction": float(overlap_waste),
        "C7_mean_token_null_plus_tail_probability": c7,
    }
    dominant = max(attribution, key=lambda name: float(attribution[name]))
    return {
        "image_id": retrieval.image_id,
        "current_scene_child_exact_recall": current_exact,
        "token_candidate_union_exact_recall_ceiling": candidate_exact,
        "current_parent_candidate_union_exact_recall_ceiling": (
            current_parent_candidate_exact
        ),
        "oracle_parent_current_candidate_exact_recall_ceiling": (
            oracle_parent_candidate_exact
        ),
        "token_candidate_unique_child_count": int(np.sum(candidate_child)),
        "current_selected_parent_count": int(np.sum(selected_parent)),
        "oracle_visible_parent_count": int(np.sum(oracle_parent)),
        "current_selected_child_count": int(np.sum(selected_child)),
        "attribution": attribution,
        "dominant_attribution": dominant,
        "area_curves": curves,
        "oracle_candidate_universe_contract": {
            "full_map_oracle": "all canonical child supports",
            "parent_oracle": "all child supports in selected physical parents",
            "candidate_oracle": "stored sparse token child candidate union",
            "budgeted_candidate_oracle": (
                "stored sparse candidates in selected parents under identical map-area budget"
            ),
            "matched_candidate_oracle_is_evaluator_only": True,
        },
        "claim_scope": {
            "evaluator_only": True,
            "ground_truth_not_returned_to_retrieval": True,
            "oracle_rows_are_upper_bound_diagnostics": True,
            "not_pose_recall": True,
        },
    }
