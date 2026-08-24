"""Lossless-carrier audit for grouping retrieved children into components."""

from __future__ import annotations

import numpy as np

from .connected_fine_support import connected_fine_support_components
from .fine_support_selection import child_surface_area_m2
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval


def audit_connected_support_carrier(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_normal_angle_degrees: float = 30.0,
    precomputed_child_surface_area_m2: np.ndarray | None = None,
) -> dict[str, object]:
    """Prove component grouping preserves the selected physical child union."""

    selected = np.asarray(retrieval.scene_child_rows, dtype=np.int64).reshape(-1)
    score = np.asarray(retrieval.scene_child_scores, dtype=np.float64).reshape(-1)
    if (
        selected.shape != score.shape
        or np.unique(selected).size != selected.size
        or np.any(selected < 0)
        or np.any(selected >= physical.child_parent_rows.size)
        or np.any(~np.isfinite(score))
        or np.any(score < 0.0)
    ):
        raise ValueError("invalid scene child set for connected carrier")
    area = (
        child_surface_area_m2(physical)
        if precomputed_child_surface_area_m2 is None
        else np.asarray(precomputed_child_surface_area_m2, dtype=np.float64)
    )
    components = connected_fine_support_components(
        selected,
        physical,
        maximum_normal_angle_degrees=float(maximum_normal_angle_degrees),
        precomputed_child_surface_area_m2=area,
    )
    flat = np.asarray(components.component_child_rows, dtype=np.int64)
    lossless = bool(np.array_equal(np.sort(flat), np.sort(selected)))
    if not lossless:
        raise ValueError("connected support carrier changed the selected child union")
    score_by_child = np.zeros((physical.child_parent_rows.size,), dtype=np.float64)
    score_by_child[selected] = score
    sizes = np.diff(components.component_offsets).astype(np.int64)
    component_score = np.add.reduceat(
        score_by_child[flat], components.component_offsets[:-1]
    ) if flat.size else np.zeros((0,), dtype=np.float64)
    component_area = np.asarray(
        components.component_surface_area_m2, dtype=np.float64
    )
    total_score = float(np.sum(score))
    total_area = float(np.sum(area[selected]))
    return {
        "selected_child_count": int(selected.size),
        "connected_component_count": int(components.component_count),
        "singleton_component_fraction": float(
            np.mean(sizes == 1) if sizes.size else 0.0
        ),
        "mean_children_per_component": float(
            np.mean(sizes) if sizes.size else 0.0
        ),
        "maximum_children_per_component": int(np.max(sizes, initial=0)),
        "largest_component_child_fraction": float(
            np.max(sizes, initial=0) / max(selected.size, 1)
        ),
        "largest_component_score_fraction": float(
            np.max(component_score, initial=0.0) / max(total_score, 1e-12)
        ),
        "largest_component_area_fraction": float(
            np.max(component_area, initial=0.0) / max(total_area, 1e-12)
        ),
        "selected_surface_area_m2": total_area,
        "component_surface_area_m2": float(np.sum(component_area)),
        "selected_score_sum": total_score,
        "component_score_sum": float(np.sum(component_score)),
        "child_union_preserved_exactly": lossless,
        "primitive_union_preserved_by_identical_child_union": lossless,
        "surface_area_preserved": bool(
            np.isclose(np.sum(component_area), total_area, rtol=1e-10, atol=1e-10)
        ),
        "score_preserved": bool(
            np.isclose(np.sum(component_score), total_score, rtol=1e-10, atol=1e-10)
        ),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
    }
