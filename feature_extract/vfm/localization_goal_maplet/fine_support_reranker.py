"""Low-capacity, pose-free features for child support reranking.

The feature extractor is deliberately query-local and translation agnostic.
It consumes only the frozen RADIO child evidence and static physical support
geometry.  Contributor visibility is never accepted by this module; labels
belong exclusively to an offline, sequence-disjoint training harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fine_support_selection import aggregate_child_evidence
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import PureRadioPhysicalRetrieval


FEATURE_SEMANTICS = "query_local_joint_evidence_and_static_geometry_v1"
FEATURE_NAMES = (
    "log1p_joint_evidence",
    "log1p_joint_evidence_density",
    "log1p_supporting_token_count",
    "maximum_token_joint_probability",
    "inverse_effective_token_count",
    "log_surface_area_m2",
    "log1p_member_primitive_count",
    "log1p_parent_joint_evidence",
    "child_fraction_of_parent_evidence",
    "log1p_extent_l1_m",
    "log1p_extent_area_m2",
)


@dataclass(frozen=True)
class ChildRerankingFeatures:
    child_rows: np.ndarray
    values: np.ndarray
    joint_evidence: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.child_rows, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.float64)
        evidence = np.asarray(self.joint_evidence, dtype=np.float64).reshape(-1)
        if (
            values.shape != (rows.size, len(FEATURE_NAMES))
            or evidence.shape != rows.shape
            or np.unique(rows).size != rows.size
            or np.any(rows < 0)
            or np.any(~np.isfinite(values))
            or np.any(~np.isfinite(evidence))
            or np.any(evidence <= 0.0)
        ):
            raise ValueError("invalid child reranking features")
        object.__setattr__(self, "child_rows", rows)
        object.__setattr__(self, "values", values.astype(np.float32))
        object.__setattr__(self, "joint_evidence", evidence.astype(np.float32))


def extract_child_reranking_features(
    retrieval: PureRadioPhysicalRetrieval,
    physical: GoalMapletPhysicalMap,
    child_surface_area_m2: np.ndarray,
) -> ChildRerankingFeatures:
    """Extract candidate features without pose, GT, RGB, or route identity."""

    child_count = int(physical.child_parent_rows.size)
    area = np.asarray(child_surface_area_m2, dtype=np.float64).reshape(-1)
    if area.shape != (child_count,) or np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("child area ledger differs")
    evidence = aggregate_child_evidence(
        retrieval.token_child_rows,
        retrieval.token_child_probabilities,
        retrieval.token_xy,
        child_count=child_count,
    )
    parent_by_id = {
        int(parent_id): row
        for row, parent_id in enumerate(physical.maplet_ids.tolist())
    }
    allowed_parent = np.zeros((physical.maplet_ids.size,), dtype=bool)
    for parent_id in retrieval.scene_parent_ids.tolist():
        row = parent_by_id.get(int(parent_id))
        if row is not None:
            allowed_parent[row] = True
    candidate = (evidence > 0.0) & allowed_parent[physical.child_parent_rows]
    child_rows = np.flatnonzero(candidate)
    if child_rows.size == 0:
        raise ValueError("reranker candidate set is empty")

    rows = np.asarray(retrieval.token_child_rows, dtype=np.int64)
    probability = np.asarray(
        retrieval.token_child_probabilities, dtype=np.float64
    )
    valid = (rows >= 0) & (rows < child_count) & (probability > 0.0)
    flat_rows = rows[valid]
    flat_probability = probability[valid]
    token_count = np.bincount(flat_rows, minlength=child_count).astype(np.float64)
    maximum = np.zeros((child_count,), dtype=np.float64)
    np.maximum.at(maximum, flat_rows, flat_probability)
    squared = np.bincount(
        flat_rows, weights=np.square(flat_probability), minlength=child_count
    ).astype(np.float64)
    inverse_effective = np.divide(
        squared,
        np.square(evidence),
        out=np.zeros_like(evidence),
        where=evidence > 0.0,
    )
    parent_evidence = np.bincount(
        np.asarray(physical.child_parent_rows, dtype=np.int64),
        weights=evidence,
        minlength=physical.maplet_ids.size,
    ).astype(np.float64)
    child_parent_evidence = parent_evidence[physical.child_parent_rows]
    parent_fraction = np.divide(
        evidence,
        child_parent_evidence,
        out=np.zeros_like(evidence),
        where=child_parent_evidence > 0.0,
    )
    member_count = np.diff(
        np.asarray(physical.child_member_offsets, dtype=np.int64)
    ).astype(np.float64)
    extents = np.maximum(
        np.asarray(physical.child_extents, dtype=np.float64), 0.0
    )
    if extents.shape != (child_count, 3):
        raise ValueError("child extent ledger differs")
    extent_l1 = np.sum(extents, axis=1)
    extent_order = np.sort(extents, axis=1)
    extent_area = 4.0 * extent_order[:, -1] * extent_order[:, -2]
    density = evidence / area
    feature = np.column_stack(
        (
            np.log1p(evidence),
            np.log1p(density),
            np.log1p(token_count),
            maximum,
            inverse_effective,
            np.log(area),
            np.log1p(member_count),
            np.log1p(child_parent_evidence),
            parent_fraction,
            np.log1p(extent_l1),
            np.log1p(extent_area),
        )
    )
    return ChildRerankingFeatures(
        child_rows=child_rows,
        values=feature[child_rows],
        joint_evidence=evidence[child_rows],
    )
