"""Pose-free scene aggregation for frozen token-to-child evidence.

The original retrieval baseline compresses the RADIO grid into sixteen large
image blocks and retains only four maxima.  This module exposes that baseline
alongside local, 4x4-token aggregation controls without changing the token
posterior.  An optional physical-parent mask prevents children outside the
already-returned scene parent set from consuming a bounded scene-child budget.

There is deliberately no pose, image, renderer, or ground-truth interface.
"""

from __future__ import annotations

import numpy as np

from .fine_support_selection import (
    EVIDENCE_BLOCK_CAPPED_SUM,
    EVIDENCE_BLOCK_MAX_SUM,
    EVIDENCE_SEMANTICS as LOCAL_EVIDENCE_SEMANTICS,
    EVIDENCE_TOKEN_SUM,
    aggregate_child_evidence,
)
from .physical_map import GoalMapletPhysicalMap
from .pure_retrieval import (
    SCENE_AGGREGATION,
    aggregate_sparse_token_evidence,
    all_radio_token_coordinates,
)


EVIDENCE_LEGACY_MACRO_TOP4 = SCENE_AGGREGATION
SCENE_CHILD_EVIDENCE_SEMANTICS = (
    EVIDENCE_LEGACY_MACRO_TOP4,
    EVIDENCE_TOKEN_SUM,
    EVIDENCE_BLOCK_CAPPED_SUM,
    EVIDENCE_BLOCK_MAX_SUM,
)
SCENE_PARENT_MASK_SEMANTICS = "positive_scene_parent_ids_only_v1"


def aggregate_scene_child_evidence(
    token_xy: np.ndarray,
    token_child_rows: np.ndarray,
    token_child_probabilities: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    token_height: int,
    token_width: int,
    semantics: str,
    local_block_size: int = 4,
    scene_parent_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Aggregate child evidence and optionally mask it to scene parents.

    ``scene_parent_ids=None`` leaves the physical child union unrestricted.
    Passing IDs applies only a deterministic hierarchy mask; parent scores are
    not multiplied into child evidence a second time because the stored child
    probabilities are already joint parent-child probabilities.
    """

    xy = np.asarray(token_xy, dtype=np.int64)
    rows = np.asarray(token_child_rows, dtype=np.int64)
    probability = np.asarray(token_child_probabilities, dtype=np.float64)
    height, width = int(token_height), int(token_width)
    mode = str(semantics)
    expected_xy = all_radio_token_coordinates(height, width).astype(np.int64)
    child_count = int(physical.child_parent_rows.size)
    if (
        xy.shape != expected_xy.shape
        or not np.array_equal(xy, expected_xy)
        or rows.ndim != 2
        or rows.shape[0] != xy.shape[0]
        or probability.shape != rows.shape
        or np.any(~np.isfinite(probability))
        or np.any((probability < 0.0) | (probability > 1.0))
        or int(local_block_size) <= 0
        or mode not in SCENE_CHILD_EVIDENCE_SEMANTICS
    ):
        raise ValueError("invalid scene-child evidence input")

    if mode == EVIDENCE_LEGACY_MACRO_TOP4:
        unmasked = aggregate_sparse_token_evidence(
            xy,
            rows,
            probability,
            entity_count=child_count,
            token_height=height,
            token_width=width,
        )
    else:
        if mode not in LOCAL_EVIDENCE_SEMANTICS:
            raise ValueError("unknown local child evidence semantics")
        unmasked = aggregate_child_evidence(
            rows,
            probability,
            xy,
            child_count=child_count,
            semantics=mode,
            block_size=int(local_block_size),
        )
    unmasked = np.asarray(unmasked, dtype=np.float64).reshape(-1)
    if (
        unmasked.shape != (child_count,)
        or np.any(~np.isfinite(unmasked))
        or np.any(unmasked < 0.0)
    ):
        raise ValueError("invalid aggregated scene-child evidence")

    parent_mask_applied = scene_parent_ids is not None
    selected_parent_rows = np.arange(physical.maplet_ids.size, dtype=np.int64)
    score = unmasked.copy()
    if parent_mask_applied:
        parent_ids = np.asarray(scene_parent_ids, dtype=np.int64).reshape(-1)
        if np.unique(parent_ids).size != parent_ids.size:
            raise ValueError("scene parent IDs must be unique")
        parent_row_by_id = {
            int(parent_id): row
            for row, parent_id in enumerate(physical.maplet_ids.tolist())
        }
        selected_parent_rows = np.asarray(
            [parent_row_by_id.get(int(parent_id), -1) for parent_id in parent_ids],
            dtype=np.int64,
        )
        if np.any(selected_parent_rows < 0):
            raise ValueError("scene parent IDs differ from the physical hierarchy")
        allowed = np.zeros((physical.maplet_ids.size,), dtype=bool)
        allowed[selected_parent_rows] = True
        score[~allowed[np.asarray(physical.child_parent_rows, dtype=np.int64)]] = 0.0

    unmasked_mass = float(np.sum(unmasked))
    masked_mass = float(np.sum(score))
    audit: dict[str, object] = {
        "evidence_semantics": mode,
        "local_block_size": int(local_block_size),
        "scene_parent_mask_applied": bool(parent_mask_applied),
        "scene_parent_mask_semantics": (
            SCENE_PARENT_MASK_SEMANTICS if parent_mask_applied else "none"
        ),
        "scene_parent_count": int(selected_parent_rows.size),
        "positive_child_count_before_parent_mask": int(np.count_nonzero(unmasked > 0.0)),
        "positive_child_count_after_parent_mask": int(np.count_nonzero(score > 0.0)),
        "summed_child_evidence_before_parent_mask": unmasked_mass,
        "summed_child_evidence_after_parent_mask": masked_mass,
        "retained_child_evidence_fraction_after_parent_mask": float(
            masked_mass / max(unmasked_mass, 1e-12)
        ),
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "parent_evidence_remultiplied": False,
    }
    return score, audit
