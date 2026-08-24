"""Pose-free spatial consistency for RADIO token-to-physical-child posteriors.

Pure appearance retrieval already places most contributor-visible children in
each token's candidate list, but its probability mass is spatially noisy.  A
candidate pose backend should not need GT geometry to repair that noise.  This
module therefore reweights only the frozen candidate rows using neighboring
tokens and the query-independent physical parent/connected-support graph.

No child is inserted, no token mass is created, and no pose is consumed.  The
method is a deterministic diagnostic control rather than a calibrated
posterior.
"""

from __future__ import annotations

import numpy as np


STRUCTURED_TOKEN_CHILD_POSTERIOR_SEMANTICS = (
    "pose_free_local_parent_and_connected_support_consistency_reweight_v1"
)


def structured_token_child_probabilities(
    child_rows: np.ndarray,
    child_probabilities: np.ndarray,
    child_parent_rows: np.ndarray,
    child_support_rows: np.ndarray,
    *,
    height: int = 36,
    width: int = 64,
    local_radius_tokens: int = 1,
    connected_support_weight: float = 2.0,
    parent_weight: float = 0.5,
) -> np.ndarray:
    """Reweight a frozen sparse child posterior without changing its support.

    The local kernel is divided by its full infinite-grid size and is not
    renormalized at image boundaries.  The final softmax preserves each
    token's original retained probability mass exactly (up to float error).
    """

    rows = np.asarray(child_rows, dtype=np.int64)
    probability = np.asarray(child_probabilities, dtype=np.float64)
    parent_by_child = np.asarray(child_parent_rows, dtype=np.int64).reshape(-1)
    support_by_child = np.asarray(child_support_rows, dtype=np.int64).reshape(-1)
    token_count = int(height) * int(width)
    radius = int(local_radius_tokens)
    support_weight = float(connected_support_weight)
    parent_coefficient = float(parent_weight)
    if (
        rows.ndim != 2 or rows.shape != probability.shape
        or rows.shape[0] != token_count
        or parent_by_child.shape != support_by_child.shape
        or parent_by_child.size == 0
        or radius < 0 or radius > 4
        or not np.isfinite(support_weight) or support_weight < 0.0
        or not np.isfinite(parent_coefficient) or parent_coefficient < 0.0
        or np.any(~np.isfinite(probability)) or np.any(probability < 0.0)
        or np.any(np.sum(probability, axis=1) > 1.0 + 2.0e-5)
    ):
        raise ValueError("invalid structured token-child posterior input")
    valid = probability > 0.0
    if np.any(valid & ((rows < 0) | (rows >= parent_by_child.size))):
        raise ValueError("positive child probability has an invalid row")
    safe = np.maximum(rows, 0)
    parent = parent_by_child[safe]
    support = support_by_child[safe]
    if np.any(valid & ((parent < 0) | (support < 0))):
        raise ValueError("positive child probability lacks physical hierarchy")
    # A repeated physical child in one token would receive duplicated mass and
    # make the reweighting depend on slot representation rather than evidence.
    for token_rows, token_valid in zip(rows, valid):
        positive_rows = token_rows[token_valid]
        if np.unique(positive_rows).size != positive_rows.size:
            raise ValueError("token child candidate rows must be unique")

    slots = int(rows.shape[1])
    probability_grid = probability.reshape(int(height), int(width), slots)
    valid_grid = valid.reshape(int(height), int(width), slots)
    parent_grid = parent.reshape(int(height), int(width), slots)
    support_grid = support.reshape(int(height), int(width), slots)
    support_agreement = np.zeros_like(probability_grid)
    parent_agreement = np.zeros_like(probability_grid)
    kernel = 1.0 / float((2 * radius + 1) ** 2)

    def matching_neighbor_mass(
        current_entity: np.ndarray,
        current_valid: np.ndarray,
        neighbor_entity: np.ndarray,
        neighbor_probability: np.ndarray,
        neighbor_valid: np.ndarray,
        entity_count: int,
    ) -> np.ndarray:
        """Sparse segment-sum lookup, avoiding a quadratic slot product."""

        aligned_tokens = int(np.prod(current_entity.shape[:2]))
        token_index = np.broadcast_to(
            np.arange(aligned_tokens, dtype=np.int64).reshape(
                current_entity.shape[0], current_entity.shape[1], 1,
            ),
            current_entity.shape,
        )
        neighbor_key = (
            token_index[neighbor_valid] * int(entity_count)
            + neighbor_entity[neighbor_valid]
        )
        neighbor_value = neighbor_probability[neighbor_valid]
        if neighbor_key.size == 0:
            return np.zeros_like(current_entity, dtype=np.float64)
        order = np.argsort(neighbor_key, kind="stable")
        ordered_key = neighbor_key[order]
        starts = np.r_[
            0, np.flatnonzero(ordered_key[1:] != ordered_key[:-1]) + 1,
        ]
        unique_key = ordered_key[starts]
        unique_mass = np.add.reduceat(neighbor_value[order], starts)
        current_key = token_index * int(entity_count) + current_entity
        location = np.searchsorted(unique_key, current_key)
        present = current_valid & (location < unique_key.size)
        present_rows = np.flatnonzero(present.reshape(-1))
        flat_location = location.reshape(-1)
        flat_key = current_key.reshape(-1)
        present.reshape(-1)[present_rows] = (
            unique_key[flat_location[present_rows]] == flat_key[present_rows]
        )
        result = np.zeros(current_entity.shape, dtype=np.float64)
        result[present] = unique_mass[location[present]]
        return result

    for shift_y in range(-radius, radius + 1):
        for shift_x in range(-radius, radius + 1):
            sy0, sy1 = max(0, -shift_y), min(int(height), int(height) - shift_y)
            sx0, sx1 = max(0, -shift_x), min(int(width), int(width) - shift_x)
            if sy1 <= sy0 or sx1 <= sx0:
                continue
            ny0, ny1 = sy0 + shift_y, sy1 + shift_y
            nx0, nx1 = sx0 + shift_x, sx1 + shift_x
            neighbor_probability = probability_grid[ny0:ny1, nx0:nx1]
            neighbor_valid = valid_grid[ny0:ny1, nx0:nx1]
            current_valid = valid_grid[sy0:sy1, sx0:sx1]
            support_agreement[sy0:sy1, sx0:sx1] += (
                matching_neighbor_mass(
                    support_grid[sy0:sy1, sx0:sx1], current_valid,
                    support_grid[ny0:ny1, nx0:nx1], neighbor_probability,
                    neighbor_valid, int(np.max(support_by_child)) + 1,
                ) * kernel
            )
            parent_agreement[sy0:sy1, sx0:sx1] += (
                matching_neighbor_mass(
                    parent_grid[sy0:sy1, sx0:sx1], current_valid,
                    parent_grid[ny0:ny1, nx0:nx1], neighbor_probability,
                    neighbor_valid, int(np.max(parent_by_child)) + 1,
                ) * kernel
            )

    total = np.sum(probability_grid, axis=2, keepdims=True)
    logit = np.full_like(probability_grid, -np.inf)
    logit[valid_grid] = (
        np.log(probability_grid[valid_grid])
        + support_weight * support_agreement[valid_grid]
        + parent_coefficient * parent_agreement[valid_grid]
    )
    maximum = np.max(logit, axis=2, keepdims=True, initial=-np.inf)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    unnormalized = np.where(
        valid_grid, np.exp(logit - finite_maximum), 0.0,
    )
    denominator = np.sum(unnormalized, axis=2, keepdims=True)
    result = np.divide(
        unnormalized * total,
        denominator,
        out=np.zeros_like(unnormalized),
        where=denominator > 0.0,
    )
    if (
        np.any(~np.isfinite(result)) or np.any(result < 0.0)
        or not np.allclose(np.sum(result, axis=2), total[..., 0], atol=2.0e-7, rtol=2.0e-7)
        or np.any((result > 0.0) != valid_grid)
    ):
        raise AssertionError("structured token-child reweighting broke mass/support conservation")
    return result.reshape(rows.shape).astype(np.float32)
