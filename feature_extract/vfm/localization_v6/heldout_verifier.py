"""Maplet-disjoint acceptance for V6 pose updates."""

from __future__ import annotations

import hashlib

import numpy as np

from feature_extract.vfm.localization_v6.local_correlation import (
    CorrelationDistribution,
)


def split_fit_heldout_maplets(
    maplet_ids: np.ndarray, *, heldout_fraction: float = 0.25, seed: int = 61
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(np.asarray(maplet_ids, dtype=np.int64))
    if ids.size < 2:
        return ids, np.zeros((0,), dtype=np.int64)
    scored = []
    for value in ids.tolist():
        digest = hashlib.sha256(f"v6-maplet-split:{seed}:{value}".encode()).digest()
        scored.append((int.from_bytes(digest[:8], "little"), int(value)))
    ordered = np.asarray([value for _score, value in sorted(scored)], dtype=np.int64)
    count = min(
        max(int(round(ids.size * float(heldout_fraction))), 1), ids.size - 1
    )
    return ordered[count:], ordered[:count]


def zero_displacement_log_likelihood(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> float:
    rows = np.flatnonzero(
        np.isin(
            correlation.maplet_ids,
            np.asarray(maplet_ids, dtype=np.int64),
        )
    )
    if rows.size == 0:
        return float("-inf")
    zero = np.linalg.norm(correlation.offsets_xy, axis=1) <= float(zero_radius)
    # ``probabilities`` is already the unconditional displacement mass from
    # the joint softmax: it sums to ``1 - null_probability``.  Multiplying by
    # non-null mass again would square that term and reject otherwise useful
    # updates merely because the verifier contains difficult regions.
    likelihood = np.clip(
        np.sum(correlation.probabilities[rows][:, zero], axis=1), 1e-8, 1.0
    )
    return float(np.mean(np.log(likelihood)))


def _fixed_surface_log_likelihoods(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> dict[int, float]:
    rows = np.flatnonzero(
        np.isin(
            correlation.maplet_ids,
            np.asarray(maplet_ids, dtype=np.int64),
        )
    )
    if rows.size == 0:
        return {}
    zero = np.linalg.norm(correlation.offsets_xy, axis=1) <= float(zero_radius)
    mass = np.clip(
        np.sum(correlation.probabilities[rows][:, zero], axis=1), 1e-8, 1.0
    )
    if correlation.surface_ids is None:
        return {int(row): float(np.log(value)) for row, value in zip(rows, mass)}
    surface_ids = np.asarray(correlation.surface_ids[rows], dtype=np.int64)
    result = {}
    for surface_id in np.unique(surface_ids[surface_ids >= 0]).tolist():
        local = surface_ids == int(surface_id)
        # A texel can cover multiple raster pixels.  Aggregate it once so
        # projected area cannot change the held-out denominator.
        result[int(surface_id)] = float(np.mean(np.log(mass[local])))
    return result


def _paired_log_likelihood(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> tuple[float, float, int]:
    before_by_id = _fixed_surface_log_likelihoods(before, maplet_ids)
    after_by_id = _fixed_surface_log_likelihoods(after, maplet_ids)
    shared = sorted(set(before_by_id) & set(after_by_id))
    if not shared:
        return float("-inf"), float("-inf"), 0
    return (
        float(np.mean([before_by_id[value] for value in shared])),
        float(np.mean([after_by_id[value] for value in shared])),
        len(shared),
    )


def accept_pose_update(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    *,
    fit_maplet_ids: np.ndarray,
    heldout_maplet_ids: np.ndarray,
    minimum_fit_gain: float = 1e-3,
    minimum_heldout_gain: float = 0.0,
) -> tuple[bool, dict[str, float]]:
    fit_before, fit_after, fit_surface_count = _paired_log_likelihood(
        before, after, fit_maplet_ids
    )
    heldout_before, heldout_after, heldout_surface_count = _paired_log_likelihood(
        before, after, heldout_maplet_ids
    )
    accepted = bool(
        np.isfinite(fit_after)
        and np.isfinite(heldout_after)
        and fit_after >= fit_before + float(minimum_fit_gain)
        and heldout_after >= heldout_before + float(minimum_heldout_gain)
    )
    return accepted, {
        "fit_before": fit_before,
        "fit_after": fit_after,
        "heldout_before": heldout_before,
        "heldout_after": heldout_after,
        "fit_surface_count": float(fit_surface_count),
        "heldout_surface_count": float(heldout_surface_count),
    }
