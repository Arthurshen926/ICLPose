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
    probability = np.sum(correlation.probabilities[rows][:, zero], axis=1)
    # Explicit null remains possible, but it cannot improve the centered-match
    # evidence merely by rejecting difficult held-out regions.
    likelihood = np.clip(
        probability * (1.0 - correlation.null_probability[rows]), 1e-8, 1.0
    )
    return float(np.mean(np.log(likelihood)))


def accept_pose_update(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    *,
    fit_maplet_ids: np.ndarray,
    heldout_maplet_ids: np.ndarray,
    minimum_fit_gain: float = 1e-3,
    minimum_heldout_gain: float = 0.0,
) -> tuple[bool, dict[str, float]]:
    fit_before = zero_displacement_log_likelihood(before, fit_maplet_ids)
    fit_after = zero_displacement_log_likelihood(after, fit_maplet_ids)
    heldout_before = zero_displacement_log_likelihood(before, heldout_maplet_ids)
    heldout_after = zero_displacement_log_likelihood(after, heldout_maplet_ids)
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
    }
