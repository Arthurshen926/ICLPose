"""Statistical helpers for VFM-MapLoc reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def paired_bootstrap_delta_ci(
    method_values: Iterable[float],
    baseline_values: Iterable[float],
    resamples: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return mean(method - baseline) and a paired bootstrap CI."""

    method = np.asarray(list(method_values), dtype=np.float64)
    baseline = np.asarray(list(baseline_values), dtype=np.float64)
    if method.shape != baseline.shape or method.ndim != 1:
        raise ValueError("method_values and baseline_values must be 1D arrays of equal length")
    if method.size == 0 or resamples <= 0:
        raise ValueError("non-empty values and positive resamples are required")
    delta = method - baseline
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        idx = rng.integers(0, delta.size, size=delta.size)
        samples.append(float(np.mean(delta[idx])))
    lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(np.mean(delta)), float(lo), float(hi)


def mcnemar_exact_pvalue(
    baseline_success: Iterable[bool],
    method_success: Iterable[bool],
) -> float:
    """Two-sided exact McNemar p-value from paired binary outcomes."""

    baseline = np.asarray(list(baseline_success), dtype=bool)
    method = np.asarray(list(method_success), dtype=bool)
    if baseline.shape != method.shape or baseline.ndim != 1:
        raise ValueError("paired success arrays must be 1D and equal length")
    b = int(np.sum(baseline & ~method))
    c = int(np.sum(~baseline & method))
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


@dataclass(frozen=True)
class WilcoxonSignedRankResult:
    signed_rank: float
    positive_rank: float
    negative_rank: float
    nonzero_count: int
    normal_approx_z: float


def _rank_abs(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    idx = 0
    while idx < values.size:
        end = idx + 1
        while end < values.size and values[order[end]] == values[order[idx]]:
            end += 1
        avg_rank = 0.5 * (idx + 1 + end)
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def wilcoxon_signed_rank(
    method_values: Iterable[float],
    baseline_values: Iterable[float],
) -> WilcoxonSignedRankResult:
    """Signed-rank statistic for paired continuous values."""

    method = np.asarray(list(method_values), dtype=np.float64)
    baseline = np.asarray(list(baseline_values), dtype=np.float64)
    if method.shape != baseline.shape or method.ndim != 1:
        raise ValueError("paired values must be 1D and equal length")
    delta = method - baseline
    delta = delta[delta != 0.0]
    if delta.size == 0:
        return WilcoxonSignedRankResult(0.0, 0.0, 0.0, 0, 0.0)
    ranks = _rank_abs(np.abs(delta))
    positive = float(np.sum(ranks[delta > 0.0]))
    negative = float(np.sum(ranks[delta < 0.0]))
    signed = positive - negative
    n = int(delta.size)
    variance = n * (n + 1) * (2 * n + 1) / 6.0
    z = signed / math.sqrt(variance)
    return WilcoxonSignedRankResult(
        signed_rank=signed,
        positive_rank=positive,
        negative_rank=negative,
        nonzero_count=n,
        normal_approx_z=float(z),
    )
