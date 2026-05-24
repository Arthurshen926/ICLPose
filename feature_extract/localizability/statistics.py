"""Paired statistical summaries for POFD-FS reports."""

from __future__ import annotations

import math

import torch


def _flat_float(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    return values[torch.isfinite(values)]


def paired_bootstrap_mean_ci(
    values: torch.Tensor,
    *,
    num_bootstrap: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Query-level paired bootstrap confidence interval for a mean statistic."""

    values = _flat_float(values)
    if values.numel() == 0:
        raise ValueError("values must contain at least one finite element")
    if int(num_bootstrap) <= 0:
        raise ValueError("num_bootstrap must be positive")
    if not (0.0 < float(ci) < 1.0):
        raise ValueError("ci must be in (0,1)")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    idx = torch.randint(0, values.numel(), (int(num_bootstrap), values.numel()), generator=generator)
    boot = values.cpu()[idx].mean(dim=1)
    alpha = (1.0 - float(ci)) / 2.0
    low = torch.quantile(boot, alpha)
    high = torch.quantile(boot, 1.0 - alpha)
    return {
        "mean": float(values.mean().item()),
        "ci_low": float(low.item()),
        "ci_high": float(high.item()),
        "num_samples": int(values.numel()),
        "num_bootstrap": int(num_bootstrap),
        "ci": float(ci),
    }


def _binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    total = 0.0
    for idx in range(0, int(k) + 1):
        total += math.comb(int(n), idx) * (float(p) ** idx) * ((1.0 - float(p)) ** (int(n) - idx))
    return float(total)


def mcnemar_test(method_success: torch.Tensor, baseline_success: torch.Tensor) -> dict[str, float | int]:
    """Exact two-sided McNemar test for paired binary successes."""

    method = torch.as_tensor(method_success).bool().reshape(-1)
    baseline = torch.as_tensor(baseline_success).bool().reshape(-1)
    if method.shape != baseline.shape:
        raise ValueError("method_success and baseline_success must have the same shape")
    method_only = int((method & ~baseline).sum().item())
    baseline_only = int((baseline & ~method).sum().item())
    discordant = method_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(1.0, 2.0 * _binomial_cdf(min(method_only, baseline_only), discordant, p=0.5))
    return {
        "method_only": method_only,
        "baseline_only": baseline_only,
        "discordant": discordant,
        "p_value": float(p_value),
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def wilcoxon_signed_rank_test(method_values: torch.Tensor, baseline_values: torch.Tensor) -> dict[str, float | int]:
    """Two-sided Wilcoxon signed-rank normal approximation for paired values."""

    method = _flat_float(method_values)
    baseline = _flat_float(baseline_values)
    if method.shape != baseline.shape:
        raise ValueError("method_values and baseline_values must have the same finite shape")
    diff = method - baseline
    nonzero = diff != 0
    diff_nz = diff[nonzero]
    n = int(diff_nz.numel())
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "z": 0.0, "p_value": 1.0, "median_delta": 0.0}
    abs_diff = diff_nz.abs()
    order = torch.argsort(abs_diff)
    ranks = torch.empty_like(abs_diff, dtype=torch.float64)
    sorted_abs = abs_diff[order]
    start = 0
    while start < n:
        end = start + 1
        while end < n and bool(sorted_abs[end] == sorted_abs[start]):
            end += 1
        avg_rank = (float(start + 1) + float(end)) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    w_plus = float(ranks[diff_nz > 0].sum().item())
    w_minus = float(ranks[diff_nz < 0].sum().item())
    w_stat = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = (w_stat - mean) / math.sqrt(var)
        p_value = min(1.0, 2.0 * _normal_cdf(z))
    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": float(z),
        "p_value": float(p_value),
        "median_delta": float(diff.median().item()),
    }


__all__ = ["mcnemar_test", "paired_bootstrap_mean_ci", "wilcoxon_signed_rank_test"]
