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
    chart_scores = zero_displacement_chart_log_likelihoods(
        correlation, maplet_ids, zero_radius=zero_radius
    )
    return (
        float(np.mean(list(chart_scores.values())))
        if chart_scores
        else float("-inf")
    )


def _uniform_zero_log_likelihood(
    correlation: CorrelationDistribution,
    *,
    zero_radius: float = 0.75,
) -> float:
    """Return the zero-flow mass of a uniform displacement-plus-null model."""

    offsets = np.asarray(correlation.offsets_xy, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.shape[0] == 0:
        raise ValueError("correlation offsets must have shape (M,2)")
    zero_count = int(
        np.sum(np.linalg.norm(offsets, axis=1) <= float(zero_radius))
    )
    if zero_count <= 0:
        raise ValueError("zero_radius contains no displacement mode")
    # The explicit null is one additional mutually exclusive categorical
    # outcome.  This baseline makes scores comparable across fixed search
    # windows without inventing a learned null probability.
    return float(np.log(zero_count / float(offsets.shape[0] + 1)))


def fixed_chart_zero_displacement_log_likelihood_ratio(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> float:
    """Score a fixed chart identity set against a uniform explicit null.

    Every requested chart contributes exactly once.  A chart that does not
    rasterize at a candidate pose contributes the uniform-null baseline (zero
    log-likelihood ratio) instead of disappearing from the denominator.  This
    is intended for comparing different pose candidates; paired update
    acceptance below uses an even stricter fixed-surface denominator.
    """

    requested = np.unique(np.asarray(maplet_ids, dtype=np.int64))
    if requested.size == 0:
        return float("-inf")
    baseline = _uniform_zero_log_likelihood(
        correlation, zero_radius=zero_radius
    )
    observed = zero_displacement_chart_log_likelihoods(
        correlation, requested, zero_radius=zero_radius
    )
    return float(
        np.mean(
            [
                float(observed.get(int(maplet_id), baseline)) - baseline
                for maplet_id in requested.tolist()
            ]
        )
    )


def _fixed_chart_row_score(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    row_score: np.ndarray,
) -> float:
    """Collapse raster pixels to surfaces, then give each chart one vote."""

    requested = np.unique(np.asarray(maplet_ids, dtype=np.int64))
    if requested.size == 0:
        return float("-inf")
    values = np.asarray(row_score, dtype=np.float64).reshape(-1)
    row_maplets = np.asarray(correlation.maplet_ids, dtype=np.int64).reshape(-1)
    if values.shape != row_maplets.shape:
        raise ValueError("row score and correlation rows differ")
    surface_ids = (
        None
        if correlation.surface_ids is None
        else np.asarray(correlation.surface_ids, dtype=np.int64).reshape(-1)
    )
    chart_scores: dict[int, float] = {}
    for maplet_id in requested.tolist():
        local = row_maplets == int(maplet_id)
        if not np.any(local):
            continue
        if surface_ids is None:
            chart_scores[int(maplet_id)] = float(np.mean(values[local]))
            continue
        local_surfaces = surface_ids[local]
        local_values = values[local]
        valid_surfaces = np.unique(local_surfaces[local_surfaces >= 0])
        if valid_surfaces.size == 0:
            chart_scores[int(maplet_id)] = float(np.mean(local_values))
            continue
        chart_scores[int(maplet_id)] = float(
            np.mean(
                [
                    np.mean(local_values[local_surfaces == int(surface_id)])
                    for surface_id in valid_surfaces.tolist()
                ]
            )
        )
    # A chart outside the rendered support is unavailable evidence, not a
    # negative match and not permission to shrink the candidate denominator.
    return float(
        np.mean(
            [
                chart_scores.get(int(maplet_id), 0.0)
                for maplet_id in requested.tolist()
            ]
        )
    )


def fixed_chart_local_match_log_bayes_factor(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> float:
    """Marginalize local displacement for coarse candidate pre-ranking.

    ``local_correlation_distribution`` is a joint posterior over a uniformly
    weighted displacement window and an explicit pair-null state.  Therefore
    ``(1-p_null)/p_null`` is exactly the Bayes factor that *some* displacement
    in that window matches.  It is the correct evidence before SE(3)
    alignment; using zero-flow mass here rejects valid but not-yet-converged
    pose basins.
    """

    non_null = np.clip(
        np.sum(np.asarray(correlation.probabilities, dtype=np.float64), axis=1),
        1e-12,
        1.0,
    )
    null = np.clip(
        np.asarray(correlation.null_probability, dtype=np.float64),
        1e-12,
        1.0,
    )
    return _fixed_chart_row_score(
        correlation, maplet_ids, np.log(non_null) - np.log(null)
    )


def fixed_chart_zero_displacement_log_bayes_factor(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> float:
    """Score exact rendered alignment against the explicit pair-null model.

    The displacement logits contain a uniform ``1 / valid_offset_count``
    search prior.  Once a pose is fixed, remove that search prior from the
    zero-flow/null odds.  This differs from posterior zero-flow probability:
    other attractive offsets must not change the feature likelihood at the
    rendered location.
    """

    row_score = _zero_displacement_row_log_bayes_factor(
        correlation, zero_radius=zero_radius
    )
    return _fixed_chart_row_score(correlation, maplet_ids, row_score)


def _zero_displacement_row_log_bayes_factor(
    correlation: CorrelationDistribution,
    *,
    zero_radius: float = 0.75,
) -> np.ndarray:
    """Return exact zero-flow evidence versus pair-null for every row."""

    offsets = np.asarray(correlation.offsets_xy, dtype=np.float64)
    zero = np.linalg.norm(offsets, axis=1) <= float(zero_radius)
    zero_count = int(np.sum(zero))
    if zero_count <= 0:
        raise ValueError("zero_radius contains no displacement mode")
    probability = np.asarray(correlation.probabilities, dtype=np.float64)
    zero_mass = np.clip(np.sum(probability[:, zero], axis=1), 1e-12, 1.0)
    null = np.clip(
        np.asarray(correlation.null_probability, dtype=np.float64),
        1e-12,
        1.0,
    )
    if correlation.valid_offset_count is None:
        # Invalid padded offsets underflow to exact zero in the float32 joint
        # posterior.  This fallback keeps hand-built test distributions and
        # legacy diagnostic payloads well-defined.
        valid_count = np.sum(probability > 0.0, axis=1)
        valid_count = np.maximum(valid_count, zero_count)
    else:
        valid_count = np.asarray(
            correlation.valid_offset_count, dtype=np.float64
        ).reshape(-1)
    return (
        np.log(zero_mass)
        - np.log(null)
        + np.log(np.maximum(valid_count, zero_count) / float(zero_count))
    )


def zero_displacement_chart_log_bayes_factors(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> dict[int, float]:
    """Return one exact pair-null evidence value per rendered chart."""

    rows = np.flatnonzero(
        np.isin(
            correlation.maplet_ids,
            np.asarray(maplet_ids, dtype=np.int64),
        )
    )
    if rows.size == 0:
        return {}
    row_score = _zero_displacement_row_log_bayes_factor(
        correlation, zero_radius=zero_radius
    )[rows]
    row_maplets = np.asarray(correlation.maplet_ids[rows], dtype=np.int64)
    surface_ids = (
        None
        if correlation.surface_ids is None
        else np.asarray(correlation.surface_ids[rows], dtype=np.int64)
    )
    result = {}
    for maplet_id in np.unique(row_maplets).tolist():
        local = row_maplets == int(maplet_id)
        if surface_ids is None:
            result[int(maplet_id)] = float(np.mean(row_score[local]))
            continue
        local_surfaces = surface_ids[local]
        local_values = row_score[local]
        valid_surfaces = np.unique(local_surfaces[local_surfaces >= 0])
        if valid_surfaces.size == 0:
            result[int(maplet_id)] = float(np.mean(local_values))
            continue
        result[int(maplet_id)] = float(
            np.mean(
                [
                    np.mean(local_values[local_surfaces == int(surface_id)])
                    for surface_id in valid_surfaces.tolist()
                ]
            )
        )
    return result


def zero_displacement_log_bayes_factor(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> float:
    """Average exact pair-null evidence over equally weighted charts."""

    chart_scores = zero_displacement_chart_log_bayes_factors(
        correlation, maplet_ids, zero_radius=zero_radius
    )
    return (
        float(np.mean(list(chart_scores.values())))
        if chart_scores
        else float("-inf")
    )


def zero_displacement_chart_log_likelihoods(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> dict[int, float]:
    """Return one area-invariant zero-flow likelihood per chart."""

    rows = np.flatnonzero(
        np.isin(
            correlation.maplet_ids,
            np.asarray(maplet_ids, dtype=np.int64),
        )
    )
    if rows.size == 0:
        return {}
    zero = np.linalg.norm(correlation.offsets_xy, axis=1) <= float(zero_radius)
    # ``probabilities`` is already the unconditional displacement mass from
    # the joint softmax: it sums to ``1 - null_probability``.  Multiplying by
    # non-null mass again would square that term and reject otherwise useful
    # updates merely because the verifier contains difficult regions.
    likelihood = np.clip(
        np.sum(correlation.probabilities[rows][:, zero], axis=1), 1e-8, 1.0
    )
    log_likelihood = np.log(likelihood)
    row_maplets = np.asarray(correlation.maplet_ids[rows], dtype=np.int64)
    surface_ids = (
        None
        if correlation.surface_ids is None
        else np.asarray(correlation.surface_ids[rows], dtype=np.int64)
    )
    # Raster area is pose-dependent and must not become evidence.  Collapse
    # repeated pixels of one atlas texel first, then give each rendered chart
    # equal mass.  Otherwise a nearby/large repetitive facade can outrank the
    # correct pose merely because it covers more feature-grid pixels.
    chart_scores = {}
    for maplet_id in np.unique(row_maplets).tolist():
        local = row_maplets == int(maplet_id)
        if surface_ids is None:
            chart_scores[int(maplet_id)] = float(
                np.mean(log_likelihood[local])
            )
            continue
        local_surfaces = surface_ids[local]
        local_values = log_likelihood[local]
        valid_surfaces = np.unique(local_surfaces[local_surfaces >= 0])
        if valid_surfaces.size == 0:
            chart_scores[int(maplet_id)] = float(np.mean(local_values))
            continue
        surface_scores = [
            float(np.mean(local_values[local_surfaces == int(surface_id)]))
            for surface_id in valid_surfaces.tolist()
        ]
        chart_scores[int(maplet_id)] = float(np.mean(surface_scores))
    return chart_scores


def _fixed_surface_log_likelihoods(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> dict[tuple[int, int], float]:
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
    row_maplets = np.asarray(correlation.maplet_ids[rows], dtype=np.int64)
    if correlation.surface_ids is None:
        return {
            (int(maplet_id), int(row)): float(np.log(value))
            for row, maplet_id, value in zip(rows, row_maplets, mass)
        }
    surface_ids = np.asarray(correlation.surface_ids[rows], dtype=np.int64)
    result = {}
    for maplet_id in np.unique(row_maplets).tolist():
        maplet_local = row_maplets == int(maplet_id)
        for surface_id in np.unique(
            surface_ids[maplet_local & (surface_ids >= 0)]
        ).tolist():
            local = maplet_local & (surface_ids == int(surface_id))
            # A texel can cover multiple raster pixels. Aggregate it once so
            # projected area cannot change the held-out denominator.
            result[(int(maplet_id), int(surface_id))] = float(
                np.mean(np.log(mass[local]))
            )
    return result


def _fixed_surface_log_bayes_factors(
    correlation: CorrelationDistribution,
    maplet_ids: np.ndarray,
    *,
    zero_radius: float = 0.75,
) -> dict[tuple[int, int], float]:
    """Return exact pair-null evidence on pose-stable surface identities."""

    rows = np.flatnonzero(
        np.isin(
            correlation.maplet_ids,
            np.asarray(maplet_ids, dtype=np.int64),
        )
    )
    if rows.size == 0:
        return {}
    values = _zero_displacement_row_log_bayes_factor(
        correlation, zero_radius=zero_radius
    )[rows]
    row_maplets = np.asarray(correlation.maplet_ids[rows], dtype=np.int64)
    if correlation.surface_ids is None:
        return {
            (int(maplet_id), int(row)): float(value)
            for row, maplet_id, value in zip(rows, row_maplets, values)
        }
    surface_ids = np.asarray(correlation.surface_ids[rows], dtype=np.int64)
    result = {}
    for maplet_id in np.unique(row_maplets).tolist():
        maplet_local = row_maplets == int(maplet_id)
        for surface_id in np.unique(
            surface_ids[maplet_local & (surface_ids >= 0)]
        ).tolist():
            local = maplet_local & (surface_ids == int(surface_id))
            result[(int(maplet_id), int(surface_id))] = float(
                np.mean(values[local])
            )
    return result


def _paired_log_bayes_factor(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> tuple[float, float, int]:
    """Compare exact evidence on one fixed set of chart identities.

    Canonical atlas texels are stable map identities, but the *raster sample*
    carrying a texel identity is not stable under a pose update.  A small
    camera motion changes triangle coverage, z-buffer winners and the
    deterministic point subsample.  Pairing only the texels rasterized at the
    seed therefore assigns a large artificial loss to every real motion.

    Candidate ranking already solves the correct invariance problem: keep the
    chart denominator fixed, average raster evidence inside each chart, and
    give an absent chart the explicit pair-null baseline (zero log Bayes
    factor).  Update acceptance must use that same objective independently at
    the two poses.  Fit and held-out chart identities remain fixed and
    disjoint, so the held-out set still only accepts or rejects a fit-selected
    proposal.
    """

    requested = np.unique(np.asarray(maplet_ids, dtype=np.int64))
    if requested.size == 0:
        return float("-inf"), float("-inf"), 0
    before_score = fixed_chart_zero_displacement_log_bayes_factor(
        before, requested
    )
    after_score = fixed_chart_zero_displacement_log_bayes_factor(
        after, requested
    )
    rows = np.isin(
        np.asarray(before.maplet_ids, dtype=np.int64), requested
    )
    if before.surface_ids is None:
        evidence_count = int(np.sum(rows))
    else:
        surfaces = np.asarray(before.surface_ids, dtype=np.int64)[rows]
        evidence_count = int(np.unique(surfaces[surfaces >= 0]).size)
    return (
        float(before_score),
        float(after_score),
        int(evidence_count),
    )


def paired_chart_log_bayes_factor_gains(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> dict[int, float]:
    """Return fixed-chart exact-evidence gain per chart identity.

    The historical function name is retained for report compatibility.  The
    compared denominator is the requested chart identity, not the
    pose-dependent set of rasterized texels.
    """

    requested = np.unique(np.asarray(maplet_ids, dtype=np.int64))
    return {
        int(maplet_id): float(
            fixed_chart_zero_displacement_log_bayes_factor(
                after, np.asarray([maplet_id], dtype=np.int64)
            )
            - fixed_chart_zero_displacement_log_bayes_factor(
                before, np.asarray([maplet_id], dtype=np.int64)
            )
        )
        for maplet_id in requested.tolist()
    }


def _paired_log_likelihood(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> tuple[float, float, int]:
    before_by_id = _fixed_surface_log_likelihoods(before, maplet_ids)
    after_by_id = _fixed_surface_log_likelihoods(after, maplet_ids)
    fixed = sorted(before_by_id)
    if not fixed:
        return float("-inf"), float("-inf"), 0
    after_null = _uniform_zero_log_likelihood(after)
    # Freeze the surfaces observed before the proposed update.  Newly entering
    # texels cannot manufacture a gain, and a difficult texel cannot improve
    # merely by leaving the image/occlusion buffer.  Missing-after evidence is
    # therefore no better than its before value and is penalized down to the
    # uniform explicit-null baseline when the before evidence was stronger.
    after_fixed = {
        key: float(
            after_by_id.get(
                key, min(float(before_by_id[key]), float(after_null))
            )
        )
        for key in fixed
    }
    # The number of verified texels differs strongly across charts. Average
    # fixed surfaces inside each chart first, then give every chart one vote.
    fixed_by_maplet: dict[int, list[tuple[int, int]]] = {}
    for key in fixed:
        fixed_by_maplet.setdefault(int(key[0]), []).append(key)
    return (
        float(
            np.mean(
                [
                    np.mean([before_by_id[key] for key in keys])
                    for keys in fixed_by_maplet.values()
                ]
            )
        ),
        float(
            np.mean(
                [
                    np.mean([after_fixed[key] for key in keys])
                    for keys in fixed_by_maplet.values()
                ]
            )
        ),
        len(fixed),
    )


def paired_chart_log_likelihood_gains(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    maplet_ids: np.ndarray,
) -> dict[int, float]:
    """Return fixed-surface update gain independently for each chart.

    This is the auditable chart-consensus counterpart of the aggregate
    verifier.  Only atlas surface cells rendered both before and after are
    compared, so entering/leaving raster area cannot manufacture a gain.
    """

    before_by_id = _fixed_surface_log_likelihoods(before, maplet_ids)
    after_by_id = _fixed_surface_log_likelihoods(after, maplet_ids)
    fixed = sorted(before_by_id)
    after_null = _uniform_zero_log_likelihood(after)
    by_maplet: dict[int, list[tuple[int, int]]] = {}
    for key in fixed:
        by_maplet.setdefault(int(key[0]), []).append(key)
    return {
        int(maplet_id): float(
            np.mean(
                [
                    float(
                        after_by_id.get(
                            key,
                            min(
                                float(before_by_id[key]),
                                float(after_null),
                            ),
                        )
                    )
                    - before_by_id[key]
                    for key in keys
                ]
            )
        )
        for maplet_id, keys in by_maplet.items()
    }


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


def accept_pose_update_bayes_factor(
    before: CorrelationDistribution,
    after: CorrelationDistribution,
    *,
    fit_maplet_ids: np.ndarray,
    heldout_maplet_ids: np.ndarray,
    minimum_fit_gain: float = 1e-3,
    minimum_heldout_gain: float = 0.0,
) -> tuple[bool, dict[str, float]]:
    """Accept an update using the same fixed-chart evidence as final ranking.

    The legacy posterior-mass verifier above remains available for historical
    diagnostics.  Continuous atlas alignment must instead compare zero-flow
    feature density to the explicit pair-null state: posterior mass is also
    divided by every competing displacement in the search window and is not
    the objective used to rank final poses.  Rasterized texels are deliberately
    re-sampled at each pose; only the disjoint chart identities stay fixed.
    """

    fit_before, fit_after, fit_surface_count = _paired_log_bayes_factor(
        before, after, fit_maplet_ids
    )
    heldout_before, heldout_after, heldout_surface_count = (
        _paired_log_bayes_factor(before, after, heldout_maplet_ids)
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
