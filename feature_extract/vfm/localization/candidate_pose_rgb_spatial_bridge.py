"""Low-capacity bridge for frozen RGB spatial and candidate-identity evidence.

The bridge intentionally contains no visual encoder and no geometry projector.
It only combines two target-free quantities emitted by frozen models after a
caller supplies a pose projection:

* per-candidate high-resolution RGB spatial log likelihood; and
* pose-independent per-candidate RADIO-final identity log likelihood.

Training code may arrange the first pose row as the correct pose and later
rows as coherent wrong poses.  That ordering is a train-only concern.  The
runtime math below receives only score tensors, immutable candidate/null mass,
and non-negative calibration weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_BRIDGE_FORMAT = "candidate_pose_rgb_spatial_bridge_v1"


@dataclass(frozen=True)
class CandidatePoseRGBSpatialBridgeWeights:
    """Non-negative source and prior weights for one frozen bridge profile."""

    spatial_weight: float
    identity_weight: float
    prior_exponent: float

    def __post_init__(self) -> None:
        values = (
            float(self.spatial_weight),
            float(self.identity_weight),
            float(self.prior_exponent),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("bridge weights must be finite and non-negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "spatial_weight": float(self.spatial_weight),
            "identity_weight": float(self.identity_weight),
            "prior_exponent": float(self.prior_exponent),
        }


@dataclass(frozen=True)
class CandidatePoseRGBSpatialBridgeQueryEvidence:
    """One train-only query's frozen correct-first pose score tensors.

    ``spatial_candidate_llrs`` has shape ``[1 + wrong_mode_count, point,
    candidate]``.  Its first row is designated by the train-only artifact as
    correct; the bridge never receives an explicit pose, residual, track ID,
    or target label.
    """

    query_id: str
    spatial_candidate_llrs: np.ndarray
    control_spatial_candidate_llrs: np.ndarray
    identity_candidate_llrs: np.ndarray
    control_identity_candidate_llrs: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray

    def __post_init__(self) -> None:
        query_id = str(self.query_id)
        spatial = np.asarray(self.spatial_candidate_llrs, dtype=np.float64)
        control_spatial = np.asarray(self.control_spatial_candidate_llrs, dtype=np.float64)
        identity = np.asarray(self.identity_candidate_llrs, dtype=np.float64)
        control_identity = np.asarray(self.control_identity_candidate_llrs, dtype=np.float64)
        candidates = np.asarray(self.candidate_probabilities, dtype=np.float64)
        null = np.asarray(self.null_probabilities, dtype=np.float64).reshape(-1)
        if (
            not query_id
            or spatial.ndim != 3
            or spatial.shape[0] < 2
            or spatial.shape[1] == 0
            or spatial.shape[2] < 2
            or control_spatial.shape != spatial.shape
            or identity.shape != spatial.shape[1:]
            or control_identity.shape != identity.shape
            or candidates.shape != identity.shape
            or null.shape != (spatial.shape[1],)
            or not np.isfinite(spatial).all()
            or not np.isfinite(control_spatial).all()
            or not np.isfinite(identity).all()
            or not np.isfinite(control_identity).all()
            or not np.isfinite(candidates).all()
            or not np.isfinite(null).all()
            or np.any(candidates < 0.0)
            or np.any(null < 0.0)
            or np.any(np.abs(candidates.sum(axis=1) + null - 1.0) > 1e-4)
        ):
            raise ValueError("bridge query evidence is invalid")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "spatial_candidate_llrs", spatial)
        object.__setattr__(self, "control_spatial_candidate_llrs", control_spatial)
        object.__setattr__(self, "identity_candidate_llrs", identity)
        object.__setattr__(self, "control_identity_candidate_llrs", control_identity)
        object.__setattr__(self, "candidate_probabilities", candidates)
        object.__setattr__(self, "null_probabilities", null)


def reweight_candidate_probabilities(
    *,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    prior_exponent: float,
) -> np.ndarray:
    """Temper fixed candidate mass without changing the explicit null mass.

    ``prior_exponent=1`` is the original coarse posterior.  ``0`` makes the
    non-null candidate mass uniform over its already-present top-L slots.  A
    zero-probability candidate is never resurrected.
    """

    candidates = np.asarray(candidate_probabilities, dtype=np.float64)
    null = np.asarray(null_probabilities, dtype=np.float64).reshape(-1)
    exponent = float(prior_exponent)
    if (
        candidates.ndim != 2
        or candidates.shape[0] == 0
        or candidates.shape[1] < 2
        or null.shape != (candidates.shape[0],)
        or not np.isfinite(candidates).all()
        or not np.isfinite(null).all()
        or not math.isfinite(exponent)
        or exponent < 0.0
        or np.any(candidates < 0.0)
        or np.any(null < 0.0)
        or np.any(np.abs(candidates.sum(axis=1) + null - 1.0) > 1e-4)
    ):
        raise ValueError("candidate prior reweighting inputs are invalid")
    nonnull = candidates.sum(axis=1, keepdims=True)
    active = nonnull[:, 0] > 0.0
    output = np.zeros_like(candidates)
    if not np.any(active):
        return output
    conditional = np.zeros_like(candidates)
    conditional[active] = candidates[active] / nonnull[active]
    positive = conditional > 0.0
    powered = np.zeros_like(conditional)
    # ``0 ** 0`` must remain absent rather than becoming a candidate slot.
    powered[positive] = np.exp(exponent * np.log(conditional[positive]))
    normalizer = powered.sum(axis=1, keepdims=True)
    if np.any(normalizer[active] <= 0.0) or not np.isfinite(normalizer).all():
        raise RuntimeError("candidate prior reweighting lost non-null mass")
    output[active] = nonnull[active] * powered[active] / normalizer[active]
    if np.any(np.abs(output.sum(axis=1) + null - 1.0) > 1e-4):
        raise RuntimeError("candidate prior reweighting changed explicit null mass")
    return output


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Numerically stable NumPy log-sum-exp with an explicit finite output."""

    raw = np.asarray(values, dtype=np.float64)
    maximum = np.max(raw, axis=axis, keepdims=True)
    if np.any(~np.isfinite(maximum)):
        raise ValueError("bridge mixture has no finite probability mass")
    summed = np.exp(raw - maximum).sum(axis=axis, keepdims=True)
    output = maximum + np.log(summed)
    return np.squeeze(output, axis=axis)


def bridge_pose_log_likelihood_ratios(
    *,
    spatial_candidate_llrs: np.ndarray,
    identity_candidate_llrs: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    weights: CandidatePoseRGBSpatialBridgeWeights,
) -> np.ndarray:
    """Return one immutable-candidate mixture score per supplied pose row."""

    spatial = np.asarray(spatial_candidate_llrs, dtype=np.float64)
    identity = np.asarray(identity_candidate_llrs, dtype=np.float64)
    candidates = reweight_candidate_probabilities(
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
        prior_exponent=float(weights.prior_exponent),
    )
    null = np.asarray(null_probabilities, dtype=np.float64).reshape(-1)
    if (
        spatial.ndim != 3
        or spatial.shape[0] == 0
        or spatial.shape[1:] != identity.shape
        or identity.shape != candidates.shape
        or null.shape != (spatial.shape[1],)
        or not np.isfinite(spatial).all()
        or not np.isfinite(identity).all()
    ):
        raise ValueError("bridge pose likelihood inputs are invalid")
    candidate_llrs = (
        float(weights.spatial_weight) * spatial
        + float(weights.identity_weight) * identity[None, :, :]
    )
    candidate_terms = np.where(
        candidates[None, :, :] > 0.0,
        np.log(np.maximum(candidates[None, :, :], np.finfo(np.float64).tiny))
        + candidate_llrs,
        -np.inf,
    )
    null_terms = np.broadcast_to(
        np.where(
            null[None, :, None] > 0.0,
            np.log(np.maximum(null[None, :, None], np.finfo(np.float64).tiny)),
            -np.inf,
        ),
        (spatial.shape[0], spatial.shape[1], 1),
    )
    point_llrs = _logsumexp(np.concatenate((candidate_terms, null_terms), axis=2), axis=2)
    return point_llrs.mean(axis=1)


def correct_minus_hardest_wrong(pose_log_likelihood_ratios: np.ndarray) -> float:
    """Return the train-only correct-first pose gap for one query."""

    values = np.asarray(pose_log_likelihood_ratios, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("correct-versus-wrong pose scores are invalid")
    return float(values[0] - np.max(values[1:]))


def bridge_query_gap(
    *,
    evidence: CandidatePoseRGBSpatialBridgeQueryEvidence,
    weights: CandidatePoseRGBSpatialBridgeWeights,
    control: bool,
) -> float:
    """Evaluate a fixed bridge profile on normal or deranged support evidence."""

    return correct_minus_hardest_wrong(
        bridge_pose_log_likelihood_ratios(
            spatial_candidate_llrs=(
                evidence.control_spatial_candidate_llrs
                if bool(control)
                else evidence.spatial_candidate_llrs
            ),
            identity_candidate_llrs=(
                evidence.control_identity_candidate_llrs
                if bool(control)
                else evidence.identity_candidate_llrs
            ),
            candidate_probabilities=evidence.candidate_probabilities,
            null_probabilities=evidence.null_probabilities,
            weights=weights,
        )
    )


def summarize_bridge_gaps(
    gaps: Sequence[float], *, catastrophic_threshold: float = -0.5
) -> dict[str, float]:
    """Summarize query-balanced correct-versus-hardest-wrong gaps."""

    values = np.asarray(tuple(gaps), dtype=np.float64).reshape(-1)
    threshold = float(catastrophic_threshold)
    if (
        len(values) == 0
        or not np.isfinite(values).all()
        or not math.isfinite(threshold)
    ):
        raise ValueError("bridge gap summary inputs are invalid")
    return {
        "query_count": float(len(values)),
        "mean_correct_minus_hardest_wrong": float(values.mean()),
        "median_correct_minus_hardest_wrong": float(np.median(values)),
        "p10_correct_minus_hardest_wrong": float(np.percentile(values, 10.0)),
        "correct_win_fraction": float(np.mean(values > 0.0)),
        "catastrophic_gap_count": float(np.sum(values <= threshold)),
        "catastrophic_gap_fraction": float(np.mean(values <= threshold)),
    }


def select_bridge_weights(
    *,
    evidences: Sequence[CandidatePoseRGBSpatialBridgeQueryEvidence],
    candidates: Iterable[CandidatePoseRGBSpatialBridgeWeights],
) -> tuple[CandidatePoseRGBSpatialBridgeWeights, dict[str, float]]:
    """Choose a profile from train-query gaps only, with deterministic ties."""

    records = tuple(evidences)
    options = tuple(candidates)
    if not records or not options:
        raise ValueError("bridge selection requires query evidence and candidate weights")
    if len({record.query_id for record in records}) != len(records):
        raise ValueError("bridge selection query IDs must be unique")
    selected: CandidatePoseRGBSpatialBridgeWeights | None = None
    selected_summary: dict[str, float] | None = None
    selected_key: tuple[float, ...] | None = None
    for weights in options:
        gaps = [bridge_query_gap(evidence=record, weights=weights, control=False) for record in records]
        summary = summarize_bridge_gaps(gaps)
        complexity = (
            abs(float(weights.spatial_weight) - 1.0)
            + float(weights.identity_weight)
            + abs(float(weights.prior_exponent) - 1.0)
        )
        key = (
            float(summary["correct_win_fraction"]),
            float(summary["mean_correct_minus_hardest_wrong"]),
            float(summary["median_correct_minus_hardest_wrong"]),
            -float(summary["catastrophic_gap_count"]),
            -complexity,
            -float(weights.spatial_weight),
            -float(weights.identity_weight),
            -float(weights.prior_exponent),
        )
        if selected_key is None or key > selected_key:
            selected = weights
            selected_summary = summary
            selected_key = key
    assert selected is not None and selected_summary is not None
    return selected, selected_summary


def deterministic_query_fold_map(
    *, query_ids: Sequence[str], fold_count: int
) -> dict[str, int]:
    """Match the existing sorted-query modulo inner-fold policy exactly."""

    values = tuple(sorted(set(str(query_id) for query_id in query_ids)))
    count = int(fold_count)
    if len(values) < 2 or count < 2 or count > len(values):
        raise ValueError("bridge cross-fit fold configuration is invalid")
    return {query_id: position % count for position, query_id in enumerate(values)}


def crossfit_bridge_profiles(
    *,
    evidences: Sequence[CandidatePoseRGBSpatialBridgeQueryEvidence],
    candidates: Iterable[CandidatePoseRGBSpatialBridgeWeights],
    fold_count: int,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    """Select source weights on disjoint query folds and score held queries."""

    records = tuple(evidences)
    options = tuple(candidates)
    if not records or len({record.query_id for record in records}) != len(records):
        raise ValueError("bridge cross-fit query evidence is invalid")
    folds = deterministic_query_fold_map(
        query_ids=[record.query_id for record in records], fold_count=int(fold_count)
    )
    rows: list[dict[str, object]] = []
    selections: dict[int, dict[str, object]] = {}
    for fold in range(int(fold_count)):
        fit = tuple(record for record in records if folds[record.query_id] != fold)
        held = tuple(record for record in records if folds[record.query_id] == fold)
        if not fit or not held:
            raise RuntimeError("bridge cross-fit produced an empty train or held fold")
        weights, fit_summary = select_bridge_weights(evidences=fit, candidates=options)
        selections[fold] = {
            "weights": weights.as_dict(),
            "fit_query_ids": [record.query_id for record in fit],
            "held_query_ids": [record.query_id for record in held],
            "fit_summary": fit_summary,
        }
        for record in held:
            rows.append(
                {
                    "query_id": record.query_id,
                    "fold": int(fold),
                    "weights": weights.as_dict(),
                    "normal_gap": bridge_query_gap(
                        evidence=record, weights=weights, control=False
                    ),
                    "control_gap": bridge_query_gap(
                        evidence=record, weights=weights, control=True
                    ),
                }
            )
    rows.sort(key=lambda row: str(row["query_id"]))
    if len(rows) != len(records) or {str(row["query_id"]) for row in rows} != {
        record.query_id for record in records
    }:
        raise RuntimeError("bridge cross-fit did not score every query exactly once")
    return rows, selections
