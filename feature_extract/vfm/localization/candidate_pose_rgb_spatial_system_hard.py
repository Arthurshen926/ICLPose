"""Train-only mining of the current RGB likelihood's coherent-repeat errors.

Static geometric hard-repeat targets are useful only when they overlap the
model's actual failure mode.  This module first receives frozen *target-free*
scores for a query, ranks a fixed target-free top-H set of coherent-wrong pose
modes, then joins train-only registered identity and projection targets to
materialize positive/negative candidate edges.  The returned rows are training
targets; the runtime scorer must never import or consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT = (
    "candidate_pose_rgb_spatial_current_system_hard_mining_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT = (
    "candidate_pose_rgb_spatial_current_system_hard_mining_v2"
)
SUPPORTED_CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMATS = frozenset(
    {
        CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MINING_FORMAT,
        CANDIDATE_POSE_RGB_SPATIAL_SYSTEM_HARD_MULTI_MODE_MINING_FORMAT,
    }
)
CURRENT_SYSTEM_HARD_TARGET_FREE_RANKED_MODES_FORMAT = (
    "candidate_pose_rgb_spatial_target_free_ranked_wrong_modes_v1"
)


@dataclass(frozen=True)
class CandidatePoseRGBSpatialSystemHardSelection:
    """Distinct exact-track and current-wrong edges from one query/mode."""

    hardest_mode_index: int
    hardest_pair_id: int
    source_point_ids: np.ndarray
    positive_candidate_indices: np.ndarray
    negative_candidate_indices: np.ndarray
    positive_offsets_xy: np.ndarray
    negative_offsets_xy: np.ndarray
    negative_candidate_posteriors: np.ndarray
    wrong_pose_log_likelihood_ratio: float
    selected_mode_rank: int = 0

    def __post_init__(self) -> None:
        mode = int(self.hardest_mode_index)
        pair_id = int(self.hardest_pair_id)
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        positive = np.asarray(self.positive_candidate_indices, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_candidate_indices, dtype=np.int64).reshape(-1)
        positive_offsets = np.asarray(self.positive_offsets_xy, dtype=np.float32)
        negative_offsets = np.asarray(self.negative_offsets_xy, dtype=np.float32)
        posterior = np.asarray(self.negative_candidate_posteriors, dtype=np.float32).reshape(-1)
        score = float(self.wrong_pose_log_likelihood_ratio)
        mode_rank = int(self.selected_mode_rank)
        count = len(source_ids)
        if (
            mode < 0
            or pair_id < 0
            or mode_rank < 0
            or positive.shape != (count,)
            or negative.shape != (count,)
            or positive_offsets.shape != (count, 2)
            or negative_offsets.shape != (count, 2)
            or posterior.shape != (count,)
            or len(np.unique(source_ids)) != count
            or np.any(positive < 0)
            or np.any(negative < 0)
            or np.any(positive == negative)
            or not np.isfinite(positive_offsets).all()
            or not np.isfinite(negative_offsets).all()
            or not np.isfinite(posterior).all()
            or np.any(posterior <= 0.0)
            or not math.isfinite(score)
        ):
            raise ValueError("current system-hard RGB spatial selection is invalid")
        object.__setattr__(self, "hardest_mode_index", mode)
        object.__setattr__(self, "hardest_pair_id", pair_id)
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "positive_candidate_indices", positive)
        object.__setattr__(self, "negative_candidate_indices", negative)
        object.__setattr__(self, "positive_offsets_xy", positive_offsets)
        object.__setattr__(self, "negative_offsets_xy", negative_offsets)
        object.__setattr__(self, "negative_candidate_posteriors", posterior)
        object.__setattr__(self, "wrong_pose_log_likelihood_ratio", score)
        object.__setattr__(self, "selected_mode_rank", mode_rank)

    @property
    def count(self) -> int:
        return int(len(self.source_point_ids))


def _validate_inputs(
    *,
    source_point_ids: np.ndarray,
    pair_ids: np.ndarray,
    wrong_pose_log_likelihood_ratios: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    observed_candidate_mask: np.ndarray,
    correct_offsets_xy: np.ndarray,
    correct_valid: np.ndarray,
    correct_edge_usable: np.ndarray,
    wrong_offsets_xy: np.ndarray,
    wrong_valid: np.ndarray,
    wrong_candidate_log_likelihood_ratios: np.ndarray,
    wrong_edge_usable: np.ndarray,
) -> tuple[int, int, int, int]:
    source_ids = np.asarray(source_point_ids, dtype=np.int64).reshape(-1)
    pairs = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    wrong_pose = np.asarray(wrong_pose_log_likelihood_ratios, dtype=np.float32).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    priors = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    observed = np.asarray(observed_candidate_mask, dtype=bool)
    correct_offsets = np.asarray(correct_offsets_xy, dtype=np.float32)
    correct_is_valid = np.asarray(correct_valid, dtype=bool)
    correct_usable = np.asarray(correct_edge_usable, dtype=bool)
    wrong_offsets = np.asarray(wrong_offsets_xy, dtype=np.float32)
    wrong_is_valid = np.asarray(wrong_valid, dtype=bool)
    wrong_candidate = np.asarray(wrong_candidate_log_likelihood_ratios, dtype=np.float32)
    wrong_usable = np.asarray(wrong_edge_usable, dtype=bool)
    point_count = len(source_ids)
    if tracks.ndim != 2:
        raise ValueError("system-hard candidate tracks are invalid")
    candidate_count = int(tracks.shape[1])
    if correct_usable.ndim != 3:
        raise ValueError("system-hard correct edge usability is invalid")
    view_count = int(correct_usable.shape[2])
    mode_count = len(pairs)
    invalid = {
        "empty_points": point_count == 0,
        "candidate_count": candidate_count < 2,
        "view_count": view_count == 0,
        "mode_count": mode_count == 0,
        "duplicate_source_ids": len(np.unique(source_ids)) != point_count,
        "duplicate_pair_ids": len(np.unique(pairs)) != mode_count,
        "negative_pair_id": bool(np.any(pairs < 0)),
        "wrong_pose_shape": wrong_pose.shape != (mode_count,),
        "track_shape": tracks.shape != (point_count, candidate_count),
        "prior_shape": priors.shape != tracks.shape,
        "null_shape": null.shape != (point_count,),
        "observed_shape": observed.shape != tracks.shape,
        "multiple_observed_candidates": bool(np.any(observed.sum(axis=1) > 1)),
        "correct_offset_shape": correct_offsets.shape != (*tracks.shape, 2),
        "correct_valid_shape": correct_is_valid.shape != tracks.shape,
        "correct_usable_shape": correct_usable.shape != (*tracks.shape, view_count),
        "wrong_offset_shape": wrong_offsets.shape
        != (mode_count, point_count, candidate_count, 2),
        "wrong_valid_shape": wrong_is_valid.shape != wrong_offsets.shape[:-1],
        "wrong_candidate_shape": wrong_candidate.shape != wrong_offsets.shape[:-1],
        "wrong_usable_shape": wrong_usable.shape
        != (*wrong_offsets.shape[:-1], view_count),
        "nonfinite_wrong_pose": not bool(np.isfinite(wrong_pose).all()),
        "nonfinite_priors": not bool(np.isfinite(priors).all()),
        "nonfinite_null": not bool(np.isfinite(null).all()),
        "nonfinite_correct_offsets": not bool(np.isfinite(correct_offsets).all()),
        "nonfinite_wrong_offsets": not bool(np.isfinite(wrong_offsets).all()),
        "nonfinite_wrong_candidate": not bool(np.isfinite(wrong_candidate).all()),
        "negative_prior": bool(np.any(priors < 0.0)),
        "negative_null": bool(np.any(null < 0.0)),
        "mixture_not_normalized": bool(
            np.any(np.abs(priors.sum(axis=1) + null - 1.0) > 1e-4)
        ),
        "observed_missing_track": bool(np.any(observed & (tracks < 0))),
    }
    failed = [name for name, value in invalid.items() if value]
    if failed:
        raise ValueError(
            "system-hard RGB spatial mining inputs are invalid: " + ", ".join(failed)
        )
    return mode_count, point_count, candidate_count, view_count


def _candidate_posterior(
    *, candidate_probabilities: np.ndarray, null_probabilities: np.ndarray, candidate_llr: np.ndarray
) -> np.ndarray:
    """Compute fixed-mixture candidate responsibilities for one pose mode."""

    priors = np.asarray(candidate_probabilities, dtype=np.float64)
    null = np.asarray(null_probabilities, dtype=np.float64).reshape(-1)
    llr = np.asarray(candidate_llr, dtype=np.float64)
    if priors.shape != llr.shape or null.shape != (len(priors),):
        raise ValueError("system-hard candidate posterior inputs are invalid")
    candidate_log_mass = np.where(priors > 0.0, np.log(np.maximum(priors, 1e-300)) + llr, -np.inf)
    null_log_mass = np.where(null > 0.0, np.log(np.maximum(null, 1e-300)), -np.inf)
    normalizer = np.logaddexp(
        np.logaddexp.reduce(candidate_log_mass, axis=1), null_log_mass
    )
    posterior = np.exp(candidate_log_mass - normalizer[:, None])
    if not np.isfinite(posterior).all() or np.any(posterior < 0.0):
        raise ValueError("system-hard candidate posterior is non-finite")
    return posterior.astype(np.float32)


def rank_current_system_hard_wrong_modes(
    *,
    pair_ids: np.ndarray,
    wrong_pose_log_likelihood_ratios: np.ndarray,
    max_wrong_modes_per_query: int,
) -> np.ndarray:
    """Return the target-free top-H coherent-wrong mode order.

    Mode selection must be fixed before any exact-track, visibility, or
    projection supervision is read.  In particular, an ineligible high-score
    mode is *not* replaced with a lower-score one after the train-only join;
    doing so would leak labels into the current-system failure distribution.
    """

    pairs = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    wrong_pose = np.asarray(wrong_pose_log_likelihood_ratios, dtype=np.float32).reshape(-1)
    max_modes = int(max_wrong_modes_per_query)
    if (
        len(pairs) == 0
        or pairs.shape != wrong_pose.shape
        or len(np.unique(pairs)) != len(pairs)
        or np.any(pairs < 0)
        or not np.isfinite(wrong_pose).all()
        or max_modes <= 0
    ):
        raise ValueError("current system-hard wrong-mode ranking inputs are invalid")
    # np.lexsort uses its last key first: descending current-system pose score,
    # then stable pair ID for exact score ties.
    mode_order = np.lexsort((pairs, -wrong_pose))
    return np.asarray(mode_order[: min(max_modes, len(mode_order))], dtype=np.int64)


def select_current_system_hard_repeat_edges_for_modes(
    *,
    source_point_ids: np.ndarray,
    pair_ids: np.ndarray,
    wrong_pose_log_likelihood_ratios: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    observed_candidate_mask: np.ndarray,
    correct_offsets_xy: np.ndarray,
    correct_valid: np.ndarray,
    correct_edge_usable: np.ndarray,
    wrong_offsets_xy: np.ndarray,
    wrong_valid: np.ndarray,
    wrong_candidate_log_likelihood_ratios: np.ndarray,
    wrong_edge_usable: np.ndarray,
    positive_radius_px: float,
    negative_radius_px: float,
    minimum_negative_candidate_posterior: float = 0.0,
    max_wrong_modes_per_query: int = 1,
) -> tuple[CandidatePoseRGBSpatialSystemHardSelection | None, ...]:
    """Materialize exact-track hard edges for target-free top-H wrong modes.

    The tuple follows the frozen target-free score order and therefore can
    contain ``None`` for a mode that has no train-only eligible edge.  The
    caller must preserve that absence rather than backfilling lower-ranked
    modes.  Each nonempty selection retains one distinct wrong candidate per
    source point for one coherent pose-pair.
    """

    mode_count, point_count, candidate_count, _view_count = _validate_inputs(
        source_point_ids=source_point_ids,
        pair_ids=pair_ids,
        wrong_pose_log_likelihood_ratios=wrong_pose_log_likelihood_ratios,
        candidate_track_ids=candidate_track_ids,
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
        observed_candidate_mask=observed_candidate_mask,
        correct_offsets_xy=correct_offsets_xy,
        correct_valid=correct_valid,
        correct_edge_usable=correct_edge_usable,
        wrong_offsets_xy=wrong_offsets_xy,
        wrong_valid=wrong_valid,
        wrong_candidate_log_likelihood_ratios=wrong_candidate_log_likelihood_ratios,
        wrong_edge_usable=wrong_edge_usable,
    )
    positive_radius = float(positive_radius_px)
    negative_radius = float(negative_radius_px)
    posterior_floor = float(minimum_negative_candidate_posterior)
    if (
        not all(math.isfinite(value) for value in (positive_radius, negative_radius, posterior_floor))
        or positive_radius <= 0.0
        or negative_radius <= 0.0
        or posterior_floor < 0.0
        or posterior_floor >= 1.0
    ):
        raise ValueError("system-hard RGB spatial mining thresholds are invalid")
    source_ids = np.asarray(source_point_ids, dtype=np.int64).reshape(-1)
    pairs = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    wrong_pose = np.asarray(wrong_pose_log_likelihood_ratios, dtype=np.float32).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    priors = np.asarray(candidate_probabilities, dtype=np.float32)
    observed = np.asarray(observed_candidate_mask, dtype=bool)
    correct_offsets = np.asarray(correct_offsets_xy, dtype=np.float32)
    correct_is_valid = np.asarray(correct_valid, dtype=bool)
    correct_usable = np.asarray(correct_edge_usable, dtype=bool)
    wrong_offsets = np.asarray(wrong_offsets_xy, dtype=np.float32)
    wrong_is_valid = np.asarray(wrong_valid, dtype=bool)
    wrong_candidate = np.asarray(wrong_candidate_log_likelihood_ratios, dtype=np.float32)
    wrong_usable = np.asarray(wrong_edge_usable, dtype=bool)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    assert mode_count == len(wrong_pose) and point_count == len(source_ids)
    assert candidate_count == tracks.shape[1]

    ranked_modes = rank_current_system_hard_wrong_modes(
        pair_ids=pairs,
        wrong_pose_log_likelihood_ratios=wrong_pose,
        max_wrong_modes_per_query=max_wrong_modes_per_query,
    )
    correct_local = (
        correct_is_valid
        & (np.max(np.abs(correct_offsets), axis=2) <= positive_radius)
        & np.any(correct_usable, axis=2)
    )
    candidate_indices = np.arange(candidate_count, dtype=np.int64)
    output: list[CandidatePoseRGBSpatialSystemHardSelection | None] = []
    for mode_rank, mode_value in enumerate(ranked_modes.tolist()):
        mode = int(mode_value)
        candidate_posterior = _candidate_posterior(
            candidate_probabilities=priors,
            null_probabilities=null,
            candidate_llr=wrong_candidate[mode],
        )
        wrong_local = (
            wrong_is_valid[mode]
            & (np.max(np.abs(wrong_offsets[mode]), axis=2) <= negative_radius)
            & np.any(wrong_usable[mode], axis=2)
            & (priors > 0.0)
            & ~observed
        )
        selected_source: list[int] = []
        positive_indices: list[int] = []
        negative_indices: list[int] = []
        positive_offsets: list[np.ndarray] = []
        negative_offsets: list[np.ndarray] = []
        negative_posteriors: list[float] = []
        for point in range(point_count):
            positive_rows = np.flatnonzero(observed[point] & correct_local[point])
            if len(positive_rows) != 1:
                continue
            positive = int(positive_rows[0])
            negative_mask = wrong_local[point].copy()
            negative_mask &= tracks[point] != tracks[point, positive]
            negative_mask &= candidate_posterior[point] >= posterior_floor
            rows = np.flatnonzero(negative_mask)
            if not len(rows):
                continue
            # Max posterior is the explicit wrong-mode explanation. Frozen
            # prior and candidate index only break exact posterior ties.
            order = np.lexsort(
                (
                    candidate_indices[rows],
                    -priors[point, rows],
                    -candidate_posterior[point, rows],
                )
            )
            negative = int(rows[int(order[0])])
            if negative == positive or tracks[point, negative] == tracks[point, positive]:
                raise RuntimeError("system-hard mining selected a non-distinct candidate identity")
            selected_source.append(int(source_ids[point]))
            positive_indices.append(positive)
            negative_indices.append(negative)
            positive_offsets.append(correct_offsets[point, positive])
            negative_offsets.append(wrong_offsets[mode, point, negative])
            negative_posteriors.append(float(candidate_posterior[point, negative]))
        if not selected_source:
            output.append(None)
            continue
        output.append(
            CandidatePoseRGBSpatialSystemHardSelection(
                hardest_mode_index=mode,
                hardest_pair_id=int(pairs[mode]),
                source_point_ids=np.asarray(selected_source, dtype=np.int64),
                positive_candidate_indices=np.asarray(positive_indices, dtype=np.int64),
                negative_candidate_indices=np.asarray(negative_indices, dtype=np.int64),
                positive_offsets_xy=np.asarray(positive_offsets, dtype=np.float32),
                negative_offsets_xy=np.asarray(negative_offsets, dtype=np.float32),
                negative_candidate_posteriors=np.asarray(negative_posteriors, dtype=np.float32),
                wrong_pose_log_likelihood_ratio=float(wrong_pose[mode]),
                selected_mode_rank=int(mode_rank),
            )
        )
    return tuple(output)


def select_current_system_hard_repeat_edges(
    *,
    source_point_ids: np.ndarray,
    pair_ids: np.ndarray,
    wrong_pose_log_likelihood_ratios: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    observed_candidate_mask: np.ndarray,
    correct_offsets_xy: np.ndarray,
    correct_valid: np.ndarray,
    correct_edge_usable: np.ndarray,
    wrong_offsets_xy: np.ndarray,
    wrong_valid: np.ndarray,
    wrong_candidate_log_likelihood_ratios: np.ndarray,
    wrong_edge_usable: np.ndarray,
    positive_radius_px: float,
    negative_radius_px: float,
    minimum_negative_candidate_posterior: float = 0.0,
) -> CandidatePoseRGBSpatialSystemHardSelection | None:
    """Mine exact-track versus model-supported wrong candidate edges.

    The frozen scorer chooses the most competitive wrong pose mode *before*
    this function sees target fields.  This function then requires that the
    registered exact track is locally usable at the correct pose and a distinct
    dustbin-labelled track is locally usable under that selected wrong mode.
    The negative is ranked by its actual fixed-mixture posterior contribution,
    rather than merely by its projection distance or coarse rank.
    """

    selections = select_current_system_hard_repeat_edges_for_modes(
        source_point_ids=source_point_ids,
        pair_ids=pair_ids,
        wrong_pose_log_likelihood_ratios=wrong_pose_log_likelihood_ratios,
        candidate_track_ids=candidate_track_ids,
        candidate_probabilities=candidate_probabilities,
        null_probabilities=null_probabilities,
        observed_candidate_mask=observed_candidate_mask,
        correct_offsets_xy=correct_offsets_xy,
        correct_valid=correct_valid,
        correct_edge_usable=correct_edge_usable,
        wrong_offsets_xy=wrong_offsets_xy,
        wrong_valid=wrong_valid,
        wrong_candidate_log_likelihood_ratios=wrong_candidate_log_likelihood_ratios,
        wrong_edge_usable=wrong_edge_usable,
        positive_radius_px=positive_radius_px,
        negative_radius_px=negative_radius_px,
        minimum_negative_candidate_posterior=minimum_negative_candidate_posterior,
        max_wrong_modes_per_query=1,
    )
    assert len(selections) == 1
    return selections[0]
