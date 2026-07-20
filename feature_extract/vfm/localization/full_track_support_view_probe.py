"""Target-free all-observation support-view expansion for frozen candidates.

The production maplet uses a small, geometry-coverage-ranked subset of a
track's support images.  This module intentionally does something narrower:
given an already frozen candidate matrix, enumerate *every* real SfM
observation for every positive-mass candidate.  It does not retrieve images,
replace candidates, use a pose, or assign labels.  The resulting sparse edge
layout is used only by an absolute-appearance separability probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)


FULL_TRACK_SUPPORT_VIEW_SUMMARY_FORMAT = (
    "frozen_fulltrack_candidate_appearance_summary_v1"
)
FULL_TRACK_VIEW_STATISTIC_NAMES = (
    "uniform_mean_ncc",
    "uniform_logmeanexp_tau0p05_ncc",
    "uniform_top4_mean_ncc",
    "uniform_max_ncc",
)
_SOFT_BEST_TEMPERATURE = 0.05
_TOP_VIEW_COUNT = 4


@dataclass(frozen=True)
class TrackObservationLookup:
    """Track-major indirection into an image-major SfM observation index."""

    track_ids: np.ndarray
    offsets: np.ndarray
    geometry_rows: np.ndarray

    def __post_init__(self) -> None:
        tracks = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(self.offsets, dtype=np.int64).reshape(-1)
        rows = np.asarray(self.geometry_rows, dtype=np.int64).reshape(-1)
        if (
            len(tracks) == 0
            or len(np.unique(tracks)) != len(tracks)
            or np.any(tracks[1:] <= tracks[:-1])
            or offsets.shape != (len(tracks) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(rows)
            or np.any(offsets[1:] <= offsets[:-1])
            or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0)
        ):
            raise ValueError("track observation lookup is invalid")
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "geometry_rows", rows)

    @property
    def observation_counts(self) -> np.ndarray:
        return np.diff(self.offsets).astype(np.int64, copy=False)


@dataclass(frozen=True)
class FullTrackCandidateEdges:
    """Sparse all-observation candidate edges for one frozen query artifact."""

    candidate_shape: tuple[int, int]
    edge_candidate_indices: np.ndarray
    geometry_rows: np.ndarray
    candidate_observation_counts: np.ndarray

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.candidate_shape)
        edge_candidates = np.asarray(self.edge_candidate_indices, dtype=np.int64).reshape(-1)
        geometry_rows = np.asarray(self.geometry_rows, dtype=np.int64).reshape(-1)
        counts = np.asarray(self.candidate_observation_counts, dtype=np.int64)
        if (
            len(shape) != 2
            or min(shape) <= 0
            or edge_candidates.shape != geometry_rows.shape
            or counts.shape != shape
            or np.any(counts < 0)
            or np.any((edge_candidates < 0) | (edge_candidates >= shape[0] * shape[1]))
            or np.any(geometry_rows < 0)
            or np.any(edge_candidates[1:] < edge_candidates[:-1])
        ):
            raise ValueError("full-track candidate edge layout is invalid")
        observed = np.bincount(
            edge_candidates, minlength=shape[0] * shape[1]
        ).reshape(shape)
        if not np.array_equal(observed, counts):
            raise ValueError("full-track edge counts do not match candidate counts")
        object.__setattr__(self, "candidate_shape", shape)
        object.__setattr__(self, "edge_candidate_indices", edge_candidates)
        object.__setattr__(self, "geometry_rows", geometry_rows)
        object.__setattr__(self, "candidate_observation_counts", counts)

    @property
    def edge_count(self) -> int:
        return int(len(self.edge_candidate_indices))

    @property
    def point_indices(self) -> np.ndarray:
        return self.edge_candidate_indices // int(self.candidate_shape[1])

    @property
    def candidate_indices(self) -> np.ndarray:
        return self.edge_candidate_indices % int(self.candidate_shape[1])


@dataclass(frozen=True)
class FullTrackViewSummary:
    """Per-candidate raw statistics while retaining missing-evidence masks."""

    statistics: dict[str, np.ndarray]
    usable_counts: np.ndarray
    usable_fractions: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.statistics) != FULL_TRACK_VIEW_STATISTIC_NAMES:
            raise ValueError("full-track summary statistics do not match the fixed schema")
        arrays = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in self.statistics.items()
        }
        counts = np.asarray(self.usable_counts, dtype=np.int64)
        fractions = np.asarray(self.usable_fractions, dtype=np.float32)
        if not arrays:
            raise ValueError("full-track summary has no statistics")
        reference_shape = next(iter(arrays.values())).shape
        if (
            len(reference_shape) != 3
            or reference_shape[0] <= 0
            or reference_shape[1] <= 0
            or reference_shape[2] <= 0
            or any(values.shape != reference_shape for values in arrays.values())
            or counts.shape != reference_shape
            or fractions.shape != reference_shape
            or np.any(counts < 0)
            or np.any((fractions < 0.0) | (fractions > 1.0))
            or any(np.any(~np.isfinite(values[counts > 0])) for values in arrays.values())
            or any(np.any(np.isfinite(values[counts == 0])) for values in arrays.values())
        ):
            raise ValueError("full-track summary arrays are invalid")
        object.__setattr__(self, "statistics", arrays)
        object.__setattr__(self, "usable_counts", counts)
        object.__setattr__(self, "usable_fractions", fractions)


def build_track_observation_lookup(
    geometry: SupportObservationGeometryIndex,
) -> TrackObservationLookup:
    """Return a deterministic track-major view of every real observation."""

    track_ids = np.asarray(geometry.track_ids, dtype=np.int64)
    if len(track_ids) == 0:
        raise ValueError("support geometry has no observations")
    order = np.argsort(track_ids, kind="stable")
    sorted_tracks = track_ids[order]
    unique_tracks, starts, counts = np.unique(
        sorted_tracks, return_index=True, return_counts=True
    )
    offsets = np.concatenate(
        (
            np.zeros((1,), dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        )
    )
    if np.any(counts <= 0):
        raise RuntimeError("track-major observation lookup has an empty track")
    return TrackObservationLookup(
        track_ids=unique_tracks,
        offsets=offsets,
        geometry_rows=order,
    )


def build_full_track_candidate_edges(
    *,
    candidate_track_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    lookup: TrackObservationLookup,
) -> FullTrackCandidateEdges:
    """Expand every positive-mass frozen candidate to all its observations.

    Candidate rows with zero posterior mass are deliberately left without
    edges.  No support image is selected, capped, reweighted, or inferred from
    query content.
    """

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    if (
        tracks.ndim != 2
        or probabilities.shape != tracks.shape
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise ValueError("frozen candidate tracks or probabilities are invalid")
    shape = tuple(int(value) for value in tracks.shape)
    active_flat = np.flatnonzero((probabilities > 0.0).reshape(-1))
    active_tracks = tracks.reshape(-1)[active_flat]
    if np.any(active_tracks < 0):
        raise ValueError("a positive-mass frozen candidate has no physical track id")
    positions = np.searchsorted(lookup.track_ids, active_tracks)
    safe_positions = np.minimum(positions, len(lookup.track_ids) - 1)
    if np.any(positions >= len(lookup.track_ids)) or not np.array_equal(
        lookup.track_ids[safe_positions], active_tracks
    ):
        raise ValueError("a positive-mass frozen candidate is absent from support geometry")
    per_candidate = np.zeros((shape[0] * shape[1],), dtype=np.int64)
    per_candidate[active_flat] = lookup.observation_counts[positions]
    if np.any(per_candidate[active_flat] <= 0):
        raise RuntimeError("a positive-mass candidate has no real support observation")
    edge_candidates = np.repeat(active_flat, per_candidate[active_flat])
    parts = [
        lookup.geometry_rows[int(lookup.offsets[position]) : int(lookup.offsets[position + 1])]
        for position in positions.tolist()
    ]
    geometry_rows = (
        np.concatenate(parts).astype(np.int64, copy=False)
        if parts
        else np.zeros((0,), dtype=np.int64)
    )
    return FullTrackCandidateEdges(
        candidate_shape=shape,
        edge_candidate_indices=edge_candidates,
        geometry_rows=geometry_rows,
        candidate_observation_counts=per_candidate.reshape(shape),
    )


def aggregate_full_track_view_scores(
    *,
    edges: FullTrackCandidateEdges,
    scores: np.ndarray,
    usable: np.ndarray,
) -> FullTrackViewSummary:
    """Summarize raw per-view NCC without treating missing evidence as low score.

    ``uniform_logmeanexp`` is a normalized soft-best diagnostic.  It has no
    count bonus: adding identical views leaves the score unchanged.  The other
    statistics are intentionally exported only for fixed raw separability
    auditing; none is a calibrated production likelihood.
    """

    values = np.asarray(scores, dtype=np.float32)
    available = np.asarray(usable, dtype=bool)
    if (
        values.ndim != 2
        or values.shape != available.shape
        or values.shape[0] != edges.edge_count
        or values.shape[1] <= 0
        or np.any(~np.isfinite(values[available]))
    ):
        raise ValueError("all-observation view scores are invalid")
    point_count, candidate_count = edges.candidate_shape
    feature_count = int(values.shape[1])
    shape = (point_count, candidate_count, feature_count)
    outputs = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in FULL_TRACK_VIEW_STATISTIC_NAMES
    }
    usable_counts = np.zeros(shape, dtype=np.int64)
    usable_fractions = np.zeros(shape, dtype=np.float32)
    edge_candidates = np.asarray(edges.edge_candidate_indices, dtype=np.int64)
    starts = np.r_[
        0,
        np.flatnonzero(edge_candidates[1:] != edge_candidates[:-1]) + 1,
        len(edge_candidates),
    ]
    for begin, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
        candidate_flat = int(edge_candidates[begin])
        point = candidate_flat // candidate_count
        candidate = candidate_flat % candidate_count
        candidate_values = values[begin:end]
        candidate_usable = available[begin:end]
        for feature in range(feature_count):
            selected = candidate_values[:, feature][candidate_usable[:, feature]]
            count = int(len(selected))
            usable_counts[point, candidate, feature] = count
            usable_fractions[point, candidate, feature] = float(count) / float(end - begin)
            if count == 0:
                continue
            maximum = float(np.max(selected))
            outputs["uniform_mean_ncc"][point, candidate, feature] = float(
                np.mean(selected, dtype=np.float64)
            )
            outputs["uniform_max_ncc"][point, candidate, feature] = maximum
            top_count = min(_TOP_VIEW_COUNT, count)
            outputs["uniform_top4_mean_ncc"][point, candidate, feature] = float(
                np.mean(np.sort(selected)[-top_count:])
            )
            centered = (selected.astype(np.float64) - maximum) / _SOFT_BEST_TEMPERATURE
            outputs["uniform_logmeanexp_tau0p05_ncc"][point, candidate, feature] = float(
                maximum
                + _SOFT_BEST_TEMPERATURE
                * (np.log(np.mean(np.exp(centered))) if count else -np.inf)
            )
    return FullTrackViewSummary(
        statistics=outputs,
        usable_counts=usable_counts,
        usable_fractions=usable_fractions,
    )
