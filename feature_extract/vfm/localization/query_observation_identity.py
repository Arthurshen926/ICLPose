"""Exact query-track targets from registered real-image SfM observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - retained for minimal runtime installs.
    cKDTree = None

from feature_extract.vfm.colmap_tracks import ColmapImageObservation


@dataclass(frozen=True)
class RegisteredQueryObservationTargets:
    track_ids: np.ndarray
    distances_px: np.ndarray
    supervised: np.ndarray

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        distances = np.asarray(self.distances_px, dtype=np.float32).reshape(-1)
        supervised = np.asarray(self.supervised, dtype=bool).reshape(-1)
        if track_ids.shape != distances.shape or distances.shape != supervised.shape:
            raise ValueError("registered query target arrays must have equal length")
        if np.any(np.isnan(distances)) or np.any(distances < 0.0):
            raise ValueError("registered query distances must be non-negative or infinity")
        if np.any(supervised & ((track_ids < 0) | ~np.isfinite(distances))):
            raise ValueError("supervised query targets require a finite track observation")
        if np.any(~supervised & (track_ids >= 0)):
            raise ValueError("unsupervised query targets must use track id -1")
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "distances_px", distances)
        object.__setattr__(self, "supervised", supervised)


def registered_query_observation_targets(
    *,
    query_ids: Sequence[str] | np.ndarray,
    query_xy: np.ndarray,
    images_by_name: Mapping[str, ColmapImageObservation],
    max_distance_px: float,
    query_chunk_size: int = 256,
) -> RegisteredQueryObservationTargets:
    """Associate detector anchors with nearby registered SfM observations.

    Query pose is not used. The target comes from COLMAP's explicit point2D to
    point3D identity and is intended exclusively as train/eval supervision.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if len(ids) != len(xy):
        raise ValueError("query ids and coordinates must have equal length")
    if not np.all(np.isfinite(xy)):
        raise ValueError("query coordinates must be finite")
    if float(max_distance_px) <= 0.0 or int(query_chunk_size) <= 0:
        raise ValueError("identity radius and chunk size must be positive")
    output_tracks = np.full((len(ids),), -1, dtype=np.int64)
    output_distances = np.full((len(ids),), np.inf, dtype=np.float32)
    for query_id in tuple(dict.fromkeys(ids.tolist())):
        image = images_by_name.get(str(query_id))
        if image is None:
            raise KeyError(f"query image missing from COLMAP model: {query_id}")
        observation_xy = np.asarray(image.xys, dtype=np.float32).reshape(-1, 2)
        observation_tracks = np.asarray(image.point3d_ids, dtype=np.int64).reshape(-1)
        if len(observation_xy) != len(observation_tracks):
            raise ValueError(f"COLMAP observation arrays differ for {query_id}")
        valid = (observation_tracks >= 0) & np.all(np.isfinite(observation_xy), axis=1)
        if not np.any(valid):
            continue
        valid_xy = observation_xy[valid]
        valid_tracks = observation_tracks[valid]
        rows = np.flatnonzero(ids == str(query_id))
        if cKDTree is not None:
            # The previous broadcasted distance matrix scaled as
            # ``chunk_size * observations_per_image`` and made all-image
            # identity supervision needlessly slow.  The KD-tree is exactly
            # the same nearest-observation definition, with no pose or target
            # information beyond the registered mapping observations.
            tree = cKDTree(valid_xy.astype(np.float64, copy=False))
            nearest_distances, nearest = tree.query(
                xy[rows].astype(np.float64, copy=False),
                k=1,
                distance_upper_bound=float(max_distance_px),
            )
            nearest_distances = np.asarray(nearest_distances, dtype=np.float32)
            nearest = np.asarray(nearest, dtype=np.int64)
            accepted = np.isfinite(nearest_distances) & (nearest < len(valid_tracks))
            accepted_rows = rows[accepted]
            output_tracks[accepted_rows] = valid_tracks[nearest[accepted]]
            output_distances[accepted_rows] = nearest_distances[accepted]
        else:
            for start in range(0, len(rows), int(query_chunk_size)):
                chunk_rows = rows[start : start + int(query_chunk_size)]
                distances2 = np.sum(
                    (xy[chunk_rows, None, :] - valid_xy[None, :, :]) ** 2, axis=2
                )
                nearest = np.argmin(distances2, axis=1)
                nearest_distances = np.sqrt(
                    distances2[np.arange(len(chunk_rows)), nearest]
                ).astype(np.float32)
                accepted = nearest_distances <= float(max_distance_px)
                accepted_rows = chunk_rows[accepted]
                output_tracks[accepted_rows] = valid_tracks[nearest[accepted]]
                output_distances[accepted_rows] = nearest_distances[accepted]
    supervised = output_tracks >= 0
    return RegisteredQueryObservationTargets(
        track_ids=output_tracks,
        distances_px=output_distances,
        supervised=supervised,
    )


def registered_candidate_identity_labels(
    candidate_track_ids: np.ndarray,
    targets: RegisteredQueryObservationTargets,
) -> np.ndarray:
    candidates = np.asarray(candidate_track_ids, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[0] != len(targets.track_ids):
        raise ValueError("candidate tracks must have shape (N, L) aligned with targets")
    return (
        targets.supervised[:, None]
        & (candidates == targets.track_ids[:, None])
        & (candidates >= 0)
    )


def registered_candidate_identity_target_membership(
    candidate_track_ids: np.ndarray,
    targets: RegisteredQueryObservationTargets,
) -> np.ndarray:
    """Build exact-track-or-null targets without inventing labels for unknown anchors.

    Rows with a nearby registered SfM observation receive exactly one target:
    the matching top-L track when it is present, otherwise the explicit null
    class.  Detector anchors with no nearby registered observation intentionally
    remain targetless so callers can exclude them from supervised fitting rather
    than treating absence of annotation as an outlier label.
    """

    candidates = np.asarray(candidate_track_ids, dtype=np.int64)
    if candidates.ndim != 2 or candidates.shape[0] != len(targets.track_ids):
        raise ValueError("candidate tracks must have shape (N, L) aligned with targets")
    labels = registered_candidate_identity_labels(candidates, targets)
    positive_count = np.sum(labels, axis=1)
    if np.any(positive_count > 1):
        raise ValueError(
            "exact registered-track supervision requires unique candidate tracks per row"
        )
    membership = np.zeros((len(candidates), candidates.shape[1] + 1), dtype=bool)
    membership[:, :-1] = labels
    membership[targets.supervised & (positive_count == 0), -1] = True
    supervised_count = np.sum(membership[targets.supervised], axis=1)
    if np.any(supervised_count != 1):
        raise RuntimeError("registered identity rows must select one candidate or explicit null")
    if np.any(membership[~targets.supervised]):
        raise RuntimeError("unsupervised anchors must not receive an identity target")
    return membership


def summarize_registered_candidate_identity(
    labels: np.ndarray,
    targets: RegisteredQueryObservationTargets,
) -> dict[str, float | int | None]:
    values = np.asarray(labels, dtype=bool)
    if values.ndim != 2 or values.shape[0] != len(targets.track_ids):
        raise ValueError("identity labels and registered targets are incompatible")
    supervised = targets.supervised
    supervised_count = int(np.sum(supervised))
    row_hit = np.any(values, axis=1)
    ranks = np.argmax(values, axis=1) + 1
    retrieved_ranks = ranks[supervised & row_hit]
    return {
        "row_count": int(len(values)),
        "supervised_row_count": supervised_count,
        "supervised_row_rate": float(np.mean(supervised)),
        "candidate_edge_positive_rate": float(np.mean(values)),
        "candidate_recall_given_supervised": (
            None if supervised_count == 0 else float(np.mean(row_hit[supervised]))
        ),
        "rank1_recall_given_supervised": (
            None
            if supervised_count == 0
            else float(np.mean(values[supervised, 0]))
        ),
        "median_positive_rank_when_retrieved": (
            None if len(retrieved_ranks) == 0 else float(np.median(retrieved_ranks))
        ),
    }
