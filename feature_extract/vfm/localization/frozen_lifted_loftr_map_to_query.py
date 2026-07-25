"""Target-free lifted LoFTR map-to-query evidence for fixed pose hypotheses.

This is deliberately a verifier, not a PnP front end.  It starts from a fixed
top-L maplet union, lifts cached real-image LoFTR support matches onto known
SfM tracks, retains multiple cross-view-consistent query modes per track, and
then evaluates a *given* pose by reprojection.  No target pose, candidate
reselection, or pose-dependent correspondence selection is permitted here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree


def _as_points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
    if np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must be finite point coordinates")
    return points


@dataclass(frozen=True)
class LiftedSupportAnchorModes:
    """Top query-side LoFTR modes attached to immutable support anchors."""

    anchor_indices: np.ndarray
    query_xy: np.ndarray
    confidence: np.ndarray
    support_distance_px: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_indices, dtype=np.int64).reshape(-1)
        query_xy = _as_points(self.query_xy, name="lifted query modes")
        confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        distances = np.asarray(self.support_distance_px, dtype=np.float32).reshape(-1)
        if (
            query_xy.shape[0] != len(anchors)
            or confidence.shape != anchors.shape
            or distances.shape != anchors.shape
            or np.any(anchors < 0)
            or np.any(~np.isfinite(confidence))
            or np.any(~np.isfinite(distances))
            or np.any((confidence < 0.0) | (confidence > 1.0))
            or np.any(distances < 0.0)
        ):
            raise ValueError("lifted support-anchor modes are invalid")
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "support_distance_px", distances)


def select_loftr_modes_near_support_anchors(
    *,
    support_anchor_xy: np.ndarray,
    matched_query_xy: np.ndarray,
    matched_support_xy: np.ndarray,
    match_confidence: np.ndarray,
    support_snap_radius_px: float,
    max_modes_per_anchor: int,
    query_mode_nms_radius_px: float,
    nearest_match_count: int = 16,
) -> LiftedSupportAnchorModes:
    """Attach support anchors to fixed cached LoFTR matches without query center.

    The query endpoint is chosen only from a match close to the *support* SfM
    observation.  This avoids the old query-center shortcut and permits modes
    anywhere in the held-out query image.
    """

    anchors = _as_points(support_anchor_xy, name="support anchors")
    query = _as_points(matched_query_xy, name="matched query coordinates")
    support = _as_points(matched_support_xy, name="matched support coordinates")
    confidence = np.asarray(match_confidence, dtype=np.float32).reshape(-1)
    if (
        query.shape != support.shape
        or confidence.shape != (len(query),)
        or not np.isfinite(float(support_snap_radius_px))
        or float(support_snap_radius_px) <= 0.0
        or int(max_modes_per_anchor) <= 0
        or int(nearest_match_count) <= 0
        or not np.isfinite(float(query_mode_nms_radius_px))
        or float(query_mode_nms_radius_px) < 0.0
        or np.any((confidence < 0.0) | (confidence > 1.0))
    ):
        raise ValueError("LoFTR support-anchor mode inputs are invalid")
    if len(anchors) == 0 or len(query) == 0:
        return LiftedSupportAnchorModes(
            anchor_indices=np.zeros((0,), dtype=np.int64),
            query_xy=np.zeros((0, 2), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            support_distance_px=np.zeros((0,), dtype=np.float32),
        )
    neighbor_count = min(int(nearest_match_count), len(query))
    distances, indices = cKDTree(support).query(
        anchors,
        k=neighbor_count,
        distance_upper_bound=float(support_snap_radius_px),
    )
    if neighbor_count == 1:
        distances = np.asarray(distances, dtype=np.float32)[:, None]
        indices = np.asarray(indices, dtype=np.int64)[:, None]
    selected_anchor: list[int] = []
    selected_query: list[np.ndarray] = []
    selected_confidence: list[float] = []
    selected_distance: list[float] = []
    for anchor_index in range(len(anchors)):
        local_distance = np.asarray(distances[anchor_index], dtype=np.float32).reshape(-1)
        local_indices = np.asarray(indices[anchor_index], dtype=np.int64).reshape(-1)
        valid = (
            np.isfinite(local_distance)
            & (local_distance <= float(support_snap_radius_px))
            & (local_indices >= 0)
            & (local_indices < len(query))
        )
        if not np.any(valid):
            continue
        candidate_distance = local_distance[valid]
        candidate_indices = local_indices[valid]
        candidate_confidence = confidence[candidate_indices]
        # The static ranking prefers a confident close support match.  It is
        # independent of query centers, candidate identity labels, and pose.
        quality = np.log(np.maximum(candidate_confidence, 1e-8)) - 0.5 * (
            candidate_distance / float(support_snap_radius_px)
        ) ** 2
        order = np.lexsort(
            (
                candidate_indices,
                query[candidate_indices, 1],
                query[candidate_indices, 0],
                -candidate_confidence,
                candidate_distance,
                -quality,
            )
        )
        accepted_query: list[np.ndarray] = []
        for position in order.tolist():
            match_index = int(candidate_indices[position])
            point = query[match_index]
            if accepted_query and np.any(
                np.linalg.norm(np.stack(accepted_query, axis=0) - point[None], axis=1)
                <= float(query_mode_nms_radius_px)
            ):
                continue
            accepted_query.append(point)
            selected_anchor.append(int(anchor_index))
            selected_query.append(point.copy())
            selected_confidence.append(float(confidence[match_index]))
            selected_distance.append(float(candidate_distance[position]))
            if len(accepted_query) >= int(max_modes_per_anchor):
                break
    return LiftedSupportAnchorModes(
        anchor_indices=np.asarray(selected_anchor, dtype=np.int64),
        query_xy=np.asarray(selected_query, dtype=np.float32).reshape(-1, 2),
        confidence=np.asarray(selected_confidence, dtype=np.float32),
        support_distance_px=np.asarray(selected_distance, dtype=np.float32),
    )


@dataclass(frozen=True)
class FrozenLiftedTrackModeBank:
    """Fixed per-view, multi-mode query observations for physical SfM tracks.

    A support view endpoint is retained as an individual likelihood component.
    Cross-view clustering is used solely to establish a fixed consensus mode
    and its reliability; it must never replace the endpoints by a centroid
    before pose scoring.
    """

    track_ids: np.ndarray
    xyz: np.ndarray
    mode_offsets: np.ndarray
    mode_query_xy: np.ndarray
    mode_weights: np.ndarray
    mode_support_image_ids: np.ndarray
    mode_support_view_counts: np.ndarray
    mode_confidence_sums: np.ndarray
    track_reliabilities: np.ndarray
    track_reference_xy: np.ndarray

    def __post_init__(self) -> None:
        tracks = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        offsets = np.asarray(self.mode_offsets, dtype=np.int64).reshape(-1)
        mode_xy = _as_points(self.mode_query_xy, name="track mode query coordinates")
        weights = np.asarray(self.mode_weights, dtype=np.float32).reshape(-1)
        support_image_ids = np.asarray(self.mode_support_image_ids).astype(str).reshape(-1)
        view_counts = np.asarray(self.mode_support_view_counts, dtype=np.int64).reshape(-1)
        confidence_sums = np.asarray(self.mode_confidence_sums, dtype=np.float32).reshape(-1)
        reliability = np.asarray(self.track_reliabilities, dtype=np.float32).reshape(-1)
        reference_xy = _as_points(self.track_reference_xy, name="track reference coordinates")
        if (
            len(tracks) == 0
            or len(np.unique(tracks)) != len(tracks)
            or np.any(tracks < 0)
            or xyz.shape != (len(tracks), 3)
            or offsets.shape != (len(tracks) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(mode_xy)
            or np.any(np.diff(offsets) <= 0)
            or weights.shape != (len(mode_xy),)
            or support_image_ids.shape != (len(mode_xy),)
            or view_counts.shape != (len(mode_xy),)
            or confidence_sums.shape != (len(mode_xy),)
            or reliability.shape != (len(tracks),)
            or reference_xy.shape != (len(tracks), 2)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(weights))
            or np.any(~np.isfinite(confidence_sums))
            or np.any(view_counts <= 0)
            or np.any(confidence_sums < 0.0)
            or np.any(weights <= 0.0)
            or np.any(support_image_ids == "")
            or np.any((reliability <= 0.0) | (reliability >= 1.0))
        ):
            raise ValueError("frozen lifted LoFTR track-mode bank is invalid")
        for track_index in range(len(tracks)):
            start, stop = int(offsets[track_index]), int(offsets[track_index + 1])
            if abs(float(weights[start:stop].sum()) - 1.0) > 2e-5:
                raise ValueError("track-mode weights must normalize independently per track")
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "mode_offsets", offsets)
        object.__setattr__(self, "mode_query_xy", mode_xy)
        object.__setattr__(self, "mode_weights", weights)
        object.__setattr__(self, "mode_support_image_ids", support_image_ids)
        object.__setattr__(self, "mode_support_view_counts", view_counts)
        object.__setattr__(self, "mode_confidence_sums", confidence_sums)
        object.__setattr__(self, "track_reliabilities", reliability)
        object.__setattr__(self, "track_reference_xy", reference_xy)

    @property
    def mode_track_indices(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.track_ids), dtype=np.int64), np.diff(self.mode_offsets))


@dataclass(frozen=True)
class FrozenLiftedSupportViewModeBank:
    """Fixed LoFTR endpoint mixtures indexed by physical track and support view.

    Unlike :class:`FrozenLiftedTrackModeBank`, this bank intentionally does
    not establish a cross-view consensus or pool a track across support views.
    It supports the stricter candidate-conditioned likelihood

    ``sum_v p(v | q, t) L(q_mode | T, t, v)``.

    Each support-view slot retains one or more frozen LoFTR query endpoints;
    a slot without an endpoint is represented by a missing slot in the group
    layout and therefore receives the neutral likelihood ratio one.
    """

    track_ids: np.ndarray
    xyz: np.ndarray
    support_slot_track_indices: np.ndarray
    support_slot_image_ids: np.ndarray
    mode_offsets: np.ndarray
    mode_query_xy: np.ndarray
    mode_weights: np.ndarray
    mode_confidence_sums: np.ndarray
    support_reliabilities: np.ndarray
    support_reference_xy: np.ndarray

    def __post_init__(self) -> None:
        tracks = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xyz = np.asarray(self.xyz, dtype=np.float32).reshape(-1, 3)
        slot_tracks = np.asarray(self.support_slot_track_indices, dtype=np.int64).reshape(-1)
        slot_ids = np.asarray(self.support_slot_image_ids).astype(str).reshape(-1)
        offsets = np.asarray(self.mode_offsets, dtype=np.int64).reshape(-1)
        mode_xy = _as_points(self.mode_query_xy, name="support-view mode query coordinates")
        weights = np.asarray(self.mode_weights, dtype=np.float32).reshape(-1)
        confidence_sums = np.asarray(self.mode_confidence_sums, dtype=np.float32).reshape(-1)
        reliabilities = np.asarray(self.support_reliabilities, dtype=np.float32).reshape(-1)
        reference_xy = _as_points(
            self.support_reference_xy,
            name="support-view mode references",
        )
        slot_count = len(slot_tracks)
        if (
            len(tracks) == 0
            or len(np.unique(tracks)) != len(tracks)
            or np.any(tracks < 0)
            or xyz.shape != (len(tracks), 3)
            or slot_count == 0
            or slot_ids.shape != (slot_count,)
            or np.any(slot_tracks < 0)
            or np.any(slot_tracks >= len(tracks))
            or np.any(slot_ids == "")
            or len(set(zip(slot_tracks.tolist(), slot_ids.tolist()))) != slot_count
            or offsets.shape != (slot_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(mode_xy)
            or np.any(np.diff(offsets) <= 0)
            or weights.shape != (len(mode_xy),)
            or confidence_sums.shape != (slot_count,)
            or reliabilities.shape != (slot_count,)
            or reference_xy.shape != (slot_count, 2)
            or np.any(~np.isfinite(xyz))
            or np.any(~np.isfinite(weights))
            or np.any(~np.isfinite(confidence_sums))
            or np.any(~np.isfinite(reliabilities))
            or np.any(weights <= 0.0)
            or np.any(confidence_sums < 0.0)
            or np.any((reliabilities < 0.0) | (reliabilities > 1.0))
        ):
            raise ValueError("frozen lifted LoFTR support-view bank is invalid")
        for slot in range(slot_count):
            start, stop = int(offsets[slot]), int(offsets[slot + 1])
            if abs(float(weights[start:stop].sum()) - 1.0) > 2e-5:
                raise ValueError(
                    "support-view mode weights must normalize independently per slot"
                )
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "support_slot_track_indices", slot_tracks)
        object.__setattr__(self, "support_slot_image_ids", slot_ids)
        object.__setattr__(self, "mode_offsets", offsets)
        object.__setattr__(self, "mode_query_xy", mode_xy)
        object.__setattr__(self, "mode_weights", weights)
        object.__setattr__(self, "mode_confidence_sums", confidence_sums)
        object.__setattr__(self, "support_reliabilities", reliabilities)
        object.__setattr__(self, "support_reference_xy", reference_xy)

    @property
    def mode_support_slot_indices(self) -> np.ndarray:
        return np.repeat(
            np.arange(len(self.support_slot_track_indices), dtype=np.int64),
            np.diff(self.mode_offsets),
        )

    @property
    def mode_track_indices(self) -> np.ndarray:
        return self.support_slot_track_indices[self.mode_support_slot_indices]


def build_frozen_lifted_support_view_mode_bank(
    *,
    track_ids: np.ndarray,
    xyz: np.ndarray,
    support_image_ids: Sequence[str],
    query_xy: np.ndarray,
    confidence: np.ndarray,
    support_distance_px: np.ndarray,
    max_modes_per_support_view: int,
    base_support_reliability: float = 0.75,
) -> FrozenLiftedSupportViewModeBank:
    """Build independent per-support-view endpoint mixtures.

    The caller has already selected endpoints exclusively by support-anchor
    proximity.  This function merely freezes their static quality weights.  It
    never clusters across support images, so a learned support-view posterior
    can remain candidate-specific until the final marginalization.
    """

    tracks = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    points_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    points = _as_points(query_xy, name="support-view lifted query coordinates")
    scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
    distances = np.asarray(support_distance_px, dtype=np.float32).reshape(-1)
    if (
        len(tracks) == 0
        or points_xyz.shape != (len(tracks), 3)
        or image_ids.shape != tracks.shape
        or points.shape[0] != len(tracks)
        or scores.shape != tracks.shape
        or distances.shape != tracks.shape
        or np.any(tracks < 0)
        or np.any(image_ids == "")
        or np.any(~np.isfinite(points_xyz))
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any(distances < 0.0)
        or int(max_modes_per_support_view) <= 0
        or not np.isfinite(float(base_support_reliability))
        or not 0.0 < float(base_support_reliability) <= 1.0
    ):
        raise ValueError("support-view lifted LoFTR bank inputs are invalid")
    unique_tracks = np.unique(tracks)
    track_lookup = {int(track_id): row for row, track_id in enumerate(unique_tracks.tolist())}
    canonical_xyz: list[np.ndarray] = []
    for track_id in unique_tracks.tolist():
        selected = np.flatnonzero(tracks == int(track_id))
        values = points_xyz[selected]
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
            raise ValueError("one physical support-view track has incompatible XYZ values")
        canonical_xyz.append(values[0].astype(np.float32))
    slot_tracks: list[int] = []
    slot_ids: list[str] = []
    offsets = [0]
    mode_xy: list[np.ndarray] = []
    mode_weights: list[float] = []
    confidence_sums: list[float] = []
    reliabilities: list[float] = []
    references: list[np.ndarray] = []
    for track_id in unique_tracks.tolist():
        selected_track = np.flatnonzero(tracks == int(track_id))
        for image_id in sorted(set(image_ids[selected_track].tolist())):
            selected = selected_track[image_ids[selected_track] == str(image_id)]
            quality = _mode_quality(
                confidence=scores[selected],
                support_distance_px=distances[selected],
            )
            order = selected[
                np.lexsort(
                    (
                        selected,
                        points[selected, 1],
                        points[selected, 0],
                        -scores[selected],
                        distances[selected],
                        -quality,
                    )
                )
            ]
            chosen = order[: int(max_modes_per_support_view)]
            chosen_quality = _mode_quality(
                confidence=scores[chosen],
                support_distance_px=distances[chosen],
            )
            unnormalized = np.exp(chosen_quality - np.max(chosen_quality))
            weights = unnormalized / float(unnormalized.sum())
            endpoint_xy = points[chosen].astype(np.float32)
            slot_tracks.append(int(track_lookup[int(track_id)]))
            slot_ids.append(str(image_id))
            mode_xy.extend(endpoint_xy)
            mode_weights.extend(weights.astype(np.float32).tolist())
            confidence_sums.append(float(scores[chosen].sum()))
            # Reliability is fixed from LoFTR's own endpoint confidence.  It
            # cannot depend on a candidate pose or a target residual.
            reliabilities.append(
                float(base_support_reliability) * float(np.max(scores[chosen]))
            )
            references.append(
                np.sum(endpoint_xy * weights[:, None], axis=0).astype(np.float32)
            )
            offsets.append(len(mode_xy))
    return FrozenLiftedSupportViewModeBank(
        track_ids=unique_tracks.astype(np.int64),
        xyz=np.stack(canonical_xyz, axis=0).astype(np.float32),
        support_slot_track_indices=np.asarray(slot_tracks, dtype=np.int64),
        support_slot_image_ids=np.asarray(slot_ids, dtype=np.str_),
        mode_offsets=np.asarray(offsets, dtype=np.int64),
        mode_query_xy=np.stack(mode_xy, axis=0).astype(np.float32),
        mode_weights=np.asarray(mode_weights, dtype=np.float32),
        mode_confidence_sums=np.asarray(confidence_sums, dtype=np.float32),
        support_reliabilities=np.asarray(reliabilities, dtype=np.float32),
        support_reference_xy=np.stack(references, axis=0).astype(np.float32),
    )


@dataclass(frozen=True)
class FrozenCandidateSupportViewGroupLayout:
    """Fixed identity and support-view latent variables for P1 pose scoring."""

    track_count: int
    support_slot_count: int
    candidate_track_indices: np.ndarray
    candidate_support_slot_indices: np.ndarray
    candidate_support_image_ids: np.ndarray
    candidate_identity_probabilities: np.ndarray
    null_probabilities: np.ndarray
    support_view_probabilities: np.ndarray
    group_reference_xy: np.ndarray
    active_group_mask: np.ndarray

    def __post_init__(self) -> None:
        track_count = int(self.track_count)
        slot_count = int(self.support_slot_count)
        candidate_tracks = np.asarray(self.candidate_track_indices, dtype=np.int64)
        slots = np.asarray(self.candidate_support_slot_indices, dtype=np.int64)
        support_ids = np.asarray(self.candidate_support_image_ids).astype(str)
        identity = np.asarray(self.candidate_identity_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        view = np.asarray(self.support_view_probabilities, dtype=np.float32)
        references = _as_points(self.group_reference_xy, name="candidate support-view references")
        active = np.asarray(self.active_group_mask, dtype=bool).reshape(-1)
        if (
            track_count <= 0
            or slot_count <= 0
            or candidate_tracks.ndim != 2
            or candidate_tracks.shape[0] == 0
            or candidate_tracks.shape[1] <= 1
            or slots.ndim != 3
            or slots.shape[:2] != candidate_tracks.shape
            or slots.shape[2] <= 0
            or support_ids.shape != slots.shape
            or identity.shape != candidate_tracks.shape
            or null.shape != (candidate_tracks.shape[0],)
            or view.shape != slots.shape
            or references.shape != (candidate_tracks.shape[0], 2)
            or active.shape != (candidate_tracks.shape[0],)
            or np.any(candidate_tracks < -1)
            or np.any(candidate_tracks >= track_count)
            or np.any(slots < -1)
            or np.any(slots >= slot_count)
            or np.any(~np.isfinite(identity))
            or np.any(~np.isfinite(null))
            or np.any(~np.isfinite(view))
            or np.any((identity < 0.0) | (identity > 1.0))
            or np.any((null < 0.0) | (null > 1.0))
            or np.any((view < 0.0) | (view > 1.0))
            or np.any(
                np.abs(identity.sum(axis=1, dtype=np.float64) + null - 1.0) > 1e-4
            )
            or np.any(np.abs(view.sum(axis=2, dtype=np.float64) - 1.0) > 1e-4)
            or np.any((slots >= 0) & (candidate_tracks[:, :, None] < 0))
            or np.any(support_ids == "")
            or not np.any(active)
        ):
            raise ValueError("frozen candidate support-view layout is invalid")
        expected_active = np.any(
            (slots >= 0)
            & (identity[:, :, None] > 0.0)
            & (view > 0.0),
            axis=(1, 2),
        )
        if np.any(active != expected_active):
            raise ValueError("candidate support-view active mask is not fixed by evidence")
        object.__setattr__(self, "track_count", track_count)
        object.__setattr__(self, "support_slot_count", slot_count)
        object.__setattr__(self, "candidate_track_indices", candidate_tracks)
        object.__setattr__(self, "candidate_support_slot_indices", slots)
        object.__setattr__(self, "candidate_support_image_ids", support_ids)
        object.__setattr__(self, "candidate_identity_probabilities", identity)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "support_view_probabilities", view)
        object.__setattr__(self, "group_reference_xy", references)
        object.__setattr__(self, "active_group_mask", active)


def build_frozen_candidate_support_view_group_layout(
    *,
    candidate_track_ids: np.ndarray,
    candidate_identity_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    support_view_probabilities: np.ndarray,
    bank: FrozenLiftedSupportViewModeBank,
) -> FrozenCandidateSupportViewGroupLayout:
    """Bind fixed group priors to explicit per-support-view LoFTR mixtures."""

    candidate_ids = np.asarray(candidate_track_ids, dtype=np.int64)
    identity = np.asarray(candidate_identity_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    support_ids = np.asarray(candidate_support_image_ids).astype(str)
    view = np.asarray(support_view_probabilities, dtype=np.float32)
    if (
        candidate_ids.ndim != 2
        or candidate_ids.shape[0] == 0
        or candidate_ids.shape[1] <= 1
        or np.any(candidate_ids < 0)
        or identity.shape != candidate_ids.shape
        or null.shape != (candidate_ids.shape[0],)
        or support_ids.ndim != 3
        or support_ids.shape[:2] != candidate_ids.shape
        or view.shape != support_ids.shape
    ):
        raise ValueError("candidate support-view group inputs are invalid")
    track_lookup = {int(track_id): row for row, track_id in enumerate(bank.track_ids.tolist())}
    slot_lookup = {
        (int(bank.track_ids[int(track_index)]), str(image_id)): slot
        for slot, (track_index, image_id) in enumerate(
            zip(
                bank.support_slot_track_indices.tolist(),
                bank.support_slot_image_ids.tolist(),
            )
        )
    }
    candidate_tracks = np.full(candidate_ids.shape, -1, dtype=np.int64)
    slots = np.full(support_ids.shape, -1, dtype=np.int64)
    for group, row in enumerate(candidate_ids.tolist()):
        for candidate, track_id in enumerate(row):
            track_index = track_lookup.get(int(track_id), -1)
            candidate_tracks[group, candidate] = int(track_index)
            if track_index < 0:
                continue
            for view_index, image_id in enumerate(support_ids[group, candidate].tolist()):
                slots[group, candidate, view_index] = int(
                    slot_lookup.get((int(track_id), str(image_id)), -1)
                )
    references = np.zeros((candidate_ids.shape[0], 2), dtype=np.float32)
    active = np.zeros((candidate_ids.shape[0],), dtype=bool)
    for group in range(candidate_ids.shape[0]):
        weights: list[float] = []
        locations: list[np.ndarray] = []
        for candidate in range(candidate_ids.shape[1]):
            for view_index in range(support_ids.shape[2]):
                slot = int(slots[group, candidate, view_index])
                weight = float(identity[group, candidate]) * float(
                    view[group, candidate, view_index]
                )
                if slot < 0 or weight <= 0.0:
                    continue
                weights.append(weight)
                locations.append(bank.support_reference_xy[slot])
        if weights:
            normalized = np.asarray(weights, dtype=np.float64)
            normalized /= float(normalized.sum())
            references[group] = np.sum(
                np.stack(locations, axis=0) * normalized[:, None], axis=0
            ).astype(np.float32)
            active[group] = True
    return FrozenCandidateSupportViewGroupLayout(
        track_count=len(bank.track_ids),
        support_slot_count=len(bank.support_slot_track_indices),
        candidate_track_indices=candidate_tracks,
        candidate_support_slot_indices=slots,
        candidate_support_image_ids=support_ids,
        candidate_identity_probabilities=identity,
        null_probabilities=null,
        support_view_probabilities=view,
        group_reference_xy=references,
        active_group_mask=active,
    )


def support_view_log_ratios_from_projected(
    *,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    bank: FrozenLiftedSupportViewModeBank,
    image_width: int,
    image_height: int,
    sigma_px: float,
    out_of_image_ratio: float,
    max_log_ratio: float,
) -> torch.Tensor:
    """Score every frozen support-view slot for every supplied pose."""

    projected = torch.as_tensor(projected_xy)
    valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=projected.device)
    if (
        projected.ndim != 3
        or projected.shape[2] != 2
        or valid.shape != projected.shape[:2]
        or projected.shape[1] != len(bank.track_ids)
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not np.isfinite([sigma_px, out_of_image_ratio, max_log_ratio]).all()
        or float(sigma_px) <= 0.0
        or not 0.0 < float(out_of_image_ratio) < 1.0
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("support-view lifted LoFTR pose-score inputs are invalid")
    dtype = projected.dtype
    device = projected.device
    mode_tracks = torch.as_tensor(bank.mode_track_indices, dtype=torch.long, device=device)
    mode_slots = torch.as_tensor(
        bank.mode_support_slot_indices,
        dtype=torch.long,
        device=device,
    )
    mode_xy = torch.as_tensor(bank.mode_query_xy, dtype=dtype, device=device)
    mode_weights = torch.as_tensor(bank.mode_weights, dtype=dtype, device=device)
    reliability = torch.as_tensor(bank.support_reliabilities, dtype=dtype, device=device)
    mode_projected = projected[:, mode_tracks, :]
    mode_valid = valid[:, mode_tracks]
    squared_error = torch.sum((mode_projected - mode_xy[None]) ** 2, dim=2)
    uniform_area = float(int(image_width) * int(image_height))
    gaussian_ratio = uniform_area / (2.0 * np.pi * float(sigma_px) ** 2)
    mode_ratio = gaussian_ratio * torch.exp(-0.5 * squared_error / float(sigma_px) ** 2)
    mode_ratio = torch.clamp(mode_ratio, min=0.0, max=float(np.exp(max_log_ratio)))
    mode_ratio = torch.where(
        mode_valid,
        mode_ratio,
        torch.full_like(mode_ratio, float(out_of_image_ratio)),
    )
    slot_ratio = torch.zeros(
        (projected.shape[0], len(bank.support_slot_track_indices)),
        dtype=dtype,
        device=device,
    )
    slot_ratio.scatter_add_(
        1,
        mode_slots[None].expand(projected.shape[0], -1),
        mode_ratio * mode_weights[None],
    )
    mixed_ratio = (1.0 - reliability[None]) + reliability[None] * slot_ratio
    return torch.log(
        torch.clamp(mixed_ratio, min=1e-12, max=float(np.exp(max_log_ratio)))
    )


def fixed_prior_group_log_ratios_from_support_view_log_ratios(
    *,
    support_view_log_ratios: torch.Tensor,
    layout: FrozenCandidateSupportViewGroupLayout,
    max_log_ratio: float,
) -> torch.Tensor:
    """Marginalize fixed identity and support-view latents without argmaxing."""

    _candidate_log_ratios, group_log_ratios = (
        candidate_support_view_log_mixture_terms_from_support_view_log_ratios(
            support_view_log_ratios=support_view_log_ratios,
            layout=layout,
            max_log_ratio=max_log_ratio,
        )
    )
    return group_log_ratios


def candidate_support_view_log_mixture_terms_from_support_view_log_ratios(
    *,
    support_view_log_ratios: torch.Tensor,
    layout: FrozenCandidateSupportViewGroupLayout,
    max_log_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed candidate and group log likelihood-ratio mixtures.

    The first result has shape ``[batch, group, candidate]`` and is useful for
    target-only diagnostics: it preserves the contribution of each fixed
    identity before the group-level null/identity marginalization.  The second
    result is the exact group score used by production-style frozen scoring.
    Missing support slots remain a unit likelihood ratio in both results.
    """

    values = torch.as_tensor(support_view_log_ratios)
    if (
        values.ndim != 2
        or values.shape[1] != int(layout.support_slot_count)
        or not np.isfinite(float(max_log_ratio))
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("candidate support-view group log-ratio inputs are invalid")
    device = values.device
    slots = torch.as_tensor(
        layout.candidate_support_slot_indices,
        dtype=torch.long,
        device=device,
    )
    flat_slots = slots.reshape(-1)
    valid = flat_slots >= 0
    view_ratios = torch.ones(
        (values.shape[0], len(flat_slots)),
        dtype=values.dtype,
        device=device,
    )
    if bool(torch.any(valid)):
        view_ratios[:, valid] = torch.exp(
            torch.clamp(
                values[:, flat_slots[valid]],
                min=-float(max_log_ratio),
                max=float(max_log_ratio),
            )
        )
    view_ratios = view_ratios.reshape(
        values.shape[0],
        slots.shape[0],
        slots.shape[1],
        slots.shape[2],
    )
    view_probabilities = torch.as_tensor(
        layout.support_view_probabilities,
        dtype=values.dtype,
        device=device,
    )
    identity = torch.as_tensor(
        layout.candidate_identity_probabilities,
        dtype=values.dtype,
        device=device,
    )
    null = torch.as_tensor(layout.null_probabilities, dtype=values.dtype, device=device)
    candidate_ratios = torch.sum(view_probabilities[None] * view_ratios, dim=3)
    group_ratio = null[None] + torch.sum(identity[None] * candidate_ratios, dim=2)
    candidate_log_ratios = torch.log(
        torch.clamp(candidate_ratios, min=1e-12, max=float(np.exp(max_log_ratio)))
    )
    group_log_ratios = torch.log(
        torch.clamp(group_ratio, min=1e-12, max=float(np.exp(max_log_ratio)))
    )
    return candidate_log_ratios, group_log_ratios


@dataclass(frozen=True)
class FrozenCandidateGroupLayout:
    """Immutable top-L identity groups expressed in the lifted track bank.

    A PnP fit row owns one mutually exclusive top-L candidate group.  The
    verifier must marginalize that latent identity *within* its group rather
    than rewarding every candidate landmark in the global union independently.
    Missing lifted tracks retain a unit likelihood-ratio component, which is a
    fixed no-evidence contribution rather than pose-dependent pruning.
    """

    track_count: int
    candidate_track_indices: np.ndarray
    group_reference_xy: np.ndarray
    active_group_mask: np.ndarray
    candidate_identity_probabilities: np.ndarray | None = None
    null_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        track_count = int(self.track_count)
        indices = np.asarray(self.candidate_track_indices, dtype=np.int64)
        reference_xy = _as_points(self.group_reference_xy, name="candidate group references")
        active = np.asarray(self.active_group_mask, dtype=bool).reshape(-1)
        candidate_probabilities = self.candidate_identity_probabilities
        null_probabilities = self.null_probabilities
        if (
            track_count <= 0
            or indices.ndim != 2
            or indices.shape[0] == 0
            or indices.shape[1] <= 1
            or reference_xy.shape != (indices.shape[0], 2)
            or active.shape != (indices.shape[0],)
            or np.any(indices < -1)
            or np.any(indices >= track_count)
            or not np.any(active)
            or np.any(active != np.any(indices >= 0, axis=1))
        ):
            raise ValueError("frozen candidate-group layout is invalid")
        if (candidate_probabilities is None) != (null_probabilities is None):
            raise ValueError("candidate identity and null priors must be supplied together")
        if candidate_probabilities is not None:
            probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
            null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
            if (
                probabilities.shape != indices.shape
                or null.shape != (indices.shape[0],)
                or np.any(~np.isfinite(probabilities))
                or np.any(~np.isfinite(null))
                or np.any((probabilities < 0.0) | (probabilities > 1.0))
                or np.any((null < 0.0) | (null > 1.0))
                or np.any(
                    np.abs(probabilities.sum(axis=1, dtype=np.float64) + null - 1.0)
                    > 1e-4
                )
            ):
                raise ValueError("frozen candidate-group identity prior is invalid")
            object.__setattr__(
                self,
                "candidate_identity_probabilities",
                probabilities,
            )
            object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "track_count", track_count)
        object.__setattr__(self, "candidate_track_indices", indices)
        object.__setattr__(self, "group_reference_xy", reference_xy)
        object.__setattr__(self, "active_group_mask", active)


def build_frozen_candidate_group_layout(
    *,
    candidate_track_ids: np.ndarray,
    bank: FrozenLiftedTrackModeBank,
    candidate_identity_probabilities: np.ndarray | None = None,
    null_probabilities: np.ndarray | None = None,
) -> FrozenCandidateGroupLayout:
    """Bind fixed PnP top-L rows to available lifted LoFTR tracks.

    The group reference only assigns a fixed block for robust aggregation.  It
    comes from held-out LoFTR mode locations and is never a PnP query center
    nor a pose-dependent match choice.
    """

    candidates = np.asarray(candidate_track_ids, dtype=np.int64)
    if (
        candidates.ndim != 2
        or candidates.shape[0] == 0
        or candidates.shape[1] <= 1
        or np.any(candidates < 0)
        or len(np.unique(bank.track_ids)) != len(bank.track_ids)
    ):
        raise ValueError("fixed candidate group track ids are invalid")
    track_lookup = {int(track_id): row for row, track_id in enumerate(bank.track_ids.tolist())}
    indices = np.full(candidates.shape, -1, dtype=np.int64)
    for group_row in range(candidates.shape[0]):
        for candidate_column, track_id in enumerate(candidates[group_row].tolist()):
            indices[group_row, candidate_column] = int(track_lookup.get(int(track_id), -1))
    references = np.zeros((candidates.shape[0], 2), dtype=np.float32)
    active = np.any(indices >= 0, axis=1)
    for group_row in range(candidates.shape[0]):
        available = indices[group_row] >= 0
        if not np.any(available):
            # This group retains a fixed null-only likelihood ratio of one.
            # It has no held-out visual position, so it is excluded only by
            # the immutable active mask during spatial aggregation.
            continue
        rows = indices[group_row, available]
        # This only partitions fixed group evidence spatially.  It is not a
        # localization measurement, and individual track likelihoods remain
        # separate inside the latent-identity mixture.
        references[group_row] = np.mean(bank.track_reference_xy[rows], axis=0)
    return FrozenCandidateGroupLayout(
        track_count=len(bank.track_ids),
        candidate_track_indices=indices,
        group_reference_xy=references,
        active_group_mask=active,
        candidate_identity_probabilities=candidate_identity_probabilities,
        null_probabilities=null_probabilities,
    )


def group_log_ratios_from_track_log_ratios(
    *,
    track_log_ratios: torch.Tensor,
    layout: FrozenCandidateGroupLayout,
    null_mass: float,
    max_log_ratio: float,
) -> torch.Tensor:
    """Marginalize one fixed identity candidate or null per PnP query group."""

    values = torch.as_tensor(track_log_ratios)
    if (
        values.ndim != 2
        or values.shape[1] != int(layout.track_count)
        or not np.isfinite([null_mass, max_log_ratio]).all()
        or not 0.0 < float(null_mass) < 1.0
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("candidate-group log-ratio inputs are invalid")
    device = values.device
    indices = torch.as_tensor(layout.candidate_track_indices, dtype=torch.long, device=device)
    flat_indices = indices.reshape(-1)
    valid = flat_indices >= 0
    candidate_ratios = torch.ones(
        (values.shape[0], len(flat_indices)), dtype=values.dtype, device=device
    )
    if bool(torch.any(valid)):
        candidate_ratios[:, valid] = torch.exp(
            torch.clamp(values[:, flat_indices[valid]], min=-float(max_log_ratio), max=float(max_log_ratio))
        )
    candidate_ratios = candidate_ratios.reshape(
        values.shape[0], indices.shape[0], indices.shape[1]
    )
    group_ratio = float(null_mass) + (1.0 - float(null_mass)) * candidate_ratios.mean(dim=2)
    return torch.log(torch.clamp(group_ratio, min=1e-12, max=float(np.exp(max_log_ratio))))


def fixed_prior_group_log_ratios_from_track_log_ratios(
    *,
    track_log_ratios: torch.Tensor,
    layout: FrozenCandidateGroupLayout,
    max_log_ratio: float,
) -> torch.Tensor:
    """Marginalize fixed learned identity/null priors with LoFTR likelihood ratios.

    The candidate posterior is fixed before any pose is evaluated.  Candidates
    absent from the lifted mode bank remain in the immutable denominator with a
    unit likelihood ratio, so missing held-out evidence cannot become an
    implicit pose-dependent rejection.
    """

    values = torch.as_tensor(track_log_ratios)
    if (
        values.ndim != 2
        or values.shape[1] != int(layout.track_count)
        or layout.candidate_identity_probabilities is None
        or layout.null_probabilities is None
        or not np.isfinite(max_log_ratio)
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("fixed-prior candidate-group log-ratio inputs are invalid")
    device = values.device
    indices = torch.as_tensor(layout.candidate_track_indices, dtype=torch.long, device=device)
    probabilities = torch.as_tensor(
        layout.candidate_identity_probabilities,
        dtype=values.dtype,
        device=device,
    )
    null = torch.as_tensor(
        layout.null_probabilities,
        dtype=values.dtype,
        device=device,
    )
    flat_indices = indices.reshape(-1)
    valid = flat_indices >= 0
    candidate_ratios = torch.ones(
        (values.shape[0], len(flat_indices)), dtype=values.dtype, device=device
    )
    if bool(torch.any(valid)):
        candidate_ratios[:, valid] = torch.exp(
            torch.clamp(
                values[:, flat_indices[valid]],
                min=-float(max_log_ratio),
                max=float(max_log_ratio),
            )
        )
    candidate_ratios = candidate_ratios.reshape(
        values.shape[0], indices.shape[0], indices.shape[1]
    )
    group_ratio = null[None, :] + torch.sum(
        probabilities[None, :, :] * candidate_ratios,
        dim=2,
    )
    return torch.log(torch.clamp(group_ratio, min=1e-12, max=float(np.exp(max_log_ratio))))


def _mode_quality(*, confidence: np.ndarray, support_distance_px: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(confidence, dtype=np.float64), 1e-8)) - 0.5 * np.square(
        np.asarray(support_distance_px, dtype=np.float64)
    )


def build_frozen_lifted_track_mode_bank(
    *,
    track_ids: np.ndarray,
    xyz: np.ndarray,
    support_image_ids: Sequence[str],
    query_xy: np.ndarray,
    confidence: np.ndarray,
    support_distance_px: np.ndarray,
    min_support_views: int,
    consensus_radius_px: float,
    max_modes_per_track: int,
    base_track_reliability: float = 0.75,
    reliability_reference_view_count: int = 3,
) -> FrozenLiftedTrackModeBank:
    """Cluster fixed endpoints without ever averaging support-view observations.

    Each consensus cluster has one static best endpoint per support image.  At
    scoring time its individual endpoints remain a mixture, so view-specific
    ambiguities stay explicit instead of being collapsed to an artificial
    query-space centroid.
    """

    tracks = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    points_xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    image_ids = np.asarray(support_image_ids).astype(str).reshape(-1)
    points = _as_points(query_xy, name="lifted query coordinates")
    scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
    distances = np.asarray(support_distance_px, dtype=np.float32).reshape(-1)
    if (
        len(tracks) == 0
        or points_xyz.shape != (len(tracks), 3)
        or image_ids.shape != tracks.shape
        or points.shape[0] != len(tracks)
        or scores.shape != tracks.shape
        or distances.shape != tracks.shape
        or np.any(tracks < 0)
        or np.any(image_ids == "")
        or np.any(~np.isfinite(points_xyz))
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any(distances < 0.0)
        or int(min_support_views) < 2
        or int(max_modes_per_track) <= 0
        or int(reliability_reference_view_count) < int(min_support_views)
        or not np.isfinite(float(consensus_radius_px))
        or float(consensus_radius_px) <= 0.0
        or not 0.0 < float(base_track_reliability) < 1.0
    ):
        raise ValueError("lifted track-mode clustering inputs are invalid")
    output_tracks: list[int] = []
    output_xyz: list[np.ndarray] = []
    output_offsets = [0]
    output_mode_xy: list[np.ndarray] = []
    output_mode_weights: list[float] = []
    output_mode_support_ids: list[str] = []
    output_mode_views: list[int] = []
    output_mode_confidence: list[float] = []
    output_reliability: list[float] = []
    output_reference_xy: list[np.ndarray] = []
    for track_id in np.unique(tracks).tolist():
        selected = np.flatnonzero(tracks == int(track_id))
        track_xyz = points_xyz[selected]
        if not np.allclose(track_xyz, track_xyz[0], rtol=0.0, atol=1e-6):
            raise ValueError("one physical track has incompatible XYZ values")
        quality = _mode_quality(
            confidence=scores[selected], support_distance_px=distances[selected]
        )
        order = selected[
            np.lexsort(
                (
                    selected,
                    points[selected, 1],
                    points[selected, 0],
                    -scores[selected],
                    distances[selected],
                    -quality,
                )
            )
        ]
        used = np.zeros((len(selected),), dtype=bool)
        local_position = {int(row): position for position, row in enumerate(selected.tolist())}
        track_clusters: list[tuple[np.ndarray, int, float, float]] = []
        for seed in order.tolist():
            seed_position = local_position[int(seed)]
            if used[seed_position]:
                continue
            candidate_rows = selected[
                (~used)
                & (
                    np.linalg.norm(points[selected] - points[int(seed)][None], axis=1)
                    <= float(consensus_radius_px)
                )
            ]
            if len(candidate_rows) == 0:
                continue
            # One fixed best mode per support image prevents a single image from
            # fabricating a cross-view consensus cluster.
            chosen_rows: list[int] = []
            for image_id in sorted(set(image_ids[candidate_rows].tolist())):
                image_rows = candidate_rows[image_ids[candidate_rows] == image_id]
                image_quality = _mode_quality(
                    confidence=scores[image_rows], support_distance_px=distances[image_rows]
                )
                best = image_rows[
                    np.lexsort(
                        (
                            image_rows,
                            points[image_rows, 1],
                            points[image_rows, 0],
                            -scores[image_rows],
                            distances[image_rows],
                            -image_quality,
                        )
                    )[0]
                ]
                chosen_rows.append(int(best))
            if len(chosen_rows) < int(min_support_views):
                used[seed_position] = True
                continue
            chosen = np.asarray(chosen_rows, dtype=np.int64)
            # The centroid is a fixed clustering aid only.  It is never
            # emitted into the likelihood bank; individual per-view endpoints
            # remain the probability components below.
            centroid_weights = np.maximum(scores[chosen], 1e-4).astype(np.float64)
            cluster_centroid = np.sum(
                points[chosen] * centroid_weights[:, None], axis=0
            ) / float(centroid_weights.sum())
            # A fixed inlier recheck prevents a loose seed from merging two image
            # structures through a chain of nearby endpoints.
            inlier = np.linalg.norm(points[chosen] - cluster_centroid[None], axis=1) <= float(
                consensus_radius_px
            )
            chosen = chosen[inlier]
            if len(chosen) < int(min_support_views):
                used[seed_position] = True
                continue
            confidence_sum = float(scores[chosen].sum())
            cluster_strength = float(len(chosen)) * max(confidence_sum, 1e-4)
            track_clusters.append(
                (chosen, int(len(chosen)), confidence_sum, cluster_strength)
            )
            for row in candidate_rows.tolist():
                used[local_position[int(row)]] = True
            if len(track_clusters) >= int(max_modes_per_track):
                break
        if not track_clusters:
            continue
        cluster_strengths = np.asarray(
            [item[3] for item in track_clusters], dtype=np.float64
        )
        cluster_weights = cluster_strengths / float(cluster_strengths.sum())
        best_view_count = max(item[1] for item in track_clusters)
        reliability = float(base_track_reliability) * min(
            1.0, float(best_view_count) / float(reliability_reference_view_count)
        )
        track_endpoint_xy: list[np.ndarray] = []
        track_endpoint_weights: list[float] = []
        track_endpoint_support_ids: list[str] = []
        track_endpoint_view_counts: list[int] = []
        track_endpoint_confidence_sums: list[float] = []
        for cluster_weight, (chosen, view_count, confidence_sum, _strength) in zip(
            cluster_weights.tolist(), track_clusters
        ):
            endpoint_quality = np.maximum(scores[chosen], 1e-4).astype(np.float64)
            endpoint_weights = endpoint_quality / float(endpoint_quality.sum())
            for row, endpoint_weight in zip(chosen.tolist(), endpoint_weights.tolist()):
                track_endpoint_xy.append(points[int(row)].astype(np.float32))
                track_endpoint_weights.append(float(cluster_weight) * float(endpoint_weight))
                track_endpoint_support_ids.append(str(image_ids[int(row)]))
                track_endpoint_view_counts.append(int(view_count))
                track_endpoint_confidence_sums.append(float(confidence_sum))
        normalized_endpoint_weights = np.asarray(track_endpoint_weights, dtype=np.float64)
        normalized_endpoint_weights /= float(normalized_endpoint_weights.sum())
        endpoint_xy = np.stack(track_endpoint_xy, axis=0)
        # This value only assigns a fixed spatial block for robust aggregation.
        # It is not used as a measurement location or mode likelihood.
        reference_xy = np.sum(
            endpoint_xy * normalized_endpoint_weights[:, None], axis=0
        ).astype(np.float32)
        output_tracks.append(int(track_id))
        output_xyz.append(track_xyz[0].astype(np.float32))
        output_mode_xy.extend(track_endpoint_xy)
        output_mode_weights.extend(normalized_endpoint_weights.astype(np.float32).tolist())
        output_mode_support_ids.extend(track_endpoint_support_ids)
        output_mode_views.extend(track_endpoint_view_counts)
        output_mode_confidence.extend(track_endpoint_confidence_sums)
        output_reliability.append(reliability)
        output_reference_xy.append(reference_xy)
        output_offsets.append(len(output_mode_xy))
    if not output_tracks:
        raise ValueError("no fixed cross-view LoFTR track modes satisfy the declared contract")
    return FrozenLiftedTrackModeBank(
        track_ids=np.asarray(output_tracks, dtype=np.int64),
        xyz=np.stack(output_xyz, axis=0).astype(np.float32),
        mode_offsets=np.asarray(output_offsets, dtype=np.int64),
        mode_query_xy=np.stack(output_mode_xy, axis=0).astype(np.float32),
        mode_weights=np.asarray(output_mode_weights, dtype=np.float32),
        mode_support_image_ids=np.asarray(output_mode_support_ids, dtype=np.str_),
        mode_support_view_counts=np.asarray(output_mode_views, dtype=np.int64),
        mode_confidence_sums=np.asarray(output_mode_confidence, dtype=np.float32),
        track_reliabilities=np.asarray(output_reliability, dtype=np.float32),
        track_reference_xy=np.stack(output_reference_xy, axis=0).astype(np.float32),
    )


def deterministic_track_xyz_permutation(track_ids: np.ndarray, *, query_id: str) -> np.ndarray:
    """Return a reproducible derangement for the geometric control arm."""

    tracks = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    if len(tracks) < 2 or len(np.unique(tracks)) != len(tracks) or not str(query_id):
        raise ValueError("XYZ permutation needs at least two unique tracks and a query id")
    seed = int.from_bytes(
        hashlib.sha256(("lifted_loftr_xyz_control_v1:" + str(query_id)).encode("utf8")).digest()[:8],
        byteorder="little",
        signed=False,
    )
    permutation = np.random.default_rng(seed).permutation(len(tracks)).astype(np.int64)
    fixed = np.flatnonzero(permutation == np.arange(len(tracks), dtype=np.int64))
    if len(fixed) == 1:
        # A swap with any other position removes the only fixed point without
        # creating one: the permutation's sole preimage of ``fixed[0]`` is
        # itself before the swap.
        partner = 0 if int(fixed[0]) != 0 else 1
        permutation[int(fixed[0])], permutation[partner] = (
            permutation[partner],
            permutation[int(fixed[0])],
        )
    elif len(fixed) > 1:
        # Rotate the values at all fixed locations, which is a deterministic
        # cycle over those tracks and removes every fixed point at once.
        permutation[fixed] = np.roll(permutation[fixed], 1)
    if np.any(permutation == np.arange(len(tracks), dtype=np.int64)):
        raise RuntimeError("deterministic XYZ permutation failed to derange tracks")
    return permutation


def track_mode_log_ratios_from_projected(
    *,
    projected_xy: torch.Tensor,
    projection_valid: torch.Tensor,
    bank: FrozenLiftedTrackModeBank,
    image_width: int,
    image_height: int,
    sigma_px: float,
    out_of_image_ratio: float,
    max_log_ratio: float,
) -> torch.Tensor:
    """Score every fixed track under every supplied pose projection.

    Invalid/out-of-image projections receive a fixed negative density ratio;
    they are retained in the denominator rather than being dropped per pose.
    Multiple per-view query modes for one track are marginalised with
    pre-frozen weights; they were never pre-averaged into a centroid.
    """

    projected = torch.as_tensor(projected_xy)
    valid = torch.as_tensor(projection_valid, dtype=torch.bool, device=projected.device)
    if (
        projected.ndim != 3
        or projected.shape[2] != 2
        or valid.shape != projected.shape[:2]
        or projected.shape[1] != len(bank.track_ids)
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not np.isfinite([sigma_px, out_of_image_ratio, max_log_ratio]).all()
        or float(sigma_px) <= 0.0
        or not 0.0 < float(out_of_image_ratio) < 1.0
        or float(max_log_ratio) <= 0.0
    ):
        raise ValueError("lifted LoFTR pose-score inputs are invalid")
    dtype = projected.dtype
    device = projected.device
    mode_tracks = torch.as_tensor(bank.mode_track_indices, dtype=torch.long, device=device)
    mode_xy = torch.as_tensor(bank.mode_query_xy, dtype=dtype, device=device)
    mode_weights = torch.as_tensor(bank.mode_weights, dtype=dtype, device=device)
    reliability = torch.as_tensor(bank.track_reliabilities, dtype=dtype, device=device)
    mode_projected = projected[:, mode_tracks, :]
    mode_valid = valid[:, mode_tracks]
    squared_error = torch.sum((mode_projected - mode_xy[None]) ** 2, dim=2)
    uniform_area = float(int(image_width) * int(image_height))
    gaussian_ratio = uniform_area / (2.0 * np.pi * float(sigma_px) ** 2)
    mode_ratio = gaussian_ratio * torch.exp(-0.5 * squared_error / float(sigma_px) ** 2)
    mode_ratio = torch.clamp(mode_ratio, min=0.0, max=float(np.exp(max_log_ratio)))
    mode_ratio = torch.where(
        mode_valid,
        mode_ratio,
        torch.full_like(mode_ratio, float(out_of_image_ratio)),
    )
    weighted_modes = mode_ratio * mode_weights[None]
    track_ratio = torch.zeros(
        (projected.shape[0], len(bank.track_ids)), dtype=dtype, device=device
    )
    track_ratio.scatter_add_(
        1, mode_tracks[None].expand(projected.shape[0], -1), weighted_modes
    )
    mixed_ratio = (1.0 - reliability[None]) + reliability[None] * track_ratio
    return torch.log(torch.clamp(mixed_ratio, min=1e-12, max=float(np.exp(max_log_ratio))))


def robust_track_log_ratio_statistics(
    *,
    track_log_ratios: torch.Tensor,
    track_reference_xy: np.ndarray,
    image_width: int,
    image_height: int,
) -> dict[str, torch.Tensor]:
    """Use fixed 2x2 blocks so one facade cannot dominate raw selection."""

    values = torch.as_tensor(track_log_ratios)
    reference_xy = _as_points(track_reference_xy, name="track reference coordinates")
    if (
        values.ndim != 2
        or values.shape[1] != len(reference_xy)
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not torch.isfinite(values).all()
    ):
        raise ValueError("robust lifted LoFTR statistics inputs are invalid")
    sorted_values = torch.sort(values, dim=1).values
    quartile_count = max(1, int(np.ceil(values.shape[1] * 0.25)))
    x_block = np.minimum((reference_xy[:, 0] >= float(image_width) * 0.5).astype(np.int64), 1)
    y_block = np.minimum((reference_xy[:, 1] >= float(image_height) * 0.5).astype(np.int64), 1)
    blocks = y_block * 2 + x_block
    block_means: list[torch.Tensor] = []
    for block in range(4):
        selected = np.flatnonzero(blocks == block)
        if len(selected) == 0:
            continue
        block_means.append(values[:, torch.as_tensor(selected, dtype=torch.long, device=values.device)].mean(dim=1))
    if not block_means:
        raise RuntimeError("lifted LoFTR tracks occupy no spatial blocks")
    block_stack = torch.stack(block_means, dim=1)
    return {
        "means": values.mean(dim=1),
        "medians": values.median(dim=1).values,
        "worst_quartile_means": sorted_values[:, :quartile_count].mean(dim=1),
        "spatial_median_of_means_2x2": block_stack.median(dim=1).values,
        "effective_track_counts": torch.full(
            (values.shape[0],), int(values.shape[1]), dtype=torch.int64, device=values.device
        ),
    }
