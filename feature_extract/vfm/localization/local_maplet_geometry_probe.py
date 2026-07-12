"""Real support-view geometry for candidate-conditioned local maplet probes.

The global retriever proposes physical SfM tracks.  This module verifies one
proposal using only its local maplet, real support observations, and a query
detector neighborhood.  Ground-truth query pose is deliberately absent from
the feature path.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.query_to_3d_matching import normalize_rows


@dataclass(frozen=True)
class SupportObservationGeometryIndex:
    """Image-major index into an external observation feature cache."""

    image_ids: tuple[str, ...]
    image_offsets: np.ndarray
    source_row_indices: np.ndarray
    track_ids: np.ndarray
    xy: np.ndarray
    viewing_rays: np.ndarray
    reprojection_errors: np.ndarray

    def __post_init__(self) -> None:
        image_ids = tuple(str(value) for value in self.image_ids)
        offsets = np.asarray(self.image_offsets, dtype=np.int64).reshape(-1)
        source_rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        tracks = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        rays = np.asarray(self.viewing_rays, dtype=np.float32).reshape(-1, 3)
        errors = np.asarray(self.reprojection_errors, dtype=np.float32).reshape(-1)
        count = int(len(source_rows))
        if offsets.shape != (len(image_ids) + 1,) or offsets[0] != 0 or offsets[-1] != count:
            raise ValueError("image_offsets must delimit every indexed observation")
        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("image_offsets must be monotonic")
        if tracks.shape[0] != count or xy.shape[0] != count or rays.shape[0] != count or errors.shape[0] != count:
            raise ValueError("all observation arrays must contain the same number of rows")
        if np.unique(source_rows).size != count or np.any(source_rows < 0):
            raise ValueError("source_row_indices must be unique and non-negative")
        if tuple(sorted(image_ids)) != image_ids or len(set(image_ids)) != len(image_ids):
            raise ValueError("image_ids must be sorted and unique")
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            image_tracks = tracks[int(start) : int(end)]
            if image_tracks.size > 1 and np.any(image_tracks[1:] <= image_tracks[:-1]):
                raise ValueError("tracks within each image must be strictly increasing")
        object.__setattr__(self, "image_ids", image_ids)
        object.__setattr__(self, "image_offsets", offsets)
        object.__setattr__(self, "source_row_indices", source_rows)
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "viewing_rays", rays)
        object.__setattr__(self, "reprojection_errors", errors)

    def __len__(self) -> int:
        return int(self.source_row_indices.size)

    def image_position(self, image_id: str) -> int | None:
        position = int(bisect_left(self.image_ids, str(image_id)))
        if position >= len(self.image_ids) or self.image_ids[position] != str(image_id):
            return None
        return position

    def image_slice(self, image_id: str) -> slice:
        position = self.image_position(str(image_id))
        if position is None:
            return slice(0, 0)
        return slice(int(self.image_offsets[position]), int(self.image_offsets[position + 1]))

    def geometry_rows_for_tracks(self, image_id: str, track_ids: np.ndarray) -> np.ndarray:
        """Return geometry rows in requested-track order, or -1 when invisible."""

        requested = np.asarray(track_ids, dtype=np.int64).reshape(-1)
        output = np.full(requested.shape, -1, dtype=np.int64)
        image_slice = self.image_slice(str(image_id))
        available = self.track_ids[image_slice]
        if available.size == 0 or requested.size == 0:
            return output
        positions = np.searchsorted(available, requested)
        clipped = np.minimum(positions, len(available) - 1)
        valid = (positions < len(available)) & (available[clipped] == requested)
        output[valid] = int(image_slice.start) + positions[valid]
        return output


def build_support_observation_geometry_index(
    observations: Sequence[ColmapTrackObservation],
) -> SupportObservationGeometryIndex:
    """Build a deterministic image/track lookup while preserving source rows."""

    count = int(len(observations))
    image_ids = tuple(sorted({str(observation.image_id) for observation in observations}))
    image_to_position = {image_id: position for position, image_id in enumerate(image_ids)}
    image_positions = np.fromiter(
        (image_to_position[str(observation.image_id)] for observation in observations),
        dtype=np.int64,
        count=count,
    )
    tracks = np.fromiter(
        (int(observation.track_id) for observation in observations),
        dtype=np.int64,
        count=count,
    )
    source_rows = np.arange(count, dtype=np.int64)
    order = np.lexsort((source_rows, tracks, image_positions))
    sorted_images = image_positions[order]
    counts = np.bincount(sorted_images, minlength=len(image_ids)) if count else np.zeros((len(image_ids),), dtype=np.int64)
    offsets = np.concatenate([np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
    xy = np.asarray([observation.xy for observation in observations], dtype=np.float32).reshape(-1, 2)
    rays = np.stack(
        [
            np.full((3,), np.nan, dtype=np.float32)
            if observation.viewing_ray is None
            else np.asarray(observation.viewing_ray, dtype=np.float32).reshape(3)
            for observation in observations
        ],
        axis=0,
    ) if count else np.zeros((0, 3), dtype=np.float32)
    errors = np.fromiter(
        (float(observation.reprojection_error) for observation in observations),
        dtype=np.float32,
        count=count,
    )
    return SupportObservationGeometryIndex(
        image_ids=image_ids,
        image_offsets=offsets,
        source_row_indices=source_rows[order],
        track_ids=tracks[order],
        xy=xy[order],
        viewing_rays=rays[order],
        reprojection_errors=errors[order],
    )


def save_support_observation_geometry_index_npz(
    index: SupportObservationGeometryIndex,
    path: Path,
    *,
    metadata: Mapping[str, object],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "support_observation_geometry_index_npz",
        "format_version": 1,
        **dict(metadata),
    }
    np.savez(
        output,
        image_ids=np.asarray(index.image_ids, dtype=np.str_),
        image_offsets=index.image_offsets,
        source_row_indices=index.source_row_indices,
        track_ids=index.track_ids,
        xy=index.xy,
        viewing_rays=index.viewing_rays,
        reprojection_errors=index.reprojection_errors,
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True), dtype=np.str_),
    )


def load_support_observation_geometry_index_npz(
    path: Path,
) -> tuple[SupportObservationGeometryIndex, dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("format") != "support_observation_geometry_index_npz":
            raise ValueError("unsupported support-observation geometry cache")
        index = SupportObservationGeometryIndex(
            image_ids=tuple(str(value) for value in data["image_ids"].tolist()),
            image_offsets=np.asarray(data["image_offsets"], dtype=np.int64),
            source_row_indices=np.asarray(data["source_row_indices"], dtype=np.int64),
            track_ids=np.asarray(data["track_ids"], dtype=np.int64),
            xy=np.asarray(data["xy"], dtype=np.float32),
            viewing_rays=np.asarray(data["viewing_rays"], dtype=np.float32),
            reprojection_errors=np.asarray(data["reprojection_errors"], dtype=np.float32),
        )
    return index, metadata


def canonical_rows_for_track_candidates(
    candidate_track_ids: np.ndarray,
    canonical_track_ids: np.ndarray,
) -> np.ndarray:
    """Resolve physical track identities into a one-row-per-track bank.

    Proposal row ids may come from a multi-prototype bank and are therefore
    intentionally not accepted here.
    """

    candidates = np.asarray(candidate_track_ids, dtype=np.int64)
    canonical = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    if np.unique(canonical).size != len(canonical):
        raise ValueError("canonical track ids must contain one row per physical track")
    order = np.argsort(canonical, kind="stable")
    sorted_tracks = canonical[order]
    output = np.full(candidates.shape, -1, dtype=np.int64)
    valid = candidates >= 0
    if not np.any(valid):
        return output
    if len(sorted_tracks) == 0:
        raise ValueError("candidate tracks cannot be resolved against an empty canonical bank")
    positions = np.searchsorted(sorted_tracks, candidates[valid])
    clipped = np.minimum(positions, len(sorted_tracks) - 1)
    if np.any(positions >= len(sorted_tracks)) or not np.array_equal(
        sorted_tracks[clipped], candidates[valid]
    ):
        raise ValueError("proposal tracks are missing from the canonical landmark bank")
    output[valid] = order[positions]
    return output


@dataclass(frozen=True)
class SupportMapletView:
    track_ids: np.ndarray
    xy: np.ndarray
    descriptors: np.ndarray
    detector_scores: np.ndarray
    reprojection_errors: np.ndarray

    def __post_init__(self) -> None:
        tracks = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        scores = np.asarray(self.detector_scores, dtype=np.float32).reshape(-1)
        errors = np.asarray(self.reprojection_errors, dtype=np.float32).reshape(-1)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(tracks):
            raise ValueError("support descriptors must have one row per track")
        if xy.shape[0] != len(tracks) or scores.shape[0] != len(tracks) or errors.shape[0] != len(tracks):
            raise ValueError("support-view arrays must have matching lengths")
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "descriptors", descriptors)
        object.__setattr__(self, "detector_scores", scores)
        object.__setattr__(self, "reprojection_errors", errors)


class SupportObservationFeatureStore:
    """Join the lightweight geometry index with its external ALIKE cache."""

    def __init__(
        self,
        geometry: SupportObservationGeometryIndex,
        *,
        source_track_ids: np.ndarray,
        source_descriptors: np.ndarray,
        source_detector_scores: np.ndarray,
    ) -> None:
        tracks = np.asarray(source_track_ids, dtype=np.int64).reshape(-1)
        descriptors = np.asarray(source_descriptors, dtype=np.float32)
        scores = np.asarray(source_detector_scores, dtype=np.float32).reshape(-1)
        if descriptors.ndim != 2 or descriptors.shape[0] != len(tracks) or len(scores) != len(tracks):
            raise ValueError("source ALIKE cache arrays have incompatible shapes")
        if len(geometry) and int(np.max(geometry.source_row_indices)) >= len(tracks):
            raise ValueError("geometry index references rows outside the source ALIKE cache")
        if not np.array_equal(tracks[geometry.source_row_indices], geometry.track_ids):
            raise ValueError("geometry index and source ALIKE cache track rows differ")
        normalized, valid = normalize_rows(descriptors)
        if not np.all(valid):
            raise ValueError("source ALIKE cache contains invalid descriptors")
        self.geometry = geometry
        self.source_descriptors = normalized
        self.source_detector_scores = scores

    @property
    def descriptor_dim(self) -> int:
        return int(self.source_descriptors.shape[1])

    def maplet_view(self, image_id: str, requested_track_ids: np.ndarray) -> SupportMapletView:
        geometry_rows = self.geometry.geometry_rows_for_tracks(str(image_id), requested_track_ids)
        valid = geometry_rows >= 0
        selected = geometry_rows[valid]
        source_rows = self.geometry.source_row_indices[selected]
        return SupportMapletView(
            track_ids=self.geometry.track_ids[selected],
            xy=self.geometry.xy[selected],
            descriptors=self.source_descriptors[source_rows],
            detector_scores=self.source_detector_scores[source_rows],
            reprojection_errors=self.geometry.reprojection_errors[selected],
        )


def select_support_maplet_view_indices(
    support_views: Sequence[SupportMapletView],
    *,
    query_anchor_descriptor: np.ndarray,
    anchor_track_id: int,
    expected_maplet_track_count: int,
    top_k: int,
    strategy: str,
    coverage_weight: float = 0.1,
) -> np.ndarray:
    """Select candidate-conditioned support views without query pose or retrieval."""

    if str(strategy) not in {"coverage", "anchor_similarity", "anchor_coverage"}:
        raise ValueError("unsupported support-view selection strategy")
    if int(top_k) <= 0 or int(expected_maplet_track_count) <= 0:
        raise ValueError("top_k and expected maplet track count must be positive")
    count = int(len(support_views))
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    if str(strategy) == "coverage":
        return np.arange(min(int(top_k), count), dtype=np.int64)
    query = np.asarray(query_anchor_descriptor, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 1e-8:
        return np.arange(min(int(top_k), count), dtype=np.int64)
    query = query / query_norm
    scores = np.full((count,), -np.inf, dtype=np.float32)
    for index, view in enumerate(support_views):
        anchor = np.flatnonzero(view.track_ids == int(anchor_track_id))
        if anchor.size != 1 or view.descriptors.shape[1] != len(query):
            continue
        descriptor = view.descriptors[int(anchor[0])]
        descriptor_norm = float(np.linalg.norm(descriptor))
        if descriptor_norm <= 1e-8:
            continue
        score = float((descriptor / descriptor_norm) @ query)
        if str(strategy) == "anchor_coverage":
            score += float(coverage_weight) * float(
                len(view.track_ids) / max(int(expected_maplet_track_count), 1)
            )
        scores[index] = score
    order = np.lexsort((np.arange(count, dtype=np.int64), -scores))
    valid = order[np.isfinite(scores[order])]
    if len(valid) < min(int(top_k), count):
        selected = valid.tolist()
        selected_set = set(int(value) for value in selected)
        selected.extend(index for index in range(count) if index not in selected_set)
        valid = np.asarray(selected, dtype=np.int64)
    return valid[: min(int(top_k), count)].astype(np.int64, copy=False)


def select_dense_query_context_neighborhood(
    *,
    anchor_xy: np.ndarray,
    anchor_descriptor: np.ndarray,
    context_xy: np.ndarray,
    context_descriptors: np.ndarray,
    context_scores: np.ndarray,
    radius_px: float,
    max_points: int,
    duplicate_radius_px: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Select high-resolution local detector context and prepend the anchor."""

    anchor_point = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    anchor_feature = np.asarray(anchor_descriptor, dtype=np.float32).reshape(-1)
    points = np.asarray(context_xy, dtype=np.float32).reshape(-1, 2)
    descriptors = np.asarray(context_descriptors, dtype=np.float32)
    scores = np.asarray(context_scores, dtype=np.float32).reshape(-1)
    if descriptors.ndim != 2 or descriptors.shape[0] != len(points) or len(scores) != len(points):
        raise ValueError("dense query context arrays have incompatible shapes")
    if descriptors.shape[1] != len(anchor_feature):
        raise ValueError("anchor and dense query context descriptor dimensions differ")
    if float(radius_px) <= 0.0 or int(max_points) <= 0 or float(duplicate_radius_px) < 0.0:
        raise ValueError("query context selection parameters are invalid")
    distances2 = np.sum((points - anchor_point[None]) ** 2, axis=1)
    valid = np.all(np.isfinite(points), axis=1) & np.isfinite(scores)
    valid &= distances2 <= float(radius_px) ** 2
    valid &= distances2 > float(duplicate_radius_px) ** 2
    candidates = np.flatnonzero(valid)
    if candidates.size:
        order = np.lexsort((candidates, -scores[candidates]))
        candidates = candidates[order[: max(int(max_points) - 1, 0)]]
    output_xy = np.concatenate([anchor_point[None], points[candidates]], axis=0)
    output_descriptors = np.concatenate([anchor_feature[None], descriptors[candidates]], axis=0)
    return output_xy.astype(np.float32), output_descriptors.astype(np.float32), 0


VIEW_EVIDENCE_NAMES = (
    "view_present",
    "support_observation_count",
    "support_coverage_fraction",
    "anchor_similarity",
    "anchor_support_nn_margin",
    "anchor_is_support_nn",
    "anchor_is_mutual_nn",
    "mutual_count",
    "mutual_similarity_mean",
    "mutual_similarity_min",
    "mutual_margin_mean",
    "context_model_found",
    "context_inlier_count",
    "context_inlier_ratio",
    "context_median_residual_px",
    "anchor_reprojection_residual_px",
    "anchor_geometry_consistent",
    "query_inlier_spread_px",
    "support_inlier_spread_px",
)


def _second_largest(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] < 2:
        return np.full(values.shape[:axis] + values.shape[axis + 1 :], -1.0, dtype=np.float32)
    partitioned = np.partition(values, kth=values.shape[axis] - 2, axis=axis)
    return np.take(partitioned, values.shape[axis] - 2, axis=axis)


def _fit_similarity_ransac(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    threshold_px: float,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Deterministic exhaustive two-point RANSAC for a 2D similarity model."""

    source = np.asarray(source_xy, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float64).reshape(-1, 2)
    count = int(len(source))
    if target.shape != source.shape or count < 2:
        return None, np.zeros((count,), dtype=bool), np.full((count,), np.inf, dtype=np.float32)
    first, second = np.triu_indices(count, k=1)
    source_delta = source[second] - source[first]
    target_delta = target[second] - target[first]
    denominator = np.sum(source_delta * source_delta, axis=1)
    target_norm2 = np.sum(target_delta * target_delta, axis=1)
    valid = (denominator >= 16.0) & (target_norm2 >= 4.0)
    if not np.any(valid):
        return None, np.zeros((count,), dtype=bool), np.full((count,), np.inf, dtype=np.float32)
    first = first[valid]
    source_delta = source_delta[valid]
    target_delta = target_delta[valid]
    denominator = denominator[valid]
    a = np.sum(source_delta * target_delta, axis=1) / denominator
    b = (source_delta[:, 0] * target_delta[:, 1] - source_delta[:, 1] * target_delta[:, 0]) / denominator
    matrices = np.stack(
        [
            np.stack([a, -b], axis=1),
            np.stack([b, a], axis=1),
        ],
        axis=1,
    )
    translation = target[first] - np.einsum("hij,hj->hi", matrices, source[first], optimize=True)
    predicted = np.einsum("hij,mj->hmi", matrices, source, optimize=True) + translation[:, None, :]
    residuals = np.linalg.norm(predicted - target[None], axis=2)
    inliers = residuals <= float(threshold_px)
    counts = np.sum(inliers, axis=1)
    clipped = np.where(inliers, residuals, float(threshold_px) * 2.0)
    costs = np.sum(clipped, axis=1)
    order = np.lexsort((costs, -counts))
    best = int(order[0])
    model = np.concatenate([matrices[best], translation[best, :, None]], axis=1)
    return model.astype(np.float32), inliers[best], residuals[best].astype(np.float32)


def score_maplet_support_view(
    *,
    query_xy: np.ndarray,
    query_descriptors: np.ndarray,
    query_anchor_index: int,
    support_view: SupportMapletView,
    anchor_track_id: int,
    expected_maplet_track_count: int,
    min_similarity: float = 0.20,
    max_tentative_matches: int = 16,
    geometry_threshold_px: float = 8.0,
) -> np.ndarray:
    """Compute deployable identity and local-pattern evidence for one view."""

    query_points = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    query_features, valid_query = normalize_rows(np.asarray(query_descriptors, dtype=np.float32))
    if query_features.shape[0] != len(query_points):
        raise ValueError("query xy and descriptors must have matching lengths")
    anchor_query = int(query_anchor_index)
    if anchor_query < 0 or anchor_query >= len(query_points):
        raise ValueError("query_anchor_index is out of range")
    output = np.zeros((len(VIEW_EVIDENCE_NAMES),), dtype=np.float32)
    if len(support_view.track_ids) == 0 or not bool(valid_query[anchor_query]):
        return output
    anchor_positions = np.flatnonzero(support_view.track_ids == int(anchor_track_id))
    if anchor_positions.size != 1:
        return output
    support_features, valid_support = normalize_rows(support_view.descriptors)
    if not np.all(valid_support):
        raise ValueError("support maplet view contains invalid descriptors")
    support_anchor = int(anchor_positions[0])
    similarities = support_features @ query_features.T
    similarities[:, ~valid_query] = -np.inf
    support_best_query = np.argmax(similarities, axis=1)
    support_best_scores = similarities[np.arange(len(support_features)), support_best_query]
    support_second_scores = _second_largest(similarities, axis=1)
    query_best_support = np.argmax(similarities, axis=0)
    mutual = query_best_support[support_best_query] == np.arange(len(support_features))
    mutual &= np.isfinite(support_best_scores) & (support_best_scores >= float(min_similarity))

    anchor_similarity = float(similarities[support_anchor, anchor_query])
    anchor_margin = anchor_similarity - float(support_second_scores[support_anchor])
    anchor_is_support_nn = int(support_best_query[support_anchor]) == anchor_query
    anchor_is_mutual = anchor_is_support_nn and int(query_best_support[anchor_query]) == support_anchor
    context = np.flatnonzero(mutual)
    context = context[(context != support_anchor) & (support_best_query[context] != anchor_query)]
    if len(context) > int(max_tentative_matches):
        order = np.argsort(-support_best_scores[context], kind="stable")[: int(max_tentative_matches)]
        context = context[order]
    context_query = support_best_query[context]
    context_scores = support_best_scores[context]
    context_margins = context_scores - support_second_scores[context]
    model, inlier_mask, context_residuals = _fit_similarity_ransac(
        support_view.xy[context],
        query_points[context_query],
        threshold_px=float(geometry_threshold_px),
    )
    model_found = model is not None
    if model_found:
        anchor_predicted = (
            model[:, :2].astype(np.float64) @ support_view.xy[support_anchor].astype(np.float64)
            + model[:, 2].astype(np.float64)
        )
        anchor_residual = float(np.linalg.norm(anchor_predicted - query_points[anchor_query]))
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / max(len(context), 1))
        median_residual = float(np.median(context_residuals[inlier_mask])) if inlier_count else np.inf
        query_inliers = query_points[context_query[inlier_mask]]
        support_inliers = support_view.xy[context[inlier_mask]]
        query_spread = float(np.sqrt(np.mean(np.sum((query_inliers - np.mean(query_inliers, axis=0)) ** 2, axis=1)))) if inlier_count else 0.0
        support_spread = float(np.sqrt(np.mean(np.sum((support_inliers - np.mean(support_inliers, axis=0)) ** 2, axis=1)))) if inlier_count else 0.0
    else:
        anchor_residual = np.inf
        inlier_count = 0
        inlier_ratio = 0.0
        median_residual = np.inf
        query_spread = 0.0
        support_spread = 0.0
    values = {
        "view_present": 1.0,
        "support_observation_count": float(len(support_view.track_ids)),
        "support_coverage_fraction": float(len(support_view.track_ids) / max(int(expected_maplet_track_count), 1)),
        "anchor_similarity": anchor_similarity,
        "anchor_support_nn_margin": anchor_margin,
        "anchor_is_support_nn": float(anchor_is_support_nn),
        "anchor_is_mutual_nn": float(anchor_is_mutual),
        "mutual_count": float(len(context)),
        "mutual_similarity_mean": float(np.mean(context_scores)) if len(context) else 0.0,
        "mutual_similarity_min": float(np.min(context_scores)) if len(context) else 0.0,
        "mutual_margin_mean": float(np.mean(context_margins)) if len(context) else 0.0,
        "context_model_found": float(model_found),
        "context_inlier_count": float(inlier_count),
        "context_inlier_ratio": inlier_ratio,
        "context_median_residual_px": min(median_residual, 64.0),
        "anchor_reprojection_residual_px": min(anchor_residual, 128.0),
        "anchor_geometry_consistent": float(model_found and anchor_residual <= float(geometry_threshold_px)),
        "query_inlier_spread_px": min(query_spread, 256.0),
        "support_inlier_spread_px": min(support_spread, 256.0),
    }
    output[:] = np.asarray([values[name] for name in VIEW_EVIDENCE_NAMES], dtype=np.float32)
    return output


def aggregate_maplet_view_evidence(view_features: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Aggregate a variable number of support views without an oracle selector."""

    values = np.asarray(view_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(VIEW_EVIDENCE_NAMES):
        raise ValueError("view_features must have shape (V, F)")
    if values.shape[0] == 0:
        values = np.zeros((1, len(VIEW_EVIDENCE_NAMES)), dtype=np.float32)
    outputs = []
    names: list[str] = []
    for reduction, reduced in (
        ("max", np.max(values, axis=0)),
        ("mean", np.mean(values, axis=0)),
        ("min", np.min(values, axis=0)),
    ):
        outputs.append(reduced.astype(np.float32))
        names.extend(f"{reduction}__{name}" for name in VIEW_EVIDENCE_NAMES)
    return np.concatenate(outputs, axis=0), tuple(names)
