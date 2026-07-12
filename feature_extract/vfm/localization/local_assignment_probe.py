"""Support-view and local-context probes for top-L landmark assignment.

This module intentionally does not train a matcher.  It measures whether a
correct track already present in a global proposal set can be recovered from
real projected support observations and local maplet context.  GT viewing rays
are exposed only through explicitly named oracle strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.local_maplet_matching import LocalMapletSupportIndex
from feature_extract.vfm.map_lifting import TrackObservation
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, normalize_rows


@dataclass(frozen=True)
class UniqueTrackCandidateSet:
    """Fixed-width top-L candidates with one row per physical track."""

    bank_row_indices: np.ndarray
    track_ids: np.ndarray
    prototype_ids: np.ndarray
    coarse_scores: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.bank_row_indices, dtype=np.int64)
        tracks = np.asarray(self.track_ids, dtype=np.int64)
        prototypes = np.asarray(self.prototype_ids, dtype=np.int64)
        scores = np.asarray(self.coarse_scores, dtype=np.float32)
        if rows.ndim != 2:
            raise ValueError("candidate arrays must have shape (Q, L)")
        if tracks.shape != rows.shape or prototypes.shape != rows.shape or scores.shape != rows.shape:
            raise ValueError("all candidate arrays must have the same shape")
        valid = rows >= 0
        if np.any(tracks[valid] < 0) or np.any(prototypes[valid] < 0):
            raise ValueError("valid candidates must have non-negative track/prototype ids")
        for row_tracks, row_valid in zip(tracks, valid):
            values = row_tracks[row_valid]
            if np.unique(values).size != values.size:
                raise ValueError("candidate rows must contain unique physical tracks")
        object.__setattr__(self, "bank_row_indices", rows)
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "prototype_ids", prototypes)
        object.__setattr__(self, "coarse_scores", scores)

    @property
    def query_count(self) -> int:
        return int(self.bank_row_indices.shape[0])

    @property
    def top_l(self) -> int:
        return int(self.bank_row_indices.shape[1])

    @property
    def valid_mask(self) -> np.ndarray:
        return self.bank_row_indices >= 0


@dataclass(frozen=True)
class ProjectedSupportDescriptor:
    track_id: int
    image_id: str
    descriptor: np.ndarray
    viewing_ray: np.ndarray | None
    reprojection_error: float

    def __post_init__(self) -> None:
        descriptor = np.asarray(self.descriptor, dtype=np.float32).reshape(-1)
        ray = None if self.viewing_ray is None else np.asarray(self.viewing_ray, dtype=np.float64).reshape(3)
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "image_id", str(self.image_id))
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "viewing_ray", ray)
        object.__setattr__(self, "reprojection_error", float(self.reprojection_error))


@dataclass(frozen=True)
class AssignmentProbeScores:
    strategy_scores: Mapping[str, np.ndarray]
    maplet_support_counts: np.ndarray
    all_support_counts: np.ndarray

    def __post_init__(self) -> None:
        maplet_counts = np.asarray(self.maplet_support_counts, dtype=np.int64)
        all_counts = np.asarray(self.all_support_counts, dtype=np.int64)
        if maplet_counts.ndim != 2 or all_counts.shape != maplet_counts.shape:
            raise ValueError("support counts must have shape (Q, L)")
        scores = {str(name): np.asarray(values, dtype=np.float32) for name, values in self.strategy_scores.items()}
        if not scores:
            raise ValueError("at least one assignment strategy is required")
        for name, values in scores.items():
            if values.shape != maplet_counts.shape:
                raise ValueError(f"strategy {name} must have shape (Q, L)")
        object.__setattr__(self, "strategy_scores", scores)
        object.__setattr__(self, "maplet_support_counts", maplet_counts)
        object.__setattr__(self, "all_support_counts", all_counts)


def _resolved_device(device: str) -> torch.device:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def retrieve_unique_track_candidates_exact(
    query_descriptors: np.ndarray,
    landmark_index: LandmarkMapIndex,
    *,
    top_l: int,
    device: str = "cuda",
    batch_size: int = 64,
) -> UniqueTrackCandidateSet:
    """Exact cosine top-L retrieval, collapsing prototype rows into tracks."""

    queries = np.asarray(query_descriptors, dtype=np.float32)
    if queries.ndim != 2:
        raise ValueError("query_descriptors must have shape (Q, C)")
    if queries.shape[1] != int(landmark_index.feature_dim):
        raise ValueError("query and landmark descriptor dimensions must match")
    if int(top_l) <= 0 or int(batch_size) <= 0:
        raise ValueError("top_l and batch_size must be positive")

    normalized_queries, valid_queries = normalize_rows(queries)
    normalized_bank, valid_bank = normalize_rows(landmark_index.features)
    valid_bank_rows = np.flatnonzero(valid_bank)
    output_rows = np.full((queries.shape[0], int(top_l)), -1, dtype=np.int64)
    output_tracks = np.full_like(output_rows, -1)
    output_prototypes = np.full_like(output_rows, -1)
    output_scores = np.full(output_rows.shape, -np.inf, dtype=np.float32)
    if valid_bank_rows.size == 0 or queries.shape[0] == 0:
        return UniqueTrackCandidateSet(output_rows, output_tracks, output_prototypes, output_scores)

    valid_track_ids = landmark_index.track_ids[valid_bank_rows]
    _unique_tracks, prototype_counts = np.unique(valid_track_ids, return_counts=True)
    max_prototypes_per_track = max(1, int(np.max(prototype_counts)))
    search_k = min(int(valid_bank_rows.size), int(top_l) * max_prototypes_per_track)
    torch_device = _resolved_device(str(device))
    bank_tensor = torch.as_tensor(
        normalized_bank[valid_bank_rows],
        dtype=torch.float32,
        device=torch_device,
    )
    with torch.no_grad():
        for start in range(0, int(queries.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(queries.shape[0]))
            query_tensor = torch.as_tensor(
                normalized_queries[start:end],
                dtype=torch.float32,
                device=torch_device,
            )
            similarities = query_tensor @ bank_tensor.T
            top_scores, top_positions = torch.topk(
                similarities,
                k=int(search_k),
                dim=1,
                largest=True,
                sorted=True,
            )
            score_rows = top_scores.cpu().numpy().astype(np.float32, copy=False)
            bank_positions = top_positions.cpu().numpy().astype(np.int64, copy=False)
            for local_row, query_row in enumerate(range(start, end)):
                if not bool(valid_queries[query_row]):
                    continue
                seen_tracks: set[int] = set()
                output_column = 0
                for position, score in zip(bank_positions[local_row], score_rows[local_row]):
                    bank_row = int(valid_bank_rows[int(position)])
                    track_id = int(landmark_index.track_ids[bank_row])
                    if track_id in seen_tracks:
                        continue
                    seen_tracks.add(track_id)
                    output_rows[query_row, output_column] = bank_row
                    output_tracks[query_row, output_column] = track_id
                    output_prototypes[query_row, output_column] = int(landmark_index.prototype_ids[bank_row])
                    output_scores[query_row, output_column] = float(score)
                    output_column += 1
                    if output_column >= int(top_l):
                        break
    del bank_tensor
    return UniqueTrackCandidateSet(output_rows, output_tracks, output_prototypes, output_scores)


def projected_support_descriptors_by_track(
    sampled_observations: Sequence[TrackObservation],
    source_observations: Sequence[ColmapTrackObservation],
) -> dict[int, tuple[ProjectedSupportDescriptor, ...]]:
    """Join projected descriptors back to deterministic SfM view metadata."""

    metadata_by_key = {
        (int(observation.track_id), str(observation.image_id)): observation
        for observation in source_observations
    }
    grouped: dict[int, list[ProjectedSupportDescriptor]] = {}
    for sampled in sampled_observations:
        key = (int(sampled.track_id), str(sampled.image_id))
        source = metadata_by_key.get(key)
        if source is None:
            raise ValueError(f"projected support observation has no SfM metadata: {key}")
        grouped.setdefault(int(sampled.track_id), []).append(
            ProjectedSupportDescriptor(
                track_id=int(sampled.track_id),
                image_id=str(sampled.image_id),
                descriptor=np.asarray(sampled.feature, dtype=np.float32),
                viewing_ray=source.viewing_ray,
                reprojection_error=float(source.reprojection_error),
            )
        )
    return {
        int(track_id): tuple(sorted(rows, key=lambda row: (row.image_id, row.reprojection_error)))
        for track_id, rows in grouped.items()
    }


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return np.zeros_like(values)
    return (values / norm).astype(np.float32, copy=False)


def _mean_descriptor(descriptors: np.ndarray) -> np.ndarray:
    if descriptors.shape[0] == 0:
        return np.zeros((descriptors.shape[1],), dtype=np.float32)
    normalized, valid = normalize_rows(descriptors)
    if not np.any(valid):
        return np.zeros((descriptors.shape[1],), dtype=np.float32)
    return _unit(np.mean(normalized[valid], axis=0))


def _selected_maplet_support_rows(
    track_id: int,
    all_rows: Sequence[ProjectedSupportDescriptor],
    maplet_index: LocalMapletSupportIndex,
    maplet_row_by_track: Mapping[int, int],
) -> tuple[ProjectedSupportDescriptor, ...]:
    maplet_row = maplet_row_by_track.get(int(track_id))
    if maplet_row is None:
        return tuple(all_rows)
    rows_by_image = {str(row.image_id): row for row in all_rows}
    selected = tuple(
        rows_by_image[image_id]
        for image_id in maplet_index.support_views(int(maplet_row))
        if image_id in rows_by_image
    )
    return selected if selected else tuple(all_rows)


def score_support_assignment_strategies(
    *,
    query_descriptors: np.ndarray,
    query_context_descriptors: np.ndarray,
    query_viewing_rays: np.ndarray,
    candidates: UniqueTrackCandidateSet,
    landmark_index: LandmarkMapIndex,
    maplet_index: LocalMapletSupportIndex,
    support_by_track: Mapping[int, Sequence[ProjectedSupportDescriptor]],
) -> AssignmentProbeScores:
    """Score fixed proposal groups using support observations and maplet context."""

    query_features, valid_queries = normalize_rows(np.asarray(query_descriptors, dtype=np.float32))
    query_context, valid_context = normalize_rows(np.asarray(query_context_descriptors, dtype=np.float32))
    query_rays = np.asarray(query_viewing_rays, dtype=np.float64).reshape(-1, 3)
    if query_features.shape[0] != candidates.query_count:
        raise ValueError("query descriptors and candidates must contain the same Q")
    if query_context.shape != query_features.shape:
        raise ValueError("query context descriptors must match query descriptor shape")
    if query_rays.shape[0] != candidates.query_count:
        raise ValueError("query_viewing_rays must contain one ray per query descriptor")

    shape = candidates.coarse_scores.shape
    strategy_names = (
        "coarse_prototype",
        "maplet_context",
        "maplet_support_first",
        "maplet_support_best",
        "maplet_support_mean",
        "maplet_support_top2_mean",
        "maplet_support_top4_mean",
        "all_support_best",
        "all_support_mean",
        "all_support_top2_mean",
        "all_support_top4_mean",
        "all_support_logmeanexp_tau0p05",
        "oracle_view_angle_best",
        "coarse_support_best_75_25",
        "coarse_support_best_50_50",
        "coarse_support_best_25_75",
        "coarse_all_support_best_75_25",
        "coarse_all_support_best_50_50",
        "coarse_all_support_best_25_75",
        "coarse_all_support_logmeanexp_50_50",
        "coarse_maplet_context_75_25",
    )
    scores = {name: np.full(shape, -np.inf, dtype=np.float32) for name in strategy_names}
    maplet_counts = np.zeros(shape, dtype=np.int64)
    all_counts = np.zeros(shape, dtype=np.int64)
    maplet_row_by_track = {
        int(track_id): int(row) for row, track_id in enumerate(maplet_index.anchor_track_ids.tolist())
    }
    maplet_context, valid_maplet_context = normalize_rows(maplet_index.maplets.context_features)

    for query_row in range(candidates.query_count):
        if not bool(valid_queries[query_row]):
            continue
        query_feature = query_features[query_row]
        query_ray = query_rays[query_row]
        query_ray_norm = float(np.linalg.norm(query_ray))
        if query_ray_norm > 1e-12:
            query_ray = query_ray / query_ray_norm
        for column in range(candidates.top_l):
            bank_row = int(candidates.bank_row_indices[query_row, column])
            if bank_row < 0:
                continue
            track_id = int(candidates.track_ids[query_row, column])
            coarse = float(candidates.coarse_scores[query_row, column])
            scores["coarse_prototype"][query_row, column] = coarse

            maplet_row = maplet_row_by_track.get(track_id)
            context_score = coarse
            if (
                maplet_row is not None
                and bool(valid_context[query_row])
                and bool(valid_maplet_context[int(maplet_row)])
            ):
                context_score = float(query_context[query_row] @ maplet_context[int(maplet_row)])
            scores["maplet_context"][query_row, column] = context_score
            scores["coarse_maplet_context_75_25"][query_row, column] = 0.75 * coarse + 0.25 * context_score

            all_rows = tuple(support_by_track.get(track_id, ()))
            selected_rows = _selected_maplet_support_rows(
                track_id,
                all_rows,
                maplet_index,
                maplet_row_by_track,
            )
            all_counts[query_row, column] = int(len(all_rows))
            maplet_counts[query_row, column] = int(len(selected_rows))
            if not selected_rows:
                for name in (
                    "maplet_support_first",
                    "maplet_support_best",
                    "maplet_support_mean",
                    "maplet_support_top2_mean",
                    "maplet_support_top4_mean",
                    "all_support_best",
                    "all_support_mean",
                    "all_support_top2_mean",
                    "all_support_top4_mean",
                    "all_support_logmeanexp_tau0p05",
                    "oracle_view_angle_best",
                    "coarse_support_best_75_25",
                    "coarse_support_best_50_50",
                    "coarse_support_best_25_75",
                    "coarse_all_support_best_75_25",
                    "coarse_all_support_best_50_50",
                    "coarse_all_support_best_25_75",
                    "coarse_all_support_logmeanexp_50_50",
                ):
                    scores[name][query_row, column] = coarse
                continue

            selected_descriptors = np.stack([_unit(row.descriptor) for row in selected_rows], axis=0)
            support_similarities = selected_descriptors @ query_feature
            first_score = float(support_similarities[0])
            best_score = float(np.max(support_similarities))
            mean_score = float(_mean_descriptor(selected_descriptors) @ query_feature)
            score_by_top_m: dict[int, float] = {}
            order = np.argsort(-support_similarities, kind="mergesort")
            for top_m in (2, 4):
                selected = selected_descriptors[order[: min(int(top_m), len(order))]]
                score_by_top_m[top_m] = float(_mean_descriptor(selected) @ query_feature)

            all_best_score = best_score
            all_mean_score = mean_score
            all_top2_score = score_by_top_m[2]
            all_top4_score = score_by_top_m[4]
            all_logmeanexp_score = best_score
            if all_rows:
                all_descriptors = np.stack([_unit(row.descriptor) for row in all_rows], axis=0)
                all_similarities = all_descriptors @ query_feature
                all_order = np.argsort(-all_similarities, kind="mergesort")
                all_best_score = float(all_similarities[all_order[0]])
                all_mean_score = float(_mean_descriptor(all_descriptors) @ query_feature)
                all_top2_score = float(
                    _mean_descriptor(all_descriptors[all_order[: min(2, len(all_order))]]) @ query_feature
                )
                all_top4_score = float(
                    _mean_descriptor(all_descriptors[all_order[: min(4, len(all_order))]]) @ query_feature
                )
                temperature = 0.05
                scaled = all_similarities.astype(np.float64) / temperature
                maximum = float(np.max(scaled))
                all_logmeanexp_score = float(
                    temperature * (maximum + np.log(np.mean(np.exp(scaled - maximum))))
                )

            view_angle_score = first_score
            ray_rows = [row for row in selected_rows if row.viewing_ray is not None]
            if query_ray_norm > 1e-12 and ray_rows:
                support_rays = np.stack([np.asarray(row.viewing_ray, dtype=np.float64) for row in ray_rows], axis=0)
                support_rays /= np.maximum(np.linalg.norm(support_rays, axis=1, keepdims=True), 1e-12)
                view_row = int(np.argmax(support_rays @ query_ray))
                view_angle_score = float(_unit(ray_rows[view_row].descriptor) @ query_feature)

            scores["maplet_support_first"][query_row, column] = first_score
            scores["maplet_support_best"][query_row, column] = best_score
            scores["maplet_support_mean"][query_row, column] = mean_score
            scores["maplet_support_top2_mean"][query_row, column] = score_by_top_m[2]
            scores["maplet_support_top4_mean"][query_row, column] = score_by_top_m[4]
            scores["all_support_best"][query_row, column] = all_best_score
            scores["all_support_mean"][query_row, column] = all_mean_score
            scores["all_support_top2_mean"][query_row, column] = all_top2_score
            scores["all_support_top4_mean"][query_row, column] = all_top4_score
            scores["all_support_logmeanexp_tau0p05"][query_row, column] = all_logmeanexp_score
            scores["oracle_view_angle_best"][query_row, column] = view_angle_score
            scores["coarse_support_best_75_25"][query_row, column] = 0.75 * coarse + 0.25 * best_score
            scores["coarse_support_best_50_50"][query_row, column] = 0.50 * coarse + 0.50 * best_score
            scores["coarse_support_best_25_75"][query_row, column] = 0.25 * coarse + 0.75 * best_score
            scores["coarse_all_support_best_75_25"][query_row, column] = 0.75 * coarse + 0.25 * all_best_score
            scores["coarse_all_support_best_50_50"][query_row, column] = 0.50 * coarse + 0.50 * all_best_score
            scores["coarse_all_support_best_25_75"][query_row, column] = 0.25 * coarse + 0.75 * all_best_score
            scores["coarse_all_support_logmeanexp_50_50"][query_row, column] = (
                0.50 * coarse + 0.50 * all_logmeanexp_score
            )

    return AssignmentProbeScores(scores, maplet_counts, all_counts)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    targets = targets[valid]
    values = values[valid]
    positive_count = int(np.sum(targets))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-values, kind="mergesort")
    sorted_targets = targets[order]
    precision = np.cumsum(sorted_targets) / np.arange(1, sorted_targets.size + 1)
    return float(np.sum(precision[sorted_targets]) / positive_count)


def score_candidate_support_feature_pooling(
    *,
    query_features: np.ndarray,
    candidates: UniqueTrackCandidateSet,
    support_track_ids: np.ndarray,
    support_features: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Pool arbitrary same-space local features over each candidate track's views."""

    query, valid_query = normalize_rows(np.asarray(query_features, dtype=np.float32))
    tracks = np.asarray(support_track_ids, dtype=np.int64).reshape(-1)
    features = np.asarray(support_features, dtype=np.float32)
    if query.shape[0] != candidates.query_count:
        raise ValueError("query_features must contain one row per candidate group")
    if features.ndim != 2 or features.shape[0] != tracks.shape[0]:
        raise ValueError("support_features must have shape (N, C) with one support_track_id per row")
    if features.shape[1] != query.shape[1]:
        raise ValueError("query and support feature dimensions must match")
    if not str(prefix):
        raise ValueError("prefix must not be empty")
    support, valid_support = normalize_rows(features)
    if tracks.size and np.any(tracks[1:] < tracks[:-1]):
        order = np.argsort(tracks, kind="stable")
        tracks = tracks[order]
        support = support[order]
        valid_support = valid_support[order]
    unique_tracks, starts, counts = np.unique(tracks, return_index=True, return_counts=True)
    names = (
        f"{prefix}_support_best",
        f"{prefix}_support_mean",
        f"{prefix}_support_top2_mean",
        f"{prefix}_support_top4_mean",
        f"{prefix}_support_logmeanexp_tau0p05",
    )
    output = {name: np.full(candidates.coarse_scores.shape, -np.inf, dtype=np.float32) for name in names}
    flat_valid = candidates.valid_mask.reshape(-1)
    flat_positions = np.flatnonzero(flat_valid)
    if flat_positions.size == 0 or unique_tracks.size == 0:
        return output
    top_l = int(candidates.top_l)
    edge_query_rows = flat_positions // top_l
    edge_track_ids = candidates.track_ids.reshape(-1)[flat_positions]
    support_positions = np.searchsorted(unique_tracks, edge_track_ids)
    support_positions_clipped = np.minimum(support_positions, max(len(unique_tracks) - 1, 0))
    edge_has_support = support_positions < len(unique_tracks)
    edge_has_support &= unique_tracks[support_positions_clipped] == edge_track_ids
    edge_has_support &= valid_query[edge_query_rows]
    active_edges = np.flatnonzero(edge_has_support)
    if active_edges.size == 0:
        return output
    edge_starts = starts[support_positions_clipped]
    edge_counts = counts[support_positions_clipped]
    flat_outputs = {name: values.reshape(-1) for name, values in output.items()}

    def normalized_dot(sums: np.ndarray, descriptor_rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        normalized = np.divide(sums, np.maximum(norms, 1e-12), out=np.zeros_like(sums), where=norms > 1e-12)
        return np.einsum("bd,bd->b", normalized, descriptor_rows, optimize=True).astype(np.float32)

    # Similar track lengths share one padded gather, avoiding a Python loop per proposal edge.
    bucket_limits = (2, 4, 8, 16, 32, 64, 128, 256, 512)
    lower = 0
    for upper in bucket_limits:
        bucket = active_edges[(edge_counts[active_edges] > lower) & (edge_counts[active_edges] <= upper)]
        lower = upper
        if bucket.size == 0:
            continue
        maximum_count = int(np.max(edge_counts[bucket]))
        feature_dim = int(support.shape[1])
        chunk_size = max(64, min(8192, int(4_000_000 / max(maximum_count * feature_dim, 1))))
        offsets = np.arange(maximum_count, dtype=np.int64)[None, :]
        for chunk_start in range(0, len(bucket), chunk_size):
            selected_edges = bucket[chunk_start : chunk_start + chunk_size]
            selected_counts = edge_counts[selected_edges]
            indices = edge_starts[selected_edges, None] + offsets
            row_mask = offsets < selected_counts[:, None]
            indices = np.minimum(indices, max(len(support) - 1, 0))
            row_mask &= valid_support[indices]
            support_rows = support[indices]
            descriptor_rows = query[edge_query_rows[selected_edges]]
            similarities = np.einsum("bmd,bd->bm", support_rows, descriptor_rows, optimize=True)
            similarities = similarities.astype(np.float32, copy=False)
            similarities[~row_mask] = -np.inf
            best = np.max(similarities, axis=1)
            valid_counts = np.maximum(np.sum(row_mask, axis=1), 1).astype(np.float32)
            all_sums = np.einsum(
                "bmd,bm->bd",
                support_rows,
                row_mask.astype(np.float32),
                optimize=True,
            )
            mean_scores = normalized_dot(all_sums, descriptor_rows)

            pooled_scores: dict[int, np.ndarray] = {}
            for pool_size in (2, 4):
                effective_size = min(int(pool_size), maximum_count)
                top_indices = np.argpartition(
                    -similarities,
                    kth=effective_size - 1,
                    axis=1,
                )[:, :effective_size]
                batch_indices = np.arange(len(selected_edges), dtype=np.int64)[:, None]
                selected_rows = support_rows[batch_indices, top_indices]
                selected_mask = row_mask[batch_indices, top_indices]
                selected_sums = np.einsum(
                    "bkd,bk->bd",
                    selected_rows,
                    selected_mask.astype(np.float32),
                    optimize=True,
                )
                pooled_scores[pool_size] = normalized_dot(selected_sums, descriptor_rows)

            temperature = 0.05
            exponent = np.zeros_like(similarities, dtype=np.float64)
            exponent[row_mask] = np.exp(
                (similarities[row_mask].astype(np.float64) - np.repeat(best, maximum_count)[row_mask.reshape(-1)])
                / temperature
            )
            logmeanexp = best.astype(np.float64) + temperature * np.log(
                np.sum(exponent, axis=1) / valid_counts.astype(np.float64)
            )
            target_positions = flat_positions[selected_edges]
            flat_outputs[f"{prefix}_support_best"][target_positions] = best
            flat_outputs[f"{prefix}_support_mean"][target_positions] = mean_scores
            flat_outputs[f"{prefix}_support_top2_mean"][target_positions] = pooled_scores[2]
            flat_outputs[f"{prefix}_support_top4_mean"][target_positions] = pooled_scores[4]
            flat_outputs[f"{prefix}_support_logmeanexp_tau0p05"][target_positions] = logmeanexp.astype(np.float32)
    return output


def summarize_assignment_strategy(
    *,
    candidates: UniqueTrackCandidateSet,
    correct_track_ids: Sequence[int],
    query_ids: Sequence[str],
    scores: np.ndarray,
    top_ks: Sequence[int] = (1, 5),
) -> dict[str, object]:
    correct = np.asarray(correct_track_ids, dtype=np.int64).reshape(-1)
    ids = tuple(str(item) for item in query_ids)
    values = np.asarray(scores, dtype=np.float32)
    if correct.shape[0] != candidates.query_count or len(ids) != candidates.query_count:
        raise ValueError("correct_track_ids and query_ids must contain one value per query row")
    if values.shape != candidates.coarse_scores.shape:
        raise ValueError("scores must match candidate shape")

    ranks: list[int | None] = []
    proposal_present: list[bool] = []
    for row in range(candidates.query_count):
        valid_columns = np.flatnonzero(candidates.valid_mask[row] & np.isfinite(values[row]))
        present = bool(np.any(candidates.track_ids[row, valid_columns] == int(correct[row])))
        proposal_present.append(present)
        if not present or valid_columns.size == 0:
            ranks.append(None)
            continue
        order = valid_columns[np.argsort(-values[row, valid_columns], kind="mergesort")]
        correct_positions = np.flatnonzero(candidates.track_ids[row, order] == int(correct[row]))
        ranks.append(None if correct_positions.size == 0 else int(correct_positions[0]) + 1)

    valid_pairs = candidates.valid_mask & np.isfinite(values)
    pair_labels = candidates.track_ids == correct[:, None]
    positives = values[valid_pairs & pair_labels]
    negatives = values[valid_pairs & ~pair_labels]
    summary: dict[str, object] = {
        "sample_count": int(candidates.query_count),
        "proposal_present_count": int(np.sum(proposal_present)),
        "proposal_recall": float(np.mean(proposal_present)) if proposal_present else 0.0,
        "mean_reciprocal_rank": float(np.mean([0.0 if rank is None else 1.0 / rank for rank in ranks])) if ranks else 0.0,
        "conditional_mean_reciprocal_rank": (
            0.0
            if not any(proposal_present)
            else float(np.mean([1.0 / rank for rank, present in zip(ranks, proposal_present) if present and rank is not None]))
        ),
        "pair_correct_average_precision": binary_average_precision(pair_labels[valid_pairs], values[valid_pairs]),
        "positive_score_mean": None if positives.size == 0 else float(np.mean(positives)),
        "negative_score_mean": None if negatives.size == 0 else float(np.mean(negatives)),
        "median_correct_rank_when_present": (
            None if not any(rank is not None for rank in ranks) else float(np.median([rank for rank in ranks if rank is not None]))
        ),
    }
    unique_query_ids = sorted(set(ids))
    for top_k in top_ks:
        k = int(top_k)
        if k <= 0:
            raise ValueError("top_ks must be positive")
        successes = [rank is not None and rank <= k for rank in ranks]
        conditional = [success for success, present in zip(successes, proposal_present) if present]
        summary[f"recall_at_{k}"] = 0.0 if not successes else float(np.mean(successes))
        summary[f"conditional_recall_at_{k}"] = 0.0 if not conditional else float(np.mean(conditional))
        macro = []
        for query_id in unique_query_ids:
            positions = [index for index, value in enumerate(ids) if value == query_id]
            macro.append(float(np.mean([successes[index] for index in positions])))
        summary[f"macro_query_recall_at_{k}"] = 0.0 if not macro else float(np.mean(macro))
    return summary


def proposal_recall_summary(
    candidates: UniqueTrackCandidateSet,
    correct_track_ids: Sequence[int],
    *,
    top_ks: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, object]:
    correct = np.asarray(correct_track_ids, dtype=np.int64).reshape(-1)
    if correct.shape[0] != candidates.query_count:
        raise ValueError("correct_track_ids must contain one value per query row")
    output: dict[str, object] = {
        "sample_count": int(candidates.query_count),
        "mean_unique_candidate_count": float(np.mean(np.sum(candidates.valid_mask, axis=1))) if candidates.query_count else 0.0,
    }
    for top_k in top_ks:
        k = min(int(top_k), candidates.top_l)
        if k <= 0:
            raise ValueError("top_ks must be positive")
        found = np.any(candidates.track_ids[:, :k] == correct[:, None], axis=1)
        output[f"recall_at_{int(top_k)}"] = float(np.mean(found)) if found.size else 0.0
    return output
