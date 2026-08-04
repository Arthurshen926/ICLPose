"""Non-redundant VFM-region to 2DGS-maplet surface alignment.

V8 treated every fixed RADIO sample as an independent observation and matched
its support box to the centre and full size of a physical maplet.  Neither is
the intended observation model.  This module keeps V8.0 immutable and defines
the corrected V8.1 likelihood:

* overlapping, appearance-consistent supports form one evidence group;
* a regional support is contained by a projected maplet footprint rather than
  forced to be co-centred and equal-sized;
* query relations are undirected and occur exactly once;
* latent maplet responsibilities are subject to projected-area capacity;
* retrieved maplets predicted visible by a pose must be explained by a query
  group (the reverse visibility direction).

Only the canonical RADIO-derived posterior and geometry enter this runtime
module.  It stores no image, downstream teacher embedding, point match or PnP
correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
)
from feature_extract.vfm.localization_v6.se3_update import se3_exp
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    PhysicalMapletGraph,
    _project_maplet_geometry,
    _rectangle_iou,
)


@dataclass(frozen=True)
class RegionEvidenceGraph:
    """Grouped query observations with a conservative posterior mixture."""

    xy: np.ndarray
    extent: np.ndarray
    candidate_rows: np.ndarray
    candidate_probabilities: np.ndarray
    null_probability: np.ndarray
    member_offsets: np.ndarray
    member_indices: np.ndarray
    representative_indices: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    # dx, dy, distance, log support-scale ratio, overlap IoU, scale class
    edge_features: np.ndarray
    grouping_feature_similarity: float

    @property
    def group_count(self) -> int:
        return int(self.xy.shape[0])


def _normalise_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def _support_overlap_matrix(xy: np.ndarray, extent: np.ndarray) -> np.ndarray:
    count = int(xy.shape[0])
    result = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            value = _rectangle_iou(
                xy[first], extent[first], xy[second], extent[second]
            )
            result[first, second] = result[second, first] = value
    return result


def _group_supports(
    xy: np.ndarray,
    extent: np.ndarray,
    descriptors: np.ndarray,
    confidence: np.ndarray,
) -> tuple[list[list[int]], float]:
    """Greedy medoid grouping without transitive connected-component chains."""

    descriptor = _normalise_rows(descriptors)
    similarity = descriptor @ descriptor.T
    overlap = _support_overlap_matrix(xy, extent)
    overlapping = similarity[np.triu(overlap >= 0.15, k=1)]
    if overlapping.size:
        # The threshold is inferred per image from pairs that actually share
        # support.  Clipping prevents a flat descriptor field from collapsing
        # the image and prevents tiny numerical differences from disabling all
        # grouping.
        feature_threshold = float(
            np.clip(np.quantile(overlapping, 0.60), 0.72, 0.94)
        )
    else:
        feature_threshold = 0.82
    order = np.argsort(-np.asarray(confidence), kind="stable")
    groups: list[list[int]] = []
    medoids: list[int] = []
    for index in order.tolist():
        winner = -1
        winner_score = -np.inf
        for group_index, medoid in enumerate(medoids):
            # Compare against the medoid rather than any member: this avoids a
            # chain of adjacent patches becoming one facade-sized component.
            centre_distance = np.max(
                np.abs(xy[index] - xy[medoid])
                / np.maximum(extent[index] + extent[medoid], 1e-6)
            )
            shared_support = (
                overlap[index, medoid] >= 0.15 or centre_distance <= 0.70
            )
            if not shared_support or similarity[index, medoid] < feature_threshold:
                continue
            score = float(similarity[index, medoid] + overlap[index, medoid])
            if score > winner_score:
                winner = group_index
                winner_score = score
        if winner < 0:
            groups.append([int(index)])
            medoids.append(int(index))
        else:
            groups[winner].append(int(index))
            members = np.asarray(groups[winner], dtype=np.int64)
            member_weight = np.maximum(confidence[members], 1e-3)
            cohesion = np.sum(
                similarity[np.ix_(members, members)] * member_weight[None],
                axis=1,
            )
            medoids[winner] = int(members[int(np.argmax(cohesion))])
    # Stable image-space order makes relation construction deterministic.
    ordering = np.argsort([min(group) for group in groups], kind="stable")
    return [groups[int(value)] for value in ordering], feature_threshold


def _aggregate_group_posterior(
    retrieval: MapletRetrievalResult,
    members: list[int],
    row_by_id: dict[int, int],
    width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    group_confidence = np.asarray(
        [1.0 - float(retrieval.groups[index].null_probability) for index in members],
        dtype=np.float64,
    )
    group_weight = np.maximum(group_confidence, 0.05)
    group_weight /= np.sum(group_weight)
    probability_by_row: dict[int, float] = {}
    for weight, index in zip(group_weight.tolist(), members):
        observation = retrieval.groups[index]
        for maplet_id, probability in zip(
            observation.maplet_ids.tolist(), observation.probabilities.tolist()
        ):
            row = row_by_id.get(int(maplet_id))
            if row is None:
                continue
            probability_by_row[row] = probability_by_row.get(row, 0.0) + (
                float(weight) * float(probability)
            )
    ranked = sorted(probability_by_row.items(), key=lambda value: (-value[1], value[0]))
    selected = ranked[:width]
    rows = np.full((width,), -1, dtype=np.int32)
    probability = np.zeros((width,), dtype=np.float32)
    if selected:
        rows[: len(selected)] = [value[0] for value in selected]
        probability[: len(selected)] = [value[1] for value in selected]
    # Averaging complete categorical distributions conserves null/omitted and
    # transfers only capacity-truncated candidate mass to the null branch.
    null = float(np.clip(1.0 - np.sum(probability), 0.0, 1.0))
    return rows, probability, null


def build_region_evidence_graph(
    retrieval: MapletRetrievalResult,
    physical: PhysicalMapletGraph,
    descriptors: np.ndarray,
    *,
    image_size_wh: tuple[int, int],
    candidates_per_group: int = 12,
    local_neighbors: int = 5,
    medium_neighbors: int = 3,
    long_neighbors: int = 2,
) -> RegionEvidenceGraph:
    """Convert fixed supports into non-redundant VFM evidence instances."""

    observations = retrieval.groups
    count = len(observations)
    descriptor = np.asarray(descriptors, dtype=np.float64)
    if descriptor.ndim != 2 or descriptor.shape[0] != count:
        raise ValueError("one VFM descriptor is required per retrieval support")
    size = np.asarray(image_size_wh, dtype=np.float64)
    raw_xy = np.asarray([value.query_region_xy for value in observations]) / size
    raw_extent = np.asarray([value.query_region_extent for value in observations]) / size
    confidence = np.asarray(
        [1.0 - float(value.null_probability) for value in observations],
        dtype=np.float64,
    )
    groups, threshold = _group_supports(
        raw_xy, raw_extent, descriptor, confidence
    )
    width = max(int(candidates_per_group), 1)
    rows = np.full((len(groups), width), -1, dtype=np.int32)
    probability = np.zeros((len(groups), width), dtype=np.float32)
    null = np.ones((len(groups),), dtype=np.float32)
    xy = np.zeros((len(groups), 2), dtype=np.float64)
    extent = np.zeros((len(groups), 2), dtype=np.float64)
    representative = np.zeros((len(groups),), dtype=np.int32)
    row_by_id = {
        int(value): int(row) for row, value in enumerate(physical.maplet_ids.tolist())
    }
    flat_members: list[int] = []
    offsets = [0]
    normalised_descriptor = _normalise_rows(descriptor)
    for group_index, members in enumerate(groups):
        member_rows = np.asarray(members, dtype=np.int64)
        member_weight = np.maximum(confidence[member_rows], 0.05)
        cohesion = np.sum(
            (normalised_descriptor[member_rows] @ normalised_descriptor[member_rows].T)
            * member_weight[None],
            axis=1,
        )
        representative[group_index] = int(member_rows[int(np.argmax(cohesion))])
        low = np.min(raw_xy[member_rows] - raw_extent[member_rows], axis=0)
        high = np.max(raw_xy[member_rows] + raw_extent[member_rows], axis=0)
        xy[group_index] = 0.5 * (low + high)
        extent[group_index] = np.maximum(0.5 * (high - low), 1.0 / size)
        rows[group_index], probability[group_index], null[group_index] = (
            _aggregate_group_posterior(retrieval, members, row_by_id, width)
        )
        flat_members.extend(members)
        offsets.append(len(flat_members))

    group_count = len(groups)
    pair_distance = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(pair_distance, np.inf)
    order = np.argsort(pair_distance, axis=1, kind="stable")
    local_end = min(max(int(local_neighbors), 1), max(group_count - 1, 0))
    medium_end = min(local_end + max(int(medium_neighbors), 0), max(group_count - 1, 0))
    long_count = min(max(int(long_neighbors), 0), max(group_count - 1 - medium_end, 0))
    # Canonical endpoint order guarantees that each undirected observation is
    # evaluated once, even if selected from both endpoint neighbourhoods.
    edges: dict[tuple[int, int], int] = {}
    for source in range(group_count):
        selected: list[tuple[int, int]] = []
        selected.extend((int(target), 0) for target in order[source, :local_end])
        selected.extend((int(target), 1) for target in order[source, local_end:medium_end])
        if long_count:
            candidates = order[source, medium_end : group_count - 1]
            positions = np.linspace(0, candidates.size - 1, long_count, dtype=np.int64)
            selected.extend((int(target), 2) for target in candidates[positions])
        for target, level in selected:
            key = (min(source, target), max(source, target))
            edges[key] = min(int(level), edges.get(key, int(level)))
    pairs = sorted(edges)
    source = np.asarray([value[0] for value in pairs], dtype=np.int32)
    target = np.asarray([value[1] for value in pairs], dtype=np.int32)
    if pairs:
        delta = xy[target] - xy[source]
        distance = np.linalg.norm(delta, axis=1)
        scale = np.log(
            np.linalg.norm(extent[target], axis=1)
            / np.maximum(np.linalg.norm(extent[source], axis=1), 1e-8)
        )
        overlap = np.asarray(
            [_rectangle_iou(xy[a], extent[a], xy[b], extent[b]) for a, b in pairs]
        )
        level = np.asarray([edges[value] for value in pairs], dtype=np.float64)
        edge_features = np.c_[delta, distance, scale, overlap, level].astype(np.float32)
    else:
        edge_features = np.zeros((0, 6), dtype=np.float32)
    return RegionEvidenceGraph(
        xy=xy.astype(np.float32),
        extent=extent.astype(np.float32),
        candidate_rows=rows,
        candidate_probabilities=probability,
        null_probability=null,
        member_offsets=np.asarray(offsets, dtype=np.int32),
        member_indices=np.asarray(flat_members, dtype=np.int32),
        representative_indices=representative,
        edge_source=source,
        edge_target=target,
        edge_features=edge_features,
        grouping_feature_similarity=float(threshold),
    )


_RELATION_LEVEL_CACHE: dict[int, tuple[PhysicalMapletGraph, np.ndarray]] = {}


def _physical_relation_levels(physical: PhysicalMapletGraph) -> np.ndarray:
    cache_key = id(physical)
    cached = _RELATION_LEVEL_CACHE.get(cache_key)
    if cached is not None and cached[0] is physical:
        return cached[1]
    count = int(physical.maplet_ids.size)
    level = np.full((count, count), 3, dtype=np.int8)
    np.fill_diagonal(level, 0)
    for source, target, value in zip(
        physical.edge_source.tolist(),
        physical.edge_target.tolist(),
        physical.edge_features[:, 6].tolist(),
    ):
        first, second = min(int(source), int(target)), max(int(source), int(target))
        edge_level = int(round(float(value)))
        level[first, second] = min(level[first, second], edge_level)
        level[second, first] = level[first, second]
    # Keep a strong reference beside the cache value so Python cannot reuse an
    # object ID for a different graph while the entry is live.
    _RELATION_LEVEL_CACHE[cache_key] = (physical, level)
    return level


def score_region_surface_pose(
    pose_w2c: np.ndarray,
    query: RegionEvidenceGraph,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    pair_weight: float = 0.50,
    reverse_visibility_weight: float = 0.25,
    capacity_weight: float = 0.25,
) -> tuple[float, dict[str, float]]:
    """Score one pose by grouped footprint containment and graph consistency."""

    pixels, projected_extent_pixels, incidence, visible = _project_maplet_geometry(
        pose_w2c, physical, camera
    )
    image_size = np.asarray([camera.width, camera.height], dtype=np.float64)
    projected_xy = pixels / image_size[None]
    projected_extent = projected_extent_pixels / image_size[None]
    candidate_rows = query.candidate_rows
    valid_candidate = candidate_rows >= 0
    safe_rows = np.maximum(candidate_rows, 0)
    candidate_visible = valid_candidate & visible[safe_rows]

    # A query support is a local observation *inside* a maplet surface.  The
    # residual is zero within the projected footprint and grows only outside
    # it.  A uniform-footprint density automatically penalises huge ambiguous
    # maplets without requiring equal query/maplet support scales.
    absolute_delta = np.abs(projected_xy[safe_rows] - query.xy[:, None])
    outside = np.maximum(
        absolute_delta - projected_extent[safe_rows], 0.0
    )
    support_sigma = np.maximum(query.extent[:, None, :] * 0.65, 0.012)
    outside_residual = outside / support_sigma
    footprint_area = np.clip(
        4.0 * np.prod(projected_extent[safe_rows], axis=2), 2e-3, 1.0
    )
    oversized_support = np.maximum(
        np.log(
            np.maximum(np.linalg.norm(query.extent, axis=1)[:, None], 1e-6)
            / np.maximum(np.linalg.norm(projected_extent[safe_rows], axis=2), 1e-6)
        ),
        0.0,
    )
    unary_ratio = (
        np.exp(np.maximum(-0.5 * np.sum(outside_residual ** 2, axis=2), -60.0))
        * np.exp(np.maximum(-0.5 * (oversized_support / 0.7) ** 2, -20.0))
        * (0.25 + 0.75 * incidence[safe_rows])
        / footprint_area
    )
    unary_ratio = np.where(candidate_visible, unary_ratio, 0.0)
    weighted_unary = query.candidate_probabilities * unary_ratio
    unary_evidence = query.null_probability + np.sum(weighted_unary, axis=1)
    unary_log = np.log(np.maximum(unary_evidence, 1e-12))
    responsibility = weighted_unary / np.maximum(unary_evidence[:, None], 1e-12)

    relation_level = _physical_relation_levels(physical)
    edge_logs: list[float] = []
    edge_weights: list[float] = []
    for edge, (source, target) in enumerate(
        zip(query.edge_source.tolist(), query.edge_target.tolist())
    ):
        rows_a = candidate_rows[source]
        rows_b = candidate_rows[target]
        safe_a = np.maximum(rows_a, 0)
        safe_b = np.maximum(rows_b, 0)
        valid_pair = (rows_a[:, None] >= 0) & (rows_b[None] >= 0)
        valid_pair &= visible[safe_a, None] & visible[safe_b[None]]
        predicted_delta = projected_xy[safe_b][None] - projected_xy[safe_a][:, None]
        observed_delta = query.edge_features[edge, :2]
        level = int(round(float(query.edge_features[edge, 5])))
        relation_sigma = (0.060, 0.095, 0.15)[min(max(level, 0), 2)]
        delta_residual = np.sum(
            ((predicted_delta - observed_delta[None, None]) / relation_sigma) ** 2,
            axis=2,
        )
        observed_distance = max(float(query.edge_features[edge, 2]), 1e-4)
        predicted_distance = np.linalg.norm(predicted_delta, axis=2)
        distance_residual = np.log(
            np.maximum(predicted_distance, 1e-4) / observed_distance
        )
        ratio = np.exp(
            np.maximum(
                -0.5 * delta_residual - 0.5 * (distance_residual / 0.60) ** 2,
                -60.0,
            )
        ) / max(2.0 * np.pi * relation_sigma ** 2, 1e-5)
        compatibility = np.asarray(
            (
                (1.00, 0.80, 0.45, 0.20),
                (0.85, 1.00, 0.80, 0.30),
                (0.65, 0.85, 1.00, 0.45),
            )[min(max(level, 0), 2)]
        )
        ratio *= compatibility[relation_level[safe_a[:, None], safe_b[None]]]
        # Distinct evidence groups represent distinct local surface regions.
        ratio = np.where(safe_a[:, None] == safe_b[None], 0.0, ratio)
        ratio = np.where(valid_pair, ratio, 0.0)
        pair_probability = (
            query.candidate_probabilities[source, :, None]
            * query.candidate_probabilities[target, None, :]
        )
        unmatched = max(1.0 - float(np.sum(pair_probability)), 0.0)
        evidence = max(unmatched + float(np.sum(pair_probability * ratio)), 1e-12)
        edge_logs.append(float(np.log(evidence)))
        edge_weights.append((0.35, 0.70, 1.0)[min(max(level, 0), 2)])
    graph_log = (
        float(np.average(edge_logs, weights=edge_weights)) if edge_logs else 0.0
    )

    # Soft global capacity.  A large projected maplet may legitimately explain
    # several local groups; a small one may not absorb the whole image.
    physical_count = int(physical.maplet_ids.size)
    occupancy = np.zeros((physical_count,), dtype=np.float64)
    np.add.at(occupancy, safe_rows[valid_candidate], responsibility[valid_candidate])
    median_support_area = max(
        float(np.median(4.0 * np.prod(query.extent, axis=1))), 1e-5
    )
    projected_area = 4.0 * np.prod(projected_extent, axis=1)
    capacity = np.clip(projected_area / median_support_area, 1.0, 8.0)
    capacity_excess = np.maximum(occupancy - capacity, 0.0)
    capacity_log = -float(
        np.sum(capacity_excess ** 2 / np.maximum(capacity, 1.0))
        / max(query.group_count, 1)
    )

    # Reverse visibility: scene-retrieved maplets which this pose places in the
    # image need an explaining group.  No unseen/non-retrieved maplet is added.
    scene_prior = np.zeros((physical_count,), dtype=np.float64)
    for group in range(query.group_count):
        for local in range(candidate_rows.shape[1]):
            row = int(candidate_rows[group, local])
            if row >= 0:
                probability = float(query.candidate_probabilities[group, local])
                scene_prior[row] = 1.0 - (1.0 - scene_prior[row]) * (1.0 - probability)
    explained = np.zeros((physical_count,), dtype=np.float64)
    for group in range(query.group_count):
        for local in range(candidate_rows.shape[1]):
            row = int(candidate_rows[group, local])
            if row >= 0:
                explained[row] = max(explained[row], float(responsibility[group, local]))
    reverse_rows = np.flatnonzero(visible & (scene_prior > 0.0))
    if reverse_rows.size and float(np.sum(scene_prior[reverse_rows])) > 0.0:
        reverse_visibility_log = float(
            np.average(
                np.log(np.maximum(0.20 + 0.80 * explained[reverse_rows], 1e-12)),
                weights=scene_prior[reverse_rows],
            )
        )
    else:
        reverse_visibility_log = 0.0

    score = float(
        np.mean(unary_log)
        + float(pair_weight) * graph_log
        + float(reverse_visibility_weight) * reverse_visibility_log
        + float(capacity_weight) * capacity_log
    )
    return score, {
        "unary_log_likelihood": float(np.mean(unary_log)),
        "graph_log_likelihood": graph_log,
        "reverse_visibility_log_likelihood": reverse_visibility_log,
        "capacity_log_likelihood": capacity_log,
        "supported_group_fraction": float(np.mean(unary_evidence > 1.0)),
        "visible_candidate_fraction": float(np.mean(candidate_visible)),
        "maximum_maplet_occupancy": float(np.max(occupancy, initial=0.0)),
        "evidence_group_count": float(query.group_count),
    }


def refine_region_surface_pose(
    initial_pose_w2c: np.ndarray,
    query: RegionEvidenceGraph,
    physical: PhysicalMapletGraph,
    camera: ColmapCamera,
    *,
    pair_weight: float = 0.50,
    reverse_visibility_weight: float = 0.25,
    capacity_weight: float = 0.25,
    stages: int = 5,
    initial_translation_step_m: float = 0.75,
    initial_rotation_step_deg: float = 8.0,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Deterministic coordinate refinement of the V8.1 surface score."""

    pose = np.asarray(initial_pose_w2c, dtype=np.float64).copy()
    score, parts = score_region_surface_pose(
        pose,
        query,
        physical,
        camera,
        pair_weight=pair_weight,
        reverse_visibility_weight=reverse_visibility_weight,
        capacity_weight=capacity_weight,
    )
    for stage in range(max(int(stages), 0)):
        step = np.asarray(
            [np.deg2rad(initial_rotation_step_deg) / (2.0 ** stage)] * 3
            + [initial_translation_step_m / (2.0 ** stage)] * 3,
            dtype=np.float64,
        )
        winner = (score, pose, parts)
        for axis in range(6):
            for sign in (-1.0, 1.0):
                delta = np.zeros((6,), dtype=np.float64)
                delta[axis] = sign * step[axis]
                candidate_pose = se3_exp(delta) @ pose
                candidate_score, candidate_parts = score_region_surface_pose(
                    candidate_pose,
                    query,
                    physical,
                    camera,
                    pair_weight=pair_weight,
                    reverse_visibility_weight=reverse_visibility_weight,
                    capacity_weight=capacity_weight,
                )
                if candidate_score > winner[0]:
                    winner = (candidate_score, candidate_pose, candidate_parts)
        score, pose, parts = winner
    return pose, float(score), parts
