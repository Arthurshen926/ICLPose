"""Identity-free query support enumeration and posterior aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval import SparseMapletPosterior


@dataclass(frozen=True)
class GroupedQuerySupports:
    xy: np.ndarray
    extent: np.ndarray
    descriptors: np.ndarray
    member_offsets: np.ndarray
    member_token_indices: np.ndarray


def aggregate_group_sparse_posteriors(
    posterior: SparseMapletPosterior,
    member_offsets: np.ndarray,
    member_token_indices: np.ndarray,
    *,
    maximum_candidates: int,
) -> SparseMapletPosterior:
    """Average correlated sparse posteriors while preserving null semantics."""

    ids = posterior.candidate_ids
    probability = posterior.candidate_probabilities.astype(np.float64)
    out_of_map = posterior.out_of_map_probabilities.astype(np.float64)
    offsets = np.asarray(member_offsets, dtype=np.int64).reshape(-1)
    members = np.asarray(member_token_indices, dtype=np.int64).reshape(-1)
    if (
        offsets.size < 1
        or offsets[0] != 0
        or offsets[-1] != members.size
        or np.any(np.diff(offsets) <= 0)
        or np.any((members < 0) | (members >= ids.shape[0]))
    ):
        raise ValueError("invalid sparse posterior grouping")
    count = offsets.size - 1
    keep = max(1, int(maximum_candidates))
    output_ids = np.full((count, keep), -1, dtype=np.int64)
    output_probability = np.zeros((count, keep), dtype=np.float64)
    output_out = np.zeros((count,), dtype=np.float64)
    output_best = np.zeros((count,), dtype=np.float64)
    for group in range(count):
        rows = members[int(offsets[group]) : int(offsets[group + 1])]
        scale = 1.0 / float(rows.size)
        accumulated: dict[int, float] = {}
        for identity, value in zip(ids[rows].reshape(-1).tolist(), probability[rows].reshape(-1).tolist()):
            if int(identity) >= 0 and float(value) > 0.0:
                accumulated[int(identity)] = accumulated.get(int(identity), 0.0) + scale * float(value)
        ranked = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))[:keep]
        if ranked:
            output_ids[group, : len(ranked)] = [item[0] for item in ranked]
            output_probability[group, : len(ranked)] = [item[1] for item in ranked]
        output_out[group] = float(np.mean(out_of_map[rows]))
        output_best[group] = float(np.mean(posterior.best_similarities[rows]))
    output_tail = np.clip(1.0 - output_out - np.sum(output_probability, axis=1), 0.0, 1.0)
    return SparseMapletPosterior(
        output_ids, output_probability, output_out, output_tail, output_best,
    )


def aggregate_group_posteriors(
    candidate_ids: np.ndarray,
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    member_offsets: np.ndarray,
    member_token_indices: np.ndarray,
    *,
    maximum_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average correlated token posteriors without performing retrieval again.

    Tokens inside a connected group are correlated measurements, so averaging
    is intentionally used instead of a product/noisy-or that would make large
    components spuriously overconfident.  The operation preserves the first
    retrieval's identity evidence and merely sparsifies its union.
    """

    ids = np.asarray(candidate_ids, dtype=np.int64)
    probability = np.asarray(candidate_probabilities, dtype=np.float64)
    null = np.asarray(null_probabilities, dtype=np.float64).reshape(-1)
    offsets = np.asarray(member_offsets, dtype=np.int64).reshape(-1)
    members = np.asarray(member_token_indices, dtype=np.int64).reshape(-1)
    if (
        ids.ndim != 2
        or probability.shape != ids.shape
        or null.shape != (ids.shape[0],)
        or offsets.size < 1
        or offsets[0] != 0
        or offsets[-1] != members.size
        or np.any(np.diff(offsets) <= 0)
        or np.any((members < 0) | (members >= ids.shape[0]))
    ):
        raise ValueError("invalid grouped posterior arrays")
    count = int(offsets.size - 1)
    keep = max(1, int(maximum_candidates))
    output_ids = np.full((count, keep), -1, dtype=np.int64)
    output_probability = np.zeros((count, keep), dtype=np.float64)
    output_null = np.zeros((count,), dtype=np.float64)
    for group in range(count):
        rows = members[int(offsets[group]) : int(offsets[group + 1])]
        accumulated: dict[int, float] = {}
        scale = 1.0 / float(rows.size)
        for maplet_id, value in zip(ids[rows].reshape(-1).tolist(), probability[rows].reshape(-1).tolist()):
            if int(maplet_id) >= 0 and float(value) > 0.0:
                accumulated[int(maplet_id)] = accumulated.get(int(maplet_id), 0.0) + scale * float(value)
        ranked = sorted(accumulated.items(), key=lambda item: (-item[1], item[0]))[:keep]
        if ranked:
            output_ids[group, : len(ranked)] = [item[0] for item in ranked]
            output_probability[group, : len(ranked)] = [item[1] for item in ranked]
        output_null[group] = float(np.mean(null[rows]))
        # Truncation mass is uncertainty, not evidence for a maplet outside the
        # retained sparse posterior.
        retained = float(np.sum(output_probability[group]))
        output_null[group] = float(np.clip(max(output_null[group], 1.0 - retained), 0.0, 1.0))
    return output_ids, output_probability.astype(np.float32), output_null.astype(np.float32)


def all_token_coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(height) * int(width), dtype=np.int64)
    xy = np.stack([indices % int(width), indices // int(width)], axis=1).astype(np.float32)
    return indices, xy


def aggregate_group_descriptors(
    descriptors: np.ndarray,
    member_offsets: np.ndarray,
    member_token_indices: np.ndarray,
) -> np.ndarray:
    """Average and normalize one descriptor per connected support group."""

    feature = np.asarray(descriptors, dtype=np.float32)
    offsets = np.asarray(member_offsets, dtype=np.int64).reshape(-1)
    members = np.asarray(member_token_indices, dtype=np.int64).reshape(-1)
    if (
        feature.ndim != 2
        or offsets.size < 1
        or offsets[0] != 0
        or offsets[-1] != members.size
        or np.any(np.diff(offsets) <= 0)
        or np.any((members < 0) | (members >= feature.shape[0]))
    ):
        raise ValueError("invalid grouped descriptor arrays")
    output = np.zeros((offsets.size - 1, feature.shape[1]), dtype=np.float32)
    for group in range(offsets.size - 1):
        rows = members[int(offsets[group]) : int(offsets[group + 1])]
        value = np.mean(feature[rows], axis=0)
        output[group] = value / max(float(np.linalg.norm(value)), 1e-8)
    return output


def group_tokens_after_retrieval(
    token_xy: np.ndarray,
    descriptors: np.ndarray,
    top1_maplet_ids: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    image_width: int,
    image_height: int,
    descriptor_half_size_tokens: float,
    minimum_descriptor_cosine: float = 0.90,
) -> GroupedQuerySupports:
    """Group adjacent correlated tokens with the same retrieved identity.

    Retrieval is deliberately performed first.  Spatially separated repeated
    facade elements never merge merely because their VFM codes are similar.
    """

    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    feature = np.asarray(descriptors, dtype=np.float32)
    identity = np.asarray(top1_maplet_ids, dtype=np.int64).reshape(-1)
    if feature.shape[0] != xy.shape[0] or identity.shape != (xy.shape[0],):
        raise ValueError("query token grouping arrays differ")
    feature = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-8)
    parent = np.arange(xy.shape[0], dtype=np.int64)
    row_by_grid = {(int(value[0]), int(value[1])): row for row, value in enumerate(xy.tolist())}

    def find(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = int(parent[row])
        return row

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for row, (x, y) in enumerate(xy.tolist()):
        for neighbor_xy in ((x + 1, y), (x, y + 1)):
            neighbor = row_by_grid.get(neighbor_xy)
            if (
                neighbor is not None
                and identity[row] >= 0
                and identity[row] == identity[neighbor]
                and float(np.dot(feature[row], feature[neighbor])) >= float(minimum_descriptor_cosine)
            ):
                union(row, int(neighbor))
    groups: dict[int, list[int]] = {}
    for row in range(xy.shape[0]):
        groups.setdefault(find(row), []).append(row)
    scale = np.asarray([image_width / token_width, image_height / token_height], dtype=np.float64)
    group_xy, group_extent, group_feature = [], [], []
    offsets = [0]
    members: list[int] = []
    for values in sorted(groups.values(), key=lambda value: min(value)):
        rows = np.asarray(values, dtype=np.int64)
        low = np.min(xy[rows], axis=0).astype(np.float64) - float(descriptor_half_size_tokens)
        high = np.max(xy[rows], axis=0).astype(np.float64) + float(descriptor_half_size_tokens) + 1.0
        low_px, high_px = low * scale, high * scale
        group_xy.append(0.5 * (low_px + high_px) / [image_width, image_height])
        group_extent.append(0.5 * (high_px - low_px) / [image_width, image_height])
        value = np.mean(feature[rows], axis=0)
        group_feature.append(value / max(float(np.linalg.norm(value)), 1e-8))
        members.extend(rows.tolist())
        offsets.append(len(members))
    return GroupedQuerySupports(
        xy=np.asarray(group_xy, dtype=np.float32),
        extent=np.asarray(group_extent, dtype=np.float32),
        descriptors=np.asarray(group_feature, dtype=np.float32),
        member_offsets=np.asarray(offsets, dtype=np.int64),
        member_token_indices=np.asarray(members, dtype=np.int64),
    )


def group_tokens_identity_free(
    token_xy: np.ndarray,
    descriptors: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    image_width: int,
    image_height: int,
    descriptor_half_size_tokens: float,
    minimum_descriptor_cosine: float = 0.96,
    maximum_group_diameter_tokens: float = 2.0,
) -> GroupedQuerySupports:
    """Group correlated query tokens without consulting any map identity.

    Adjacent edges are processed from strongest to weakest.  A merge is
    accepted only if the merged component satisfies a complete-link feature
    threshold and a bounded token-grid diameter.  These two constraints avoid
    transitive facade-wide chaining while keeping the query partition fixed
    for every physical-map hypothesis.
    """

    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    feature = np.asarray(descriptors, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] != xy.shape[0]:
        raise ValueError("identity-free grouping arrays differ")
    if float(maximum_group_diameter_tokens) < 1.0:
        raise ValueError("identity-free maximum group diameter is too small")
    feature = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-8)
    row_by_grid = {(int(value[0]), int(value[1])): row for row, value in enumerate(xy.tolist())}
    edges: list[tuple[float, int, int]] = []
    for row, (x, y) in enumerate(xy.tolist()):
        for neighbor_xy in ((x + 1, y), (x, y + 1)):
            neighbor = row_by_grid.get(neighbor_xy)
            if neighbor is not None:
                cosine = float(np.dot(feature[row], feature[int(neighbor)]))
                if cosine >= float(minimum_descriptor_cosine):
                    edges.append((-cosine, int(row), int(neighbor)))
    edges.sort()
    parent = np.arange(xy.shape[0], dtype=np.int64)
    components: dict[int, list[int]] = {row: [row] for row in range(xy.shape[0])}

    def find(row: int) -> int:
        while int(parent[row]) != row:
            parent[row] = parent[int(parent[row])]
            row = int(parent[row])
        return row

    for _, left, right in edges:
        a, b = find(left), find(right)
        if a == b:
            continue
        rows_a = np.asarray(components[a], dtype=np.int64)
        rows_b = np.asarray(components[b], dtype=np.int64)
        merged = np.concatenate([rows_a, rows_b])
        span = np.ptp(xy[merged], axis=0)
        if np.any(span > float(maximum_group_diameter_tokens)):
            continue
        cross_cosine = feature[rows_a] @ feature[rows_b].T
        if float(np.min(cross_cosine)) < float(minimum_descriptor_cosine):
            continue
        # Stable root selection makes the result independent of union order.
        root, absorbed = (a, b) if min(components[a]) <= min(components[b]) else (b, a)
        parent[absorbed] = root
        components[root] = sorted(components[root] + components[absorbed])
        del components[absorbed]

    groups = sorted(components.values(), key=lambda value: min(value))
    scale = np.asarray([image_width / token_width, image_height / token_height], dtype=np.float64)
    group_xy, group_extent, group_feature = [], [], []
    offsets = [0]
    members: list[int] = []
    for values in groups:
        rows = np.asarray(values, dtype=np.int64)
        low = np.min(xy[rows], axis=0).astype(np.float64) - float(descriptor_half_size_tokens)
        high = np.max(xy[rows], axis=0).astype(np.float64) + float(descriptor_half_size_tokens) + 1.0
        low_px, high_px = low * scale, high * scale
        group_xy.append(0.5 * (low_px + high_px) / [image_width, image_height])
        group_extent.append(0.5 * (high_px - low_px) / [image_width, image_height])
        value = np.mean(feature[rows], axis=0)
        group_feature.append(value / max(float(np.linalg.norm(value)), 1e-8))
        members.extend(rows.tolist())
        offsets.append(len(members))
    return GroupedQuerySupports(
        xy=np.asarray(group_xy, dtype=np.float32),
        extent=np.asarray(group_extent, dtype=np.float32),
        descriptors=np.asarray(group_feature, dtype=np.float32),
        member_offsets=np.asarray(offsets, dtype=np.int64),
        member_token_indices=np.asarray(members, dtype=np.int64),
    )
