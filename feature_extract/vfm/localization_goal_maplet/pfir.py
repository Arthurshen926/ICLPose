"""Physical Feature-to-Instance Retrieval (PFIR) labels and metrics.

Labels are multi-positive contributor distributions from the declared clean
2DGS rasterizer.  They are not nearest-maplet-center or single-ID proxies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physical_map import GoalMapletPhysicalMap
from .visibility import camera_center_from_w2c


@dataclass(frozen=True)
class ContributorLabels:
    topk_primitive_ids: np.ndarray
    topk_weights: np.ndarray
    pose_w2c: np.ndarray

    @classmethod
    def load_npz(cls, path: Path) -> "ContributorLabels":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                topk_primitive_ids=np.asarray(data["topk_ids"], dtype=np.int64),
                topk_weights=np.asarray(data["topk_weights"], dtype=np.float32),
                pose_w2c=np.asarray(data["pose_w2c"], dtype=np.float64),
            )


@dataclass(frozen=True)
class QuerySupportPosterior:
    xy: np.ndarray
    extent: np.ndarray
    candidate_maplet_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    support_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64)
        extent = np.asarray(self.extent, dtype=np.float64)
        ids = np.asarray(self.candidate_maplet_ids, dtype=np.int64)
        probability = np.asarray(self.candidate_probabilities, dtype=np.float64)
        null = np.asarray(self.null_probabilities, dtype=np.float64).reshape(-1)
        count = xy.shape[0]
        support_weight = (
            np.ones((count,), dtype=np.float64)
            if self.support_weights is None
            else np.asarray(self.support_weights, dtype=np.float64).reshape(-1)
        )
        if (
            xy.shape != (count, 2)
            or extent.shape != (count, 2)
            or ids.ndim != 2
            or ids.shape[0] != count
            or probability.shape != ids.shape
            or null.shape != (count,)
            or support_weight.shape != (count,)
            or np.any(~np.isfinite(support_weight))
            or np.any(support_weight <= 0.0)
            or np.any(probability < 0.0)
            or np.any((null < 0.0) | (null > 1.0))
        ):
            raise ValueError("invalid query support posterior")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "candidate_maplet_ids", ids)
        object.__setattr__(self, "candidate_probabilities", probability)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "support_weights", support_weight)


def _primitive_to_maplet_links(
    physical_map: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR primitive row -> maplet rows, normalized for overlapping parents."""

    primitive_links: list[list[tuple[int, float]]] = [
        [] for _ in range(physical_map.primitive_ids.size)
    ]
    for maplet_row in range(physical_map.maplet_ids.size):
        link = physical_map.member_slice(maplet_row)
        for primitive_row, weight in zip(
            physical_map.membership_primitive_rows[link].tolist(),
            physical_map.membership_weights[link].tolist(),
        ):
            primitive_links[int(primitive_row)].append((maplet_row, float(weight)))
    offsets = [0]
    rows: list[int] = []
    weights: list[float] = []
    for values in primitive_links:
        total = max(sum(value[1] for value in values), 1e-12)
        rows.extend(value[0] for value in values)
        weights.extend(value[1] / total for value in values)
        offsets.append(len(rows))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def _primitive_to_child_links(
    physical_map: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR primitive row -> child rows, normalized across overlapping parents."""

    primitive_links: list[list[tuple[int, float]]] = [
        [] for _ in range(physical_map.primitive_ids.size)
    ]
    for child_row in range(physical_map.child_parent_rows.size):
        start, end = int(physical_map.child_member_offsets[child_row]), int(physical_map.child_member_offsets[child_row + 1])
        for primitive_row, weight in zip(
            physical_map.child_member_primitive_rows[start:end].tolist(),
            physical_map.child_member_weights[start:end].tolist(),
        ):
            primitive_links[int(primitive_row)].append((child_row, float(weight)))
    offsets, rows, weights = [0], [], []
    for values in primitive_links:
        total = max(sum(value[1] for value in values), 1e-12)
        rows.extend(value[0] for value in values)
        weights.extend(value[1] / total for value in values)
        offsets.append(len(rows))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def contributor_maplet_distribution(
    labels: ContributorLabels,
    physical_map: GoalMapletPhysicalMap,
    support_xy: np.ndarray,
    support_extent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate exact top-k raster contributors inside every support mask."""

    xy = np.asarray(support_xy, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(support_extent, dtype=np.float64).reshape(-1, 2)
    height, width, topk = labels.topk_primitive_ids.shape
    primitive_row_by_id = {
        int(value): row for row, value in enumerate(physical_map.primitive_ids.tolist())
    }
    link_offsets, link_rows, link_weights = _primitive_to_maplet_links(physical_map)
    output = np.zeros((xy.shape[0], physical_map.maplet_ids.size), dtype=np.float64)
    null = np.ones((xy.shape[0],), dtype=np.float64)
    for support in range(xy.shape[0]):
        low = np.floor((xy[support] - extent[support]) * [width, height]).astype(np.int64)
        high = np.ceil((xy[support] + extent[support]) * [width, height]).astype(np.int64)
        x0, y0 = np.maximum(low, 0)
        x1, y1 = np.minimum(high, [width, height])
        if x1 <= x0 or y1 <= y0:
            continue
        ids = labels.topk_primitive_ids[y0:y1, x0:x1].reshape(-1, topk)
        weights = labels.topk_weights[y0:y1, x0:x1].reshape(-1, topk)
        total_mass = float(np.sum(np.maximum(weights, 0.0)))
        if total_mass <= 1e-12:
            continue
        for primitive_id, contribution in zip(ids.reshape(-1).tolist(), weights.reshape(-1).tolist()):
            primitive_row = primitive_row_by_id.get(int(primitive_id))
            if primitive_row is None or contribution <= 0.0:
                continue
            start, end = int(link_offsets[primitive_row]), int(link_offsets[primitive_row + 1])
            if end > start:
                output[support, link_rows[start:end]] += float(contribution) * link_weights[start:end]
        output[support] /= total_mass
        null[support] = float(np.clip(1.0 - np.sum(output[support]), 0.0, 1.0))
    return output.astype(np.float32), null.astype(np.float32)


def contributor_multiscale_maplet_distribution(
    labels: ContributorLabels,
    physical_map: GoalMapletPhysicalMap,
    token_xy: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    pool_sizes: tuple[int, ...],
    pool_weights: tuple[float, ...],
    group_member_offsets: np.ndarray | None = None,
    group_member_token_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PFIR truth under the exact weighted union of VFM pooling masks.

    A multi-scale descriptor is not supported by the box corresponding to the
    weighted-average kernel diameter.  Its physical target is the mixture of
    the clipped per-scale masks.  For post-retrieval groups we average the
    member masks, preserving irregular connected components without filling
    their bounding boxes.
    """

    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    if len(pool_sizes) != len(pool_weights) or not pool_sizes:
        raise ValueError("pool sizes and weights differ")
    if np.any((xy[:, 0] < 0) | (xy[:, 0] >= int(token_width))) or np.any(
        (xy[:, 1] < 0) | (xy[:, 1] >= int(token_height))
    ):
        raise ValueError("token coordinate outside feature grid")
    if group_member_offsets is None:
        offsets = np.arange(xy.shape[0] + 1, dtype=np.int64)
        members = np.arange(xy.shape[0], dtype=np.int64)
    else:
        offsets = np.asarray(group_member_offsets, dtype=np.int64).reshape(-1)
        members = np.asarray(group_member_token_indices, dtype=np.int64).reshape(-1)
        if (
            offsets.size < 1
            or offsets[0] != 0
            or offsets[-1] != members.size
            or np.any(np.diff(offsets) <= 0)
            or np.any((members < 0) | (members >= xy.shape[0]))
        ):
            raise ValueError("invalid exact-mask group membership")

    height, width, topk = labels.topk_primitive_ids.shape
    primitive_row_by_id = {
        int(value): row for row, value in enumerate(physical_map.primitive_ids.tolist())
    }
    link_offsets, link_rows, link_weights = _primitive_to_maplet_links(physical_map)
    output = np.zeros((offsets.size - 1, physical_map.maplet_ids.size), dtype=np.float64)
    null = np.ones((offsets.size - 1,), dtype=np.float64)
    for support in range(offsets.size - 1):
        selected_members = members[int(offsets[support]) : int(offsets[support + 1])]
        pixel_weight = np.zeros((height, width), dtype=np.float64)
        member_scale = 1.0 / float(selected_members.size)
        for member in selected_members.tolist():
            x, y = xy[int(member)].tolist()
            for size, scale_weight in zip(pool_sizes, pool_weights):
                if float(scale_weight) <= 0.0:
                    continue
                radius = int(size) // 2
                tx0, tx1 = max(0, x - radius), min(int(token_width), x + radius + 1)
                ty0, ty1 = max(0, y - radius), min(int(token_height), y + radius + 1)
                px0 = int(np.floor(tx0 * width / float(token_width)))
                px1 = int(np.ceil(tx1 * width / float(token_width)))
                py0 = int(np.floor(ty0 * height / float(token_height)))
                py1 = int(np.ceil(ty1 * height / float(token_height)))
                area = max((px1 - px0) * (py1 - py0), 1)
                pixel_weight[py0:py1, px0:px1] += member_scale * float(scale_weight) / float(area)
        selected_pixels = pixel_weight > 0.0
        ids = labels.topk_primitive_ids[selected_pixels].reshape(-1, topk)
        weights = labels.topk_weights[selected_pixels].reshape(-1, topk)
        weights = weights * pixel_weight[selected_pixels, None]
        total_mass = float(np.sum(np.maximum(weights, 0.0)))
        if total_mass <= 1e-12:
            continue
        for primitive_id, contribution in zip(ids.reshape(-1).tolist(), weights.reshape(-1).tolist()):
            primitive_row = primitive_row_by_id.get(int(primitive_id))
            if primitive_row is None or contribution <= 0.0:
                continue
            start, end = int(link_offsets[primitive_row]), int(link_offsets[primitive_row + 1])
            if end > start:
                output[support, link_rows[start:end]] += float(contribution) * link_weights[start:end]
        output[support] /= total_mass
        null[support] = float(np.clip(1.0 - np.sum(output[support]), 0.0, 1.0))
    return output.astype(np.float32), null.astype(np.float32)


def contributor_multiscale_child_distribution(
    labels: ContributorLabels,
    physical_map: GoalMapletPhysicalMap,
    token_xy: np.ndarray,
    *,
    token_height: int,
    token_width: int,
    pool_sizes: tuple[int, ...] = (1,),
    pool_weights: tuple[float, ...] = (1.0,),
    group_member_offsets: np.ndarray | None = None,
    group_member_token_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact contributor distribution over metric child surface tiles."""

    xy = np.asarray(token_xy, dtype=np.int64).reshape(-1, 2)
    if len(pool_sizes) != len(pool_weights) or not pool_sizes:
        raise ValueError("pool sizes and weights differ")
    if np.any((xy[:, 0] < 0) | (xy[:, 0] >= int(token_width))) or np.any(
        (xy[:, 1] < 0) | (xy[:, 1] >= int(token_height))
    ):
        raise ValueError("token coordinate outside feature grid")
    if group_member_offsets is None:
        offsets = np.arange(xy.shape[0] + 1, dtype=np.int64)
        members = np.arange(xy.shape[0], dtype=np.int64)
    else:
        offsets = np.asarray(group_member_offsets, dtype=np.int64).reshape(-1)
        members = np.asarray(group_member_token_indices, dtype=np.int64).reshape(-1)
        if (
            offsets.size < 1
            or offsets[0] != 0
            or offsets[-1] != members.size
            or np.any(np.diff(offsets) <= 0)
            or np.any((members < 0) | (members >= xy.shape[0]))
        ):
            raise ValueError("invalid exact-mask group membership")
    height, width, topk = labels.topk_primitive_ids.shape
    primitive_row_by_id = {int(value): row for row, value in enumerate(physical_map.primitive_ids.tolist())}
    link_offsets, link_rows, link_weights = _primitive_to_child_links(physical_map)
    output = np.zeros((offsets.size - 1, physical_map.child_parent_rows.size), dtype=np.float64)
    null = np.ones((offsets.size - 1,), dtype=np.float64)
    for support in range(offsets.size - 1):
        selected_members = members[int(offsets[support]) : int(offsets[support + 1])]
        pixel_weight = np.zeros((height, width), dtype=np.float64)
        member_scale = 1.0 / float(selected_members.size)
        for member in selected_members.tolist():
            x, y = xy[int(member)].tolist()
            for size, scale_weight in zip(pool_sizes, pool_weights):
                if float(scale_weight) <= 0.0:
                    continue
                radius = int(size) // 2
                tx0, tx1 = max(0, x - radius), min(int(token_width), x + radius + 1)
                ty0, ty1 = max(0, y - radius), min(int(token_height), y + radius + 1)
                px0 = int(np.floor(tx0 * width / float(token_width)))
                px1 = int(np.ceil(tx1 * width / float(token_width)))
                py0 = int(np.floor(ty0 * height / float(token_height)))
                py1 = int(np.ceil(ty1 * height / float(token_height)))
                area = max((px1 - px0) * (py1 - py0), 1)
                pixel_weight[py0:py1, px0:px1] += member_scale * float(scale_weight) / float(area)
        selected_pixels = pixel_weight > 0.0
        ids = labels.topk_primitive_ids[selected_pixels].reshape(-1, topk)
        weights = labels.topk_weights[selected_pixels].reshape(-1, topk) * pixel_weight[selected_pixels, None]
        total_mass = float(np.sum(np.maximum(weights, 0.0)))
        if total_mass <= 1e-12:
            continue
        for primitive_id, contribution in zip(ids.reshape(-1).tolist(), weights.reshape(-1).tolist()):
            primitive_row = primitive_row_by_id.get(int(primitive_id))
            if primitive_row is None or contribution <= 0.0:
                continue
            start, end = int(link_offsets[primitive_row]), int(link_offsets[primitive_row + 1])
            if end > start:
                output[support, link_rows[start:end]] += float(contribution) * link_weights[start:end]
        output[support] /= total_mass
        null[support] = float(np.clip(1.0 - np.sum(output[support]), 0.0, 1.0))
    return output.astype(np.float32), null.astype(np.float32)


def _weighted_ap(relevance: np.ndarray) -> float:
    rel = np.maximum(np.asarray(relevance, dtype=np.float64), 0.0)
    total = float(np.sum(rel))
    if total <= 0.0:
        return 0.0
    precision = np.cumsum(rel) / np.arange(1, rel.size + 1)
    return float(np.sum(precision * rel) / total)


def _ndcg(relevance: np.ndarray, ideal: np.ndarray) -> float:
    discount = 1.0 / np.log2(np.arange(2, relevance.size + 2))
    dcg = float(np.sum(np.asarray(relevance) * discount))
    idcg = float(np.sum(np.sort(np.asarray(ideal))[::-1][: relevance.size] * discount))
    return dcg / max(idcg, 1e-12)


def _ece(
    confidence: np.ndarray,
    target: np.ndarray,
    bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weight = np.ones_like(confidence) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    total_weight = max(float(np.sum(weight)), 1e-12)
    value = 0.0
    for index in range(int(bins)):
        low, high = index / bins, (index + 1) / bins
        selected = (confidence >= low) & (confidence < high if index + 1 < bins else confidence <= high)
        if np.any(selected):
            selected_weight = weight[selected]
            mass = float(np.sum(selected_weight))
            predicted = float(np.average(confidence[selected], weights=selected_weight))
            observed = float(np.average(target[selected], weights=selected_weight))
            value += mass / total_weight * abs(predicted - observed)
    return value


def evaluate_pfir(
    posterior: QuerySupportPosterior,
    ground_truth_distribution: np.ndarray,
    ground_truth_null: np.ndarray,
    physical_map: GoalMapletPhysicalMap,
    pose_w2c: np.ndarray,
    *,
    recall_k: tuple[int, ...] = (1, 5, 20, 64),
) -> dict[str, object]:
    truth = np.asarray(ground_truth_distribution, dtype=np.float64)
    truth_null = np.asarray(ground_truth_null, dtype=np.float64).reshape(-1)
    if truth.shape != (posterior.xy.shape[0], physical_map.maplet_ids.size):
        raise ValueError("PFIR truth/posterior shapes differ")
    row_by_id = {int(value): row for row, value in enumerate(physical_map.maplet_ids.tolist())}
    per_support: list[dict[str, float]] = []
    support_weight = np.asarray(posterior.support_weights, dtype=np.float64)
    recalls = {k: [] for k in recall_k}
    ap, ndcg, mrr, surface_distance, surface_distance_weight = [], [], [], [], []
    for support in range(posterior.xy.shape[0]):
        candidate_ids = posterior.candidate_maplet_ids[support]
        candidate_rows = np.asarray([row_by_id.get(int(value), -1) for value in candidate_ids], dtype=np.int64)
        relevance = np.asarray([truth[support, row] if row >= 0 else 0.0 for row in candidate_rows])
        # Duplicate candidates do not get credit twice.
        seen: set[int] = set()
        unique_relevance = relevance.copy()
        for rank, row in enumerate(candidate_rows.tolist()):
            if row < 0 or row in seen:
                unique_relevance[rank] = 0.0
            seen.add(row)
        positive_mass = max(float(np.sum(truth[support])), 1e-12)
        for k in recall_k:
            recalls[k].append(float(np.sum(unique_relevance[:k]) / positive_mass))
        ap.append(_weighted_ap(unique_relevance))
        ndcg.append(_ndcg(unique_relevance, truth[support]))
        positive = np.flatnonzero(unique_relevance > 1e-6)
        mrr.append(0.0 if positive.size == 0 else 1.0 / float(positive[0] + 1))
        gt_rows = np.flatnonzero(truth[support] > 0.0)
        if candidate_rows.size and candidate_rows[0] >= 0 and gt_rows.size:
            distance = np.linalg.norm(
                physical_map.maplet_centers[gt_rows] - physical_map.maplet_centers[candidate_rows[0]],
                axis=1,
            )
            surface_distance.append(float(np.sum(distance * truth[support, gt_rows]) / positive_mass))
            surface_distance_weight.append(float(support_weight[support]))
        per_support.append({
            "weight": float(support_weight[support]),
            "positive_mass": float(np.sum(truth[support])),
            "ap": ap[-1],
            "ndcg": ndcg[-1],
            "mrr": mrr[-1],
        })

    scene_score = np.zeros((physical_map.maplet_ids.size,), dtype=np.float64)
    for support in range(posterior.xy.shape[0]):
        for maplet_id, probability in zip(
            posterior.candidate_maplet_ids[support].tolist(),
            posterior.candidate_probabilities[support].tolist(),
        ):
            row = row_by_id.get(int(maplet_id))
            if row is not None:
                evidence = 1.0 - (1.0 - float(np.clip(probability, 0.0, 1.0))) ** float(support_weight[support])
                scene_score[row] = 1.0 - (1.0 - scene_score[row]) * (1.0 - evidence)
    scene_truth = np.sum(truth * support_weight[:, None], axis=0)
    scene_truth /= max(float(np.sum(scene_truth)), 1e-12)
    ranking = np.argsort(-scene_score, kind="stable")
    scene_coverage = {f"coverage_at_{k}": float(np.sum(scene_truth[ranking[:k]])) for k in recall_k}
    top64 = ranking[: min(64, ranking.size)]
    correct = top64[scene_truth[top64] > 0.0]
    centers = physical_map.maplet_centers[correct]
    camera_center = camera_center_from_w2c(pose_w2c)
    bearings = centers - camera_center[None] if centers.size else np.zeros((0, 3))
    bearing_rank = int(np.linalg.matrix_rank(bearings - np.mean(bearings, axis=0), tol=0.10)) if len(bearings) >= 2 else 0
    spread = float(np.max(np.linalg.norm(centers[:, None] - centers[None], axis=2))) if len(centers) >= 2 else 0.0
    normal_gram_rank = int(np.linalg.matrix_rank(physical_map.maplet_normals[correct], tol=0.20)) if correct.size else 0
    pose_sufficient = bool(correct.size >= 3 and spread >= 0.5 and (bearing_rank >= 2 or normal_gram_rank >= 2))
    valid_target = 1.0 - truth_null
    confidence = 1.0 - posterior.null_probabilities
    def weighted_mean(value: list[float] | np.ndarray) -> float:
        array = np.asarray(value, dtype=np.float64)
        return float(np.average(array, weights=support_weight)) if array.size else 0.0

    return {
        "support_count": int(posterior.xy.shape[0]),
        "support_effective_count": float(np.sum(support_weight)),
        **{f"weighted_recall_at_{k}": weighted_mean(value) for k, value in recalls.items()},
        "multi_positive_ap": weighted_mean(ap),
        "ndcg": weighted_mean(ndcg),
        "mrr": weighted_mean(mrr),
        "null_ece": _ece(confidence, valid_target, sample_weight=support_weight),
        "null_brier": float(np.average(np.square(confidence - valid_target), weights=support_weight)),
        "top1_expected_maplet_center_distance_m": (
            float(np.average(surface_distance, weights=surface_distance_weight)) if surface_distance else None
        ),
        "candidate_entropy": weighted_mean(-np.sum(
            posterior.candidate_probabilities * np.log(np.maximum(posterior.candidate_probabilities, 1e-12)), axis=1
        )),
        "scene": {
            **scene_coverage,
            "retrieved_unique_maplets_at_64": int(np.sum(scene_score[top64] > 0.0)),
            "correct_maplets_at_64": int(correct.size),
            "correct_center_spread_m": spread,
            "bearing_rank": bearing_rank,
            "normal_rank": normal_gram_rank,
            "pose_sufficient_at_64": pose_sufficient,
        },
        "per_support": per_support,
    }
