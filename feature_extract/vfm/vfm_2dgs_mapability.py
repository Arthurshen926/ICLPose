"""Mapping-only diagnostics for VFM-2DGS anchor maps."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap, Vfm2DgsObservationBank


def _stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
    }


def _weighted_iou_dict(
    first_ids: np.ndarray,
    first_weights: np.ndarray,
    second_ids: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    first = {int(idx): float(weight) for idx, weight in zip(first_ids.tolist(), first_weights.tolist())}
    second = {int(idx): float(weight) for idx, weight in zip(second_ids.tolist(), second_weights.tolist())}
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    numerator = sum(min(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    denominator = sum(max(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    return float(numerator / max(denominator, 1e-12))


def assign_observations_to_anchors(
    anchor_map: Vfm2DgsAnchorMap,
    observation_bank: Vfm2DgsObservationBank,
    min_support_iou: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each token observation to the anchor with maximum support IoU."""
    assigned_anchor_ids = np.full((len(observation_bank),), -1, dtype=np.int64)
    assigned_ious = np.zeros((len(observation_bank),), dtype=np.float32)
    if len(anchor_map) == 0 or len(observation_bank) == 0:
        return assigned_anchor_ids, assigned_ious
    anchor_support = []
    for row in range(len(anchor_map)):
        start = int(anchor_map.support_offsets[row])
        end = int(anchor_map.support_offsets[row + 1])
        anchor_support.append((anchor_map.support_element_ids[start:end], anchor_map.support_weights[start:end]))
    for obs_idx in range(len(observation_bank)):
        obs_start = int(observation_bank.support_offsets[obs_idx])
        obs_end = int(observation_bank.support_offsets[obs_idx + 1])
        obs_ids = observation_bank.element_ids[obs_start:obs_end]
        obs_weights = observation_bank.element_weights[obs_start:obs_end]
        best_row = -1
        best_iou = 0.0
        for anchor_row, (anchor_ids, anchor_weights) in enumerate(anchor_support):
            iou = _weighted_iou_dict(obs_ids, obs_weights, anchor_ids, anchor_weights)
            if iou > best_iou:
                best_iou = iou
                best_row = anchor_row
        if best_row >= 0 and best_iou >= float(min_support_iou):
            assigned_anchor_ids[obs_idx] = int(anchor_map.anchor_ids[best_row])
            assigned_ious[obs_idx] = float(best_iou)
    return assigned_anchor_ids, assigned_ious


def _leave_one_out_retrieval(
    observation_bank: Vfm2DgsObservationBank,
    assigned_anchor_ids: np.ndarray,
    top_ks: Sequence[int] = (1, 5),
    query_rows: Sequence[int] | None = None,
) -> dict[str, float | int]:
    features = np.asarray(observation_bank.features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return {
            "query_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mean_positive_cosine": 0.0,
        }
    features, _valid = normalize_rows(features)
    anchors = sorted(int(item) for item in set(np.asarray(assigned_anchor_ids, dtype=np.int64).tolist()) if int(item) >= 0)
    if not anchors:
        return {
            "query_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mean_positive_cosine": 0.0,
        }
    rows_by_anchor: dict[int, list[int]] = defaultdict(list)
    for row, anchor_id in enumerate(assigned_anchor_ids.tolist()):
        if int(anchor_id) >= 0:
            rows_by_anchor[int(anchor_id)].append(int(row))

    ranks: list[int] = []
    positive_scores: list[float] = []
    recall_hits = {int(k): 0 for k in top_ks}
    if query_rows is None:
        query_indices = range(len(assigned_anchor_ids))
    else:
        query_indices = [int(row) for row in query_rows]
    for query_row in query_indices:
        positive_anchor = int(assigned_anchor_ids[int(query_row)])
        positive_anchor = int(positive_anchor)
        if positive_anchor < 0:
            continue
        prototype_rows = []
        prototype_anchor_ids = []
        for anchor_id in anchors:
            rows = [row for row in rows_by_anchor[anchor_id] if row != int(query_row)]
            if not rows:
                continue
            proto = np.mean(features[np.asarray(rows, dtype=np.int64)], axis=0)
            norm = float(np.linalg.norm(proto))
            if norm <= 1e-8:
                continue
            prototype_rows.append(proto / norm)
            prototype_anchor_ids.append(int(anchor_id))
        if positive_anchor not in prototype_anchor_ids:
            continue
        prototypes = np.stack(prototype_rows, axis=0).astype(np.float32)
        scores = prototypes @ features[query_row].reshape(-1, 1)
        scores = scores.reshape(-1)
        order = np.argsort(-scores, kind="mergesort")
        ranked_anchor_ids = [prototype_anchor_ids[int(idx)] for idx in order.tolist()]
        rank = int(ranked_anchor_ids.index(positive_anchor) + 1)
        ranks.append(rank)
        positive_scores.append(float(scores[prototype_anchor_ids.index(positive_anchor)]))
        for k in recall_hits:
            if rank <= int(k):
                recall_hits[k] += 1
    query_count = len(ranks)
    if query_count == 0:
        return {
            "query_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mean_positive_cosine": 0.0,
        }
    result: dict[str, float | int] = {
        "query_count": int(query_count),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_positive_cosine": float(np.mean(positive_scores)),
    }
    for k, hits in sorted(recall_hits.items()):
        result[f"recall_at_{int(k)}"] = float(hits / max(query_count, 1))
    result.setdefault("recall_at_1", 0.0)
    result.setdefault("recall_at_5", 0.0)
    return result


def _source_breakdown(
    observation_bank: Vfm2DgsObservationBank,
    assigned_anchor_ids: np.ndarray,
    assigned_ious: np.ndarray,
) -> dict[str, object]:
    sources = tuple(str(item) for item in observation_bank.source_ids)
    named_sources = sorted(set(item for item in sources if item))
    if not named_sources:
        return {}
    output: dict[str, object] = {}
    for source_name in named_sources:
        rows = [idx for idx, item in enumerate(sources) if item == source_name]
        assigned = assigned_anchor_ids[np.asarray(rows, dtype=np.int64)] >= 0 if rows else np.zeros((0,), dtype=bool)
        source_ious = assigned_ious[np.asarray(rows, dtype=np.int64)] if rows else np.zeros((0,), dtype=np.float32)
        output[source_name] = {
            "observation_count": int(len(rows)),
            "assigned_observation_count": int(np.sum(assigned)),
            "assigned_fraction": float(np.mean(assigned)) if assigned.size else 0.0,
            "support_iou_stats": _stats(source_ious[assigned]),
            "retrieval": _leave_one_out_retrieval(
                observation_bank,
                assigned_anchor_ids,
                query_rows=rows,
            ),
        }
    return output


def _coverage_stats(observation_bank: Vfm2DgsObservationBank, token_grid_shape: tuple[int, int] | None) -> dict[str, float | int]:
    view_to_rows: dict[str, list[int]] = defaultdict(list)
    for row, image_id in enumerate(observation_bank.image_ids):
        view_to_rows[str(image_id)].append(int(row))
    per_view_counts = np.asarray([len(rows) for rows in view_to_rows.values()], dtype=np.float32)
    output: dict[str, float | int] = {
        "view_count": int(len(view_to_rows)),
        "mean_observations_per_view": float(np.mean(per_view_counts)) if per_view_counts.size else 0.0,
        "median_observations_per_view": float(np.median(per_view_counts)) if per_view_counts.size else 0.0,
    }
    if token_grid_shape is not None:
        height, width = int(token_grid_shape[0]), int(token_grid_shape[1])
        total = max(height * width, 1)
        occupancies = []
        for rows in view_to_rows.values():
            tokens = set(int(observation_bank.token_indices[row]) for row in rows)
            occupancies.append(len(tokens) / float(total))
        output["mean_token_grid_occupancy"] = float(np.mean(occupancies)) if occupancies else 0.0
        output["median_token_grid_occupancy"] = float(np.median(occupancies)) if occupancies else 0.0
    return output


def _surface_seed_consensus(
    anchor_map: Vfm2DgsAnchorMap,
    observation_bank: Vfm2DgsObservationBank,
    surface_elements: SurfaceElementMap,
    min_views: int = 2,
) -> dict[str, float | int]:
    row_by_id = surface_elements.row_by_element_id
    views_by_element: dict[int, set[str]] = defaultdict(set)
    views_by_parent: dict[int, set[str]] = defaultdict(set)
    obs_by_element: dict[int, int] = defaultdict(int)
    obs_by_parent: dict[int, int] = defaultdict(int)
    for obs_row, image_id in enumerate(observation_bank.image_ids):
        start = int(observation_bank.support_offsets[obs_row])
        end = int(observation_bank.support_offsets[obs_row + 1])
        seen_elements = set(int(item) for item in observation_bank.element_ids[start:end].tolist())
        seen_parents = set()
        for element_id in seen_elements:
            if element_id not in row_by_id:
                continue
            parent_id = int(surface_elements.parent_gaussian_indices[row_by_id[element_id]])
            views_by_element[element_id].add(str(image_id))
            obs_by_element[element_id] += 1
            seen_parents.add(parent_id)
        for parent_id in seen_parents:
            views_by_parent[parent_id].add(str(image_id))
            obs_by_parent[parent_id] += 1

    anchor_obs_by_element: dict[int, int] = defaultdict(int)
    anchor_obs_by_parent: dict[int, int] = defaultdict(int)
    for anchor_row in range(len(anchor_map)):
        start = int(anchor_map.support_offsets[anchor_row])
        end = int(anchor_map.support_offsets[anchor_row + 1])
        obs_count = int(anchor_map.observation_counts[anchor_row])
        parent_ids = set()
        for element_id in anchor_map.support_element_ids[start:end].astype(np.int64).tolist():
            element_id = int(element_id)
            anchor_obs_by_element[element_id] = max(anchor_obs_by_element[element_id], obs_count)
            if element_id in row_by_id:
                parent_ids.add(int(surface_elements.parent_gaussian_indices[row_by_id[element_id]]))
        for parent_id in parent_ids:
            anchor_obs_by_parent[parent_id] = max(anchor_obs_by_parent[parent_id], obs_count)

    multi_view_elements = [key for key, views in views_by_element.items() if len(views) >= int(min_views)]
    multi_view_parents = [key for key, views in views_by_parent.items() if len(views) >= int(min_views)]
    element_hits = [key for key in multi_view_elements if anchor_obs_by_element.get(key, 0) >= int(min_views)]
    parent_hits = [key for key in multi_view_parents if anchor_obs_by_parent.get(key, 0) >= int(min_views)]
    return {
        "min_views": int(min_views),
        "element_seed_count": int(len(views_by_element)),
        "parent_seed_count": int(len(views_by_parent)),
        "multi_view_element_seed_count": int(len(multi_view_elements)),
        "multi_view_parent_seed_count": int(len(multi_view_parents)),
        "multi_view_element_anchor_recall": float(len(element_hits) / max(len(multi_view_elements), 1)),
        "multi_view_parent_anchor_recall": float(len(parent_hits) / max(len(multi_view_parents), 1)),
        "element_observation_count_stats": _stats(np.asarray(list(obs_by_element.values()), dtype=np.float32)),
        "parent_observation_count_stats": _stats(np.asarray(list(obs_by_parent.values()), dtype=np.float32)),
        "element_view_count_stats": _stats(np.asarray([len(views) for views in views_by_element.values()], dtype=np.float32)),
        "parent_view_count_stats": _stats(np.asarray([len(views) for views in views_by_parent.values()], dtype=np.float32)),
    }


def evaluate_vfm_2dgs_mapability(
    anchor_map: Vfm2DgsAnchorMap,
    observation_bank: Vfm2DgsObservationBank,
    token_grid_shape: tuple[int, int] | None = None,
    min_support_iou: float = 1e-6,
    surface_elements: SurfaceElementMap | None = None,
    surface_seed_min_views: int = 2,
) -> dict[str, object]:
    assigned_anchor_ids, assigned_ious = assign_observations_to_anchors(
        anchor_map,
        observation_bank,
        min_support_iou=float(min_support_iou),
    )
    assigned = assigned_anchor_ids >= 0
    result: dict[str, object] = {
        "assignment": {
            "observation_count": int(len(observation_bank)),
            "assigned_observation_count": int(np.sum(assigned)),
            "assigned_fraction": float(np.mean(assigned)) if assigned.size else 0.0,
            "support_iou_stats": _stats(assigned_ious[assigned]),
        },
        "retrieval": _leave_one_out_retrieval(observation_bank, assigned_anchor_ids),
        "coverage": _coverage_stats(observation_bank, token_grid_shape),
        "anchors": {
            "anchor_count": int(len(anchor_map)),
            "observation_count_stats": _stats(anchor_map.observation_counts),
            "support_count_stats": _stats(anchor_map.surface_support_counts),
            "quality_stats": _stats(anchor_map.quality_scores),
            "purity_stats": _stats(anchor_map.purity_scores),
            "feature_variance_stats": _stats(anchor_map.feature_variances),
            "distinctiveness_stats": _stats(anchor_map.distinctiveness_scores),
            "stability_stats": _stats(anchor_map.stability_scores),
        },
        "observations": {
            "support_count_stats": _stats(np.diff(observation_bank.support_offsets)),
            "purity_stats": _stats(observation_bank.purity_scores),
            "component_concentration_stats": _stats(observation_bank.component_concentrations),
            "quality_stats": _stats(observation_bank.quality_scores),
        },
    }
    source_breakdown = _source_breakdown(observation_bank, assigned_anchor_ids, assigned_ious)
    if source_breakdown:
        result["source_breakdown"] = source_breakdown
    if surface_elements is not None:
        result["surface_seed_consensus"] = _surface_seed_consensus(
            anchor_map,
            observation_bank,
            surface_elements,
            min_views=int(surface_seed_min_views),
        )
    return result
