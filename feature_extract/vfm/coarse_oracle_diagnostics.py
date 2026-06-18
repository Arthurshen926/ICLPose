"""Coarse correspondence oracle-rank diagnostics for render-pose refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from feature_extract.vfm.matcha_coarse_to_fine import _dual_softmax_confidence, feature_map_to_coarse_grid
from feature_extract.vfm.query_to_3d_matching import normalize_rows


def _rank_desc(values: np.ndarray, index: int) -> int:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    idx = int(index)
    if idx < 0 or idx >= vals.shape[0] or not np.isfinite(vals[idx]):
        return int(vals.shape[0] + 1)
    return int(np.count_nonzero(vals > vals[idx]) + 1)


def _cell_distance(index_a: int, index_b: int, width: int) -> int:
    row_a, col_a = divmod(int(index_a), int(width))
    row_b, col_b = divmod(int(index_b), int(width))
    return int(max(abs(row_a - row_b), abs(col_a - col_b)))


def coarse_oracle_rank_rows(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    oracle_render_indices: Sequence[int] | np.ndarray,
    query_image_width: int = 1,
    query_image_height: int = 1,
    render_image_width: int = 1,
    render_image_height: int = 1,
    logit_scale: float = 10.0,
) -> list[dict[str, Any]]:
    """Measure where each oracle render cell ranks in the coarse score matrix.

    ``oracle_render_indices`` is indexed by full query-cell index and should
    contain ``-1`` for invalid/invisible query cells.
    """

    query_grid = feature_map_to_coarse_grid(
        query_feature_map,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    render_grid = feature_map_to_coarse_grid(
        render_feature_map,
        image_width=int(render_image_width),
        image_height=int(render_image_height),
    )
    oracle = np.asarray(oracle_render_indices, dtype=np.int64).reshape(-1)
    if oracle.shape[0] != query_grid.xy.shape[0]:
        raise ValueError("oracle_render_indices must contain one value per query cell")
    qdesc, qvalid = normalize_rows(query_grid.descriptors)
    rdesc, rvalid = normalize_rows(render_grid.descriptors)
    qrows = np.flatnonzero(qvalid & (oracle >= 0) & (oracle < render_grid.xy.shape[0]))
    rrows = np.flatnonzero(rvalid)
    if qrows.size == 0 or rrows.size == 0:
        return []
    qlocal_by_index = {int(qidx): int(pos) for pos, qidx in enumerate(qrows)}
    rlocal_by_index = {int(ridx): int(pos) for pos, ridx in enumerate(rrows)}
    qdesc_local = qdesc[qrows]
    rdesc_local = rdesc[rrows]
    scores = qdesc_local @ rdesc_local.T
    confidence = _dual_softmax_confidence(scores, float(logit_scale))
    top_render_local = np.argmax(confidence, axis=1)
    rows: list[dict[str, Any]] = []
    for qidx in qrows:
        oracle_idx = int(oracle[int(qidx)])
        if oracle_idx not in rlocal_by_index:
            continue
        local_q = qlocal_by_index[int(qidx)]
        oracle_local = rlocal_by_index[oracle_idx]
        top_local = int(top_render_local[local_q])
        top_idx = int(rrows[top_local])
        sim_rank = _rank_desc(scores[local_q], oracle_local)
        conf_rank = _rank_desc(confidence[local_q], oracle_local)
        reciprocal_rank = _rank_desc(confidence[:, oracle_local], local_q)
        cell_distance = _cell_distance(top_idx, oracle_idx, int(render_grid.width))
        rows.append(
            {
                "query_index": int(qidx),
                "oracle_render_index": oracle_idx,
                "coarse_top1_render_index": top_idx,
                "cell_distance_top1_to_oracle": int(cell_distance),
                "oracle_rank_by_similarity": int(sim_rank),
                "oracle_rank_by_dual_softmax": int(conf_rank),
                "oracle_score": float(scores[local_q, oracle_local]),
                "oracle_confidence": float(confidence[local_q, oracle_local]),
                "top1_score": float(scores[local_q, top_local]),
                "top1_confidence": float(confidence[local_q, top_local]),
                "score_gap_top1_minus_oracle": float(scores[local_q, top_local] - scores[local_q, oracle_local]),
                "confidence_gap_top1_minus_oracle": float(confidence[local_q, top_local] - confidence[local_q, oracle_local]),
                "mutual_rank": int(reciprocal_rank),
                "mutual_dropped_oracle": bool(reciprocal_rank != 1),
                "coarse_hit_radius0": bool(cell_distance <= 0),
                "coarse_hit_radius1": bool(cell_distance <= 1),
                "coarse_hit_radius2": bool(cell_distance <= 2),
            }
        )
    return rows


def coarse_oracle_candidate_rows(
    query_feature_map: np.ndarray,
    render_feature_map: np.ndarray,
    *,
    oracle_render_indices: Sequence[int] | np.ndarray,
    top_k: int = 5,
    positive_radius: int = 0,
    query_image_width: int = 1,
    query_image_height: int = 1,
    render_image_width: int = 1,
    render_image_height: int = 1,
    logit_scale: float = 10.0,
) -> list[dict[str, Any]]:
    """Export top-K coarse candidate rows with oracle-cell distance labels."""

    query_grid = feature_map_to_coarse_grid(
        query_feature_map,
        image_width=int(query_image_width),
        image_height=int(query_image_height),
    )
    render_grid = feature_map_to_coarse_grid(
        render_feature_map,
        image_width=int(render_image_width),
        image_height=int(render_image_height),
    )
    oracle = np.asarray(oracle_render_indices, dtype=np.int64).reshape(-1)
    if oracle.shape[0] != query_grid.xy.shape[0]:
        raise ValueError("oracle_render_indices must contain one value per query cell")
    qdesc, qvalid = normalize_rows(query_grid.descriptors)
    rdesc, rvalid = normalize_rows(render_grid.descriptors)
    qrows = np.flatnonzero(qvalid & (oracle >= 0) & (oracle < render_grid.xy.shape[0]))
    rrows = np.flatnonzero(rvalid)
    if qrows.size == 0 or rrows.size == 0:
        return []
    rlocal_by_index = {int(ridx): int(pos) for pos, ridx in enumerate(rrows)}
    qdesc_local = qdesc[qrows]
    rdesc_local = rdesc[rrows]
    scores = qdesc_local @ rdesc_local.T
    confidence = _dual_softmax_confidence(scores, float(logit_scale))
    keep_k = max(1, int(top_k))
    rows: list[dict[str, Any]] = []
    for local_q, qidx in enumerate(qrows):
        oracle_idx = int(oracle[int(qidx)])
        if oracle_idx not in rlocal_by_index:
            continue
        order = np.lexsort((-scores[local_q], -confidence[local_q]))
        top1_local = int(order[0])
        top1_score = float(scores[local_q, top1_local])
        for rank, local_r in enumerate(order[:keep_k]):
            candidate_idx = int(rrows[int(local_r)])
            distance = _cell_distance(candidate_idx, oracle_idx, int(render_grid.width))
            reciprocal_rank = _rank_desc(confidence[:, int(local_r)], local_q) - 1
            positive = bool(distance <= int(positive_radius))
            query_xy = np.asarray(query_grid.xy[int(qidx)], dtype=np.float64).reshape(2)
            render_xy = np.asarray(render_grid.xy[candidate_idx], dtype=np.float64).reshape(2)
            rows.append(
                {
                    "query_index": int(qidx),
                    "render_index": candidate_idx,
                    "candidate_render_index": candidate_idx,
                    "oracle_render_index": oracle_idx,
                    "query_x": float(query_xy[0]),
                    "query_y": float(query_xy[1]),
                    "render_x": float(render_xy[0]),
                    "render_y": float(render_xy[1]),
                    "similarity": float(scores[local_q, int(local_r)]),
                    "similarity_margin": float(top1_score - scores[local_q, int(local_r)]),
                    "confidence": float(confidence[local_q, int(local_r)]),
                    "coarse_rank": int(rank),
                    "coarse_score": float(scores[local_q, int(local_r)]),
                    "coarse_score_gap": float(top1_score - scores[local_q, int(local_r)]),
                    "mutual_rank": int(reciprocal_rank),
                    "cell_delta_x": 0,
                    "cell_delta_y": 0,
                    "cell_distance_to_oracle": int(distance),
                    "gt_reproj_error_stride": 0.0 if positive else float(max(int(positive_radius) + 3, distance)),
                    "patch_correct": positive,
                    "patch_positive_label": positive,
                }
            )
    return rows


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


@dataclass
class CoarseOracleRankAccumulator:
    """Online summary for large coarse-oracle rank CSV exports."""

    count: int = 0
    hit_radius0: int = 0
    hit_radius1: int = 0
    hit_radius2: int = 0
    rank_at_1: int = 0
    rank_at_3: int = 0
    rank_at_5: int = 0
    rank_at_10: int = 0
    rank_at_20: int = 0
    mutual_dropped: int = 0
    score_gap_sum: float = 0.0
    score_gap_count: int = 0
    ranks: list[int] = field(default_factory=list)

    def update(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            rank = int(row["oracle_rank_by_dual_softmax"])
            self.count += 1
            self.ranks.append(rank)
            self.hit_radius0 += int(bool(row.get("coarse_hit_radius0", False)))
            self.hit_radius1 += int(bool(row.get("coarse_hit_radius1", False)))
            self.hit_radius2 += int(bool(row.get("coarse_hit_radius2", False)))
            self.rank_at_1 += int(rank <= 1)
            self.rank_at_3 += int(rank <= 3)
            self.rank_at_5 += int(rank <= 5)
            self.rank_at_10 += int(rank <= 10)
            self.rank_at_20 += int(rank <= 20)
            self.mutual_dropped += int(bool(row.get("mutual_dropped_oracle", False)))
            gap = row.get("score_gap_top1_minus_oracle")
            if gap is not None:
                gap_value = float(gap)
                if np.isfinite(gap_value):
                    self.score_gap_sum += gap_value
                    self.score_gap_count += 1

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "coarse_oracle_count": 0,
                "coarse_hit_radius0": None,
                "coarse_hit_radius1": None,
                "coarse_hit_radius2": None,
                "oracle_rank_at_1": None,
                "oracle_rank_at_3": None,
                "oracle_rank_at_5": None,
                "oracle_rank_at_10": None,
                "oracle_rank_at_20": None,
                "median_oracle_rank": None,
                "mean_score_gap_top1_minus_oracle": None,
                "mutual_dropped_oracle_rate": None,
            }
        denom = float(self.count)
        return {
            "coarse_oracle_count": int(self.count),
            "coarse_hit_radius0": float(self.hit_radius0 / denom),
            "coarse_hit_radius1": float(self.hit_radius1 / denom),
            "coarse_hit_radius2": float(self.hit_radius2 / denom),
            "oracle_rank_at_1": float(self.rank_at_1 / denom),
            "oracle_rank_at_3": float(self.rank_at_3 / denom),
            "oracle_rank_at_5": float(self.rank_at_5 / denom),
            "oracle_rank_at_10": float(self.rank_at_10 / denom),
            "oracle_rank_at_20": float(self.rank_at_20 / denom),
            "median_oracle_rank": float(np.median(np.asarray(self.ranks, dtype=np.float64))),
            "mean_score_gap_top1_minus_oracle": (
                float(self.score_gap_sum / float(self.score_gap_count)) if self.score_gap_count else None
            ),
            "mutual_dropped_oracle_rate": float(self.mutual_dropped / denom),
        }


def summarize_coarse_oracle_rank_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    values = list(rows)
    count = len(values)
    if count == 0:
        return {
            "coarse_oracle_count": 0,
            "coarse_hit_radius0": None,
            "coarse_hit_radius1": None,
            "coarse_hit_radius2": None,
            "oracle_rank_at_1": None,
            "oracle_rank_at_3": None,
            "oracle_rank_at_5": None,
            "oracle_rank_at_10": None,
            "oracle_rank_at_20": None,
            "median_oracle_rank": None,
            "mean_score_gap_top1_minus_oracle": None,
            "mutual_dropped_oracle_rate": None,
        }
    ranks = np.asarray([int(row["oracle_rank_by_dual_softmax"]) for row in values], dtype=np.float64)
    gap_values = [
        float(row["score_gap_top1_minus_oracle"])
        for row in values
        if row.get("score_gap_top1_minus_oracle") is not None
    ]
    gaps = np.asarray(gap_values, dtype=np.float64)
    return {
        "coarse_oracle_count": int(count),
        "coarse_hit_radius0": _mean_bool(values, "coarse_hit_radius0"),
        "coarse_hit_radius1": _mean_bool(values, "coarse_hit_radius1"),
        "coarse_hit_radius2": _mean_bool(values, "coarse_hit_radius2"),
        "oracle_rank_at_1": float(np.mean(ranks <= 1)),
        "oracle_rank_at_3": float(np.mean(ranks <= 3)),
        "oracle_rank_at_5": float(np.mean(ranks <= 5)),
        "oracle_rank_at_10": float(np.mean(ranks <= 10)),
        "oracle_rank_at_20": float(np.mean(ranks <= 20)),
        "median_oracle_rank": float(np.median(ranks)),
        "mean_score_gap_top1_minus_oracle": float(np.mean(gaps)) if gaps.size else None,
        "mutual_dropped_oracle_rate": _mean_bool(values, "mutual_dropped_oracle"),
    }
