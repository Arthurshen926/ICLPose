"""Oracle diagnostics for query-observation to 3D-landmark descriptor recall."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from feature_extract.vfm.query_to_3d_matching import normalize_rows


@dataclass(frozen=True)
class LandmarkRecallRecord:
    query_id: str
    correct_track_id: int
    correct_rank: int | None
    correct_score: float | None
    rank1_track_id: int | None
    rank1_score: float | None
    score_gap_to_rank1: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def rank_correct_landmark(
    *,
    query_descriptor: np.ndarray,
    landmark_features: np.ndarray,
    landmark_track_ids: np.ndarray,
    correct_track_id: int,
    query_id: str = "",
    landmark_features_are_normalized: bool = False,
    valid_landmark_mask: np.ndarray | None = None,
) -> LandmarkRecallRecord:
    """Rank one GT landmark among a descriptor bank using cosine similarity."""

    features = np.asarray(landmark_features, dtype=np.float32)
    track_ids = np.asarray(landmark_track_ids, dtype=np.int64).reshape(-1)
    if features.ndim != 2:
        raise ValueError("landmark_features must have shape (N, C)")
    if track_ids.shape[0] != features.shape[0]:
        raise ValueError("landmark_track_ids must have one id per feature")
    query = np.asarray(query_descriptor, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != features.shape[1]:
        raise ValueError("query descriptor and landmark features must have the same dimension")
    norm_query, valid_query = normalize_rows(query)
    if landmark_features_are_normalized:
        norm_features = features
        if valid_landmark_mask is None:
            _normed, valid_features = normalize_rows(features)
        else:
            valid_features = np.asarray(valid_landmark_mask, dtype=bool).reshape(-1)
            if valid_features.shape[0] != features.shape[0]:
                raise ValueError("valid_landmark_mask must contain one value per landmark feature")
    else:
        norm_features, valid_features = normalize_rows(features)
    if not bool(valid_query[0]) or not np.any(valid_features):
        return LandmarkRecallRecord(str(query_id), int(correct_track_id), None, None, None, None, None)
    scores = (norm_features @ norm_query[0].astype(np.float32)).astype(np.float64)
    scores[~valid_features] = -np.inf
    rank1_index = int(np.argmax(scores)) if scores.size else None
    if rank1_index is not None and not np.isfinite(scores[rank1_index]):
        rank1_index = None
    rank1_track_id = None if rank1_index is None else int(track_ids[rank1_index])
    rank1_score = None if rank1_index is None else float(scores[rank1_index])

    correct_indices = np.flatnonzero((track_ids == int(correct_track_id)) & valid_features)
    if correct_indices.size == 0:
        return LandmarkRecallRecord(
            str(query_id),
            int(correct_track_id),
            None,
            None,
            rank1_track_id,
            rank1_score,
            None,
        )
    best_rank: int | None = None
    correct_index: int | None = None
    for candidate_index in correct_indices:
        score = float(scores[int(candidate_index)])
        if not np.isfinite(score):
            continue
        higher = int(np.sum(scores > score))
        tied_before = int(np.sum((scores == score) & (np.arange(scores.shape[0]) < int(candidate_index))))
        rank = int(higher + tied_before + 1)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            correct_index = int(candidate_index)
    if correct_index is None or best_rank is None:
        return LandmarkRecallRecord(str(query_id), int(correct_track_id), None, None, rank1_track_id, rank1_score, None)
    correct_score = float(scores[correct_index])
    score_gap = None if rank1_score is None else float(rank1_score - correct_score)
    return LandmarkRecallRecord(
        str(query_id),
        int(correct_track_id),
        int(best_rank),
        correct_score,
        rank1_track_id,
        rank1_score,
        score_gap,
    )


def summarize_landmark_recall_records(
    records: Sequence[LandmarkRecallRecord],
    *,
    top_ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
) -> dict[str, object]:
    values = list(records)
    ranks = [record.correct_rank for record in values if record.correct_rank is not None]
    summary: dict[str, object] = {
        "sample_count": int(len(values)),
        "found_count": int(len(ranks)),
        "missing_count": int(len(values) - len(ranks)),
        "median_correct_rank": None if not ranks else float(np.median(np.asarray(ranks, dtype=np.float64))),
    }
    for k in top_ks:
        kk = int(k)
        if kk <= 0:
            raise ValueError("top_ks must contain positive integers")
        summary[f"recall_at_{kk}"] = (
            0.0 if not values else float(np.mean([(rank is not None and rank <= kk) for rank in (r.correct_rank for r in values)]))
        )
    gaps = [record.score_gap_to_rank1 for record in values if record.score_gap_to_rank1 is not None]
    summary["median_score_gap_to_rank1"] = None if not gaps else float(np.median(np.asarray(gaps, dtype=np.float64)))
    return summary
