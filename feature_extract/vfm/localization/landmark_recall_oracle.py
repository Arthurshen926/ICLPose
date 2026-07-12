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
    query_x: float | None = None
    query_y: float | None = None
    landmark_x: float | None = None
    landmark_y: float | None = None
    landmark_z: float | None = None

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
    if np.unique(track_ids).size != track_ids.size:
        # Multi-prototype banks are evaluated in track space. A track receives
        # the score of its best prototype and therefore occupies one rank.
        if np.any(track_ids[1:] < track_ids[:-1]):
            order = np.argsort(track_ids, kind="stable")
            track_ids = track_ids[order]
            scores = scores[order]
        starts = np.flatnonzero(np.r_[True, track_ids[1:] != track_ids[:-1]])
        scores = np.maximum.reduceat(scores, starts)
        track_ids = track_ids[starts]
        valid_features = np.isfinite(scores)
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


def rank_correct_landmarks_batch(
    *,
    query_descriptors: np.ndarray,
    landmark_features: np.ndarray,
    landmark_track_ids: np.ndarray,
    correct_track_ids: Sequence[int],
    query_ids: Sequence[str] | None = None,
    landmark_features_are_normalized: bool = False,
    valid_landmark_mask: np.ndarray | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> list[LandmarkRecallRecord]:
    """Compute exact track-level ranks in batches, including multi-prototype banks."""

    import torch

    queries = np.asarray(query_descriptors, dtype=np.float32)
    features = np.asarray(landmark_features, dtype=np.float32)
    track_ids = np.asarray(landmark_track_ids, dtype=np.int64).reshape(-1)
    correct_ids = np.asarray(correct_track_ids, dtype=np.int64).reshape(-1)
    if queries.ndim != 2 or features.ndim != 2:
        raise ValueError("query_descriptors and landmark_features must have shape (N, C)")
    if queries.shape[1] != features.shape[1]:
        raise ValueError("query and landmark descriptor dimensions must match")
    if track_ids.shape[0] != features.shape[0]:
        raise ValueError("landmark_track_ids must contain one id per landmark feature")
    if correct_ids.shape[0] != queries.shape[0]:
        raise ValueError("correct_track_ids must contain one id per query descriptor")
    if query_ids is None:
        ids = tuple("" for _ in range(queries.shape[0]))
    else:
        ids = tuple(str(item) for item in query_ids)
        if len(ids) != queries.shape[0]:
            raise ValueError("query_ids must contain one id per query descriptor")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")

    normalized_queries, valid_queries = normalize_rows(queries)
    if landmark_features_are_normalized:
        normalized_features = features
        if valid_landmark_mask is None:
            _normalized, valid_features = normalize_rows(features)
        else:
            valid_features = np.asarray(valid_landmark_mask, dtype=bool).reshape(-1)
            if valid_features.shape[0] != features.shape[0]:
                raise ValueError("valid_landmark_mask must contain one value per landmark feature")
    else:
        normalized_features, valid_features = normalize_rows(features)

    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    feature_tensor = torch.as_tensor(normalized_features, dtype=torch.float32, device=requested)
    valid_feature_tensor = torch.as_tensor(valid_features, dtype=torch.bool, device=requested)
    unique_track_ids, track_inverse = np.unique(track_ids, return_inverse=True)
    has_multiple_prototypes = bool(unique_track_ids.size != track_ids.size)
    score_track_ids = unique_track_ids if has_multiple_prototypes else track_ids
    track_position = {int(track_id): int(index) for index, track_id in enumerate(score_track_ids.tolist())}
    track_inverse_tensor = (
        torch.as_tensor(track_inverse, dtype=torch.long, device=requested) if has_multiple_prototypes else None
    )
    score_positions = torch.arange(int(score_track_ids.size), dtype=torch.long, device=requested)

    output: list[LandmarkRecallRecord] = []
    with torch.no_grad():
        for start in range(0, int(queries.shape[0]), int(batch_size)):
            end = min(start + int(batch_size), int(queries.shape[0]))
            query_tensor = torch.as_tensor(normalized_queries[start:end], dtype=torch.float32, device=requested)
            scores = query_tensor @ feature_tensor.T
            scores[:, ~valid_feature_tensor] = -torch.inf
            if track_inverse_tensor is not None:
                track_scores = torch.full(
                    (end - start, int(unique_track_ids.size)),
                    -torch.inf,
                    dtype=torch.float32,
                    device=requested,
                )
                track_scores.scatter_reduce_(
                    1,
                    track_inverse_tensor[None, :].expand(end - start, -1),
                    scores,
                    reduce="amax",
                    include_self=True,
                )
            else:
                track_scores = scores
            for local_index, global_index in enumerate(range(start, end)):
                query_id = ids[global_index]
                correct_track_id = int(correct_ids[global_index])
                row = track_scores[local_index]
                finite = torch.isfinite(row)
                if not bool(valid_queries[global_index]) or not bool(torch.any(finite)):
                    output.append(
                        LandmarkRecallRecord(query_id, correct_track_id, None, None, None, None, None)
                    )
                    continue
                rank1_position = int(torch.argmax(row).item())
                rank1_track_id = int(score_track_ids[rank1_position])
                rank1_score = float(row[rank1_position].item())
                correct_position = track_position.get(correct_track_id)
                if correct_position is None or not bool(finite[correct_position]):
                    output.append(
                        LandmarkRecallRecord(
                            query_id,
                            correct_track_id,
                            None,
                            None,
                            rank1_track_id,
                            rank1_score,
                            None,
                        )
                    )
                    continue
                correct_score_tensor = row[correct_position]
                correct_score = float(correct_score_tensor.item())
                higher = int(torch.sum(row > correct_score_tensor).item())
                tied_before = int(
                    torch.sum((row == correct_score_tensor) & (score_positions < int(correct_position))).item()
                )
                output.append(
                    LandmarkRecallRecord(
                        query_id,
                        correct_track_id,
                        int(higher + tied_before + 1),
                        correct_score,
                        rank1_track_id,
                        rank1_score,
                        float(rank1_score - correct_score),
                    )
                )
    return output


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
        "mean_correct_rank_found": None if not ranks else float(np.mean(np.asarray(ranks, dtype=np.float64))),
        "mean_reciprocal_rank": (
            0.0
            if not values
            else float(np.mean([0.0 if record.correct_rank is None else 1.0 / float(record.correct_rank) for record in values]))
        ),
    }
    query_ids = sorted({str(record.query_id) for record in values})
    for k in top_ks:
        kk = int(k)
        if kk <= 0:
            raise ValueError("top_ks must contain positive integers")
        summary[f"recall_at_{kk}"] = (
            0.0 if not values else float(np.mean([(rank is not None and rank <= kk) for rank in (r.correct_rank for r in values)]))
        )
        per_query = []
        for query_id in query_ids:
            query_records = [record for record in values if str(record.query_id) == query_id]
            per_query.append(
                float(np.mean([record.correct_rank is not None and record.correct_rank <= kk for record in query_records]))
            )
        summary[f"macro_query_recall_at_{kk}"] = 0.0 if not per_query else float(np.mean(per_query))
    gaps = [record.score_gap_to_rank1 for record in values if record.score_gap_to_rank1 is not None]
    summary["median_score_gap_to_rank1"] = None if not gaps else float(np.median(np.asarray(gaps, dtype=np.float64)))
    return summary
