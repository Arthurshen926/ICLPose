"""Inference-safe assignment and selection of 2D-3D matches before PnP."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch


def resolve_global_query_track_assignment(
    candidate_track_ids: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    dustbin_score: float | np.ndarray | None = None,
) -> np.ndarray:
    """Resolve a whole-image sparse query-track bipartite assignment.

    The returned value contains one candidate column per query row, or ``-1``
    for dustbin. Each physical track can be selected at most once. Every query
    receives its own dustbin column, so unmatched queries do not compete with
    one another.

    ``dustbin_score=None`` means "prefer any finite candidate", while still
    allowing dustbin when all of a row's tracks are consumed by better rows.
    A scalar or one value per row gives an explicit no-match threshold in the
    same score space as ``candidate_scores``.
    """

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if tracks.ndim != 2 or scores.shape != tracks.shape:
        raise ValueError("candidate tracks and scores must have equal shape (Nq, L)")
    query_count = int(tracks.shape[0])
    if query_count == 0:
        return np.empty((0,), dtype=np.int64)
    valid = (tracks >= 0) & np.isfinite(scores)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != tracks.shape:
            raise ValueError("valid_mask must match candidate arrays")
        valid &= supplied

    finite_candidate_scores = scores[valid]
    if dustbin_score is None:
        if finite_candidate_scores.size:
            span = max(float(np.ptp(finite_candidate_scores)), 1.0)
            dustbins = np.full(
                (query_count,),
                float(np.min(finite_candidate_scores) - span),
                dtype=np.float64,
            )
        else:
            dustbins = np.zeros((query_count,), dtype=np.float64)
    else:
        dustbins = np.asarray(dustbin_score, dtype=np.float64)
        if dustbins.ndim == 0:
            dustbins = np.full((query_count,), float(dustbins), dtype=np.float64)
        else:
            dustbins = dustbins.reshape(-1)
            if dustbins.shape != (query_count,):
                raise ValueError("dustbin_score must be scalar or contain one value per query")
        if not np.all(np.isfinite(dustbins)):
            raise ValueError("explicit dustbin scores must be finite")

    unique_tracks = np.unique(tracks[valid])
    track_to_column = {int(track): column for column, track in enumerate(unique_tracks.tolist())}
    allowed_values = np.concatenate([finite_candidate_scores, dustbins])
    value_span = max(float(np.ptp(allowed_values)), 1.0)
    forbidden = float(np.min(allowed_values) - value_span * (query_count + 1))
    objective = np.full(
        (query_count, len(unique_tracks) + query_count),
        forbidden,
        dtype=np.float64,
    )
    source_columns = np.full(
        (query_count, len(unique_tracks)), -1, dtype=np.int64
    )

    # Multiple prototypes of one physical track are collapsed to the best
    # candidate edge before enforcing the cross-query one-to-one constraint.
    for query_row in range(query_count):
        for candidate_column in np.flatnonzero(valid[query_row]).tolist():
            track_column = track_to_column[int(tracks[query_row, candidate_column])]
            score = float(scores[query_row, candidate_column])
            previous = int(source_columns[query_row, track_column])
            if score > float(objective[query_row, track_column]) or (
                score == float(objective[query_row, track_column])
                and (previous < 0 or int(candidate_column) < previous)
            ):
                objective[query_row, track_column] = score
                source_columns[query_row, track_column] = int(candidate_column)
        objective[query_row, len(unique_tracks) + query_row] = float(
            dustbins[query_row]
        )

    row_indices, assignment_columns = linear_sum_assignment(objective, maximize=True)
    if not np.array_equal(row_indices, np.arange(query_count, dtype=np.int64)):
        raise RuntimeError("global assignment did not return one decision per query")
    selected = np.full((query_count,), -1, dtype=np.int64)
    is_track = assignment_columns < len(unique_tracks)
    selected[is_track] = source_columns[
        row_indices[is_track], assignment_columns[is_track]
    ]
    if np.any(selected[is_track] < 0):
        raise RuntimeError("global assignment selected a forbidden query-track edge")
    chosen = selected >= 0
    if len(np.unique(tracks[np.arange(query_count)[chosen], selected[chosen]])) != int(
        np.sum(chosen)
    ):
        raise RuntimeError("global assignment returned duplicate physical tracks")
    return selected


def global_assignment_score_matrix(
    candidate_track_ids: np.ndarray,
    candidate_scores: np.ndarray,
    query_ids: Sequence[str],
    *,
    valid_mask: np.ndarray | None = None,
    dustbin_score: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve independent whole-image assignments and retain chosen scores."""

    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    ids = np.asarray([str(value) for value in query_ids], dtype=np.str_)
    if tracks.ndim != 2 or scores.shape != tracks.shape or ids.shape != (len(tracks),):
        raise ValueError("global assignment inputs have incompatible shapes")
    valid = None if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid is not None and valid.shape != tracks.shape:
        raise ValueError("valid_mask must match candidate arrays")
    resolved = np.full(scores.shape, -np.inf, dtype=np.float32)
    selected = np.full((len(tracks),), -1, dtype=np.int64)
    for query_id in dict.fromkeys(ids.tolist()):
        rows = np.flatnonzero(ids == str(query_id))
        local = resolve_global_query_track_assignment(
            tracks[rows],
            scores[rows],
            valid_mask=None if valid is None else valid[rows],
            dustbin_score=dustbin_score,
        )
        selected[rows] = local
        accepted = local >= 0
        accepted_rows = rows[accepted]
        accepted_columns = local[accepted]
        resolved[accepted_rows, accepted_columns] = scores[
            accepted_rows, accepted_columns
        ].astype(np.float32)
    return resolved, selected


def stable_uniform_ransac_order(
    matches: Sequence[QueryTo3DMatch],
) -> list[QueryTo3DMatch]:
    """Canonical solver order that is independent of learned confidence."""

    return sorted(
        matches,
        key=lambda item: (
            int(item.token_index),
            int(item.track_id),
            -1 if item.prototype_id is None else int(item.prototype_id),
        ),
    )


def resolve_pose_match_conflicts(
    matches: Sequence[QueryTo3DMatch],
) -> list[QueryTo3DMatch]:
    """Resolve duplicate query/track hypotheses by confidence before PnP."""

    ordered = sorted(
        matches,
        key=lambda item: (
            -float(item.similarity),
            int(item.token_index),
            int(item.track_id),
        ),
    )
    unique: list[QueryTo3DMatch] = []
    seen_tokens: set[int] = set()
    seen_tracks: set[int] = set()
    for match in ordered:
        token = int(match.token_index)
        track = int(match.track_id)
        if token in seen_tokens or track in seen_tracks:
            continue
        seen_tokens.add(token)
        seen_tracks.add(track)
        unique.append(match)
    return unique


def select_pose_safe_matches(
    matches: Sequence[QueryTo3DMatch],
    *,
    max_matches: int,
    image_width: int,
    image_height: int,
    mode: str = "spatial_round_robin",
    grid_rows: int = 4,
    grid_cols: int = 4,
    min_matches: int | None = None,
    min_confidence: float | None = None,
) -> list[QueryTo3DMatch]:
    """Resolve conflicts and retain a confidence/coverage-safe adaptive subset."""

    limit = int(max_matches)
    if limit <= 0 or int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("max_matches and image dimensions must be positive")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    if str(mode) not in {"score_topk", "spatial_round_robin"}:
        raise ValueError("unsupported pose-safe selection mode")
    floor = limit if min_matches is None else int(min_matches)
    if floor <= 0 or floor > limit:
        raise ValueError("min_matches must be positive and no greater than max_matches")
    if min_confidence is not None and not np.isfinite(float(min_confidence)):
        raise ValueError("min_confidence must be finite")

    unique = resolve_pose_match_conflicts(matches)
    adaptive_limit = limit
    if min_confidence is not None:
        confident_count = sum(
            np.isfinite(float(match.similarity))
            and float(match.similarity) >= float(min_confidence)
            for match in unique
        )
        adaptive_limit = max(floor, min(limit, int(confident_count)))
    adaptive_limit = min(adaptive_limit, len(unique))
    if len(unique) <= adaptive_limit:
        return unique
    if str(mode) == "score_topk":
        return unique[:adaptive_limit]

    buckets: dict[tuple[int, int], deque[QueryTo3DMatch]] = defaultdict(deque)
    for match in unique:
        x, y = np.asarray(match.xy, dtype=np.float64).reshape(2)
        col = int(
            np.clip(
                np.floor(float(x) / float(image_width) * int(grid_cols)),
                0,
                int(grid_cols) - 1,
            )
        )
        row = int(
            np.clip(
                np.floor(float(y) / float(image_height) * int(grid_rows)),
                0,
                int(grid_rows) - 1,
            )
        )
        buckets[(row, col)].append(match)

    selected: list[QueryTo3DMatch] = []
    while len(selected) < adaptive_limit:
        active = [key for key, values in buckets.items() if values]
        if not active:
            break
        active.sort(
            key=lambda key: float(buckets[key][0].similarity),
            reverse=True,
        )
        for key in active:
            selected.append(buckets[key].popleft())
            if len(selected) >= adaptive_limit:
                break
    selected.sort(key=lambda item: float(item.similarity), reverse=True)
    return selected
