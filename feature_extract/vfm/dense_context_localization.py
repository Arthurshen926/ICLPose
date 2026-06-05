"""Helpers for dense-context reranked patch-to-3D localization."""

from __future__ import annotations

import numpy as np

from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch


def select_top1_candidate_rows(
    top_indices: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the best valid candidate per query token from NxK score matrices."""

    indices = np.asarray(top_indices, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if indices.shape != values.shape or indices.shape != valid.shape or indices.ndim != 2:
        raise ValueError("top_indices, scores and valid_mask must all have shape (N, K)")
    active_tokens = np.flatnonzero(np.any(valid, axis=1))
    if active_tokens.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    masked = np.where(valid, values, -np.inf)
    ranks = np.argmax(masked[active_tokens], axis=1)
    landmark_rows = indices[active_tokens, ranks].astype(np.int64, copy=False)
    selected_scores = masked[active_tokens, ranks].astype(np.float32, copy=False)
    return active_tokens.astype(np.int64, copy=False), landmark_rows, selected_scores


def build_selected_matches(
    token_rows: np.ndarray,
    token_indices: np.ndarray,
    landmark_rows: np.ndarray,
    scores: np.ndarray,
    landmark_index: LandmarkMapIndex,
    token_centers: np.ndarray,
    source: str,
    max_matches: int,
) -> list[QueryTo3DMatch]:
    """Convert selected token-landmark rows to sorted QueryTo3DMatch objects."""

    rows = np.asarray(token_rows, dtype=np.int64).reshape(-1)
    tokens = np.asarray(token_indices, dtype=np.int64).reshape(-1)
    landmarks = np.asarray(landmark_rows, dtype=np.int64).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (rows.shape[0] == landmarks.shape[0] == score_values.shape[0]):
        raise ValueError("token_rows, landmark_rows and scores must have the same length")
    centers = np.asarray(token_centers, dtype=np.float64)
    order = np.argsort(-score_values, kind="mergesort")
    if int(max_matches) > 0:
        order = order[: int(max_matches)]
    matches: list[QueryTo3DMatch] = []
    for item in order.tolist():
        token_row = int(rows[item])
        landmark_row = int(landmarks[item])
        score = float(score_values[item])
        if landmark_row < 0 or landmark_row >= len(landmark_index):
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=int(tokens[token_row]),
                xy=centers[token_row].astype(np.float64, copy=True),
                track_id=int(landmark_index.track_ids[landmark_row]),
                xyz=landmark_index.xyz[landmark_row].astype(np.float64, copy=True),
                similarity=score,
                ratio=0.0,
                landmark_variance=float(landmark_index.mean_variances[landmark_row]),
                source=str(source),
                observation_count=int(landmark_index.observation_counts[landmark_row]),
                visibility_count=len(landmark_index.observation_image_ids[landmark_row]),
                landmark_reprojection_error=float(landmark_index.reprojection_errors[landmark_row]),
                quality_weighted_similarity=score,
                pnp_soft_score=score,
            )
        )
    return matches
