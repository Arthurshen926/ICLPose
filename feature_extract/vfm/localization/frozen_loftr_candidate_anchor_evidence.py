"""Target-free candidate observation evidence read from a global LoFTR cache.

For a fixed query anchor and one fixed candidate support observation, this
module evaluates only whether an already-cached LoFTR correspondence lies near
both image locations.  It does not project a 3-D point, change a candidate,
rank support images, or consume a pose.  The returned values are raw features
for a later train-only calibration, never probabilities by themselves.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.frozen_loftr_pair_cache import FrozenLoFTRPairCache


FROZEN_LOFTR_CANDIDATE_ANCHOR_EVIDENCE_FORMAT = (
    "frozen_loftr_candidate_anchor_evidence_v1"
)
LOFTR_ANCHOR_FEATURE_NAMES = (
    "loftr_joint_peak_sigma8",
    "loftr_joint_peak_sigma16",
    "loftr_joint_peak_sigma32",
    "loftr_joint_kernel_sigma16",
    "loftr_nearest_joint_distance_px",
    "loftr_nearest_query_distance_px",
    "loftr_nearest_support_distance_px",
    "loftr_nearest_match_confidence",
)


def loftr_anchor_pair_features(
    *,
    query_xy: np.ndarray,
    support_xy: np.ndarray,
    matched_query_xy: np.ndarray,
    matched_support_xy: np.ndarray,
    match_confidence: np.ndarray,
    chunk_size: int,
    device: torch.device,
    peak_sigmas_px: Sequence[float] = (8.0, 16.0, 32.0),
    kernel_sigma_px: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw 4-D correspondence features for fixed anchor pairs.

    A candidate is strong only when one LoFTR match is jointly close to the
    held-out query anchor and the candidate's own SfM support-observation
    anchor.  Query-only or support-only proximity cannot receive a high joint
    peak.  Empty pair output is represented as unavailable/NaN rather than a
    geometry-dependent fallback.
    """

    query = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    support = np.asarray(support_xy, dtype=np.float32).reshape(-1, 2)
    matched_query = np.asarray(matched_query_xy, dtype=np.float32).reshape(-1, 2)
    matched_support = np.asarray(matched_support_xy, dtype=np.float32).reshape(-1, 2)
    confidence = np.asarray(match_confidence, dtype=np.float32).reshape(-1)
    sigmas = tuple(float(value) for value in peak_sigmas_px)
    if (
        support.shape != query.shape
        or matched_support.shape != matched_query.shape
        or len(confidence) != len(matched_query)
        or int(chunk_size) <= 0
        or len(sigmas) != 3
        or any(not np.isfinite(value) or value <= 0.0 for value in sigmas)
        or not np.isfinite(float(kernel_sigma_px))
        or float(kernel_sigma_px) <= 0.0
        or np.any(~np.isfinite(query))
        or np.any(~np.isfinite(support))
        or np.any(~np.isfinite(matched_query))
        or np.any(~np.isfinite(matched_support))
        or np.any(~np.isfinite(confidence))
        or np.any(confidence < 0.0)
        or np.any(confidence > 1.0)
    ):
        raise ValueError("LoFTR anchor feature inputs are invalid")
    feature_count = len(LOFTR_ANCHOR_FEATURE_NAMES)
    output = np.full((len(query), feature_count), np.nan, dtype=np.float32)
    usable = np.zeros((len(query),), dtype=bool)
    if len(query) == 0 or len(matched_query) == 0:
        return output, usable
    query_matches = torch.as_tensor(matched_query, dtype=torch.float32, device=device)
    support_matches = torch.as_tensor(matched_support, dtype=torch.float32, device=device)
    scores = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    score_sum = scores.sum().clamp_min(1e-12)
    for start in range(0, len(query), int(chunk_size)):
        stop = min(start + int(chunk_size), len(query))
        query_chunk = torch.as_tensor(query[start:stop], dtype=torch.float32, device=device)
        support_chunk = torch.as_tensor(support[start:stop], dtype=torch.float32, device=device)
        query_delta_sq = torch.sum(
            (query_chunk[:, None, :] - query_matches[None, :, :]) ** 2, dim=2
        )
        support_delta_sq = torch.sum(
            (support_chunk[:, None, :] - support_matches[None, :, :]) ** 2, dim=2
        )
        joint_delta_sq = query_delta_sq + support_delta_sq
        nearest_joint_sq, nearest_index = torch.min(joint_delta_sq, dim=1)
        nearest_query_sq = torch.gather(query_delta_sq, 1, nearest_index[:, None]).squeeze(1)
        nearest_support_sq = torch.gather(support_delta_sq, 1, nearest_index[:, None]).squeeze(1)
        nearest_confidence = scores.index_select(0, nearest_index)
        peak_values = []
        for sigma in sigmas:
            kernel = torch.exp(-0.5 * joint_delta_sq / float(sigma * sigma))
            peak_values.append(torch.max(kernel * scores[None, :], dim=1).values)
        kernel16 = torch.sum(
            torch.exp(
                -0.5 * joint_delta_sq / float(kernel_sigma_px * kernel_sigma_px)
            )
            * scores[None, :],
            dim=1,
        ) / score_sum
        values = torch.stack(
            [
                *peak_values,
                kernel16,
                torch.sqrt(nearest_joint_sq),
                torch.sqrt(nearest_query_sq),
                torch.sqrt(nearest_support_sq),
                nearest_confidence,
            ],
            dim=1,
        )
        output[start:stop] = values.detach().cpu().numpy().astype(np.float32, copy=False)
        usable[start:stop] = True
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, usable


def frozen_loftr_candidate_view_features(
    *,
    cache: FrozenLoFTRPairCache,
    query_xy: np.ndarray,
    candidate_support_xy: np.ndarray,
    candidate_support_image_ids: np.ndarray,
    candidate_view_valid: np.ndarray,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize fixed per-view anchor features without image selection."""

    query = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    support_xy = np.asarray(candidate_support_xy, dtype=np.float32)
    support_ids = np.asarray(candidate_support_image_ids).astype(str)
    valid = np.asarray(candidate_view_valid, dtype=bool)
    if (
        support_xy.ndim != 4
        or support_xy.shape != (*support_ids.shape, 2)
        or support_ids.shape != valid.shape
        or support_ids.shape[0] != len(query)
        or int(chunk_size) <= 0
        or np.any(~np.isfinite(query))
        or np.any(~np.isfinite(support_xy[valid]))
        or np.any(support_ids[valid] == "")
    ):
        raise ValueError("frozen LoFTR candidate view inputs are invalid")
    feature_count = len(LOFTR_ANCHOR_FEATURE_NAMES)
    features = np.full((*support_ids.shape, feature_count), np.nan, dtype=np.float32)
    usable = np.zeros(support_ids.shape, dtype=bool)
    pair_match_counts = np.zeros(support_ids.shape, dtype=np.int32)
    flat_ids = support_ids.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_support_xy = support_xy.reshape(-1, 2)
    point_indices = np.broadcast_to(
        np.arange(len(query), dtype=np.int64)[:, None, None], support_ids.shape
    ).reshape(-1)
    for image_id in np.unique(flat_ids[flat_valid]).tolist():
        selected = np.flatnonzero(flat_valid & (flat_ids == str(image_id)))
        pair_index = cache.image_index(str(image_id))
        matched_query, matched_support, confidence = cache.matches_for_index(pair_index)
        values, available = loftr_anchor_pair_features(
            query_xy=query[point_indices[selected]],
            support_xy=flat_support_xy[selected],
            matched_query_xy=matched_query,
            matched_support_xy=matched_support,
            match_confidence=confidence,
            chunk_size=int(chunk_size),
            device=device,
        )
        features.reshape(-1, feature_count)[selected] = values
        usable.reshape(-1)[selected] = available
        pair_match_counts.reshape(-1)[selected] = int(len(confidence))
    if np.any(usable & ~valid) or np.any(~np.isfinite(features[usable])):
        raise RuntimeError("frozen LoFTR candidate feature materialization is invalid")
    return features, usable, pair_match_counts
