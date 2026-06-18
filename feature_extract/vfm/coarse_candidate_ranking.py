"""Calibrated ranking utilities for coarse render-cell candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


@dataclass(frozen=True)
class CoarseCandidateLabel:
    target: int
    ignore: bool
    strong_positive: bool
    weak_positive: bool


def _float_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return float(default)
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float(default)
    return output if math.isfinite(output) else float(default)


def _int_value(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def label_coarse_candidate_row(
    row: Mapping[str, Any],
    *,
    stride_positive: float = 1.0,
    weak_positive_stride: float = 2.0,
) -> CoarseCandidateLabel:
    """Label a coarse candidate by GT reprojection while ignoring the ambiguous band."""

    gt_stride = _float_value(row, "gt_reproj_error_stride", float("inf"))
    strong = bool(_bool_value(row, "patch_correct") or _bool_value(row, "patch_positive_label"))
    strong = bool(strong or gt_stride <= float(stride_positive))
    weak = bool((not strong) and gt_stride <= float(weak_positive_stride))
    if strong:
        return CoarseCandidateLabel(target=1, ignore=False, strong_positive=True, weak_positive=False)
    if weak:
        return CoarseCandidateLabel(target=0, ignore=True, strong_positive=False, weak_positive=True)
    return CoarseCandidateLabel(target=0, ignore=False, strong_positive=False, weak_positive=False)


def _rank_score(value: int) -> float:
    return float(1.0 / (1.0 + max(int(value), 0)))


def _feature_specs(feature_set: str = "coarse") -> list[tuple[str, Any]]:
    base = [
        ("similarity", lambda row: _float_value(row, "similarity")),
        ("similarity_margin", lambda row: _float_value(row, "similarity_margin")),
        ("candidate_confidence", lambda row: _float_value(row, "confidence", _float_value(row, "dual_softmax_confidence"))),
    ]
    coarse = [
        ("coarse_score", lambda row: _float_value(row, "coarse_score", _float_value(row, "similarity"))),
        ("coarse_score_gap", lambda row: _float_value(row, "coarse_score_gap")),
        ("coarse_rank", lambda row: float(_int_value(row, "coarse_rank", 0))),
        ("coarse_rank_score", lambda row: _rank_score(_int_value(row, "coarse_rank", 0))),
        ("mutual_rank", lambda row: float(_int_value(row, "mutual_rank", 999))),
        ("mutual_rank_score", lambda row: _rank_score(_int_value(row, "mutual_rank", 999))),
        ("mutual_is_top1", lambda row: 1.0 if _int_value(row, "mutual_rank", 999) == 0 else 0.0),
    ]
    local = [
        ("cell_delta_x", lambda row: float(_int_value(row, "cell_delta_x", 0))),
        ("cell_delta_y", lambda row: float(_int_value(row, "cell_delta_y", 0))),
        (
            "cell_delta_chebyshev",
            lambda row: float(max(abs(_int_value(row, "cell_delta_x", 0)), abs(_int_value(row, "cell_delta_y", 0)))),
        ),
        ("query_x", lambda row: _float_value(row, "query_x")),
        ("query_y", lambda row: _float_value(row, "query_y")),
        ("render_x", lambda row: _float_value(row, "render_x")),
        ("render_y", lambda row: _float_value(row, "render_y")),
    ]
    if feature_set == "descriptor":
        return base
    if feature_set == "coarse":
        return base + coarse
    if feature_set == "coarse_local":
        return base + coarse + local
    raise ValueError("feature_set must be one of: descriptor, coarse, coarse_local")


def vectorize_coarse_candidate_row_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_set: str = "coarse",
) -> tuple[np.ndarray, list[str]]:
    specs = _feature_specs(feature_set)
    features = np.zeros((len(rows), len(specs)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, (_name, getter) in enumerate(specs):
            features[row_idx, col_idx] = float(getter(row))
    return features, [name for name, _getter in specs]


def vectorize_coarse_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_set: str = "coarse",
    stride_positive: float = 1.0,
    weak_positive_stride: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features, names = vectorize_coarse_candidate_row_features(rows, feature_set=feature_set)
    labels = np.zeros((len(rows),), dtype=np.int64)
    keep = np.zeros((len(rows),), dtype=bool)
    for idx, row in enumerate(rows):
        label = label_coarse_candidate_row(
            row,
            stride_positive=float(stride_positive),
            weak_positive_stride=float(weak_positive_stride),
        )
        labels[idx] = int(label.target)
        keep[idx] = not bool(label.ignore)
    return features, labels, keep, names


def _match_to_row(match: KeypointFeatureMatch) -> dict[str, Any]:
    return {
        "similarity": float(match.similarity),
        "similarity_margin": match.similarity_margin,
        "confidence": match.dual_softmax_confidence,
        "coarse_rank": match.coarse_rank,
        "coarse_score": match.coarse_score,
        "coarse_score_gap": match.coarse_score_gap,
        "mutual_rank": match.mutual_rank,
        "cell_delta_x": match.cell_delta_x,
        "cell_delta_y": match.cell_delta_y,
        "query_x": float(match.query_xy[0]),
        "query_y": float(match.query_xy[1]),
        "render_x": float(match.render_xy[0]),
        "render_y": float(match.render_xy[1]),
    }


def vectorize_keypoint_matches(
    matches: Sequence[KeypointFeatureMatch],
    *,
    feature_set: str = "coarse",
) -> tuple[np.ndarray, list[str]]:
    return vectorize_coarse_candidate_row_features([_match_to_row(match) for match in matches], feature_set=feature_set)


def annotate_keypoint_matches_with_coarse_candidate_ranker(
    matches: Sequence[KeypointFeatureMatch],
    model: CalibratedLogisticConfidence,
    *,
    feature_set: str = "coarse",
    blend: float = 1.0,
) -> list[KeypointFeatureMatch]:
    """Replace or blend coarse candidate confidence with a calibrated ranker score."""

    values = list(matches)
    if not values:
        return []
    features, _names = vectorize_keypoint_matches(values, feature_set=feature_set)
    probabilities = model.predict_proba(features)
    alpha = float(np.clip(float(blend), 0.0, 1.0))
    updated: list[KeypointFeatureMatch] = []
    for match, probability in zip(values, probabilities):
        learned = float(np.clip(float(probability), 0.0, 1.0))
        base = (
            float(match.dual_softmax_confidence)
            if match.dual_softmax_confidence is not None
            else float(np.clip((float(match.similarity) + 1.0) * 0.5, 0.0, 1.0))
        )
        confidence = (1.0 - alpha) * base + alpha * learned
        updated.append(replace(match, dual_softmax_confidence=float(np.clip(confidence, 0.0, 1.0))))
    updated.sort(key=lambda item: (float(item.dual_softmax_confidence or 0.0), float(item.similarity)), reverse=True)
    return updated
