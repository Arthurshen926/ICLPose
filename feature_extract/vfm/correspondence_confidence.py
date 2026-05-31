"""Calibrated confidence utilities for patch-to-3D correspondences."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatchConfidenceLabel:
    target: int
    ignore: bool
    strong_positive: bool
    weak_positive: bool
    hard_negative: bool


def _float_value(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool_value(row: Mapping[str, Any], key: str) -> bool:
    return bool(row.get(key))


def label_match_row(
    row: Mapping[str, Any],
    stride_positive: float = 1.0,
    weak_positive_stride: float = 2.0,
) -> MatchConfidenceLabel:
    """Label a match by GT patch geometry while preserving an ambiguous band."""

    patch_correct = _bool_value(row, "patch_correct") or _bool_value(row, "patch_positive_label")
    gt_stride = _float_value(row, "gt_reproj_error_stride", float("inf"))
    strong = bool(patch_correct or gt_stride <= float(stride_positive))
    weak = bool(not strong and gt_stride <= float(weak_positive_stride))
    hard_negative = bool((gt_stride > float(weak_positive_stride)) and _bool_value(row, "pnp_inlier"))
    if strong:
        return MatchConfidenceLabel(target=1, ignore=False, strong_positive=True, weak_positive=False, hard_negative=False)
    if weak:
        return MatchConfidenceLabel(target=0, ignore=True, strong_positive=False, weak_positive=True, hard_negative=False)
    return MatchConfidenceLabel(
        target=0,
        ignore=False,
        strong_positive=False,
        weak_positive=False,
        hard_negative=hard_negative,
    )


def _rank_score(row: Mapping[str, Any]) -> float:
    rank = row.get("token_match_rank", row.get("match_rank", row.get("match_index", 0)))
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        rank_int = 0
    return float(1.0 / (1.0 + max(rank_int, 0)))


def _xy_norm(row: Mapping[str, Any], axis: int) -> float:
    xy = row.get("xy")
    if not isinstance(xy, Sequence) or len(xy) <= axis:
        return 0.0
    return _float_value({str(axis): xy[axis]}, str(axis), 0.0)


def _feature_specs(feature_set: str) -> list[tuple[str, Any]]:
    descriptor = [
        ("similarity", lambda row: _float_value(row, "similarity")),
        ("similarity_margin", lambda row: _float_value(row, "similarity_margin")),
        ("rank_score", _rank_score),
        ("mnn_flag", lambda row: 1.0 if str(row.get("source", "")).startswith("sparse") else 0.0),
    ]
    map_stats = [
        ("log_observation_count", lambda row: math.log1p(_float_value(row, "observation_count"))),
        ("log_visibility_count", lambda row: math.log1p(_float_value(row, "visibility_count"))),
        ("landmark_variance", lambda row: _float_value(row, "landmark_variance")),
        ("landmark_reprojection_error", lambda row: _float_value(row, "landmark_reprojection_error")),
        ("landmark_ambiguity", lambda row: _float_value(row, "landmark_ambiguity")),
        ("landmark_quality", lambda row: _float_value(row, "landmark_quality", 1.0)),
        ("map_reliability", lambda row: _float_value(row, "map_reliability", 1.0)),
    ]
    local = [
        ("x_px", lambda row: _xy_norm(row, 0)),
        ("y_px", lambda row: _xy_norm(row, 1)),
        ("distance_to_boundary_px", lambda row: _float_value(row, "distance_to_boundary_px")),
        ("local_consistency_support", lambda row: _float_value(row, "local_consistency_support")),
        ("local_consistency_score", lambda row: _float_value(row, "local_consistency_score")),
        ("baseline_ransac_inlier", lambda row: 1.0 if _bool_value(row, "pnp_inlier") else 0.0),
        ("baseline_reproj_residual_px", lambda row: _float_value(row, "baseline_reproj_residual_px", 1e3)),
    ]
    if feature_set == "descriptor":
        return descriptor
    if feature_set == "map_stats":
        return map_stats
    if feature_set == "descriptor_map":
        return descriptor + map_stats
    if feature_set == "full":
        return descriptor + map_stats + local
    raise ValueError("feature_set must be one of: descriptor, map_stats, descriptor_map, full")


def vectorize_match_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_set: str = "full",
    stride_positive: float = 1.0,
    weak_positive_stride: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    specs = _feature_specs(feature_set)
    matrix = np.zeros((len(rows), len(specs)), dtype=np.float32)
    labels = np.zeros((len(rows),), dtype=np.int64)
    keep = np.zeros((len(rows),), dtype=bool)
    for row_idx, row in enumerate(rows):
        for col_idx, (_name, getter) in enumerate(specs):
            matrix[row_idx, col_idx] = float(getter(row))
        label = label_match_row(row, stride_positive=stride_positive, weak_positive_stride=weak_positive_stride)
        labels[row_idx] = int(label.target)
        keep[row_idx] = not bool(label.ignore)
    names = [name for name, _getter in specs]
    return matrix, labels, keep, names


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    return np.where(logits >= 0.0, 1.0 / (1.0 + np.exp(-logits)), np.exp(logits) / (1.0 + np.exp(logits)))


@dataclass
class CalibratedLogisticConfidence:
    learning_rate: float = 0.05
    max_iter: int = 800
    l2: float = 1e-3
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    weights_: np.ndarray | None = None
    bias_: float = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "CalibratedLogisticConfidence":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64).reshape(-1)
        if x.ndim != 2:
            raise ValueError("features must have shape (N, C)")
        if y.shape[0] != x.shape[0]:
            raise ValueError("labels must have one value per feature row")
        if x.shape[0] == 0:
            raise ValueError("at least one training sample is required")
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        z = (x - self.mean_) / self.scale_
        weights = np.zeros((z.shape[1],), dtype=np.float64)
        bias = 0.0
        pos = max(float(np.sum(y > 0.5)), 1.0)
        neg = max(float(np.sum(y <= 0.5)), 1.0)
        sample_weight = np.where(y > 0.5, 0.5 * (pos + neg) / pos, 0.5 * (pos + neg) / neg)
        denominator = max(float(np.sum(sample_weight)), 1e-12)
        for _ in range(int(self.max_iter)):
            pred = _sigmoid(z @ weights + bias)
            residual = (pred - y) * sample_weight
            grad_w = (z.T @ residual) / denominator + float(self.l2) * weights
            grad_b = float(np.sum(residual) / denominator)
            weights -= float(self.learning_rate) * grad_w
            bias -= float(self.learning_rate) * grad_b
        self.weights_ = weights.astype(np.float64, copy=False)
        self.bias_ = float(bias)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise ValueError("model is not fitted")
        x = np.asarray(features, dtype=np.float64)
        z = (x - self.mean_) / self.scale_
        return _sigmoid(z @ self.weights_ + float(self.bias_)).astype(np.float64, copy=False)

    def to_json_dict(self, feature_names: Sequence[str] | None = None) -> dict[str, Any]:
        if self.mean_ is None or self.scale_ is None or self.weights_ is None:
            raise ValueError("model is not fitted")
        return {
            "model_type": "calibrated_logistic_confidence",
            "feature_names": list(feature_names or []),
            "mean": [float(value) for value in self.mean_],
            "scale": [float(value) for value in self.scale_],
            "weights": [float(value) for value in self.weights_],
            "bias": float(self.bias_),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "CalibratedLogisticConfidence":
        model = cls()
        model.mean_ = np.asarray(data["mean"], dtype=np.float64)
        model.scale_ = np.asarray(data["scale"], dtype=np.float64)
        model.weights_ = np.asarray(data["weights"], dtype=np.float64)
        model.bias_ = float(data["bias"])
        return model

    def save_json(self, path: str | Path, feature_names: Sequence[str] | None = None) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(feature_names), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load_json(cls, path: str | Path) -> "CalibratedLogisticConfidence":
        return cls.from_json_dict(json.loads(Path(path).read_text()))


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[pos]))
    n_pos = float(np.sum(pos))
    n_neg = float(np.sum(neg))
    return float((pos_rank_sum - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1):
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = y[order]
    tp = np.cumsum(ranked == 1)
    positions = np.arange(1, ranked.shape[0] + 1, dtype=np.float64)
    precision = tp / positions
    return float(np.sum(precision[ranked == 1]) / max(float(np.sum(y == 1)), 1.0))


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int) -> float:
    y = np.asarray(labels, dtype=np.float64)
    s = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(float(y.shape[0]), 1.0)
    error = 0.0
    for idx in range(int(bins)):
        if idx == int(bins) - 1:
            mask = (s >= edges[idx]) & (s <= edges[idx + 1])
        else:
            mask = (s >= edges[idx]) & (s < edges[idx + 1])
        if not np.any(mask):
            continue
        error += float(np.sum(mask)) / total * abs(float(np.mean(s[mask])) - float(np.mean(y[mask])))
    return float(error)


def confidence_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    top_fraction: float = 0.10,
    num_ece_bins: int = 10,
) -> dict[str, float | int | None]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if y.shape[0] != p.shape[0]:
        raise ValueError("labels and scores must have the same length")
    count = int(y.shape[0])
    if count == 0:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "positive_prior": None,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "ece": None,
            "precision_at_top_fraction": None,
            "recall_at_top_fraction": None,
        }
    keep_count = max(1, int(math.ceil(count * float(top_fraction))))
    order = np.argsort(-p, kind="mergesort")
    top = order[:keep_count]
    positive_count = int(np.sum(y == 1))
    top_positive = int(np.sum(y[top] == 1))
    return {
        "sample_count": count,
        "positive_count": positive_count,
        "positive_prior": float(positive_count / max(count, 1)),
        "auroc": _binary_auroc(y, p),
        "auprc": _average_precision(y, p),
        "brier": float(np.mean((p - y.astype(np.float64)) ** 2)),
        "ece": _ece(y, p, int(num_ece_bins)),
        "precision_at_top_fraction": float(top_positive / max(keep_count, 1)),
        "recall_at_top_fraction": None if positive_count == 0 else float(top_positive / positive_count),
    }


def select_confident_matches_coverage_preserving(
    rows: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    max_matches: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    per_cell: int = 8,
    confidence_key: str = "confidence",
) -> list[Mapping[str, Any]]:
    """Keep high-confidence matches while preserving 2D image coverage."""

    if int(max_matches) <= 0:
        raise ValueError("max_matches must be positive")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0 or int(per_cell) <= 0:
        raise ValueError("grid_rows, grid_cols, and per_cell must be positive")
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    cells: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        xy = row.get("xy", (0.0, 0.0))
        x = float(xy[0]) if isinstance(xy, Sequence) and len(xy) > 0 else 0.0
        y = float(xy[1]) if isinstance(xy, Sequence) and len(xy) > 1 else 0.0
        x_cell = int(np.clip(np.floor(x / width * int(grid_cols)), 0, int(grid_cols) - 1))
        y_cell = int(np.clip(np.floor(y / height * int(grid_rows)), 0, int(grid_rows) - 1))
        cells.setdefault((y_cell, x_cell), []).append(row)
    selected_ids: set[int] = set()
    selected: list[Mapping[str, Any]] = []
    for cell_rows in cells.values():
        ordered = sorted(cell_rows, key=lambda item: _float_value(item, confidence_key), reverse=True)
        for row in ordered[: int(per_cell)]:
            selected.append(row)
            selected_ids.add(id(row))
    if len(selected) < int(max_matches):
        ordered_all = sorted(rows, key=lambda item: _float_value(item, confidence_key), reverse=True)
        for row in ordered_all:
            if id(row) in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(id(row))
            if len(selected) >= int(max_matches):
                break
    selected = sorted(selected[: int(max_matches)], key=lambda item: _float_value(item, confidence_key), reverse=True)
    return selected
