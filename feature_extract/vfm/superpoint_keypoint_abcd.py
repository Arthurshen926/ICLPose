"""Diagnostics for landmark-conditioned SuperPoint keypoint usefulness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


FEATURE_COLUMNS = (
    "selector_score",
    "selector_margin",
    "descriptor_similarity",
    "query_keypoint_score",
    "support_keypoint_score",
    "distance_to_center_px",
    "distance_to_center_norm",
    "support_bias_px",
    "support_bias_norm",
    "match_similarity",
    "match_ratio",
    "similarity_margin",
    "landmark_variance",
    "landmark_reprojection_error",
    "landmark_quality",
    "landmark_ambiguity",
    "observation_count",
    "visibility_count",
    "baseline_reproj_residual_px",
    "support_delta_x",
    "support_delta_y",
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("values must be NxD")
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float64), std=std.astype(np.float64))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return (array - self.mean[None, :]) / self.std[None, :]


@dataclass(frozen=True)
class BinaryLinearModel:
    weights: np.ndarray
    bias: float
    standardizer: Standardizer
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def predict_proba(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float64)
        x = self.standardizer.transform(feature_matrix(rows, self.feature_columns))
        logits = x @ self.weights.reshape(-1) + float(self.bias)
        return sigmoid(logits)


@dataclass(frozen=True)
class SoftmaxNoSnapModel:
    weights: np.ndarray
    no_snap_bias: float
    standardizer: Standardizer
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def candidate_logits(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if not rows:
            return np.zeros((0,), dtype=np.float64)
        x = self.standardizer.transform(feature_matrix(rows, self.feature_columns))
        return x @ self.weights.reshape(-1)


@dataclass(frozen=True)
class RidgeResidualModel:
    weights: np.ndarray
    bias: np.ndarray
    standardizer: Standardizer
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    max_norm_px: float = 16.0

    def predict(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 2), dtype=np.float64)
        x = self.standardizer.transform(feature_matrix(rows, self.feature_columns))
        residuals = x @ self.weights + self.bias[None, :]
        norms = np.linalg.norm(residuals, axis=1)
        limit = float(self.max_norm_px)
        if limit > 0.0:
            scale = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
            residuals = residuals * scale[:, None]
        return residuals.astype(np.float64)


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, data: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def finite_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def feature_value(row: Mapping[str, object], column: str) -> float:
    if column == "support_delta_x":
        delta = row.get("support_delta_xy") or [0.0, 0.0]
        return finite_float(delta[0] if isinstance(delta, Sequence) and len(delta) >= 1 else 0.0)
    if column == "support_delta_y":
        delta = row.get("support_delta_xy") or [0.0, 0.0]
        return finite_float(delta[1] if isinstance(delta, Sequence) and len(delta) >= 2 else 0.0)
    return finite_float(row.get(column))


def feature_matrix(rows: Sequence[Mapping[str, object]], columns: Sequence[str] = FEATURE_COLUMNS) -> np.ndarray:
    return np.asarray([[feature_value(row, column) for column in columns] for row in rows], dtype=np.float64)


def snap_candidate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row.get("action") == "snap"]


def group_key(row: Mapping[str, object]) -> tuple[str, int]:
    return str(row.get("query_id")), int(row.get("match_index", -1))


def group_rows(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(dict(row))
    return grouped


def query_split(query_ids: Sequence[str], train_fraction: float = 0.5, seed: int = 0) -> tuple[set[str], set[str]]:
    unique = np.asarray(sorted({str(query_id) for query_id in query_ids}), dtype=object)
    rng = np.random.default_rng(int(seed))
    order = np.arange(unique.shape[0])
    rng.shuffle(order)
    split = int(round(float(train_fraction) * unique.shape[0]))
    split = min(max(split, 1), max(unique.shape[0] - 1, 1))
    train = {str(unique[idx]) for idx in order[:split]}
    eval_ids = {str(unique[idx]) for idx in order[split:]}
    return train, eval_ids


def is_useful_snap(row: Mapping[str, object], *, positive_px: float = 8.0, margin_px: float = 2.0) -> bool:
    candidate_error = row.get("candidate_error_px")
    center_error = row.get("center_error_px")
    if candidate_error is None or center_error is None:
        return False
    return bool(float(candidate_error) < float(positive_px) and float(candidate_error) + float(margin_px) < float(center_error))


def best_action_label(
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    positive_px: float = 8.0,
    margin_px: float = 2.0,
) -> int:
    best_idx = -1
    best_error = float("inf")
    center_error = None
    for idx, row in enumerate(candidate_rows):
        if center_error is None:
            center_error = row.get("center_error_px")
        err = row.get("candidate_error_px")
        if err is None:
            continue
        value = float(err)
        if value < best_error:
            best_error = value
            best_idx = idx
    if best_idx < 0 or center_error is None:
        return len(candidate_rows)
    if best_error < float(positive_px) and best_error + float(margin_px) < float(center_error):
        return best_idx
    return len(candidate_rows)


def sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def train_snap_improvement_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    positive_px: float = 8.0,
    margin_px: float = 2.0,
    iterations: int = 400,
    learning_rate: float = 0.15,
    l2: float = 1e-4,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> tuple[BinaryLinearModel, dict[str, float]]:
    candidates = snap_candidate_rows(rows)
    if not candidates:
        raise ValueError("no snap candidates available")
    x = feature_matrix(candidates, feature_columns)
    y = np.asarray([1.0 if is_useful_snap(row, positive_px=positive_px, margin_px=margin_px) else 0.0 for row in candidates])
    standardizer = Standardizer.fit(x)
    z = standardizer.transform(x)
    weights = np.zeros((z.shape[1],), dtype=np.float64)
    bias = float(np.log((float(y.mean()) + 1e-4) / max(1.0 - float(y.mean()) + 1e-4, 1e-4)))
    for _ in range(int(iterations)):
        pred = sigmoid(z @ weights + bias)
        error = pred - y
        weights -= float(learning_rate) * ((z.T @ error) / max(z.shape[0], 1) + float(l2) * weights)
        bias -= float(learning_rate) * float(np.mean(error))
    model = BinaryLinearModel(weights=weights, bias=float(bias), standardizer=standardizer, feature_columns=tuple(feature_columns))
    probs = model.predict_proba(candidates)
    return model, {
        "positive_prior": float(y.mean()),
        "auc": roc_auc(probs, y),
        "precision_top20": precision_at_fraction(probs, y, 0.20),
        "precision_top40": precision_at_fraction(probs, y, 0.40),
    }


def train_no_snap_softmax_selector(
    rows: Sequence[Mapping[str, object]],
    *,
    positive_px: float = 8.0,
    margin_px: float = 2.0,
    iterations: int = 120,
    learning_rate: float = 0.08,
    l2: float = 1e-4,
    max_groups: int | None = None,
    seed: int = 0,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> SoftmaxNoSnapModel:
    grouped = [snap_candidate_rows(items) for items in group_rows(rows).values()]
    grouped = [items for items in grouped if items]
    if not grouped:
        raise ValueError("no candidate groups available")
    rng = np.random.default_rng(int(seed))
    if max_groups is not None and len(grouped) > int(max_groups):
        indices = rng.choice(len(grouped), size=int(max_groups), replace=False)
        grouped = [grouped[int(idx)] for idx in indices]
    all_candidates = [row for group in grouped for row in group]
    standardizer = Standardizer.fit(feature_matrix(all_candidates, feature_columns))
    weights = np.zeros((len(tuple(feature_columns)),), dtype=np.float64)
    no_snap_bias = 0.0
    for _ in range(int(iterations)):
        rng.shuffle(grouped)
        grad_w = np.zeros_like(weights)
        grad_b = 0.0
        count = 0
        for group in grouped:
            x = standardizer.transform(feature_matrix(group, feature_columns))
            logits = np.concatenate([x @ weights, np.asarray([no_snap_bias], dtype=np.float64)])
            logits = logits - float(np.max(logits))
            probs = np.exp(logits)
            probs /= max(float(np.sum(probs)), 1e-12)
            label = best_action_label(group, positive_px=positive_px, margin_px=margin_px)
            target = np.zeros_like(probs)
            target[int(label)] = 1.0
            error = probs - target
            grad_w += x.T @ error[: x.shape[0]]
            grad_b += float(error[-1])
            count += 1
        weights -= float(learning_rate) * (grad_w / max(count, 1) + float(l2) * weights)
        no_snap_bias -= float(learning_rate) * (grad_b / max(count, 1))
    return SoftmaxNoSnapModel(
        weights=weights,
        no_snap_bias=float(no_snap_bias),
        standardizer=standardizer,
        feature_columns=tuple(feature_columns),
    )


def train_ridge_residual_model(
    rows: Sequence[Mapping[str, object]],
    *,
    positive_px: float = 12.0,
    l2: float = 1e-2,
    max_norm_px: float = 16.0,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> RidgeResidualModel:
    candidates = []
    targets = []
    for row in snap_candidate_rows(rows):
        gt_xy = row.get("gt_xy")
        candidate_xy = row.get("candidate_xy")
        if gt_xy is None or candidate_xy is None:
            continue
        err = finite_float(row.get("candidate_error_px"), default=1e9)
        if err > float(positive_px):
            continue
        candidates.append(row)
        targets.append(np.asarray(gt_xy, dtype=np.float64).reshape(2) - np.asarray(candidate_xy, dtype=np.float64).reshape(2))
    if not candidates:
        raise ValueError("no residual training candidates available")
    x = feature_matrix(candidates, feature_columns)
    y = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
    standardizer = Standardizer.fit(x)
    z = standardizer.transform(x)
    design = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    reg[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return RidgeResidualModel(
        weights=params[:-1],
        bias=params[-1],
        standardizer=standardizer,
        feature_columns=tuple(feature_columns),
        max_norm_px=float(max_norm_px),
    )


def select_heuristic_candidate(group: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    candidates = [row for row in group if row.get("action") == "snap" and bool(row.get("heuristic_applied", row.get("selected_by_heuristic", False)))]
    if candidates:
        return min(candidates, key=lambda row: int(row.get("candidate_rank", 1_000_000)))
    candidates = [row for row in group if row.get("action") == "snap" and bool(row.get("selected_by_heuristic", False))]
    if candidates:
        return min(candidates, key=lambda row: int(row.get("candidate_rank", 1_000_000)))
    return None


def select_softmax_candidate(
    group: Sequence[Mapping[str, object]],
    model: SoftmaxNoSnapModel,
) -> dict[str, object] | None:
    candidates = [row for row in group if row.get("action") == "snap"]
    if not candidates:
        return None
    logits = np.concatenate([model.candidate_logits(candidates), np.asarray([model.no_snap_bias], dtype=np.float64)])
    selected = int(np.argmax(logits))
    if selected >= len(candidates):
        return None
    return candidates[selected]


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1) > 0.5
    pos = int(y.sum())
    neg = int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_sum = float(ranks[y].sum())
    return float((rank_sum - pos * (pos + 1) / 2.0) / max(pos * neg, 1))


def precision_at_fraction(scores: Sequence[float], labels: Sequence[float], fraction: float) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1) > 0.5
    if s.size == 0:
        return float("nan")
    keep = max(1, int(round(float(fraction) * s.size)))
    order = np.argsort(-s)[:keep]
    return float(np.mean(y[order]))


def snap_improvement_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    selected = [row for group in group_rows(rows).values() if (row := select_heuristic_candidate(group)) is not None]
    if not selected:
        return {"selected_count": 0.0}
    improvements = np.asarray([finite_float(row.get("snap_improvement_px")) for row in selected], dtype=np.float64)
    improve = improvements > 0.0
    worsen = improvements < 0.0
    return {
        "selected_count": float(len(selected)),
        "snap_improve_ratio": float(np.mean(improve)),
        "snap_worsen_ratio": float(np.mean(worsen)),
        "mean_improvement_if_improve_px": 0.0 if not np.any(improve) else float(np.mean(improvements[improve])),
        "mean_damage_if_worsen_px": 0.0 if not np.any(worsen) else float(-np.mean(improvements[worsen])),
        "mean_delta_px": float(np.mean(improvements)),
    }


def support_bias_bucket_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float | str]]:
    buckets = [
        ("<2", 0.0, 2.0),
        ("2-4", 2.0, 4.0),
        ("4-8", 4.0, 8.0),
        ("8-16", 8.0, 16.0),
        (">16", 16.0, float("inf")),
    ]
    selected = [row for group in group_rows(rows).values() if (row := select_heuristic_candidate(group)) is not None]
    output = []
    for name, lo, hi in buckets:
        items = [
            row
            for row in selected
            if finite_float(row.get("support_bias_px"), default=float("inf")) >= lo
            and finite_float(row.get("support_bias_px"), default=float("inf")) < hi
        ]
        improvements = np.asarray([finite_float(row.get("snap_improvement_px")) for row in items], dtype=np.float64)
        output.append(
            {
                "bucket": name,
                "count": float(len(items)),
                "improve_ratio": float(np.mean(improvements > 0.0)) if improvements.size else float("nan"),
                "snap8": float(np.mean([bool(row.get("snap_correct_at_8px", False)) for row in items])) if items else float("nan"),
                "mean_delta_px": float(np.mean(improvements)) if improvements.size else float("nan"),
            }
        )
    return output
