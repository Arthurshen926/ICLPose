#!/usr/bin/env python3
"""Score labeled LoFTR correspondence reliability stores."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pose_refine.tools.score_loftr_correspondence_feature_consistency import (  # noqa: E402
    binary_auc_from_scores,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def average_precision_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(labels.size, scores.size)
    labels = labels[:count]
    scores = scores[:count]
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    num_pos = int(labels.sum())
    if labels.size == 0 or num_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels.astype(np.float64))
    ranks = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    precision = tp / ranks
    return float(precision[sorted_labels].sum() / float(num_pos))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    count = min(labels.size, probabilities.size)
    labels = labels[:count]
    probabilities = probabilities[:count]
    finite = np.isfinite(probabilities)
    if not finite.any():
        return float("nan")
    return float(np.mean(np.square(probabilities[finite] - labels[finite])))


def _normalized_xy(xy: np.ndarray, hw: Sequence[int]) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64)
    h = int(hw[0]) if len(hw) > 0 else 0
    w = int(hw[1]) if len(hw) > 1 else 0
    denom_x = float(max(w - 1, 1))
    denom_y = float(max(h - 1, 1))
    out = np.zeros_like(coords, dtype=np.float64)
    out[:, 0] = coords[:, 0] / denom_x
    out[:, 1] = coords[:, 1] / denom_y
    return out


def build_reliability_features(
    payload: Mapping[str, np.ndarray],
    *,
    feature_set: str = "confidence_query_xy",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    query_xy = np.asarray(payload["query_xy"], dtype=np.float64)
    confidence = np.asarray(payload["confidence"], dtype=np.float64).reshape(-1)
    labels = np.asarray(payload["pnp_inlier_mask"], dtype=bool).reshape(-1)
    count = min(len(query_xy), len(confidence), len(labels))
    query_xy = query_xy[:count]
    confidence = confidence[:count]
    labels = labels[:count]

    query_hw = np.asarray(payload.get("query_hw", [1, 1]), dtype=np.int64).reshape(-1)
    q_norm = _normalized_xy(query_xy, query_hw)
    columns = [confidence, q_norm[:, 0], q_norm[:, 1]]
    names = ["confidence", "query_x_norm", "query_y_norm"]

    if feature_set == "confidence":
        columns = [confidence]
        names = ["confidence"]
    elif feature_set == "confidence_query_xy":
        pass
    elif feature_set == "confidence_query_xy_map_xy":
        map_xy = np.asarray(payload["map_xy"], dtype=np.float64)[:count]
        map_hw = np.asarray(payload.get("map_hw", query_hw), dtype=np.int64).reshape(-1)
        m_norm = _normalized_xy(map_xy, map_hw)
        columns.extend([m_norm[:, 0], m_norm[:, 1], q_norm[:, 0] - m_norm[:, 0], q_norm[:, 1] - m_norm[:, 1]])
        names.extend(["map_x_norm", "map_y_norm", "delta_x_norm", "delta_y_norm"])
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    features = np.stack(columns, axis=1).astype(np.float32)
    finite = np.isfinite(features).all(axis=1) & np.isfinite(confidence)
    return features[finite], labels[finite], names


@dataclass
class LogisticReliabilityModel:
    weights: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        x = (x - self.feature_mean) / self.feature_std
        return x @ self.weights + float(self.bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logits(features))

    def to_json(self, feature_names: Sequence[str]) -> Dict[str, object]:
        return {
            "feature_names": list(feature_names),
            "weights": [float(v) for v in self.weights],
            "bias": float(self.bias),
            "feature_mean": [float(v) for v in self.feature_mean],
            "feature_std": [float(v) for v in self.feature_std],
        }


def train_logistic_reliability(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 300,
    lr: float = 0.1,
    l2: float = 1.0e-4,
    seed: int = 0,
) -> LogisticReliabilityModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=bool).reshape(-1).astype(np.float64)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    count = min(len(x), len(y))
    x = x[:count]
    y = y[:count]
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    y = y[finite]
    if len(x) == 0 or y.min() == y.max():
        raise ValueError("training data must contain finite positive and negative samples")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    xs = (x - mean) / std

    rng = np.random.default_rng(int(seed))
    weights = rng.normal(0.0, 0.01, size=(xs.shape[1],))
    bias = 0.0
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float((1.0 - y).sum()), 1.0)
    sample_weight = np.where(y > 0.5, 0.5 / n_pos, 0.5 / n_neg)
    sample_weight = sample_weight / sample_weight.mean()

    for _ in range(int(epochs)):
        logits = xs @ weights + bias
        pred = _sigmoid(logits)
        err = (pred - y) * sample_weight
        grad_w = (xs.T @ err) / float(len(xs)) + float(l2) * weights
        grad_b = float(err.mean())
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b

    return LogisticReliabilityModel(weights=weights, bias=float(bias), feature_mean=mean, feature_std=std)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_labeled_correspondence_dataset(
    corr_dirs: Iterable[str],
    *,
    feature_set: str,
    max_points_per_file: int = 0,
    max_total_points: int = 0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object]]:
    rng = np.random.default_rng(int(seed))
    features_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    feature_names: List[str] = []
    num_files = 0
    missing_or_empty = 0
    for corr_dir in corr_dirs:
        for path in sorted(Path(corr_dir).glob("*.npz")):
            num_files += 1
            try:
                features, labels, names = build_reliability_features(_load_npz(path), feature_set=feature_set)
            except Exception:
                missing_or_empty += 1
                continue
            if len(features) == 0:
                missing_or_empty += 1
                continue
            if not feature_names:
                feature_names = names
            if int(max_points_per_file) > 0 and len(features) > int(max_points_per_file):
                idx = rng.choice(len(features), size=int(max_points_per_file), replace=False)
                features = features[idx]
                labels = labels[idx]
            features_all.append(features)
            labels_all.append(labels)
    if not features_all:
        raise ValueError("No labeled correspondence features were loaded")
    features_cat = np.concatenate(features_all, axis=0)
    labels_cat = np.concatenate(labels_all, axis=0)
    if int(max_total_points) > 0 and len(features_cat) > int(max_total_points):
        idx = rng.choice(len(features_cat), size=int(max_total_points), replace=False)
        features_cat = features_cat[idx]
        labels_cat = labels_cat[idx]
    stats = {
        "num_files": int(num_files),
        "num_missing_or_empty_files": int(missing_or_empty),
        "num_points": int(len(labels_cat)),
        "num_inliers": int(labels_cat.sum()),
        "num_outliers": int((~labels_cat).sum()),
        "inlier_ratio": float(labels_cat.mean()) if len(labels_cat) else None,
    }
    return features_cat.astype(np.float32), labels_cat.astype(bool), feature_names, stats


def evaluate_reliability_scores(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "auc": float(binary_auc_from_scores(labels, scores, lower_score_is_positive=False)),
        "average_precision": float(average_precision_from_scores(labels, scores)),
        "brier": float(brier_score(labels, scores)),
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_inlier_mean": float(np.mean(scores[labels])) if labels.any() else float("nan"),
        "score_outlier_mean": float(np.mean(scores[~labels])) if (~labels).any() else float("nan"),
    }


def run_reliability_experiment(
    *,
    train_corr_dirs: Sequence[str],
    eval_corr_dirs: Sequence[str],
    output_json: str,
    feature_set: str = "confidence_query_xy",
    max_points_per_file: int = 0,
    max_total_train_points: int = 0,
    max_total_eval_points: int = 0,
    epochs: int = 300,
    lr: float = 0.1,
    seed: int = 0,
) -> Dict[str, object]:
    train_x, train_y, feature_names, train_stats = load_labeled_correspondence_dataset(
        train_corr_dirs,
        feature_set=feature_set,
        max_points_per_file=max_points_per_file,
        max_total_points=max_total_train_points,
        seed=seed,
    )
    model = train_logistic_reliability(train_x, train_y, epochs=epochs, lr=lr, seed=seed)
    eval_results = []
    for corr_dir in eval_corr_dirs:
        eval_x, eval_y, _, eval_stats = load_labeled_correspondence_dataset(
            [corr_dir],
            feature_set=feature_set,
            max_points_per_file=max_points_per_file,
            max_total_points=max_total_eval_points,
            seed=seed + 17,
        )
        confidence_scores = np.clip(eval_x[:, 0].astype(np.float64), 0.0, 1.0)
        model_scores = model.predict_proba(eval_x)
        eval_results.append(
            {
                "corr_dir": str(corr_dir),
                "stats": eval_stats,
                "confidence": evaluate_reliability_scores(eval_y, confidence_scores),
                "model": evaluate_reliability_scores(eval_y, model_scores),
            }
        )
    summary = {
        "feature_set": feature_set,
        "train_corr_dirs": [str(p) for p in train_corr_dirs],
        "eval_corr_dirs": [str(p) for p in eval_corr_dirs],
        "max_points_per_file": int(max_points_per_file),
        "max_total_train_points": int(max_total_train_points),
        "max_total_eval_points": int(max_total_eval_points),
        "epochs": int(epochs),
        "lr": float(lr),
        "seed": int(seed),
        "train_stats": train_stats,
        "model": model.to_json(feature_names),
        "eval": eval_results,
    }
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_corr_dir", nargs="+", required=True)
    parser.add_argument("--eval_corr_dir", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--feature_set",
        choices=("confidence", "confidence_query_xy", "confidence_query_xy_map_xy"),
        default="confidence_query_xy",
    )
    parser.add_argument("--max_points_per_file", type=int, default=0)
    parser.add_argument("--max_total_train_points", type=int, default=0)
    parser.add_argument("--max_total_eval_points", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_reliability_experiment(
        train_corr_dirs=args.train_corr_dir,
        eval_corr_dirs=args.eval_corr_dir,
        output_json=args.output_json,
        feature_set=args.feature_set,
        max_points_per_file=args.max_points_per_file,
        max_total_train_points=args.max_total_train_points,
        max_total_eval_points=args.max_total_eval_points,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
