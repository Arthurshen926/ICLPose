from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.dense_depth_measurement_fusion import dense_depth_matches_from_rows
from feature_extract.vfm.dense_depth_pose_error_budget import _run_pnp_solver_ablation_safe


RAW_FEATURE_COLUMNS = [
    "radio_match_score",
    "measurement_valid_prob",
    "local_cost_entropy",
    "inverse_local_cost_entropy",
    "local_cost_peak_prob",
    "local_cost_top2_gap",
    "render_depth",
    "log_render_depth",
    "query_center_x_norm",
    "query_center_y_norm",
]

RANK_SOURCE_COLUMNS = [
    "radio_match_score",
    "measurement_valid_prob",
    "inverse_local_cost_entropy",
    "local_cost_peak_prob",
    "local_cost_top2_gap",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter=str(delimiter))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def _bool_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    return int(str(value).strip().lower() in {"1", "true", "yes", "y"})


def _candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id")
    return "" if value is None else str(value)


def _group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("query_id", "")), _candidate_id(row)


def _rank_percentiles(values: Sequence[float | None]) -> list[float]:
    finite_indices = [index for index, value in enumerate(values) if value is not None and np.isfinite(float(value))]
    out = [0.0 for _ in values]
    count = len(finite_indices)
    if count == 0:
        return out
    ordered = sorted(finite_indices, key=lambda index: float(values[index]), reverse=True)
    if count == 1:
        out[ordered[0]] = 1.0
        return out
    for rank, index in enumerate(ordered):
        out[index] = float((count - rank) / count)
    return out


def build_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    image_width: int = 1920,
    image_height: int = 1080,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build non-leakage center-validity features from validity rows.

    Labels and `center_error_px` are copied for supervision, but they are
    intentionally not included in `feature_columns`.
    """

    output: list[dict[str, Any]] = []
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    for row in rows:
        item = dict(row)
        entropy = _finite_float(row.get("local_cost_entropy"))
        depth = _finite_float(row.get("render_depth"))
        qx = _finite_float(row.get("query_center_x"))
        qy = _finite_float(row.get("query_center_y"))
        item["inverse_local_cost_entropy"] = "" if entropy is None else float(1.0 - entropy)
        item["log_render_depth"] = "" if depth is None or depth <= 0.0 else float(math.log(depth))
        item["query_center_x_norm"] = "" if qx is None else float(qx / width)
        item["query_center_y_norm"] = "" if qy is None else float(qy / height)
        output.append(item)
    by_group: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(output):
        by_group.setdefault(_group_key(row), []).append(index)
    rank_columns: list[str] = []
    for column in RANK_SOURCE_COLUMNS:
        rank_column = f"{column}_rank_pct"
        rank_columns.append(rank_column)
        for indices in by_group.values():
            ranks = _rank_percentiles([_finite_float(output[index].get(column)) for index in indices])
            for index, rank in zip(indices, ranks):
                output[index][rank_column] = float(rank)
    feature_columns = list(RAW_FEATURE_COLUMNS) + rank_columns
    return output, feature_columns


def split_rows_by_query(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.7,
    calibration_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    queries = sorted({str(row.get("query_id", "")) for row in rows})
    rng = random.Random(int(seed))
    rng.shuffle(queries)
    train_end = int(round(len(queries) * float(train_fraction)))
    cal_end = train_end + int(round(len(queries) * float(calibration_fraction)))
    if queries and train_end <= 0:
        train_end = 1
    train_queries = set(queries[:train_end])
    calibration_queries = set(queries[train_end:cal_end])
    validation_queries = set(queries[cal_end:])
    if queries and not validation_queries and calibration_queries:
        moved = sorted(calibration_queries)[-1]
        calibration_queries.remove(moved)
        validation_queries.add(moved)
    splits = {"train": [], "calibration": [], "validation": []}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        item = dict(row)
        if query_id in train_queries:
            splits["train"].append(item)
        elif query_id in calibration_queries:
            splits["calibration"].append(item)
        else:
            splits["validation"].append(item)
    return splits


def _matrix(rows: Sequence[Mapping[str, Any]], feature_columns: Sequence[str]) -> np.ndarray:
    values = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for col_index, name in enumerate(feature_columns):
            value = _finite_float(row.get(name))
            values[row_index, col_index] = 0.0 if value is None else float(value)
    return values


def _labels(rows: Sequence[Mapping[str, Any]], target: str) -> np.ndarray:
    return np.asarray([_bool_label(row.get(target, False)) for row in rows], dtype=np.float32)


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True) if train.size else np.zeros((1, train.shape[1]), dtype=np.float32)
    std = train.std(axis=0, keepdims=True) if train.size else np.ones((1, train.shape[1]), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return (train - mean) / std, [(item - mean) / std for item in others], mean.reshape(-1), std.reshape(-1)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = scores[valid]
    y = labels[valid].astype(bool)
    positives = int(np.count_nonzero(y))
    if positives == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted) / (np.arange(y_sorted.size, dtype=np.float64) + 1.0)
    return float(np.sum(precision[y_sorted]) / positives)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = scores[valid]
    y = labels[valid].astype(bool)
    pos = int(np.count_nonzero(y))
    neg = int(y.size - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    del unique
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        ranks = (rank_sums / counts)[inverse]
    pos_rank_sum = float(np.sum(ranks[y]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _ece(scores: np.ndarray, labels: np.ndarray, *, bin_count: int = 10) -> float | None:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return None
    s = np.clip(scores[valid], 0.0, 1.0)
    y = labels[valid].astype(np.float64)
    ece = 0.0
    for index in range(int(bin_count)):
        lo = index / float(bin_count)
        hi = (index + 1) / float(bin_count)
        mask = (s >= lo) & (s <= hi if index == int(bin_count) - 1 else s < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(s[mask])) - float(np.mean(y[mask])))
    return float(ece)


def _precision_at_coverages(scores: np.ndarray, labels: np.ndarray, coverages: Sequence[float]) -> list[dict[str, Any]]:
    valid = np.isfinite(scores) & np.isfinite(labels)
    if not np.any(valid):
        return []
    s = scores[valid]
    y = labels[valid].astype(bool)
    order = np.argsort(-s, kind="mergesort")
    out: list[dict[str, Any]] = []
    for coverage in coverages:
        k = max(1, int(math.ceil(float(coverage) * y.size)))
        selected = y[order[:k]]
        out.append(
            {
                "coverage": float(coverage),
                "selected_count": int(k),
                "positive_count": int(np.count_nonzero(selected)),
                "precision": float(np.mean(selected)) if selected.size else None,
            }
        )
    return out


def evaluate_scores(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    coverages: Sequence[float] = (0.05, 0.1, 0.2, 0.4, 0.6, 1.0),
) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.shape[0] != y.shape[0]:
        raise ValueError("scores and labels must have the same length")
    valid = np.isfinite(s) & np.isfinite(y)
    clipped = np.clip(s[valid], 0.0, 1.0)
    return {
        "row_count": int(s.shape[0]),
        "finite_count": int(np.count_nonzero(valid)),
        "positive_count": int(np.count_nonzero(y[valid] > 0.5)),
        "positive_rate": float(np.mean(y[valid])) if np.any(valid) else None,
        "auroc": _auroc(s, y),
        "auprc": _average_precision(s, y),
        "brier_clipped": None if not np.any(valid) else float(np.mean((clipped - y[valid]) ** 2)),
        "ece_clipped": _ece(s, y),
        "precision_at_coverage": _precision_at_coverages(s, y, coverages),
    }


class LogisticValidityModel(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(input_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPValidityModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(int(input_dim), int(hidden_dim)), nn.ReLU(), nn.Linear(int(hidden_dim), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    if logits.size == 0:
        return 1.0
    logit_tensor = torch.as_tensor(logits, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    best_temp = 1.0
    best_loss = float("inf")
    for temp in np.geomspace(0.25, 8.0, num=40):
        loss = F.binary_cross_entropy_with_logits(logit_tensor / float(temp), label_tensor).item()
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp


def train_validity_model(
    feature_rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    *,
    target: str = "valid_5px",
    model_type: str = "logistic",
    seed: int = 0,
    steps: int = 500,
    learning_rate: float = 1e-2,
    train_fraction: float = 0.7,
    calibration_fraction: float = 0.15,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    splits = split_rows_by_query(feature_rows, train_fraction=train_fraction, calibration_fraction=calibration_fraction, seed=seed)
    train_rows = splits["train"]
    cal_rows = splits["calibration"]
    val_rows = splits["validation"]
    x_train = _matrix(train_rows, feature_columns)
    x_cal = _matrix(cal_rows, feature_columns)
    x_val = _matrix(val_rows, feature_columns)
    y_train = _labels(train_rows, target)
    y_cal = _labels(cal_rows, target)
    y_val = _labels(val_rows, target)
    x_train_std, (x_cal_std, x_val_std), mean, std = _standardize(x_train, x_cal, x_val)
    model: nn.Module
    if str(model_type) == "mlp":
        model = MLPValidityModel(len(feature_columns))
    elif str(model_type) == "logistic":
        model = LogisticValidityModel(len(feature_columns))
    else:
        raise ValueError("model_type must be 'logistic' or 'mlp'")
    if x_train_std.shape[0] == 0:
        raise ValueError("training split is empty")
    pos = max(float(np.sum(y_train)), 1.0)
    neg = max(float(y_train.shape[0] - np.sum(y_train)), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    x_tensor = torch.as_tensor(x_train_std, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_tensor)
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_logits = model(torch.as_tensor(x_train_std, dtype=torch.float32)).numpy()
        cal_logits = model(torch.as_tensor(x_cal_std, dtype=torch.float32)).numpy() if x_cal_std.shape[0] else np.zeros((0,), dtype=np.float32)
        val_logits = model(torch.as_tensor(x_val_std, dtype=torch.float32)).numpy() if x_val_std.shape[0] else np.zeros((0,), dtype=np.float32)
    temperature = _fit_temperature(cal_logits, y_cal)

    def probs(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)))

    return {
        "model": model,
        "feature_columns": list(feature_columns),
        "target": str(target),
        "model_type": str(model_type),
        "mean": mean,
        "std": std,
        "temperature": float(temperature),
        "splits": splits,
        "metrics": {
            "train": evaluate_scores(scores=probs(train_logits), labels=y_train),
            "calibration": evaluate_scores(scores=probs(cal_logits), labels=y_cal),
            "validation": evaluate_scores(scores=probs(val_logits), labels=y_val),
        },
        "logits": {
            "train": train_logits,
            "calibration": cal_logits,
            "validation": val_logits,
        },
    }


def predict_validity_scores(model_result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    feature_columns = list(model_result["feature_columns"])
    x = _matrix(rows, feature_columns)
    mean = np.asarray(model_result["mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(model_result["std"], dtype=np.float32).reshape(1, -1)
    x_std = (x - mean) / np.where(std < 1e-6, 1.0, std)
    model = model_result["model"]
    temperature = float(model_result.get("temperature", 1.0))
    with torch.no_grad():
        logits = model(torch.as_tensor(x_std, dtype=torch.float32)).numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64) / max(temperature, 1e-6)))


def _cell(row: Mapping[str, Any], *, image_width: int, image_height: int) -> tuple[int, int]:
    x = _finite_float(row.get("query_center_x"))
    y = _finite_float(row.get("query_center_y"))
    width = max(int(image_width), 1)
    height = max(int(image_height), 1)
    if x is None or y is None:
        return 0, 0
    return min(max(int(math.floor(x / width * 4.0)), 0), 3), min(max(int(math.floor(y / height * 4.0)), 0), 3)


def grid4_coverage(rows: Sequence[Mapping[str, Any]], *, image_width: int, image_height: int) -> int:
    return len({_cell(row, image_width=image_width, image_height=image_height) for row in rows})


def depth_bin(row: Mapping[str, Any], *, rows: Sequence[Mapping[str, Any]], bin_count: int = 4) -> int:
    finite = [_finite_float(item.get("render_depth")) for item in rows]
    values = [float(value) for value in finite if value is not None and np.isfinite(float(value))]
    if not values:
        return 0
    depth = _finite_float(row.get("render_depth"))
    if depth is None:
        return 0
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1e-9:
        return 0
    normalized = (float(depth) - lo) / (hi - lo)
    return min(max(int(math.floor(normalized * int(bin_count))), 0), int(bin_count) - 1)


def select_rows_by_score(rows: Sequence[Mapping[str, Any]], *, score_key: str, budget: int) -> list[dict[str, Any]]:
    ordered = sorted([dict(row) for row in rows], key=lambda row: _finite_float(row.get(score_key)) or -float("inf"), reverse=True)
    return ordered[: max(int(budget), 0)]


def select_rows_grid_balanced(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    budget: int,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    remaining = sorted(values, key=lambda row: _finite_float(row.get(score_key)) or -float("inf"), reverse=True)
    selected: list[dict[str, Any]] = []
    used_cells: set[tuple[int, int]] = set()
    for row in list(remaining):
        if len(selected) >= int(budget):
            break
        cell = _cell(row, image_width=image_width, image_height=image_height)
        if cell in used_cells:
            continue
        selected.append(row)
        used_cells.add(cell)
        remaining.remove(row)
    for row in remaining:
        if len(selected) >= int(budget):
            break
        selected.append(row)
    return selected


def select_rows_grid_depth_balanced(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    budget: int,
    image_width: int,
    image_height: int,
    depth_bin_count: int = 4,
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    remaining = sorted(values, key=lambda row: _finite_float(row.get(score_key)) or -float("inf"), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()

    def take_unique(key_fn) -> None:
        for row in list(remaining):
            if len(selected) >= int(budget):
                return
            key = key_fn(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)
            remaining.remove(row)

    take_unique(lambda row: ("grid", *_cell(row, image_width=image_width, image_height=image_height)))
    selected_keys.clear()
    take_unique(
        lambda row: (
            "grid_depth",
            *_cell(row, image_width=image_width, image_height=image_height),
            depth_bin(row, rows=values, bin_count=int(depth_bin_count)),
        )
    )
    for row in remaining:
        if len(selected) >= int(budget):
            break
        selected.append(row)
    return selected


def coverage_gate_selection(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    fallback_rows: Sequence[Mapping[str, Any]],
    score_key: str,
    image_width: int,
    image_height: int,
    min_kept: int = 20,
    min_grid4_coverage: int = 6,
    min_depth_span_m: float = 0.1,
    min_score_threshold: float | None = 0.5,
    min_score_count: int = 20,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in selected_rows]
    coverage = _coverage_summary(selected, image_width=int(image_width), image_height=int(image_height))
    if int(coverage["kept_row_count"]) < int(min_kept):
        return [dict(row) for row in fallback_rows]
    if int(coverage["grid4_coverage"]) < int(min_grid4_coverage):
        return [dict(row) for row in fallback_rows]
    depth_span = _finite_float(coverage.get("depth_span_m"))
    if depth_span is None or depth_span < float(min_depth_span_m):
        return [dict(row) for row in fallback_rows]
    if min_score_threshold is not None and int(min_score_count) > 0:
        confident = sum(1 for row in selected if (_finite_float(row.get(score_key)) or -float("inf")) >= float(min_score_threshold))
        if confident < int(min_score_count):
            return [dict(row) for row in fallback_rows]
    return selected


def select_rows_p5_then_secondary_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    p5_score_key: str,
    secondary_score_key: str,
    budget: int,
    image_width: int,
    image_height: int,
    p5_pool_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    if int(budget) <= 0:
        return []
    pool_size = max(int(budget), int(math.ceil(len(values) * float(p5_pool_fraction))))
    pool = select_rows_by_score(values, score_key=str(p5_score_key), budget=pool_size)
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in pool:
        by_cell.setdefault(_cell(row, image_width=int(image_width), image_height=int(image_height)), []).append(row)
    for cell_rows in by_cell.values():
        cell_rows.sort(
            key=lambda row: (
                _finite_float(row.get(secondary_score_key)) or -float("inf"),
                _finite_float(row.get(p5_score_key)) or -float("inf"),
            ),
            reverse=True,
        )
    selected: list[dict[str, Any]] = []
    cell_heads = [rows_for_cell[0] for rows_for_cell in by_cell.values() if rows_for_cell]
    cell_heads.sort(
        key=lambda row: (
            _finite_float(row.get(secondary_score_key)) or -float("inf"),
            _finite_float(row.get(p5_score_key)) or -float("inf"),
        ),
        reverse=True,
    )
    selected.extend(cell_heads[: int(budget)])
    selected_ids = {id(row) for row in selected}
    if len(selected) >= int(budget):
        return selected[: int(budget)]
    remaining = [row for cell_rows in by_cell.values() for row in cell_rows if id(row) not in selected_ids]
    remaining.sort(
        key=lambda row: (
            _finite_float(row.get(secondary_score_key)) or -float("inf"),
            _finite_float(row.get(p5_score_key)) or -float("inf"),
        ),
        reverse=True,
    )
    for row in remaining:
        if len(selected) >= int(budget):
            break
        selected.append(row)
    return selected


def write_feature_dataset(
    *,
    center_validity_rows_csv: Path,
    output_dir: Path,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    rows = _read_csv(Path(center_validity_rows_csv))
    feature_rows, feature_columns = build_feature_rows(rows, image_width=int(image_width), image_height=int(image_height))
    output = Path(output_dir)
    fieldnames = list(feature_rows[0].keys()) if feature_rows else []
    _write_csv(output / "features.csv", feature_rows, fieldnames)
    (output / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "center_validity_feature_dataset",
        "input_rows": int(len(rows)),
        "output_rows": int(len(feature_rows)),
        "feature_columns": feature_columns,
        "outputs": {
            "features_csv": str(output / "features.csv"),
            "feature_columns_json": str(output / "feature_columns.json"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row.get("query_id", "")), _candidate_id(row), str(row.get("match_index", ""))


def _center_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        x = _finite_float(item.get("query_center_x"))
        y = _finite_float(item.get("query_center_y"))
        if x is None or y is None:
            continue
        item["query_refined_x"] = float(x)
        item["query_refined_y"] = float(y)
        item["measurement_dx"] = 0.0
        item["measurement_dy"] = 0.0
        out.append(item)
    return out


def _xy_pair(row: Mapping[str, Any], x_name: str, y_name: str) -> tuple[float, float] | None:
    x = _finite_float(row.get(x_name))
    y = _finite_float(row.get(y_name))
    if x is None or y is None:
        return None
    return float(x), float(y)


def _set_query_refined_xy(row: Mapping[str, Any], xy: tuple[float, float]) -> dict[str, Any]:
    item = dict(row)
    item["query_refined_x"] = float(xy[0])
    item["query_refined_y"] = float(xy[1])
    center = _xy_pair(item, "query_center_x", "query_center_y")
    if center is not None:
        item["measurement_dx"] = float(xy[0] - center[0])
        item["measurement_dy"] = float(xy[1] - center[1])
    return item


def _oracle_update_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    update_variant: str,
    seed: int,
    source_variant: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(abs(hash((int(seed), str(source_variant), str(update_variant)))) % (2**32 - 1))
    for row in rows:
        if str(update_variant) == "center":
            xy = _xy_pair(row, "query_center_x", "query_center_y")
            if xy is None:
                continue
            out.append(_set_query_refined_xy(row, xy))
            continue
        gt_xy = _xy_pair(row, "query_gt_x", "query_gt_y")
        if gt_xy is None:
            continue
        if str(update_variant) == "oracle_xy":
            out.append(_set_query_refined_xy(row, gt_xy))
            continue
        if str(update_variant).startswith("oracle_noise_") and str(update_variant).endswith("px"):
            sigma_text = str(update_variant)[len("oracle_noise_") : -2].replace("p", ".")
            sigma = float(sigma_text)
            noise = rng.normal(loc=0.0, scale=float(sigma), size=2)
            out.append(_set_query_refined_xy(row, (float(gt_xy[0] + noise[0]), float(gt_xy[1] + noise[1]))))
            continue
        raise ValueError(f"unsupported update variant: {update_variant}")
    return out


def _rows_by_pose_group(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("query_id", "")), _candidate_id(row)), []).append(dict(row))
    return grouped


def _coverage_summary(rows: Sequence[Mapping[str, Any]], *, image_width: int, image_height: int) -> dict[str, Any]:
    depths = [_finite_float(row.get("render_depth")) for row in rows]
    finite_depths = [float(value) for value in depths if value is not None]
    return {
        "kept_row_count": int(len(rows)),
        "grid4_coverage": int(grid4_coverage(rows, image_width=int(image_width), image_height=int(image_height))),
        "depth_span_m": None if not finite_depths else float(max(finite_depths) - min(finite_depths)),
    }


def _valid_label(row: Mapping[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "y"}


def _group_coverage_row(
    *,
    query_id: str,
    candidate_id: str,
    variant: str,
    selected: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    coverage = _coverage_summary(selected, image_width=int(image_width), image_height=int(image_height))
    selected_valid_5 = sum(1 for row in selected if _valid_label(row, "valid_5px"))
    selected_valid_2 = sum(1 for row in selected if _valid_label(row, "valid_2px"))
    total_valid_5 = sum(1 for row in group_rows if _valid_label(row, "valid_5px"))
    total_valid_2 = sum(1 for row in group_rows if _valid_label(row, "valid_2px"))
    kept = max(len(selected), 1)
    return {
        "query_id": str(query_id),
        "candidate_id": str(candidate_id),
        "variant": str(variant),
        **coverage,
        "group_row_count": int(len(group_rows)),
        "selected_valid_5px_count": int(selected_valid_5),
        "selected_valid_2px_count": int(selected_valid_2),
        "group_valid_5px_count": int(total_valid_5),
        "group_valid_2px_count": int(total_valid_2),
        "valid_5px_precision": float(selected_valid_5 / kept),
        "valid_2px_precision": float(selected_valid_2 / kept),
        "valid_5px_recall": float(selected_valid_5 / total_valid_5) if total_valid_5 > 0 else None,
        "valid_2px_recall": float(selected_valid_2 / total_valid_2) if total_valid_2 > 0 else None,
    }


def _aggregate_pose_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row.get("variant", "")), str(row.get("solver", ""))), []).append(row)
    out: list[dict[str, Any]] = []
    for (variant, solver), values in sorted(by_key.items()):
        translations = [_finite_float(row.get("translation_error_m")) for row in values]
        rotations = [_finite_float(row.get("rotation_error_deg")) for row in values]
        successes = [str(row.get("success", "")).lower() in {"1", "true", "yes"} for row in values]
        kept = [_finite_float(row.get("kept_row_count")) for row in values]
        grids = [_finite_float(row.get("grid4_coverage")) for row in values]

        def percentile(items: Sequence[float | None], p: float) -> float | None:
            finite = np.asarray([float(item) for item in items if item is not None and np.isfinite(float(item))], dtype=np.float64)
            return None if finite.size == 0 else float(np.percentile(finite, p))

        def success_rate(t_threshold: float, r_threshold: float) -> float:
            count = 0
            for ok, t_err, r_err in zip(successes, translations, rotations):
                if ok and t_err is not None and r_err is not None and t_err <= t_threshold and r_err <= r_threshold:
                    count += 1
            return float(count / len(values)) if values else 0.0

        out.append(
            {
                "variant": variant,
                "solver": solver,
                "query_count": int(len(values)),
                "pnp_success_rate": float(sum(1 for item in successes if item) / len(values)) if values else 0.0,
                "median_translation_error_m": percentile(translations, 50.0),
                "p90_translation_error_m": percentile(translations, 90.0),
                "median_rotation_error_deg": percentile(rotations, 50.0),
                "p90_rotation_error_deg": percentile(rotations, 90.0),
                "success_3cm_1deg": success_rate(0.03, 1.0),
                "success_5cm_2deg": success_rate(0.05, 2.0),
                "success_10cm_5deg": success_rate(0.10, 5.0),
                "median_kept_row_count": percentile(kept, 50.0),
                "median_grid4_coverage": percentile(grids, 50.0),
            }
        )
    return out


def evaluate_center_validity_pose_filtering(
    *,
    match_table_csv: Path,
    score_rows_csv: Path,
    output_dir: Path,
    camera: Any,
    query_pose_w2c_by_id: Mapping[str, np.ndarray] | None = None,
    score_key: str = "p_valid_5px",
    secondary_score_key: str | None = None,
    budgets: Sequence[int] = (20, 50, 100, 200),
    solvers: Sequence[str] = ("ransac",),
    reprojection_error_px: float = 8.0,
    image_width: int = 1920,
    image_height: int = 1080,
    geometry_source: str = "prefer_world_xyz",
    seed: int = 20260706,
    score_thresholds: Sequence[float] = (0.2, 0.5, 0.8),
) -> dict[str, Any]:
    match_rows = _read_csv(Path(match_table_csv))
    score_rows = _read_csv(Path(score_rows_csv))
    score_by_key = {_row_key(row): row for row in score_rows}
    annotated: list[dict[str, Any]] = []
    missing_score_count = 0
    for row in match_rows:
        item = dict(row)
        score_row = score_by_key.get(_row_key(row))
        if score_row is None:
            missing_score_count += 1
            item[score_key] = ""
            item["valid_5px"] = ""
            item["valid_2px"] = ""
        else:
            item[score_key] = score_row.get(score_key, score_row.get("p_valid_5px", ""))
            item["valid_5px"] = score_row.get("valid_5px", "")
            item["valid_2px"] = score_row.get("valid_2px", "")
            for key, value in score_row.items():
                if str(key).startswith("p_") or str(key).startswith("pred_"):
                    item[str(key)] = value
        entropy = _finite_float(item.get("local_cost_entropy"))
        item["inverse_local_cost_entropy"] = "" if entropy is None else float(1.0 - entropy)
        annotated.append(item)
    groups = _rows_by_pose_group(annotated)
    pose_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    group_coverage_rows: list[dict[str, Any]] = []

    def add_variant_rows(query_id: str, candidate_id: str, variant: str, rows: Sequence[Mapping[str, Any]]) -> None:
        prepared = _center_rows(rows)
        coverage = _coverage_summary(prepared, image_width=int(image_width), image_height=int(image_height))
        group_coverage_rows.append(
            _group_coverage_row(
                query_id=str(query_id),
                candidate_id=str(candidate_id),
                variant=str(variant),
                selected=prepared,
                group_rows=groups.get((str(query_id), str(candidate_id)), []),
                image_width=int(image_width),
                image_height=int(image_height),
            )
        )
        matches, _match_summary = dense_depth_matches_from_rows(
            prepared,
            camera=camera,
            render_pose_w2c=None,
            source=f"center_validity_pose_filtering:{variant}",
            geometry_source=str(geometry_source),
        )
        report = _run_pnp_solver_ablation_safe(
            matches,
            camera,
            gt_pose_w2c=None if query_pose_w2c_by_id is None else query_pose_w2c_by_id.get(str(query_id)),
            solvers=tuple(str(value) for value in solvers),
            reprojection_error_px=float(reprojection_error_px),
        )
        for solver, row in report.items():
            pose_rows.append(
                {
                    "query_id": str(query_id),
                    "candidate_id": str(candidate_id),
                    "variant": str(variant),
                    "solver": str(solver),
                    **coverage,
                    **row,
                }
            )
        for row in prepared:
            selected = dict(row)
            selected["variant"] = str(variant)
            selected_rows.append(selected)

    for (query_id, candidate_id), rows in sorted(groups.items()):
        add_variant_rows(query_id, candidate_id, "center_all", rows)
        oracle_rows = [row for row in rows if str(row.get("valid_5px", "")).strip().lower() == "true"]
        add_variant_rows(query_id, candidate_id, "oracle_center_valid_5px", oracle_rows)
        for threshold in score_thresholds:
            threshold_rows = [row for row in rows if (_finite_float(row.get(score_key)) or -float("inf")) >= float(threshold)]
            add_variant_rows(query_id, candidate_id, f"learned_threshold_{float(threshold):.3f}".replace(".", "p"), threshold_rows)
        for budget in budgets:
            random_rows = [dict(row) for row in rows]
            rng = random.Random(f"{int(seed)}:{query_id}:{candidate_id}:{int(budget)}")
            rng.shuffle(random_rows)
            learned_top = select_rows_by_score(rows, score_key=score_key, budget=int(budget))
            learned_grid = select_rows_grid_balanced(rows, score_key=score_key, budget=int(budget), image_width=int(image_width), image_height=int(image_height))
            learned_grid_depth = select_rows_grid_depth_balanced(
                rows,
                score_key=score_key,
                budget=int(budget),
                image_width=int(image_width),
                image_height=int(image_height),
            )
            gate_kwargs = {
                "score_key": score_key,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "min_kept": min(20, int(budget)),
                "min_grid4_coverage": min(6, int(budget)),
                "min_depth_span_m": 0.1,
                "min_score_threshold": 0.5,
                "min_score_count": min(20, int(budget)),
            }
            add_variant_rows(query_id, candidate_id, f"learned_top{int(budget)}", learned_top)
            add_variant_rows(
                query_id,
                candidate_id,
                f"learned_top{int(budget)}_gated",
                coverage_gate_selection(selected_rows=learned_top, fallback_rows=rows, **gate_kwargs),
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"learned_grid_top{int(budget)}",
                learned_grid,
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"learned_grid_top{int(budget)}_gated",
                coverage_gate_selection(selected_rows=learned_grid, fallback_rows=rows, **gate_kwargs),
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"learned_grid_depth_top{int(budget)}",
                learned_grid_depth,
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"learned_grid_depth_top{int(budget)}_gated",
                coverage_gate_selection(selected_rows=learned_grid_depth, fallback_rows=rows, **gate_kwargs),
            )
            if secondary_score_key is not None and str(secondary_score_key).strip():
                secondary_key = str(secondary_score_key)
                secondary_top = select_rows_by_score(rows, score_key=secondary_key, budget=int(budget))
                secondary_grid = select_rows_grid_balanced(
                    rows,
                    score_key=secondary_key,
                    budget=int(budget),
                    image_width=int(image_width),
                    image_height=int(image_height),
                )
                p5_then_secondary_grid = select_rows_p5_then_secondary_grid(
                    rows,
                    p5_score_key=score_key,
                    secondary_score_key=secondary_key,
                    budget=int(budget),
                    image_width=int(image_width),
                    image_height=int(image_height),
                    p5_pool_fraction=0.5,
                )
                add_variant_rows(query_id, candidate_id, f"p2_top{int(budget)}", secondary_top)
                add_variant_rows(query_id, candidate_id, f"p2_grid_top{int(budget)}", secondary_grid)
                add_variant_rows(query_id, candidate_id, f"p5_then_p2_grid_top{int(budget)}", p5_then_secondary_grid)
            add_variant_rows(
                query_id,
                candidate_id,
                f"oracle_grid_top{int(budget)}",
                select_rows_grid_balanced(oracle_rows, score_key=score_key, budget=int(budget), image_width=int(image_width), image_height=int(image_height)),
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"oracle_grid_depth_top{int(budget)}",
                select_rows_grid_depth_balanced(
                    oracle_rows,
                    score_key=score_key,
                    budget=int(budget),
                    image_width=int(image_width),
                    image_height=int(image_height),
                ),
            )
            add_variant_rows(
                query_id,
                candidate_id,
                f"heuristic_entropy_top{int(budget)}",
                select_rows_by_score(rows, score_key="inverse_local_cost_entropy", budget=int(budget)),
            )
            add_variant_rows(query_id, candidate_id, f"random_top{int(budget)}", random_rows[: int(budget)])
    output = Path(output_dir)
    pose_fieldnames = list(pose_rows[0].keys()) if pose_rows else []
    selected_fieldnames = list(selected_rows[0].keys()) if selected_rows else []
    group_fieldnames = list(group_coverage_rows[0].keys()) if group_coverage_rows else []
    _write_csv(output / "pose_rows.csv", pose_rows, pose_fieldnames)
    _write_csv(output / "selected_rows.csv", selected_rows, selected_fieldnames)
    _write_csv(output / "group_coverage.tsv", group_coverage_rows, group_fieldnames, delimiter="\t")
    aggregate = _aggregate_pose_rows(pose_rows)
    _write_csv(output / "pose_filtering_summary.tsv", aggregate, list(aggregate[0].keys()) if aggregate else [], delimiter="\t")
    summary = {
        "stage": "center_validity_pose_filtering",
        "match_table_csv": str(match_table_csv),
        "score_rows_csv": str(score_rows_csv),
        "score_key": str(score_key),
        "row_count": int(len(match_rows)),
        "missing_score_count": int(missing_score_count),
        "pose_rows": pose_rows,
        "pose_summary": aggregate,
        "group_coverage_rows": group_coverage_rows,
        "outputs": {
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "selected_rows_csv": str(output / "selected_rows.csv"),
            "group_coverage_tsv": str(output / "group_coverage.tsv"),
            "pose_filtering_summary_tsv": str(output / "pose_filtering_summary.tsv"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def evaluate_selection_oracle_update_gap(
    *,
    selected_rows_csv: Path,
    output_dir: Path,
    camera: Any,
    query_pose_w2c_by_id: Mapping[str, np.ndarray] | None = None,
    source_variants: Sequence[str] = ("learned_top50", "learned_grid_top200_gated"),
    update_variants: Sequence[str] = ("center", "oracle_xy", "oracle_noise_1px", "oracle_noise_2px"),
    solvers: Sequence[str] = ("ransac",),
    reprojection_error_px: float = 8.0,
    image_width: int = 1920,
    image_height: int = 1080,
    geometry_source: str = "prefer_world_xyz",
    seed: int = 20260706,
) -> dict[str, Any]:
    rows = _read_csv(Path(selected_rows_csv))
    wanted = {str(value) for value in source_variants}
    filtered = [row for row in rows if str(row.get("variant", "")) in wanted]
    by_source_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in filtered:
        by_source_group.setdefault(
            (str(row.get("variant", "")), str(row.get("query_id", "")), _candidate_id(row)),
            [],
        ).append(dict(row))

    pose_rows: list[dict[str, Any]] = []

    for (source_variant, query_id, candidate_id), group_rows in sorted(by_source_group.items()):
        for update_variant in update_variants:
            prepared = _oracle_update_rows(
                group_rows,
                update_variant=str(update_variant),
                seed=int(seed),
                source_variant=str(source_variant),
            )
            coverage = _coverage_summary(prepared, image_width=int(image_width), image_height=int(image_height))
            matches, _match_summary = dense_depth_matches_from_rows(
                prepared,
                camera=camera,
                render_pose_w2c=None,
                source=f"selection_oracle_update_gap:{source_variant}:{update_variant}",
                geometry_source=str(geometry_source),
            )
            report = _run_pnp_solver_ablation_safe(
                matches,
                camera,
                gt_pose_w2c=None if query_pose_w2c_by_id is None else query_pose_w2c_by_id.get(str(query_id)),
                solvers=tuple(str(value) for value in solvers),
                reprojection_error_px=float(reprojection_error_px),
            )
            for solver, row in report.items():
                pose_rows.append(
                    {
                        "query_id": str(query_id),
                        "candidate_id": str(candidate_id),
                        "source_variant": str(source_variant),
                        "update_variant": str(update_variant),
                        "variant": f"{source_variant}_{update_variant}",
                        "solver": str(solver),
                        **coverage,
                        **row,
                    }
                )

    output = Path(output_dir)
    pose_fieldnames = list(pose_rows[0].keys()) if pose_rows else []
    _write_csv(output / "pose_rows.csv", pose_rows, pose_fieldnames)
    aggregate = _aggregate_pose_rows(pose_rows)
    _write_csv(output / "pose_summary.tsv", aggregate, list(aggregate[0].keys()) if aggregate else [], delimiter="\t")
    summary = {
        "stage": "selection_oracle_update_gap",
        "selected_rows_csv": str(selected_rows_csv),
        "source_variants": [str(value) for value in source_variants],
        "update_variants": [str(value) for value in update_variants],
        "row_count": int(len(rows)),
        "filtered_row_count": int(len(filtered)),
        "pose_rows": pose_rows,
        "pose_summary": aggregate,
        "outputs": {
            "pose_rows_csv": str(output / "pose_rows.csv"),
            "pose_summary_tsv": str(output / "pose_summary.tsv"),
            "summary": str(output / "summary.json"),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
