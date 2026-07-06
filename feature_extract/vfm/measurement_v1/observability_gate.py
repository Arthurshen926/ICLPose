from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.measurement_v1.rgb_patch_training import _binary_auroc, _binary_ece


DEFAULT_OBSERVABILITY_GATE_FEATURES = (
    "mode_probability",
    "entropy_norm",
    "peak_gap_z",
    "logit_std",
    "render_texture",
    "radio_match_score",
    "mode_abs_dx",
    "mode_abs_dy",
)

POSITIVE_OBSERVABILITY_CLASSES = frozenset({"observable_subpixel", "observable_coarse_only"})

FORBIDDEN_OBSERVABILITY_GATE_FEATURES = frozenset(
    {
        "gt_rank",
        "gt_probability",
        "gt_in_window",
        "mode_epe_px",
        "center_residual_px",
        "target_dx",
        "target_dy",
        "query_gt_x",
        "query_gt_y",
        "target_is_dustbin",
        "requested_residual_px",
    }
)


@dataclass(frozen=True)
class ObservabilityGateDataset:
    rows: list[dict[str, str]]
    feature_names: list[str]
    features: torch.Tensor
    labels: torch.Tensor


class ObservabilityGate(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        if hidden > 0:
            self.net = nn.Sequential(nn.Linear(int(input_dim), hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(int(input_dim), 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).reshape(-1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        return float(default)
    value = float(text)
    if math.isnan(value) or math.isinf(value):
        return float(default)
    return float(value)


def _feature_value(row: Mapping[str, object], name: str) -> float:
    if name == "mode_abs_dx":
        return abs(_float(row, "mode_dx", 0.0))
    if name == "mode_abs_dy":
        return abs(_float(row, "mode_dy", 0.0))
    return _float(row, name, 0.0)


def _validate_feature_names(feature_names: Sequence[str]) -> list[str]:
    selected = [str(name) for name in feature_names]
    forbidden = sorted(set(selected) & FORBIDDEN_OBSERVABILITY_GATE_FEATURES)
    if forbidden:
        raise ValueError(f"observability gate features must be inference-available; GT-derived fields are forbidden: {forbidden}")
    if not selected:
        raise ValueError("at least one observability gate feature is required")
    return selected


def load_observability_gate_dataset(
    rows_csv: Path,
    *,
    feature_names: Sequence[str] = DEFAULT_OBSERVABILITY_GATE_FEATURES,
    max_rows: int | None = None,
) -> ObservabilityGateDataset:
    selected_features = _validate_feature_names(feature_names)
    rows = _read_csv(Path(rows_csv))
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    values = [[_feature_value(row, name) for name in selected_features] for row in rows]
    labels = [1.0 if str(row.get("observability_class", "")).strip() in POSITIVE_OBSERVABILITY_CLASSES else 0.0 for row in rows]
    feature_tensor = torch.tensor(values, dtype=torch.float32) if values else torch.empty((0, len(selected_features)), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0,), dtype=torch.float32)
    return ObservabilityGateDataset(rows=[dict(row) for row in rows], feature_names=selected_features, features=feature_tensor, labels=label_tensor)


def _standardize(features: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features.float() - mean.reshape(1, -1)) / std.reshape(1, -1).clamp_min(1e-6)


def _fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(features.numel()) == 0:
        raise ValueError("cannot fit observability gate on an empty dataset")
    mean = torch.mean(features.float(), dim=0)
    std = torch.std(features.float(), dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _brier(probabilities: torch.Tensor, labels: torch.Tensor) -> float | None:
    probs = probabilities.detach().cpu().float().reshape(-1)
    target = labels.detach().cpu().float().reshape(-1)
    if int(probs.numel()) == 0:
        return None
    return float(torch.mean((probs - target) ** 2).item())


def _threshold_stats(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float | int]:
    probs = probabilities.detach().cpu().float().reshape(-1)
    target = labels.detach().cpu().bool().reshape(-1)
    accept = probs >= float(threshold)
    accepted = int(torch.sum(accept).item())
    positives = int(torch.sum(target).item())
    true_positive = int(torch.sum(accept & target).item())
    precision = float(true_positive) / float(accepted) if accepted else 0.0
    recall = float(true_positive) / float(positives) if positives else 0.0
    coverage = float(accepted) / float(probs.numel()) if int(probs.numel()) else 0.0
    return {"accepted": accepted, "true_positive": true_positive, "precision": precision, "recall": recall, "coverage": coverage}


def _select_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_precision: float,
    minimum_threshold: float = 0.5,
) -> tuple[float, dict[str, float | int | str]]:
    probs = probabilities.detach().cpu().float().reshape(-1)
    if int(probs.numel()) == 0:
        return 1.0, {"strategy": "empty", "accepted": 0, "precision": 0.0, "recall": 0.0, "coverage": 0.0}
    all_candidates = sorted({float(value) for value in probs.tolist()}, reverse=True)
    candidates = [value for value in all_candidates if value >= float(minimum_threshold)]
    if not candidates:
        candidates = all_candidates[:1]
    best: tuple[float, dict[str, float | int]] | None = None
    for threshold in candidates:
        stats = _threshold_stats(probs, labels, threshold)
        if int(stats["accepted"]) > 0 and float(stats["precision"]) >= float(target_precision):
            if best is None or float(stats["recall"]) > float(best[1]["recall"]) or (
                float(stats["recall"]) == float(best[1]["recall"]) and float(stats["coverage"]) > float(best[1]["coverage"])
            ):
                best = (float(threshold), stats)
    if best is not None:
        threshold, stats = best
        return threshold, {"strategy": "target_precision", "minimum_threshold": float(minimum_threshold), **stats}
    f1_best: tuple[float, dict[str, float | int], float] | None = None
    for threshold in candidates:
        stats = _threshold_stats(probs, labels, threshold)
        precision = float(stats["precision"])
        recall = float(stats["recall"])
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1_best is None or f1 > f1_best[2]:
            f1_best = (float(threshold), stats, f1)
    assert f1_best is not None
    threshold, stats, f1 = f1_best
    return threshold, {"strategy": "max_f1", "minimum_threshold": float(minimum_threshold), "f1": float(f1), **stats}


def _row_fieldnames(rows: Sequence[Mapping[str, object]], appended: Sequence[str]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(str(key))
    for key in appended:
        if key not in names:
            names.append(str(key))
    return names


def _accepted_quality_metrics(
    rows: Sequence[Mapping[str, object]],
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    probs = probabilities.detach().cpu().float().reshape(-1)
    target = labels.detach().cpu().float().reshape(-1)
    accept = probs >= float(threshold)
    stats = _threshold_stats(probs, labels, float(threshold))
    out: dict[str, float | int | None] = {
        "threshold": float(threshold),
        "accepted_count": int(stats["accepted"]),
        "accepted_coverage": float(stats["coverage"]),
        "accepted_precision": float(stats["precision"]),
        "accepted_recall": float(stats["recall"]),
        "positive_count": int(torch.sum(target > 0.5).item()),
        "positive_fraction": float(torch.mean(target).item()) if int(target.numel()) else None,
        "accuracy_at_0p5": float(torch.mean(((probs >= 0.5).float() == target).float()).item()) if int(target.numel()) else None,
        "auroc": _binary_auroc(probs, target),
        "ece": _binary_ece(probs, target),
        "brier": _brier(probs, target),
    }
    accepted_indices = torch.nonzero(accept, as_tuple=False).reshape(-1).tolist()
    mode_epes: list[float] = []
    improve_flags: list[bool] = []
    for index in accepted_indices:
        row = rows[int(index)]
        mode_text = str(row.get("mode_epe_px", "")).strip()
        if mode_text:
            mode_epe = float(mode_text)
            mode_epes.append(mode_epe)
            center_text = str(row.get("center_residual_px", "")).strip()
            if center_text:
                improve_flags.append(mode_epe < float(center_text))
    if mode_epes:
        values = np.asarray(mode_epes, dtype=np.float64)
        out["accepted_mode_epe_median_px"] = float(np.median(values))
        out["accepted_mode_epe_p90_px"] = float(np.percentile(values, 90.0))
    else:
        out["accepted_mode_epe_median_px"] = None
        out["accepted_mode_epe_p90_px"] = None
    out["accepted_mode_improve_ratio"] = float(np.mean(improve_flags)) if improve_flags else None
    return out


def _score_rows(
    rows: Sequence[Mapping[str, object]],
    probabilities: torch.Tensor,
    *,
    threshold: float,
) -> list[dict[str, object]]:
    probs = probabilities.detach().cpu().float().reshape(-1)
    out: list[dict[str, object]] = []
    for row, prob in zip(rows, probs.tolist()):
        scored = dict(row)
        scored["observability_prob"] = f"{float(prob):.9g}"
        scored["observability_accept"] = "1" if float(prob) >= float(threshold) else "0"
        out.append(scored)
    return out


def train_observability_gate(
    *,
    train_rows_csv: Path,
    val_rows_csv: Path,
    output_dir: Path,
    feature_names: Sequence[str] = DEFAULT_OBSERVABILITY_GATE_FEATURES,
    steps: int = 1000,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dim: int = 16,
    target_precision: float = 0.85,
    minimum_threshold: float = 0.5,
    threshold_split: str = "val",
    device: str = "cuda",
    seed: int = 0,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
) -> dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    train = load_observability_gate_dataset(Path(train_rows_csv), feature_names=feature_names, max_rows=max_train_rows)
    val = load_observability_gate_dataset(Path(val_rows_csv), feature_names=train.feature_names, max_rows=max_val_rows)
    if int(train.features.shape[0]) == 0:
        raise ValueError("train_rows_csv has no rows")
    if int(val.features.shape[0]) == 0:
        raise ValueError("val_rows_csv has no rows")
    positive_count = int(torch.sum(train.labels > 0.5).item())
    negative_count = int(train.labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("observability gate training requires both positive and negative rows")
    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    mean, std = _fit_standardizer(train.features)
    model = ObservabilityGate(input_dim=len(train.feature_names), hidden_dim=int(hidden_dim)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x_train = _standardize(train.features, mean=mean, std=std).to(torch_device)
    y_train = train.labels.to(torch_device)
    pos_weight = torch.tensor([float(negative_count) / max(float(positive_count), 1.0)], dtype=torch.float32, device=torch_device)
    total = int(x_train.shape[0])
    batch = max(1, min(int(batch_size), total))
    for _ in range(max(1, int(steps))):
        if batch >= total:
            indices = torch.arange(total, device=torch_device)
        else:
            indices = torch.randint(0, total, (batch,), device=torch_device)
        logits = model(x_train[indices])
        loss = F.binary_cross_entropy_with_logits(logits, y_train[indices], pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(x_train)).detach().cpu()
        val_x = _standardize(val.features, mean=mean, std=std).to(torch_device)
        val_probs = torch.sigmoid(model(val_x)).detach().cpu()
    threshold_source = str(threshold_split).strip().lower()
    if threshold_source not in {"train", "val"}:
        raise ValueError("threshold_split must be 'train' or 'val'")
    threshold_probs = val_probs if threshold_source == "val" else train_probs
    threshold_labels = val.labels if threshold_source == "val" else train.labels
    threshold, threshold_stats = _select_threshold(threshold_probs, threshold_labels, target_precision=float(target_precision), minimum_threshold=float(minimum_threshold))
    train_metrics = _accepted_quality_metrics(train.rows, train_probs, train.labels, threshold=float(threshold))
    val_metrics = _accepted_quality_metrics(val.rows, val_probs, val.labels, threshold=float(threshold))
    scored_train = _score_rows(train.rows, train_probs, threshold=float(threshold))
    scored_val = _score_rows(val.rows, val_probs, threshold=float(threshold))
    accepted_train = [row for row in scored_train if str(row.get("observability_accept", "")) == "1"]
    accepted_val = [row for row in scored_val if str(row.get("observability_accept", "")) == "1"]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_train_csv = out_dir / "scored_train_rows.csv"
    scored_val_csv = out_dir / "scored_val_rows.csv"
    accepted_train_csv = out_dir / "accepted_train_rows.csv"
    accepted_val_csv = out_dir / "accepted_val_rows.csv"
    appended = ["observability_prob", "observability_accept"]
    _write_csv(scored_train_csv, scored_train, fieldnames=_row_fieldnames(scored_train, appended))
    _write_csv(scored_val_csv, scored_val, fieldnames=_row_fieldnames(scored_val, appended))
    _write_csv(accepted_train_csv, accepted_train, fieldnames=_row_fieldnames(scored_train, appended))
    _write_csv(accepted_val_csv, accepted_val, fieldnames=_row_fieldnames(scored_val, appended))
    checkpoint_path = out_dir / "observability_gate.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": train.feature_names,
            "feature_mean": mean,
            "feature_std": std,
            "hidden_dim": int(hidden_dim),
            "threshold": float(threshold),
        },
        checkpoint_path,
    )
    summary: dict[str, Any] = {
        "train_rows_csv": str(train_rows_csv),
        "val_rows_csv": str(val_rows_csv),
        "feature_names": train.feature_names,
        "forbidden_features": sorted(FORBIDDEN_OBSERVABILITY_GATE_FEATURES),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "hidden_dim": int(hidden_dim),
        "device": str(torch_device),
        "threshold_selection": {"target_precision": float(target_precision), "threshold": float(threshold), "threshold_split": threshold_source, **threshold_stats},
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "outputs": {
            "checkpoint": str(checkpoint_path),
            "scored_train_rows_csv": str(scored_train_csv),
            "scored_val_rows_csv": str(scored_val_csv),
            "accepted_train_rows_csv": str(accepted_train_csv),
            "accepted_val_rows_csv": str(accepted_val_csv),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary
