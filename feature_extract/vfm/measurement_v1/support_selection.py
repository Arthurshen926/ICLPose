from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.artifacts import file_sha256_short


SUPPORT_SELECTOR_FEATURE_NAMES: tuple[str, ...] = (
    "support_view_probability",
    "support_reprojection_quality",
    "measurement_accept_probability",
    "likelihood_confidence",
    "likelihood_peak_probability",
    "likelihood_peak_margin",
    "likelihood_covariance_quality",
    "mean_offset_norm",
    "mode_offset_norm",
)


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    text = str(row.get(key, "")).strip()
    return float(default) if not text else float(text)


def _bool(row: Mapping[str, object], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "t", "yes", "y"}


def support_selector_features(row: Mapping[str, object]) -> np.ndarray:
    pred_dx = _float(row, "pred_dx")
    pred_dy = _float(row, "pred_dy")
    peak_dx = _float(row, "peak_dx")
    peak_dy = _float(row, "peak_dy")
    return np.asarray(
        [
            _float(row, "support_view_probability"),
            1.0 / (1.0 + max(_float(row, "support_reprojection_error", 4.0), 0.0)),
            1.0 - _float(row, "dustbin_probability", 0.5),
            1.0 - _float(row, "likelihood_normalized_entropy", 1.0),
            _float(row, "likelihood_peak_probability"),
            _float(row, "likelihood_peak_margin"),
            1.0 / (1.0 + max(_float(row, "likelihood_covariance_max_sigma_px", 4.0), 0.0)),
            math.hypot(pred_dx, pred_dy),
            math.hypot(peak_dx, peak_dy),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MeasurementSupportSelector:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    temperature: float

    def scores(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        features = np.stack([support_selector_features(row) for row in rows])
        standardized = (features - np.asarray(self.mean)[None, :]) / np.asarray(self.scale)[None, :]
        return standardized @ np.asarray(self.weights) + float(self.bias)

    def probabilities(self, rows: Sequence[Mapping[str, object]]) -> np.ndarray:
        scores = self.scores(rows)
        scaled = (scores - float(np.max(scores))) / max(float(self.temperature), 1e-8)
        values = np.exp(np.clip(scaled, -80.0, 0.0))
        return values / max(float(np.sum(values)), 1e-12)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "temperature": float(self.temperature),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MeasurementSupportSelector":
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weights=tuple(float(value) for value in payload["weights"]),
            bias=float(payload["bias"]),
            temperature=float(payload["temperature"]),
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _groups(rows: Sequence[Mapping[str, object]], *, valid_only: bool) -> dict[int, list[Mapping[str, object]]]:
    output: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if valid_only and _bool(row, "target_is_dustbin"):
            continue
        policy_row = int(_float(row, "policy_row_index", -1.0))
        if policy_row < 0:
            raise ValueError("diagnostic row is missing policy_row_index")
        output[policy_row].append(row)
    return dict(output)


def _diagnostic_manifest(path: Path) -> dict[str, object]:
    summary_path = Path(path).parent / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"diagnostic summary is required: {summary_path}")
    summary = json.loads(summary_path.read_text())
    actual_hash = file_sha256_short(Path(path))
    expected_hash = str(summary.get("outputs", {}).get("diagnostic_rows_sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError("diagnostic rows hash does not match its summary")
    checkpoint_hash = str(summary.get("checkpoint_sha256", ""))
    if not checkpoint_hash:
        raise ValueError("diagnostic summary is missing checkpoint_sha256")
    return {
        "diagnostic_rows": str(path),
        "diagnostic_rows_sha256": actual_hash,
        "summary": str(summary_path),
        "summary_sha256": file_sha256_short(summary_path),
        "checkpoint_sha256": checkpoint_hash,
    }


def _fit_pairwise(rows: Sequence[Mapping[str, object]], *, c_value: float) -> MeasurementSupportSelector:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for group_rows in _groups(rows, valid_only=True).values():
        for first_index in range(len(group_rows)):
            for second_index in range(first_index + 1, len(group_rows)):
                first = group_rows[first_index]
                second = group_rows[second_index]
                first_epe = _float(first, "likelihood_epe_px")
                second_epe = _float(second, "likelihood_epe_px")
                if abs(first_epe - second_epe) <= 1e-8:
                    continue
                difference = support_selector_features(first) - support_selector_features(second)
                differences.extend((difference, -difference))
                labels.extend((int(first_epe < second_epe), int(second_epe < first_epe)))
    if not differences or len(set(labels)) < 2:
        raise ValueError("support selector requires at least one non-tied multi-view training pair")
    features = np.stack(differences)
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (features - mean[None, :]) / scale[None, :]
    classifier = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000, random_state=0)
    classifier.fit(standardized, np.asarray(labels, dtype=np.int64))
    return MeasurementSupportSelector(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in classifier.coef_[0]),
        bias=float(classifier.intercept_[0]),
        temperature=1.0,
    )


def _group_metrics(rows: Sequence[Mapping[str, object]], selector: MeasurementSupportSelector) -> dict[str, object]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for group_rows in _groups(rows, valid_only=True).values():
        target = np.asarray([_float(group_rows[0], "target_dx"), _float(group_rows[0], "target_dy")])
        offsets = np.asarray([[_float(row, "pred_dx"), _float(row, "pred_dy")] for row in group_rows])
        epes = np.asarray([_float(row, "likelihood_epe_px") for row in group_rows])
        static = np.asarray([max(_float(row, "support_view_probability"), 0.0) for row in group_rows])
        if float(np.sum(static)) <= 0.0:
            static = np.ones_like(static)
        static /= float(np.sum(static))
        learned = selector.probabilities(group_rows)
        metrics["static_top1"].append(float(epes[int(np.argmax(static))]))
        metrics["static_fused"].append(float(np.linalg.norm(np.sum(static[:, None] * offsets, axis=0) - target)))
        metrics["learned_top1"].append(float(epes[int(np.argmax(learned))]))
        metrics["learned_fused"].append(float(np.linalg.norm(np.sum(learned[:, None] * offsets, axis=0) - target)))
        metrics["oracle_best_view_TARGET_ONLY"].append(float(np.min(epes)))
    output: dict[str, object] = {"group_count": len(next(iter(metrics.values()), []))}
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        output[name] = {
            "median_epe_px": float(np.median(array)),
            "mean_epe_px": float(np.mean(array)),
            "p90_epe_px": float(np.quantile(array, 0.9)),
            "recall_1px": float(np.mean(array <= 1.0)),
        }
    return output


def _with_temperature(selector: MeasurementSupportSelector, temperature: float) -> MeasurementSupportSelector:
    return MeasurementSupportSelector(
        mean=selector.mean,
        scale=selector.scale,
        weights=selector.weights,
        bias=selector.bias,
        temperature=float(temperature),
    )


def _write_selector_rows(path: Path, rows: Sequence[Mapping[str, object]], selector: MeasurementSupportSelector) -> None:
    fields = [
        "diagnostic_row_index",
        "policy_row_index",
        "query_id",
        "track_id",
        "support_view_rank",
        "selector_score",
        "selector_probability",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_rows in _groups(rows, valid_only=False).values():
            scores = selector.scores(group_rows)
            probabilities = selector.probabilities(group_rows)
            for row, score, probability in zip(group_rows, scores, probabilities):
                writer.writerow(
                    {
                        "diagnostic_row_index": int(_float(row, "row_index")),
                        "policy_row_index": int(_float(row, "policy_row_index")),
                        "query_id": str(row.get("query_id", "")),
                        "track_id": str(row.get("track_id", "")),
                        "support_view_rank": str(row.get("support_view_rank", "")),
                        "selector_score": float(score),
                        "selector_probability": float(probability),
                    }
                )


def fit_measurement_support_selector(
    *,
    train_diagnostics_csv: Path,
    validation_diagnostics_csv: Path,
    output_dir: Path,
    c_value: float = 0.25,
    temperatures: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
) -> dict[str, object]:
    train_manifest = _diagnostic_manifest(Path(train_diagnostics_csv))
    validation_manifest = _diagnostic_manifest(Path(validation_diagnostics_csv))
    if train_manifest["checkpoint_sha256"] != validation_manifest["checkpoint_sha256"]:
        raise ValueError("train and validation diagnostics use different measurement checkpoints")
    train_rows = _read_csv(Path(train_diagnostics_csv))
    validation_rows = _read_csv(Path(validation_diagnostics_csv))
    base_selector = _fit_pairwise(train_rows, c_value=float(c_value))
    candidates = []
    for temperature in temperatures:
        selector = _with_temperature(base_selector, float(temperature))
        metrics = _group_metrics(validation_rows, selector)
        candidates.append((metrics["learned_fused"]["median_epe_px"], metrics["learned_fused"]["mean_epe_px"], selector, metrics))
    _median, _mean, selector, validation_metrics = min(candidates, key=lambda value: (value[0], value[1]))
    train_metrics = _group_metrics(train_rows, selector)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "measurement_support_selector.json"
    model_payload = {
        "format": "measurement_support_selector_v1",
        "feature_names": list(SUPPORT_SELECTOR_FEATURE_NAMES),
        "model": selector.to_dict(),
        "measurement_checkpoint_sha256": train_manifest["checkpoint_sha256"],
        "train_diagnostics_sha256": train_manifest["diagnostic_rows_sha256"],
        "validation_diagnostics_sha256": validation_manifest["diagnostic_rows_sha256"],
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    train_predictions = output / "train_selector_rows.csv"
    validation_predictions = output / "validation_selector_rows.csv"
    _write_selector_rows(train_predictions, train_rows, selector)
    _write_selector_rows(validation_predictions, validation_rows, selector)
    passes = bool(
        validation_metrics["learned_fused"]["median_epe_px"]
        < validation_metrics["static_fused"]["median_epe_px"]
        and validation_metrics["learned_fused"]["mean_epe_px"]
        < validation_metrics["static_fused"]["mean_epe_px"]
    )
    summary = {
        "stage": "measurement_support_selector_fit",
        "protocol": {
            "fit_split": "train",
            "temperature_selection_split": "validation",
            "target_fields_used_as_features": False,
            "query_pose_used_as_feature": False,
            "render": False,
        },
        "feature_names": list(SUPPORT_SELECTOR_FEATURE_NAMES),
        "c_value": float(c_value),
        "temperature": float(selector.temperature),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "promotion_passes": passes,
        "inputs": {"train": train_manifest, "validation": validation_manifest},
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "train_predictions": str(train_predictions),
            "train_predictions_sha256": file_sha256_short(train_predictions),
            "validation_predictions": str(validation_predictions),
            "validation_predictions_sha256": file_sha256_short(validation_predictions),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def apply_measurement_support_selector(
    *, diagnostics_csv: Path, selector_json: Path, output_dir: Path
) -> dict[str, object]:
    manifest = _diagnostic_manifest(Path(diagnostics_csv))
    payload = json.loads(Path(selector_json).read_text())
    if payload.get("format") != "measurement_support_selector_v1":
        raise ValueError("unsupported measurement support selector format")
    if str(payload.get("measurement_checkpoint_sha256")) != str(manifest["checkpoint_sha256"]):
        raise ValueError("measurement support selector checkpoint mismatch")
    if tuple(payload.get("feature_names", ())) != SUPPORT_SELECTOR_FEATURE_NAMES:
        raise ValueError("measurement support selector feature schema mismatch")
    selector = MeasurementSupportSelector.from_dict(payload["model"])
    rows = _read_csv(Path(diagnostics_csv))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "selector_rows.csv"
    _write_selector_rows(predictions, rows, selector)
    summary = {
        "stage": "measurement_support_selector_apply",
        "protocol": {"target_fields_used_as_features": False, "threshold_search": False},
        "group_count": len(_groups(rows, valid_only=False)),
        "inputs": {
            "diagnostics": manifest,
            "selector_json": str(selector_json),
            "selector_sha256": file_sha256_short(Path(selector_json)),
        },
        "outputs": {
            "predictions": str(predictions),
            "predictions_sha256": file_sha256_short(predictions),
        },
    }
    summary_path = output / "summary.json"
    summary["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
