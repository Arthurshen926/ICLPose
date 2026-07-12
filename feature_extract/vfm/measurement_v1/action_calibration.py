from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics


ACTION_FEATURE_NAMES: tuple[str, ...] = (
    "assignment_score",
    "pose_selection_score",
    "geometry_p01",
    "geometry_p02",
    "geometry_p05",
    "log_track_length",
    "query_reprojection_quality",
    "support_reprojection_quality",
    "support_view_top_probability",
    "support_view_normalized_entropy",
    "support_view_angle_quality",
    "measurement_accept_probability",
    "measurement_update_gate_probability",
    "likelihood_confidence",
    "likelihood_peak_probability",
    "likelihood_peak_margin",
    "likelihood_covariance_quality",
    "fused_offset_norm",
    "view_offset_disagreement",
    "mean_mode_offset_disagreement",
    "coarse_pose_success",
    "coarse_pose_inlier",
    "coarse_pose_query_inlier_ratio",
    "coarse_pose_reprojection_quality",
    "coarse_pose_reprojection_log",
    "coarse_pose_center_le_1px",
    "coarse_pose_center_1_to_5px",
    "coarse_pose_center_gt_5px",
    "measurement_pose_offset_cosine",
    "measurement_pose_offset_agreement_quality",
    "measurement_pose_offset_scale_compatibility",
    "switched_from_baseline",
)


@dataclass(frozen=True)
class BinaryLinearProbabilityModel:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    constant_probability: float | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, len(self.mean))
        if self.constant_probability is not None:
            return np.full(
                (len(values),), float(self.constant_probability), dtype=np.float64
            )
        standardized = (
            values - np.asarray(self.mean, dtype=np.float64)[None, :]
        ) / np.asarray(self.scale, dtype=np.float64)[None, :]
        logits = standardized @ np.asarray(self.weights, dtype=np.float64) + float(
            self.bias
        )
        return _sigmoid(logits)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "constant_probability": self.constant_probability,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "BinaryLinearProbabilityModel":
        constant = payload.get("constant_probability")
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            scale=tuple(float(value) for value in payload["scale"]),
            weights=tuple(float(value) for value in payload["weights"]),
            bias=float(payload["bias"]),
            constant_probability=(
                None if constant is None else float(constant)
            ),
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_diagnostic_manifest(
    diagnostic_rows_csv: Path,
    *,
    expected_rows_csv: Path | None = None,
    required: bool,
) -> dict[str, object] | None:
    diagnostic_path = Path(diagnostic_rows_csv)
    summary_path = diagnostic_path.parent / "summary.json"
    if not summary_path.exists():
        if required:
            raise ValueError(
                f"diagnostic manifest is required but missing: {summary_path}"
            )
        return None
    payload = json.loads(summary_path.read_text())
    if payload.get("stage") != "measurement_v1_rgb_patch_diagnostics":
        raise ValueError("unsupported measurement diagnostic manifest")
    expected_diagnostic_hash = str(
        payload.get("outputs", {}).get("diagnostic_rows_sha256", "")
    )
    actual_diagnostic_hash = file_sha256_short(diagnostic_path)
    if not expected_diagnostic_hash:
        if required:
            raise ValueError("diagnostic manifest is missing diagnostic_rows_sha256")
    elif expected_diagnostic_hash != actual_diagnostic_hash:
        raise ValueError("stale or modified measurement diagnostic rows")
    if expected_rows_csv is not None:
        expected_rows_hash = str(payload.get("rows_csv_sha256", ""))
        actual_rows_hash = file_sha256_short(Path(expected_rows_csv))
        if not expected_rows_hash:
            if required:
                raise ValueError("diagnostic manifest is missing rows_csv_sha256")
        elif expected_rows_hash != actual_rows_hash:
            raise ValueError("measurement diagnostic source rows do not match")
    checkpoint_hash = str(payload.get("checkpoint_sha256", ""))
    if required and not checkpoint_hash:
        raise ValueError("diagnostic manifest is missing checkpoint_sha256")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256_short(summary_path),
        "diagnostic_rows_sha256": actual_diagnostic_hash,
        "rows_csv_sha256": str(payload.get("rows_csv_sha256", "")),
        "checkpoint": str(payload.get("checkpoint", "")),
        "checkpoint_sha256": checkpoint_hash,
    }


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    try:
        value = float(str(row.get(key, "")).strip())
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _optional_finite_float(
    row: Mapping[str, object], key: str
) -> float | None:
    text = str(row.get(key, "")).strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return float(value) if math.isfinite(value) else None


def _bool(row: Mapping[str, object], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }


def _weighted(values: Sequence[float], weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64) * weights))


def _merge_source_and_diagnostics(
    source_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    diagnostics_by_row: dict[int, Mapping[str, object]] = {}
    for row in diagnostic_rows:
        row_index = int(str(row.get("row_index", "-1")))
        if row_index in diagnostics_by_row:
            raise ValueError(f"duplicate diagnostic row_index: {row_index}")
        diagnostics_by_row[row_index] = row
    invalid_rows = sorted(
        row_index
        for row_index in diagnostics_by_row
        if row_index < 0 or row_index >= len(source_rows)
    )
    if invalid_rows:
        raise ValueError(
            f"diagnostic row_index is outside the source CSV: {invalid_rows[:5]}"
        )
    merged: list[dict[str, object]] = []
    for row_index in sorted(diagnostics_by_row):
        source = source_rows[row_index]
        diagnostic = diagnostics_by_row[row_index]
        for key in ("query_id", "track_id", "support_track_id"):
            if str(source.get(key, "")) != str(diagnostic.get(key, "")):
                raise ValueError(f"source/diagnostic identity mismatch at row {row_index}: {key}")
        merged.append({**source, **diagnostic})
    return merged


def _coarse_pose_context_by_policy_row(
    path: Path | None,
) -> dict[int, dict[str, str]]:
    if path is None or not str(path):
        return {}
    rows = _read_csv(Path(path))
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        policy_row = int(str(row.get("policy_row_index", "-1")))
        if policy_row < 0:
            raise ValueError("coarse-pose context is missing policy_row_index")
        if policy_row in output:
            raise ValueError(
                f"duplicate coarse-pose context policy row: {policy_row}"
            )
        output[policy_row] = dict(row)
    return output


def _measurement_support_selector_rows(
    path: Path | None,
) -> tuple[dict[int, dict[str, str]], dict[str, object] | None]:
    if path is None or not str(path):
        return {}, None
    rows = _read_csv(Path(path))
    by_diagnostic_row: dict[int, dict[str, str]] = {}
    for row in rows:
        row_index = int(str(row.get("diagnostic_row_index", "-1")))
        if row_index < 0:
            raise ValueError("measurement support selector row is missing diagnostic_row_index")
        if row_index in by_diagnostic_row:
            raise ValueError(f"duplicate measurement support selector row: {row_index}")
        probability = float(str(row.get("selector_probability", "nan")))
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError("measurement support selector probability is invalid")
        by_diagnostic_row[row_index] = dict(row)
    summary_path = Path(path).parent / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"measurement support selector summary is required: {summary_path}")
    summary = json.loads(summary_path.read_text())
    stage = str(summary.get("stage", ""))
    outputs = summary.get("outputs", {})
    actual_rows_hash = file_sha256_short(Path(path))
    if stage == "measurement_support_selector_fit":
        candidates = (
            (outputs.get("train_predictions"), outputs.get("train_predictions_sha256")),
            (outputs.get("validation_predictions"), outputs.get("validation_predictions_sha256")),
        )
        selector_hash = str(outputs.get("model_sha256", ""))
    elif stage == "measurement_support_selector_apply":
        candidates = ((outputs.get("predictions"), outputs.get("predictions_sha256")),)
        selector_hash = str(summary.get("inputs", {}).get("selector_sha256", ""))
    else:
        raise ValueError("unsupported measurement support selector summary")
    matching = [expected_hash for expected_path, expected_hash in candidates if str(expected_path) == str(path)]
    if len(matching) != 1 or str(matching[0]) != actual_rows_hash:
        raise ValueError("measurement support selector rows do not match their summary")
    if not selector_hash:
        raise ValueError("measurement support selector summary is missing selector hash")
    return by_diagnostic_row, {
        "rows_csv": str(path),
        "rows_sha256": actual_rows_hash,
        "summary": str(summary_path),
        "summary_sha256": file_sha256_short(summary_path),
        "selector_sha256": selector_hash,
    }


def build_action_examples(
    *,
    source_rows_csv: Path,
    diagnostic_rows_csv: Path,
    coarse_pose_context_csv: Path | None = None,
    measurement_support_selector_rows_csv: Path | None = None,
    minimum_update_gain_px: float = 0.1,
    maximum_safe_worsening_px: float = 0.1,
) -> list[dict[str, object]]:
    source_rows = _read_csv(Path(source_rows_csv))
    diagnostic_rows = _read_csv(Path(diagnostic_rows_csv))
    merged = _merge_source_and_diagnostics(source_rows, diagnostic_rows)
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in merged:
        policy_row = int(str(row.get("policy_row_index", "-1")))
        if policy_row < 0:
            raise ValueError("selected-policy measurement row is missing policy_row_index")
        grouped.setdefault(policy_row, []).append(row)
    coarse_pose_context = _coarse_pose_context_by_policy_row(
        coarse_pose_context_csv
    )
    support_selector_rows, _support_selector_manifest = _measurement_support_selector_rows(
        measurement_support_selector_rows_csv
    )

    examples: list[dict[str, object]] = []
    for policy_row, rows in sorted(grouped.items()):
        first = rows[0]
        pose_context = coarse_pose_context.get(policy_row)
        if coarse_pose_context_csv is not None and pose_context is None:
            raise ValueError(
                f"coarse-pose context is missing policy row {policy_row}"
            )
        if pose_context is not None:
            if str(pose_context.get("query_id", "")) != str(
                first.get("query_id", "")
            ):
                raise ValueError("coarse-pose context query identity mismatch")
            if int(float(str(pose_context.get("track_id", "-1")))) != int(
                float(str(first.get("track_id", "-1")))
            ):
                raise ValueError("coarse-pose context track identity mismatch")
        if measurement_support_selector_rows_csv is None:
            probabilities = np.asarray(
                [max(_float(row, "support_view_probability", 0.0), 0.0) for row in rows],
                dtype=np.float64,
            )
        else:
            selected_rows = []
            for row in rows:
                diagnostic_row_index = int(_float(row, "row_index", -1.0))
                selector_row = support_selector_rows.get(diagnostic_row_index)
                if selector_row is None:
                    raise ValueError(f"measurement support selector is missing diagnostic row {diagnostic_row_index}")
                if str(selector_row.get("query_id", "")) != str(row.get("query_id", "")):
                    raise ValueError("measurement support selector query identity mismatch")
                if int(float(str(selector_row.get("policy_row_index", "-1")))) != int(policy_row):
                    raise ValueError("measurement support selector policy identity mismatch")
                if int(float(str(selector_row.get("track_id", "-1")))) != int(float(str(row.get("track_id", "-1")))):
                    raise ValueError("measurement support selector track identity mismatch")
                selected_rows.append(float(selector_row["selector_probability"]))
            probabilities = np.asarray(selected_rows, dtype=np.float64)
        if float(np.sum(probabilities)) <= 0.0:
            probabilities = np.ones_like(probabilities)
        probabilities /= float(np.sum(probabilities))
        predicted_offsets = np.asarray(
            [[_float(row, "pred_dx"), _float(row, "pred_dy")] for row in rows],
            dtype=np.float64,
        )
        fused_offset = np.sum(predicted_offsets * probabilities[:, None], axis=0)
        mode_offsets = np.asarray(
            [[_float(row, "peak_dx"), _float(row, "peak_dy")] for row in rows],
            dtype=np.float64,
        )
        fused_mode_offset = np.sum(mode_offsets * probabilities[:, None], axis=0)
        disagreement = float(
            np.sum(
                np.linalg.norm(predicted_offsets - fused_offset[None, :], axis=1)
                * probabilities
            )
        )
        probability_entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
        )
        normalized_probability_entropy = (
            0.0
            if len(probabilities) <= 1
            else probability_entropy / math.log(float(len(probabilities)))
        )
        coarse_pose_success = bool(
            pose_context is not None
            and _bool(pose_context, "coarse_pose_success")
            and _bool(pose_context, "coarse_pose_projection_in_front")
        )
        coarse_pose_residual = (
            _optional_finite_float(
                pose_context or {}, "coarse_pose_reprojection_residual_px"
            )
            if coarse_pose_success
            else None
        )
        coarse_pose_offset = np.asarray(
            [
                _float(pose_context or {}, "coarse_pose_offset_dx"),
                _float(pose_context or {}, "coarse_pose_offset_dy"),
            ],
            dtype=np.float64,
        )
        fused_offset_norm = float(np.linalg.norm(fused_offset))
        coarse_pose_offset_norm = (
            0.0
            if coarse_pose_residual is None
            else float(np.linalg.norm(coarse_pose_offset))
        )
        if fused_offset_norm > 1e-8 and coarse_pose_offset_norm > 1e-8:
            offset_cosine = float(
                np.dot(fused_offset, coarse_pose_offset)
                / (fused_offset_norm * coarse_pose_offset_norm)
            )
            offset_scale_compatibility = float(
                min(fused_offset_norm, coarse_pose_offset_norm)
                / max(fused_offset_norm, coarse_pose_offset_norm)
            )
        else:
            offset_cosine = 0.0
            offset_scale_compatibility = 0.0
        offset_agreement_quality = (
            0.0
            if coarse_pose_residual is None
            else 1.0
            / (1.0 + float(np.linalg.norm(fused_offset - coarse_pose_offset)))
        )
        center = np.asarray(
            [_float(first, "center_x"), _float(first, "center_y")],
            dtype=np.float64,
        )
        target_x = _optional_finite_float(first, "target_gt_projected_x")
        target_y = _optional_finite_float(first, "target_gt_projected_y")
        projected_residual = _optional_finite_float(
            first, "target_gt_projected_residual_px"
        )
        projection_valid = bool(
            target_x is not None
            and target_y is not None
            and projected_residual is not None
            and _bool(first, "target_gt_projection_in_front")
            and _bool(first, "target_gt_projection_in_image")
        )
        target = np.asarray(
            [
                center[0] if target_x is None else target_x,
                center[1] if target_y is None else target_y,
            ],
            dtype=np.float64,
        )
        baseline_residual = (
            float(projected_residual) if projection_valid else float("inf")
        )
        updated_residual = (
            float(np.linalg.norm(center + fused_offset - target))
            if projection_valid
            else float("inf")
        )
        mode_updated_residual = (
            float(np.linalg.norm(center + fused_mode_offset - target))
            if projection_valid
            else float("inf")
        )
        track_length = max(_float(first, "track_length", 1.0), 1.0)
        query_reprojection_error = max(
            _float(first, "query_reprojection_error", 4.0), 0.0
        )
        support_reprojection_error = _weighted(
            [max(_float(row, "support_reprojection_error", 4.0), 0.0) for row in rows],
            probabilities,
        )
        support_view_angle = _weighted(
            [_float(row, "support_view_angle_deg", 90.0) for row in rows],
            probabilities,
        )
        measurement_dustbin = _weighted(
            [_float(row, "dustbin_probability", 0.5) for row in rows], probabilities
        )
        likelihood_entropy = _weighted(
            [_float(row, "likelihood_normalized_entropy", 1.0) for row in rows],
            probabilities,
        )
        covariance_sigma = _weighted(
            [
                _float(row, "likelihood_covariance_max_sigma_px", 4.0)
                for row in rows
            ],
            probabilities,
        )
        feature_values = {
            "assignment_score": _float(first, "assignment_score"),
            "pose_selection_score": _float(first, "pose_selection_score"),
            "geometry_p01": _float(first, "geometry_p01"),
            "geometry_p02": _float(first, "geometry_p02"),
            "geometry_p05": _float(first, "geometry_p05"),
            "log_track_length": math.log1p(track_length),
            "query_reprojection_quality": 1.0 / (1.0 + query_reprojection_error),
            "support_reprojection_quality": 1.0
            / (1.0 + support_reprojection_error),
            "support_view_top_probability": float(np.max(probabilities)),
            "support_view_normalized_entropy": normalized_probability_entropy,
            "support_view_angle_quality": 1.0 / (1.0 + support_view_angle / 30.0),
            "measurement_accept_probability": 1.0 - measurement_dustbin,
            "measurement_update_gate_probability": _weighted(
                [_float(row, "measurement_gate_probability", 0.0) for row in rows],
                probabilities,
            ),
            "likelihood_confidence": 1.0 - likelihood_entropy,
            "likelihood_peak_probability": _weighted(
                [_float(row, "likelihood_peak_probability") for row in rows],
                probabilities,
            ),
            "likelihood_peak_margin": _weighted(
                [_float(row, "likelihood_peak_margin") for row in rows],
                probabilities,
            ),
            "likelihood_covariance_quality": 1.0 / (1.0 + covariance_sigma),
            "fused_offset_norm": fused_offset_norm,
            "view_offset_disagreement": disagreement,
            "mean_mode_offset_disagreement": float(
                np.linalg.norm(fused_offset - fused_mode_offset)
            ),
            "coarse_pose_success": float(coarse_pose_success),
            "coarse_pose_inlier": float(
                pose_context is not None
                and _bool(pose_context, "coarse_pose_inlier")
            ),
            "coarse_pose_query_inlier_ratio": _float(
                pose_context or {}, "coarse_pose_query_inlier_ratio"
            ),
            "coarse_pose_reprojection_quality": (
                0.0
                if coarse_pose_residual is None
                else 1.0 / (1.0 + float(coarse_pose_residual))
            ),
            "coarse_pose_reprojection_log": (
                0.0
                if coarse_pose_residual is None
                else min(
                    1.0,
                    math.log1p(float(coarse_pose_residual))
                    / math.log(101.0),
                )
            ),
            "coarse_pose_center_le_1px": float(
                coarse_pose_residual is not None
                and coarse_pose_residual <= 1.0
            ),
            "coarse_pose_center_1_to_5px": float(
                coarse_pose_residual is not None
                and 1.0 < coarse_pose_residual <= 5.0
            ),
            "coarse_pose_center_gt_5px": float(
                coarse_pose_residual is not None
                and coarse_pose_residual > 5.0
            ),
            "measurement_pose_offset_cosine": offset_cosine,
            "measurement_pose_offset_agreement_quality": offset_agreement_quality,
            "measurement_pose_offset_scale_compatibility": offset_scale_compatibility,
            "switched_from_baseline": float(_bool(first, "switched_from_baseline")),
        }
        examples.append(
            {
                "policy_row_index": int(policy_row),
                "query_id": str(first.get("query_id", "")),
                "track_id": int(float(str(first.get("track_id", "-1")))),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "updated_x": float(center[0] + fused_offset[0]),
                "updated_y": float(center[1] + fused_offset[1]),
                "mode_updated_x": float(center[0] + fused_mode_offset[0]),
                "mode_updated_y": float(center[1] + fused_mode_offset[1]),
                "baseline_residual_px": float(baseline_residual),
                "updated_residual_px": float(updated_residual),
                "mode_updated_residual_px": float(
                    mode_updated_residual
                ),
                "has_valid_geometry_target": bool(projection_valid),
                "target_geometry_correct_5px": bool(
                    projection_valid and baseline_residual <= 5.0
                ),
                "target_center_correct_1px": bool(
                    projection_valid and baseline_residual <= 1.0
                ),
                "target_update_beneficial": bool(
                    projection_valid
                    and updated_residual
                    <= baseline_residual - float(minimum_update_gain_px)
                ),
                "target_update_safe": bool(
                    projection_valid
                    and updated_residual
                    <= baseline_residual + float(maximum_safe_worsening_px)
                ),
                "target_mode_update_beneficial": bool(
                    projection_valid
                    and mode_updated_residual
                    <= baseline_residual - float(minimum_update_gain_px)
                ),
                "target_mode_update_safe": bool(
                    projection_valid
                    and mode_updated_residual
                    <= baseline_residual + float(maximum_safe_worsening_px)
                ),
                "support_view_count": int(len(rows)),
                "coarse_pose_success": bool(coarse_pose_success),
                "coarse_pose_inlier": bool(
                    pose_context is not None
                    and _bool(pose_context, "coarse_pose_inlier")
                ),
                "coarse_pose_reprojection_residual_px": (
                    "" if coarse_pose_residual is None else coarse_pose_residual
                ),
                "features": feature_values,
            }
        )
    return examples


def _feature_matrix(
    examples: Sequence[Mapping[str, object]],
    feature_names: Sequence[str] = ACTION_FEATURE_NAMES,
) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"][name]) for name in feature_names]
            for example in examples
        ],
        dtype=np.float64,
    )


def _fit_binary_model(features: np.ndarray, labels: np.ndarray, *, c_value: float) -> BinaryLinearProbabilityModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-6] = 1.0
    if len(np.unique(y)) < 2:
        probability = float(np.mean(y)) if len(y) else 0.5
        return BinaryLinearProbabilityModel(
            mean=tuple(mean.tolist()),
            scale=tuple(scale.tolist()),
            weights=tuple(np.zeros((x.shape[1],), dtype=np.float64).tolist()),
            bias=0.0,
            constant_probability=probability,
        )
    model = LogisticRegression(
        C=float(c_value),
        class_weight=None,
        max_iter=4000,
        solver="lbfgs",
        random_state=0,
    )
    model.fit((x - mean[None, :]) / scale[None, :], y)
    return BinaryLinearProbabilityModel(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in model.coef_[0]),
        bias=float(model.intercept_[0]),
    )


def _residual_policy_metrics(
    examples: Sequence[Mapping[str, object]],
    final_residual: np.ndarray,
    mean_update: np.ndarray,
    mode_update: np.ndarray,
) -> dict[str, object]:
    baseline = np.asarray(
        [float(example["baseline_residual_px"]) for example in examples],
        dtype=np.float64,
    )
    final = np.asarray(final_residual, dtype=np.float64).reshape(-1)
    mean_mask = np.asarray(mean_update, dtype=bool).reshape(-1)
    mode_mask = np.asarray(mode_update, dtype=bool).reshape(-1)
    mask = mean_mask | mode_mask
    finite = np.isfinite(baseline)

    def metrics(rows: np.ndarray) -> dict[str, object]:
        before = baseline[rows]
        after = final[rows]
        updated_rows = mask[rows]
        if not np.any(rows):
            return {
                "count": 0,
                "update_count": 0,
                "median_improvement_fraction": None,
                "improve_ratio": None,
                "worsen_ratio": None,
            }
        before_median = float(np.median(before))
        return {
            "count": int(np.sum(rows)),
            "update_count": int(np.sum(updated_rows)),
            "mean_update_count": int(np.sum(mean_mask[rows])),
            "mode_update_count": int(np.sum(mode_mask[rows])),
            "update_fraction": float(np.mean(updated_rows)),
            "baseline_median_px": before_median,
            "final_median_px": float(np.median(after)),
            "median_improvement_fraction": float(
                (before_median - float(np.median(after)))
                / max(before_median, 1e-12)
            ),
            "improve_ratio": float(np.mean(after < before - 1e-9)),
            "worsen_ratio": float(np.mean(after > before + 1e-9)),
        }

    return {
        "valid_geometry_target_count": int(np.sum(finite)),
        "invalid_geometry_target_count": int(np.sum(~finite)),
        "all": metrics(finite),
        "center_le_1px": metrics(finite & (baseline <= 1.0)),
        "center_1_to_5px": metrics(
            finite & (baseline > 1.0) & (baseline <= 5.0)
        ),
        "center_gt_5px": metrics(finite & (baseline > 5.0)),
    }


def _choose_update_policy(
    examples: Sequence[Mapping[str, object]],
    geometry_probability: np.ndarray,
    center_probability: np.ndarray,
    mean_update_probability: np.ndarray,
    mean_safe_probability: np.ndarray,
    mode_update_probability: np.ndarray,
    mode_safe_probability: np.ndarray,
    *,
    maximum_low_residual_worsen_ratio: float,
    minimum_mid_residual_improvement_fraction: float,
    minimum_mid_residual_improve_ratio: float,
) -> dict[str, object]:
    mean_score = np.sqrt(
        np.clip(mean_update_probability, 0.0, 1.0)
        * np.clip(mean_safe_probability, 0.0, 1.0)
    )
    mode_score = np.sqrt(
        np.clip(mode_update_probability, 0.0, 1.0)
        * np.clip(mode_safe_probability, 0.0, 1.0)
    )
    baseline = np.asarray(
        [float(example["baseline_residual_px"]) for example in examples],
        dtype=np.float64,
    )
    mean_residual = np.asarray(
        [float(example["updated_residual_px"]) for example in examples],
        dtype=np.float64,
    )
    mode_residual = np.asarray(
        [float(example["mode_updated_residual_px"]) for example in examples],
        dtype=np.float64,
    )
    candidates: list[dict[str, object]] = []
    for geometry_threshold in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        for center_threshold in np.linspace(0.05, 0.95, 19):
            for mean_update_threshold in np.linspace(0.50, 0.95, 10):
                for mode_update_threshold in np.linspace(0.50, 0.95, 10):
                    eligible = geometry_probability >= float(geometry_threshold)
                    mode_update = (
                        eligible
                        & (center_probability < float(center_threshold))
                        & (mode_score >= float(mode_update_threshold))
                    )
                    mean_update = (
                        eligible
                        & (~mode_update)
                        & (mean_score >= float(mean_update_threshold))
                    )
                    final_residual = np.where(
                        mode_update,
                        mode_residual,
                        np.where(mean_update, mean_residual, baseline),
                    )
                    metrics = _residual_policy_metrics(
                        examples, final_residual, mean_update, mode_update
                    )
                    low = metrics["center_le_1px"]
                    mid = metrics["center_1_to_5px"]
                    passes = bool(
                        low["worsen_ratio"] is not None
                        and low["worsen_ratio"]
                        <= float(maximum_low_residual_worsen_ratio)
                        and mid["median_improvement_fraction"] is not None
                        and mid["median_improvement_fraction"]
                        >= float(minimum_mid_residual_improvement_fraction)
                        and mid["improve_ratio"] is not None
                        and mid["improve_ratio"]
                        >= float(minimum_mid_residual_improve_ratio)
                    )
                    candidates.append(
                        {
                            "geometry_threshold": float(geometry_threshold),
                            "center_threshold": float(center_threshold),
                            "mean_update_threshold": float(
                                mean_update_threshold
                            ),
                            "mode_update_threshold": float(
                                mode_update_threshold
                            ),
                            "passes": passes,
                            "metrics": metrics,
                            "mean_update_mask": mean_update,
                            "mode_update_mask": mode_update,
                        }
                    )
    feasible = [candidate for candidate in candidates if bool(candidate["passes"])]
    if feasible:
        chosen = max(
            feasible,
            key=lambda candidate: (
                float(candidate["metrics"]["center_1_to_5px"]["median_improvement_fraction"]),
                float(candidate["metrics"]["center_1_to_5px"]["improve_ratio"]),
                int(candidate["metrics"]["all"]["update_count"]),
            ),
        )
        promotion = True
    else:
        chosen = {
            "geometry_threshold": 1.0,
            "center_threshold": 0.0,
            "mean_update_threshold": 1.0,
            "mode_update_threshold": 1.0,
            "passes": False,
            "metrics": _residual_policy_metrics(
                examples,
                baseline,
                np.zeros((len(examples),), dtype=bool),
                np.zeros((len(examples),), dtype=bool),
            ),
            "mean_update_mask": np.zeros((len(examples),), dtype=bool),
            "mode_update_mask": np.zeros((len(examples),), dtype=bool),
        }
        promotion = False
    return {
        "promotion_passes": bool(promotion),
        "geometry_threshold": float(chosen["geometry_threshold"]),
        "center_threshold": float(chosen["center_threshold"]),
        "mean_update_threshold": float(chosen["mean_update_threshold"]),
        "mode_update_threshold": float(chosen["mode_update_threshold"]),
        "metrics": chosen["metrics"],
        "mean_update_mask": chosen["mean_update_mask"],
        "mode_update_mask": chosen["mode_update_mask"],
        "candidate_count": int(len(candidates)),
        "feasible_candidate_count": int(len(feasible)),
    }


def _choose_drop_policy(
    examples: Sequence[Mapping[str, object]],
    geometry_probability: np.ndarray,
    *,
    minimum_drop_precision: float,
) -> dict[str, object]:
    labels_wrong = np.asarray(
        [not bool(example["target_geometry_correct_5px"]) for example in examples],
        dtype=bool,
    )
    choices: list[tuple[int, float, float, float]] = []
    for threshold in np.linspace(0.01, 0.50, 50):
        drop = geometry_probability <= float(threshold)
        if not np.any(drop):
            continue
        precision = float(np.mean(labels_wrong[drop]))
        recall = float(np.sum(labels_wrong & drop) / max(np.sum(labels_wrong), 1))
        if precision >= float(minimum_drop_precision):
            choices.append((int(np.sum(drop)), float(threshold), precision, recall))
    if not choices:
        return {
            "promotion_passes": False,
            "threshold": 0.0,
            "drop_count": 0,
            "precision": None,
            "recall": 0.0,
        }
    count, threshold, precision, recall = max(choices)
    return {
        "promotion_passes": True,
        "threshold": threshold,
        "drop_count": count,
        "precision": precision,
        "recall": recall,
    }


def _write_prediction_rows(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    geometry_probability: np.ndarray,
    center_probability: np.ndarray,
    mean_update_probability: np.ndarray,
    mean_safe_probability: np.ndarray,
    mode_update_probability: np.ndarray,
    mode_safe_probability: np.ndarray,
    mean_update_mask: np.ndarray,
    mode_update_mask: np.ndarray,
    drop_threshold: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy_row_index",
        "query_id",
        "track_id",
        "center_x",
        "center_y",
        "updated_x",
        "updated_y",
        "mode_updated_x",
        "mode_updated_y",
        "geometry_probability",
        "center_probability",
        "mean_update_probability",
        "mean_safe_probability",
        "mode_update_probability",
        "mode_safe_probability",
        "coarse_pose_success",
        "coarse_pose_inlier",
        "coarse_pose_reprojection_residual_px",
        "action",
        "baseline_residual_px_TARGET_ONLY",
        "updated_residual_px_TARGET_ONLY",
        "mode_updated_residual_px_TARGET_ONLY",
        "target_geometry_correct_5px_TARGET_ONLY",
        "target_center_correct_1px_TARGET_ONLY",
        "target_update_beneficial_TARGET_ONLY",
        "target_update_safe_TARGET_ONLY",
        "target_mode_update_beneficial_TARGET_ONLY",
        "target_mode_update_safe_TARGET_ONLY",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, example in enumerate(examples):
            action = (
                "DROP"
                if float(geometry_probability[index]) <= float(drop_threshold)
                else "UPDATE_MODE"
                if bool(mode_update_mask[index])
                else "UPDATE_MEAN"
                if bool(mean_update_mask[index])
                else "KEEP"
            )
            writer.writerow(
                {
                    **{key: example[key] for key in fields[:9]},
                    "geometry_probability": float(geometry_probability[index]),
                    "center_probability": float(center_probability[index]),
                    "mean_update_probability": float(
                        mean_update_probability[index]
                    ),
                    "mean_safe_probability": float(mean_safe_probability[index]),
                    "mode_update_probability": float(
                        mode_update_probability[index]
                    ),
                    "mode_safe_probability": float(mode_safe_probability[index]),
                    "coarse_pose_success": example["coarse_pose_success"],
                    "coarse_pose_inlier": example["coarse_pose_inlier"],
                    "coarse_pose_reprojection_residual_px": example[
                        "coarse_pose_reprojection_residual_px"
                    ],
                    "action": action,
                    "baseline_residual_px_TARGET_ONLY": example[
                        "baseline_residual_px"
                    ],
                    "updated_residual_px_TARGET_ONLY": example[
                        "updated_residual_px"
                    ],
                    "mode_updated_residual_px_TARGET_ONLY": example[
                        "mode_updated_residual_px"
                    ],
                    "target_geometry_correct_5px_TARGET_ONLY": example[
                        "target_geometry_correct_5px"
                    ],
                    "target_center_correct_1px_TARGET_ONLY": example[
                        "target_center_correct_1px"
                    ],
                    "target_update_beneficial_TARGET_ONLY": example[
                        "target_update_beneficial"
                    ],
                    "target_update_safe_TARGET_ONLY": example[
                        "target_update_safe"
                    ],
                    "target_mode_update_beneficial_TARGET_ONLY": example[
                        "target_mode_update_beneficial"
                    ],
                    "target_mode_update_safe_TARGET_ONLY": example[
                        "target_mode_update_safe"
                    ],
                }
            )


def fit_measurement_action_calibrator(
    *,
    train_rows_csv: Path,
    train_diagnostics_csv: Path,
    validation_rows_csv: Path,
    validation_diagnostics_csv: Path,
    train_coarse_pose_context_csv: Path | None = None,
    validation_coarse_pose_context_csv: Path | None = None,
    train_measurement_support_selector_rows_csv: Path | None = None,
    validation_measurement_support_selector_rows_csv: Path | None = None,
    output_dir: Path,
    minimum_update_gain_px: float = 0.1,
    maximum_safe_worsening_px: float = 0.1,
    c_value: float = 0.25,
    maximum_low_residual_worsen_ratio: float = 0.10,
    minimum_mid_residual_improvement_fraction: float = 0.30,
    minimum_mid_residual_improve_ratio: float = 0.60,
    minimum_drop_precision: float = 0.90,
    require_diagnostic_manifests: bool = False,
) -> dict[str, Any]:
    train_diagnostic_manifest = _load_diagnostic_manifest(
        Path(train_diagnostics_csv),
        expected_rows_csv=Path(train_rows_csv),
        required=bool(require_diagnostic_manifests),
    )
    validation_diagnostic_manifest = _load_diagnostic_manifest(
        Path(validation_diagnostics_csv),
        expected_rows_csv=Path(validation_rows_csv),
        required=bool(require_diagnostic_manifests),
    )
    diagnostic_checkpoint_hashes = {
        str(manifest.get("checkpoint_sha256", ""))
        for manifest in (
            train_diagnostic_manifest,
            validation_diagnostic_manifest,
        )
        if manifest is not None and str(manifest.get("checkpoint_sha256", ""))
    }
    if len(diagnostic_checkpoint_hashes) > 1:
        raise ValueError(
            "train and validation diagnostics use different measurement checkpoints"
        )
    measurement_checkpoint_sha256 = (
        ""
        if not diagnostic_checkpoint_hashes
        else next(iter(diagnostic_checkpoint_hashes))
    )
    context_hashes = {
        file_sha256_short(Path(path))
        for path in (
            train_coarse_pose_context_csv,
            validation_coarse_pose_context_csv,
        )
        if path is not None
    }
    if len(context_hashes) > 1:
        raise ValueError(
            "train and validation coarse-pose context artifacts differ"
        )
    coarse_pose_context_sha256 = (
        "" if not context_hashes else next(iter(context_hashes))
    )
    if (train_measurement_support_selector_rows_csv is None) != (
        validation_measurement_support_selector_rows_csv is None
    ):
        raise ValueError("train and validation measurement support selector rows must be supplied together")
    train_support_selector_manifest = None
    validation_support_selector_manifest = None
    measurement_support_selector_sha256 = ""
    if train_measurement_support_selector_rows_csv is not None:
        _, train_support_selector_manifest = _measurement_support_selector_rows(
            Path(train_measurement_support_selector_rows_csv)
        )
        _, validation_support_selector_manifest = _measurement_support_selector_rows(
            Path(validation_measurement_support_selector_rows_csv)
        )
        selector_hashes = {
            str(train_support_selector_manifest["selector_sha256"]),
            str(validation_support_selector_manifest["selector_sha256"]),
        }
        if len(selector_hashes) != 1:
            raise ValueError("train and validation use different measurement support selectors")
        measurement_support_selector_sha256 = next(iter(selector_hashes))
    train = build_action_examples(
        source_rows_csv=Path(train_rows_csv),
        diagnostic_rows_csv=Path(train_diagnostics_csv),
        coarse_pose_context_csv=(
            None
            if train_coarse_pose_context_csv is None
            else Path(train_coarse_pose_context_csv)
        ),
        measurement_support_selector_rows_csv=(
            None
            if train_measurement_support_selector_rows_csv is None
            else Path(train_measurement_support_selector_rows_csv)
        ),
        minimum_update_gain_px=float(minimum_update_gain_px),
        maximum_safe_worsening_px=float(maximum_safe_worsening_px),
    )
    validation = build_action_examples(
        source_rows_csv=Path(validation_rows_csv),
        diagnostic_rows_csv=Path(validation_diagnostics_csv),
        coarse_pose_context_csv=(
            None
            if validation_coarse_pose_context_csv is None
            else Path(validation_coarse_pose_context_csv)
        ),
        measurement_support_selector_rows_csv=(
            None
            if validation_measurement_support_selector_rows_csv is None
            else Path(validation_measurement_support_selector_rows_csv)
        ),
        minimum_update_gain_px=float(minimum_update_gain_px),
        maximum_safe_worsening_px=float(maximum_safe_worsening_px),
    )
    if not train or not validation:
        raise ValueError("action calibration requires non-empty train and validation groups")
    train_x = _feature_matrix(train)
    validation_x = _feature_matrix(validation)
    targets = {
        "geometry": "target_geometry_correct_5px",
        "center": "target_center_correct_1px",
        "mean_update": "target_update_beneficial",
        "mean_safe": "target_update_safe",
        "mode_update": "target_mode_update_beneficial",
        "mode_safe": "target_mode_update_safe",
    }
    models: dict[str, BinaryLinearProbabilityModel] = {}
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, object] = {}
    for model_name, target_key in targets.items():
        train_y = np.asarray([bool(row[target_key]) for row in train], dtype=np.int64)
        validation_y = np.asarray(
            [bool(row[target_key]) for row in validation], dtype=np.int64
        )
        model = _fit_binary_model(train_x, train_y, c_value=float(c_value))
        models[model_name] = model
        train_probability = model.predict(train_x)
        validation_probability = model.predict(validation_x)
        probabilities[model_name] = {
            "train": train_probability,
            "validation": validation_probability,
        }
        reports[model_name] = {
            "target": target_key,
            "train": confidence_metrics(train_y, train_probability),
            "validation": confidence_metrics(
                validation_y, validation_probability
            ),
        }
    update_policy = _choose_update_policy(
        validation,
        probabilities["geometry"]["validation"],
        probabilities["center"]["validation"],
        probabilities["mean_update"]["validation"],
        probabilities["mean_safe"]["validation"],
        probabilities["mode_update"]["validation"],
        probabilities["mode_safe"]["validation"],
        maximum_low_residual_worsen_ratio=float(
            maximum_low_residual_worsen_ratio
        ),
        minimum_mid_residual_improvement_fraction=float(
            minimum_mid_residual_improvement_fraction
        ),
        minimum_mid_residual_improve_ratio=float(
            minimum_mid_residual_improve_ratio
        ),
    )
    drop_policy = _choose_drop_policy(
        validation,
        probabilities["geometry"]["validation"],
        minimum_drop_precision=float(minimum_drop_precision),
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "format": "selected_measurement_action_calibrator_v4",
        "feature_names": list(ACTION_FEATURE_NAMES),
        "models": {name: model.to_dict() for name, model in models.items()},
        "update_policy": {
            key: value
            for key, value in update_policy.items()
            if key not in {"mean_update_mask", "mode_update_mask"}
        },
        "drop_policy": drop_policy,
        "target_only_fields_are_forbidden_at_inference": True,
        "coarse_pose_context_required": bool(
            train_coarse_pose_context_csv is not None
            or validation_coarse_pose_context_csv is not None
        ),
        "measurement_checkpoint_sha256": measurement_checkpoint_sha256,
        "coarse_pose_context_sha256": coarse_pose_context_sha256,
        "measurement_support_selector_required": bool(
            train_measurement_support_selector_rows_csv is not None
        ),
        "measurement_support_selector_sha256": measurement_support_selector_sha256,
    }
    model_path = output / "measurement_action_calibrator.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    validation_mean_update = np.asarray(
        update_policy["mean_update_mask"], dtype=bool
    )
    validation_mode_update = np.asarray(
        update_policy["mode_update_mask"], dtype=bool
    )
    train_mean_score = np.sqrt(
        probabilities["mean_update"]["train"]
        * probabilities["mean_safe"]["train"]
    )
    train_mode_score = np.sqrt(
        probabilities["mode_update"]["train"]
        * probabilities["mode_safe"]["train"]
    )
    train_eligible = probabilities["geometry"]["train"] >= float(
        update_policy["geometry_threshold"]
    )
    train_mode_update = (
        train_eligible
        & (
            probabilities["center"]["train"]
            < float(update_policy["center_threshold"])
        )
        & (
            train_mode_score
            >= float(update_policy["mode_update_threshold"])
        )
    )
    train_mean_update = (
        train_eligible
        & (~train_mode_update)
        & (
            train_mean_score
            >= float(update_policy["mean_update_threshold"])
        )
    )
    _write_prediction_rows(
        output / "train_action_predictions.csv",
        train,
        probabilities["geometry"]["train"],
        probabilities["center"]["train"],
        probabilities["mean_update"]["train"],
        probabilities["mean_safe"]["train"],
        probabilities["mode_update"]["train"],
        probabilities["mode_safe"]["train"],
        train_mean_update,
        train_mode_update,
        float(drop_policy["threshold"]),
    )
    _write_prediction_rows(
        output / "validation_action_predictions.csv",
        validation,
        probabilities["geometry"]["validation"],
        probabilities["center"]["validation"],
        probabilities["mean_update"]["validation"],
        probabilities["mean_safe"]["validation"],
        probabilities["mode_update"]["validation"],
        probabilities["mode_safe"]["validation"],
        validation_mean_update,
        validation_mode_update,
        float(drop_policy["threshold"]),
    )
    summary = {
        "stage": "selected_measurement_action_calibration",
        "protocol": {
            "fit_split": "train",
            "threshold_selection_split": "validation",
            "test_used": False,
            "render": False,
            "image_retrieval": False,
            "submap": False,
            "grouping": (
                "one_row_per_policy_assignment_with_measurement_support_selector_fusion"
                if train_measurement_support_selector_rows_csv is not None
                else "one_row_per_policy_assignment_with_support_posterior_fusion"
            ),
            "coarse_pose_context": bool(
                train_coarse_pose_context_csv is not None
                or validation_coarse_pose_context_csv is not None
            ),
            "coarse_pose_context_uses_gt_pose": False,
            "diagnostic_manifests_required": bool(
                require_diagnostic_manifests
            ),
            "geometry_target": "GT_pose_projected_3D_track_residual_le_5px",
            "update_target": "GT_pose_residual_improvement_after_frozen_RGB_measurement",
        },
        "train_group_count": int(len(train)),
        "validation_group_count": int(len(validation)),
        "feature_names": list(ACTION_FEATURE_NAMES),
        "target_fields_excluded_from_features": True,
        "model_reports": reports,
        "update_policy": {
            key: value
            for key, value in update_policy.items()
            if key not in {"mean_update_mask", "mode_update_mask"}
        },
        "drop_policy": drop_policy,
        "promotion_passes": bool(
            update_policy["promotion_passes"]
            and drop_policy["promotion_passes"]
        ),
        "inputs": {
            "train_rows_csv": str(train_rows_csv),
            "train_rows_sha256": file_sha256_short(Path(train_rows_csv)),
            "train_diagnostics_csv": str(train_diagnostics_csv),
            "train_diagnostics_sha256": file_sha256_short(
                Path(train_diagnostics_csv)
            ),
            "validation_rows_csv": str(validation_rows_csv),
            "validation_rows_sha256": file_sha256_short(
                Path(validation_rows_csv)
            ),
            "validation_diagnostics_csv": str(validation_diagnostics_csv),
            "validation_diagnostics_sha256": file_sha256_short(
                Path(validation_diagnostics_csv)
            ),
            "train_coarse_pose_context_csv": (
                ""
                if train_coarse_pose_context_csv is None
                else str(train_coarse_pose_context_csv)
            ),
            "train_coarse_pose_context_sha256": (
                ""
                if train_coarse_pose_context_csv is None
                else file_sha256_short(Path(train_coarse_pose_context_csv))
            ),
            "validation_coarse_pose_context_csv": (
                ""
                if validation_coarse_pose_context_csv is None
                else str(validation_coarse_pose_context_csv)
            ),
            "validation_coarse_pose_context_sha256": (
                ""
                if validation_coarse_pose_context_csv is None
                else file_sha256_short(
                    Path(validation_coarse_pose_context_csv)
                )
            ),
            "train_diagnostic_manifest": (
                None
                if train_diagnostic_manifest is None
                else train_diagnostic_manifest
            ),
            "validation_diagnostic_manifest": (
                None
                if validation_diagnostic_manifest is None
                else validation_diagnostic_manifest
            ),
            "train_measurement_support_selector": train_support_selector_manifest,
            "validation_measurement_support_selector": validation_support_selector_manifest,
        },
        "outputs": {
            "model": str(model_path),
            "train_predictions": str(output / "train_action_predictions.csv"),
            "validation_predictions": str(
                output / "validation_action_predictions.csv"
            ),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def apply_measurement_action_calibrator(
    *,
    rows_csv: Path,
    diagnostics_csv: Path,
    calibrator_json: Path,
    output_dir: Path,
    coarse_pose_context_csv: Path | None = None,
    measurement_support_selector_rows_csv: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(calibrator_json).read_text())
    if payload.get("format") not in {
        "selected_measurement_action_calibrator_v3",
        "selected_measurement_action_calibrator_v4",
    }:
        raise ValueError("unsupported selected measurement action calibrator")
    feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
    if not feature_names or any(name not in ACTION_FEATURE_NAMES for name in feature_names):
        raise ValueError("measurement action feature contract mismatch")
    model_payloads = payload.get("models")
    if not isinstance(model_payloads, Mapping):
        raise ValueError("measurement action calibrator is missing models")
    required_models = {
        "geometry",
        "center",
        "mean_update",
        "mean_safe",
        "mode_update",
        "mode_safe",
    }
    if set(model_payloads) != required_models:
        raise ValueError("measurement action calibrator model set mismatch")
    models = {
        name: BinaryLinearProbabilityModel.from_dict(model_payloads[name])
        for name in required_models
    }
    context_required = bool(payload.get("coarse_pose_context_required", False))
    if context_required and coarse_pose_context_csv is None:
        raise ValueError("measurement action calibrator requires coarse-pose context")
    expected_context_hash = str(
        payload.get("coarse_pose_context_sha256", "")
    )
    if expected_context_hash and coarse_pose_context_csv is not None:
        actual_context_hash = file_sha256_short(Path(coarse_pose_context_csv))
        if actual_context_hash != expected_context_hash:
            raise ValueError("coarse-pose context does not match calibrator")
    expected_checkpoint_hash = str(
        payload.get("measurement_checkpoint_sha256", "")
    )
    diagnostic_manifest = _load_diagnostic_manifest(
        Path(diagnostics_csv),
        expected_rows_csv=Path(rows_csv),
        required=bool(expected_checkpoint_hash),
    )
    if expected_checkpoint_hash:
        actual_checkpoint_hash = str(
            (diagnostic_manifest or {}).get("checkpoint_sha256", "")
        )
        if actual_checkpoint_hash != expected_checkpoint_hash:
            raise ValueError(
                "measurement diagnostics use a different checkpoint than calibrator"
            )
    selector_required = bool(payload.get("measurement_support_selector_required", False))
    if selector_required and measurement_support_selector_rows_csv is None:
        raise ValueError("measurement action calibrator requires measurement support selector rows")
    support_selector_manifest = None
    if measurement_support_selector_rows_csv is not None:
        _, support_selector_manifest = _measurement_support_selector_rows(
            Path(measurement_support_selector_rows_csv)
        )
        expected_selector_hash = str(payload.get("measurement_support_selector_sha256", ""))
        if expected_selector_hash and str(support_selector_manifest["selector_sha256"]) != expected_selector_hash:
            raise ValueError("measurement support selector does not match action calibrator")
    examples = build_action_examples(
        source_rows_csv=Path(rows_csv),
        diagnostic_rows_csv=Path(diagnostics_csv),
        coarse_pose_context_csv=(
            None
            if coarse_pose_context_csv is None
            else Path(coarse_pose_context_csv)
        ),
        measurement_support_selector_rows_csv=(
            None
            if measurement_support_selector_rows_csv is None
            else Path(measurement_support_selector_rows_csv)
        ),
    )
    features = _feature_matrix(examples, feature_names=feature_names)
    probabilities = {
        name: model.predict(features) for name, model in models.items()
    }
    update_policy = payload.get("update_policy")
    drop_policy = payload.get("drop_policy")
    if not isinstance(update_policy, Mapping) or not isinstance(
        drop_policy, Mapping
    ):
        raise ValueError("measurement action calibrator is missing policies")
    mean_score = np.sqrt(
        probabilities["mean_update"] * probabilities["mean_safe"]
    )
    mode_score = np.sqrt(
        probabilities["mode_update"] * probabilities["mode_safe"]
    )
    if bool(update_policy.get("promotion_passes", False)):
        eligible = probabilities["geometry"] >= float(
            update_policy["geometry_threshold"]
        )
        mode_update = (
            eligible
            & (
                probabilities["center"]
                < float(update_policy["center_threshold"])
            )
            & (mode_score >= float(update_policy["mode_update_threshold"]))
        )
        mean_update = (
            eligible
            & (~mode_update)
            & (mean_score >= float(update_policy["mean_update_threshold"]))
        )
    else:
        mean_update = np.zeros((len(examples),), dtype=bool)
        mode_update = np.zeros((len(examples),), dtype=bool)
    drop_threshold = (
        float(drop_policy["threshold"])
        if bool(drop_policy.get("promotion_passes", False))
        else -1.0
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "action_predictions.csv"
    _write_prediction_rows(
        predictions_path,
        examples,
        probabilities["geometry"],
        probabilities["center"],
        probabilities["mean_update"],
        probabilities["mean_safe"],
        probabilities["mode_update"],
        probabilities["mode_safe"],
        mean_update,
        mode_update,
        drop_threshold,
    )
    action_counts = {
        "KEEP": int(
            np.sum(
                (~mean_update)
                & (~mode_update)
                & (probabilities["geometry"] > drop_threshold)
            )
        ),
        "UPDATE_MEAN": int(np.sum(mean_update)),
        "UPDATE_MODE": int(np.sum(mode_update)),
        "DROP": int(np.sum(probabilities["geometry"] <= drop_threshold)),
    }
    summary = {
        "stage": "selected_measurement_action_calibrator_apply",
        "protocol": {
            "thresholds_frozen": True,
            "threshold_search": False,
            "target_fields_used_as_features": False,
            "feature_names": list(feature_names),
            "drop_disabled_when_calibration_gate_fails": True,
            "coarse_pose_context_required": context_required,
            "coarse_pose_context_supplied": coarse_pose_context_csv is not None,
            "measurement_checkpoint_sha256": expected_checkpoint_hash,
            "measurement_support_selector_required": selector_required,
            "measurement_support_selector_supplied": measurement_support_selector_rows_csv is not None,
        },
        "group_count": int(len(examples)),
        "action_counts": action_counts,
        "inputs": {
            "rows_csv": str(rows_csv),
            "rows_sha256": file_sha256_short(Path(rows_csv)),
            "diagnostics_csv": str(diagnostics_csv),
            "diagnostics_sha256": file_sha256_short(Path(diagnostics_csv)),
            "calibrator_json": str(calibrator_json),
            "calibrator_sha256": file_sha256_short(Path(calibrator_json)),
            "coarse_pose_context_csv": (
                ""
                if coarse_pose_context_csv is None
                else str(coarse_pose_context_csv)
            ),
            "coarse_pose_context_sha256": (
                ""
                if coarse_pose_context_csv is None
                else file_sha256_short(Path(coarse_pose_context_csv))
            ),
            "diagnostic_manifest": diagnostic_manifest,
            "measurement_support_selector": support_selector_manifest,
        },
        "outputs": {
            "predictions": str(predictions_path),
            "predictions_sha256": file_sha256_short(predictions_path),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
