"""Load frozen pose-free measurement probabilities for PnP verification."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


AssignmentKey = Tuple[str, int, int]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _source_assignment_keys(path: Path) -> dict[int, AssignmentKey]:
    grouped: dict[int, AssignmentKey] = {}
    for row in _read_csv(Path(path)):
        policy_row = int(row.get("policy_row_index", "-1"))
        key = (
            str(row.get("query_id", "")),
            int(row.get("source_query_row", "-1")),
            int(row.get("track_id", "-1")),
        )
        if policy_row < 0 or key[1] < 0 or key[2] < 0 or not key[0]:
            raise ValueError("measurement source row has an invalid assignment identity")
        previous = grouped.setdefault(policy_row, key)
        if previous != key:
            raise ValueError(
                f"measurement support views disagree for policy row {policy_row}"
            )
    if not grouped:
        raise ValueError("measurement source rows are empty")
    return grouped


def _load_prediction_split(
    *,
    source_rows_csv: Path,
    expected_source_hash: str,
    predictions_csv: Path,
    expected_predictions_hash: str,
    probability_column: str,
) -> dict[AssignmentKey, float]:
    if file_sha256_short(Path(source_rows_csv)) != str(expected_source_hash):
        raise ValueError("measurement source rows hash mismatch")
    if file_sha256_short(Path(predictions_csv)) != str(expected_predictions_hash):
        raise ValueError("measurement prediction rows hash mismatch")
    source_keys = _source_assignment_keys(Path(source_rows_csv))
    output: dict[AssignmentKey, float] = {}
    seen_policy_rows: set[int] = set()
    for row in _read_csv(Path(predictions_csv)):
        policy_row = int(row.get("policy_row_index", "-1"))
        if policy_row in seen_policy_rows:
            raise ValueError(f"duplicate measurement prediction row: {policy_row}")
        seen_policy_rows.add(policy_row)
        key = source_keys.get(policy_row)
        if key is None:
            raise ValueError(
                f"measurement prediction has no source assignment: {policy_row}"
            )
        if str(row.get("query_id", "")) != key[0] or int(
            row.get("track_id", "-1")
        ) != key[2]:
            raise ValueError("measurement prediction/source assignment mismatch")
        value_text = str(row.get(probability_column, "")).strip()
        if not value_text:
            raise ValueError(
                "measurement prediction is missing; rebuild with complete query-fold OOF"
            )
        probability = float(value_text)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("measurement geometry probability must be in [0, 1]")
        if key in output:
            raise ValueError(f"duplicate measurement assignment key: {key}")
        output[key] = probability
    if set(source_keys) != seen_policy_rows:
        missing = sorted(set(source_keys) - seen_policy_rows)
        raise ValueError(f"measurement predictions are incomplete: {missing[:5]}")
    return output


@dataclass(frozen=True)
class FrozenMeasurementPoseEvidence:
    feature_set: str
    verification_threshold: float
    probability_by_assignment: Mapping[AssignmentKey, float]
    manifest: Mapping[str, object]

    def candidate_probability_matrix(
        self,
        *,
        query_id: str,
        token_indices: Sequence[int],
        measured_track_ids: Sequence[int],
        candidate_track_ids: np.ndarray,
    ) -> np.ndarray:
        tokens = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        measured_tracks = np.asarray(measured_track_ids, dtype=np.int64).reshape(-1)
        candidates = np.asarray(candidate_track_ids, dtype=np.int64)
        if candidates.ndim != 2 or candidates.shape[0] != len(tokens):
            raise ValueError("candidate tracks must have shape (N, L)")
        if len(measured_tracks) != len(tokens):
            raise ValueError("measured tracks must have one value per query token")
        output = np.full(candidates.shape, np.nan, dtype=np.float64)
        for row, (token, track) in enumerate(zip(tokens, measured_tracks)):
            key = (str(query_id), int(token), int(track))
            if key not in self.probability_by_assignment:
                # Some frozen coarse assignments have no usable real-image
                # support observation. Missing evidence is unknown, not a
                # negative measurement target.
                continue
            columns = np.flatnonzero(candidates[row] == int(track))
            if len(columns) == 0:
                raise ValueError(
                    f"measured track is absent from its frozen top-L pool: {key}"
                )
            output[row, columns] = float(self.probability_by_assignment[key])
        return output


@dataclass(frozen=True)
class FrozenMeasurementUpdateEvidence:
    update_threshold: float
    update_by_assignment: Mapping[AssignmentKey, tuple[float, tuple[float, float]]]
    manifest: Mapping[str, object]

    def assignment_updates(
        self,
        *,
        query_id: str,
        token_indices: Sequence[int],
        measured_track_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = np.asarray(token_indices, dtype=np.int64).reshape(-1)
        tracks = np.asarray(measured_track_ids, dtype=np.int64).reshape(-1)
        if len(tokens) != len(tracks):
            raise ValueError("update evidence identity arrays have unequal length")
        probabilities = np.full((len(tokens),), np.nan, dtype=np.float64)
        refined_xy = np.full((len(tokens), 2), np.nan, dtype=np.float64)
        for row, (token, track) in enumerate(zip(tokens, tracks)):
            value = self.update_by_assignment.get(
                (str(query_id), int(token), int(track))
            )
            if value is None:
                continue
            probabilities[row] = float(value[0])
            refined_xy[row] = np.asarray(value[1], dtype=np.float64)
        return probabilities, refined_xy


def _load_update_prediction_split(
    *,
    source_rows_csv: Path,
    expected_source_hash: str,
    predictions_csv: Path,
    expected_predictions_hash: str,
) -> dict[AssignmentKey, tuple[float, tuple[float, float]]]:
    if file_sha256_short(Path(source_rows_csv)) != str(expected_source_hash):
        raise ValueError("measurement update source rows hash mismatch")
    if file_sha256_short(Path(predictions_csv)) != str(expected_predictions_hash):
        raise ValueError("measurement update prediction rows hash mismatch")
    source_keys = _source_assignment_keys(Path(source_rows_csv))
    output: dict[AssignmentKey, tuple[float, tuple[float, float]]] = {}
    seen: set[int] = set()
    for row in _read_csv(Path(predictions_csv)):
        policy_row = int(row.get("policy_row_index", "-1"))
        if policy_row in seen or policy_row not in source_keys:
            raise ValueError("duplicate or unknown measurement update policy row")
        seen.add(policy_row)
        key = source_keys[policy_row]
        if str(row.get("query_id", "")) != key[0] or int(
            row.get("track_id", "-1")
        ) != key[2]:
            raise ValueError("measurement update prediction/source mismatch")
        probability = float(row.get("update_beneficial_probability", "nan"))
        xy = (
            float(row.get("updated_x", "nan")),
            float(row.get("updated_y", "nan")),
        )
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("measurement update probability must be in [0, 1]")
        if not np.all(np.isfinite(np.asarray(xy))):
            raise ValueError("measurement updated xy is invalid")
        output[key] = (probability, xy)
    if set(source_keys) != seen:
        raise ValueError("measurement update predictions are incomplete")
    return output


def load_frozen_measurement_update_evidence(
    *,
    fit_summary_path: Path,
    late_apply_summary_path: Path | None = None,
) -> FrozenMeasurementUpdateEvidence:
    fit_path = Path(fit_summary_path)
    fit = json.loads(fit_path.read_text())
    if fit.get("stage") != "pose_free_measurement_update_verifier_fit":
        raise ValueError("unsupported measurement update fit summary")
    protocol = fit.get("protocol", {})
    if any(
        bool(protocol.get(key, True))
        for key in (
            "query_pose_used_as_feature",
            "coarse_pose_used_as_feature",
            "assignment_score_used_as_feature",
            "coordinate_update_applied",
        )
    ):
        raise ValueError("measurement update verifier is not pose-free")
    if not bool(fit.get("promotion_passes", False)):
        raise ValueError("measurement update verifier did not pass calibration gates")
    outputs = fit.get("outputs", {})
    model_path = Path(str(outputs.get("model", "")))
    model_hash = file_sha256_short(model_path)
    if model_hash != str(outputs.get("model_sha256", "")):
        raise ValueError("measurement update model hash mismatch")
    model = json.loads(model_path.read_text())
    threshold = float(model.get("update_threshold", float("nan")))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("measurement update threshold is invalid")
    values: dict[AssignmentKey, tuple[float, tuple[float, float]]] = {}
    split_manifest: dict[str, object] = {}
    for split_name, output_key in (
        ("train", "train_predictions"),
        ("validation", "validation_predictions"),
    ):
        source = fit.get("inputs", {}).get(split_name, {})
        prediction_path = Path(str(outputs.get(output_key, "")))
        split_values = _load_update_prediction_split(
            source_rows_csv=Path(str(source.get("source_rows_csv", ""))),
            expected_source_hash=str(source.get("source_rows_sha256", "")),
            predictions_csv=prediction_path,
            expected_predictions_hash=str(outputs.get(f"{output_key}_sha256", "")),
        )
        if set(values) & set(split_values):
            raise ValueError("measurement update development splits overlap")
        values.update(split_values)
        split_manifest[split_name] = {
            "predictions": str(prediction_path),
            "predictions_sha256": str(outputs.get(f"{output_key}_sha256", "")),
            "assignment_count": len(split_values),
        }
    late_manifest = None
    if late_apply_summary_path is not None:
        late_path = Path(late_apply_summary_path)
        late = json.loads(late_path.read_text())
        if late.get("stage") != "pose_free_measurement_update_verifier_apply":
            raise ValueError("unsupported late measurement update summary")
        if str(late.get("inputs", {}).get("verifier_sha256", "")) != model_hash:
            raise ValueError("late update predictions use a different verifier")
        late_protocol = late.get("protocol", {})
        if bool(late_protocol.get("threshold_search", True)) or any(
            bool(late_protocol.get(key, True))
            for key in (
                "target_fields_used_as_features",
                "query_pose_used_as_feature",
                "coordinate_update_applied",
            )
        ):
            raise ValueError("late update predictions are not frozen and pose-free")
        data = late.get("inputs", {}).get("data", {})
        late_outputs = late.get("outputs", {})
        prediction_path = Path(str(late_outputs.get("predictions", "")))
        split_values = _load_update_prediction_split(
            source_rows_csv=Path(str(data.get("source_rows_csv", ""))),
            expected_source_hash=str(data.get("source_rows_sha256", "")),
            predictions_csv=prediction_path,
            expected_predictions_hash=str(late_outputs.get("predictions_sha256", "")),
        )
        if set(values) & set(split_values):
            raise ValueError("measurement update late assignments overlap development")
        values.update(split_values)
        late_manifest = {
            "summary": str(late_path),
            "summary_sha256": file_sha256_short(late_path),
            "predictions": str(prediction_path),
            "predictions_sha256": str(late_outputs.get("predictions_sha256", "")),
            "assignment_count": len(split_values),
        }
    manifest = {
        "fit_summary": str(fit_path),
        "fit_summary_sha256": file_sha256_short(fit_path),
        "model": str(model_path),
        "model_sha256": model_hash,
        "update_threshold": threshold,
        "splits": split_manifest,
        "late": late_manifest,
        "assignment_count": len(values),
    }
    return FrozenMeasurementUpdateEvidence(threshold, values, manifest)


def load_frozen_measurement_pose_evidence(
    *,
    fit_summary_path: Path,
    late_apply_summary_path: Path | None = None,
    feature_set: str = "measurement_plus_support",
) -> FrozenMeasurementPoseEvidence:
    fit_path = Path(fit_summary_path)
    fit = json.loads(fit_path.read_text())
    if fit.get("stage") != "pose_free_measurement_geometry_verifier_fit":
        raise ValueError("unsupported measurement fit summary")
    protocol = fit.get("protocol", {})
    required_false = (
        "query_pose_used_as_feature",
        "coarse_pose_used_as_feature",
        "assignment_score_used_as_feature",
        "coordinate_update_enabled",
        "render",
    )
    if any(bool(protocol.get(key, True)) for key in required_false):
        raise ValueError("measurement fit summary is not pose-free inference evidence")
    if not bool(fit.get("promotion_gate", {}).get("passes", False)):
        raise ValueError("measurement verifier did not pass its independent signal gate")
    model_record = fit.get("outputs", {}).get("models", {}).get(feature_set)
    if not isinstance(model_record, dict):
        raise ValueError(f"measurement fit summary has no model: {feature_set}")
    model_path = Path(str(model_record.get("path", "")))
    model_hash = file_sha256_short(model_path)
    if model_hash != str(model_record.get("sha256", "")):
        raise ValueError("measurement verifier model hash mismatch")
    model = json.loads(model_path.read_text())
    if str(model.get("feature_set", "")) != str(feature_set):
        raise ValueError("measurement verifier feature set mismatch")
    threshold = float(model.get("verification_threshold", float("nan")))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("measurement verifier threshold is invalid")
    probability_column = f"{feature_set}_geometry_probability"

    probabilities: dict[AssignmentKey, float] = {}
    split_manifests: dict[str, object] = {}
    outputs = fit.get("outputs", {})
    for split_name, output_key in (
        ("train", "train_oof_predictions"),
        ("validation", "validation_predictions"),
    ):
        input_record = fit.get("inputs", {}).get(split_name, {})
        predictions_path = Path(str(outputs.get(output_key, "")))
        split_values = _load_prediction_split(
            source_rows_csv=Path(str(input_record.get("source_rows_csv", ""))),
            expected_source_hash=str(input_record.get("source_rows_sha256", "")),
            predictions_csv=predictions_path,
            expected_predictions_hash=str(outputs.get(f"{output_key}_sha256", "")),
            probability_column=probability_column,
        )
        overlap = set(probabilities) & set(split_values)
        if overlap:
            raise ValueError(f"measurement split assignment overlap: {next(iter(overlap))}")
        probabilities.update(split_values)
        split_manifests[split_name] = {
            "source_rows_csv": str(input_record.get("source_rows_csv", "")),
            "source_rows_sha256": str(input_record.get("source_rows_sha256", "")),
            "predictions_csv": str(predictions_path),
            "predictions_sha256": str(outputs.get(f"{output_key}_sha256", "")),
            "assignment_count": len(split_values),
        }

    late_manifest: dict[str, object] | None = None
    if late_apply_summary_path is not None:
        late_path = Path(late_apply_summary_path)
        late = json.loads(late_path.read_text())
        if late.get("stage") != "pose_free_measurement_geometry_verifier_apply":
            raise ValueError("unsupported late measurement apply summary")
        if str(late.get("feature_set", "")) != str(feature_set):
            raise ValueError("late measurement feature set mismatch")
        if str(late.get("inputs", {}).get("verifier_sha256", "")) != model_hash:
            raise ValueError("late measurement predictions use a different verifier")
        late_protocol = late.get("protocol", {})
        if bool(late_protocol.get("threshold_search", True)) or any(
            bool(late_protocol.get(key, True))
            for key in (
                "target_fields_used_as_features",
                "query_pose_used_as_feature",
                "coarse_pose_used_as_feature",
                "coordinate_update_enabled",
            )
        ):
            raise ValueError("late measurement application is not frozen and pose-free")
        data = late.get("inputs", {}).get("data", {})
        late_outputs = late.get("outputs", {})
        predictions_path = Path(str(late_outputs.get("predictions", "")))
        split_values = _load_prediction_split(
            source_rows_csv=Path(str(data.get("source_rows_csv", ""))),
            expected_source_hash=str(data.get("source_rows_sha256", "")),
            predictions_csv=predictions_path,
            expected_predictions_hash=str(
                late_outputs.get("predictions_sha256", "")
            ),
            probability_column=probability_column,
        )
        overlap = set(probabilities) & set(split_values)
        if overlap:
            raise ValueError("late measurement assignments overlap development splits")
        probabilities.update(split_values)
        late_manifest = {
            "summary": str(late_path),
            "summary_sha256": file_sha256_short(late_path),
            "source_rows_csv": str(data.get("source_rows_csv", "")),
            "source_rows_sha256": str(data.get("source_rows_sha256", "")),
            "predictions_csv": str(predictions_path),
            "predictions_sha256": str(
                late_outputs.get("predictions_sha256", "")
            ),
            "assignment_count": len(split_values),
        }

    manifest = {
        "feature_set": str(feature_set),
        "verification_threshold": threshold,
        "fit_summary": str(fit_path),
        "fit_summary_sha256": file_sha256_short(fit_path),
        "verifier_model": str(model_path),
        "verifier_model_sha256": model_hash,
        "splits": split_manifests,
        "late": late_manifest,
        "assignment_count": len(probabilities),
    }
    return FrozenMeasurementPoseEvidence(
        feature_set=str(feature_set),
        verification_threshold=threshold,
        probability_by_assignment=probabilities,
        manifest=manifest,
    )
