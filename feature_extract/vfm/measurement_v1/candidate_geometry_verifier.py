"""Learn candidate geometry probability from frozen coarse and RGB evidence."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics


CANDIDATE_GEOMETRY_FEATURE_NAMES = (
    "assignment_probability",
    "retrieval_similarity",
    "prior_geometry_p01",
    "prior_geometry_p02",
    "prior_geometry_p05",
    "inverse_score_rank",
    "rgb_validity_max",
    "rgb_validity_mean",
    "rgb_validity_posterior",
    "rgb_geometry_probability_max",
    "rgb_geometry_probability_mean",
    "rgb_geometry_probability_posterior",
    "likelihood_peak_max",
    "likelihood_peak_mean",
    "likelihood_margin_max",
    "likelihood_confidence_mean",
    "uncertainty_quality_mean",
    "view_offset_agreement_quality",
    "mean_mode_agreement_quality",
    "log_support_view_count",
)


@dataclass(frozen=True)
class CandidateGeometryVerifier:
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    geometry_threshold_px: float
    verification_threshold: float
    promotion_probability_min: float
    promotion_margin_min: float
    measurement_checkpoint_sha256: str

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        if self.feature_names != CANDIDATE_GEOMETRY_FEATURE_NAMES:
            raise ValueError("candidate geometry feature schema mismatch")
        if any(
            len(value) != count
            for value in (self.feature_mean, self.feature_scale, self.coefficients)
        ):
            raise ValueError("candidate geometry model vector lengths differ")
        if any(float(value) <= 0.0 for value in self.feature_scale):
            raise ValueError("candidate geometry feature scales must be positive")
        if len(str(self.measurement_checkpoint_sha256)) < 8:
            raise ValueError("candidate geometry verifier lacks checkpoint identity")

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64).reshape(-1, len(self.feature_names))
        standardized = (
            values - np.asarray(self.feature_mean, dtype=np.float64)[None]
        ) / np.asarray(self.feature_scale, dtype=np.float64)[None]
        base_logit = (
            standardized @ np.asarray(self.coefficients, dtype=np.float64)
            + float(self.intercept)
        )
        calibrated = (
            float(self.calibration_slope) * base_logit
            + float(self.calibration_intercept)
        )
        output = np.empty_like(calibrated)
        positive = calibrated >= 0.0
        output[positive] = 1.0 / (1.0 + np.exp(-calibrated[positive]))
        exp_values = np.exp(calibrated[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "candidate_geometry_verifier_v2",
            "feature_names": list(self.feature_names),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "calibration_slope": float(self.calibration_slope),
            "calibration_intercept": float(self.calibration_intercept),
            "geometry_threshold_px": float(self.geometry_threshold_px),
            "verification_threshold": float(self.verification_threshold),
            "promotion_probability_min": float(self.promotion_probability_min),
            "promotion_margin_min": float(self.promotion_margin_min),
            "measurement_checkpoint_sha256": str(
                self.measurement_checkpoint_sha256
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CandidateGeometryVerifier":
        if payload.get("format") not in {
            "candidate_geometry_verifier_v1",
            "candidate_geometry_verifier_v2",
        }:
            raise ValueError("unsupported candidate geometry verifier format")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            calibration_slope=float(payload["calibration_slope"]),
            calibration_intercept=float(payload["calibration_intercept"]),
            geometry_threshold_px=float(payload["geometry_threshold_px"]),
            verification_threshold=float(payload["verification_threshold"]),
            promotion_probability_min=float(
                payload.get("promotion_probability_min", payload["verification_threshold"])
            ),
            promotion_margin_min=float(payload.get("promotion_margin_min", 0.0)),
            measurement_checkpoint_sha256=str(
                payload["measurement_checkpoint_sha256"]
            ),
        )


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(row: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    text = str(row.get(key, "")).strip()
    if text:
        return float(text)
    if default is None:
        raise ValueError(f"candidate diagnostic row lacks {key}")
    return float(default)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.mean(values)) if total <= 0.0 else float(np.sum(values * weights) / total)


def build_candidate_geometry_examples(
    diagnostic_rows_csv: Path,
) -> list[dict[str, object]]:
    with Path(diagnostic_rows_csv).open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        identity = str(row.get("candidate_identity_key", "")).strip()
        if not identity:
            raise ValueError("candidate diagnostic row lacks candidate_identity_key")
        grouped.setdefault(identity, []).append(row)
    examples: list[dict[str, object]] = []
    for identity, views in grouped.items():
        first = views[0]
        query_id = str(first.get("query_id", ""))
        token = int(first["source_query_row"])
        measurement_rank = int(first["candidate_measurement_rank"])
        labels_2px = {_bool_text(view["target_geometry_correct_2px"]) for view in views}
        labels_5px = {_bool_text(view["target_geometry_correct_5px"]) for view in views}
        checkpoint_hashes = {
            str(view.get("measurement_checkpoint_sha256", "")).strip()
            for view in views
            if str(view.get("measurement_checkpoint_sha256", "")).strip()
        }
        if len(checkpoint_hashes) > 1:
            raise ValueError("candidate support views use different measurement checkpoints")
        if len(labels_2px) != 1 or len(labels_5px) != 1:
            raise ValueError("candidate support views disagree on geometry target")
        for key in (
            "candidate_assignment_probability",
            "candidate_retrieval_similarity",
            "candidate_geometry_p01",
            "candidate_geometry_p02",
            "candidate_geometry_p05",
            "candidate_score_rank",
        ):
            values = {_float(view, key) for view in views}
            if len(values) != 1:
                raise ValueError(f"candidate support views disagree on {key}")
        weights = np.asarray(
            [_float(view, "support_view_probability", default=1.0) for view in views],
            dtype=np.float64,
        )
        weights = np.clip(weights, 0.0, None)
        validity = np.asarray(
            [1.0 - _float(view, "dustbin_probability") for view in views],
            dtype=np.float64,
        )
        geometry_probability = np.asarray(
            [
                _float(
                    view,
                    "measurement_geometry_probability",
                    default=float(validity[index]),
                )
                for index, view in enumerate(views)
            ],
            dtype=np.float64,
        )
        peak = np.asarray(
            [_float(view, "likelihood_peak_probability") for view in views],
            dtype=np.float64,
        )
        margin = np.asarray(
            [_float(view, "likelihood_peak_margin") for view in views],
            dtype=np.float64,
        )
        confidence = 1.0 - np.asarray(
            [_float(view, "likelihood_normalized_entropy") for view in views],
            dtype=np.float64,
        )
        sigma = np.asarray(
            [_float(view, "likelihood_covariance_max_sigma_px") for view in views],
            dtype=np.float64,
        )
        predicted_offsets = np.asarray(
            [[_float(view, "pred_dx"), _float(view, "pred_dy")] for view in views],
            dtype=np.float64,
        )
        mode_offsets = np.asarray(
            [[_float(view, "peak_dx"), _float(view, "peak_dy")] for view in views],
            dtype=np.float64,
        )
        offset_center = np.mean(predicted_offsets, axis=0)
        view_disagreement = float(
            np.mean(np.linalg.norm(predicted_offsets - offset_center[None], axis=1))
        )
        mean_mode_disagreement = float(
            np.mean(np.linalg.norm(predicted_offsets - mode_offsets, axis=1))
        )
        features = {
            "assignment_probability": _float(first, "candidate_assignment_probability"),
            "retrieval_similarity": _float(first, "candidate_retrieval_similarity"),
            "prior_geometry_p01": _float(first, "candidate_geometry_p01"),
            "prior_geometry_p02": _float(first, "candidate_geometry_p02"),
            "prior_geometry_p05": _float(first, "candidate_geometry_p05"),
            "inverse_score_rank": 1.0 / max(_float(first, "candidate_score_rank"), 1.0),
            "rgb_validity_max": float(np.max(validity)),
            "rgb_validity_mean": float(np.mean(validity)),
            "rgb_validity_posterior": _weighted_mean(validity, weights),
            "rgb_geometry_probability_max": float(np.max(geometry_probability)),
            "rgb_geometry_probability_mean": float(np.mean(geometry_probability)),
            "rgb_geometry_probability_posterior": _weighted_mean(
                geometry_probability, weights
            ),
            "likelihood_peak_max": float(np.max(peak)),
            "likelihood_peak_mean": float(np.mean(peak)),
            "likelihood_margin_max": float(np.max(margin)),
            "likelihood_confidence_mean": float(np.mean(confidence)),
            "uncertainty_quality_mean": float(np.mean(1.0 / (1.0 + sigma))),
            "view_offset_agreement_quality": 1.0 / (1.0 + view_disagreement),
            "mean_mode_agreement_quality": 1.0 / (1.0 + mean_mode_disagreement),
            "log_support_view_count": math.log1p(float(len(views))),
        }
        values = np.asarray([features[name] for name in CANDIDATE_GEOMETRY_FEATURE_NAMES])
        if not np.all(np.isfinite(values)):
            raise ValueError("candidate geometry features must be finite")
        examples.append(
            {
                "candidate_identity_key": identity,
                "query_id": query_id,
                "source_query_row": token,
                "candidate_measurement_rank": measurement_rank,
                "track_id": int(first["track_id"]),
                "prototype_id": int(first["candidate_prototype_id"]),
                "label_2px": bool(next(iter(labels_2px))),
                "label_5px": bool(next(iter(labels_5px))),
                "measurement_checkpoint_sha256": (
                    "" if not checkpoint_hashes else next(iter(checkpoint_hashes))
                ),
                "features": features,
            }
        )
    return examples


def _matrix(examples: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"][name]) for name in CANDIDATE_GEOMETRY_FEATURE_NAMES]
            for example in examples
        ],
        dtype=np.float64,
    )


def _fit_base(features: np.ndarray, labels: np.ndarray, *, c_value: float) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000, random_state=0)
    model.fit((features - mean[None]) / scale[None], labels)
    return mean, scale, model


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def _fit_calibration(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=2000, random_state=0)
    model.fit(_logit(probabilities).reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _threshold_at_precision(labels: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    valid = np.flatnonzero(precision >= float(target))
    if not len(valid):
        return 1.0
    return float(probabilities[order[int(valid[-1])]])


def candidate_selection_metrics(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    promotion_probability_min: float | None = None,
    promotion_margin_min: float = 0.0,
) -> dict[str, object]:
    token_groups: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        token_groups.setdefault(int(example["source_query_row"]), []).append(index)
    output: dict[str, object] = {}
    for threshold in (2, 5):
        label_key = f"label_{threshold}px"
        baseline: list[bool] = []
        chosen: list[bool] = []
        oracle: list[bool] = []
        changed: list[bool] = []
        skipped = 0
        for indices in token_groups.values():
            frozen = [
                index
                for index in indices
                if int(examples[index]["candidate_measurement_rank"]) == 1
            ]
            if len(frozen) != 1:
                skipped += 1
                continue
            alternative_indices = [index for index in indices if index != frozen[0]]
            if not alternative_indices:
                skipped += 1
                continue
            best_alternative = max(
                alternative_indices,
                key=lambda index: (
                    float(probabilities[index]),
                    -int(examples[index]["candidate_measurement_rank"]),
                ),
            )
            should_promote = (
                promotion_probability_min is None
                or float(probabilities[best_alternative])
                >= float(promotion_probability_min)
            ) and (
                float(probabilities[best_alternative])
                - float(probabilities[frozen[0]])
                >= float(promotion_margin_min)
            )
            chosen_index = best_alternative if should_promote else frozen[0]
            baseline.append(bool(examples[frozen[0]][label_key]))
            chosen.append(bool(examples[chosen_index][label_key]))
            oracle.append(any(bool(examples[index][label_key]) for index in indices))
            changed.append(int(chosen_index) != int(frozen[0]))
        baseline_values = np.asarray(baseline, dtype=bool)
        chosen_values = np.asarray(chosen, dtype=bool)
        oracle_values = np.asarray(oracle, dtype=bool)
        rescue_eligible = (~baseline_values) & oracle_values
        rescued = (~baseline_values) & chosen_values
        harmed = baseline_values & (~chosen_values)
        output[f"geometry_correct_{threshold}px"] = {
            "comparable_token_count": int(len(baseline_values)),
            "skipped_token_count": int(skipped),
            "frozen_selected_correct_rate": float(np.mean(baseline_values)),
            "predicted_selected_correct_rate": float(np.mean(chosen_values)),
            "oracle_top_m_correct_rate": float(np.mean(oracle_values)),
            "candidate_change_rate": float(np.mean(changed)),
            "rescue_eligible_count": int(np.sum(rescue_eligible)),
            "rescued_count": int(np.sum(rescued)),
            "rescue_recall": float(np.sum(rescued) / max(np.sum(rescue_eligible), 1)),
            "harmed_count": int(np.sum(harmed)),
            "net_correct_change": int(np.sum(chosen_values) - np.sum(baseline_values)),
            "promotion_probability_min": promotion_probability_min,
            "promotion_margin_min": float(promotion_margin_min),
        }
    return output


def _fit_abstaining_promotion_margin(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    geometry_threshold_px: float,
    probability_min: float,
    target_precision: float,
) -> tuple[float, dict[str, object]]:
    """Freeze a probability-difference gate using OOF predictions only."""

    token_groups: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        token_groups.setdefault(int(example["source_query_row"]), []).append(index)
    label_key = f"label_{int(geometry_threshold_px)}px"
    records: list[tuple[float, bool, bool]] = []
    for indices in token_groups.values():
        frozen = [
            index
            for index in indices
            if int(examples[index]["candidate_measurement_rank"]) == 1
        ]
        if len(frozen) != 1:
            continue
        alternatives = [index for index in indices if index != frozen[0]]
        if not alternatives:
            continue
        alternative = max(alternatives, key=lambda index: float(probabilities[index]))
        if float(probabilities[alternative]) < float(probability_min):
            continue
        records.append(
            (
                float(probabilities[alternative] - probabilities[frozen[0]]),
                bool(examples[frozen[0]][label_key]),
                bool(examples[alternative][label_key]),
            )
        )
    empty = {
        "promotion_count": 0,
        "beneficial_count": 0,
        "harmful_count": 0,
        "neutral_count": 0,
        "outcome_precision": None,
        "net_correct_change": 0,
    }
    if not records:
        return float("inf"), empty
    margins = np.asarray([record[0] for record in records], dtype=np.float64)
    candidate_margins = np.unique(
        np.concatenate(
            [
                np.asarray([0.0], dtype=np.float64),
                np.quantile(margins, np.linspace(0.0, 1.0, 201)),
            ]
        )
    )
    best_key: tuple[int, int, float] | None = None
    best_margin = float("inf")
    best_report = empty
    for margin in candidate_margins.tolist():
        selected = [record for record in records if record[0] >= float(margin)]
        beneficial = sum((not before) and after for _delta, before, after in selected)
        harmful = sum(before and (not after) for _delta, before, after in selected)
        decisive = beneficial + harmful
        precision = None if decisive == 0 else float(beneficial / decisive)
        if precision is None or precision < float(target_precision):
            continue
        net = int(beneficial - harmful)
        selection_key = (net, int(beneficial), -float(margin))
        if best_key is not None and selection_key <= best_key:
            continue
        best_key = selection_key
        best_margin = float(margin)
        best_report = {
            "promotion_count": int(len(selected)),
            "beneficial_count": int(beneficial),
            "harmful_count": int(harmful),
            "neutral_count": int(len(selected) - decisive),
            "outcome_precision": precision,
            "net_correct_change": net,
        }
    return best_margin, best_report


def _evaluate(
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    geometry_threshold_px: float,
    promotion_probability_min: float | None = None,
    promotion_margin_min: float = 0.0,
) -> dict[str, object]:
    key = f"label_{int(geometry_threshold_px)}px"
    labels = np.asarray([bool(example[key]) for example in examples], dtype=bool)
    return {
        **confidence_metrics(labels, probabilities),
        "positive_count": int(np.sum(labels)),
        "sample_count": int(len(labels)),
        "selection": candidate_selection_metrics(
            examples,
            probabilities,
            promotion_probability_min=promotion_probability_min,
            promotion_margin_min=float(promotion_margin_min),
        ),
    }


def _write_probabilities_csv(
    path: Path,
    examples: Sequence[Mapping[str, object]],
    probabilities: np.ndarray,
    *,
    verification_threshold: float,
) -> None:
    fieldnames = [
        "candidate_identity_key",
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "geometry_probability",
        "verified",
    ]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example, probability in zip(examples, probabilities.tolist()):
            writer.writerow(
                {
                    **{name: example[name] for name in fieldnames[:6]},
                    "geometry_probability": float(probability),
                    "verified": bool(probability >= float(verification_threshold)),
                }
            )


def fit_candidate_geometry_verifier(
    *,
    train_diagnostic_rows_csv: Path,
    validation_diagnostic_rows_csv: Path,
    measurement_checkpoint: Path,
    output_dir: Path,
    geometry_threshold_px: float = 5.0,
    c_value: float = 1.0,
    fold_count: int = 5,
    target_precision: float = 0.8,
    allow_legacy_missing_checkpoint_hash: bool = False,
) -> dict[str, object]:
    if float(geometry_threshold_px) not in {2.0, 5.0}:
        raise ValueError("geometry_threshold_px must be 2 or 5")
    train = build_candidate_geometry_examples(Path(train_diagnostic_rows_csv))
    validation = build_candidate_geometry_examples(Path(validation_diagnostic_rows_csv))
    expected_checkpoint_hash = file_sha256_short(Path(measurement_checkpoint))
    observed_checkpoint_hashes = {
        str(example["measurement_checkpoint_sha256"])
        for example in (*train, *validation)
        if str(example["measurement_checkpoint_sha256"])
    }
    missing_checkpoint_hash_count = sum(
        not bool(str(example["measurement_checkpoint_sha256"]))
        for example in (*train, *validation)
    )
    if observed_checkpoint_hashes and observed_checkpoint_hashes != {
        expected_checkpoint_hash
    }:
        raise ValueError("candidate diagnostics and supplied measurement checkpoint differ")
    if missing_checkpoint_hash_count and not bool(allow_legacy_missing_checkpoint_hash):
        raise ValueError(
            "candidate diagnostics lack checkpoint identity; use only an explicit legacy baseline override"
        )
    features = _matrix(train)
    key = f"label_{int(geometry_threshold_px)}px"
    labels = np.asarray([bool(example[key]) for example in train], dtype=np.int64)
    groups = np.asarray([str(example["query_id"]) for example in train], dtype=object)
    if len(np.unique(labels)) != 2:
        raise ValueError("candidate geometry verifier requires both classes")
    splitter = GroupKFold(n_splits=min(int(fold_count), len(np.unique(groups))))
    oof = np.zeros((len(train),), dtype=np.float64)
    for train_indices, heldout_indices in splitter.split(features, labels, groups):
        mean, scale, model = _fit_base(
            features[train_indices], labels[train_indices], c_value=float(c_value)
        )
        oof[heldout_indices] = model.predict_proba(
            (features[heldout_indices] - mean[None]) / scale[None]
        )[:, 1]
    calibration_slope, calibration_intercept = _fit_calibration(oof, labels)
    calibrated_oof = 1.0 / (
        1.0 + np.exp(-(calibration_slope * _logit(oof) + calibration_intercept))
    )
    verification_threshold = _threshold_at_precision(
        labels, calibrated_oof, float(target_precision)
    )
    promotion_margin_min, promotion_fit_report = _fit_abstaining_promotion_margin(
        train,
        calibrated_oof,
        geometry_threshold_px=float(geometry_threshold_px),
        probability_min=float(verification_threshold),
        target_precision=float(target_precision),
    )
    mean, scale, model = _fit_base(features, labels, c_value=float(c_value))
    verifier = CandidateGeometryVerifier(
        feature_names=CANDIDATE_GEOMETRY_FEATURE_NAMES,
        feature_mean=tuple(mean.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(model.coef_[0].tolist()),
        intercept=float(model.intercept_[0]),
        calibration_slope=calibration_slope,
        calibration_intercept=calibration_intercept,
        geometry_threshold_px=float(geometry_threshold_px),
        verification_threshold=verification_threshold,
        promotion_probability_min=verification_threshold,
        promotion_margin_min=promotion_margin_min,
        measurement_checkpoint_sha256=expected_checkpoint_hash,
    )
    validation_probabilities = verifier.predict(_matrix(validation))
    validation_features = _matrix(validation)
    validation_baselines = {
        name: _evaluate(
            validation,
            validation_features[:, CANDIDATE_GEOMETRY_FEATURE_NAMES.index(feature_name)],
            geometry_threshold_px=float(geometry_threshold_px),
        )
        for name, feature_name in (
            ("assignment_probability", "assignment_probability"),
            ("prior_geometry_probability", f"prior_geometry_p{int(geometry_threshold_px):02d}"),
            ("rgb_validity_max", "rgb_validity_max"),
            ("rgb_validity_posterior", "rgb_validity_posterior"),
        )
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "candidate_geometry_verifier.json"
    model_path.write_text(json.dumps(verifier.to_dict(), indent=2, sort_keys=True) + "\n")
    train_oof_probabilities_path = output / "train_oof_probabilities.csv"
    _write_probabilities_csv(
        train_oof_probabilities_path,
        train,
        calibrated_oof,
        verification_threshold=float(verification_threshold),
    )
    summary = {
        "stage": "candidate_geometry_verifier_fit",
        "protocol": {
            "pose_features": False,
            "ground_truth_features": False,
            "labels": f"GT_pose_projection_residual_le_{geometry_threshold_px:g}px_TARGET_ONLY",
            "candidate_selection_frozen": True,
            "grouped_oof_by_query": True,
            "legacy_missing_checkpoint_hash_override": bool(
                allow_legacy_missing_checkpoint_hash
            ),
        },
        "geometry_threshold_px": float(geometry_threshold_px),
        "verification_threshold": float(verification_threshold),
        "promotion_probability_min": float(verification_threshold),
        "promotion_margin_min": float(promotion_margin_min),
        "promotion_fit_oof": promotion_fit_report,
        "train_oof": _evaluate(
            train,
            calibrated_oof,
            geometry_threshold_px=float(geometry_threshold_px),
            promotion_probability_min=float(verification_threshold),
            promotion_margin_min=float(promotion_margin_min),
        ),
        "validation": _evaluate(
            validation,
            validation_probabilities,
            geometry_threshold_px=float(geometry_threshold_px),
            promotion_probability_min=float(verification_threshold),
            promotion_margin_min=float(promotion_margin_min),
        ),
        "validation_unconditional_argmax_DIAGNOSTIC_ONLY": _evaluate(
            validation,
            validation_probabilities,
            geometry_threshold_px=float(geometry_threshold_px),
        ),
        "validation_single_signal_baselines": validation_baselines,
        "inputs": {
            "train_diagnostic_rows_csv": str(train_diagnostic_rows_csv),
            "train_diagnostic_rows_sha256": file_sha256_short(Path(train_diagnostic_rows_csv)),
            "validation_diagnostic_rows_csv": str(validation_diagnostic_rows_csv),
            "validation_diagnostic_rows_sha256": file_sha256_short(Path(validation_diagnostic_rows_csv)),
            "measurement_checkpoint": str(measurement_checkpoint),
            "measurement_checkpoint_sha256": expected_checkpoint_hash,
        },
        "outputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(model_path),
            "train_oof_probabilities": str(train_oof_probabilities_path),
            "train_oof_probabilities_sha256": file_sha256_short(
                train_oof_probabilities_path
            ),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def apply_candidate_geometry_verifier(
    *,
    model_path: Path,
    diagnostic_rows_csv: Path,
    output_path: Path,
    allow_legacy_missing_checkpoint_hash: bool = False,
) -> dict[str, object]:
    verifier = CandidateGeometryVerifier.from_dict(json.loads(Path(model_path).read_text()))
    examples = build_candidate_geometry_examples(Path(diagnostic_rows_csv))
    diagnostic_checkpoint_hashes = {
        str(example["measurement_checkpoint_sha256"])
        for example in examples
        if str(example["measurement_checkpoint_sha256"])
    }
    if diagnostic_checkpoint_hashes and diagnostic_checkpoint_hashes != {
        verifier.measurement_checkpoint_sha256
    }:
        raise ValueError(
            "candidate diagnostics use a different measurement checkpoint"
        )
    if not diagnostic_checkpoint_hashes and not bool(
        allow_legacy_missing_checkpoint_hash
    ):
        raise ValueError("candidate diagnostics are missing measurement checkpoint identity")
    probabilities = verifier.predict(_matrix(examples))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_probabilities_csv(
        output,
        examples,
        probabilities,
        verification_threshold=float(verifier.verification_threshold),
    )
    summary = {
        "stage": "candidate_geometry_verifier_apply",
        "metrics_TARGET_ONLY": _evaluate(
            examples,
            probabilities,
            geometry_threshold_px=verifier.geometry_threshold_px,
            promotion_probability_min=verifier.promotion_probability_min,
            promotion_margin_min=verifier.promotion_margin_min,
        ),
        "verification_threshold": float(verifier.verification_threshold),
        "legacy_missing_checkpoint_hash_override": bool(
            allow_legacy_missing_checkpoint_hash
        ),
        "promotion_probability_min": float(verifier.promotion_probability_min),
        "promotion_margin_min": float(verifier.promotion_margin_min),
        "verified_count": int(np.sum(probabilities >= verifier.verification_threshold)),
        "inputs": {
            "model": str(model_path),
            "model_sha256": file_sha256_short(Path(model_path)),
            "diagnostic_rows_csv": str(diagnostic_rows_csv),
            "diagnostic_rows_sha256": file_sha256_short(Path(diagnostic_rows_csv)),
        },
        "outputs": {
            "probabilities_csv": str(output),
            "probabilities_sha256": file_sha256_short(output),
            "summary": str(output.with_suffix(".summary.json")),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
