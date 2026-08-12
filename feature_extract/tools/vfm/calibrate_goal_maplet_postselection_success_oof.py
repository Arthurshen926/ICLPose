"""Fit low-capacity success calibration from complete train-only OOF outcomes.

Route-crossfit metrics must consume nested policy-selected rows.  A separate
all-OOF-selected policy report can be supplied only for fitting the final
deployment calibrator; its outcomes are never used for the reported crossfit
calibration metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256
from feature_extract.vfm.postselection_success_calibration import (
    FEATURE_NAMES,
    postselection_feature_row,
    sigmoid_logistic_payload,
)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    regularization_c: float,
) -> tuple[np.ndarray, dict[str, object]]:
    if np.unique(train_y).size != 2:
        raise ValueError("success calibration fold needs both outcome classes")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=float(regularization_c),
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        random_state=0,
    ).fit(scaler.transform(train_x), train_y)
    probability = model.predict_proba(scaler.transform(test_x))[:, 1]
    return probability, sigmoid_logistic_payload(model, scaler.mean_, scaler.scale_)


def _parse_regularization_candidates(value: str) -> list[float]:
    candidates = sorted({float(item) for item in str(value).split(",") if item.strip()})
    if not candidates or any(item <= 0.0 for item in candidates):
        raise ValueError("regularization candidates must be positive")
    return candidates


def _select_regularization_nested(
    features: np.ndarray,
    labels: np.ndarray,
    trajectories: np.ndarray,
    folds: Sequence[dict[str, object]],
    candidates: Sequence[float],
) -> dict[str, object]:
    """Select logistic C using route folds wholly inside the supplied rows."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    trajectories = np.asarray(trajectories)
    records = []
    for candidate in sorted(float(value) for value in candidates):
        total_nll = 0.0
        validation_count = 0
        fold_records = []
        viable = True
        for fold in folds:
            held = set(str(value) for value in fold["held_query_trajectories"])
            validation_mask = np.asarray([
                str(value) in held for value in trajectories
            ])
            if not np.any(validation_mask):
                continue
            training_mask = ~validation_mask
            if np.unique(labels[training_mask]).size != 2:
                viable = False
                fold_records.append({
                    "fold_id": str(fold["fold_id"]),
                    "viable": False,
                    "reason": "inner_training_has_one_outcome_class",
                })
                break
            probability, _payload = _fit_predict(
                features[training_mask], labels[training_mask],
                features[validation_mask], regularization_c=candidate,
            )
            validation_label = labels[validation_mask]
            nll_sum = float(-np.sum(
                validation_label * np.log(np.maximum(probability, 1.0e-12))
                + (~validation_label) * np.log(
                    np.maximum(1.0 - probability, 1.0e-12)
                )
            ))
            total_nll += nll_sum
            validation_count += int(np.sum(validation_mask))
            fold_records.append({
                "fold_id": str(fold["fold_id"]),
                "viable": True,
                "training_count": int(np.sum(training_mask)),
                "validation_count": int(np.sum(validation_mask)),
                "validation_negative_log_likelihood": float(
                    nll_sum / max(int(np.sum(validation_mask)), 1)
                ),
            })
        if validation_count != labels.size:
            viable = False
        records.append({
            "C": candidate,
            "viable": viable,
            "validation_count": validation_count,
            "mean_negative_log_likelihood": (
                float(total_nll / validation_count)
                if viable and validation_count else None
            ),
            "folds": fold_records,
        })
    viable_records = [value for value in records if bool(value["viable"])]
    if not viable_records:
        raise ValueError("no regularization candidate has complete inner-route evidence")
    # A tie prefers the smaller C and therefore stronger regularization.
    selected = min(
        viable_records,
        key=lambda value: (float(value["mean_negative_log_likelihood"]), float(value["C"])),
    )
    return {
        "selection_semantics": "minimum_inner_route_crossfit_negative_log_likelihood",
        "selected_C": float(selected["C"]),
        "candidates": records,
    }


def _route_crossfit_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    trajectories: np.ndarray,
    folds: Sequence[dict[str, object]],
    regularization_candidates: Sequence[float],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    probability = np.full((labels.size,), np.nan, dtype=np.float64)
    fold_models = []
    for fold in folds:
        held = set(str(value) for value in fold["held_query_trajectories"])
        test_mask = np.asarray([str(value) in held for value in trajectories])
        train_mask = ~test_mask
        if not np.any(test_mask):
            raise ValueError("calibrator route fold contains no held outcomes")
        regularization_selection = _select_regularization_nested(
            features[train_mask], labels[train_mask], trajectories[train_mask],
            folds, regularization_candidates,
        )
        selected_c = float(regularization_selection["selected_C"])
        held_probability, model_payload = _fit_predict(
            features[train_mask], labels[train_mask], features[test_mask],
            regularization_c=selected_c,
        )
        probability[test_mask] = held_probability
        fold_models.append({
            "fold_id": str(fold["fold_id"]),
            "held_query_trajectories": sorted(held),
            "training_count": int(np.sum(train_mask)),
            "query_count": int(np.sum(test_mask)),
            "regularization_selection": regularization_selection,
            "model": model_payload,
        })
    if np.any(~np.isfinite(probability)):
        raise ValueError("calibrator route cross-fit left predictions missing")
    return probability, fold_models


def _calibration_metrics(probability: np.ndarray, label: np.ndarray) -> dict[str, object]:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probability >= lower) & (
            probability <= upper if upper == 1.0 else probability < upper
        )
        if not np.any(mask):
            continue
        confidence = float(np.mean(probability[mask]))
        success = float(np.mean(label[mask]))
        ece += float(np.mean(mask)) * abs(confidence - success)
        rows.append({
            "lower": float(lower), "upper": float(upper),
            "count": int(np.sum(mask)), "mean_probability": confidence,
            "empirical_success": success,
        })
    order = np.argsort(-probability, kind="stable")
    risk_coverage = []
    for coverage in (0.25, 0.50, 0.75, 0.90, 1.0):
        take = max(1, int(np.ceil(coverage * label.size)))
        selected = order[:take]
        risk_coverage.append({
            "requested_coverage": coverage,
            "accepted_count": take,
            "success_precision": float(np.mean(label[selected])),
            "success_yield": float(np.sum(label[selected]) / label.size),
            "minimum_probability": float(np.min(probability[selected])),
            "accepted_success_wilson95": _wilson(
                int(np.sum(label[selected])), take
            ),
        })
    return {
        "count": int(label.size),
        "positive_count": int(np.sum(label)),
        "base_rate": float(np.mean(label)),
        "brier": float(np.mean(np.square(probability - label.astype(np.float64)))),
        "negative_log_likelihood": float(-np.mean(
            label * np.log(np.maximum(probability, 1.0e-12))
            + (~label) * np.log(np.maximum(1.0 - probability, 1.0e-12))
        )),
        "expected_calibration_error_10bin": float(ece),
        "calibration_bins": rows,
        "risk_coverage": risk_coverage,
    }


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count <= 0:
        return [float("nan"), float("nan")]
    probability = float(successes) / float(count)
    denominator = 1.0 + z * z / count
    center = (probability + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(
        probability * (1.0 - probability) / count + z * z / (4.0 * count**2)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def _parse_success_targets(value: str) -> list[float]:
    targets = sorted({float(item) for item in value.split(",") if item.strip()})
    if not targets or any(item <= 0.0 or item >= 1.0 for item in targets):
        raise ValueError("minimum success targets must lie strictly between zero and one")
    return targets


def _acceptance_metrics(decision: np.ndarray, label: np.ndarray) -> dict[str, object]:
    decision = np.asarray(decision, dtype=bool)
    label = np.asarray(label, dtype=bool)
    accepted = int(np.sum(decision))
    successes = int(np.sum(label & decision))
    return {
        "query_count": int(label.size),
        "accepted_count": accepted,
        "accepted_coverage": float(accepted / label.size) if label.size else 0.0,
        "accepted_success_count": successes,
        "accepted_success_precision": float(successes / accepted) if accepted else None,
        "accepted_success_yield": float(successes / label.size) if label.size else 0.0,
        "accepted_success_wilson95": (
            _wilson(successes, accepted) if accepted else [None, None]
        ),
    }


def _select_risk_threshold(
    probability: np.ndarray, label: np.ndarray, *, minimum_success: float,
) -> dict[str, object]:
    probability = np.asarray(probability, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    best = None
    # Evaluate deployed >= thresholds at unique observed probabilities so ties
    # cannot make the deployed accepted set larger than the fitted set.
    for threshold in np.unique(probability)[::-1]:
        decision = probability >= float(threshold)
        metrics = _acceptance_metrics(decision, label)
        lower = metrics["accepted_success_wilson95"][0]
        if metrics["accepted_count"] and lower >= float(minimum_success):
            if best is None or metrics["accepted_count"] > best["accepted_count"]:
                best = {"probability_threshold": float(threshold), **metrics}
    if best is None:
        best = {
            "probability_threshold": 1.0000001,
            **_acceptance_metrics(np.zeros(label.shape, dtype=bool), label),
        }
    best["minimum_wilson95_success_precision"] = float(minimum_success)
    return best


def _nested_risk_threshold_evaluation(
    probability: np.ndarray,
    label: np.ndarray,
    trajectory: np.ndarray,
    folds: list[dict[str, object]],
    *,
    minimum_success: float,
) -> dict[str, object]:
    decision = np.zeros(label.shape, dtype=bool)
    records = []
    for fold in folds:
        held = set(str(value) for value in fold["held_query_trajectories"])
        held_mask = np.asarray([value in held for value in trajectory])
        selected = _select_risk_threshold(
            probability[~held_mask], label[~held_mask],
            minimum_success=float(minimum_success),
        )
        held_decision = probability[held_mask] >= float(
            selected["probability_threshold"]
        )
        decision[held_mask] = held_decision
        records.append({
            "fold_id": str(fold["fold_id"]),
            "held_query_trajectories": sorted(held),
            "threshold_selected_without_held_route_outcomes": True,
            "training_selection": selected,
            "held_evaluation": _acceptance_metrics(
                held_decision, label[held_mask]
            ),
        })
    return {
        "selection_semantics": "nested_route_crossfit_without_held_route_outcomes",
        "eligible_for_unbiased_train_side_evaluation": True,
        "aggregate": _acceptance_metrics(decision, label),
        "folds": records,
    }


def _load_postselection_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        if str(report.get("postselection_evidence_contract", "")) != (
            "proposal_to_pruning_to_exact_ranking_to_refinement_to_winner_v1"
        ):
            raise ValueError("calibration input lacks complete post-selection evidence")
        rows.extend(report.get("rows", []))
    return rows


def _validate_complete_rows(
    rows: list[dict[str, object]],
    expected: dict[str, object],
    *,
    role: str,
) -> list[str]:
    image_ids = [str(row["image_id"]) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"{role} reports contain duplicate images")
    if (
        len(image_ids) != int(expected["count"])
        or ordered_id_sha256(image_ids) != str(expected["image_ids_sha256"])
    ):
        raise ValueError(f"{role} reports do not cover official train exactly")
    return image_ids


def _features_and_labels(
    rows: list[dict[str, object]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = np.stack([postselection_feature_row(row) for row in rows])
    if np.any(~np.isfinite(features)):
        raise ValueError("post-selection calibration contains non-finite features")
    translation = np.asarray([row["final_translation_m"] for row in rows])
    rotation = np.asarray([row["final_rotation_deg"] for row in rows])
    return features, {
        "strict_0.5m_5deg": (translation <= 0.5) & (rotation <= 5.0),
        "loose_1m_10deg": (translation <= 1.0) & (rotation <= 10.0),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement_reports", required=True, nargs="+")
    parser.add_argument(
        "--final_fit_refinement_report",
        default="",
        help=(
            "Optional all-OOF-selected frozen-policy replay used only to fit "
            "the final model. Crossfit metrics still use --refinement_reports."
        ),
    )
    parser.add_argument("--protocol_json", required=True)
    parser.add_argument(
        "--regularization_candidates", default="0.01,0.03,0.1,0.3,1.0",
    )
    parser.add_argument(
        "--regularization_c", type=float, default=None,
        help="Deprecated fixed-C override; prefer nested candidate selection.",
    )
    parser.add_argument("--minimum_success_targets", default="0.8,0.9,0.95")
    parser.add_argument(
        "--require_nested_policy_evaluation",
        action="store_true",
        help="Reject evaluation reports whose refinement policy saw held-route outcomes.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output_json)
    if output.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite calibrator: {output}")
    protocol_path = Path(args.protocol_json)
    protocol = json.loads(protocol_path.read_text())
    report_paths = [Path(value) for value in args.refinement_reports]
    evaluation_reports = [json.loads(path.read_text()) for path in report_paths]
    evaluation_policy_is_nested = all(
        bool(report.get("eligible_for_unbiased_train_side_evaluation", False))
        and str(report.get("selection_semantics", ""))
        == "nested_route_crossfit_without_held_route_outcomes"
        for report in evaluation_reports
    )
    if bool(args.require_nested_policy_evaluation) and not evaluation_policy_is_nested:
        raise ValueError("success calibration evaluation requires nested policy selection")
    rows = _load_postselection_rows(report_paths)
    expected = protocol["official_train"]
    image_ids = _validate_complete_rows(rows, expected, role="evaluation OOF")
    features, labels = _features_and_labels(rows)

    final_fit_path = (
        Path(args.final_fit_refinement_report)
        if str(args.final_fit_refinement_report) else None
    )
    if final_fit_path is None:
        final_fit_rows = rows
        final_fit_image_ids = image_ids
    else:
        final_fit_rows = _load_postselection_rows([final_fit_path])
        final_fit_image_ids = _validate_complete_rows(
            final_fit_rows, expected, role="final-fit policy"
        )
    final_features, final_labels = _features_and_labels(final_fit_rows)
    trajectory = np.asarray([value.split("/", 1)[0] for value in image_ids])
    final_trajectory = np.asarray([
        value.split("/", 1)[0] for value in final_fit_image_ids
    ])
    success_targets = _parse_success_targets(str(args.minimum_success_targets))
    regularization_candidates = (
        [float(args.regularization_c)]
        if args.regularization_c is not None
        else _parse_regularization_candidates(str(args.regularization_candidates))
    )
    heads = {}
    predictions_by_head = {}
    final_policy_predictions_by_head = {}
    for name, label in labels.items():
        folds = list(protocol["development"]["folds"])
        oof_probability, fold_models = _route_crossfit_predictions(
            features, label, trajectory, folds, regularization_candidates,
        )
        final_policy_probability, final_policy_fold_models = (
            _route_crossfit_predictions(
                final_features, final_labels[name], final_trajectory,
                folds, regularization_candidates,
            )
        )
        final_regularization_selection = _select_regularization_nested(
            final_features, final_labels[name], final_trajectory,
            list(protocol["development"]["folds"]),
            regularization_candidates,
        )
        _, final_model = _fit_predict(
            final_features, final_labels[name], final_features[:1],
            regularization_c=float(final_regularization_selection["selected_C"]),
        )
        heads[name] = {
            "metrics": _calibration_metrics(oof_probability, label),
            "metrics_by_trajectory": {
                route: _calibration_metrics(
                    oof_probability[trajectory == route], label[trajectory == route]
                )
                for route in sorted(set(trajectory.tolist()))
            },
            "fold_models": fold_models,
            "final_policy_oof_fit_diagnostic": {
                "eligible_for_unbiased_train_side_evaluation": False,
                "reason": (
                    "deployment_policy_was_selected_on_all_train_oof_outcomes"
                ),
                "metrics": _calibration_metrics(
                    final_policy_probability, final_labels[name]
                ),
                "fold_models": final_policy_fold_models,
            },
            "final_regularization_selection": final_regularization_selection,
            "final_all_official_train_oof_model": final_model,
            "selective_operating_points": {
                f"wilson95_min_success_{target:g}": {
                    "final_deployment_threshold": _select_risk_threshold(
                        final_policy_probability, final_labels[name],
                        minimum_success=target,
                    ),
                    "final_threshold_selection_semantics": (
                        "selected_from_crossfit_probabilities_of_the_all_train_"
                        "selected_deployment_policy_for_final_freeze_only"
                    ),
                    "nested_route_crossfit_evaluation": (
                        _nested_risk_threshold_evaluation(
                            oof_probability, label, trajectory,
                            list(protocol["development"]["folds"]),
                            minimum_success=target,
                        )
                    ),
                }
                for target in success_targets
            },
        }
        predictions_by_head[name] = oof_probability
        final_policy_predictions_by_head[name] = final_policy_probability
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "image_id": image_ids[index],
            "trajectory_id": str(trajectory[index]),
            "strict_label": bool(labels["strict_0.5m_5deg"][index]),
            "loose_label": bool(labels["loose_1m_10deg"][index]),
            "strict_oof_probability": float(
                predictions_by_head["strict_0.5m_5deg"][index]
            ),
            "loose_oof_probability": float(
                predictions_by_head["loose_1m_10deg"][index]
            ),
        })
    final_policy_prediction_rows = []
    for index, row in enumerate(final_fit_rows):
        final_policy_prediction_rows.append({
            "image_id": final_fit_image_ids[index],
            "trajectory_id": str(final_trajectory[index]),
            "strict_label": bool(final_labels["strict_0.5m_5deg"][index]),
            "loose_label": bool(final_labels["loose_1m_10deg"][index]),
            "strict_oof_probability": float(
                final_policy_predictions_by_head["strict_0.5m_5deg"][index]
            ),
            "loose_oof_probability": float(
                final_policy_predictions_by_head["loose_1m_10deg"][index]
            ),
        })
    payload = {
        "artifact_type": "goal_maplet_postselection_success_calibration_oof_v1",
        "probability_semantics": "P(pose_error_within_threshold|post_selection_evidence)",
        "pose_posterior_claimed": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "regularization": {
            "kind": "l2_logistic",
            "candidate_C": regularization_candidates,
            "selection": (
                "nested_inner_route_crossfit_for_each_outer_fold_and_"
                "all_train_route_crossfit_for_final_model"
                if args.regularization_c is None else "fixed_C_deprecated_override"
            ),
        },
        "selective_risk_targets": success_targets,
        "selective_threshold_semantics": (
            "maximum_coverage_probability_threshold_whose_training_wilson95_lower_bound_meets_target"
        ),
        "protocol_json": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "source_reports": [str(path) for path in report_paths],
        "source_report_sha256": [file_sha256(path) for path in report_paths],
        "evaluation_policy_semantics": (
            "nested_route_crossfit_policy_selection_without_held_route_outcomes"
            if evaluation_policy_is_nested else "not_declared_nested_by_source_report"
        ),
        "final_fit_policy_report": (
            str(final_fit_path) if final_fit_path is not None else None
        ),
        "final_fit_policy_report_sha256": (
            file_sha256(final_fit_path) if final_fit_path is not None else None
        ),
        "final_fit_query_count": len(final_fit_rows),
        "final_fit_image_ids_sha256": ordered_id_sha256(final_fit_image_ids),
        "final_fit_outcomes_used_for_crossfit_metrics": final_fit_path is None,
        "training_outcomes_are_base_model_oof": True,
        "calibrator_evaluation_is_route_grouped_oof": True,
        "policy_selection_is_nested_for_calibrator_evaluation": (
            evaluation_policy_is_nested
        ),
        "query_count": len(rows),
        "heads": heads,
        "predictions": prediction_rows,
        "final_policy_fit_only_predictions": final_policy_prediction_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: value for key, value in payload.items()
        if key not in {"predictions", "final_policy_fit_only_predictions"}
    }, indent=2))


if __name__ == "__main__":
    main()
