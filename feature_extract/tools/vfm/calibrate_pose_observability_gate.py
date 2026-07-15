"""Calibrate an inference-safe P22/P24 promotion policy on a development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short


MINIMUM_METRICS = (
    "information_match_count",
    "translation_information_min_eigenvalue",
    "bearing_max_angle_deg",
    "camera_depth_span_ratio",
    "xyz_second_singular_ratio",
    "xyz_third_singular_ratio",
)
MAXIMUM_METRICS = (
    "translation_information_condition",
    "joint_information_condition",
)
OBSERVABILITY_METRICS = (*MINIMUM_METRICS, *MAXIMUM_METRICS)

DEFAULT_LIKELIHOOD_DELTA_GRID = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.5,
    1.0,
)
DEFAULT_EFFECTIVE_GROUP_GRID = (4, 6, 8, 10, 12, 16, 24, 32)

OBSERVABILITY_CLI_FLAGS = {
    "information_match_count": "--grouped_observability_min_information_matches",
    "translation_information_min_eigenvalue": (
        "--grouped_observability_min_translation_eigenvalue"
    ),
    "translation_information_condition": (
        "--grouped_observability_max_translation_condition"
    ),
    "joint_information_condition": "--grouped_observability_max_joint_condition",
    "bearing_max_angle_deg": "--grouped_observability_min_bearing_span_deg",
    "camera_depth_span_ratio": "--grouped_observability_min_depth_span_ratio",
    "xyz_second_singular_ratio": "--grouped_observability_min_xyz_second_ratio",
    "xyz_third_singular_ratio": "--grouped_observability_min_xyz_third_ratio",
}


def _policy_rows(payload: dict, split_name: str, policy_key: str) -> list[dict]:
    split = payload.get(split_name)
    if not isinstance(split, dict):
        return []
    policy = split.get(policy_key)
    if not isinstance(policy, dict) or not isinstance(policy.get("verified"), list):
        return []
    return list(policy["verified"])


def _records(rows: Sequence[dict]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        backend = row.get("inference_verification")
        promotion = (
            None
            if not isinstance(backend, dict)
            else backend.get("crossfit_likelihood_promotion")
        )
        if not isinstance(promotion, dict):
            continue
        observability = promotion.get("optional_observability")
        output.append(
            {
                "query_id": str(row["query_id"]),
                "baseline_translation_m": row.get(
                    "crossfit_baseline_translation_m_TARGET_ONLY"
                ),
                "optional_translation_m": row.get(
                    "crossfit_optional_translation_m_TARGET_ONLY"
                ),
                "baseline_rotation_deg": row.get(
                    "crossfit_baseline_rotation_deg_TARGET_ONLY"
                ),
                "optional_rotation_deg": row.get(
                    "crossfit_optional_rotation_deg_TARGET_ONLY"
                ),
                "base_promoted": bool(promotion.get("promoted")),
                "baseline_success": bool(promotion.get("baseline_success")),
                "optional_success": bool(promotion.get("optional_success")),
                "likelihood_delta": promotion.get(
                    "optional_minus_baseline_log_likelihood_mean"
                ),
                "baseline_likelihood": promotion.get(
                    "baseline_log_likelihood_mean"
                ),
                "optional_likelihood": promotion.get(
                    "optional_log_likelihood_mean"
                ),
                "effective_group_count": int(
                    promotion.get("effective_group_count", 0)
                ),
                "observability": (
                    dict(observability) if isinstance(observability, dict) else {}
                ),
            }
        )
    return output


def _validate_record_set(
    records: Sequence[dict[str, object]], *, split_name: str
) -> None:
    query_ids = [str(record["query_id"]) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError(f"{split_name} contains duplicate promotion query IDs")


def _parse_observability_metrics(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(value).split(",") if name.strip())
    if not names:
        raise ValueError("at least one observability metric must be enabled")
    if len(names) != len(set(names)):
        raise ValueError("observability metric names must be unique")
    unknown = set(names).difference(OBSERVABILITY_METRICS)
    if unknown:
        raise ValueError(f"unknown observability metrics: {sorted(unknown)}")
    return names


def _finite(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _passes(record: dict[str, object], thresholds: dict[str, float]) -> bool:
    values = record["observability"]
    assert isinstance(values, dict)
    for name, threshold in thresholds.items():
        value = _finite(values.get(name))
        if value is None:
            return False
        if name in MINIMUM_METRICS and value < float(threshold):
            return False
        if name in MAXIMUM_METRICS and value > float(threshold):
            return False
    return True


def _likelihood_gate_passes(
    record: dict[str, object],
    *,
    min_likelihood_delta: float,
    min_effective_group_count: int,
) -> bool:
    delta = _finite(record.get("likelihood_delta"))
    baseline_success = bool(record.get("baseline_success", True))
    optional_success = bool(record.get("optional_success", True))
    enough_groups = int(record.get("effective_group_count", 0)) >= int(
        min_effective_group_count
    )
    if not optional_success or not enough_groups:
        return False
    if not baseline_success:
        return _finite(record.get("optional_likelihood")) is not None
    return bool(delta is not None and delta >= float(min_likelihood_delta))


def _with_likelihood_gate(
    records: Sequence[dict[str, object]],
    *,
    min_likelihood_delta: float,
    min_effective_group_count: int,
) -> list[dict[str, object]]:
    return [
        {
            **record,
            "base_promoted": _likelihood_gate_passes(
                record,
                min_likelihood_delta=min_likelihood_delta,
                min_effective_group_count=min_effective_group_count,
            ),
        }
        for record in records
    ]


def evaluate_policy(
    records: Sequence[dict[str, object]],
    thresholds: dict[str, float],
    *,
    min_likelihood_delta: float | None = None,
    min_effective_group_count: int | None = None,
    catastrophic_threshold_m: float = 0.5,
    failure_penalty_m: float = 10.0,
) -> dict[str, object]:
    selected = []
    baseline = []
    optional = []
    promoted = []
    for record in records:
        baseline_error = _finite(record["baseline_translation_m"])
        optional_error = _finite(record["optional_translation_m"])
        if min_likelihood_delta is None and min_effective_group_count is None:
            likelihood_gate = bool(record["base_promoted"])
        else:
            likelihood_gate = _likelihood_gate_passes(
                record,
                min_likelihood_delta=float(min_likelihood_delta or 0.0),
                min_effective_group_count=int(min_effective_group_count or 0),
            )
        use_optional = likelihood_gate and _passes(record, thresholds)
        selected_error = optional_error if use_optional else baseline_error
        selected.append(
            float(failure_penalty_m)
            if selected_error is None
            else float(selected_error)
        )
        baseline.append(
            float(failure_penalty_m)
            if baseline_error is None
            else float(baseline_error)
        )
        optional.append(
            float(failure_penalty_m)
            if optional_error is None
            else float(optional_error)
        )
        promoted.append(bool(use_optional))
    values = np.asarray(selected, dtype=np.float64)
    baseline_values = np.asarray(baseline, dtype=np.float64)
    optional_values = np.asarray(optional, dtype=np.float64)
    promoted_mask = np.asarray(promoted, dtype=bool)
    wins = promoted_mask & (optional_values < baseline_values)
    losses = promoted_mask & (optional_values > baseline_values)
    newly_catastrophic = (
        promoted_mask
        & (baseline_values <= float(catastrophic_threshold_m))
        & (optional_values > float(catastrophic_threshold_m))
    )
    rescued_catastrophic = (
        promoted_mask
        & (baseline_values > float(catastrophic_threshold_m))
        & (optional_values <= float(catastrophic_threshold_m))
    )
    promotion_regressions = np.where(
        promoted_mask,
        optional_values - baseline_values,
        0.0,
    )
    return {
        "query_count": int(len(values)),
        "promotion_count": int(np.count_nonzero(promoted_mask)),
        "translation_median_m": float(np.median(values)),
        "translation_p90_m": float(np.percentile(values, 90)),
        "catastrophic_count": int(
            np.count_nonzero(values > float(catastrophic_threshold_m))
        ),
        "recall_3cm": float(np.mean(values <= 0.03)),
        "recall_5cm": float(np.mean(values <= 0.05)),
        "recall_10cm": float(np.mean(values <= 0.10)),
        "recall_25cm": float(np.mean(values <= 0.25)),
        "promotion_win_count": int(np.count_nonzero(wins)),
        "promotion_loss_count": int(np.count_nonzero(losses)),
        "new_catastrophic_promotion_count": int(
            np.count_nonzero(newly_catastrophic)
        ),
        "rescued_catastrophic_promotion_count": int(
            np.count_nonzero(rescued_catastrophic)
        ),
        "max_promotion_regression_m": float(np.max(promotion_regressions)),
        "selected_translation_m_by_query": {
            str(record["query_id"]): float(value)
            for record, value in zip(records, values)
        },
    }


def _objective(metrics: dict[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["new_catastrophic_promotion_count"]),
        float(metrics["catastrophic_count"]),
        float(metrics["translation_p90_m"]),
        float(metrics["translation_median_m"]),
        -float(metrics["recall_10cm"]),
    )


def _candidate_thresholds(
    records: Sequence[dict[str, object]],
    *,
    allowed_metrics: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    promoted = [record for record in records if bool(record["base_promoted"])]
    output: dict[str, list[float]] = {}
    metric_names = (
        tuple(str(name) for name in allowed_metrics)
        if allowed_metrics is not None
        else (*MINIMUM_METRICS, *MAXIMUM_METRICS)
    )
    unknown = set(metric_names).difference((*MINIMUM_METRICS, *MAXIMUM_METRICS))
    if unknown:
        raise ValueError(f"unknown observability metrics: {sorted(unknown)}")
    for name in metric_names:
        values = np.asarray(
            [
                value
                for record in promoted
                if (value := _finite(record["observability"].get(name))) is not None
            ],
            dtype=np.float64,
        )
        if len(values) < 3:
            continue
        quantiles = (10, 20, 30, 40, 50) if name in MINIMUM_METRICS else (90, 80, 70, 60, 50)
        output[name] = sorted(
            set(float(np.percentile(values, quantile)) for quantile in quantiles)
        )
    return output


def calibrate_thresholds(
    records: Sequence[dict[str, object]],
    *,
    max_gate_count: int = 2,
    catastrophic_threshold_m: float = 0.5,
    max_median_regression_m: float = 0.0,
    min_likelihood_delta: float | None = None,
    min_effective_group_count: int | None = None,
    allowed_metrics: Sequence[str] | None = None,
) -> tuple[dict[str, float], dict[str, object]]:
    if (min_likelihood_delta is None) != (min_effective_group_count is None):
        raise ValueError("likelihood delta and effective group count must be set together")
    if min_likelihood_delta is not None:
        records = _with_likelihood_gate(
            records,
            min_likelihood_delta=float(min_likelihood_delta),
            min_effective_group_count=int(min_effective_group_count),
        )
    baseline_metrics = evaluate_policy(
        records, {}, catastrophic_threshold_m=catastrophic_threshold_m
    )
    immutable_metrics = evaluate_policy(
        [{**record, "base_promoted": False} for record in records],
        {},
        catastrophic_threshold_m=catastrophic_threshold_m,
    )
    selected_thresholds: dict[str, float] = {}
    selected_metrics = baseline_metrics
    candidates = _candidate_thresholds(records, allowed_metrics=allowed_metrics)
    for _step in range(int(max_gate_count)):
        best = None
        for name, values in candidates.items():
            if name in selected_thresholds:
                continue
            for value in values:
                trial_thresholds = {**selected_thresholds, name: float(value)}
                trial_metrics = evaluate_policy(
                    records,
                    trial_thresholds,
                    catastrophic_threshold_m=catastrophic_threshold_m,
                )
                if float(trial_metrics["translation_median_m"]) > float(
                    immutable_metrics["translation_median_m"]
                ) + float(max_median_regression_m):
                    continue
                item = (_objective(trial_metrics), name, float(value), trial_metrics)
                if best is None or item[:3] < best[:3]:
                    best = item
        if best is None or best[0] >= _objective(selected_metrics):
            break
        _score, name, value, selected_metrics = best
        selected_thresholds[str(name)] = float(value)
    return selected_thresholds, {
        "immutable_baseline_policy": immutable_metrics,
        "ungated_likelihood_policy": baseline_metrics,
        "calibrated_observability_policy": selected_metrics,
    }


def _promotion_candidate_grids(
    records: Sequence[dict[str, object]],
) -> tuple[list[float], list[int]]:
    deltas = np.asarray(
        [
            value
            for record in records
            if (value := _finite(record.get("likelihood_delta"))) is not None
            and value >= 0.0
        ],
        dtype=np.float64,
    )
    data_deltas: list[float] = []
    if len(deltas) >= 3:
        data_deltas = [
            float(np.percentile(deltas, quantile))
            for quantile in (10, 25, 50, 75, 90)
        ]
    delta_grid = sorted(
        set(float(value) for value in (*DEFAULT_LIKELIHOOD_DELTA_GRID, *data_deltas))
    )
    max_groups = max(
        (int(record.get("effective_group_count", 0)) for record in records),
        default=0,
    )
    group_grid = sorted(
        set(
            int(value)
            for value in DEFAULT_EFFECTIVE_GROUP_GRID
            if int(value) <= max_groups
        )
    )
    if max_groups > 0 and not group_grid:
        group_grid = [max_groups]
    return delta_grid, group_grid


def _policy_complexity_key(candidate: dict[str, object]) -> tuple[float, ...]:
    thresholds = candidate["observability_thresholds"]
    assert isinstance(thresholds, dict)
    metrics = candidate["metrics"]
    assert isinstance(metrics, dict)
    return (
        float(len(thresholds)),
        float(metrics["promotion_loss_count"]),
        -float(metrics["promotion_win_count"]),
        float(metrics["promotion_count"]),
        -float(candidate["min_likelihood_delta"]),
        -float(candidate["min_effective_group_count"]),
    )


def calibrate_promotion_policy(
    records: Sequence[dict[str, object]],
    *,
    max_gate_count: int = 1,
    catastrophic_threshold_m: float = 0.5,
    max_median_regression_m: float = 0.0,
    allowed_observability_metrics: Sequence[str] = (
        "translation_information_min_eigenvalue",
    ),
) -> dict[str, object]:
    """Select a small safety-first policy without consulting the audit split."""
    immutable_records = [{**record, "base_promoted": False} for record in records]
    immutable_metrics = evaluate_policy(
        immutable_records,
        {},
        catastrophic_threshold_m=catastrophic_threshold_m,
    )
    delta_grid, group_grid = _promotion_candidate_grids(records)
    candidates: list[dict[str, object]] = [
        {
            "policy_kind": "immutable_baseline",
            "min_likelihood_delta": 0.0,
            "min_effective_group_count": 0,
            "observability_thresholds": {},
            "metrics": immutable_metrics,
        }
    ]
    for delta in delta_grid:
        for group_count in group_grid:
            gated_records = _with_likelihood_gate(
                records,
                min_likelihood_delta=float(delta),
                min_effective_group_count=int(group_count),
            )
            thresholds, calibration = calibrate_thresholds(
                gated_records,
                max_gate_count=int(max_gate_count),
                catastrophic_threshold_m=float(catastrophic_threshold_m),
                max_median_regression_m=float(max_median_regression_m),
                allowed_metrics=allowed_observability_metrics,
            )
            metrics = calibration["calibrated_observability_policy"]
            assert isinstance(metrics, dict)
            if float(metrics["translation_median_m"]) > float(
                immutable_metrics["translation_median_m"]
            ) + float(max_median_regression_m):
                continue
            candidates.append(
                {
                    "policy_kind": "crossfit_likelihood_with_observability_veto",
                    "min_likelihood_delta": float(delta),
                    "min_effective_group_count": int(group_count),
                    "observability_thresholds": thresholds,
                    "metrics": metrics,
                }
            )
    selected = min(
        candidates,
        key=lambda candidate: (
            _objective(candidate["metrics"]),
            _policy_complexity_key(candidate),
        ),
    )
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            _objective(candidate["metrics"]),
            _policy_complexity_key(candidate),
        ),
    )
    return {
        "selected_policy": selected,
        "immutable_baseline_policy": immutable_metrics,
        "candidate_count": int(len(candidates)),
        "likelihood_delta_grid": delta_grid,
        "effective_group_count_grid": group_grid,
        "top_candidates": ranked[: min(20, len(ranked))],
    }


def evaluate_selected_promotion_policy(
    records: Sequence[dict[str, object]],
    policy: dict[str, object],
    *,
    catastrophic_threshold_m: float = 0.5,
) -> dict[str, object]:
    policy_kind = str(policy["policy_kind"])
    if policy_kind == "immutable_baseline":
        records = [{**record, "base_promoted": False} for record in records]
        return evaluate_policy(
            records,
            {},
            catastrophic_threshold_m=catastrophic_threshold_m,
        )
    thresholds = policy["observability_thresholds"]
    assert isinstance(thresholds, dict)
    if policy_kind == "legacy_likelihood_with_observability_veto":
        return evaluate_policy(
            records,
            {str(name): float(value) for name, value in thresholds.items()},
            catastrophic_threshold_m=catastrophic_threshold_m,
        )
    if policy_kind != "crossfit_likelihood_with_observability_veto":
        raise ValueError(f"unsupported promotion policy kind: {policy_kind}")
    return evaluate_policy(
        records,
        {str(name): float(value) for name, value in thresholds.items()},
        min_likelihood_delta=float(policy["min_likelihood_delta"]),
        min_effective_group_count=int(policy["min_effective_group_count"]),
        catastrophic_threshold_m=catastrophic_threshold_m,
    )


def _evaluator_cli(policy: dict[str, object]) -> list[str]:
    policy_kind = str(policy["policy_kind"])
    if policy_kind == "immutable_baseline":
        return []
    if policy_kind not in {
        "crossfit_likelihood_with_observability_veto",
        "legacy_likelihood_with_observability_veto",
    }:
        raise ValueError(f"unsupported promotion policy kind: {policy_kind}")
    output = ["--enable_grouped_crossfit_likelihood_fallback"]
    if policy_kind == "crossfit_likelihood_with_observability_veto":
        output.extend(
            [
                "--grouped_likelihood_min_mean_delta",
                str(float(policy["min_likelihood_delta"])),
                "--grouped_likelihood_min_effective_groups",
                str(int(policy["min_effective_group_count"])),
            ]
        )
    thresholds = policy["observability_thresholds"]
    assert isinstance(thresholds, dict)
    for name, value in sorted(thresholds.items()):
        formatted = (
            str(int(round(float(value))))
            if str(name) == "information_match_count"
            else str(float(value))
        )
        output.extend((OBSERVABILITY_CLI_FLAGS[str(name)], formatted))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_rows", required=True)
    parser.add_argument("--policy_key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration_split", default="validation")
    parser.add_argument("--audit_split", default="late_development")
    parser.add_argument("--max_gate_count", type=int, default=1)
    parser.add_argument("--catastrophic_threshold_m", type=float, default=0.5)
    parser.add_argument("--max_median_regression_m", type=float, default=0.0)
    parser.add_argument(
        "--allowed_observability_metrics",
        default="translation_information_min_eigenvalue",
        help=(
            "comma-separated physical diagnostics eligible for calibration; "
            "the selected names are frozen in the output artifact"
        ),
    )
    likelihood_mode = parser.add_mutually_exclusive_group()
    likelihood_mode.add_argument(
        "--calibrate_likelihood_gate",
        dest="calibrate_likelihood_gate",
        action="store_true",
    )
    likelihood_mode.add_argument(
        "--no_calibrate_likelihood_gate",
        dest="calibrate_likelihood_gate",
        action="store_false",
    )
    parser.set_defaults(calibrate_likelihood_gate=True)
    args = parser.parse_args()
    if int(args.max_gate_count) < 0:
        raise ValueError("max_gate_count must be non-negative")
    if float(args.catastrophic_threshold_m) <= 0.0:
        raise ValueError("catastrophic threshold must be positive")
    if float(args.max_median_regression_m) < 0.0:
        raise ValueError("maximum median regression must be non-negative")
    allowed_observability_metrics = _parse_observability_metrics(
        args.allowed_observability_metrics
    )

    rows_path = Path(args.pose_rows)
    payload = json.loads(rows_path.read_text())
    calibration_records = _records(
        _policy_rows(payload, str(args.calibration_split), str(args.policy_key))
    )
    if not calibration_records:
        raise ValueError("calibration split contains no cross-fit promotion rows")
    _validate_record_set(
        calibration_records, split_name=str(args.calibration_split)
    )
    audit_records = _records(
        _policy_rows(payload, str(args.audit_split), str(args.policy_key))
    )
    _validate_record_set(audit_records, split_name=str(args.audit_split))
    overlap = {str(record["query_id"]) for record in calibration_records}.intersection(
        str(record["query_id"]) for record in audit_records
    )
    if overlap:
        raise ValueError(
            "calibration and audit splits overlap: "
            f"{sorted(overlap)[:5]}"
        )
    if bool(args.calibrate_likelihood_gate):
        calibration = calibrate_promotion_policy(
            calibration_records,
            max_gate_count=int(args.max_gate_count),
            catastrophic_threshold_m=float(args.catastrophic_threshold_m),
            max_median_regression_m=float(args.max_median_regression_m),
            allowed_observability_metrics=allowed_observability_metrics,
        )
        selected_policy = calibration["selected_policy"]
        assert isinstance(selected_policy, dict)
        thresholds = selected_policy["observability_thresholds"]
        assert isinstance(thresholds, dict)
        audit = None
        if audit_records:
            audit = {
                "immutable_baseline_policy": evaluate_policy(
                    [
                        {**record, "base_promoted": False}
                        for record in audit_records
                    ],
                    {},
                    catastrophic_threshold_m=float(args.catastrophic_threshold_m),
                ),
                "embedded_evaluator_policy": evaluate_policy(
                    audit_records,
                    {},
                    catastrophic_threshold_m=float(args.catastrophic_threshold_m),
                ),
                "selected_calibrated_policy": evaluate_selected_promotion_policy(
                    audit_records,
                    selected_policy,
                    catastrophic_threshold_m=float(args.catastrophic_threshold_m),
                ),
            }
        stage = "pose_promotion_policy_calibration_v2"
    else:
        thresholds, calibration = calibrate_thresholds(
            calibration_records,
            max_gate_count=int(args.max_gate_count),
            catastrophic_threshold_m=float(args.catastrophic_threshold_m),
            max_median_regression_m=float(args.max_median_regression_m),
        )
        selected_policy = {
            "policy_kind": "legacy_likelihood_with_observability_veto",
            "observability_thresholds": thresholds,
        }
        audit = (
            None
            if not audit_records
            else evaluate_policy(
                audit_records,
                thresholds,
                catastrophic_threshold_m=float(args.catastrophic_threshold_m),
            )
        )
        stage = "pose_observability_gate_calibration_v1"
    result = {
        "stage": stage,
        "protocol": {
            "selection_data": str(args.calibration_split),
            "audit_data_used_for_selection": False,
            "gate_semantics": (
                "joint_crossfit_likelihood_and_observability_veto"
                if bool(args.calibrate_likelihood_gate)
                else "veto_existing_likelihood_promotions_only"
            ),
            "maximum_gate_count": int(args.max_gate_count),
            "catastrophic_threshold_m": float(args.catastrophic_threshold_m),
            "max_median_regression_m": float(args.max_median_regression_m),
            "allowed_observability_metrics": list(
                allowed_observability_metrics
            ),
        },
        "thresholds": thresholds,
        "selected_policy": selected_policy,
        "evaluator_cli": _evaluator_cli(selected_policy),
        "calibration": calibration,
        "audit": audit,
        "inputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "policy_key": str(args.policy_key),
            "calibration_split": str(args.calibration_split),
            "audit_split": str(args.audit_split),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
