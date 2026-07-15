from feature_extract.tools.vfm.calibrate_pose_observability_gate import (
    _evaluator_cli,
    _likelihood_gate_passes,
    _parse_observability_metrics,
    calibrate_promotion_policy,
    calibrate_thresholds,
    evaluate_policy,
    evaluate_selected_promotion_policy,
)


def test_likelihood_replay_can_rescue_failed_immutable_baseline() -> None:
    record = {
        "baseline_success": False,
        "optional_success": True,
        "baseline_likelihood": None,
        "optional_likelihood": -3.2,
        "likelihood_delta": None,
        "effective_group_count": 16,
    }

    assert _likelihood_gate_passes(
        record,
        min_likelihood_delta=0.5,
        min_effective_group_count=8,
    )
    assert not _likelihood_gate_passes(
        {**record, "optional_likelihood": None},
        min_likelihood_delta=0.0,
        min_effective_group_count=8,
    )


def test_observability_metric_parser_rejects_unknown_and_duplicate_names() -> None:
    assert _parse_observability_metrics(
        "translation_information_min_eigenvalue,camera_depth_span_ratio"
    ) == (
        "translation_information_min_eigenvalue",
        "camera_depth_span_ratio",
    )
    for value in ("", "camera_depth_span_ratio,camera_depth_span_ratio", "unknown"):
        try:
            _parse_observability_metrics(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid observability metric set accepted: {value}")


def _record(index: int, *, baseline: float, optional: float, eigenvalue: float):
    return {
        "query_id": f"q{index}",
        "baseline_translation_m": baseline,
        "optional_translation_m": optional,
        "baseline_rotation_deg": None,
        "optional_rotation_deg": None,
        "base_promoted": True,
        "baseline_success": True,
        "optional_success": True,
        "likelihood_delta": 0.1,
        "effective_group_count": 16,
        "observability": {
            "information_match_count": 16,
            "translation_information_min_eigenvalue": eigenvalue,
            "translation_information_condition": 10.0,
            "joint_information_condition": 100.0,
            "bearing_max_angle_deg": 30.0,
            "camera_depth_span_ratio": 0.5,
            "xyz_second_singular_ratio": 0.2,
            "xyz_third_singular_ratio": 0.05,
        },
    }


def test_observability_calibration_vetoes_low_information_harmful_promotion() -> None:
    records = [
        _record(0, baseline=0.20, optional=1.00, eigenvalue=1.0),
        _record(1, baseline=0.20, optional=0.10, eigenvalue=10.0),
        _record(2, baseline=0.30, optional=0.15, eigenvalue=20.0),
        _record(3, baseline=0.10, optional=0.08, eigenvalue=30.0),
        _record(4, baseline=0.15, optional=0.09, eigenvalue=40.0),
    ]

    thresholds, metrics = calibrate_thresholds(records, max_gate_count=2)

    assert "translation_information_min_eigenvalue" in thresholds
    assert metrics["ungated_likelihood_policy"]["catastrophic_count"] == 1
    assert metrics["calibrated_observability_policy"]["catastrophic_count"] == 0
    assert metrics["calibrated_observability_policy"]["translation_p90_m"] < metrics[
        "ungated_likelihood_policy"
    ]["translation_p90_m"]


def test_observability_policy_treats_missing_gate_metric_as_abstain() -> None:
    record = _record(0, baseline=0.2, optional=0.1, eigenvalue=10.0)
    record["observability"].pop("bearing_max_angle_deg")

    result = evaluate_policy([record], {"bearing_max_angle_deg": 5.0})

    assert result["promotion_count"] == 0
    assert result["translation_median_m"] == 0.2


def test_joint_calibration_reestimates_likelihood_scale() -> None:
    records = [
        _record(0, baseline=0.10, optional=1.00, eigenvalue=10.0),
        _record(1, baseline=0.20, optional=0.05, eigenvalue=10.0),
        _record(2, baseline=0.30, optional=0.06, eigenvalue=10.0),
        _record(3, baseline=0.08, optional=0.07, eigenvalue=10.0),
        _record(4, baseline=0.15, optional=0.50, eigenvalue=10.0),
    ]
    for record, delta in zip(records, (0.03, 0.20, 0.30, 0.40, 0.01)):
        record["likelihood_delta"] = delta

    calibration = calibrate_promotion_policy(records, max_gate_count=1)
    selected = calibration["selected_policy"]

    assert selected["policy_kind"] == (
        "crossfit_likelihood_with_observability_veto"
    )
    assert selected["min_likelihood_delta"] > 0.03
    assert selected["metrics"]["new_catastrophic_promotion_count"] == 0
    assert selected["metrics"]["translation_median_m"] < calibration[
        "immutable_baseline_policy"
    ]["translation_median_m"]


def test_audit_evaluation_cannot_change_selected_policy() -> None:
    validation = [
        _record(0, baseline=0.20, optional=0.05, eigenvalue=20.0),
        _record(1, baseline=0.30, optional=0.06, eigenvalue=20.0),
        _record(2, baseline=0.10, optional=0.08, eigenvalue=20.0),
    ]
    selected = calibrate_promotion_policy(validation)["selected_policy"]
    frozen = dict(selected)
    late = [_record(3, baseline=0.05, optional=2.0, eigenvalue=20.0)]

    audit = evaluate_selected_promotion_policy(late, selected)

    assert selected == frozen
    assert audit["query_count"] == 1


def test_legacy_observability_policy_exports_replay_cli_without_joint_fields() -> None:
    policy = {
        "policy_kind": "legacy_likelihood_with_observability_veto",
        "observability_thresholds": {
            "translation_information_min_eigenvalue": 12.5,
        },
    }

    cli = _evaluator_cli(policy)

    assert cli == [
        "--enable_grouped_crossfit_likelihood_fallback",
        "--grouped_observability_min_translation_eigenvalue",
        "12.5",
    ]
    assert "--grouped_likelihood_min_mean_delta" not in cli


def test_legacy_observability_policy_reuses_embedded_likelihood_decision() -> None:
    harmful = _record(0, baseline=0.2, optional=1.0, eigenvalue=1.0)
    helpful = _record(1, baseline=0.2, optional=0.1, eigenvalue=20.0)
    policy = {
        "policy_kind": "legacy_likelihood_with_observability_veto",
        "observability_thresholds": {
            "translation_information_min_eigenvalue": 10.0,
        },
    }

    result = evaluate_selected_promotion_policy([harmful, helpful], policy)

    assert result["promotion_count"] == 1
    assert result["promotion_win_count"] == 1
    assert result["promotion_loss_count"] == 0
