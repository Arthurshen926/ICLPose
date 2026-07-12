from __future__ import annotations

from feature_extract.vfm.localization.pairwise_pose_promotion import (
    PAIRWISE_FEATURE_NAMES,
    PairwisePosePromotionGate,
    _is_beneficial,
)


def test_pairwise_target_requires_significant_safe_improvement() -> None:
    baseline = {"success": True, "translation_m": 0.20, "rotation_deg": 0.40}
    assert _is_beneficial(
        baseline, {"success": True, "translation_m": 0.17, "rotation_deg": 0.44}
    )
    assert not _is_beneficial(
        baseline, {"success": True, "translation_m": 0.17, "rotation_deg": 0.60}
    )


def test_pairwise_gate_schema_round_trip_payload() -> None:
    count = len(PAIRWISE_FEATURE_NAMES)
    gate = PairwisePosePromotionGate(
        feature_names=PAIRWISE_FEATURE_NAMES,
        feature_mean=tuple([0.0] * count),
        feature_scale=tuple([1.0] * count),
        coefficients=tuple([0.0] * count),
        intercept=0.0,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        promotion_threshold=0.8,
        target_precision=0.8,
    )
    assert gate.to_dict()["format"] == "pairwise_pose_promotion_gate_v1"
