from __future__ import annotations

import pytest

from feature_extract.tools.vfm.audit_robust_pose_hypothesis_selectors import (
    audit_pose_rows,
    select_hypothesis_by_immutable_statistic,
)


def _hypothesis(
    index: int,
    score: float,
    translation: float,
    *,
    strict_inliers: int,
) -> dict[str, object]:
    return {
        "hypothesis_index": index,
        "verified_for_final_selection": True,
        "fixed_posterior_log_likelihood_mean": score,
        "fixed_posterior_log_likelihood_median": score,
        "fixed_posterior_log_likelihood_trimmed_mean_10": score,
        "fixed_posterior_log_likelihood_worst_quartile_mean": score,
        "fixed_posterior_log_likelihood_lcb95": score,
        "fixed_posterior_spatial_median_of_means_2x2": score,
        "translation_m_TARGET_ONLY": translation,
        "rotation_deg_TARGET_ONLY": 0.5,
        "strict_inlier_count": strict_inliers,
    }


def test_selector_uses_stable_index_not_self_consistency_for_exact_ties() -> None:
    row = {
        "hypothesis_information_audit_with_TARGET_ONLY_errors": [
            _hypothesis(3, -1.0, 0.20, strict_inliers=2),
            _hypothesis(8, -1.0, 0.02, strict_inliers=100),
        ]
    }

    selected = select_hypothesis_by_immutable_statistic(
        row, "fixed_posterior_log_likelihood_mean"
    )

    assert selected is not None
    assert selected["hypothesis_index"] == 3


def test_audit_reports_target_metrics_only_after_selection() -> None:
    payload = {
        "validation": {
            "label": {
                "verified": [
                    {
                        "query_id": "q0",
                        "hypothesis_information_audit_with_TARGET_ONLY_errors": [
                            _hypothesis(0, -2.0, 0.02, strict_inliers=4),
                            _hypothesis(1, -1.0, 0.20, strict_inliers=12),
                        ],
                    },
                    {
                        "query_id": "q1",
                        "hypothesis_information_audit_with_TARGET_ONLY_errors": [
                            _hypothesis(0, -1.0, 0.08, strict_inliers=4),
                            _hypothesis(1, -2.0, 0.01, strict_inliers=12),
                        ],
                    },
                ]
            }
        }
    }

    result = audit_pose_rows(payload, splits=("validation",))
    mean = result["validation"]["selectors"]["mean"]

    assert result["validation"]["selector_uses_target_pose"] is False
    assert mean["summary"]["median_translation_m_TARGET_ONLY"] == pytest.approx(
        0.14
    )
    assert mean["summary"]["recall_10cm_5deg_TARGET_ONLY"] == 0.5
    assert mean["summary"]["median_selection_regret_m_TARGET_ONLY"] == pytest.approx(
        0.125
    )
