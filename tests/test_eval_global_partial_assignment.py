from __future__ import annotations

import json

import pytest

from feature_extract.tools.vfm.eval_global_partial_assignment import (
    _has_finite_pose_errors,
    _load_frozen_baseline_policy,
    _relative_pose_risk,
    _validate_frozen_baseline_pose,
)
from feature_extract.vfm.artifacts import file_sha256_short


def _pose(median: float | None, p90: float | None, rotation: float | None):
    return {
        "median_translation_m_success": median,
        "p90_translation_m_success": p90,
        "median_rotation_deg_success": rotation,
    }


def test_relative_pose_risk_marks_failed_pnp_without_crashing() -> None:
    risk = _relative_pose_risk(
        _pose(None, None, None),
        _pose(0.3, 0.7, 0.4),
    )

    assert risk["valid"] is False
    assert risk["failure_reason"] == "missing_pose_error_metric"
    assert risk["worst_error_ratio"] is None
    assert not _has_finite_pose_errors(_pose(None, None, None))


def test_relative_pose_risk_reports_finite_ratios() -> None:
    pose = _pose(0.2, 0.8, 0.5)
    risk = _relative_pose_risk(pose, _pose(0.4, 0.8, 0.4))

    assert risk["valid"] is True
    assert risk["median_translation_m_success"] == 0.5
    assert risk["p90_translation_m_success"] == 1.0
    assert risk["median_rotation_deg_success"] == 1.25
    assert risk["worst_error_ratio"] == 1.25
    assert _has_finite_pose_errors(pose)


def _complete_pose() -> dict[str, float]:
    return {
        "query_count": 21,
        "success_count": 21,
        "success_rate": 1.0,
        "median_translation_m_success": 0.2,
        "p90_translation_m_success": 0.6,
        "median_rotation_deg_success": 0.4,
        "recall_25cm_2deg": 0.5,
        "recall_10cm_5deg": 0.2,
        "recall_5cm_5deg": 0.1,
    }


def test_frozen_baseline_summary_binds_inputs_and_pose(tmp_path) -> None:
    inputs = {}
    paths = {}
    for key in ("proposals", "candidate", "bank", "split"):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(key.encode())
        paths[key] = path
    inputs.update(
        {
            "proposals_sha256": file_sha256_short(paths["proposals"]),
            "candidate_artifact_sha256": file_sha256_short(paths["candidate"]),
            "projected_landmark_bank_sha256": file_sha256_short(paths["bank"]),
            "split_json_sha256": file_sha256_short(paths["split"]),
            "baseline_score_key": "strategy__baseline",
        }
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "stage": "whole_image_global_partial_assignment_audit",
                "protocol": {"policy_selected_on_validation_only": True},
                "inputs": inputs,
                "baseline": {
                    "frozen_validation_policy": {
                        "policy_key": "baseline_global_max128_score_topk",
                        "max_matches": 128,
                        "selection_mode": "score_topk",
                        "pose": _complete_pose(),
                    }
                },
            }
        )
    )

    source = _load_frozen_baseline_policy(
        summary_path,
        proposals_path=paths["proposals"],
        candidate_path=paths["candidate"],
        bank_path=paths["bank"],
        split_path=paths["split"],
        baseline_score_key="strategy__baseline",
    )
    _validate_frozen_baseline_pose(_complete_pose(), source)
    assert source["max_matches"] == 128

    paths["split"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="different evaluation inputs"):
        _load_frozen_baseline_policy(
            summary_path,
            proposals_path=paths["proposals"],
            candidate_path=paths["candidate"],
            bank_path=paths["bank"],
            split_path=paths["split"],
            baseline_score_key="strategy__baseline",
        )


def test_frozen_baseline_pose_rejects_metric_drift() -> None:
    actual = _complete_pose()
    source = {"pose": _complete_pose()}
    actual["median_translation_m_success"] += 1e-6

    with pytest.raises(ValueError, match="baseline replay differs"):
        _validate_frozen_baseline_pose(actual, source)
