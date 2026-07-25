from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_p1_rgb_radio_bridge_crossfit import (
    _BASELINE_WEIGHTS,
    _bridge_gate,
    _candidate_weight_grid,
    _feature_filename,
    _load_training_feature,
    _write_training_feature,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_bridge import (
    CandidatePoseRGBSpatialBridgeQueryEvidence,
)


def _evidence() -> CandidatePoseRGBSpatialBridgeQueryEvidence:
    return CandidatePoseRGBSpatialBridgeQueryEvidence(
        query_id="query/a.png",
        spatial_candidate_llrs=np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]]),
        control_spatial_candidate_llrs=np.zeros((2, 1, 2)),
        identity_candidate_llrs=np.asarray([[0.25, -0.25]]),
        control_identity_candidate_llrs=np.zeros((1, 2)),
        candidate_probabilities=np.asarray([[0.6, 0.3]]),
        null_probabilities=np.asarray([0.1]),
    )


def test_bridge_profile_grid_contains_the_exact_rgb_baseline_once() -> None:
    profiles = _candidate_weight_grid()
    assert profiles.count(_BASELINE_WEIGHTS) == 1
    assert len(profiles) > 20
    assert all(profile.spatial_weight >= 0.0 for profile in profiles)
    assert all(profile.identity_weight >= 0.0 for profile in profiles)


def test_training_feature_is_explicitly_runtime_ineligible(tmp_path) -> None:
    evidence = _evidence()
    lineage = {"layout_sha256": "layout", "candidate_count": 2}
    path = tmp_path / _feature_filename(evidence.query_id)
    _write_training_feature(
        path=path,
        evidence=evidence,
        source_point_ids=np.asarray([9]),
        lineage=lineage,
    )
    loaded = _load_training_feature(
        path=path,
        expected_query_id=evidence.query_id,
        expected_lineage=lineage,
    )
    assert loaded.query_id == evidence.query_id
    assert loaded.spatial_candidate_llrs == pytest.approx(evidence.spatial_candidate_llrs)
    with pytest.raises(ValueError, match="contract"):
        _load_training_feature(
            path=path,
            expected_query_id=evidence.query_id,
            expected_lineage={"layout_sha256": "other"},
        )


def test_bridge_gate_rejects_a_better_mean_when_tail_and_pairing_fail() -> None:
    bridge = {
        "normal": {
            "mean_correct_minus_hardest_wrong": 0.3,
            "correct_win_fraction": 0.7,
            "catastrophic_gap_count": 2.0,
        },
        "normal_minus_control_mean_gap": 0.2,
    }
    baseline = {"normal": {"catastrophic_gap_count": 1.0}}
    gate = _bridge_gate(
        bridge_summary=bridge,
        baseline_summary=baseline,
        paired={"win_count": 3.0, "loss_count": 4.0},
        minimum_normal_gap=0.05,
        minimum_win_fraction=0.55,
        minimum_visual_gap_delta=0.05,
    )
    assert gate["passed"] is False
    assert gate["checks"]["catastrophic_tail_not_worse_than_rgb_baseline"] is False
    assert gate["checks"]["paired_rgb_baseline_wins_exceed_losses"] is False
