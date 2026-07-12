import copy

import numpy as np

from feature_extract.vfm.localization.pairwise_pose_promotion_v2 import (
    _validation_threshold,
    absolute_pose_evidence,
    build_pairwise_examples,
    select_rows,
)


def _row(query_id: str, translation: float, rotation: float, inliers: int = 30):
    return {
        "query_id": query_id,
        "success": True,
        "match_count": 64,
        "inlier_count": inliers,
        "inlier_ratio": inliers / 64.0,
        "all_grid_4x4_occupancy_frac": 0.75,
        "inlier_grid_4x4_occupancy_frac": 0.5,
        "pnp_reproj_inlier_median_px": 2.0,
        "pnp_reproj_inlier_p90_px": 4.0,
        "selection_confidence_min": 0.2,
        "selection_confidence_median": 0.5,
        "translation_m": translation,
        "rotation_deg": rotation,
    }


def test_absolute_evidence_does_not_read_pose_error_targets():
    row = _row("q", 0.1, 0.2)
    changed = dict(row, translation_m=100.0, rotation_deg=180.0)
    assert np.array_equal(absolute_pose_evidence(row), absolute_pose_evidence(changed))


def test_abstention_returns_immutable_baseline_values():
    baseline = [_row("q", 0.1, 0.2)]
    optional = [_row("q", 0.05, 0.1)]
    examples = build_pairwise_examples(baseline, optional)
    before = copy.deepcopy(baseline[0])
    selected, report = select_rows(
        examples, np.asarray([0.9]), threshold=float("inf")
    )
    assert selected == [before]
    assert baseline[0] == before
    assert report["promotion_count"] == 0


def test_validation_policy_can_choose_exact_fallback_when_optional_is_harmful():
    baseline = [_row(f"q{i}", 0.1 + i * 0.01, 0.2) for i in range(5)]
    optional = [_row(f"q{i}", 1.0 + i * 0.1, 2.0) for i in range(5)]
    examples = build_pairwise_examples(baseline, optional)
    threshold, selected, report = _validation_threshold(
        examples, np.linspace(0.1, 0.9, len(examples))
    )
    assert threshold > 0.9
    assert selected == baseline
    assert report["promotion"]["promotion_count"] == 0
    assert report["pose_gate"]["passes"]
