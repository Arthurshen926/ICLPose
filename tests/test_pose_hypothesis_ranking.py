from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.pose_hypothesis_ranking import (
    HypothesisRankingGroup,
    LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES,
    MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES,
    POSE_HYPOTHESIS_FEATURE_NAMES,
    PoseHypothesisRanker,
    evaluate_pose_hypothesis_ranker,
    fit_pose_hypothesis_ranker,
    pose_hypothesis_features,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    HypothesisVerification,
    PoseHypothesisRecord,
)


def _record(strict: int, loose: int, residual: float) -> PoseHypothesisRecord:
    return PoseHypothesisRecord(
        fit_match_count_limit=32,
        fit_match_count=32,
        selection_mode="geometry_diverse",
        ransac_threshold_px=4.0,
        rng_seed_offset=0,
        solver_success=True,
        fit_inlier_count=loose + 4,
        verification=HypothesisVerification(
            verification_count=16,
            finite_count=16,
            positive_depth_count=16,
            positive_depth_ratio=1.0,
            strict_inlier_count=strict,
            loose_inlier_count=loose,
            strict_grid_cell_count=min(strict, 8),
            loose_grid_cell_count=min(loose, 12),
            soft_consensus=float(strict),
            clipped_median_residual_px=residual,
            depth_range_m=5.0,
        ),
    )


def _pose(translation: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(translation)
    return pose


def _group(index: int) -> HypothesisRankingGroup:
    records = (
        _record(14, 15, 0.5),
        _record(8, 11, 2.0),
        _record(2, 5, 8.0),
    )
    return HypothesisRankingGroup(
        query_id=f"query-{index:02d}",
        records=records,
        poses_w2c=(_pose(0.0), _pose(0.2), _pose(1.0)),
        translation_errors_m=(0.01 + index * 1e-4, 0.20, 1.0),
        rotation_errors_deg=(0.1, 0.8, 4.0),
    )


def test_pose_hypothesis_features_are_finite_and_gt_free() -> None:
    group = _group(0)

    indices, features = pose_hypothesis_features(group.records, group.poses_w2c)

    assert indices.tolist() == [0, 1, 2]
    assert features.shape[0] == 3
    assert features.shape[1] == len(POSE_HYPOTHESIS_FEATURE_NAMES)
    assert np.all(features[:, -len(MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES) :] == 0.0)
    assert np.all(np.isfinite(features))


def test_pairwise_pose_hypothesis_ranker_learns_heldout_order() -> None:
    groups = [_group(index) for index in range(12)]

    ranker, summary = fit_pose_hypothesis_ranker(
        groups,
        c_values=(0.03, 0.3),
        cross_validation_folds=3,
    )
    metrics = evaluate_pose_hypothesis_ranker(groups, ranker)
    restored = PoseHypothesisRanker.from_dict(ranker.to_dict())

    assert summary["fit"]["query_count_with_pairs"] == 12
    assert metrics["median_translation_m"] < 0.02
    assert all(row["chosen_hypothesis_index"] == 0 for row in metrics["rows"])
    assert restored.select(
        groups[0].records, groups[0].poses_w2c, (0, 1, 2)
    ) == 0


def test_legacy_pose_ranker_loads_with_zero_weight_measurement_features() -> None:
    payload = {
        "feature_names": list(LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES),
        "scale": [1.0] * len(LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES),
        "weights": [0.5] * len(LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES),
        "c_value": 0.1,
        "rotation_equivalent_m_per_deg": 0.02,
        "minimum_pair_gap_m": 0.005,
    }

    restored = PoseHypothesisRanker.from_dict(payload)

    assert len(restored.weights) == len(POSE_HYPOTHESIS_FEATURE_NAMES)
    assert restored.weights[-len(MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES) :] == (
        0.0,
    ) * len(MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES)
