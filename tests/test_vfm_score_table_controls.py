import numpy as np
import pytest

from feature_extract.vfm.controls import (
    mask_channels_by_utility,
    metadata_only_scores,
    pca_channel_projection,
    random_channel_projection,
    shuffle_features,
)
from feature_extract.vfm.protocols import ProtocolKind
from feature_extract.vfm.score_table import (
    ScoreRow,
    evaluate_score_table,
    validate_candidate_groups,
)


def test_score_table_groups_queries_and_reports_mean_metrics():
    rows = [
        ScoreRow("q1", "a", 0.9, 0.10, True, ProtocolKind.REAL_RETRIEVAL, "selected"),
        ScoreRow("q1", "b", 0.2, 0.70, False, ProtocolKind.REAL_RETRIEVAL, "selected"),
        ScoreRow("q2", "a", 0.4, 0.60, False, ProtocolKind.REAL_RETRIEVAL, "selected"),
        ScoreRow("q2", "b", 0.8, 0.20, True, ProtocolKind.REAL_RETRIEVAL, "selected"),
    ]

    validate_candidate_groups(rows, min_candidates=2)
    report = evaluate_score_table(rows)

    assert report.query_count == 2
    assert report.mean_pred_cost_m == pytest.approx(0.15)
    assert report.mean_oracle_gap_m == pytest.approx(0.0)
    assert report.mean_top1_acc == pytest.approx(1.0)
    assert report.protocol_kind == ProtocolKind.REAL_RETRIEVAL


def test_score_table_refuses_mixed_protocols_or_short_candidate_groups():
    rows = [
        ScoreRow("q1", "a", 0.9, 0.10, True, ProtocolKind.REAL_RETRIEVAL, "selected"),
        ScoreRow("q2", "a", 0.9, 0.10, True, ProtocolKind.CONTROLLED_LATTICE, "selected"),
    ]

    with pytest.raises(ValueError, match="protocol"):
        evaluate_score_table(rows)

    with pytest.raises(ValueError, match="candidate"):
        validate_candidate_groups(rows[:1], min_candidates=2)


def test_metadata_only_scores_use_whitelisted_fields():
    metadata = [
        {"retrieval_score": 0.3, "candidate_rank": 2},
        {"retrieval_score": 0.8, "candidate_rank": 1},
    ]

    scores = metadata_only_scores(metadata, weights={"retrieval_score": 1.0, "candidate_rank": -0.1})

    assert scores[1] > scores[0]
    with pytest.raises(ValueError, match="not whitelisted"):
        metadata_only_scores([{"oracle_cost": 0.0}], weights={"oracle_cost": 1.0})


def test_shuffle_and_utility_mask_controls_are_deterministic():
    features = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    shuffled = shuffle_features(features, axis=0, seed=7)
    shuffled_again = shuffle_features(features, axis=0, seed=7)

    assert not np.array_equal(shuffled, features)
    np.testing.assert_array_equal(shuffled, shuffled_again)

    channel_features = np.ones((4, 2, 2), dtype=np.float32)
    utility = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    high_removed = mask_channels_by_utility(channel_features, utility, fraction=0.5, remove="high")
    low_removed = mask_channels_by_utility(channel_features, utility, fraction=0.5, remove="low")

    assert high_removed[[0, 2]].sum() == 0.0
    assert low_removed[[1, 3]].sum() == 0.0


def test_random_and_pca_projection_keep_requested_channel_dim():
    features = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)

    random_projected = random_channel_projection(features, output_dim=2, seed=0)
    pca_projected = pca_channel_projection(features, output_dim=2)

    assert random_projected.shape == (2, 3, 2)
    assert pca_projected.shape == (2, 3, 2)
