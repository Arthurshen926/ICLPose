import numpy as np

from feature_extract.vfm.dense_context_oracle_metrics import score_metrics_from_matrices


def test_score_metrics_from_matrices_reports_ranking_quality() -> None:
    labels = np.asarray([[False, True], [True, False]], dtype=bool)
    scores = np.asarray([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)

    metrics = score_metrics_from_matrices(labels, scores)

    assert metrics["candidate_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["mean_best_positive_rank"] == 1.0
    assert metrics["auroc"] == 1.0


def test_score_metrics_from_matrices_handles_missing_positive_tokens() -> None:
    labels = np.asarray([[False, False], [True, False]], dtype=bool)
    scores = np.asarray([[0.8, 0.7], [0.1, 0.2]], dtype=np.float32)

    metrics = score_metrics_from_matrices(labels, scores)

    assert metrics["top1_accuracy"] == 0.0
    assert metrics["mean_best_positive_rank"] == 2.0
    assert metrics["token_count"] == 2
    assert metrics["positive_token_count"] == 1


def test_score_metrics_from_matrices_excludes_invalid_candidates() -> None:
    labels = np.asarray([[True, False], [False, True]], dtype=bool)
    scores = np.asarray([[0.1, 99.0], [0.3, 0.2]], dtype=np.float32)
    valid = np.asarray([[True, False], [True, True]], dtype=bool)

    metrics = score_metrics_from_matrices(labels, scores, valid_mask=valid)

    assert metrics["candidate_count"] == 3
    assert metrics["positive_count"] == 2
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["mean_best_positive_rank"] == 1.5
