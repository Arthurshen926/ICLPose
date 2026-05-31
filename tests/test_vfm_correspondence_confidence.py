import numpy as np

from feature_extract.vfm.correspondence_confidence import (
    CalibratedLogisticConfidence,
    confidence_metrics,
    label_match_row,
    select_confident_matches_coverage_preserving,
    vectorize_match_rows,
)


def test_label_match_row_separates_positive_negative_and_ambiguous_zone() -> None:
    strong = label_match_row({"patch_correct": True, "gt_reproj_error_stride": 1.4}, stride_positive=1.0)
    weak = label_match_row({"patch_correct": False, "gt_reproj_error_stride": 0.7}, stride_positive=1.0)
    ambiguous = label_match_row({"patch_correct": False, "gt_reproj_error_stride": 1.5}, stride_positive=1.0)
    hard_negative = label_match_row(
        {"patch_correct": False, "gt_reproj_error_stride": 3.0, "pnp_inlier": True},
        stride_positive=1.0,
    )

    assert strong.target == 1
    assert strong.ignore is False
    assert weak.target == 1
    assert ambiguous.target == 0
    assert ambiguous.ignore is True
    assert hard_negative.target == 0
    assert hard_negative.ignore is False
    assert hard_negative.hard_negative is True


def test_confidence_metrics_report_ranking_calibration_and_top_fraction() -> None:
    metrics = confidence_metrics(
        labels=np.asarray([1, 1, 0, 0], dtype=np.int64),
        scores=np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64),
        top_fraction=0.5,
        num_ece_bins=2,
    )

    assert metrics["positive_prior"] == 0.5
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["precision_at_top_fraction"] == 1.0
    assert metrics["recall_at_top_fraction"] == 1.0
    assert metrics["brier"] < 0.05
    assert metrics["ece"] >= 0.0


def test_vectorize_and_train_logistic_confidence_on_observable_features() -> None:
    rows = [
        {"similarity": 0.95, "similarity_margin": 0.30, "match_rank": 0, "landmark_variance": 0.01, "observation_count": 8, "gt_reproj_error_stride": 0.2},
        {"similarity": 0.90, "similarity_margin": 0.25, "match_rank": 1, "landmark_variance": 0.02, "observation_count": 7, "gt_reproj_error_stride": 0.4},
        {"similarity": 0.45, "similarity_margin": 0.02, "match_rank": 2, "landmark_variance": 0.90, "observation_count": 1, "gt_reproj_error_stride": 3.0},
        {"similarity": 0.40, "similarity_margin": 0.01, "match_rank": 3, "landmark_variance": 0.80, "observation_count": 1, "gt_reproj_error_stride": 4.0},
    ]
    matrix, labels, keep, names = vectorize_match_rows(rows, feature_set="descriptor_map")
    model = CalibratedLogisticConfidence(max_iter=300, learning_rate=0.2, l2=1e-4)

    model.fit(matrix[keep], labels[keep])
    scores = model.predict_proba(matrix)

    assert "similarity" in names
    assert scores[0] > scores[2]
    assert scores[1] > scores[3]


def test_coverage_preserving_selection_keeps_high_confidence_per_grid_cell() -> None:
    rows = [
        {"match_index": 0, "xy": [10.0, 10.0], "confidence": 0.1},
        {"match_index": 1, "xy": [20.0, 20.0], "confidence": 0.9},
        {"match_index": 2, "xy": [80.0, 20.0], "confidence": 0.8},
        {"match_index": 3, "xy": [80.0, 80.0], "confidence": 0.7},
    ]
    selected = select_confident_matches_coverage_preserving(
        rows,
        image_width=100,
        image_height=100,
        max_matches=3,
        grid_rows=2,
        grid_cols=2,
        per_cell=1,
    )

    assert [row["match_index"] for row in selected] == [1, 2, 3]
