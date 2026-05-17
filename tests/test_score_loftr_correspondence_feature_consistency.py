from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.tools.score_loftr_correspondence_feature_consistency import (
    binary_auc_from_scores,
    sample_feature_vectors_at_xy,
    summarize_labeled_residuals,
)


def test_binary_auc_from_scores_treats_lower_residual_as_better_for_inliers():
    labels = np.array([True, True, False, False])
    residuals = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    assert binary_auc_from_scores(labels, residuals, lower_score_is_positive=True) == 1.0
    assert binary_auc_from_scores(labels, residuals, lower_score_is_positive=False) == 0.0


def test_binary_auc_from_scores_handles_ties_as_half_credit():
    labels = np.array([True, False, True, False])
    residuals = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    assert binary_auc_from_scores(labels, residuals, lower_score_is_positive=True) == 0.5


def test_summarize_labeled_residuals_reports_inlier_outlier_gap_and_auc():
    residuals = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    inliers = np.array([True, True, False, False])

    summary = summarize_labeled_residuals(residuals, inliers)

    assert summary["num_points"] == 4
    assert summary["num_inliers"] == 2
    assert summary["num_outliers"] == 2
    assert summary["inlier_residual_mean"] < summary["outlier_residual_mean"]
    assert summary["inlier_lower_residual_auc"] == 1.0


def test_sample_feature_vectors_at_xy_scales_source_coordinates():
    feature = torch.zeros(1, 2, 2, 2)
    feature[0, :, 0, 0] = torch.tensor([1.0, 0.0])
    feature[0, :, 0, 1] = torch.tensor([0.0, 1.0])
    xy = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])
    valid = torch.ones(1, 2)

    vectors, inside = sample_feature_vectors_at_xy(
        feature,
        xy,
        valid,
        source_hw=(4, 4),
    )

    assert inside.tolist() == [[True, True]]
    assert torch.allclose(vectors[0, 0], torch.tensor([1.0, 0.0]), atol=1.0e-5)
    assert torch.allclose(vectors[0, 1], torch.tensor([0.0, 1.0]), atol=1.0e-5)
