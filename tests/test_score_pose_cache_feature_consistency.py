from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_refine.tools.score_pose_cache_feature_consistency import (
    compute_feature_consistency_metrics,
)


def test_compute_feature_consistency_metrics_uses_valid_mask_only():
    query = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    rendered = torch.tensor(
        [[[[1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[[[True, False], [True, True]]]])

    metrics = compute_feature_consistency_metrics(query, rendered, valid)

    assert metrics["valid_frac"] == 0.75
    assert metrics["cosine_mean"] == 1.0
    assert metrics["residual_mean"] == 0.0


def test_compute_feature_consistency_metrics_reports_empty_mask():
    query = torch.randn(1, 4, 2, 2)
    rendered = torch.randn(1, 4, 2, 2)
    valid = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    metrics = compute_feature_consistency_metrics(query, rendered, valid)

    assert metrics["valid_frac"] == 0.0
    assert metrics["cosine_mean"] == 0.0
    assert metrics["residual_mean"] == float("inf")
