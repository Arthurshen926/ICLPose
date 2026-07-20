from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance import (
    paired_rank,
    rank_metrics,
)


def test_raw_fulltrack_rank_metrics_use_common_explicit_coverage() -> None:
    labels = np.asarray([[False, True, False], [True, False, False]], dtype=bool)
    valid = np.asarray([[True, True, False], [True, False, False]], dtype=bool)
    baseline = np.asarray([[0.9, 0.8, np.nan], [0.7, np.nan, np.nan]], dtype=np.float32)
    raw = np.asarray([[0.1, 0.8, np.nan], [0.9, np.nan, np.nan]], dtype=np.float32)
    rows = np.asarray([True, True])
    metrics = rank_metrics(
        scores=raw,
        labels=labels,
        valid=valid,
        rows=rows,
    )
    paired = paired_rank(
        baseline_scores=baseline,
        probe_scores=raw,
        labels=labels,
        valid=valid,
        rows=rows,
    )
    assert metrics["positive_row_count"] == 2
    assert metrics["top1_positive_rate_given_positive"] == 1.0
    assert paired["rank_win_count"] == 1
    assert paired["rank_loss_count"] == 0
    assert np.array_equal(rows, np.asarray([True, True]))
