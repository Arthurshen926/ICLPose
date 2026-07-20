from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_radio_intermediate_fixed_candidate_rerank import (
    paired_rank_audit,
    rank_metrics,
    top_and_first_positive_rank,
)


def test_raw_rank_audit_reports_rank2_rescue_and_harm() -> None:
    labels = np.asarray([[False, True, False], [True, False, False]], dtype=bool)
    valid = np.ones_like(labels, dtype=bool)
    baseline = np.asarray([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]], dtype=np.float32)
    probe = np.asarray([[0.1, 0.95, 0.2], [0.1, 0.8, 0.9]], dtype=np.float32)
    top, rank = top_and_first_positive_rank(baseline, labels, valid)
    assert top.tolist() == [0, 0]
    assert rank.tolist() == [2, 1]
    row_mask = np.asarray([True, True])
    metrics = rank_metrics(
        scores=probe,
        labels=labels,
        valid=valid,
        row_mask=row_mask,
    )
    assert metrics["top1_positive_rate_given_positive"] == 0.5
    paired = paired_rank_audit(
        baseline_scores=baseline,
        probe_scores=probe,
        labels=labels,
        valid=valid,
        row_mask=row_mask,
    )
    assert paired["rank_win_count"] == 1
    assert paired["rank_loss_count"] == 1
    assert paired["top1_rescue_count"] == 1
    assert paired["top1_harm_count"] == 1
    assert np.array_equal(row_mask, np.asarray([True, True]))
