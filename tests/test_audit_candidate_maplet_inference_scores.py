from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_maplet_inference_scores import (
    _identity_metrics,
)


def test_external_identity_audit_reports_rescue_harm_and_null_nll() -> None:
    labels = np.asarray(
        [[True, False], [False, True], [False, False]], dtype=bool
    )
    valid = np.ones_like(labels)
    baseline = np.asarray(
        [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=np.float64
    )
    candidate = np.asarray(
        [[0.6, 0.1], [0.1, 0.7], [0.1, 0.1]], dtype=np.float64
    )
    result = _identity_metrics(
        candidate,
        labels=labels,
        valid=valid,
        row_mask=np.ones((3,), dtype=bool),
        baseline_probabilities=baseline,
        null_probabilities=np.asarray([0.3, 0.2, 0.8]),
    )

    assert result["conditional_top1_accuracy_mappable"] == pytest.approx(1.0)
    assert result["transition_vs_baseline"]["rescued_count"] == 1
    assert result["transition_vs_baseline"]["harmed_count"] == 0
    assert result["transition_vs_baseline"]["candidate_correct_count"] == 2
    assert result["target_nll"] == pytest.approx(
        float(np.mean(-np.log([0.6, 0.7, 0.8])))
    )


def test_external_identity_audit_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _identity_metrics(
            np.asarray([[np.nan, 0.5]]),
            labels=np.asarray([[True, False]]),
            valid=np.asarray([[True, True]]),
            row_mask=np.asarray([True]),
            baseline_probabilities=np.asarray([[0.5, 0.5]]),
            null_probabilities=None,
        )
