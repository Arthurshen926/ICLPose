from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.measurement_v1.candidate_evidence_v3_audit import (
    marginalize_candidate_views,
)


def test_view_marginalization_preserves_valid_invalid_and_missing_mass() -> None:
    result = marginalize_candidate_views(
        local_log_probabilities=np.log(
            np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
        ),
        support_view_probabilities=np.asarray([0.6, 0.3]),
        dustbin_probabilities=np.asarray([0.25, 0.5]),
    )
    assert result["valid_probability"] == pytest.approx(0.6)
    assert result["invalid_probability"] == pytest.approx(0.3)
    assert result["missing_probability"] == pytest.approx(0.1)
    probability = np.exp(result["conditional_local_log_probabilities"])
    assert probability.sum() == pytest.approx(1.0)
    assert probability.tolist() == pytest.approx([0.625, 0.375])


def test_view_marginalization_rejects_renormalized_available_view_mass() -> None:
    with pytest.raises(ValueError, match="probability mass"):
        marginalize_candidate_views(
            local_log_probabilities=np.log(np.asarray([[0.5, 0.5], [0.5, 0.5]])),
            support_view_probabilities=np.asarray([0.7, 0.5]),
            dustbin_probabilities=np.asarray([0.0, 0.0]),
        )
