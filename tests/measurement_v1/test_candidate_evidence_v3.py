from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.measurement_v1.candidate_evidence_v3 import (
    _rank_top_m,
    _validate_probability_contract,
)


def test_top_m_keeps_joint_probability_mass_without_renormalizing() -> None:
    probabilities = np.asarray([[0.40, 0.20, 0.10, 0.05]], dtype=np.float32)
    valid = np.ones_like(probabilities, dtype=bool)
    selected, selected_valid = _rank_top_m(probabilities, valid, top_m=2)
    retained = np.take_along_axis(probabilities, selected, axis=1).sum(axis=1)
    dustbin = np.asarray([0.25], dtype=np.float32)
    omitted = probabilities.sum(axis=1) - retained
    unknown = dustbin + omitted
    assert selected.tolist() == [[0, 1]]
    assert selected_valid.tolist() == [[True, True]]
    assert retained[0] == pytest.approx(0.60)
    assert unknown[0] == pytest.approx(0.40)
    assert retained[0] + unknown[0] == pytest.approx(1.0)


def test_joint_probability_contract_rejects_per_candidate_dustbin() -> None:
    candidates = np.asarray([[0.4, 0.3]], dtype=np.float32)
    valid = np.ones_like(candidates, dtype=bool)
    with pytest.raises(ValueError, match="constant"):
        _validate_probability_contract(
            candidates,
            np.asarray([[0.3, 0.2]], dtype=np.float32),
            valid,
        )


def test_joint_probability_contract_rejects_missing_mass() -> None:
    candidates = np.asarray([[0.4, 0.3]], dtype=np.float32)
    valid = np.ones_like(candidates, dtype=bool)
    with pytest.raises(ValueError, match="joint softmax"):
        _validate_probability_contract(
            candidates,
            np.asarray([[0.1, 0.1]], dtype=np.float32),
            valid,
        )
