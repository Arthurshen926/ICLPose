import numpy as np
import pytest

from feature_extract.tools.vfm.mine_coherent_hard_modes_from_global_proposals import (
    _candidate_probabilities,
)


def test_candidate_probabilities_preserve_fixed_null_mass() -> None:
    scores = np.asarray([[0.9, 0.8, -1.0], [0.7, 0.6, 0.5]], dtype=np.float64)
    valid = np.asarray([[True, True, False], [True, True, True]])
    probabilities = _candidate_probabilities(
        scores, valid, temperature=0.1, null_probability=0.07
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 0.93)
    assert probabilities[0, 2] == 0.0
    assert probabilities[0, 0] > probabilities[0, 1]


@pytest.mark.parametrize("temperature,null", [(0.0, 0.1), (0.1, 0.0), (0.1, 1.0)])
def test_candidate_probabilities_reject_invalid_calibration(
    temperature: float, null: float
) -> None:
    with pytest.raises(ValueError):
        _candidate_probabilities(
            np.ones((1, 2)),
            np.ones((1, 2), dtype=bool),
            temperature=temperature,
            null_probability=null,
        )
