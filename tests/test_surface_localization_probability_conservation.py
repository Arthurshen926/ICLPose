from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.surface_localization import (
    _probability_conserving_top_l,
    _stable_top_indices,
)


def test_top_l_moves_omitted_candidate_mass_to_null():
    indices, _logits, probabilities, null, valid = _probability_conserving_top_l(
        np.asarray([[3.0, 2.0, 1.0]], dtype=np.float64),
        np.ones((1, 3), dtype=bool),
        keep=1,
        null_logits=0.0,
    )
    assert indices.tolist() == [[0]]
    assert valid.tolist() == [[True]]
    assert probabilities[0, 0] < 1.0
    assert np.isclose(float(probabilities.sum()) + float(null[0]), 1.0)


def test_invalid_rows_never_receive_candidate_probability():
    _indices, _logits, probabilities, null, valid = _probability_conserving_top_l(
        np.asarray([[2.0, -np.inf]], dtype=np.float64),
        np.asarray([[True, False]], dtype=bool),
        keep=2,
        null_logits=0.0,
    )
    assert valid.tolist() == [[True, False]]
    assert probabilities[0, 1] == 0.0
    assert np.isclose(float(probabilities.sum()) + float(null[0]), 1.0)


def test_stable_top_indices_break_equal_score_ties_by_metric_id():
    result = _stable_top_indices(
        np.asarray([[1.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        np.asarray([40, 10, 20, 5], dtype=np.int64),
        keep=3,
    )
    assert result.tolist() == [[1, 2, 0]]
