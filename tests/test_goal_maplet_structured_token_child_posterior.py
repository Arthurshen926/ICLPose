from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.structured_token_child_posterior import (
    structured_token_child_probabilities,
)


def test_structured_reweighting_preserves_token_mass_and_candidate_support():
    rows = np.asarray([[0, 1], [0, 1], [2, 3], [2, 3]])
    probability = np.asarray([[0.3, 0.2], [0.1, 0.4], [0.2, 0.1], [0.1, 0.2]])
    result = structured_token_child_probabilities(
        rows, probability,
        child_parent_rows=np.asarray([0, 1, 0, 1]),
        child_support_rows=np.asarray([0, 1, 0, 1]),
        height=2, width=2,
    )
    np.testing.assert_allclose(result.sum(1), probability.sum(1), atol=1e-7)
    np.testing.assert_array_equal(result > 0.0, probability > 0.0)


def test_neighbor_support_consistency_increases_consistent_candidate_share():
    # Child 0 and child 2 share one connected physical support even though
    # they straddle a parent seam.  The bottom tokens therefore reinforce the
    # top candidate 0, while the unrelated candidate 1 receives no support.
    rows = np.asarray([[0, 1], [0, 1], [2, 3], [2, 3]])
    probability = np.asarray([[0.25, 0.25], [0.25, 0.25], [0.45, 0.05], [0.45, 0.05]])
    result = structured_token_child_probabilities(
        rows, probability,
        child_parent_rows=np.asarray([0, 1, 2, 3]),
        child_support_rows=np.asarray([7, 8, 7, 9]),
        height=2, width=2, connected_support_weight=3.0, parent_weight=0.0,
    )
    assert result[0, 0] > probability[0, 0]
    assert result[0, 1] < probability[0, 1]


def test_zero_weights_are_identity_and_invalid_duplicates_fail_closed():
    rows = np.asarray([[0, 1], [0, 1], [1, 0], [1, 0]])
    probability = np.asarray([[0.3, 0.2]] * 4)
    result = structured_token_child_probabilities(
        rows, probability, np.asarray([0, 1]), np.asarray([0, 1]),
        height=2, width=2, connected_support_weight=0.0, parent_weight=0.0,
    )
    np.testing.assert_allclose(result, probability, atol=1e-7)
    bad = rows.copy()
    bad[0] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        structured_token_child_probabilities(
            bad, probability, np.asarray([0, 1]), np.asarray([0, 1]),
            height=2, width=2,
        )
