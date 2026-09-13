import numpy as np
import pytest
from feature_extract.tools.vfm.audit_goal_maplet_candidate_pool import pool_decomposition


def test_joint_threshold_oracle_and_factorization():
    # Translation and rotation minima from different poses must never be combined.
    e = [[[.1, 20], [2, .1]], [[.3, 3], [.1, .5]], [[np.inf, np.inf], [np.nan, 0]]]
    r = pool_decomposition(e, [0, 0, 0])
    assert r['0.1m_1deg']['pool_recall_percent'] == pytest.approx(100 / 3)
    assert r['0.1m_1deg']['selected_recall_percent'] == 0
    for row in r.values():
        assert row['pool_recall_percent'] + row['pool_missing_percent'] == pytest.approx(100)
        assert row['selected_recall_percent'] + row['selection_loss_pp'] == pytest.approx(row['pool_recall_percent'])
        if row['conditional_selection_success_percent'] is not None:
            assert row['pool_recall_percent'] * row['conditional_selection_success_percent'] / 100 == pytest.approx(row['selected_recall_percent'])


def test_duplicate_invariance_and_retention_monotonicity():
    e = np.array([[[.2, 1]], [[3, 50]]])
    old = pool_decomposition(e, [0, 0])
    duplicate = pool_decomposition(np.repeat(e, 2, axis=1), [0, 0])
    assert old == duplicate
    enlarged = pool_decomposition(np.concatenate([e, np.zeros_like(e)], axis=1), [0, 0])
    for key in old:
        assert enlarged[key]['pool_recall_percent'] >= old[key]['pool_recall_percent']
        assert enlarged[key]['selected_recall_percent'] == old[key]['selected_recall_percent']


def test_invalid_candidate_index_rejected():
    with pytest.raises(ValueError):
        pool_decomposition([[[0, 0]]], [-1])
