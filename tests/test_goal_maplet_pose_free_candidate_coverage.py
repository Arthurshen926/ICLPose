from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_pose_free_candidate_coverage import (
    _coverage_rows,
)


def test_candidate_coverage_is_joint_and_budget_monotone():
    translation = np.asarray([[3.0, 0.4], [1.5, 0.7]], dtype=np.float64)
    rotation = np.asarray([[1.0, 4.0], [30.0, 20.0]], dtype=np.float64)
    valid = np.ones_like(translation, dtype=bool)
    rows = _coverage_rows(translation, rotation, valid, (1, 2))
    assert rows[0]["strict_0_5m_5deg"]["hits"] == 0
    assert rows[1]["strict_0_5m_5deg"]["hits"] == 1
    assert rows[0]["region_2m_45deg"]["hits"] == 1
    assert rows[1]["region_2m_45deg"]["hits"] == 2


def test_candidate_coverage_respects_invalid_padding():
    translation = np.asarray([[0.1, 0.1]], dtype=np.float64)
    rotation = np.asarray([[1.0, 1.0]], dtype=np.float64)
    valid = np.asarray([[False, True]])
    rows = _coverage_rows(translation, rotation, valid, (1, 2))
    assert rows[0]["strict_0_5m_5deg"]["hits"] == 0
    assert rows[1]["strict_0_5m_5deg"]["hits"] == 1
