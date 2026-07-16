from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_oracle_maplet_hypothesis_prior import (
    _maplet_coverage,
    _query_maplet_members,
)


def test_query_maplet_members_uses_only_gt_near_candidates() -> None:
    proposals = {
        "query_ids": np.asarray(["q", "q", "other"]),
        "candidate_track_ids": np.asarray([[10, 20], [30, 40], [50, 60]]),
        "candidate_gt_residuals_px": np.asarray(
            [[1.0, 3.0], [1.5, 8.0], [0.5, 9.0]], dtype=np.float32
        ),
    }
    members = _query_maplet_members(
        proposals,
        np.asarray([0, 1]),
        {10: np.asarray([11, 12]), 30: np.asarray([31, -1])},
        threshold_px=2.0,
    )

    assert members == {"q": {10, 11, 12, 30, 31}}


def test_maplet_coverage_ignores_padding() -> None:
    assert _maplet_coverage(np.asarray([10, 20, -1, -1]), {10, 30}) == 0.5
    assert _maplet_coverage(np.asarray([-1, -1]), {10}) == 0.0
