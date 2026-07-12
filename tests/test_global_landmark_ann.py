from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.global_landmark_ann import (
    audit_ann_against_exact_candidates,
    collapse_unique_track_search,
)


def test_collapse_unique_track_search_removes_duplicate_prototypes() -> None:
    rows, tracks, prototypes, scores = collapse_unique_track_search(
        np.asarray([[1, 0, 2, 3]], dtype=np.int64),
        np.asarray([[0.95, 0.90, 0.80, 0.70]], dtype=np.float32),
        np.asarray([10, 10, 20, 30], dtype=np.int64),
        np.asarray([0, 1, 0, 0], dtype=np.int64),
        proposal_top_l=3,
    )
    np.testing.assert_array_equal(rows, [[1, 2, 3]])
    np.testing.assert_array_equal(tracks, [[10, 20, 30]])
    np.testing.assert_array_equal(prototypes, [[1, 0, 0]])
    np.testing.assert_allclose(scores, [[0.95, 0.80, 0.70]])


def test_ann_audit_conditions_retention_on_exact_coverage() -> None:
    metrics = audit_ann_against_exact_candidates(
        np.asarray([[1, 3], [4, 8], [9, 2]], dtype=np.int64),
        np.asarray([[1, 2], [4, 5], [7, 8]], dtype=np.int64),
        np.asarray([1, 5, 6], dtype=np.int64),
    )
    assert metrics["exact_correct_track_recall"] == 2.0 / 3.0
    assert metrics["ann_correct_track_recall"] == 1.0 / 3.0
    assert metrics["correct_track_retention_given_exact"] == 0.5
