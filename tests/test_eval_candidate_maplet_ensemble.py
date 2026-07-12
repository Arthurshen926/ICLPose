import numpy as np

from feature_extract.tools.vfm.eval_candidate_maplet_ensemble import (
    _baseline_assignment_scores,
)
from feature_extract.vfm.localization.local_assignment_probe import (
    UniqueTrackCandidateSet,
)


def _candidates() -> UniqueTrackCandidateSet:
    return UniqueTrackCandidateSet(
        bank_row_indices=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        track_ids=np.asarray([[10, 11], [10, 12]], dtype=np.int64),
        prototype_ids=np.asarray([[100, 101], [102, 103]], dtype=np.int64),
        coarse_scores=np.asarray([[0.9, 0.8], [0.85, 0.7]], dtype=np.float32),
    )


def test_frozen_global_baseline_changes_pose_scores_and_enforces_unique_tracks():
    candidates = _candidates()
    scores = np.asarray([[0.9, 0.8], [0.85, 0.7]], dtype=np.float32)
    query_ids = np.asarray(["query.png", "query.png"])
    valid = np.ones_like(scores, dtype=bool)

    row_scores, row_columns = _baseline_assignment_scores(
        candidates=candidates,
        scores=scores,
        query_ids=query_ids,
        valid_edges=valid,
        formal_global_assignment=False,
    )
    global_scores, global_columns = _baseline_assignment_scores(
        candidates=candidates,
        scores=scores,
        query_ids=query_ids,
        valid_edges=valid,
        formal_global_assignment=True,
    )

    assert np.array_equal(row_scores, scores)
    assert np.array_equal(row_columns, np.asarray([0, 0]))
    selected_tracks = candidates.track_ids[np.arange(2), global_columns]
    assert len(np.unique(selected_tracks)) == 2
    assert np.sum(np.isfinite(global_scores)) == 2
    assert not np.array_equal(global_scores, row_scores)
