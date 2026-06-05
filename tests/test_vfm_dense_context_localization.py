import numpy as np

from feature_extract.vfm.dense_context_localization import build_selected_matches, select_top1_candidate_rows
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _toy_index() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11, 12], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [2.0, 0.0, 4.0]], dtype=np.float64),
        features=np.eye(3, dtype=np.float32),
        mean_variances=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        observation_counts=np.asarray([5, 6, 7], dtype=np.int64),
        observation_image_ids=(("a",), ("b",), ("c",)),
        reprojection_errors=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
    )


def test_select_top1_candidate_rows_ignores_invalid_candidates() -> None:
    top_indices = np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    scores = np.asarray([[0.1, 0.9, 0.2], [9.0, 0.3, 0.4]], dtype=np.float32)
    valid = np.asarray([[True, True, True], [False, True, True]], dtype=bool)

    token_rows, landmark_rows, selected_scores = select_top1_candidate_rows(top_indices, scores, valid)

    assert token_rows.tolist() == [0, 1]
    assert landmark_rows.tolist() == [1, 2]
    assert np.allclose(selected_scores, [0.9, 0.4])


def test_build_selected_matches_sorts_by_score_and_preserves_geometry() -> None:
    token_rows = np.asarray([0, 1], dtype=np.int64)
    token_indices = np.asarray([5, 7], dtype=np.int64)
    landmark_rows = np.asarray([1, 2], dtype=np.int64)
    scores = np.asarray([0.3, 0.9], dtype=np.float32)
    centers = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)

    matches = build_selected_matches(
        token_rows,
        token_indices,
        landmark_rows,
        scores,
        _toy_index(),
        centers,
        source="test_source",
        max_matches=10,
    )

    assert [match.track_id for match in matches] == [12, 11]
    assert [match.token_index for match in matches] == [7, 5]
    assert matches[0].xy.tolist() == [30.0, 40.0]
    assert matches[0].source == "test_source"
