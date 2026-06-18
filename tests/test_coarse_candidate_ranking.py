from __future__ import annotations

import numpy as np

from feature_extract.vfm.coarse_candidate_ranking import (
    annotate_keypoint_matches_with_coarse_candidate_ranker,
    label_coarse_candidate_row,
    vectorize_coarse_candidate_rows,
    vectorize_keypoint_matches,
)
from feature_extract.vfm.correspondence_confidence import CalibratedLogisticConfidence
from feature_extract.vfm.rendered_keypoint_matching import KeypointFeatureMatch


def test_label_coarse_candidate_row_uses_patch_positive_and_ignores_weak_band() -> None:
    positive = label_coarse_candidate_row({"patch_correct": True, "gt_reproj_error_stride": 3.0})
    weak = label_coarse_candidate_row({"patch_correct": False, "gt_reproj_error_stride": 1.5})
    negative = label_coarse_candidate_row({"patch_correct": False, "gt_reproj_error_stride": 3.0})

    assert positive.target == 1
    assert positive.ignore is False
    assert weak.target == 0
    assert weak.ignore is True
    assert negative.target == 0
    assert negative.ignore is False


def test_vectorize_coarse_candidate_rows_includes_rank_and_mutual_metadata() -> None:
    rows = [
        {
            "similarity": 0.8,
            "similarity_margin": 0.1,
            "confidence": 0.3,
            "coarse_rank": 2,
            "coarse_score": 0.8,
            "coarse_score_gap": 0.2,
            "mutual_rank": 4,
            "cell_delta_x": -1,
            "cell_delta_y": 2,
            "patch_correct": True,
        }
    ]

    features, labels, keep, names = vectorize_coarse_candidate_rows(rows, feature_set="coarse_local")

    assert features.shape == (1, len(names))
    assert labels.tolist() == [1]
    assert keep.tolist() == [True]
    assert "coarse_rank_score" in names
    assert "mutual_is_top1" in names
    assert "cell_delta_chebyshev" in names


def test_vectorize_keypoint_matches_matches_row_feature_schema() -> None:
    match = KeypointFeatureMatch(
        query_index=3,
        render_index=7,
        query_xy=np.asarray([12.0, 20.0]),
        render_xy=np.asarray([30.0, 40.0]),
        similarity=0.7,
        ratio=1.0,
        similarity_margin=0.05,
        dual_softmax_confidence=0.2,
        coarse_rank=1,
        coarse_score=0.7,
        coarse_score_gap=0.1,
        mutual_rank=2,
        cell_delta_x=0,
        cell_delta_y=1,
    )

    features, names = vectorize_keypoint_matches([match])

    assert features.shape == (1, len(names))
    assert float(features[0, names.index("coarse_rank_score")]) == 0.5
    assert float(features[0, names.index("mutual_is_top1")]) == 0.0


def test_annotate_keypoint_matches_with_coarse_candidate_ranker_updates_confidence() -> None:
    matches = [
        KeypointFeatureMatch(
            query_index=0,
            render_index=0,
            query_xy=np.asarray([0.0, 0.0]),
            render_xy=np.asarray([0.0, 0.0]),
            similarity=0.9,
            ratio=1.0,
            dual_softmax_confidence=0.1,
            coarse_rank=0,
            coarse_score=0.9,
            coarse_score_gap=0.0,
            mutual_rank=0,
        ),
        KeypointFeatureMatch(
            query_index=0,
            render_index=1,
            query_xy=np.asarray([0.0, 0.0]),
            render_xy=np.asarray([10.0, 0.0]),
            similarity=0.6,
            ratio=1.0,
            dual_softmax_confidence=0.1,
            coarse_rank=4,
            coarse_score=0.6,
            coarse_score_gap=0.3,
            mutual_rank=3,
        ),
    ]
    train_x, _names = vectorize_keypoint_matches(matches)
    model = CalibratedLogisticConfidence(max_iter=50, learning_rate=0.2, l2=0.0).fit(
        train_x,
        np.asarray([1, 0], dtype=np.int64),
    )

    updated = annotate_keypoint_matches_with_coarse_candidate_ranker(matches, model)

    assert updated[0].dual_softmax_confidence is not None
    assert updated[1].dual_softmax_confidence is not None
    assert updated[0].dual_softmax_confidence > updated[1].dual_softmax_confidence
    assert matches[0].dual_softmax_confidence == 0.1
