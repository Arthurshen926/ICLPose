import numpy as np
import pytest

from feature_extract.vfm.colmap_tracks import ColmapImageObservation
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


def _image(name: str) -> ColmapImageObservation:
    return ColmapImageObservation(
        image_id=1,
        image_name=name,
        camera_id=2,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros((3,)),
        xys=np.asarray([[10.0, 10.0], [20.0, 20.0], [11.0, 10.0]]),
        point3d_ids=np.asarray([101, 202, -1]),
    )


def test_registered_query_targets_keep_only_reliable_track_observations() -> None:
    targets = registered_query_observation_targets(
        query_ids=["q.png", "q.png", "q.png"],
        query_xy=np.asarray([[10.5, 10.0], [21.0, 20.0], [30.0, 30.0]]),
        images_by_name={"q.png": _image("q.png")},
        max_distance_px=1.1,
    )

    assert targets.track_ids.tolist() == [101, 202, -1]
    assert targets.supervised.tolist() == [True, True, False]
    np.testing.assert_allclose(targets.distances_px[:2], [0.5, 1.0])
    assert np.isinf(targets.distances_px[2])


def test_registered_candidate_identity_distinguishes_unsupervised_from_no_match() -> None:
    targets = registered_query_observation_targets(
        query_ids=["q.png", "q.png", "q.png"],
        query_xy=np.asarray([[10.0, 10.0], [20.0, 20.0], [40.0, 40.0]]),
        images_by_name={"q.png": _image("q.png")},
        max_distance_px=0.25,
    )
    labels = registered_candidate_identity_labels(
        np.asarray([[9, 101], [8, 7], [202, 101]], dtype=np.int64), targets
    )
    summary = summarize_registered_candidate_identity(labels, targets)

    assert labels.tolist() == [[False, True], [False, False], [False, False]]
    assert summary["supervised_row_count"] == 2
    assert summary["candidate_recall_given_supervised"] == 0.5
    assert summary["rank1_recall_given_supervised"] == 0.0
    assert summary["median_positive_rank_when_retrieved"] == 2.0


def test_registered_query_targets_reject_missing_colmap_image() -> None:
    with pytest.raises(KeyError, match="missing"):
        registered_query_observation_targets(
            query_ids=["missing.png"],
            query_xy=np.asarray([[0.0, 0.0]]),
            images_by_name={},
            max_distance_px=2.0,
        )
