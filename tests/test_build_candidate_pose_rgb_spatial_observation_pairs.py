from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_observation_pairs import (
    partition_train_query_ids,
    select_distinct_ann_tracks,
)


def test_inner_partition_is_deterministic_and_query_disjoint() -> None:
    train, validation = partition_train_query_ids(
        query_ids=["q/c.png", "q/a.png", "q/b.png", "q/d.png", "q/e.png"],
        fold_count=3,
        fold_index=1,
    )
    assert train and validation
    assert set(train).isdisjoint(validation)
    assert (train, validation) == partition_train_query_ids(
        query_ids=["q/e.png", "q/d.png", "q/c.png", "q/b.png", "q/a.png"],
        fold_count=3,
        fold_index=1,
    )


def test_ann_track_selection_skips_positive_and_duplicate_tracks() -> None:
    selected = select_distinct_ann_tracks(
        ann_track_ids=np.asarray([17, 17, 9, 21, 9, 33], dtype=np.int64),
        positive_track_id=17,
        negative_count=3,
    )
    np.testing.assert_array_equal(selected, [9, 21, 33])


def test_inner_partition_rejects_empty_or_invalid_fold() -> None:
    with pytest.raises(ValueError, match="invalid"):
        partition_train_query_ids(query_ids=["q/a.png"], fold_count=2, fold_index=0)
