from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization_goal_maplet.fine_support_reranker import (
    FEATURE_NAMES,
    extract_child_reranking_features,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    all_radio_token_coordinates,
)
from test_goal_maplet_pure_retrieval import _physical, _result


def test_query_local_features_are_finite_and_translation_agnostic() -> None:
    physical = _physical()
    retrieval = _result(physical)
    area = np.full(physical.child_parent_rows.shape, 2.0, dtype=np.float64)
    first = extract_child_reranking_features(retrieval, physical, area)
    shifted = physical.__class__(
        **{
            **physical.__dict__,
            "maplet_centers": physical.maplet_centers + 1000.0,
            "child_centers": physical.child_centers + 1000.0,
        }
    )
    second = extract_child_reranking_features(retrieval, shifted, area)
    assert first.values.shape[1] == len(FEATURE_NAMES)
    assert np.array_equal(first.child_rows, second.child_rows)
    assert np.array_equal(first.values, second.values)
    assert np.all(np.isfinite(first.values))


def test_features_reject_noncanonical_token_layout() -> None:
    physical = _physical()
    retrieval = _result(physical)
    bad_xy = all_radio_token_coordinates(
        int(retrieval.metadata["token_height"]),
        int(retrieval.metadata["token_width"]),
    ).copy()
    bad_xy[[0, 1]] = bad_xy[[1, 0]]
    try:
        bad = retrieval.__class__(**{**retrieval.__dict__, "token_xy": bad_xy})
    except ValueError:
        return
    with np.testing.assert_raises(ValueError):
        extract_child_reranking_features(
            bad, physical, np.ones(physical.child_parent_rows.shape)
        )
