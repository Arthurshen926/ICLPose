from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_direct_radio_plane_ranking import (
    _aggregate_observation_scores,
    _query_subset_names,
)


def test_top2_mean_requires_two_consistent_observations() -> None:
    scores = np.asarray([0.99, 0.10, 0.80, 0.79, 0.78], np.float64)
    offsets = np.asarray([0, 2, 5], np.int64)
    maximum = _aggregate_observation_scores(scores, offsets, "max")
    robust = _aggregate_observation_scores(scores, offsets, "top2_mean")
    assert maximum.tolist() == [0.99, 0.80]
    assert np.allclose(robust, [0.545, 0.795])
    assert int(np.argmax(maximum)) == 0
    assert int(np.argmax(robust)) == 1


def test_query_subset_is_pose_free_and_preserves_frozen_order(tmp_path) -> None:
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_direct_radio_to_finite_plane_ranking_v1",
        "uses_pose_or_ground_truth": False,
        "contains_postlabel_fields": False,
        "rows": [{"image": "b.npz"}, {"image": "a.npz"}],
    }))
    assert _query_subset_names(path) == ["b.npz", "a.npz"]


def test_query_subset_rejects_postlabel_or_duplicates(tmp_path) -> None:
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_direct_radio_to_finite_plane_ranking_v2",
        "uses_pose_or_ground_truth": False,
        "contains_postlabel_fields": False,
        "rows": [{"image": "a.npz"}, {"image": "a.npz"}],
    }))
    with pytest.raises(ValueError, match="empty or duplicated"):
        _query_subset_names(path)
    payload = json.loads(path.read_text())
    payload["rows"] = [{"image": "a.npz"}]
    payload["contains_postlabel_fields"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not pose/label-free"):
        _query_subset_names(path)
