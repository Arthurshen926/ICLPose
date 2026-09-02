import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_radio_planar_support_pose import (
    _match_plane_regions,
    _token_matches,
)


def _region(descriptor: np.ndarray, token_features: np.ndarray, *, query: bool) -> dict:
    row = {
        "descriptor": np.asarray(descriptor, np.float32),
        "token_ids": np.arange(len(token_features)),
        "token_features": np.asarray(token_features, np.float32),
    }
    if query:
        row["token_points_camera"] = np.arange(len(token_features) * 3, dtype=float).reshape(-1, 3)
    else:
        row["token_points_world"] = np.arange(len(token_features) * 3, dtype=float).reshape(-1, 3) + 1
        row["chart_name"] = "seq4__frame00001.png"
    return row


def test_plane_retrieval_preserves_top_candidates_for_homography() -> None:
    mapping = [
        _region(np.asarray([1.0, 0.0, 0.0]), np.eye(3), query=False),
        _region(np.asarray([0.8, 0.6, 0.0]), np.eye(3), query=False),
        _region(np.asarray([0.7, 0.0, 0.71414284]), np.eye(3), query=False),
        _region(np.asarray([0.6, 0.0, 0.8]), np.eye(3), query=False),
        _region(np.asarray([0.0, 1.0, 0.0]), np.eye(3), query=False),
    ]
    query = [_region(np.asarray([1.0, 0.0, 0.0]), np.eye(3), query=True)]
    matches = _match_plane_regions(query, mapping)
    assert [row["candidate_rank"] for row in matches] == [1, 2, 3, 4]
    assert matches[0]["map_region"] == 0


def test_token_matching_is_mutual_and_stays_inside_retrieved_plane() -> None:
    feature = np.eye(6, dtype=np.float32)
    query = [_region(np.asarray([1.0, 0.0, 0.0]), feature, query=True)]
    mapping = [_region(np.asarray([1.0, 0.0, 0.0]), feature, query=False)]
    query[0]["token_ids"] = np.asarray([0, 1, 2, 64, 65, 66])
    mapping[0]["token_ids"] = np.asarray([0, 1, 2, 64, 65, 66])
    source, target, rows = _token_matches(
        query,
        mapping,
        [{
            "query_region": 0,
            "map_region": 0,
            "candidate_rank": 1,
            "plane_cosine": 1.0,
            "plane_margin_to_next": 0.5,
        }],
    )
    assert source.shape == target.shape == (6, 3)
    assert rows[0]["mutual_token_match_count"] == 6
    assert rows[0]["homography_accepted"] is True
