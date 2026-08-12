from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.mapping_view_graph import (
    MappingViewGraph,
    mapping_view_pose_modes,
    retrieve_mapping_view_posterior,
)


def _fixture():
    physical = SimpleNamespace(
        content_sha256="map",
        maplet_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
    )
    pose = np.tile(np.eye(4, dtype=np.float64), (2, 1, 1))
    pose[1, 0, 3] = 5.0
    graph = MappingViewGraph(
        poses_w2c=pose,
        parent_offsets=np.asarray([0, 2, 4]),
        parent_rows=np.asarray([0, 1, 2, 3]),
        parent_weights=np.ones((4,), dtype=np.float32),
        physical_map_sha256="map",
        canonical_field_sha256="field",
        metadata={"artifact_type": "goal_maplet_mapping_view_graph_v1"},
    )
    return physical, graph


def test_mapping_view_retrieval_preserves_ambiguous_identity_until_view_factor():
    physical, graph = _fixture()
    result = retrieve_mapping_view_posterior(
        graph,
        physical,
        candidate_maplet_ids=np.asarray([[10, 12], [11, 13]]),
        # The correct identities have low absolute mass but dominate only
        # after the two supports are explained by the same view node.
        candidate_probabilities=np.asarray([[0.02, 0.01], [0.03, 0.01]]),
        out_of_map_probabilities=np.asarray([0.6, 0.5]),
        unresolved_probabilities=np.asarray([0.97, 0.96]),
        maximum_views=2,
    )
    assert result.view_rows.tolist() == [0, 1]
    assert result.scores[0] > result.scores[1]
    assert np.isclose(np.sum(result.probabilities) + result.null_probability, 1.0)


def test_mapping_view_retrieval_is_invariant_to_per_support_retained_scale():
    physical, graph = _fixture()
    common = dict(
        graph=graph,
        physical=physical,
        candidate_maplet_ids=np.asarray([[10, 12], [11, 13]]),
        out_of_map_probabilities=np.asarray([0.2, 0.2]),
        unresolved_probabilities=np.asarray([0.4, 0.4]),
        maximum_views=2,
    )
    first = retrieve_mapping_view_posterior(
        candidate_probabilities=np.asarray([[0.4, 0.1], [0.3, 0.2]]),
        **common,
    )
    second = retrieve_mapping_view_posterior(
        candidate_probabilities=np.asarray([[0.04, 0.01], [0.03, 0.02]]),
        **common,
    )
    np.testing.assert_array_equal(first.view_rows, second.view_rows)
    np.testing.assert_allclose(first.scores, second.scores)


def test_mapping_view_graph_roundtrip(tmp_path):
    _physical, graph = _fixture()
    path = tmp_path / "view_graph.npz"
    graph.save_npz(path)
    restored = MappingViewGraph.load_npz(path)
    assert restored.content_sha256 == graph.content_sha256
    np.testing.assert_allclose(restored.poses_w2c, graph.poses_w2c)


def test_mapping_view_null_is_one_without_resolved_support_and_view_count_invariant():
    physical, graph = _fixture()
    common = dict(
        graph=graph,
        physical=physical,
        candidate_maplet_ids=np.asarray([[-1, -1], [-1, -1]]),
        candidate_probabilities=np.zeros((2, 2), dtype=np.float64),
        out_of_map_probabilities=np.ones((2,), dtype=np.float64),
        unresolved_probabilities=np.ones((2,), dtype=np.float64),
    )
    one = retrieve_mapping_view_posterior(maximum_views=1, **common)
    two = retrieve_mapping_view_posterior(maximum_views=2, **common)
    assert one.null_probability == 1.0
    assert two.null_probability == 1.0
    assert np.all(one.probabilities == 0.0)
    assert np.all(two.probabilities == 0.0)
    assert mapping_view_pose_modes(graph, two).poses_w2c.shape[0] == 0


def test_mapping_view_h0_is_normalized_before_conditional_view_distribution():
    physical, graph = _fixture()
    common = dict(
        graph=graph,
        physical=physical,
        candidate_maplet_ids=np.asarray([[10, 12]]),
        candidate_probabilities=np.asarray([[0.15, 0.05]]),
        out_of_map_probabilities=np.asarray([0.70]),
        unresolved_probabilities=np.asarray([0.80]),
    )
    one = retrieve_mapping_view_posterior(maximum_views=1, **common)
    two = retrieve_mapping_view_posterior(maximum_views=2, **common)
    assert np.isclose(one.null_probability, 0.8)
    assert np.isclose(two.null_probability, 0.8)
    assert np.isclose(np.sum(one.probabilities), 0.2)
    assert np.isclose(np.sum(two.probabilities), 0.2)
