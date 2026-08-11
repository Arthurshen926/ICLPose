from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.localization_goal_maplet.query_edge_factor import QueryEdgeGraph
from feature_extract.vfm.localization_goal_maplet.structured_parent_posterior import (
    refine_parent_posteriors_with_query_graph,
)


def _fixture():
    physical = SimpleNamespace(
        content_sha256="map",
        maplet_ids=np.asarray([10, 11, 12, 13]),
        maplet_centers=np.asarray([[0, 0, 5], [1, 0, 5], [20, 0, 5], [21, 0, 5]], dtype=float),
        maplet_normals=np.tile(np.asarray([[0, 0, 1.0]]), (4, 1)),
        maplet_extents=np.ones((4, 3)),
    )
    covis = np.asarray([
        [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1],
    ], dtype=float)
    graph = SimpleNamespace(physical_map_sha256="map", covisibility_matrix=lambda: covis)
    edge = QueryEdgeGraph(
        source=np.asarray([0]), target=np.asarray([1]), kind=np.asarray([0], dtype=np.uint8),
        high_displacement=np.asarray([False]), distinctive_context=np.asarray([False]),
    )
    return physical, graph, edge


def test_structured_parent_posterior_preserves_retained_mass_and_promotes_coherence():
    physical, graph, edge = _fixture()
    ids = np.asarray([[10, 12], [13, 11]])
    probability = np.asarray([[0.36, 0.24], [0.31, 0.29]])
    refined, diagnostics = refine_parent_posteriors_with_query_graph(
        ids, probability, np.asarray([[0.2, 0.5], [0.3, 0.5]]), np.eye(2),
        physical, graph, iterations=3, query_graph=edge,
    )
    np.testing.assert_allclose(np.sum(refined, axis=1), np.sum(probability, axis=1))
    assert refined[1, 1] > probability[1, 1]
    assert diagnostics.edge_count == 1


def test_zero_iterations_is_identity():
    physical, graph, edge = _fixture()
    ids = np.asarray([[10, 12], [13, 11]])
    probability = np.asarray([[0.36, 0.24], [0.31, 0.29]])
    refined, _ = refine_parent_posteriors_with_query_graph(
        ids, probability, np.asarray([[0.2, 0.5], [0.3, 0.5]]), np.eye(2),
        physical, graph, iterations=0, query_graph=edge,
    )
    np.testing.assert_allclose(refined, probability)
