from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.query_edge_factor import (
    LOCAL_EDGE,
    QueryEdgeGraph,
    build_query_edge_graph,
    score_pose_conditioned_query_edges,
)


def test_query_edge_graph_is_candidate_independent_unique_and_multiscale():
    xy = np.asarray([
        [0.1, 0.1], [0.2, 0.1], [0.8, 0.8], [0.9, 0.8],
    ])
    descriptor = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
    ])
    graph = build_query_edge_graph(
        xy, descriptor, local_neighbors=1, long_neighbors=1,
        minimum_long_displacement=0.3,
    )
    assert np.all(graph.source < graph.target)
    assert len(set(zip(graph.source.tolist(), graph.target.tolist()))) == graph.source.size
    assert np.any(graph.kind == LOCAL_EDGE)
    assert np.any(graph.high_displacement)
    assert np.any(graph.distinctive_context)


def test_pose_conditioned_edge_score_prefers_correct_surface_configuration():
    camera = ColmapCamera(
        camera_id=1, model_id=1, width=100, height=100,
        params=(100.0, 100.0, 50.0, 50.0),
    )
    physical = SimpleNamespace(
        child_centers=np.asarray([
            [-1.0, -1.0, 5.0], [1.0, -1.0, 5.0],
            [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0],
        ]),
        child_frames=np.repeat(np.eye(3)[None], 4, axis=0),
        child_extents=np.repeat(np.asarray([[0.1, 0.1, 0.01]]), 4, axis=0),
    )
    xy = np.asarray([
        [0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7],
    ])
    extent = np.full((4, 2), 0.03)
    graph = QueryEdgeGraph(
        source=np.asarray([0, 0, 1, 2]),
        target=np.asarray([1, 2, 3, 3]),
        kind=np.asarray([0, 1, 1, 0], dtype=np.uint8),
        high_displacement=np.ones((4,), dtype=bool),
        distinctive_context=np.ones((4,), dtype=bool),
    )
    pose = np.eye(4)
    correct = score_pose_conditioned_query_edges(
        graph, xy, extent, np.arange(4), pose, physical, camera,
    )
    shifted = score_pose_conditioned_query_edges(
        graph, xy, extent, np.asarray([3, 2, 1, 0]), pose, physical, camera,
    )
    assert correct["all"]["vector_score"] > shifted["all"]["vector_score"]
    assert correct["all"]["fixed_vector_score"] > shifted["all"]["fixed_vector_score"]
    assert correct["all"]["normalized_vector_residual_median"] < 1e-8
    assert correct["all"]["order_consistency"] == 1.0


def test_fixed_edge_score_penalizes_unexplained_assignments():
    camera = ColmapCamera(
        camera_id=1, model_id=1, width=100, height=100,
        params=(100.0, 100.0, 50.0, 50.0),
    )
    physical = SimpleNamespace(
        child_centers=np.asarray([[-1.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
        child_frames=np.repeat(np.eye(3)[None], 2, axis=0),
        child_extents=np.repeat(np.asarray([[0.1, 0.1, 0.01]]), 2, axis=0),
    )
    graph = QueryEdgeGraph(
        source=np.asarray([0]), target=np.asarray([1]),
        kind=np.asarray([0], dtype=np.uint8),
        high_displacement=np.asarray([True]),
        distinctive_context=np.asarray([True]),
    )
    xy = np.asarray([[0.3, 0.5], [0.7, 0.5]])
    extent = np.full((2, 2), 0.03)
    complete = score_pose_conditioned_query_edges(
        graph, xy, extent, np.asarray([0, 1]), np.eye(4), physical, camera,
    )
    missing = score_pose_conditioned_query_edges(
        graph, xy, extent, np.asarray([0, -1]), np.eye(4), physical, camera,
    )
    assert complete["all"]["planned_edge_count"] == 1
    assert missing["all"]["edge_count"] == 0
    assert missing["all"]["missing_edge_count"] == 1
    assert missing["all"]["fixed_vector_score"] == -2.0
    assert complete["all"]["fixed_vector_score"] > missing["all"]["fixed_vector_score"]
