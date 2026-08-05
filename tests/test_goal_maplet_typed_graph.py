import numpy as np

from feature_extract.vfm.localization_goal_maplet.typed_graph import MAPPING_COVISIBILITY, TypedParentGraph


def test_typed_graph_covisibility_is_symmetric_and_same_parent_neutral():
    graph = TypedParentGraph(
        edge_source=np.asarray([0], dtype=np.int32),
        edge_target=np.asarray([1], dtype=np.int32),
        edge_type=np.asarray([MAPPING_COVISIBILITY], dtype=np.uint8),
        edge_features=np.asarray([[1.0, 0.9, 0.6, 0.7, 0.1, 0.2]], dtype=np.float32),
        parent_view_count=np.asarray([3, 4], dtype=np.int32),
        parent_distinctiveness=np.asarray([0.1, 0.2], dtype=np.float32),
        physical_map_sha256="a" * 64,
        canonical_field_sha256="b" * 64,
        metadata={"artifact_type": "goal_maplet_typed_parent_graph_v1"},
    )
    matrix = graph.covisibility_matrix()
    assert np.allclose(matrix, [[1.0, 0.6], [0.6, 1.0]])
