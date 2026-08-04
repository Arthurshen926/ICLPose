import numpy as np
import torch

from feature_extract.vfm.localization.surface_retrieval_maplets import SurfaceRetrievalMapletBank
from feature_extract.vfm.localization_v6.maplet_retrieval import MapletRetrievalResult, QueryMapletGroup
from feature_extract.vfm.localization_v8.multi_teacher_student import MapletRetrievalAdaptor, MapletRetrievalAdaptorConfig
from feature_extract.vfm.localization_v8.structured_maplet_graph import build_physical_maplet_graph, build_query_region_graph


def _bank():
    return SurfaceRetrievalMapletBank(
        maplet_ids=np.arange(5), centers=np.c_[np.arange(5), np.zeros((5, 2))],
        normals=np.tile([0, 0, 1], (5, 1)), extents=np.ones((5, 3)),
        tangent_frames=np.tile(np.eye(3), (5, 1, 1)), descriptor_offsets=np.arange(6),
        descriptors=np.eye(5), descriptor_weights=np.ones(5), quality_scores=np.ones(5),
        descriptor_uncertainties=np.zeros(5), metadata={"has_canonical_tangent_frames": True},
    )


def test_physical_graph_stores_geometry_not_downstream_features(tmp_path):
    feature_path = tmp_path / "features.npz"; feature_path.write_bytes(b"one canonical feature")
    graph = build_physical_maplet_graph(_bank(), feature_bank_path=feature_path, local_neighbors=1, medium_neighbors=1, long_neighbors=1)
    path = tmp_path / "graph.npz"; graph.save_npz(path)
    with np.load(path) as data:
        assert "descriptors" not in data.files
        assert "dino" not in " ".join(data.files).lower()
        assert "sam" not in " ".join(data.files).lower()
        assert "siglip" not in " ".join(data.files).lower()


def test_query_graph_preserves_duplicate_candidate_instances(tmp_path):
    feature_path = tmp_path / "features.npz"; feature_path.write_bytes(b"feature")
    physical = build_physical_maplet_graph(_bank(), feature_bank_path=feature_path)
    groups = tuple(QueryMapletGroup(
        query_region_xy=np.array([10.0 + index, 20.0]), query_region_extent=np.array([3.0, 3.0]),
        maplet_ids=np.array([2, 3]), probabilities=np.array([0.6, 0.2]), null_probability=0.2,
        omitted_probability=0.0,
    ) for index in range(3))
    query = build_query_region_graph(MapletRetrievalResult(groups, np.array([2, 3]), np.ones(2)), physical, image_size_wh=(100, 100))
    assert query.xy.shape[0] == 3
    assert np.all(query.candidate_rows[:, 0] == 2)


def test_single_student_is_identity_initialized():
    model = MapletRetrievalAdaptor(MapletRetrievalAdaptorConfig(feature_dim=8, hidden_dim=16))
    value = torch.randn(4, 8)
    expected = torch.nn.functional.normalize(value, dim=-1)
    assert torch.allclose(model(value), expected, atol=1e-6)
