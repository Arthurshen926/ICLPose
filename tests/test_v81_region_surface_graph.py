import numpy as np

from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    QueryMapletGroup,
)
from feature_extract.vfm.localization_v8.structured_maplet_graph import (
    build_physical_maplet_graph,
)
from feature_extract.vfm.localization_v81.region_surface_graph import (
    build_region_evidence_graph,
)


def _bank(count=8):
    return SurfaceRetrievalMapletBank(
        maplet_ids=np.arange(count),
        centers=np.c_[np.arange(count), np.zeros((count, 2))],
        normals=np.tile([0, 0, 1], (count, 1)),
        extents=np.ones((count, 3)),
        tangent_frames=np.tile(np.eye(3), (count, 1, 1)),
        descriptor_offsets=np.arange(count + 1),
        descriptors=np.eye(count),
        descriptor_weights=np.ones(count),
        quality_scores=np.ones(count),
        descriptor_uncertainties=np.zeros(count),
        metadata={"has_canonical_tangent_frames": True},
    )


def _group(x, identity, probability=0.8):
    return QueryMapletGroup(
        query_region_xy=np.asarray([x, 50.0]),
        query_region_extent=np.asarray([8.0, 8.0]),
        maplet_ids=np.asarray([identity, (identity + 1) % 8]),
        probabilities=np.asarray([probability, 0.1]),
        null_probability=1.0 - probability - 0.1,
        omitted_probability=0.0,
    )


def test_overlapping_vfm_supports_become_one_evidence_group(tmp_path):
    feature_path = tmp_path / "feature.npz"
    feature_path.write_bytes(b"canonical")
    physical = build_physical_maplet_graph(_bank(), feature_bank_path=feature_path)
    observations = (_group(30.0, 2), _group(33.0, 2), _group(75.0, 5))
    retrieval = MapletRetrievalResult(observations, np.arange(8), np.ones(8))
    descriptor = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    graph = build_region_evidence_graph(
        retrieval, physical, descriptor, image_size_wh=(100, 100)
    )
    assert graph.group_count == 2
    assert sorted(np.diff(graph.member_offsets).tolist()) == [1, 2]


def test_query_edges_are_unique_and_undirected(tmp_path):
    feature_path = tmp_path / "feature.npz"
    feature_path.write_bytes(b"canonical")
    physical = build_physical_maplet_graph(_bank(), feature_bank_path=feature_path)
    observations = tuple(_group(float(10 + 12 * index), index) for index in range(6))
    retrieval = MapletRetrievalResult(observations, np.arange(8), np.ones(8))
    graph = build_region_evidence_graph(
        retrieval,
        physical,
        np.eye(6),
        image_size_wh=(100, 100),
        local_neighbors=2,
        medium_neighbors=1,
        long_neighbors=1,
    )
    pairs = list(zip(graph.edge_source.tolist(), graph.edge_target.tolist()))
    assert all(first < second for first, second in pairs)
    assert len(pairs) == len(set(pairs))


def test_group_posterior_conserves_probability_mass(tmp_path):
    feature_path = tmp_path / "feature.npz"
    feature_path.write_bytes(b"canonical")
    physical = build_physical_maplet_graph(_bank(), feature_bank_path=feature_path)
    observations = (_group(30.0, 2), _group(33.0, 2))
    graph = build_region_evidence_graph(
        MapletRetrievalResult(observations, np.arange(8), np.ones(8)),
        physical,
        np.asarray([[1.0, 0.0], [0.99, 0.01]]),
        image_size_wh=(100, 100),
        candidates_per_group=1,
    )
    total = graph.null_probability + np.sum(graph.candidate_probabilities, axis=1)
    assert np.allclose(total, 1.0)
