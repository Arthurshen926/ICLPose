import numpy as np
import pytest

from feature_extract.vfm.localization.surface_feature_field import SurfaceFeatureField
from feature_extract.vfm.localization.surface_metric_feature_mapper import (
    SurfaceMetricFeatureMapper,
    SurfaceMetricFeatureMapperConfig,
    load_surface_metric_feature_mapper,
    save_surface_metric_feature_mapper,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
    retrieve_surface_maplets,
)


def _field(metadata=None):
    return SurfaceFeatureField(
        source_indices=np.asarray([3, 9]),
        centers=np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.float64),
        normals=np.asarray([[0, 0, 2], [0, 0, 1]], dtype=np.float32),
        tangent1=np.asarray([[2, 0, 0], [1, 0, 0]], dtype=np.float32),
        tangent2=np.asarray([[0, 1, 0], [0, 2, 0]], dtype=np.float32),
        scale1=np.ones(2),
        scale2=np.ones(2),
        opacity=np.ones(2),
        features=np.asarray([[2, 0], [0, 3]], dtype=np.float32),
        uncertainty=np.zeros(2),
        confidence=np.ones(2),
        support_weight=np.ones(2),
        support_count=np.ones(2),
        owner_maplet_ids=np.asarray([4, 5]),
        metadata=metadata
        or {
            "artifact_type": "radio_final_2dgs_surface_feature_field",
            "vfm_layer": "radio_final",
        },
    )


def test_surface_feature_field_roundtrip(tmp_path):
    field = _field()
    path = tmp_path / "field.npz"
    field.save_npz(path)
    loaded = SurfaceFeatureField.load_npz(path)
    assert loaded.feature_dim == 2
    assert np.allclose(np.linalg.norm(loaded.features, axis=1), 1.0)
    assert loaded.owner_maplet_ids.tolist() == [4, 5]


def test_surface_feature_field_rejects_anchor_identity():
    with pytest.raises(ValueError, match="stable_anchor"):
        _field(
            {
                "artifact_type": "radio_final_2dgs_surface_feature_field",
                "vfm_layer": "radio_final",
                "uses_stable_anchor_identity": True,
            }
        )


def test_surface_metric_mapper_roundtrip_and_dense_projection(tmp_path):
    model = SurfaceMetricFeatureMapper(
        SurfaceMetricFeatureMapperConfig(input_dim=4, hidden_dim=8, output_dim=4)
    )
    path = tmp_path / "metric.pt"
    save_surface_metric_feature_mapper(
        path,
        model,
        {
            "uses_alike_descriptors": False,
            "uses_radio_intermediate": False,
            "uses_sfm_points": False,
            "uses_sfm_tracks": False,
        },
    )
    loaded = load_surface_metric_feature_mapper(path)
    output = loaded.project_map(np.ones((4, 2, 3), dtype=np.float32))
    assert output.shape == (4, 2, 3)
    assert np.allclose(np.linalg.norm(output, axis=0), 1.0, atol=1e-5)


def test_anchor_free_retrieval_maplets_roundtrip_and_retrieval(tmp_path):
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([4, 9]),
        centers=np.zeros((2, 3), dtype=np.float32),
        normals=np.asarray([[0, 0, 1], [0, 1, 0]], dtype=np.float32),
        extents=np.ones((2, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 2, 3]),
        descriptors=np.asarray([[1, 0], [-1, 0], [0, 1]], dtype=np.float32),
        descriptor_weights=np.asarray([0.6, 0.4, 1.0], dtype=np.float32),
        quality_scores=np.ones((2,), dtype=np.float32),
        descriptor_uncertainties=np.zeros((2,), dtype=np.float32),
        metadata={"vfm_layer": "radio_final"},
    )
    path = tmp_path / "maplets.npz"
    bank.save_npz(path)
    loaded = SurfaceRetrievalMapletBank.load_npz(path)
    selected, diagnostics = retrieve_surface_maplets(
        np.asarray([[0.99, 0.01]], dtype=np.float32),
        loaded,
        maximum_maplets=1,
    )
    assert selected.tolist() == [4]
    assert diagnostics["representation"] == "compact_radio_final_mixture_per_maplet"
    selected_second_mode, _ = retrieve_surface_maplets(
        np.asarray([[-0.99, 0.01]], dtype=np.float32),
        loaded,
        maximum_maplets=1,
    )
    assert selected_second_mode.tolist() == [4]


def test_anchor_free_retrieval_maplets_reject_anchor_contract():
    with pytest.raises(ValueError, match="stable_anchor"):
        SurfaceRetrievalMapletBank(
            maplet_ids=np.asarray([1]),
            centers=np.zeros((1, 3), dtype=np.float32),
            normals=np.ones((1, 3), dtype=np.float32),
            extents=np.ones((1, 3), dtype=np.float32),
            descriptor_offsets=np.asarray([0, 1]),
            descriptors=np.ones((1, 2), dtype=np.float32),
            descriptor_weights=np.ones((1,), dtype=np.float32),
            quality_scores=np.ones((1,), dtype=np.float32),
            descriptor_uncertainties=np.zeros((1,), dtype=np.float32),
            metadata={"uses_stable_anchor_identity": True},
        )
