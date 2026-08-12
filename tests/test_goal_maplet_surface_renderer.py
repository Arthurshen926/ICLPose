import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera, colmap_camera_focal_lengths
from feature_extract.vfm.localization_goal_maplet.surface_renderer import (
    _deterministic_contribution_order,
    _pool_rendered_surface,
)
from feature_extract.vfm.localization_v6.atlas_renderer import RenderedMapletAtlases


def test_supersampled_surface_pooling_is_mask_aware():
    mask = np.asarray([[1, 0], [1, 0]], dtype=bool)
    rendered = RenderedMapletAtlases(
        feature=np.asarray([[[1, 0], [1, 0]], [[0, 0], [0, 0]]], dtype=np.float32),
        xyz=np.dstack([np.ones((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32),
        normal=np.dstack([np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32),
        uncertainty=np.zeros((2, 2), dtype=np.float32),
        maplet_id=np.asarray([[3, -1], [3, -1]]),
        mask=mask,
        depth=np.asarray([[2, 0], [1, 0]], dtype=np.float32),
        surface_id=np.asarray([[4, -1], [5, -1]]),
        primitive_id=np.asarray([[14, -1], [15, -1]]),
        visibility=np.ones((2, 2), dtype=bool),
        field_missing=np.asarray([[0, 1], [0, 1]], dtype=bool),
        incidence=np.asarray([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32),
        projected_scale=np.asarray([[2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
    )
    pooled = _pool_rendered_surface(rendered, 2)
    assert pooled.mask[0, 0]
    np.testing.assert_allclose(pooled.feature[:, 0, 0], [1, 0], atol=1e-6)
    assert pooled.primitive_id[0, 0] == 15
    np.testing.assert_allclose(pooled.xyz[0, 0], [1, 0, 1], atol=1e-6)
    assert pooled.visibility[0, 0]
    assert pooled.field_missing[0, 0]
    np.testing.assert_allclose(pooled.feature_fraction[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(pooled.visibility_fraction[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled.missing_fraction[0, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(pooled.background_fraction[0, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        pooled.feature_fraction + pooled.missing_fraction + pooled.background_fraction,
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(pooled.dominant_surface_fraction[0, 0], 0.5, atol=1e-6)
    assert pooled.mixed_surface[0, 0]
    np.testing.assert_allclose(pooled.depth[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled.incidence[0, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(pooled.projected_scale[0, 0], 4.0, atol=1e-6)


def test_supersampled_geometry_uses_dominant_component_not_frontmost_outlier():
    shape = (2, 2)
    rendered = RenderedMapletAtlases(
        feature=np.ones((1, *shape), dtype=np.float32),
        xyz=np.dstack([
            np.zeros(shape), np.zeros(shape), np.asarray([[5.0, 5.0], [5.0, 1.0]])
        ]).astype(np.float32),
        normal=np.dstack([np.zeros(shape), np.zeros(shape), np.ones(shape)]).astype(np.float32),
        uncertainty=np.zeros(shape, dtype=np.float32),
        maplet_id=np.asarray([[3, 3], [3, 9]]),
        mask=np.ones(shape, dtype=bool),
        depth=np.asarray([[5.0, 5.0], [5.0, 1.0]], dtype=np.float32),
        surface_id=np.asarray([[4, 4], [4, 8]]),
        primitive_id=np.asarray([[14, 14], [14, 18]]),
        child_id=np.asarray([[7, 7], [7, 8]]),
        visibility=np.ones(shape, dtype=bool),
    )
    pooled = _pool_rendered_surface(rendered, 2)
    assert pooled.child_id[0, 0] == 7
    assert pooled.surface_id[0, 0] == 4
    np.testing.assert_allclose(pooled.depth[0, 0], 5.0)
    np.testing.assert_allclose(pooled.dominant_surface_fraction[0, 0], 0.75)
    assert pooled.mixed_surface[0, 0]


def test_colmap_focal_parsing_does_not_treat_principal_point_as_fy():
    simple = ColmapCamera(1, 0, 640, 480, (500.0, 320.0, 240.0))
    radial = ColmapCamera(2, 3, 640, 480, (450.0, 315.0, 235.0, 0.01, -0.001))
    pinhole = ColmapCamera(3, 1, 640, 480, (510.0, 490.0, 320.0, 240.0))
    assert colmap_camera_focal_lengths(simple) == (500.0, 500.0)
    assert colmap_camera_focal_lengths(radial) == (450.0, 450.0)
    assert colmap_camera_focal_lengths(pinhole) == (510.0, 490.0)


def test_surface_contribution_order_uses_stable_primitive_identity():
    pixel = np.asarray([1, 0, 1, 0], dtype=np.int64)
    primitive = np.asarray([20, 20, 10, 10], dtype=np.int64)
    weight = np.asarray([0.4, 0.3, 0.5, 0.2], dtype=np.float32)
    expected = [(0, 10, 0.2), (0, 20, 0.3), (1, 10, 0.5), (1, 20, 0.4)]
    for permutation in (
        np.arange(4, dtype=np.int64),
        np.asarray([2, 0, 3, 1], dtype=np.int64),
    ):
        order = _deterministic_contribution_order(
            pixel[permutation], primitive[permutation], weight[permutation],
        )
        actual = [
            (int(pixel[permutation][row]), int(primitive[permutation][row]),
             round(float(weight[permutation][row]), 1))
            for row in order.tolist()
        ]
        assert actual == expected
