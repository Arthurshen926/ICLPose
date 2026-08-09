import numpy as np

from feature_extract.vfm.localization_goal_maplet.surface_renderer import _pool_rendered_surface
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
    np.testing.assert_allclose(pooled.incidence[0, 0], 0.375, atol=1e-6)
    np.testing.assert_allclose(pooled.projected_scale[0, 0], 1.5, atol=1e-6)
