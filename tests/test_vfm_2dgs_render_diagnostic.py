import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap
from feature_extract.vfm.vfm_2dgs_render_diagnostic import (
    Vfm2DgsRenderConfig,
    render_vfm_2dgs_anchor_features,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(4.0, 4.0, 2.0, 2.0))


def _surface_elements() -> SurfaceElementMap:
    return SurfaceElementMap(
        element_ids=np.asarray([10, 20], dtype=np.int64),
        parent_gaussian_indices=np.asarray([10, 20], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        tangent1=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        tangent2=np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        scale1=np.asarray([1.0, 1.0], dtype=np.float32),
        scale2=np.asarray([1.0, 1.0], dtype=np.float32),
        opacity=np.asarray([1.0, 1.0], dtype=np.float32),
        area=np.asarray([1.0, 1.0], dtype=np.float32),
        adjacency=(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)),
    )


def _anchor_map() -> Vfm2DgsAnchorMap:
    return Vfm2DgsAnchorMap(
        anchor_ids=np.asarray([1, 2], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0]], dtype=np.float64),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None, :, :], (2, 1, 1)),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        feature_variances=np.asarray([0.1, 0.2], dtype=np.float32),
        quality_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        purity_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        observation_counts=np.asarray([2, 2], dtype=np.int64),
        surface_support_counts=np.asarray([1, 1], dtype=np.int64),
        support_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        support_element_ids=np.asarray([10, 20], dtype=np.int64),
        support_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        observed_view_ids=(("a.png",), ("b.png",)),
    )


def test_vfm_2dgs_feature_render_uses_surface_footprint_and_depth_gate() -> None:
    rendered = render_vfm_2dgs_anchor_features(
        _anchor_map(),
        _surface_elements(),
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
        config=Vfm2DgsRenderConfig(width=4, height=4, min_radius_px=1.0, depth_epsilon=0.1),
    )

    assert rendered.feature_map.shape == (2, 4, 4)
    assert rendered.xyz_map.shape == (4, 4, 3)
    assert rendered.visibility_mask[2, 2]
    np.testing.assert_allclose(rendered.feature_map[:, 2, 2], np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-5)
    assert rendered.depth_map[2, 2] < 5.0
    assert rendered.support_count_map[2, 2] >= 1
