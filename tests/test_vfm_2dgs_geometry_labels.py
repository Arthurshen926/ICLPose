from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap
from feature_extract.vfm.vfm_2dgs_geometry_labels import (
    TwoDgsGeometryRenderConfig,
    render_2dgs_geometry_label,
)


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=32, height=24, params=(32.0, 32.0, 16.0, 12.0))


def _surface() -> SurfaceElementMap:
    return SurfaceElementMap(
        element_ids=np.asarray([0], dtype=np.int64),
        parent_gaussian_indices=np.asarray([0], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        tangent1=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        tangent2=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        scale1=np.asarray([0.5], dtype=np.float32),
        scale2=np.asarray([0.5], dtype=np.float32),
        opacity=np.asarray([1.0], dtype=np.float32),
        area=np.asarray([1.0], dtype=np.float32),
        adjacency=(np.zeros((0,), dtype=np.int64),),
    )


def test_render_2dgs_geometry_label_produces_depth_alpha_and_camera_normals() -> None:
    label = render_2dgs_geometry_label(
        _surface(),
        np.eye(4, dtype=np.float64),
        _camera(),
        width=32,
        height=24,
        config=TwoDgsGeometryRenderConfig(sigma_scale=1.0, max_radius_px=8.0, min_opacity=0.0),
    )

    assert label.depth.shape == (24, 32)
    assert label.normal_cam.shape == (24, 32, 3)
    assert label.alpha.shape == (24, 32)
    assert label.valid.shape == (24, 32)
    assert int(label.valid.sum()) > 1
    assert np.allclose(label.depth[label.valid], 4.0)
    center_normal = label.normal_cam[12, 16]
    assert np.isclose(np.linalg.norm(center_normal), 1.0, atol=1e-5)
    assert center_normal[2] < -0.99
    assert float(label.alpha[label.valid].max()) > 0.0
