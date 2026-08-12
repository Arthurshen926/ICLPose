import numpy as np

from feature_extract.tools.vfm.build_2dgs_geometry_labels_from_contributors import (
    _downsample_front_surface,
)


def test_downsample_selects_physical_front_surface_and_preserves_support():
    ids = np.asarray([[1, 2], [-1, 3]])
    depth = np.asarray([[4.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    alpha = np.asarray([[0.8, 0.7], [0.0, 0.9]], dtype=np.float32)
    out_ids, out_depth, out_alpha, support = _downsample_front_surface(
        ids, depth, alpha, output_height=1, output_width=1, minimum_alpha=0.05,
    )
    assert out_ids.tolist() == [[2]]
    assert out_depth.tolist() == [[2.0]]
    np.testing.assert_allclose(out_alpha, [[0.6]])
    assert support.tolist() == [[3]]
