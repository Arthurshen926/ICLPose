from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_atlas import (
    _fuse_plane_texels,
    _fuse_plane_texel_prototypes,
)


def test_plane_texel_fusion_is_view_balanced_and_metric() -> None:
    uv = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.2], [1.2, 0.2]])
    feature = np.asarray([[1, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    view = np.asarray([0, 0, 1, 2])
    texel, descriptor, support, tokens = _fuse_plane_texels(
        uv, feature, view, cell_size_m=1.0, minimum_views=2
    )
    np.testing.assert_allclose(texel, [[0.225, 0.175]])
    np.testing.assert_array_equal(support, [2])
    np.testing.assert_array_equal(tokens, [3])
    expected = np.asarray([1.0, 1.0]); expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(descriptor[0], expected)


def test_plane_texel_fusion_is_stable_to_input_order() -> None:
    uv = np.asarray([[0.1, 0.1], [0.2, 0.2], [1.2, 0.2], [0.3, 0.2]])
    feature = np.eye(4, dtype=np.float32)
    view = np.asarray([0, 1, 2, 3])
    first = _fuse_plane_texels(uv, feature, view, cell_size_m=1.0, minimum_views=1)
    permutation = np.asarray([2, 0, 3, 1])
    second = _fuse_plane_texels(
        uv[permutation], feature[permutation], view[permutation], cell_size_m=1.0, minimum_views=1
    )
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right)


def test_plane_texel_prototypes_share_identity_and_preserve_view_modes() -> None:
    uv = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.3, 0.2], [1.2, 0.2]])
    feature = np.asarray([[1, 0], [0.9, 0.1], [0, 1], [1, 1]], np.float32)
    result = _fuse_plane_texel_prototypes(
        uv, feature, np.asarray([0, 1, 2, 3]), cell_size_m=1.0,
        minimum_views=2, maximum_prototypes=2,
    )
    texel, descriptor, identity, rank, support, tokens = result
    np.testing.assert_allclose(texel, [[0.2, 1.0 / 6.0], [0.2, 1.0 / 6.0]])
    np.testing.assert_array_equal(identity, [0, 0])
    np.testing.assert_array_equal(rank, [0, 1])
    np.testing.assert_array_equal(support, [3, 3])
    np.testing.assert_array_equal(tokens, [3, 3])
    assert float(descriptor[0] @ descriptor[1]) < 0.2
