import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_large_plane_loftr_pose import (
    _grid_rows,
    _plane_crop,
    _plane_pair,
)


def test_pixel_center_downscale_contract():
    xy = np.asarray([[.5, .5], [2.5, 2.5], [511.0, 287.0]])
    yy, xx, valid = _grid_rows(xy, factor=2.0, height=144, width=256)
    assert valid.tolist() == [True, True, True]
    assert xx.tolist() == [0, 1, 255]
    assert yy.tolist() == [0, 1, 143]


def test_plane_pair_rejects_matches_outside_finite_support():
    pair = {
        "query_xy": np.stack((np.arange(12) * 8 + 8, np.arange(12) * 4 + 8), axis=1).astype(float),
        "map_xy": np.stack((np.arange(12) * 8 + 8, np.arange(12) * 4 + 8), axis=1).astype(float),
        "confidence": np.ones((12,)),
    }
    query_mask = np.zeros((144, 256), bool)
    map_mask = np.zeros((72, 128), bool)
    query = {"grid_mask": query_mask, "grid_points": np.zeros((144, 256, 3))}
    mapping = {"grid_mask": map_mask, "grid_points": np.zeros((72, 128, 3))}
    source, target, audit = _plane_pair(pair, query, mapping)
    assert len(source) == len(target) == 0
    assert audit["accepted"] is False
    assert audit["masked_match_count"] == 0


def test_plane_crop_preserves_aspect_and_has_invertible_pixel_transform():
    image = np.arange(432 * 768, dtype=np.uint32).reshape(432, 768).astype(np.uint8)
    mask = np.zeros((144, 256), bool)
    mask[30:90, 40:180] = True
    crop, (scale, ox, oy) = _plane_crop(image, mask)
    assert crop.shape == (384, 384)
    assert scale > 0
    point = np.asarray([200.0, 150.0])
    encoded = point * scale + np.asarray([ox, oy])
    decoded = (encoded - np.asarray([ox, oy])) / scale
    assert np.allclose(decoded, point)
