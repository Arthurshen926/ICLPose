import numpy as np

from feature_extract.vfm.dense_gaussian_field_diagnostics import (
    encode_feature_map_with_selector,
    gaussian_field_coverage_stats,
    gaussian_field_to_semidense_anchor_map,
    pca_feature_rgb,
    visibility_overlay_rgb,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField, GaussianVFMSource


class _ToySelector:
    def encode_rows(self, rows, device="cpu", batch_size=65536):
        values = np.asarray(rows, dtype=np.float32)
        return values[:, :2] + values[:, 2:4]


def test_encode_feature_map_with_selector_preserves_grid_shape() -> None:
    feature_map = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)

    encoded = encode_feature_map_with_selector(feature_map, _ToySelector(), device="cpu", batch_size=4)

    assert encoded.shape == (2, 2, 3)
    assert np.allclose(encoded[:, 0, 0], feature_map[:2, 0, 0] + feature_map[2:4, 0, 0])


def test_gaussian_field_coverage_stats_reports_feature_bearing_ratio() -> None:
    source = GaussianVFMSource(
        xyz=np.zeros((5, 3), dtype=np.float64),
        opacity=np.ones((5,), dtype=np.float32),
        scale=np.ones((5,), dtype=np.float32),
        gaussian_indices=np.arange(5, dtype=np.int64),
    )
    field = GaussianVFMField(
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.ones((2, 3), dtype=np.float32),
        opacity=np.ones((2,), dtype=np.float32),
        scale=np.ones((2,), dtype=np.float32),
        gaussian_indices=np.asarray([1, 3], dtype=np.int64),
        nearest_track_ids=np.asarray([-1, -1], dtype=np.int64),
        support_counts=np.asarray([2, 4], dtype=np.int64),
        mean_distances=np.asarray([0.5, 0.25], dtype=np.float32),
    )

    stats = gaussian_field_coverage_stats(source, field)

    assert stats["source_gaussian_count"] == 5
    assert stats["feature_bearing_gaussian_count"] == 2
    assert stats["feature_bearing_fraction"] == 0.4
    assert stats["mean_samples"] == 3.0


def test_visibility_overlay_rgb_blends_only_visible_pixels() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.asarray([[True, False], [False, True]], dtype=bool)

    overlay = visibility_overlay_rgb(image, mask, color=(100, 50, 0), alpha=0.5)

    assert overlay[0, 0].tolist() == [50, 25, 0]
    assert overlay[0, 1].tolist() == [0, 0, 0]
    assert overlay[1, 1].tolist() == [50, 25, 0]


def test_pca_feature_rgb_respects_visibility_mask() -> None:
    features = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[True, False], [True, True]], dtype=bool)

    rgb = pca_feature_rgb(features, mask)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 1].tolist() == [0, 0, 0]
    assert int(rgb[mask].sum()) > 0


def test_gaussian_field_to_semidense_anchor_map_preserves_feature_bearing_gaussians() -> None:
    field = GaussianVFMField(
        xyz=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        opacity=np.asarray([0.5, 1.0], dtype=np.float32),
        scale=np.asarray([0.01, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([7, 9], dtype=np.int64),
        nearest_track_ids=np.asarray([-1, -1], dtype=np.int64),
        support_counts=np.asarray([2, 8], dtype=np.int64),
        mean_distances=np.asarray([1.5, 0.5], dtype=np.float32),
    )

    anchor_map = gaussian_field_to_semidense_anchor_map(field)

    assert len(anchor_map) == 2
    assert anchor_map.source_types.tolist() == ["gaussian_ray", "gaussian_ray"]
    assert anchor_map.source_gaussian_indices.tolist() == [7, 9]
    assert anchor_map.support_counts.tolist() == [2, 8]
    assert np.allclose(np.linalg.norm(anchor_map.features, axis=1), 1.0)
    assert float(anchor_map.quality_scores[1]) > float(anchor_map.quality_scores[0])
