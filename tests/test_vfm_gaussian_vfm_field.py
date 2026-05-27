import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import (
    GaussianVFMFieldConfig,
    GaussianVFMRenderConfig,
    GaussianVFMRenderResult,
    GaussianVFMFeatureView,
    GaussianVFMRayContributionConfig,
    GaussianVFMSource,
    aggregate_ray_contributed_gaussian_vfm_features,
    associate_landmarks_to_gaussians,
    export_gaussian_vfm_field_to_ply,
    load_gaussian_rgb_source_from_ply,
    merge_gaussian_vfm_fields,
    render_gaussian_rgb_image_soft,
    render_gaussian_vfm_feature_map_gsplat,
    render_gaussian_vfm_feature_map,
    GaussianVFMField,
)
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _landmark_index() -> LandmarkMapIndex:
    bank = SelectedTrackFeatureBank(
        feature_dim=2,
        tracks={
            10: TrackFeature(
                track_id=10,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.asarray([0.01, 0.01], dtype=np.float32),
                observation_count=4,
                mean_utility=1.0,
                observation_image_ids=("r0.png",),
            ),
            20: TrackFeature(
                track_id=20,
                mean_feature=np.asarray([0.0, 1.0], dtype=np.float32),
                variance=np.asarray([0.02, 0.02], dtype=np.float32),
                observation_count=5,
                mean_utility=1.0,
                observation_image_ids=("r1.png",),
            ),
        },
    )
    return LandmarkMapIndex.from_track_bank(
        bank,
        {
            10: np.asarray([0.0, 0.0, 2.0], dtype=np.float64),
            20: np.asarray([2.0, 0.0, 2.0], dtype=np.float64),
        },
    )


def test_associate_landmarks_to_gaussians_keeps_only_nearby_supported_gaussians():
    source = GaussianVFMSource(
        xyz=np.asarray(
            [
                [0.02, 0.0, 2.0],
                [2.03, 0.0, 2.0],
                [9.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        ),
        opacity=np.ones((3,), dtype=np.float32),
        scale=np.ones((3,), dtype=np.float32) * 0.1,
        gaussian_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )

    field = associate_landmarks_to_gaussians(
        source,
        _landmark_index(),
        GaussianVFMFieldConfig(max_distance=0.1, k_neighbors=2, min_support=1),
    )

    assert field.feature_dim == 2
    assert field.gaussian_indices.tolist() == [0, 1]
    assert field.nearest_track_ids.tolist() == [10, 20]
    assert field.support_counts.tolist() == [1, 1]
    np.testing.assert_allclose(field.features, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_render_gaussian_vfm_feature_map_uses_depth_order_for_overlapping_splats():
    field = associate_landmarks_to_gaussians(
        GaussianVFMSource(
            xyz=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float64,
            ),
            opacity=np.ones((2,), dtype=np.float32),
            scale=np.ones((2,), dtype=np.float32) * 0.1,
            gaussian_indices=np.asarray([0, 1], dtype=np.int64),
        ),
        LandmarkMapIndex(
            track_ids=np.asarray([1, 2], dtype=np.int64),
            xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64),
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            mean_variances=np.asarray([0.0, 0.0], dtype=np.float32),
            observation_counts=np.asarray([2, 2], dtype=np.int64),
            observation_image_ids=((), ()),
        ),
        GaussianVFMFieldConfig(max_distance=0.01, k_neighbors=1),
    )

    rendered = render_gaussian_vfm_feature_map(
        field,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=ColmapCamera(camera_id=1, model_id=1, width=5, height=5, params=(1.0, 1.0, 2.0, 2.0)),
        config=GaussianVFMRenderConfig(width=5, height=5, radius_px=0.75, depth_epsilon=0.01),
    )

    assert isinstance(rendered, GaussianVFMRenderResult)
    assert rendered.feature_map.shape == (2, 5, 5)
    assert rendered.xyz_map.shape == (5, 5, 3)
    assert rendered.visibility_mask[2, 2]
    assert rendered.dominant_gaussian_index[2, 2] == 0
    assert rendered.feature_map[:, 2, 2].tolist() == pytest.approx([1.0, 0.0])
    assert rendered.xyz_map[2, 2].tolist() == pytest.approx([0.0, 0.0, 1.0])


def _write_minimal_gaussian_ply(path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    rows = np.zeros((3,), dtype=dtype)
    rows["x"] = [0.0, 1.0, 2.0]
    rows["y"] = [0.0, 0.0, 0.0]
    rows["z"] = [1.0, 1.0, 1.0]
    rows["opacity"] = [1.0, 2.0, 3.0]
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = -2.0
    rows["rot_0"] = 1.0
    PlyData([PlyElement.describe(rows, "vertex")]).write(path)


def test_export_gaussian_vfm_field_to_ply_preserves_gaussian_rows_and_writes_loc_features(tmp_path):
    source_ply = tmp_path / "gaussians.ply"
    output_ply = tmp_path / "gaussians_loc.ply"
    _write_minimal_gaussian_ply(source_ply)
    field = associate_landmarks_to_gaussians(
        GaussianVFMSource(
            xyz=np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            opacity=np.ones((2,), dtype=np.float32),
            scale=np.ones((2,), dtype=np.float32),
            gaussian_indices=np.asarray([0, 2], dtype=np.int64),
        ),
        LandmarkMapIndex(
            track_ids=np.asarray([1, 2], dtype=np.int64),
            xyz=np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            mean_variances=np.asarray([0.0, 0.0], dtype=np.float32),
            observation_counts=np.asarray([2, 2], dtype=np.int64),
            observation_image_ids=((), ()),
        ),
        GaussianVFMFieldConfig(max_distance=0.01, k_neighbors=1),
    )

    export_gaussian_vfm_field_to_ply(source_ply, field, output_ply)

    vertex = PlyData.read(output_ply).elements[0]
    assert vertex.count == 3
    names = vertex.data.dtype.names
    assert "f_dc_0" in names
    assert "loc_0" in names
    assert "loc_1" in names
    np.testing.assert_allclose(np.asarray(vertex["loc_0"]), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(vertex["loc_1"]), np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_load_gaussian_rgb_source_and_soft_render(tmp_path):
    source_ply = tmp_path / "gaussians.ply"
    _write_minimal_gaussian_ply(source_ply)

    source = load_gaussian_rgb_source_from_ply(source_ply)
    rendered_rgb, alpha = render_gaussian_rgb_image_soft(
        source,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=ColmapCamera(camera_id=1, model_id=1, width=9, height=9, params=(4.0, 4.0, 4.0, 4.0)),
        config=GaussianVFMRenderConfig(width=9, height=9, radius_px=1.0),
    )

    assert source.rgb.shape == (3, 3)
    assert rendered_rgb.shape == (9, 9, 3)
    assert alpha.shape == (9, 9)
    assert np.any(alpha > 0.0)
    assert float(rendered_rgb.max()) <= 1.0


def test_render_gaussian_vfm_feature_map_gsplat_smoke():
    field = associate_landmarks_to_gaussians(
        GaussianVFMSource(
            xyz=np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64),
            opacity=np.ones((1,), dtype=np.float32),
            scale=np.ones((1,), dtype=np.float32) * 0.05,
            gaussian_indices=np.asarray([7], dtype=np.int64),
        ),
        LandmarkMapIndex(
            track_ids=np.asarray([1], dtype=np.int64),
            xyz=np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64),
            features=np.asarray([[1.0, 0.0]], dtype=np.float32),
            mean_variances=np.asarray([0.0], dtype=np.float32),
            observation_counts=np.asarray([2], dtype=np.int64),
            observation_image_ids=((),),
        ),
        GaussianVFMFieldConfig(max_distance=0.01, k_neighbors=1),
    )

    rendered = render_gaussian_vfm_feature_map_gsplat(
        field,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=ColmapCamera(camera_id=1, model_id=1, width=9, height=9, params=(4.0, 4.0, 4.0, 4.0)),
        config=GaussianVFMRenderConfig(width=9, height=9),
        device="cpu",
    )

    assert rendered.feature_map.shape == (2, 9, 9)
    assert rendered.xyz_map.shape == (9, 9, 3)
    assert rendered.visibility_mask.any()
    visible = rendered.feature_map[:, rendered.visibility_mask]
    assert np.max(visible[0]) > 0.5


def test_aggregate_ray_contributed_gaussian_vfm_features_assigns_token_features_to_visible_gaussians():
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=np.float64),
        opacity=np.ones((2,), dtype=np.float32),
        scale=np.ones((2,), dtype=np.float32) * 0.1,
        gaussian_indices=np.asarray([3, 5], dtype=np.int64),
    )
    feature_map = np.zeros((2, 3, 3), dtype=np.float32)
    feature_map[:, 1, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    feature_map[:, 1, 2] = np.asarray([0.0, 1.0], dtype=np.float32)

    field = aggregate_ray_contributed_gaussian_vfm_features(
        source,
        [
            GaussianVFMFeatureView(
                image_id="r0.png",
                feature_map=feature_map,
                pose_w2c=np.eye(4, dtype=np.float64),
                camera=ColmapCamera(camera_id=1, model_id=1, width=3, height=3, params=(4.0, 4.0, 1.0, 1.0)),
            )
        ],
        GaussianVFMRayContributionConfig(radius_px=0.6, min_samples=1, l2_normalize_observations=False),
    )

    assert field.gaussian_indices.tolist() == [3, 5]
    assert field.support_counts.tolist() == [1, 1]
    np.testing.assert_allclose(field.features, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_merge_gaussian_vfm_fields_prefers_primary_and_fills_fallback():
    primary = GaussianVFMField(
        xyz=np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        opacity=np.ones((2,), dtype=np.float32),
        scale=np.ones((2,), dtype=np.float32),
        gaussian_indices=np.asarray([0, 2], dtype=np.int64),
        nearest_track_ids=np.asarray([10, 20], dtype=np.int64),
        support_counts=np.asarray([3, 4], dtype=np.int64),
        mean_distances=np.asarray([0.01, 0.02], dtype=np.float32),
        metadata={"source": "landmark"},
    )
    fallback = GaussianVFMField(
        xyz=np.asarray([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
        features=np.asarray([[0.0, 1.0], [0.0, 0.9]], dtype=np.float32),
        opacity=np.ones((2,), dtype=np.float32) * 0.5,
        scale=np.ones((2,), dtype=np.float32) * 2.0,
        gaussian_indices=np.asarray([1, 2], dtype=np.int64),
        nearest_track_ids=np.asarray([-1, -1], dtype=np.int64),
        support_counts=np.asarray([7, 8], dtype=np.int64),
        mean_distances=np.asarray([0.3, 0.4], dtype=np.float32),
        metadata={"source": "ray"},
    )

    hybrid = merge_gaussian_vfm_fields(primary, fallback)

    assert hybrid.gaussian_indices.tolist() == [0, 1, 2]
    assert hybrid.nearest_track_ids.tolist() == [10, -1, 20]
    assert hybrid.support_counts.tolist() == [3, 7, 4]
    np.testing.assert_allclose(
        hybrid.features,
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
    )
    assert hybrid.metadata["primary_count"] == 2
    assert hybrid.metadata["fallback_added_count"] == 1
