import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, GaussianVFMSource
from feature_extract.vfm.vfm_2dgs_mapping import (
    Vfm2DgsDescriptorIndex,
    Vfm2DgsContributionBuffer,
    Vfm2DgsObservationBank,
    Vfm2DgsAnchorFusionConfig,
    Vfm2DgsMappingConfig,
    Vfm2DgsAnchorMap,
    SurfaceElementMap,
    TokenSurfaceObservation,
    _signed_safe_denominator,
    _surface_element_quaternions_and_scales,
    build_anchor_covisibility_graph,
    build_anchor_descriptor_index,
    build_surface_elements_from_2dgs_source,
    compute_renderer_token_surface_contribution_buffer,
    compute_token_surface_contribution_buffer,
    compute_token_surface_observations,
    estimate_virtual_cell_max_scale_for_token_projection,
    merge_vfm_2dgs_observation_banks,
    fuse_token_surface_observations,
    surface_supported_token_indices_from_hits,
    spatial_nms_anchor_map,
    token_surface_observations_from_contribution_buffer,
    vfm_2dgs_anchor_map_to_semidense,
)
from feature_extract.vfm.vfm_2dgs_diagnostics import token_purity_diagnostic_grid
from feature_extract.vfm.gaussian_vfm_field import _quaternion_rotation_matrices


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(1.0, 1.0, 2.0, 2.0))


def _source() -> GaussianVFMSource:
    return GaussianVFMSource(
        xyz=np.asarray([[-0.5, 0.0, 4.0], [0.5, 0.0, 4.0], [3.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        scale=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scale_xyz=np.asarray([[0.1, 0.2, 0.1], [0.1, 0.2, 0.1], [0.1, 0.2, 0.1]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )


def _view(image_id: str = "ref.png") -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    fmap[:, 2, 3] = np.asarray([0.8, 0.2, 0.0], dtype=np.float32)
    return GaussianVFMFeatureView(
        image_id=image_id,
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def _single_token_view(image_id: str, feature) -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray(feature, dtype=np.float32)
    return GaussianVFMFeatureView(
        image_id=image_id,
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def _projected_source_for_grid() -> GaussianVFMSource:
    xyz = []
    for u, v in [(0.0, 0.0), (1.0, 1.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)]:
        xyz.append([(u - 2.0) * 4.0, (v - 2.0) * 4.0, 4.0])
    return GaussianVFMSource(
        xyz=np.asarray(xyz, dtype=np.float64),
        opacity=np.ones((5,), dtype=np.float32),
        scale=np.ones((5,), dtype=np.float32) * 0.1,
        gaussian_indices=np.arange(100, 105, dtype=np.int64),
        scale_xyz=np.ones((5, 3), dtype=np.float32) * 0.1,
        normal=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (5, 1)),
    )


def _grid_view() -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 0, 0] = [10.0, 0.0, 0.0]
    fmap[:, 0, 1] = [9.0, 0.0, 0.0]
    fmap[:, 1, 0] = [8.0, 0.0, 0.0]
    fmap[:, 1, 1] = [7.0, 0.0, 0.0]
    fmap[:, 0, 3] = [4.0, 0.0, 0.0]
    fmap[:, 3, 0] = [3.0, 0.0, 0.0]
    fmap[:, 3, 3] = [2.0, 0.0, 0.0]
    return GaussianVFMFeatureView(
        image_id="grid.png",
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def _manual_anchor_map(
    centers: np.ndarray,
    quality_scores: np.ndarray,
    observed_view_ids,
    anchor_ids=None,
) -> Vfm2DgsAnchorMap:
    centers = np.asarray(centers, dtype=np.float64)
    count = centers.shape[0]
    if anchor_ids is None:
        anchor_ids = np.arange(count, dtype=np.int64)
    support_offsets = np.arange(count + 1, dtype=np.int64)
    features = np.zeros((count, 2), dtype=np.float32)
    if count:
        features[:, 0] = 1.0
    return Vfm2DgsAnchorMap(
        anchor_ids=np.asarray(anchor_ids, dtype=np.int64),
        centers=centers,
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None, :, :], (count, 1, 1)),
        features=features,
        feature_variances=np.zeros((count,), dtype=np.float32),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        purity_scores=np.ones((count,), dtype=np.float32),
        observation_counts=np.ones((count,), dtype=np.int64),
        surface_support_counts=np.ones((count,), dtype=np.int64),
        support_offsets=support_offsets,
        support_element_ids=np.arange(count, dtype=np.int64),
        support_weights=np.ones((count,), dtype=np.float32),
        observed_view_ids=tuple(tuple(row) for row in observed_view_ids),
    )


def test_vfm_2dgs_semidense_export_can_expand_feature_prototypes() -> None:
    anchor_map = Vfm2DgsAnchorMap(
        anchor_ids=np.asarray([7, 8], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None, :, :], (2, 1, 1)),
        features=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        feature_prototypes=np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        feature_prototype_counts=np.asarray([2, 3], dtype=np.int64),
        feature_variances=np.asarray([0.1, 0.2], dtype=np.float32),
        quality_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        purity_scores=np.asarray([0.7, 0.6], dtype=np.float32),
        observation_counts=np.asarray([4, 5], dtype=np.int64),
        surface_support_counts=np.asarray([2, 3], dtype=np.int64),
        support_offsets=np.asarray([0, 1, 3], dtype=np.int64),
        support_element_ids=np.asarray([11, 12, 13], dtype=np.int64),
        support_parent_gaussian_indices=np.asarray([101, 102, 102], dtype=np.int64),
        support_weights=np.asarray([1.0, 0.4, 0.6], dtype=np.float32),
        observed_view_ids=(("a.png", "b.png"), ("b.png", "c.png")),
    )

    semidense = vfm_2dgs_anchor_map_to_semidense(anchor_map, descriptor_mode="prototypes")

    assert len(semidense) == 5
    assert semidense.anchor_ids.tolist() == [7000, 7001, 8000, 8001, 8002]
    assert semidense.source_types.tolist() == ["vfm_2dgs_prototype"] * 5
    assert semidense.source_gaussian_indices.tolist() == [101, 101, 102, 102, 102]
    assert semidense.observation_image_ids[0] == ("a.png", "b.png")
    assert semidense.observation_counts.tolist() == [4, 4, 5, 5, 5]
    np.testing.assert_allclose(semidense.features[1], np.asarray([0.8, 0.2, 0.0], dtype=np.float32))
    index = semidense.to_landmark_index()
    assert index.track_ids.tolist()[0] == index.track_ids.tolist()[1]
    assert len(set(index.track_ids.tolist()[2:])) == 1


def test_vfm_2dgs_semidense_export_cli_accepts_descriptor_mode(tmp_path) -> None:
    from feature_extract.tools.vfm.export_vfm_2dgs_to_semidense import main
    from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap

    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        quality_scores=np.asarray([0.8], dtype=np.float32),
        observed_view_ids=(("a.png",),),
        anchor_ids=np.asarray([3], dtype=np.int64),
    )
    anchor_npz = tmp_path / "anchor_map.npz"
    output_npz = tmp_path / "semidense.npz"
    summary_json = tmp_path / "summary.json"
    anchor_map.save_npz(anchor_npz)

    main(
        [
            "--anchor_map",
            str(anchor_npz),
            "--output_npz",
            str(output_npz),
            "--summary_json",
            str(summary_json),
            "--descriptor_mode",
            "prototypes",
        ]
    )

    semidense = SemiDenseAnchorMap.load_npz(output_npz)
    assert semidense.metadata["descriptor_mode"] == "prototypes"
    assert semidense.source_types.tolist() == ["vfm_2dgs_prototype"]
    assert '"descriptor_mode": "prototypes"' in summary_json.read_text()


def test_surface_elements_keep_2dgs_geometry_and_adjacency() -> None:
    elements = build_surface_elements_from_2dgs_source(
        _source(),
        adjacency_radius=1.1,
        normal_cosine_threshold=0.9,
    )

    assert elements.centers.shape == (3, 3)
    assert elements.element_ids.tolist() == [10, 11, 12]
    np.testing.assert_allclose(elements.normals[0], np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    assert elements.adjacency[0].tolist() == [1]
    assert elements.adjacency[1].tolist() == [0]
    assert elements.adjacency[2].tolist() == []


def test_surface_adjacency_uses_disk_support_extent_not_only_center_distance() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.35, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.2, 0.2], dtype=np.float32),
        gaussian_indices=np.asarray([30, 31], dtype=np.int64),
        scale_xyz=np.asarray([[0.2, 0.2, 0.02], [0.2, 0.2, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    elements = build_surface_elements_from_2dgs_source(
        source,
        adjacency_radius=0.0,
        normal_cosine_threshold=0.9,
    )

    assert elements.adjacency[0].tolist() == [1]
    assert elements.adjacency[1].tolist() == [0]


def test_large_2dgs_disk_is_split_into_virtual_surface_cells_and_round_trips(tmp_path) -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        gaussian_indices=np.asarray([42], dtype=np.int64),
        scale_xyz=np.asarray([[1.0, 0.5, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    elements = build_surface_elements_from_2dgs_source(
        source,
        virtual_cell_max_scale=0.25,
        virtual_cell_grid_cap=8,
        adjacency_radius=0.4,
    )

    assert len(elements) > 1
    assert set(elements.parent_gaussian_indices.tolist()) == {42}
    assert elements.tangent1.shape == elements.centers.shape
    assert elements.tangent2.shape == elements.centers.shape
    assert np.all(elements.scale1 <= 0.25 + 1e-6)
    assert np.all(elements.scale2 <= 0.25 + 1e-6)
    assert len(set(elements.element_ids.tolist())) == len(elements)

    output = tmp_path / "surface_elements.npz"
    elements.save_npz(output)
    loaded = SurfaceElementMap.load_npz(output)

    assert len(loaded) == len(elements)
    np.testing.assert_allclose(loaded.centers, elements.centers)
    np.testing.assert_allclose(loaded.tangent1, elements.tangent1)
    np.testing.assert_allclose(loaded.tangent2, elements.tangent2)
    assert loaded.parent_gaussian_indices.tolist() == elements.parent_gaussian_indices.tolist()


def test_projected_token_virtual_cell_scale_estimator_tracks_depth_and_focal() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 8.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([1.0, 1.0], dtype=np.float32),
        gaussian_indices=np.asarray([1, 2], dtype=np.int64),
        scale_xyz=np.asarray([[1.0, 0.5, 0.02], [1.0, 0.5, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    view = GaussianVFMFeatureView(
        image_id="ref.png",
        feature_map=np.zeros((3, 4, 4), dtype=np.float32),
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(2.0, 2.0, 2.0, 2.0)),
    )

    threshold = estimate_virtual_cell_max_scale_for_token_projection(
        source,
        [view],
        target_projected_radius_px=1.0,
        depth_quantile=0.5,
    )

    assert np.isclose(threshold, 3.0)


def test_2dgs_surface_basis_is_right_handed_when_scale_axes_are_swapped() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([0.1], dtype=np.float32),
        gaussian_indices=np.asarray([7], dtype=np.int64),
        scale_xyz=np.asarray([[0.1, 0.3, 0.01]], dtype=np.float32),
        rotation=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    elements = build_surface_elements_from_2dgs_source(source)

    basis = np.stack([elements.tangent1, elements.tangent2, elements.normals], axis=2)
    assert float(np.linalg.det(basis[0])) > 0.99
    quats, _scales = _surface_element_quaternions_and_scales(elements)
    reconstructed = _quaternion_rotation_matrices(quats)
    np.testing.assert_allclose(reconstructed[0], basis[0], atol=1e-5)


def test_signed_safe_denominator_preserves_negative_renderer_denominators() -> None:
    import torch

    values = torch.tensor([2.0, -3.0, 0.0, 1e-15, -1e-15], dtype=torch.float32)

    safe = _signed_safe_denominator(values, eps=1e-6)

    expected = torch.tensor([2.0, -3.0, 1e-6, 1e-6, -1e-6], dtype=torch.float32)
    torch.testing.assert_close(safe, expected)


def test_token_surface_observation_uses_soft_multi_element_responsibility() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            min_component_concentration=0.5,
            max_elements_per_token=8,
            min_responsibility=0.01,
        ),
    )

    assert len(observations) == 1
    obs = observations[0]
    assert obs.image_id == "ref.png"
    assert obs.token_xy.tolist() == [2.0, 2.0]
    assert len(obs.element_ids) >= 2
    assert set(obs.element_ids.tolist()).issuperset({10, 11})
    assert np.isclose(float(np.sum(obs.element_weights)), 1.0)
    assert obs.quality_score > 0.0


def test_full_component_concentration_is_checked_before_truncating_token_support() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.2, 0.0, 4.0], [0.2, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([201, 202], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.01)

    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            support_mode="surface_component",
            max_elements_per_token=1,
            min_component_concentration=0.0,
            min_full_component_concentration=0.75,
        ),
    )

    assert observations == []


def test_contribution_buffer_records_full_support_ambiguity_rejections() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.2, 0.0, 4.0], [0.2, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([301, 302], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.01)

    buffer = compute_token_surface_contribution_buffer(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            support_mode="surface_component",
            max_elements_per_token=1,
            min_component_concentration=0.0,
            min_full_component_concentration=0.75,
        ),
    )

    assert len(buffer) == 0
    assert buffer.metadata["rejection_stats"]["full_component_concentration"] == 1


def test_full_support_element_capacity_is_checked_before_topk_truncation() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray(
            [[-0.15, 0.0, 4.0], [0.0, 0.0, 4.0], [0.15, 0.0, 4.0]],
            dtype=np.float64,
        ),
        opacity=np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([401, 402, 403], dtype=np.int64),
        scale_xyz=np.ones((3, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)

    buffer = compute_token_surface_contribution_buffer(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            max_elements_per_token=1,
            max_full_support_elements=2,
            min_component_concentration=0.0,
        ),
    )

    assert len(buffer) == 0
    assert buffer.metadata["rejection_stats"]["full_support_elements"] == 1


def test_mixed_token_can_be_kept_as_weak_observation_without_descriptor_weight() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.2, 0.0, 4.0], [0.2, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([501, 502], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.01)

    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            support_mode="surface_component",
            max_elements_per_token=8,
            min_component_concentration=0.0,
            min_full_component_concentration=0.75,
            weak_observation_mode="keep",
            weak_min_full_component_concentration=0.45,
        ),
    )

    assert len(observations) == 1
    assert observations[0].observation_strength == "weak"
    assert observations[0].descriptor_weight == 0.0
    assert observations[0].quality_score > 0.0


def test_weak_observations_do_not_pollute_descriptor_bearing_support() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    strong = TokenSurfaceObservation(
        image_id="strong.png",
        token_index=1,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.9,
        component_concentration=0.9,
        quality_score=1.0,
        observation_strength="strong",
        descriptor_weight=1.0,
    )
    weak = TokenSurfaceObservation(
        image_id="weak.png",
        token_index=2,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 12], dtype=np.int64),
        element_weights=np.asarray([0.2, 0.8], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.5,
        component_concentration=0.5,
        quality_score=0.5,
        observation_strength="weak",
        descriptor_weight=0.0,
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        [strong, weak],
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.1, min_observations=1),
    )

    assert len(anchor_map) == 1
    np.testing.assert_allclose(anchor_map.features[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert set(anchor_map.support_element_ids.tolist()) == {10, 11}
    assert anchor_map.observation_counts.tolist() == [2]


def test_anchor_support_core_keeps_repeated_descriptor_surface_elements() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    first = TokenSurfaceObservation(
        image_id="first.png",
        token_index=1,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.7, 0.3], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.9,
        component_concentration=0.9,
        quality_score=1.0,
        observation_strength="strong",
        descriptor_weight=1.0,
    )
    second = TokenSurfaceObservation(
        image_id="second.png",
        token_index=2,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 12], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.9,
        component_concentration=0.9,
        quality_score=1.0,
        observation_strength="strong",
        descriptor_weight=1.0,
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        [first, second],
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.1,
            min_observations=1,
            support_core_min_observations=2,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.support_element_ids.tolist() == [10]
    assert anchor_map.surface_support_counts.tolist() == [1]


def test_surface_first_fusion_groups_by_dominant_surface_seed() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    first = TokenSurfaceObservation(
        image_id="first.png",
        token_index=1,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.8, 0.2], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.9,
        component_concentration=0.9,
        quality_score=1.0,
        observation_strength="strong",
        descriptor_weight=1.0,
    )
    second = TokenSurfaceObservation(
        image_id="second.png",
        token_index=2,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.2, 0.8], dtype=np.float32),
        center=center,
        normal=normal,
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.9,
        component_concentration=0.9,
        quality_score=1.0,
        observation_strength="strong",
        descriptor_weight=1.0,
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        [first, second],
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="surface_first",
            min_observations=1,
            surface_first_max_seeds_per_observation=1,
        ),
    )

    assert len(anchor_map) == 2
    assert anchor_map.observation_counts.tolist() == [1, 1]
    assert {tuple(ids) for ids in np.split(anchor_map.support_element_ids, anchor_map.support_offsets[1:-1])} == {
        (10, 11),
    }


def test_surface_first_fusion_applies_stable_support_core() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id=f"view{idx}.png",
            token_index=idx,
            token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=np.asarray([10, other], dtype=np.int64),
            element_weights=np.asarray([0.8, 0.2], dtype=np.float32),
            center=center,
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=0.9,
            component_concentration=0.9,
            quality_score=1.0,
            observation_strength="strong",
            descriptor_weight=1.0,
        )
        for idx, other in enumerate([11, 12], start=1)
    ]

    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="surface_first",
            min_observations=2,
            support_core_min_observations=2,
            support_core_min_fraction=0.5,
            surface_first_max_seeds_per_observation=1,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.observation_counts.tolist() == [2]
    assert anchor_map.support_element_ids.tolist() == [10]
    assert anchor_map.surface_support_counts.tolist() == [1]


def test_fusion_can_reject_weak_only_anchor_states() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    weak = TokenSurfaceObservation(
        image_id="weak.png",
        token_index=2,
        token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.5, 0.5], dtype=np.float32),
        center=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.5,
        component_concentration=0.5,
        quality_score=0.5,
        observation_strength="weak",
        descriptor_weight=0.0,
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        [weak],
        Vfm2DgsAnchorFusionConfig(min_observations=1, min_descriptor_observations=1),
    )

    assert len(anchor_map) == 0


def test_grid_balanced_token_selection_spreads_surface_observations() -> None:
    elements = build_surface_elements_from_2dgs_source(_projected_source_for_grid(), adjacency_radius=1.1)
    observations = compute_token_surface_observations(
        elements,
        _grid_view(),
        Vfm2DgsMappingConfig(
            token_selection_mode="grid_top",
            token_grid_rows=2,
            token_grid_cols=2,
            max_tokens_per_cell=1,
            token_top_fraction=0.25,
            footprint_radius_px=0.75,
            max_projected_disk_radius_px=0.5,
            depth_epsilon=0.1,
        ),
    )

    selected_cells = {(int(obs.token_xy[0]) // 2, int(obs.token_xy[1]) // 2) for obs in observations}

    assert selected_cells == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_surface_supported_token_indices_come_from_renderer_contribution_not_saliency() -> None:
    feature_map = np.zeros((3, 4, 4), dtype=np.float32)
    feature_map[:, 0, 0] = [100.0, 0.0, 0.0]
    pixel_ids = np.asarray([5, 5, 10, 10, 10, 15], dtype=np.int64)
    weights = np.asarray([0.1, 0.2, 0.1, 0.2, 0.3, 0.01], dtype=np.float32)

    selected = surface_supported_token_indices_from_hits(
        feature_map,
        pixel_ids,
        weights,
        Vfm2DgsMappingConfig(
            token_selection_mode="surface",
            min_surface_token_contribution=0.05,
            max_surface_tokens=2,
        ),
    )

    assert selected.tolist() == [10, 5]


def test_surface_supported_token_indices_can_balance_contribution_and_saliency() -> None:
    feature_map = np.zeros((3, 4, 4), dtype=np.float32)
    feature_map[:, 5 // 4, 5 % 4] = [100.0, 0.0, 0.0]
    feature_map[:, 10 // 4, 10 % 4] = [1.0, 0.0, 0.0]
    pixel_ids = np.asarray([5, 10], dtype=np.int64)
    weights = np.asarray([0.2, 0.3], dtype=np.float32)

    selected = surface_supported_token_indices_from_hits(
        feature_map,
        pixel_ids,
        weights,
        Vfm2DgsMappingConfig(
            token_selection_mode="surface",
            surface_token_saliency_power=1.0,
            max_surface_tokens=1,
        ),
    )

    assert selected.tolist() == [5]


def test_token_purity_diagnostic_grid_marks_cross_surface_suspects() -> None:
    buffer = Vfm2DgsContributionBuffer(
        image_id="a.png",
        renderer="unit",
        token_indices=np.asarray([0, 5], dtype=np.int64),
        token_xy=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        support_offsets=np.asarray([0, 1, 4], dtype=np.int64),
        element_ids=np.asarray([10, 20, 21, 22], dtype=np.int64),
        element_weights=np.asarray([1.0, 0.34, 0.33, 0.33], dtype=np.float32),
        purity_scores=np.asarray([0.9, 0.4], dtype=np.float32),
        component_concentrations=np.asarray([0.95, 0.45], dtype=np.float32),
        quality_scores=np.asarray([0.8, 0.2], dtype=np.float32),
        top_alpha=np.asarray([0.9, 0.25], dtype=np.float32),
        alpha_entropy=np.asarray([0.1, 0.8], dtype=np.float32),
    )

    diagnostic = token_purity_diagnostic_grid(
        buffer,
        grid_shape=(2, 3),
        component_threshold=0.6,
        entropy_threshold=0.7,
        support_threshold=2,
    )

    assert diagnostic["token_count"] == 2
    assert diagnostic["cross_surface_suspect_count"] == 1
    assert diagnostic["cross_surface_suspect_fraction"] == 0.5
    assert diagnostic["low_component_count"] == 1
    assert diagnostic["high_entropy_count"] == 1
    assert diagnostic["large_support_count"] == 1
    assert diagnostic["support_count_grid"][1, 2] == 3
    assert bool(diagnostic["cross_surface_suspect_grid"][1, 2])


def test_purity_components_reduce_quality_for_mixed_depth_and_normals() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.2, 0.0, 4.5]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([20, 21], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1, normal_cosine_threshold=-1.0)
    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.2,
            depth_epsilon=1.0,
            depth_purity_sigma=0.1,
            normal_purity_power=1.0,
            min_purity=0.0,
        ),
    )

    obs = observations[0]

    assert obs.purity_score < 1.0
    assert obs.purity_components["depth"] < 1.0
    assert obs.purity_components["normal"] < 1.0


def test_purity_components_downweight_tokens_with_too_much_effective_surface_support() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray(
            [[-0.2, 0.0, 4.0], [0.0, 0.0, 4.0], [0.2, 0.0, 4.0]],
            dtype=np.float64,
        ),
        opacity=np.asarray([0.9, 0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([701, 702, 703], dtype=np.int64),
        scale_xyz=np.ones((3, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)
    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            max_elements_per_token=8,
            min_component_concentration=0.0,
            max_effective_support_elements=1.1,
        ),
    )

    assert len(observations) == 1
    assert observations[0].purity_components["effective_support"] > 1.1
    assert observations[0].purity_components["capacity"] < 1.0


def test_bidirectional_responsibility_penalizes_large_disk_overreach() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.1, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.95, 0.95], dtype=np.float32),
        scale=np.asarray([32.0, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([30, 31], dtype=np.int64),
        scale_xyz=np.asarray([[32.0, 32.0, 0.02], [0.1, 0.1, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)
    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            max_projected_disk_radius_px=8.0,
            depth_epsilon=0.1,
            bidirectional_lambda=0.5,
            element_coverage_power=1.0,
            min_responsibility=0.0,
        ),
    )

    obs = observations[0]
    weights = {int(idx): float(weight) for idx, weight in zip(obs.element_ids.tolist(), obs.element_weights.tolist())}

    assert weights[31] > weights[30]


def test_token_anchor_competition_keeps_only_winning_surface_element() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.15, 0.0, 4.0], [0.15, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.95, 0.95], dtype=np.float32),
        scale=np.asarray([0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([801, 802], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)

    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            max_elements_per_token=8,
            min_component_concentration=0.0,
            token_anchor_competition="winner_element",
        ),
    )

    assert len(observations) == 1
    assert observations[0].element_ids.shape == (1,)
    np.testing.assert_allclose(observations[0].element_weights, np.asarray([1.0], dtype=np.float32))


def test_token_anchor_competition_can_keep_winner_local_neighborhood() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[-0.15, 0.0, 4.0], [0.15, 0.0, 4.0], [3.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.95, 0.95, 0.95], dtype=np.float32),
        scale=np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        gaussian_indices=np.asarray([811, 812, 813], dtype=np.int64),
        scale_xyz=np.ones((3, 3), dtype=np.float32) * 0.1,
        normal=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.5)

    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            max_elements_per_token=8,
            min_component_concentration=0.0,
            token_anchor_competition="winner_neighborhood",
            token_anchor_neighborhood_hops=1,
            token_anchor_max_support_elements=2,
        ),
    )

    assert len(observations) == 1
    assert observations[0].element_ids.shape == (2,)
    assert 813 not in observations[0].element_ids.tolist()
    np.testing.assert_allclose(np.sum(observations[0].element_weights), 1.0)


def test_footprint_sampling_collects_surface_elements_beyond_token_center_radius() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [3.2, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.05, 0.05], dtype=np.float32),
        gaussian_indices=np.asarray([50, 51], dtype=np.int64),
        scale_xyz=np.ones((2, 3), dtype=np.float32) * 0.05,
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)

    center_only = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=0.55,
            max_projected_disk_radius_px=0.0,
            depth_epsilon=0.1,
            footprint_sample_grid=1,
            min_responsibility=0.0,
        ),
    )[0]
    sampled = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=0.55,
            max_projected_disk_radius_px=0.0,
            depth_epsilon=0.1,
            footprint_sample_grid=3,
            footprint_sample_extent_px=1.0,
            min_responsibility=0.0,
        ),
    )[0]

    assert 51 not in center_only.element_ids.tolist()
    assert 51 in sampled.element_ids.tolist()


def test_contribution_buffer_preserves_multi_element_token_weights_and_round_trips(tmp_path) -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    cfg = Vfm2DgsMappingConfig(
        token_top_fraction=0.10,
        footprint_radius_px=1.1,
        depth_epsilon=0.1,
        min_component_concentration=0.5,
        max_elements_per_token=8,
        min_responsibility=0.01,
    )

    buffer = compute_token_surface_contribution_buffer(elements, _view(), cfg)
    direct = compute_token_surface_observations(elements, _view(), cfg)
    restored = token_surface_observations_from_contribution_buffer(elements, _view(), buffer, cfg)

    assert len(buffer) == 1
    assert buffer.renderer == "projection_depth_soft"
    assert buffer.token_xy.tolist() == [[2.0, 2.0]]
    assert set(buffer.element_ids.tolist()).issuperset({10, 11})
    np.testing.assert_allclose(np.sum(buffer.element_weights), 1.0)
    assert float(buffer.top_alpha[0]) > 0.0
    assert len(restored) == len(direct)
    assert restored[0].element_ids.tolist() == direct[0].element_ids.tolist()
    np.testing.assert_allclose(restored[0].element_weights, direct[0].element_weights)

    output = tmp_path / "contribution_buffer.npz"
    buffer.save_npz(output)
    loaded = buffer.load_npz(output)

    assert len(loaded) == len(buffer)
    assert loaded.image_id == buffer.image_id
    assert loaded.renderer == buffer.renderer
    assert loaded.support_offsets.tolist() == buffer.support_offsets.tolist()
    assert loaded.element_ids.tolist() == buffer.element_ids.tolist()
    np.testing.assert_allclose(loaded.element_weights, buffer.element_weights)


def test_renderer_contribution_buffer_uses_alpha_compositing_order() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.25, 0.25], dtype=np.float32),
        gaussian_indices=np.asarray([100, 101], dtype=np.int64),
        scale_xyz=np.asarray([[0.25, 0.25, 1e-4], [0.25, 0.25, 1e-4]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=1.1)
    view = _view()

    buffer = compute_renderer_token_surface_contribution_buffer(
        elements,
        view,
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            min_component_concentration=0.0,
            min_purity=0.0,
            min_responsibility=0.0,
        ),
        device="cuda",
        renderer="gsplat_2dgs",
    )

    assert len(buffer) == 1
    weights = {
        int(element_id): float(weight)
        for element_id, weight in zip(buffer.element_ids.tolist(), buffer.element_weights.tolist())
    }
    assert buffer.renderer in {"gsplat_2dgs", "projection_depth_soft"}
    assert set(weights).issuperset({100, 101})
    assert weights[100] > weights[101]
    assert float(buffer.top_alpha[0]) > 0.5


def test_observation_bank_persists_token_features_geometry_and_support(tmp_path) -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    observations = compute_token_surface_observations(
        elements,
        _view(),
        Vfm2DgsMappingConfig(
            token_top_fraction=0.10,
            footprint_radius_px=1.1,
            depth_epsilon=0.1,
            min_component_concentration=0.5,
        ),
    )

    bank = Vfm2DgsObservationBank.from_observations(observations)

    assert len(bank) == len(observations)
    assert bank.features.shape[0] == len(observations)
    assert bank.centers.shape == (len(observations), 3)
    assert bank.view_directions.shape == (len(observations), 3)
    assert bank.support_offsets[-1] == bank.element_ids.shape[0]

    output = tmp_path / "observation_bank.npz"
    bank.save_npz(output)
    loaded = Vfm2DgsObservationBank.load_npz(output)
    restored = loaded.to_observations()

    assert loaded.image_ids == bank.image_ids
    np.testing.assert_allclose(loaded.features, bank.features)
    assert restored[0].element_ids.tolist() == observations[0].element_ids.tolist()
    np.testing.assert_allclose(restored[0].center, observations[0].center)


def test_observation_bank_merge_deduplicates_by_view_token_and_support_and_keeps_source_labels(tmp_path) -> None:
    duplicate_low_quality = TokenSurfaceObservation(
        image_id="ref.png",
        source_id="broad",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.4,
        component_concentration=0.5,
        quality_score=0.2,
    )
    duplicate_high_quality = TokenSurfaceObservation(
        image_id="ref.png",
        source_id="surface_balanced",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.9,
    )
    unique_surface = TokenSurfaceObservation(
        image_id="ref.png",
        source_id="surface_balanced",
        token_index=9,
        token_xy=np.asarray([1.0, 2.0], dtype=np.float32),
        feature=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        element_ids=np.asarray([12], dtype=np.int64),
        element_weights=np.asarray([1.0], dtype=np.float32),
        center=np.asarray([1.0, 0.0, 4.0], dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.7,
        component_concentration=0.7,
        quality_score=0.7,
    )
    broad = Vfm2DgsObservationBank.from_observations([duplicate_low_quality], metadata={"name": "broad"})
    surface = Vfm2DgsObservationBank.from_observations(
        [duplicate_high_quality, unique_surface],
        metadata={"name": "surface_balanced"},
    )

    merged = merge_vfm_2dgs_observation_banks(
        [broad, surface],
        source_names=["broad", "surface_balanced"],
        deduplicate=True,
    )

    assert len(merged) == 2
    assert merged.source_ids == ("surface_balanced", "surface_balanced")
    np.testing.assert_allclose(merged.features[0], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    assert merged.metadata["duplicate_observation_count"] == 1
    assert merged.metadata["source_observation_counts"] == {"broad": 1, "surface_balanced": 2}

    output = tmp_path / "merged_observation_bank.npz"
    merged.save_npz(output)
    loaded = Vfm2DgsObservationBank.load_npz(output)

    assert loaded.source_ids == merged.source_ids
    assert loaded.metadata["stage"] == "vfm_2dgs_merged_observation_layer"


def test_observation_source_id_round_trips_through_bank_and_observations(tmp_path) -> None:
    observation = TokenSurfaceObservation(
        image_id="ref.png",
        source_id="surface_balanced",
        token_index=3,
        token_xy=np.asarray([1.0, 2.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10], dtype=np.int64),
        element_weights=np.asarray([1.0], dtype=np.float32),
        center=np.asarray([0.0, 0.0, 4.0], dtype=np.float64),
        normal=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )

    bank = Vfm2DgsObservationBank.from_observations([observation])
    output = tmp_path / "bank.npz"
    bank.save_npz(output)
    loaded = Vfm2DgsObservationBank.load_npz(output)

    assert loaded.source_ids == ("surface_balanced",)
    assert loaded.to_observations()[0].source_id == "surface_balanced"


def test_source_aware_fusion_can_keep_broad_and_surface_observations_as_separate_layers() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    broad = TokenSurfaceObservation(
        image_id="a.png",
        source_id="broad",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=elements.centers[0],
        normal=elements.normals[0],
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )
    surface = TokenSurfaceObservation(
        image_id="b.png",
        source_id="surface_balanced",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=elements.centers[0],
        normal=elements.normals[0],
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )

    blind = fuse_token_surface_observations(
        elements,
        [broad, surface],
        Vfm2DgsAnchorFusionConfig(fusion_mode="graph", min_surface_iou=0.3, min_observations=1),
    )
    source_aware = fuse_token_surface_observations(
        elements,
        [broad, surface],
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="graph",
            source_merge_policy="same_source",
            min_surface_iou=0.3,
            min_observations=1,
        ),
    )

    assert len(blind) == 1
    assert len(source_aware) == 2
    assert source_aware.metadata["fusion_config"]["source_merge_policy"] == "same_source"


def test_feature_agreement_source_policy_only_merges_cross_source_observations_when_descriptors_agree() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    broad = TokenSurfaceObservation(
        image_id="a.png",
        source_id="broad",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=elements.centers[0],
        normal=elements.normals[0],
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )
    disagreeing_surface = TokenSurfaceObservation(
        image_id="b.png",
        source_id="surface_balanced",
        token_index=7,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=elements.centers[0],
        normal=elements.normals[0],
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )
    agreeing_surface = TokenSurfaceObservation(
        image_id="c.png",
        source_id="surface_balanced",
        token_index=8,
        token_xy=np.asarray([3.0, 1.0], dtype=np.float32),
        feature=np.asarray([0.98, 0.02, 0.0], dtype=np.float32),
        element_ids=np.asarray([10, 11], dtype=np.int64),
        element_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        center=elements.centers[0],
        normal=elements.normals[0],
        covariance=np.eye(3, dtype=np.float32),
        purity_score=0.8,
        component_concentration=0.8,
        quality_score=0.7,
    )

    blocked = fuse_token_surface_observations(
        elements,
        [broad, disagreeing_surface],
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="graph",
            source_merge_policy="feature_agree",
            cross_source_min_feature_cosine=0.8,
            min_surface_iou=0.3,
            min_observations=1,
        ),
    )
    merged = fuse_token_surface_observations(
        elements,
        [broad, agreeing_surface],
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="graph",
            source_merge_policy="feature_agree",
            cross_source_min_feature_cosine=0.8,
            min_surface_iou=0.3,
            min_observations=1,
        ),
    )

    assert len(blocked) == 2
    assert len(merged) == 1
    assert merged.metadata["fusion_config"]["cross_source_min_feature_cosine"] == 0.8


def test_vfm_2dgs_anchor_map_exports_to_semidense_landmark_index() -> None:
    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        quality_scores=np.asarray([0.7, 0.6], dtype=np.float32),
        observed_view_ids=(("a.png", "b.png"), ("b.png",)),
        anchor_ids=np.asarray([10, 20], dtype=np.int64),
    )

    semidense = vfm_2dgs_anchor_map_to_semidense(anchor_map)
    index = semidense.to_landmark_index()

    assert semidense.source_types.tolist() == ["vfm_2dgs", "vfm_2dgs"]
    assert semidense.source_track_ids.tolist() == [-1, -1]
    assert semidense.support_counts.tolist() == [1, 1]
    assert index.track_ids.tolist() == [-100000010, -100000020]
    np.testing.assert_allclose(index.xyz, anchor_map.centers)
    np.testing.assert_allclose(index.features, anchor_map.features)
    assert index.observation_image_ids == anchor_map.observed_view_ids


def test_vfm_2dgs_to_semidense_export_cli_round_trips(tmp_path) -> None:
    from feature_extract.tools.vfm.export_vfm_2dgs_to_semidense import main as export_main
    from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap

    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        quality_scores=np.asarray([0.7], dtype=np.float32),
        observed_view_ids=(("a.png", "b.png"),),
        anchor_ids=np.asarray([42], dtype=np.int64),
    )
    anchor_path = tmp_path / "anchor_map.npz"
    output_path = tmp_path / "semidense.npz"
    summary_path = tmp_path / "summary.json"
    anchor_map.save_npz(anchor_path)

    export_main(["--anchor_map", str(anchor_path), "--output_npz", str(output_path), "--summary_json", str(summary_path)])
    loaded = SemiDenseAnchorMap.load_npz(output_path)

    assert len(loaded) == 1
    assert loaded.anchor_ids.tolist() == [42]
    assert loaded.observation_image_ids == (("a.png", "b.png"),)
    assert summary_path.exists()


def test_observation_bank_fusion_cli_merges_sources_and_outputs_anchor_map(tmp_path) -> None:
    from feature_extract.tools.vfm.fuse_vfm_2dgs_observation_banks import main as fuse_main
    from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap

    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    observations_a = compute_token_surface_observations(
        elements,
        _view("a.png"),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )
    observations_b = compute_token_surface_observations(
        elements,
        _view("b.png"),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )
    surface_path = tmp_path / "surface_elements.npz"
    broad_path = tmp_path / "broad_observations.npz"
    surface_bank_path = tmp_path / "surface_observations.npz"
    output_path = tmp_path / "anchor_map.npz"
    merged_bank_path = tmp_path / "merged_observations.npz"
    semidense_path = tmp_path / "semidense_view_bins.npz"
    semidense_summary_path = tmp_path / "semidense_summary.json"
    summary_path = tmp_path / "summary.json"
    elements.save_npz(surface_path)
    Vfm2DgsObservationBank.from_observations(observations_a).save_npz(broad_path)
    Vfm2DgsObservationBank.from_observations(observations_b).save_npz(surface_bank_path)

    fuse_main(
        [
            "--surface_npz",
            str(surface_path),
            "--observation_bank",
            f"broad={broad_path}",
            "--observation_bank",
            f"surface={surface_bank_path}",
            "--fusion_mode",
            "graph",
            "--source_merge_policy",
            "feature_agree",
            "--cross_source_min_feature_cosine",
            "0.8",
            "--min_surface_iou",
            "0.3",
            "--min_observations",
            "1",
            "--output_npz",
            str(output_path),
            "--merged_observation_bank_npz",
            str(merged_bank_path),
            "--semidense_output_npz",
            str(semidense_path),
            "--semidense_summary_json",
            str(semidense_summary_path),
            "--semidense_descriptor_mode",
            "view_bins",
            "--summary_json",
            str(summary_path),
        ]
    )

    merged = Vfm2DgsObservationBank.load_npz(merged_bank_path)
    anchor_map = Vfm2DgsAnchorMap.load_npz(output_path)
    semidense = SemiDenseAnchorMap.load_npz(semidense_path)

    assert len(merged) == len(observations_a) + len(observations_b)
    assert set(merged.source_ids) == {"broad", "surface"}
    assert len(anchor_map) == 1
    assert anchor_map.observation_counts.tolist() == [2]
    assert semidense.metadata["descriptor_mode"] == "view_bins"
    assert len(semidense) >= 1
    assert summary_path.exists()
    assert semidense_summary_path.exists()
    import json

    summary = json.loads(summary_path.read_text())
    assert summary["fusion_config"]["source_merge_policy"] == "feature_agree"
    assert summary["fusion_config"]["cross_source_min_feature_cosine"] == 0.8
    assert summary["outputs"]["semidense_anchor_map"] == str(semidense_path)
    assert summary["semidense_export"]["descriptor_mode"] == "view_bins"


def test_overlapping_observations_fuse_to_anchor_and_round_trip(tmp_path) -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    first = compute_token_surface_observations(
        elements,
        _view("a.png"),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )
    second = compute_token_surface_observations(
        elements,
        _view("b.png"),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        first + second,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.3, min_observations=1),
    )

    assert len(anchor_map) == 1
    assert anchor_map.features.shape == (1, 3)
    assert anchor_map.observation_counts.tolist() == [2]
    assert anchor_map.surface_support_counts[0] >= 2
    assert anchor_map.distinctiveness_scores.shape == (1,)
    assert 0.0 <= float(anchor_map.distinctiveness_scores[0]) <= 1.0
    output = tmp_path / "anchor_map.npz"
    anchor_map.save_npz(output)

    loaded = Vfm2DgsAnchorMap.load_npz(output)

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded.features, anchor_map.features)
    np.testing.assert_allclose(loaded.centers, anchor_map.centers)
    assert loaded.observed_view_ids == anchor_map.observed_view_ids


def test_adjacent_surface_observations_can_fuse_with_dilated_support_iou() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.35, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.2, 0.2], dtype=np.float32),
        gaussian_indices=np.asarray([30, 31], dtype=np.int64),
        scale_xyz=np.asarray([[0.2, 0.2, 0.02], [0.2, 0.2, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.0, normal_cosine_threshold=0.9)
    observations = [
        TokenSurfaceObservation(
            image_id="a.png",
            token_index=0,
            token_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=np.asarray([30], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[0],
            normal=elements.normals[0],
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
        TokenSurfaceObservation(
            image_id="b.png",
            token_index=1,
            token_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            feature=np.asarray([0.99, 0.01, 0.0], dtype=np.float32),
            element_ids=np.asarray([31], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[1],
            normal=elements.normals[1],
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
    ]

    strict = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.1, min_observations=1),
    )
    tolerant = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.1,
            min_dilated_surface_iou=0.1,
            support_iou_dilation_hops=1,
            min_observations=1,
        ),
    )

    assert len(strict) == 2
    assert len(tolerant) == 1
    assert tolerant.observation_counts.tolist() == [2]


def test_same_parent_virtual_cell_observations_can_fuse_by_parent_support_iou() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([1.0], dtype=np.float32),
        gaussian_indices=np.asarray([42], dtype=np.int64),
        scale_xyz=np.asarray([[1.0, 0.5, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(
        source,
        virtual_cell_max_scale=0.25,
        virtual_cell_grid_cap=4,
        adjacency_radius=0.0,
    )
    first_id = int(elements.element_ids[0])
    second_id = int(elements.element_ids[-1])
    observations = [
        TokenSurfaceObservation(
            image_id="a.png",
            token_index=0,
            token_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=np.asarray([first_id], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[0],
            normal=elements.normals[0],
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
        TokenSurfaceObservation(
            image_id="b.png",
            token_index=1,
            token_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            feature=np.asarray([0.99, 0.01, 0.0], dtype=np.float32),
            element_ids=np.asarray([second_id], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[-1],
            normal=elements.normals[-1],
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
    ]

    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.1,
            min_parent_surface_iou=0.9,
            max_center_distance=2.0,
            min_observations=1,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.observation_counts.tolist() == [2]


def test_graph_fusion_recovers_transitive_surface_consensus_independent_of_order() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [0.3, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.9], dtype=np.float32),
        scale=np.asarray([0.2, 0.2], dtype=np.float32),
        gaussian_indices=np.asarray([70, 71], dtype=np.int64),
        scale_xyz=np.asarray([[0.2, 0.2, 0.02], [0.2, 0.2, 0.02]], dtype=np.float32),
        normal=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )
    elements = build_surface_elements_from_2dgs_source(source, adjacency_radius=0.0, normal_cosine_threshold=0.9)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id="a.png",
            token_index=0,
            token_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=np.asarray([70], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[0],
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
        TokenSurfaceObservation(
            image_id="b.png",
            token_index=1,
            token_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            feature=np.asarray([0.99, 0.01, 0.0], dtype=np.float32),
            element_ids=np.asarray([71], dtype=np.int64),
            element_weights=np.asarray([1.0], dtype=np.float32),
            center=elements.centers[1],
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
        TokenSurfaceObservation(
            image_id="c.png",
            token_index=2,
            token_xy=np.asarray([0.5, 0.0], dtype=np.float32),
            feature=np.asarray([0.98, 0.02, 0.0], dtype=np.float32),
            element_ids=np.asarray([70, 71], dtype=np.int64),
            element_weights=np.asarray([0.5, 0.5], dtype=np.float32),
            center=np.mean(elements.centers[:2], axis=0),
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
        ),
    ]

    greedy = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.3, max_center_distance=1.0, min_observations=1),
    )
    graph = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            fusion_mode="graph",
            min_surface_iou=0.3,
            max_center_distance=1.0,
            min_observations=1,
        ),
    )

    assert len(greedy) == 2
    assert len(graph) == 1
    assert graph.observation_counts.tolist() == [3]


def test_anchor_keeps_multiple_feature_prototypes_for_view_dependent_observations() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    first = compute_token_surface_observations(
        elements,
        _single_token_view("a.png", [1.0, 0.0, 0.0]),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )
    second = compute_token_surface_observations(
        elements,
        _single_token_view("b.png", [0.0, 1.0, 0.0]),
        Vfm2DgsMappingConfig(token_top_fraction=0.10, footprint_radius_px=1.1, min_component_concentration=0.5),
    )

    anchor_map = fuse_token_surface_observations(
        elements,
        first + second,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.3,
            min_observations=1,
            max_feature_prototypes=2,
            prototype_min_cosine=0.5,
        ),
    )

    assert anchor_map.feature_prototypes.shape == (1, 2, 3)
    assert anchor_map.feature_prototype_counts.tolist() == [2]
    similarities = anchor_map.feature_prototypes[0] @ np.asarray([[1.0], [0.0], [0.0]], dtype=np.float32)
    assert float(np.max(similarities)) > 0.9


def test_anchor_robust_feature_trim_removes_low_consistency_observation() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    element_ids = np.asarray([10, 11], dtype=np.int64)
    element_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id=f"{idx}.png",
            token_index=idx,
            token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            feature=np.asarray(feature, dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=0.9,
            component_concentration=0.9,
            quality_score=1.0,
        )
        for idx, feature in enumerate(([1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0]))
    ]

    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.3,
            min_observations=1,
            robust_feature_trim_fraction=0.34,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.features[0, 0] > 0.99
    assert abs(float(anchor_map.features[0, 1])) < 0.05
    assert anchor_map.feature_variances[0] < 0.01


def test_anchor_consensus_weighted_feature_fusion_prefers_stable_descriptor_cluster() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    element_ids = np.asarray([10, 11], dtype=np.int64)
    element_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id=f"{idx}.png",
            token_index=idx,
            token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            feature=np.asarray(feature, dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=0.9,
            component_concentration=0.9,
            quality_score=1.0,
            descriptor_weight=1.0,
        )
        for idx, feature in enumerate(([1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0]))
    ]

    plain = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.3, min_observations=1),
    )
    consensus = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.3,
            min_observations=1,
            feature_fusion_mode="consensus_weighted_mean",
            feature_consensus_weight_power=2.0,
        ),
    )

    assert len(consensus) == 1
    assert consensus.features[0, 0] > plain.features[0, 0]
    assert consensus.features[0, 1] < plain.features[0, 1]


def test_anchor_high_variance_observation_removal_drops_low_consensus_descriptor() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    element_ids = np.asarray([10, 11], dtype=np.int64)
    element_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id=f"{idx}.png",
            token_index=idx,
            token_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            feature=np.asarray(feature, dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=np.eye(3, dtype=np.float32),
            purity_score=0.9,
            component_concentration=0.9,
            quality_score=1.0,
            descriptor_weight=1.0,
        )
        for idx, feature in enumerate(([1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0]))
    ]

    plain = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.3, min_observations=1),
    )
    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.3,
            min_observations=1,
            min_feature_consensus_cosine=0.6,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.features[0, 0] > 0.99
    assert abs(float(anchor_map.features[0, 1])) < 0.02
    assert anchor_map.feature_variances[0] < plain.feature_variances[0]
    assert anchor_map.feature_variances[0] < 0.011


def test_anchor_stores_view_bin_feature_prototypes() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    element_ids = np.asarray([10, 11], dtype=np.int64)
    element_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    covariance = np.eye(3, dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id="x.png",
            token_index=0,
            token_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=covariance,
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
            view_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ),
        TokenSurfaceObservation(
            image_id="y.png",
            token_index=1,
            token_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=covariance,
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
            view_direction=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        ),
    ]

    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(min_surface_iou=0.3, min_observations=1, view_bin_count=4),
    )

    assert anchor_map.view_bin_features.shape == (1, 4, 3)
    assert anchor_map.view_bin_counts.tolist() == [[1, 1, 0, 0]]
    np.testing.assert_allclose(anchor_map.view_bin_features[0, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(anchor_map.view_bin_features[0, 1], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))


def test_view_bin_medoid_keeps_observation_descriptor_not_average() -> None:
    elements = build_surface_elements_from_2dgs_source(_source(), adjacency_radius=1.1)
    element_ids = np.asarray([10, 11], dtype=np.int64)
    element_weights = np.asarray([0.5, 0.5], dtype=np.float32)
    center = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    covariance = np.eye(3, dtype=np.float32)
    observations = [
        TokenSurfaceObservation(
            image_id="a.png",
            token_index=0,
            token_xy=np.asarray([0.0, 0.0], dtype=np.float32),
            feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=covariance,
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=2.0,
            view_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            descriptor_weight=2.0,
        ),
        TokenSurfaceObservation(
            image_id="b.png",
            token_index=1,
            token_xy=np.asarray([1.0, 0.0], dtype=np.float32),
            feature=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            element_ids=element_ids,
            element_weights=element_weights,
            center=center,
            normal=normal,
            covariance=covariance,
            purity_score=1.0,
            component_concentration=1.0,
            quality_score=1.0,
            view_direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            descriptor_weight=1.0,
        ),
    ]

    anchor_map = fuse_token_surface_observations(
        elements,
        observations,
        Vfm2DgsAnchorFusionConfig(
            min_surface_iou=0.3,
            min_observations=1,
            view_bin_count=4,
            view_bin_feature_mode="medoid",
        ),
    )

    assert anchor_map.view_bin_counts.tolist() == [[2, 0, 0, 0]]
    np.testing.assert_allclose(anchor_map.view_bin_features[0, 0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_spatial_nms_keeps_high_quality_spatially_separated_anchors() -> None:
    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        quality_scores=np.asarray([0.2, 0.9, 0.8], dtype=np.float32),
        observed_view_ids=(("a.png",), ("b.png",), ("c.png",)),
        anchor_ids=np.asarray([10, 11, 12], dtype=np.int64),
    )

    selected = spatial_nms_anchor_map(anchor_map, radius=0.10)

    assert selected.anchor_ids.tolist() == [11, 12]
    np.testing.assert_allclose(selected.quality_scores, np.asarray([0.9, 0.8], dtype=np.float32))
    assert selected.support_offsets.tolist() == [0, 1, 2]
    assert selected.support_element_ids.tolist() == [1, 2]
    assert selected.observed_view_ids == (("b.png",), ("c.png",))


def test_covisibility_graph_uses_shared_observed_views_and_round_trips(tmp_path) -> None:
    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        quality_scores=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        observed_view_ids=(("v1", "v2"), ("v2", "v3"), ("v4",)),
        anchor_ids=np.asarray([10, 20, 30], dtype=np.int64),
    )

    with_graph = build_anchor_covisibility_graph(anchor_map, min_score=0.10, max_neighbors=4)

    first_start, first_end = with_graph.covisibility_offsets[0], with_graph.covisibility_offsets[1]
    second_start, second_end = with_graph.covisibility_offsets[1], with_graph.covisibility_offsets[2]
    third_start, third_end = with_graph.covisibility_offsets[2], with_graph.covisibility_offsets[3]

    assert with_graph.covisibility_anchor_ids[first_start:first_end].tolist() == [20]
    assert with_graph.covisibility_anchor_ids[second_start:second_end].tolist() == [10]
    assert with_graph.covisibility_anchor_ids[third_start:third_end].tolist() == []
    np.testing.assert_allclose(with_graph.covisibility_scores[first_start:first_end], np.asarray([1.0 / 3.0]))

    output = tmp_path / "anchor_map_with_graph.npz"
    with_graph.save_npz(output)
    loaded = Vfm2DgsAnchorMap.load_npz(output)

    assert loaded.covisibility_offsets.tolist() == with_graph.covisibility_offsets.tolist()
    assert loaded.covisibility_anchor_ids.tolist() == with_graph.covisibility_anchor_ids.tolist()
    np.testing.assert_allclose(loaded.covisibility_scores, with_graph.covisibility_scores)


def test_descriptor_index_exports_anchor_prototypes_and_round_trips(tmp_path) -> None:
    anchor_map = Vfm2DgsAnchorMap(
        anchor_ids=np.asarray([10, 20], dtype=np.int64),
        centers=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None, :, :], (2, 1, 1)),
        features=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        feature_prototypes=np.asarray(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        feature_prototype_counts=np.asarray([2, 1], dtype=np.int64),
        feature_variances=np.zeros((2,), dtype=np.float32),
        quality_scores=np.asarray([0.9, 0.4], dtype=np.float32),
        purity_scores=np.ones((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        surface_support_counts=np.ones((2,), dtype=np.int64),
        support_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        support_element_ids=np.asarray([1, 2], dtype=np.int64),
        support_weights=np.ones((2,), dtype=np.float32),
        observed_view_ids=(("a.png",), ("b.png",)),
    )

    index = build_anchor_descriptor_index(anchor_map, include_prototypes=True)
    anchor_ids, prototype_ids, scores = index.search(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), top_k=1)

    assert isinstance(index, Vfm2DgsDescriptorIndex)
    assert index.descriptors.shape == (3, 3)
    assert index.anchor_ids.tolist() == [10, 10, 20]
    assert anchor_ids.tolist() == [[10]]
    assert prototype_ids.tolist() == [[1]]
    np.testing.assert_allclose(scores, np.asarray([[1.0]], dtype=np.float32))

    output = tmp_path / "descriptor_index.npz"
    index.save_npz(output)
    loaded = Vfm2DgsDescriptorIndex.load_npz(output)

    assert loaded.anchor_ids.tolist() == index.anchor_ids.tolist()
    assert loaded.prototype_ids.tolist() == index.prototype_ids.tolist()
    np.testing.assert_allclose(loaded.descriptors, index.descriptors)


def test_descriptor_index_exports_faiss_inner_product_index(tmp_path) -> None:
    anchor_map = _manual_anchor_map(
        centers=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        quality_scores=np.asarray([0.9, 0.5], dtype=np.float32),
        observed_view_ids=(("a.png",), ("b.png",)),
        anchor_ids=np.asarray([10, 20], dtype=np.int64),
    )
    index = build_anchor_descriptor_index(anchor_map, include_prototypes=False)

    output = tmp_path / "descriptor.faiss"
    index.save_faiss(output)

    import faiss

    faiss_index = faiss.read_index(str(output))
    scores, rows = faiss_index.search(index.descriptors[:1].astype(np.float32), 1)

    assert faiss_index.ntotal == len(index)
    assert rows.tolist() == [[0]]
    np.testing.assert_allclose(scores, np.asarray([[1.0]], dtype=np.float32))
