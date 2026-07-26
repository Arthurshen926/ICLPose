from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_maplet_mapper import (
    SurfaceMapletMapper,
    SurfaceMapletMapperConfig,
    load_surface_maplet_mapper,
    maplet_prototype_retrieval_metrics,
    pool_radio_final_context_torch,
    save_surface_maplet_mapper,
    select_spatially_balanced_radio_final_regions,
    surface_maplet_contrastive_loss,
)
from feature_extract.vfm.localization.surface_localization import (
    LocalFeatureFrame,
    SurfaceMapletMatchConfig,
    SurfacePoseConfig,
    build_anchor_local_descriptor_bank,
    build_maplet_conditioned_surface_anchor_candidate_pool,
    build_surface_anchor_candidate_pool,
    estimate_vfm_query_to_support_layout,
    generate_grouped_surface_pose_hypotheses,
    match_radio_final_regions_to_maplets,
    select_vfm_surface_feature_modes,
)
from feature_extract.vfm.surface_maplet_bank import (
    RadioFinalRegionConfig,
    StableSurfaceAnchorMap,
    SurfaceMapletBuildConfig,
    TwoDGSPrimitiveQuality,
    load_2dgs_primitive_quality,
    VfmSurfaceMapletBank,
    build_track_free_surface_map,
    encode_radio_final_regions,
)


def test_clean_2dgs_quality_uses_source_index_as_explicit_mask(
    tmp_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    vertices = np.zeros(
        (2,),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("source_index", "i4"),
            ("primitive_class", "i2"),
        ],
    )
    vertices["source_index"] = np.asarray([2, 5], dtype=np.int32)
    path = tmp_path / "clean.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
    quality = load_2dgs_primitive_quality(path)
    assert len(quality) == 6
    np.testing.assert_array_equal(
        quality.geometry_confidence,
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert quality.metadata["clean_mask_active"] is True
    assert quality.metadata["field_presence"]["geometry_confidence"] is False
from feature_extract.vfm.vfm_2dgs_mapping import (
    SurfaceElementMap,
    Vfm2DgsAnchorMap,
    Vfm2DgsObservationBank,
    _build_adjacency,
)


def _camera(width: int = 100, height: int = 100, focal: float = 80.0) -> ColmapCamera:
    return ColmapCamera(
        camera_id=1,
        model_id=1,
        width=width,
        height=height,
        params=(focal, focal, width / 2.0, height / 2.0),
    )


def _synthetic_surface_inputs():
    element_ids = np.arange(10, 18, dtype=np.int64)
    centers = np.asarray(
        [
            [-0.8, -0.2, 5.0],
            [-0.5, 0.2, 5.0],
            [-0.2, -0.2, 5.0],
            [-0.4, -0.5, 5.0],
            [0.2, -0.2, 5.0],
            [0.5, 0.2, 5.0],
            [0.8, -0.2, 5.0],
            [0.4, -0.5, 5.0],
        ],
        dtype=np.float64,
    )
    surface = SurfaceElementMap(
        element_ids=element_ids,
        parent_gaussian_indices=np.arange(8, dtype=np.int64),
        centers=centers,
        tangent1=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (8, 1)),
        tangent2=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (8, 1)),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (8, 1)),
        scale1=np.full((8,), 0.04, dtype=np.float32),
        scale2=np.full((8,), 0.03, dtype=np.float32),
        opacity=np.full((8,), 0.9, dtype=np.float32),
        area=np.full((8,), 0.004, dtype=np.float32),
        adjacency=tuple(np.zeros((0,), dtype=np.int64) for _ in range(8)),
    )
    maplet_features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    region_map = Vfm2DgsAnchorMap(
        anchor_ids=np.asarray([100, 200], dtype=np.int64),
        centers=np.asarray([[-0.475, -0.175, 5.0], [0.475, -0.175, 5.0]], dtype=np.float64),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        covariances=np.tile(np.diag([0.1, 0.1, 0.001])[None], (2, 1, 1)).astype(np.float32),
        features=maplet_features,
        feature_variances=np.asarray([0.01, 0.01], dtype=np.float32),
        quality_scores=np.asarray([0.9, 0.9], dtype=np.float32),
        purity_scores=np.asarray([0.9, 0.9], dtype=np.float32),
        observation_counts=np.asarray([2, 2], dtype=np.int64),
        surface_support_counts=np.asarray([4, 4], dtype=np.int64),
        support_offsets=np.asarray([0, 4, 8], dtype=np.int64),
        support_element_ids=element_ids,
        support_weights=np.full((8,), 0.25, dtype=np.float32),
        observed_view_ids=(("view0", "view1"), ("view0", "view1")),
    )
    observation_bank = Vfm2DgsObservationBank(
        image_ids=("view0", "view1", "view0", "view1"),
        token_indices=np.asarray([6, 6, 18, 18], dtype=np.int64),
        token_xy=np.asarray([[1, 1], [1, 1], [3, 3], [3, 3]], dtype=np.float32),
        features=np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
        centers=np.asarray(
            [
                [-0.475, -0.175, 5.0],
                [-0.475, -0.175, 5.0],
                [0.475, -0.175, 5.0],
                [0.475, -0.175, 5.0],
            ],
            dtype=np.float64,
        ),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
        covariances=np.tile(np.diag([0.1, 0.1, 0.001])[None], (4, 1, 1)).astype(np.float32),
        support_offsets=np.asarray([0, 4, 8, 12, 16], dtype=np.int64),
        element_ids=np.concatenate([element_ids[:4], element_ids[:4], element_ids[4:], element_ids[4:]]),
        element_weights=np.full((16,), 0.25, dtype=np.float32),
        purity_scores=np.full((4,), 0.9, dtype=np.float32),
        component_concentrations=np.full((4,), 1.0, dtype=np.float32),
        quality_scores=np.full((4,), 0.9, dtype=np.float32),
        view_directions=np.tile(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), (4, 1)),
    )
    feature0 = np.zeros((2, 5, 5), dtype=np.float32)
    feature0[0, :, :2] = 1.0
    feature0[1, :, 2:] = 1.0
    feature1 = feature0.copy()
    poses = {"view0": np.eye(4), "view1": np.eye(4)}
    cameras = {"view0": _camera(), "view1": _camera()}
    return surface, region_map, observation_bank, {"view0": feature0, "view1": feature1}, poses, cameras


def test_radio_final_multiscale_regions_use_context_without_intermediate() -> None:
    feature = np.zeros((2, 7, 7), dtype=np.float32)
    feature[1, 1:6, 1:6] = 1.0
    feature[:, 3, 3] = np.asarray([1.0, 0.0])
    center_only = encode_radio_final_regions(
        feature,
        np.asarray([[3.0, 3.0]], dtype=np.float32),
        RadioFinalRegionConfig(pool_sizes=(1,), pool_weights=(1.0,)),
    )
    contextual = encode_radio_final_regions(
        feature,
        np.asarray([[3.0, 3.0]], dtype=np.float32),
        RadioFinalRegionConfig(pool_sizes=(1, 5), pool_weights=(0.5, 0.5)),
    )
    assert center_only[0, 0] > center_only[0, 1]
    assert contextual[0, 1] > center_only[0, 1]


def test_surface_maplet_mapper_is_full_map_normalized_and_roundtrips(tmp_path) -> None:
    torch.manual_seed(2)
    model = SurfaceMapletMapper(
        SurfaceMapletMapperConfig(input_dim=4, hidden_dim=8, output_dim=3)
    )
    feature = torch.randn(2, 4, 5, 7)
    mapped = model(feature)
    assert mapped.shape == (2, 3, 5, 7)
    torch.testing.assert_close(torch.linalg.vector_norm(mapped, dim=1), torch.ones(2, 5, 7))
    pooled = pool_radio_final_context_torch(
        mapped[0],
        torch.tensor([[1.0, 2.0], [5.0, 4.0]]),
        pool_sizes=(1, 3),
        pool_weights=(0.5, 0.5),
    )
    assert pooled.shape == (2, 3)
    pooled.sum().backward()
    assert model.shortcut.weight.grad is not None

    checkpoint = tmp_path / "surface_mapper.pt"
    save_surface_maplet_mapper(
        checkpoint,
        model,
        metadata={"uses_sfm_tracks": False, "uses_radio_intermediate": False},
    )
    restored, metadata = load_surface_maplet_mapper(checkpoint)
    np.testing.assert_allclose(
        restored.project(feature[0].detach().numpy()).coarse_descriptors,
        model(feature[:1]).detach().numpy()[0],
        atol=1e-6,
    )
    assert metadata["uses_sfm_tracks"] is False


def test_query_radio_region_selection_is_spatially_balanced() -> None:
    feature = np.zeros((2, 8, 8), dtype=np.float32)
    feature[0] = np.arange(64, dtype=np.float32).reshape(8, 8)
    indices, xy = select_spatially_balanced_radio_final_regions(
        feature,
        grid_rows=2,
        grid_cols=2,
        regions_per_cell=2,
    )
    assert indices.shape == (8,)
    assert xy.shape == (8, 2)
    cell_ids = (xy[:, 1] >= 4).astype(np.int64) * 2 + (xy[:, 0] >= 4).astype(np.int64)
    np.testing.assert_array_equal(np.bincount(cell_ids, minlength=4), np.full((4,), 2))


def test_direct_vfm_support_layout_recovers_full_affine_geometry() -> None:
    query = np.eye(4, dtype=np.float32).reshape(4, 2, 2)
    query_xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    matrix = np.asarray([[1.2, 0.2], [-0.1, 0.9]], dtype=np.float64)
    translation = np.asarray([2.0, 3.0], dtype=np.float64)
    support_xy = query_xy @ matrix.T + translation
    bank = Vfm2DgsObservationBank(
        image_ids=("support",) * 4,
        token_indices=np.arange(4, dtype=np.int64),
        token_xy=support_xy,
        features=np.eye(4, dtype=np.float32),
        centers=np.zeros((4, 3), dtype=np.float64),
        normals=np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1)),
        covariances=np.tile(np.eye(3, dtype=np.float32)[None], (4, 1, 1)),
        support_offsets=np.zeros((5,), dtype=np.int64),
        element_ids=np.zeros((0,), dtype=np.int64),
        element_weights=np.zeros((0,), dtype=np.float32),
        purity_scores=np.ones((4,), dtype=np.float32),
        component_concentrations=np.ones((4,), dtype=np.float32),
        quality_scores=np.ones((4,), dtype=np.float32),
        view_directions=np.tile(np.asarray([[0, 0, -1]], dtype=np.float32), (4, 1)),
    )
    estimated_matrix, estimated_translation, inliers, residual = (
        estimate_vfm_query_to_support_layout(
            query,
            bank,
            "support",
            minimum_similarity=0.5,
            ransac_threshold_tokens=0.01,
        )
    )
    assert inliers == 4
    assert residual < 1e-5
    np.testing.assert_allclose(estimated_matrix, matrix, atol=1e-5)
    np.testing.assert_allclose(estimated_translation, translation, atol=1e-5)


def test_surface_feature_modes_are_map_only_and_can_be_maplet_scoped() -> None:
    (
        _surface,
        _region_map,
        observation_bank,
        features,
        _poses,
        _cameras,
    ) = _synthetic_surface_inputs()
    modes, scores = select_vfm_surface_feature_modes(
        features["view0"],
        observation_bank,
        maximum_global_modes=2,
        maximum_modes=2,
        allowed_mode_ids=("view1",),
    )
    assert modes == ("view1",)
    assert scores.shape == (1,)
    assert np.isfinite(scores).all()


def test_surface_maplet_contrastive_supervision_and_disjoint_retrieval() -> None:
    descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([10, 10, 20, 20])
    images = torch.tensor([0, 1, 0, 1])
    loss, stats = surface_maplet_contrastive_loss(
        descriptors,
        labels,
        images,
        maplet_centers=torch.tensor(
            [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.2, 0.0, 5.0], [0.2, 0.0, 5.0]]
        ),
    )
    assert float(loss) > 0.0
    assert stats["valid_anchor_count"] == 4
    loss.backward()
    assert descriptors.grad is not None
    metrics = maplet_prototype_retrieval_metrics(
        descriptors.detach().numpy(),
        labels.numpy(),
        train_mask=np.asarray([True, False, True, False]),
        query_mask=np.asarray([False, True, False, True]),
    )
    assert metrics["query_count"] == 2
    assert metrics["recall_at_1"] == 1.0


def test_surface_adjacency_cap_is_robust_to_one_huge_disk() -> None:
    adjacency = _build_adjacency(
        centers=np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1)),
        radius=0.05,
        normal_cosine_threshold=0.8,
        element_radius=np.asarray([100.0, 0.1, 0.1]),
        element_radius_cap=0.2,
    )
    assert all(len(row) == 0 for row in adjacency)


def test_track_free_surface_map_separates_maplets_from_metric_anchors(tmp_path) -> None:
    surface, region_map, observations, features, poses, cameras = _synthetic_surface_inputs()
    maplets, anchors, summary = build_track_free_surface_map(
        surface,
        region_map,
        observations,
        features,
        poses,
        cameras,
        primitive_quality=TwoDGSPrimitiveQuality.neutral(8),
        build_config=SurfaceMapletBuildConfig(
            min_anchor_views=2,
            min_anchors_per_maplet=2,
            max_anchors_per_maplet=4,
            min_anchor_separation=0.01,
        ),
    )
    assert len(maplets) == 2
    assert len(anchors) == 8
    assert set(maplets.anchor_ids.tolist()) == set(anchors.anchor_ids.tolist())
    assert np.all(np.diff(anchors.observation_offsets) == 2)
    assert summary["radio_intermediate_used"] is False
    assert summary["sfm_tracks_used"] is False
    assert maplets.metadata["vfm_layer"] == "radio_final"

    maplet_path = tmp_path / "maplets.npz"
    anchor_path = tmp_path / "anchors.npz"
    maplets.save_npz(maplet_path)
    anchors.save_npz(anchor_path)
    restored_maplets = VfmSurfaceMapletBank.load_npz(maplet_path)
    restored_anchors = StableSurfaceAnchorMap.load_npz(anchor_path)
    np.testing.assert_allclose(restored_maplets.descriptors, maplets.descriptors)
    np.testing.assert_allclose(restored_anchors.xyz, anchors.xyz)


def test_surface_maplet_contract_rejects_radio_intermediate() -> None:
    with pytest.raises(ValueError, match="RADIO intermediate"):
        VfmSurfaceMapletBank(
            maplet_ids=np.asarray([1], dtype=np.int64),
            centers=np.zeros((1, 3), dtype=np.float64),
            normals=np.asarray([[0, 0, 1]], dtype=np.float32),
            tangent_frames=np.eye(3, dtype=np.float32)[None],
            extents=np.ones((1, 3), dtype=np.float32),
            descriptors=np.asarray([[1, 0]], dtype=np.float32),
            quality_scores=np.ones((1,), dtype=np.float32),
            descriptor_variances=np.zeros((1,), dtype=np.float32),
            anchor_offsets=np.asarray([0, 0], dtype=np.int64),
            anchor_ids=np.zeros((0,), dtype=np.int64),
            support_offsets=np.asarray([0, 0], dtype=np.int64),
            support_element_ids=np.zeros((0,), dtype=np.int64),
            view_offsets=np.asarray([0, 0], dtype=np.int64),
            view_image_ids=(),
            view_token_xy=np.zeros((0, 2), dtype=np.float32),
            view_grid_sizes=np.zeros((0, 2), dtype=np.int32),
            view_descriptors=np.zeros((0, 2), dtype=np.float32),
            view_quality_scores=np.zeros((0,), dtype=np.float32),
            metadata={"vfm_layer": "radio_final", "uses_radio_intermediate": True},
        )


def test_maplet_matching_uses_whole_image_support_layout() -> None:
    surface, region_map, observations, features, poses, cameras = _synthetic_surface_inputs()
    maplets, _anchors, _summary = build_track_free_surface_map(
        surface,
        region_map,
        observations,
        features,
        poses,
        cameras,
        primitive_quality=TwoDGSPrimitiveQuality.neutral(8),
        build_config=SurfaceMapletBuildConfig(
            min_anchor_views=2,
            min_anchors_per_maplet=2,
            max_anchors_per_maplet=4,
            min_anchor_separation=0.01,
        ),
    )
    result = match_radio_final_regions_to_maplets(
        query_region_xy=np.asarray([[1, 1], [3, 3]], dtype=np.float32),
        query_descriptors=maplets.descriptors.copy(),
        query_grid_size=(5, 5),
        bank=maplets,
        config=SurfaceMapletMatchConfig(top_k=2, minimum_layout_pairs=2),
    )
    np.testing.assert_array_equal(result.selected_maplet_ids, maplets.maplet_ids)
    assert result.support_view_id in {"view0", "view1"}
    assert np.all(result.null_probabilities < np.max(result.candidate_probabilities, axis=1))


def test_maplet_matching_can_skip_support_layout_for_point_aligned_vfm() -> None:
    surface, region_map, observations, features, poses, cameras = _synthetic_surface_inputs()
    maplets, _anchors, _summary = build_track_free_surface_map(
        surface,
        region_map,
        observations,
        features,
        poses,
        cameras,
        primitive_quality=TwoDGSPrimitiveQuality.neutral(8),
        build_config=SurfaceMapletBuildConfig(
            min_anchor_views=2,
            min_anchors_per_maplet=2,
            max_anchors_per_maplet=4,
            min_anchor_separation=0.01,
        ),
    )
    result = match_radio_final_regions_to_maplets(
        query_region_xy=np.asarray([[1, 1], [3, 3]], dtype=np.float32),
        query_descriptors=maplets.descriptors.copy(),
        query_grid_size=(5, 5),
        bank=maplets,
        config=SurfaceMapletMatchConfig(
            top_k=2,
            minimum_layout_pairs=2,
            enable_support_layout=False,
        ),
    )
    assert result.support_view_id is None
    assert result.support_mode_view_ids == ()
    assert np.all(np.isinf(result.layout_residuals))


def test_local_anchor_candidates_are_conditioned_on_nearest_vfm_maplet() -> None:
    surface, region_map, observations, features, poses, cameras = _synthetic_surface_inputs()
    maplets, anchors, _summary = build_track_free_surface_map(
        surface,
        region_map,
        observations,
        features,
        poses,
        cameras,
        primitive_quality=TwoDGSPrimitiveQuality.neutral(8),
        build_config=SurfaceMapletBuildConfig(
            min_anchor_views=2,
            min_anchors_per_maplet=2,
            max_anchors_per_maplet=4,
            min_anchor_separation=0.01,
        ),
    )
    match = match_radio_final_regions_to_maplets(
        query_region_xy=np.asarray([[1, 1], [3, 3]], dtype=np.float32),
        query_descriptors=maplets.descriptors.copy(),
        query_grid_size=(5, 5),
        bank=maplets,
        config=SurfaceMapletMatchConfig(top_k=2, minimum_layout_pairs=2),
    )
    support_frames = {}
    for image_id in ("view0", "view1"):
        observation_rows = np.asarray(
            [
                row
                for row, value in enumerate(anchors.observation_image_ids)
                if value == image_id
            ],
            dtype=np.int64,
        )
        anchor_rows = np.asarray(
            [
                row
                for row in range(len(anchors))
                if image_id
                in anchors.observation_image_ids[
                    int(anchors.observation_offsets[row]) : int(anchors.observation_offsets[row + 1])
                ]
            ],
            dtype=np.int64,
        )
        local_descriptors = np.stack(
            [
                np.asarray([1.0, 0.0], dtype=np.float32)
                if int(anchors.owner_maplet_ids[row]) == int(maplets.maplet_ids[0])
                else np.asarray([0.0, 1.0], dtype=np.float32)
                for row in anchor_rows
            ]
        )
        support_frames[image_id] = LocalFeatureFrame(
            image_id,
            anchors.observation_xy[observation_rows],
            local_descriptors,
        )
    local_bank = build_anchor_local_descriptor_bank(
        anchors,
        support_frames,
        maximum_pixel_distance=0.1,
        minimum_observations=2,
    )
    query_descriptors = np.stack(
        [
            np.asarray([1.0, 0.0], dtype=np.float32)
            if int(owner) == int(maplets.maplet_ids[0])
            else np.asarray([0.0, 1.0], dtype=np.float32)
            for owner in anchors.owner_maplet_ids.tolist()
        ]
    )
    query = LocalFeatureFrame(
        "query",
        anchors.observation_xy[::2],
        query_descriptors,
    )
    pool = build_maplet_conditioned_surface_anchor_candidate_pool(
        query,
        query_image_size=(100, 100),
        query_region_xy=np.asarray([[1, 2], [3, 2]], dtype=np.float32),
        query_region_grid_size=(5, 5),
        maplet_match=match,
        maplets=maplets,
        descriptor_bank=local_bank,
        anchors=anchors,
        top_l=2,
        maximum_maplets_per_region=1,
        support_spatial_sigma=1.0,
        maximum_support_distance=2.0,
    )
    row_by_id = anchors.row_by_id()
    selected_owners = np.asarray(
        [
            anchors.owner_maplet_ids[row_by_id[int(anchor_id)]]
            for anchor_id in pool.anchor_ids[:, 0]
        ]
    )
    np.testing.assert_array_equal(selected_owners, anchors.owner_maplet_ids)


def _project(xyz: np.ndarray, camera: ColmapCamera) -> np.ndarray:
    z = xyz[:, 2:3]
    fx, fy, cx, cy = camera.params[:4]
    return np.concatenate(
        [fx * xyz[:, 0:1] / z + cx, fy * xyz[:, 1:2] / z + cy],
        axis=1,
    ).astype(np.float32)


def test_local_anchor_assignment_and_independent_grouped_pnp() -> None:
    rng = np.random.default_rng(4)
    camera = _camera(width=640, height=480, focal=500.0)
    xyz = rng.uniform([-1.0, -0.7, 4.0], [1.0, 0.7, 7.0], size=(20, 3))
    xy = _project(xyz, camera)
    descriptor = rng.normal(size=(20, 32)).astype(np.float32)
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8)
    observation_offsets = np.arange(0, 41, 2, dtype=np.int64)
    anchors = StableSurfaceAnchorMap(
        anchor_ids=np.arange(100, 120, dtype=np.int64),
        owner_maplet_ids=np.zeros((20,), dtype=np.int64),
        surface_element_ids=np.arange(100, 120, dtype=np.int64),
        parent_primitive_indices=np.arange(20, dtype=np.int64),
        xyz=xyz,
        normals=np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (20, 1)),
        tangent_covariances=np.tile(np.diag([0.01, 0.01, 0.001])[None], (20, 1, 1)).astype(np.float32),
        support_radii=np.full((20,), 0.05, dtype=np.float32),
        quality_scores=np.ones((20,), dtype=np.float32),
        geometry_confidence=np.ones((20,), dtype=np.float32),
        opacity=np.ones((20,), dtype=np.float32),
        observation_offsets=observation_offsets,
        observation_image_ids=tuple(value for _ in range(20) for value in ("support0", "support1")),
        observation_xy=np.repeat(xy, 2, axis=0),
        observation_depth=np.repeat(xyz[:, 2], 2),
        observation_weights=np.ones((40,), dtype=np.float32),
    )
    support_frames = {
        "support0": LocalFeatureFrame("support0", xy, descriptor),
        "support1": LocalFeatureFrame("support1", xy, descriptor),
    }
    local_bank = build_anchor_local_descriptor_bank(
        anchors,
        support_frames,
        maximum_pixel_distance=0.5,
        minimum_observations=2,
    )
    query = LocalFeatureFrame("query", xy, descriptor)
    pool = build_surface_anchor_candidate_pool(
        query,
        anchors.anchor_ids.tolist(),
        local_bank,
        anchors,
        top_l=3,
        descriptor_temperature=0.05,
    )
    np.testing.assert_array_equal(pool.anchor_ids[:, 0], anchors.anchor_ids)
    result = generate_grouped_surface_pose_hypotheses(
        pool,
        camera,
        SurfacePoseConfig(
            hypothesis_count=32,
            minimum_fit_groups=8,
            minimum_verification_groups=3,
            heldout_stride=4,
        ),
    )
    assert result.success
    np.testing.assert_allclose(result.pose_w2c, np.eye(4), atol=1e-4)
    assert np.all(~(result.fit_mask & result.verification_mask))
    assert len(result.hypotheses) >= 1
