import numpy as np
import pytest
import torch
from types import SimpleNamespace

from feature_extract.tools.vfm.evaluate_v6_retrieval_pose_basin import (
    _coarse_observation_oracles,
)
from feature_extract.tools.vfm.train_v6_metric_encoder import (
    TrainingView,
    _query_matchable_at_grid,
    _validation_selection_key,
)
from feature_extract.tools.vfm.train_v6_surface_spatial_projection import (
    _balanced_surface_rows,
    _mode_marginalized_logits,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMSource
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView
from feature_extract.vfm.localization_v6.atlas_baking import bake_feature_atlas
from feature_extract.vfm.localization_v6.atlas_renderer import (
    RenderedMapletAtlases,
    render_selected_maplet_atlases,
)
from feature_extract.vfm.localization_v6.local_correlation import (
    CorrelationDistribution,
    local_correlation_distribution,
)
from feature_extract.vfm.localization_v6.heldout_verifier import (
    accept_pose_update,
    zero_displacement_log_likelihood,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
    canonical_maplet_geometry,
)
from feature_extract.vfm.localization_v6.maplet_pose_proposal import (
    propose_maplet_surface_mode_poses,
)
from feature_extract.vfm.localization_v6.maplet_footprint_pose import (
    score_maplet_footprint_pose,
)
from feature_extract.vfm.localization_v6.maplet_pose_voting import (
    AnonymousMapletPoseVoteBank,
    _bearing_from_normalized_image_xy,
    retrieval_descriptor_sha256,
    retrieval_geometry_sha256,
    vote_maplet_poses,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    V6MetricEncoder,
    V6MetricEncoderConfig,
)
from feature_extract.vfm.localization_v6.metric_training import (
    analytic_one_step_pose_loss,
    local_correlation_training_loss,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    PrimitiveContributorBuffer,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    QueryMapletGroup,
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
    package_surface_feature_atlas,
)
from feature_extract.vfm.localization_v6.se3_update import (
    projection_jacobian,
    se3_exp,
    solve_correlation_se3_update,
)
from feature_extract.vfm.localization_v6.surface_spatial_projection import (
    SurfaceSpatialProjection,
    SurfaceSpatialProjectionConfig,
    load_surface_spatial_projection,
    save_surface_spatial_projection,
)
from feature_extract.vfm.localization_v6.probability_calibration import (
    DistributionNullCalibration,
    V6ProbabilityCalibration,
)
from feature_extract.vfm.surface_maplet_bank import VfmSurfaceMapletBank


def _maplets():
    return VfmSurfaceMapletBank(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0, 0, 2]], dtype=np.float64),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        tangent_frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.2, 0.1]], dtype=np.float32),
        descriptors=np.asarray([[1, 0]], dtype=np.float32),
        quality_scores=np.ones(1),
        descriptor_variances=np.zeros(1),
        anchor_offsets=np.asarray([0, 1]),
        anchor_ids=np.asarray([0]),
        support_offsets=np.asarray([0, 1]),
        support_element_ids=np.asarray([0]),
        view_offsets=np.asarray([0, 1]),
        view_image_ids=("view",),
        view_token_xy=np.zeros((1, 2)),
        view_grid_sizes=np.ones((1, 2), dtype=np.int32),
        view_descriptors=np.asarray([[1, 0]], dtype=np.float32),
        view_quality_scores=np.ones(1),
        metadata={"vfm_layer": "radio_final"},
    )


def _source():
    return GaussianVFMSource(
        xyz=np.asarray([[0, 0, 2]], dtype=np.float64),
        opacity=np.ones(1),
        scale=np.asarray([0.2]),
        gaussian_indices=np.asarray([0]),
        scale_xyz=np.asarray([[0.2, 0.2, 0.01]], dtype=np.float32),
        rotation=np.asarray([[1, 0, 0, 0]], dtype=np.float32),
        normal=np.asarray([[0, 0, 1]], dtype=np.float32),
    )


def _camera():
    return ColmapCamera(
        camera_id=1,
        model_id=2,
        width=100,
        height=80,
        params=(50.0, 50.0, 40.0, 0.0),
    )


def _probability_calibration(
    *,
    identity_bias: float = -20.0,
    spatial_bias: float = -20.0,
) -> V6ProbabilityCalibration:
    return V6ProbabilityCalibration(
        identity=DistributionNullCalibration(
            weights=np.asarray([identity_bias, 0, 0, 0, 0])
        ),
        spatial=DistributionNullCalibration(
            weights=np.asarray([spatial_bias, 0, 0, 0, 0])
        ),
        metadata={
            "artifact_type": "v6_probability_calibration",
            "calibration_trajectory_ids": ["validation"],
            "strict_holdout_trajectory_ids": ["test"],
            "spatial_calibration_condition": (
                "identity_is_true_and_atlas_available"
            ),
            "calibration_objective": "unweighted_bernoulli_nll",
        },
    )


def test_query_matchability_uses_clean_surface_support_not_pair_validity():
    view = TrainingView(
        image_id="seq/frame.png",
        trajectory_id="seq",
        rgb=torch.zeros(3, 4, 4),
        radio=torch.zeros(2, 1, 1),
        pose_w2c=np.eye(4),
        camera=_camera(),
        visible_rows=np.zeros((0,), dtype=np.int64),
        image_xy=np.zeros((0, 2), dtype=np.float32),
        clean_surface_mask=np.asarray(
            [[True, False], [False, True]], dtype=bool
        ),
    )
    result = _query_matchable_at_grid(
        view,
        np.asarray(
            [[0, 0], [3, 0], [0, 3], [3, 3], [-1, 0], [np.nan, 0]],
            dtype=np.float32,
        ),
        width=4,
        height=4,
    )
    assert result.tolist() == [True, False, False, True, False, False]


def test_checkpoint_selection_prioritizes_spatial_mode_signal():
    lower_loss = {
        "mode_recall": 0.20,
        "direction_cosine": 0.9,
        "one_step_pose": 0.01,
        "null_auprc": 0.9,
        "flow_epe": 0.1,
        "total": 0.1,
    }
    higher_mode = {
        **lower_loss,
        "mode_recall": 0.21,
        "direction_cosine": 0.1,
        "total": 10.0,
    }
    assert _validation_selection_key(higher_mode) < _validation_selection_key(
        lower_loss
    )


def test_surface_spatial_projection_round_trip_and_contract(tmp_path):
    initial = np.eye(3, 5, dtype=np.float32)
    model = SurfaceSpatialProjection(
        SurfaceSpatialProjectionConfig(input_dim=5, output_dim=3),
        initial_projection=initial,
    )
    descriptors = model(torch.eye(5)[:2])
    assert descriptors.shape == (2, 3)
    assert torch.allclose(
        torch.linalg.norm(descriptors, dim=1), torch.ones(2)
    )
    path = tmp_path / "surface-spatial.pt"
    save_surface_spatial_projection(
        path,
        model,
        {
            "vfm_layer": "radio_final",
            "stores_mapping_rgb": False,
            "uses_sfm_tracks": False,
        },
    )
    restored, metadata = load_surface_spatial_projection(path)
    assert metadata["vfm_layer"] == "radio_final"
    assert torch.allclose(restored.projection, model.projection)


def test_surface_row_sampler_balances_maplets_and_has_no_duplicates():
    flat_cells = 16
    rows = np.asarray(
        [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35], dtype=np.int64
    )
    xyz = np.zeros((48, 3), dtype=np.float32)
    xyz[:, 0] = np.arange(48)
    sampled = _balanced_surface_rows(
        rows,
        flat_cells=flat_cells,
        xyz=xyz,
        maplets_per_batch=2,
        cells_per_maplet=3,
        rng=np.random.default_rng(4),
    )
    _, counts = np.unique(sampled // flat_cells, return_counts=True)
    assert sampled.size == 6
    assert np.unique(sampled).size == sampled.size
    assert counts.tolist() == [3, 3]


def test_surface_spatial_training_marginalizes_reference_modes():
    model = SurfaceSpatialProjection(
        SurfaceSpatialProjectionConfig(input_dim=2, output_dim=2),
        initial_projection=np.eye(2, dtype=np.float32),
    )
    reference_modes = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]]]
    )
    mask = torch.tensor([[True, False], [True, False]])
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits, similarity = _mode_marginalized_logits(
        model,
        reference_modes,
        mask,
        query,
        temperature=0.1,
    )
    assert logits.shape == (2, 2)
    assert similarity.shape == (2, 2)
    assert torch.argmax(logits, dim=1).tolist() == [0, 1]


def test_canonical_atlas_geometry_does_not_depend_on_observations():
    xyz, primitive_ids, valid, audit = canonical_maplet_geometry(
        _source(), _maplets(), resolution=4
    )
    assert valid.all()
    assert np.all(primitive_ids == 0)
    assert np.allclose(xyz[..., 2], 2.0)
    assert audit["observation_dependent_geometry"] is False


def test_area_atlas_renderer_fills_surface_pixels():
    xyz, primitive_ids, valid, _audit = canonical_maplet_geometry(
        _source(), _maplets(), resolution=4
    )
    features = np.zeros((1, 2, 4, 4), dtype=np.float32)
    features[:, 0] = 1.0
    atlas = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0, 0, 2]], dtype=np.float32),
        frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.2, 0.1]], dtype=np.float32),
        xyz=xyz,
        primitive_ids=primitive_ids,
        features=features,
        variance=np.zeros((1, 4, 4), dtype=np.float32),
        support_count=np.ones((1, 4, 4), dtype=np.int32),
        valid_mask=valid,
        metadata={},
    )
    rendered = render_selected_maplet_atlases(
        atlas, np.asarray([7]), np.eye(4), _camera(), width=100, height=80
    )
    assert int(np.sum(rendered.mask)) > 20
    assert np.all(rendered.maplet_id[rendered.mask] == 7)
    assert np.allclose(rendered.xyz[rendered.mask, 2], 2.0)
    spatial_bank = package_surface_feature_atlas(
        atlas,
        spatial_stride=2,
        feature_space="v6_metric_fine",
        feature_space_sha256="query-encoder",
        representation="exact_canonical_v6_metric_surface_texture",
    )
    assert spatial_bank.maplet_ids.tolist() == [7]
    assert spatial_bank.descriptor_centers.shape == (4, 3)
    assert np.allclose(spatial_bank.descriptor_centers[:, 2], 2.0)


def test_atlas_baking_requires_exact_raster_identity_and_keeps_geometry():
    xyz, primitive_ids, valid, _audit = canonical_maplet_geometry(
        _source(), _maplets(), resolution=4
    )
    geometry = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0, 0, 2]], dtype=np.float32),
        frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.2, 0.1]], dtype=np.float32),
        xyz=xyz,
        primitive_ids=primitive_ids,
        features=np.zeros((1, 2, 4, 4), dtype=np.float32),
        variance=np.ones((1, 4, 4), dtype=np.float32),
        support_count=np.zeros((1, 4, 4), dtype=np.int32),
        valid_mask=valid,
        metadata={},
    )
    feature = np.zeros((2, 20, 25), dtype=np.float32)
    feature[0] = 1.0
    view = GaussianVFMFeatureView(
        image_id="view",
        feature_map=feature,
        pose_w2c=np.eye(4),
        camera=_camera(),
    )
    top_ids = np.zeros((20, 25, 1), dtype=np.int64)
    top_weights = np.ones((20, 25, 1), dtype=np.float32)
    contributor = PrimitiveContributorBuffer(
        dominant_ids=top_ids[..., 0],
        dominant_weights=top_weights[..., 0],
        topk_ids=top_ids,
        topk_weights=top_weights,
        primitive_depth=np.ones((20, 25), dtype=np.float32) * 2,
        metadata={},
    )
    atlas, report = bake_feature_atlas(
        geometry, [view], [contributor], minimum_support=1
    )
    assert np.array_equal(atlas.xyz, geometry.xyz)
    assert atlas.valid_mask.all()
    assert np.allclose(atlas.features[:, 0], 1.0)
    assert (
        report["assignment"]
        == "bilinear_gsplat_topk_contributor_source_index"
    )


def test_view_conditioned_atlas_keeps_anonymous_appearance_modes(tmp_path):
    xyz, primitive_ids, valid, _audit = canonical_maplet_geometry(
        _source(), _maplets(), resolution=4
    )
    geometry = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0, 0, 2]], dtype=np.float32),
        frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.2, 0.1]], dtype=np.float32),
        xyz=xyz,
        primitive_ids=primitive_ids,
        features=np.zeros((1, 2, 4, 4), dtype=np.float32),
        variance=np.ones((1, 4, 4), dtype=np.float32),
        support_count=np.zeros((1, 4, 4), dtype=np.int32),
        valid_mask=valid,
        metadata={},
    )
    contributor = PrimitiveContributorBuffer(
        dominant_ids=np.zeros((20, 25), dtype=np.int64),
        dominant_weights=np.ones((20, 25), dtype=np.float32),
        topk_ids=np.zeros((20, 25, 1), dtype=np.int64),
        topk_weights=np.ones((20, 25, 1), dtype=np.float32),
        primitive_depth=np.ones((20, 25), dtype=np.float32) * 2,
        metadata={},
    )
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = -0.5
    views = []
    for index, pose in enumerate(poses):
        feature = np.zeros((2, 20, 25), dtype=np.float32)
        feature[index] = 1.0
        views.append(
            GaussianVFMFeatureView(
                image_id=f"view-{index}",
                feature_map=feature,
                pose_w2c=pose,
                camera=_camera(),
            )
        )
    atlas, report = bake_feature_atlas(
        geometry,
        views,
        [contributor, contributor],
        minimum_support=1,
        appearance_modes=2,
    )
    assert atlas.appearance_mode_count == 2
    assert np.all(np.sum(atlas.mode_valid_mask, axis=1) == 2)
    assert np.allclose(np.sum(atlas.mode_weights, axis=1), 1.0)
    assert report["view_conditioned_appearance"] is True
    path = tmp_path / "modes.npz"
    atlas.save_npz(path)
    restored = MapletFeatureAtlasBank.load_npz(path)
    assert np.allclose(restored.mode_features, atlas.mode_features, atol=1e-3)


def test_local_correlation_recovers_shift_distribution():
    rng = np.random.default_rng(4)
    channels, height, width = 32, 12, 14
    feature = rng.normal(size=(channels, height, width)).astype(np.float32)
    feature /= np.maximum(np.linalg.norm(feature, axis=0, keepdims=True), 1e-8)
    query = np.zeros_like(feature)
    query[:, :, 2:] = feature[:, :, :-2]
    mask = np.zeros((height, width), dtype=bool)
    mask[3:-3, 3:-4] = True
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xyz = np.stack([xx, yy, np.ones_like(xx) * 4], axis=2).astype(np.float32)
    rendered = RenderedMapletAtlases(
        feature=feature,
        xyz=xyz,
        normal=np.zeros((height, width, 3), dtype=np.float32),
        uncertainty=np.zeros((height, width), dtype=np.float32),
        maplet_id=np.where(mask, 7, -1),
        mask=mask,
        depth=np.where(mask, 4.0, 0.0).astype(np.float32),
    )
    result = local_correlation_distribution(
        rendered, query, radius=3, temperature=0.02, device="cpu"
    )
    assert np.median(result.mean_displacement[:, 0]) > 1.8
    assert np.median(np.abs(result.mean_displacement[:, 1])) < 0.2


def test_candidate_matchability_changes_offset_posterior():
    feature = np.ones((1, 5, 5), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    rendered = RenderedMapletAtlases(
        feature=feature,
        xyz=np.zeros((5, 5, 3), dtype=np.float32),
        normal=np.zeros((5, 5, 3), dtype=np.float32),
        uncertainty=np.zeros((5, 5), dtype=np.float32),
        maplet_id=np.where(mask, 7, -1),
        mask=mask,
        depth=mask.astype(np.float32),
    )
    matchability = np.full((5, 5), 1e-4, dtype=np.float32)
    matchability[2, 3] = 1.0
    result = local_correlation_distribution(
        rendered,
        feature,
        radius=1,
        temperature=1.0,
        query_matchability=matchability,
        null_logit=-10.0,
        device="cpu",
    )
    assert result.mean_displacement[0, 0] > 0.99
    assert abs(float(result.mean_displacement[0, 1])) < 1e-3


def test_heldout_zero_mass_is_not_multiplied_by_non_null_twice():
    correlation = CorrelationDistribution(
        pixel_xy=np.zeros((1, 2), dtype=np.float32),
        xyz=np.zeros((1, 3), dtype=np.float32),
        maplet_ids=np.asarray([7]),
        offsets_xy=np.asarray([[0, 0], [1, 0]], dtype=np.float32),
        probabilities=np.asarray([[0.4, 0.1]], dtype=np.float32),
        null_probability=np.asarray([0.5], dtype=np.float32),
        mean_displacement=np.zeros((1, 2), dtype=np.float32),
        covariance=np.zeros((1, 2, 2), dtype=np.float32),
        entropy=np.zeros(1, dtype=np.float32),
        matchability=np.ones(1, dtype=np.float32),
    )
    value = zero_displacement_log_likelihood(correlation, np.asarray([7]))
    assert np.isclose(value, np.log(0.4))


def test_heldout_verifier_uses_fixed_surface_identity_denominator():
    def distribution(probability, surface_ids):
        count = len(probability)
        return CorrelationDistribution(
            pixel_xy=np.zeros((count, 2), dtype=np.float32),
            xyz=np.zeros((count, 3), dtype=np.float32),
            maplet_ids=np.full(count, 7, dtype=np.int64),
            offsets_xy=np.asarray([[0, 0]], dtype=np.float32),
            probabilities=np.asarray(probability, dtype=np.float32)[:, None],
            null_probability=1.0 - np.asarray(probability, dtype=np.float32),
            mean_displacement=np.zeros((count, 2), dtype=np.float32),
            covariance=np.zeros((count, 2, 2), dtype=np.float32),
            entropy=np.zeros(count, dtype=np.float32),
            matchability=np.ones(count, dtype=np.float32),
            surface_ids=np.asarray(surface_ids, dtype=np.int64),
        )

    before = distribution([0.9, 0.9, 0.2], [1, 1, 2])
    after = distribution([0.8, 0.3, 0.3, 0.3], [1, 2, 2, 2])
    _accepted, evidence = accept_pose_update(
        before,
        after,
        fit_maplet_ids=np.asarray([7]),
        heldout_maplet_ids=np.asarray([8]),
    )
    assert evidence["fit_surface_count"] == 2
    assert np.isclose(
        evidence["fit_before"], np.mean(np.log([0.9, 0.2]))
    )
    assert np.isclose(
        evidence["fit_after"], np.mean(np.log([0.8, 0.3]))
    )


def test_density_ratio_null_is_not_overwhelmed_by_search_window_size():
    rng = np.random.default_rng(19)
    channels, height, width = 64, 12, 12
    rendered_feature = rng.normal(size=(channels, height, width)).astype(
        np.float32
    )
    query_feature = rng.normal(size=(channels, height, width)).astype(np.float32)
    rendered_feature /= np.linalg.norm(
        rendered_feature, axis=0, keepdims=True
    )
    query_feature /= np.linalg.norm(query_feature, axis=0, keepdims=True)
    mask = np.zeros((height, width), dtype=bool)
    mask[3:-3, 3:-3] = True
    rendered = RenderedMapletAtlases(
        feature=rendered_feature,
        xyz=np.zeros((height, width, 3), dtype=np.float32),
        normal=np.zeros((height, width, 3), dtype=np.float32),
        uncertainty=np.zeros((height, width), dtype=np.float32),
        maplet_id=np.where(mask, 7, -1),
        mask=mask,
        depth=mask.astype(np.float32),
    )
    result = local_correlation_distribution(
        rendered, query_feature, radius=4, temperature=0.07, device="cpu"
    )
    assert 0.25 < float(np.median(result.null_probability)) < 0.75


def test_geometry_solver_recovers_joint_pose_step():
    camera = _camera()
    xyz = np.asarray(
        [
            [-1, -0.5, 4],
            [-0.5, 0.5, 5],
            [0.2, -0.7, 4.5],
            [0.8, 0.6, 6],
            [1.1, -0.2, 5.5],
            [-0.8, 0.9, 6.5],
            [0.4, 0.2, 3.5],
            [-0.2, -0.1, 7],
        ],
        dtype=np.float32,
    )
    expected = np.asarray([0.002, -0.003, 0.001, 0.03, -0.02, 0.01])
    pixels, jacobian = projection_jacobian(xyz, np.eye(4), camera)
    displacement = np.einsum("nij,j->ni", jacobian, expected)
    count = len(xyz)
    correlation = CorrelationDistribution(
        pixel_xy=pixels.astype(np.float32),
        xyz=xyz,
        maplet_ids=np.arange(count) % 3,
        offsets_xy=np.asarray([[0, 0]], dtype=np.float32),
        probabilities=np.ones((count, 1), dtype=np.float32),
        null_probability=np.zeros(count, dtype=np.float32),
        mean_displacement=displacement.astype(np.float32),
        covariance=np.tile(np.eye(2, dtype=np.float32)[None] * 0.01, (count, 1, 1)),
        entropy=np.zeros(count, dtype=np.float32),
        matchability=np.ones(count, dtype=np.float32),
    )
    result = solve_correlation_se3_update(
        correlation,
        np.eye(4),
        camera,
        minimum_points=6,
        dominant_mode_conditioning=False,
    )
    assert result.success
    assert np.allclose(result.delta, expected, atol=2e-3)


def test_raster_subpixel_origin_is_accounted_for_by_geometry_solver():
    xyz, primitive_ids, valid, _audit = canonical_maplet_geometry(
        _source(), _maplets(), resolution=8
    )
    atlas = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([7]),
        centers=np.asarray([[0, 0, 2]], dtype=np.float32),
        frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.2, 0.1]], dtype=np.float32),
        xyz=xyz,
        primitive_ids=primitive_ids,
        features=np.ones((1, 1, 8, 8), dtype=np.float32),
        variance=np.zeros((1, 8, 8), dtype=np.float32),
        support_count=np.ones((1, 8, 8), dtype=np.int32),
        valid_mask=valid,
        metadata={},
    )
    initial_pose = se3_exp(
        np.asarray([0.002, -0.003, 0.001, 0.02, -0.01, 0.005])
    )
    rendered = render_selected_maplet_atlases(
        atlas,
        np.asarray([7]),
        initial_pose,
        _camera(),
        width=25,
        height=20,
    )
    yy, xx = np.nonzero(rendered.mask)
    gt_pixels, _depth = projection_jacobian(
        rendered.xyz[yy, xx], np.eye(4), _camera()
    )
    gt_grid = np.stack(
        [
            (gt_pixels[:, 0] + 0.5) / 4.0 - 0.5,
            (gt_pixels[:, 1] + 0.5) / 4.0 - 0.5,
        ],
        axis=1,
    )
    displacement = gt_grid - np.stack([xx, yy], axis=1)
    count = len(xx)
    correlation = CorrelationDistribution(
        pixel_xy=np.stack([xx, yy], axis=1).astype(np.float32),
        xyz=rendered.xyz[yy, xx],
        maplet_ids=rendered.maplet_id[yy, xx],
        offsets_xy=np.zeros((1, 2), dtype=np.float32),
        probabilities=np.ones((count, 1), dtype=np.float32),
        null_probability=np.zeros(count, dtype=np.float32),
        mean_displacement=displacement.astype(np.float32),
        covariance=np.tile(
            np.eye(2, dtype=np.float32)[None] * 0.01, (count, 1, 1)
        ),
        entropy=np.zeros(count, dtype=np.float32),
        matchability=np.ones(count, dtype=np.float32),
    )
    result = solve_correlation_se3_update(
        correlation,
        initial_pose,
        _camera(),
        minimum_points=6,
        displacement_scale_xy=(4.0, 4.0),
        dominant_mode_conditioning=False,
    )
    assert result.success
    assert np.linalg.norm(result.updated_pose_w2c[:3, 3]) < 1e-3


def test_v6_metric_encoder_preserves_stride4_and_trains_matchability():
    model = V6MetricEncoder(
        V6MetricEncoderConfig(radio_dim=8, hidden_dim=16, output_dim=8)
    )
    output = model(torch.randn(2, 8, 2, 3), torch.randn(2, 3, 32, 48))
    assert output["fine"].shape == (2, 8, 8, 12)
    assert output["middle"].shape == (2, 8, 4, 6)
    assert output["coarse"].shape == (2, 8, 2, 3)
    labels = torch.zeros_like(output["matchability_logits"])
    labels[:, :, 2:4, 3:6] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["matchability_logits"], labels
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "null_logits" not in output
    assert "log_variance" not in output


def test_v6_independent_pyramid_starts_from_legacy_average_pool():
    model = V6MetricEncoder(
        V6MetricEncoderConfig(radio_dim=8, hidden_dim=16, output_dim=8)
    )
    output = model(torch.randn(1, 8, 2, 3), torch.randn(1, 3, 32, 48))
    expected_middle = torch.nn.functional.normalize(
        torch.nn.functional.avg_pool2d(output["fine"], 2, 2), dim=1
    )
    expected_coarse = torch.nn.functional.normalize(
        torch.nn.functional.avg_pool2d(output["fine"], 4, 4), dim=1
    )
    assert torch.allclose(output["middle"], expected_middle, atol=1e-6)
    assert torch.allclose(output["coarse"], expected_coarse, atol=1e-6)


def test_local_correlation_training_supervises_null_and_displacement():
    torch.manual_seed(9)
    features = torch.nn.functional.normalize(
        torch.randn(1, 8, 12, 12), dim=1
    )
    shifted = torch.zeros_like(features)
    shifted[:, :, :, 2:] = features[:, :, :, :-2]
    map_output = {"fine": features}
    query_output = {
        "fine": shifted,
        "null_logits": torch.zeros(1, 1, 12, 12, requires_grad=True),
        "matchability_logits": torch.zeros(
            1, 1, 12, 12, requires_grad=True
        ),
    }
    coordinates = torch.tensor([[[4.0, 4.0], [6.0, 7.0]]])
    loss = local_correlation_training_loss(
        map_output,
        query_output,
        map_xy=coordinates,
        proposal_xy=coordinates,
        target_xy=coordinates + torch.tensor([2.0, 0.0]),
        positive_mask=torch.tensor([[True, False]]),
        radius=3,
    )
    assert torch.isfinite(loss["total"])
    assert 0.0 < float(loss["positive_fraction"]) < 1.0
    assert torch.allclose(
        loss["correlation_nll"],
        0.5
        * (
            loss["positive_correlation_nll"]
            + loss["null_correlation_nll"]
        ),
    )
    assert 0.0 <= float(loss["mode_recall_radius1"]) <= 1.0
    loss["total"].backward()


def test_training_marginalizes_rendered_appearance_modes():
    torch.manual_seed(13)
    query = torch.nn.functional.normalize(torch.randn(1, 4, 8, 8), dim=1)
    coordinates = torch.tensor([[[3.0, 3.0], [5.0, 4.0]]])
    descriptor = torch.nn.functional.normalize(
        torch.randn(1, 2, 4), dim=-1
    )
    modes = torch.stack([-descriptor, descriptor], dim=2)
    output = {
        "fine": query,
        "null_logits": torch.zeros(1, 1, 8, 8, requires_grad=True),
        "matchability_logits": torch.zeros(
            1, 1, 8, 8, requires_grad=True
        ),
    }
    loss = local_correlation_training_loss(
        {"fine": query},
        output,
        map_xy=coordinates,
        proposal_xy=coordinates,
        target_xy=coordinates,
        positive_mask=torch.ones(1, 2, dtype=torch.bool),
        map_descriptor_modes_override=modes,
        map_mode_log_prior=torch.log(torch.full((1, 2, 2), 0.5)),
        radius=1,
    )
    assert torch.isfinite(loss["total"])
    loss["total"].backward()


def test_analytic_one_step_pose_loss_is_differentiable():
    torch.manual_seed(11)
    count = 12
    jacobian = torch.randn(1, count, 2, 6)
    expected = torch.tensor(
        [[0.01, -0.02, 0.005, 0.03, -0.01, 0.02]]
    )
    displacement = torch.einsum("bnij,bj->bni", jacobian, expected)
    displacement.requires_grad_(True)
    covariance = (
        torch.eye(2)[None, None].repeat(1, count, 1, 1) * 0.01
    )
    pose_loss = analytic_one_step_pose_loss(
        displacement,
        covariance,
        jacobian,
        torch.ones(1, count, dtype=torch.bool),
        expected,
        displacement_scale=1.0,
    )
    assert float(pose_loss) < 1e-5
    pose_loss.backward()
    assert displacement.grad is not None


def test_v6_retrieval_preserves_groups_and_transfers_omitted_mass_to_null(
    tmp_path,
):
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10, 11, 12]),
        centers=np.zeros((3, 3), dtype=np.float32),
        normals=np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (3, 1)),
        extents=np.ones((3, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1, 2, 3]),
        descriptors=np.asarray([[1, 0], [0.8, 0.2], [-1, 0]], dtype=np.float32),
        descriptor_weights=np.ones(3),
        descriptor_centers=np.asarray(
            [[0, 0, 4], [1, 0, 4], [2, 0, 4]], dtype=np.float32
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (3, 1, 1)
        ),
        query_projection=np.eye(2, dtype=np.float32),
        quality_scores=np.ones(3),
        descriptor_uncertainties=np.zeros(3),
        metadata={"vfm_layer": "radio_final"},
    )
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        bank,
        preliminary_candidates=3,
        maximum_maplets=1,
        probability_calibration=_probability_calibration(),
    )
    group = result.groups[0]
    assert group.query_region_xy.tolist() == [20, 30]
    assert group.maplet_ids.tolist() == result.ranked_maplet_ids.tolist()
    assert group.maplet_ids.size == 1
    assert group.omitted_probability > 0.0
    assert np.isclose(group.probabilities.sum() + group.null_probability, 1.0)
    assert group.candidate_centers.shape == (1, 3)
    assert group.component_offsets.tolist() == [0, 1]
    assert group.component_centers.shape == (1, 3)
    assert group.component_covariances.shape == (1, 3, 3)
    assert np.isclose(
        group.component_probabilities.sum()
        + group.spatial_null_probabilities[0],
        1.0,
    )
    path = tmp_path / "retrieval-maplets.npz"
    bank.save_npz(path)
    restored = SurfaceRetrievalMapletBank.load_npz(path)
    assert np.allclose(restored.descriptor_centers, bank.descriptor_centers)
    assert np.allclose(
        restored.descriptor_covariances, bank.descriptor_covariances
    )
    assert np.allclose(restored.query_projection, bank.query_projection)
    projected = restored.project_query_feature_map(
        np.asarray([[[3.0]], [[4.0]]], dtype=np.float32)
    )
    assert np.allclose(projected[:, 0, 0], [0.6, 0.8])
    subset = restored.subset_maplets(np.asarray([11]))
    assert subset.maplet_ids.tolist() == [11]
    assert subset.descriptor_offsets.tolist() == [0, 1]
    assert np.allclose(subset.descriptor_centers[0], [1, 0, 4])


def test_retrieval_marginalizes_descriptor_conditioned_surface_location():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 2]),
        descriptors=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        descriptor_weights=np.asarray([0.5, 0.5]),
        descriptor_centers=np.asarray(
            [[-0.5, 0, 4], [0.5, 0, 4]], dtype=np.float32
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (2, 1, 1)
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    result = retrieve_candidate_groups(
        np.asarray([[0, 1]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        bank,
        preliminary_candidates=1,
        probability_calibration=_probability_calibration(),
    )
    assert result.groups[0].candidate_centers[0, 0] > 0.49
    assert result.groups[0].component_offsets.tolist() == [0, 2]
    best = int(np.argmax(result.groups[0].component_probabilities))
    assert result.groups[0].component_probabilities[best] > 0.99
    assert result.groups[0].component_centers[best, 0] > 0.49


def test_retrieval_aggregates_appearance_modes_at_one_surface_location():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 3]),
        descriptors=np.asarray(
            [[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32
        ),
        descriptor_weights=np.asarray([0.4, 0.4, 0.2]),
        descriptor_centers=np.asarray(
            [[-0.5, 0, 4], [-0.5, 0, 4], [0.5, 0, 4]],
            dtype=np.float32,
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (3, 1, 1)
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        bank,
        preliminary_candidates=1,
        probability_calibration=_probability_calibration(),
    )
    group = result.groups[0]
    assert group.component_offsets.tolist() == [0, 2]
    assert group.component_centers.shape == (2, 3)
    assert np.isclose(
        group.component_probabilities.sum()
        + group.spatial_null_probabilities[0],
        1.0,
    )
    assert group.component_probabilities[0] > 0.99


def test_retrieval_separates_maplet_identity_from_spatial_likelihood():
    common = dict(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 2]),
        descriptor_weights=np.asarray([0.5, 0.5]),
        descriptor_centers=np.asarray(
            [[-0.5, 0, 4], [0.5, 0, 4]], dtype=np.float32
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (2, 1, 1)
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    identity_bank = SurfaceRetrievalMapletBank(
        descriptors=np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        **common,
    )
    spatial_bank = SurfaceRetrievalMapletBank(
        descriptors=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
        **common,
    )
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        identity_bank,
        preliminary_candidates=1,
        spatial_query_descriptors=np.asarray([[0, 1]], dtype=np.float32),
        spatial_bank=spatial_bank,
        probability_calibration=_probability_calibration(),
    )
    group = result.groups[0]
    best = int(np.argmax(group.component_probabilities))
    assert group.component_centers[best, 0] > 0.49
    assert group.candidate_centers[0, 0] > 0.49


def test_spatially_unavailable_maplet_remains_in_identity_posterior():
    identity_bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10, 11]),
        centers=np.asarray([[0, 0, 4], [1, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
        extents=np.ones((2, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1, 2]),
        descriptors=np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        descriptor_weights=np.ones(2, dtype=np.float32),
        descriptor_centers=np.asarray(
            [[0, 0, 4], [1, 0, 4]], dtype=np.float32
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (2, 1, 1)
        ),
        quality_scores=np.ones(2),
        descriptor_uncertainties=np.zeros(2),
        metadata={"vfm_layer": "radio_final"},
    )
    spatial_bank = identity_bank.subset_maplets(np.asarray([10]))
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        identity_bank,
        preliminary_candidates=2,
        maximum_maplets=2,
        spatial_query_descriptors=np.asarray([[1, 0]], dtype=np.float32),
        spatial_bank=spatial_bank,
        probability_calibration=_probability_calibration(),
    )
    group = result.groups[0]
    row = int(np.flatnonzero(group.maplet_ids == 11)[0])
    assert 11 in result.ranked_maplet_ids
    assert not bool(group.spatial_available[row])
    assert float(group.spatial_null_probabilities[row]) == 1.0
    assert int(group.component_offsets[row]) == int(
        group.component_offsets[row + 1]
    )


def test_surface_mode_truncation_transfers_mass_to_spatial_null():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 3]),
        descriptors=np.asarray(
            [[1, 0], [1, 0], [1, 0]], dtype=np.float32
        ),
        descriptor_weights=np.ones(3, dtype=np.float32) / 3.0,
        descriptor_centers=np.asarray(
            [[-1, 0, 4], [0, 0, 4], [1, 0, 4]], dtype=np.float32
        ),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (3, 1, 1)
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        bank,
        preliminary_candidates=1,
        maximum_components_per_maplet=1,
        component_nms_distance_m=0.0,
        probability_calibration=_probability_calibration(),
    )
    group = result.groups[0]
    retained = float(np.sum(group.component_probabilities))
    spatial_null = float(group.spatial_null_probabilities[0])
    assert retained < 0.34
    assert spatial_null > 0.66
    assert np.isclose(retained + spatial_null, 1.0, atol=1e-6)


def test_garbage_spatial_scores_increase_calibrated_spatial_null():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1, 0]], dtype=np.float32),
        descriptor_weights=np.ones(1, dtype=np.float32),
        descriptor_centers=np.asarray([[0, 0, 4]], dtype=np.float32),
        descriptor_covariances=np.eye(3, dtype=np.float32)[None] * 0.01,
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    calibration = V6ProbabilityCalibration(
        identity=_probability_calibration().identity,
        spatial=DistributionNullCalibration(
            weights=np.asarray([5.0, -0.6, 0.0, 0.0, 0.0])
        ),
        metadata={
            "artifact_type": "v6_probability_calibration",
            "calibration_trajectory_ids": ["validation"],
            "strict_holdout_trajectory_ids": ["test"],
            "spatial_calibration_condition": (
                "identity_is_true_and_atlas_available"
            ),
            "calibration_objective": "unweighted_bernoulli_nll",
        },
    )

    def spatial_null(descriptor):
        return float(
            retrieve_candidate_groups(
                np.asarray([[1, 0]], dtype=np.float32),
                np.asarray([[20, 30]], dtype=np.float32),
                np.asarray([[5, 7]], dtype=np.float32),
                bank,
                preliminary_candidates=1,
                spatial_query_descriptors=np.asarray(
                    [descriptor], dtype=np.float32
                ),
                spatial_bank=bank,
                probability_calibration=calibration,
            ).groups[0].spatial_null_probabilities[0]
        )

    assert spatial_null([0, 1]) > spatial_null([1, 0])


def test_descriptor_oracle_deduplicates_by_surface_position_not_local_rank():
    xyz = np.asarray(
        [[[[-0.1, 0.0, 5.0], [0.1, 0.0, 5.0]]]], dtype=np.float32
    )
    atlas = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        frames=np.asarray([np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.2, 0.1, 0.1]], dtype=np.float32),
        xyz=xyz,
        primitive_ids=np.asarray([[[0, 1]]], dtype=np.int64),
        features=np.ones((1, 2, 1, 2), dtype=np.float32),
        variance=np.zeros((1, 1, 2), dtype=np.float32),
        support_count=np.ones((1, 1, 2), dtype=np.int32),
        valid_mask=np.ones((1, 1, 2), dtype=bool),
    )
    spatial_bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 2]),
        descriptors=np.eye(2, dtype=np.float32),
        descriptor_weights=np.ones(2, dtype=np.float32),
        descriptor_centers=xyz.reshape(2, 3),
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01, (2, 1, 1)
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    identity_bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1, dtype=np.float32),
        descriptor_centers=np.asarray(
            [[0.0, 0.0, 5.0]], dtype=np.float32
        ),
        descriptor_covariances=np.eye(3, dtype=np.float32)[None] * 0.01,
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    camera = _camera()
    image_xy = np.asarray([[49.0, 40.0], [51.0, 40.0]])
    view = SimpleNamespace(
        visible_rows=np.asarray([0, 1], dtype=np.int64),
        image_xy=image_xy,
        pose_w2c=np.eye(4),
        camera=camera,
    )
    groups = tuple(
        QueryMapletGroup(
            query_region_xy=image_xy[index],
            query_region_extent=np.asarray([0.9, 0.9]),
            maplet_ids=np.asarray([10]),
            probabilities=np.asarray([1.0]),
            null_probability=0.0,
            omitted_probability=0.0,
            component_offsets=np.asarray([0, 1]),
            component_centers=xyz.reshape(2, 3)[index : index + 1],
            component_covariances=np.eye(3)[None],
            component_probabilities=np.asarray([1.0]),
        )
        for index in range(2)
    )
    result = _coarse_observation_oracles(
        groups, view, atlas, identity_bank, spatial_bank
    )
    assert (
        result["true_maplet_oracle_spatial_component"][
            "correspondence_count"
        ]
        == 2
    )
    assert (
        result["true_maplet_descriptor_map_component"][
            "correspondence_count"
        ]
        == 2
    )


def test_deterministic_regional_proposal_does_not_require_random_trials():
    points = np.asarray(
        [
            [-0.8, -0.5, 4.0],
            [0.7, -0.6, 4.2],
            [-0.6, 0.7, 4.5],
            [0.8, 0.6, 4.8],
            [-0.3, -0.2, 5.2],
            [0.4, -0.1, 5.5],
            [-0.2, 0.4, 5.8],
            [0.5, 0.3, 6.0],
        ],
        dtype=np.float32,
    )
    camera = _camera()
    image_xy = np.stack(
        [
            50.0 * points[:, 0] / points[:, 2] + 50.0,
            50.0 * points[:, 1] / points[:, 2] + 40.0,
        ],
        axis=1,
    )
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.mean(points, axis=0, keepdims=True),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.ones((1, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, points.shape[0]]),
        descriptors=np.ones((points.shape[0], 2), dtype=np.float32),
        descriptor_weights=np.ones(points.shape[0], dtype=np.float32),
        descriptor_centers=points,
        descriptor_covariances=np.tile(
            np.eye(3, dtype=np.float32)[None] * 0.01,
            (points.shape[0], 1, 1),
        ),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    groups = tuple(
        QueryMapletGroup(
            query_region_xy=image_xy[index],
            query_region_extent=np.asarray([4.0, 4.0]),
            maplet_ids=np.asarray([10]),
            probabilities=np.asarray([1.0]),
            null_probability=0.0,
            omitted_probability=0.0,
            component_offsets=np.asarray([0, 1]),
            component_centers=points[index : index + 1],
            component_covariances=np.eye(3)[None] * 0.01,
            component_probabilities=np.asarray([1.0]),
        )
        for index in range(points.shape[0])
    )
    hypotheses = propose_maplet_surface_mode_poses(
        groups,
        bank,
        camera,
        trials=0,
        em_iterations=0,
    )
    assert hypotheses
    assert all(
        item.source.startswith("regional_center_pseudo_pnp")
        for item in hypotheses
    )


def test_maplet_footprint_score_uses_region_area_without_surface_points():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        tangent_frames=np.eye(3, dtype=np.float32)[None],
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1),
        descriptor_centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        descriptor_covariances=np.eye(3, dtype=np.float32)[None] * 0.01,
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={
            "vfm_layer": "radio_final",
            "has_canonical_tangent_frames": True,
        },
    )
    group = QueryMapletGroup(
        query_region_xy=np.asarray([50.0, 40.0]),
        query_region_extent=np.asarray([8.0, 8.0]),
        maplet_ids=np.asarray([10]),
        probabilities=np.asarray([0.9]),
        null_probability=0.1,
        omitted_probability=0.0,
    )
    correct, correct_support = score_maplet_footprint_pose(
        np.eye(4), (group,), bank, _camera()
    )
    wrong_pose = np.eye(4)
    wrong_pose[0, 3] = 5.0
    wrong, wrong_support = score_maplet_footprint_pose(
        wrong_pose, (group,), bank, _camera()
    )
    assert correct > wrong
    assert correct > 0.0
    assert correct_support == 1
    assert wrong_support == 0


def test_retrieval_maplet_roundtrip_preserves_canonical_frames(tmp_path):
    frame = np.asarray(
        [[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.asarray([[0.6, 0.2, 0.1]], dtype=np.float32),
        tangent_frames=frame,
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={
            "vfm_layer": "radio_final",
            "has_canonical_tangent_frames": True,
        },
    )
    path = tmp_path / "maplets.npz"
    bank.save_npz(path)
    loaded = SurfaceRetrievalMapletBank.load_npz(path)
    np.testing.assert_allclose(loaded.tangent_frames, frame)
    assert loaded.metadata["has_canonical_tangent_frames"] is True
    with np.load(path, allow_pickle=False) as payload:
        assert "tangent_frames" in payload.files
        assert not any(
            value in payload.files
            for value in (
                "mapping_rgb",
                "mapping_image_ids",
                "observation_descriptors",
            )
        )


def test_anonymous_pose_vote_uses_query_region_geometry():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1),
        descriptor_centers=np.asarray(
            [[0.0, 0.0, 5.0]], dtype=np.float32
        ),
        descriptor_covariances=np.eye(3, dtype=np.float32)[None],
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    votes = AnonymousMapletPoseVoteBank(
        component_maplet_ids=np.asarray([10]),
        vote_offsets=np.asarray([0, 2]),
        camera_centers=np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
        ),
        rotations_w2c=np.tile(np.eye(3)[None], (2, 1, 1)),
        vote_weights=np.asarray([0.5, 0.5]),
        translation_sigma_m=np.zeros(2),
        rotation_sigma_deg=np.zeros(2),
        region_xy_mean=np.asarray([[0.2, 0.5], [0.8, 0.5]]),
        region_xy_covariance=np.tile(
            np.eye(2, dtype=np.float32)[None] * 1e-4, (2, 1, 1)
        ),
        region_support_count=np.asarray([3, 3]),
        component_descriptor_sha256=retrieval_descriptor_sha256(bank),
        retrieval_geometry_sha256=retrieval_geometry_sha256(bank),
        metadata={
            "artifact_type": "v6_anonymous_maplet_pose_vote_bank",
            "representation": (
                "maplet_component_pose_and_region_sufficient_statistics"
            ),
            "uses_query_region_geometry": True,
            "mapping_trajectory_ids": ["mapping"],
            "stores_mapping_rgb": False,
            "stores_mapping_image_ids": False,
        },
    )
    group = QueryMapletGroup(
        query_region_xy=np.asarray([19.5, 49.5]),
        query_region_extent=np.asarray([2.0, 2.0]),
        maplet_ids=np.asarray([10]),
        probabilities=np.asarray([0.9]),
        null_probability=0.1,
        omitted_probability=0.0,
    )
    hypotheses = vote_maplet_poses(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        bank,
        votes,
        candidate_groups=(group,),
        image_size_wh=(100, 100),
        translation_kernel_m=0.1,
        rotation_kernel_deg=0.1,
        maximum_modes=2,
    )
    assert len(hypotheses) == 2
    top_center = (
        -hypotheses[0].pose_w2c[:3, :3].T
        @ hypotheses[0].pose_w2c[:3, 3]
    )
    assert np.linalg.norm(top_center) < 1e-3


def test_region_conditioned_vote_rotates_source_bearing_to_query():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        tangent_frames=np.eye(3, dtype=np.float32)[None],
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={
            "vfm_layer": "radio_final",
            "has_canonical_tangent_frames": True,
        },
    )
    votes = AnonymousMapletPoseVoteBank(
        component_maplet_ids=np.asarray([10]),
        vote_offsets=np.asarray([0, 1]),
        camera_centers=np.zeros((1, 3)),
        rotations_w2c=np.eye(3)[None],
        vote_weights=np.ones(1),
        translation_sigma_m=np.zeros(1),
        rotation_sigma_deg=np.zeros(1),
        region_xy_mean=np.asarray([[0.2, 0.5]]),
        region_xy_covariance=np.eye(2, dtype=np.float32)[None] * 1e-4,
        region_support_count=np.asarray([4]),
        component_descriptor_sha256=retrieval_descriptor_sha256(bank),
        retrieval_geometry_sha256=retrieval_geometry_sha256(bank),
        metadata={
            "artifact_type": "v6_anonymous_maplet_pose_vote_bank",
            "representation": (
                "maplet_component_pose_and_region_sufficient_statistics"
            ),
            "uses_query_region_geometry": True,
            "mapping_trajectory_ids": ["mapping"],
        },
    )
    group = QueryMapletGroup(
        query_region_xy=np.asarray([79.5, 49.5]),
        query_region_extent=np.asarray([2.0, 2.0]),
        maplet_ids=np.asarray([10]),
        probabilities=np.asarray([0.9]),
        null_probability=0.1,
        omitted_probability=0.0,
    )
    camera = _camera()
    hypotheses = vote_maplet_poses(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        bank,
        votes,
        candidate_groups=(group,),
        image_size_wh=(100, 80),
        camera=camera,
        maximum_modes=1,
    )
    assert len(hypotheses) == 1
    source = _bearing_from_normalized_image_xy(
        np.asarray([[0.2, 0.5]]), camera
    )[0]
    target = _bearing_from_normalized_image_xy(
        np.asarray([[0.8, 0.625]]), camera
    )[0]
    rotated = hypotheses[0].pose_w2c[:3, :3] @ source
    assert np.dot(rotated, target) > 1.0 - 1e-6


def test_pose_vote_rejects_retrieval_geometry_lineage_mismatch():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10]),
        centers=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        extents=np.asarray([[0.5, 0.5, 0.1]], dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1]),
        descriptors=np.asarray([[1.0, 0.0]], dtype=np.float32),
        descriptor_weights=np.ones(1),
        quality_scores=np.ones(1),
        descriptor_uncertainties=np.zeros(1),
        metadata={"vfm_layer": "radio_final"},
    )
    votes = AnonymousMapletPoseVoteBank(
        component_maplet_ids=np.asarray([10]),
        vote_offsets=np.asarray([0, 1]),
        camera_centers=np.zeros((1, 3)),
        rotations_w2c=np.eye(3)[None],
        vote_weights=np.ones(1),
        translation_sigma_m=np.zeros(1),
        rotation_sigma_deg=np.zeros(1),
        region_xy_mean=np.asarray([[0.5, 0.5]]),
        region_xy_covariance=np.eye(2, dtype=np.float32)[None],
        region_support_count=np.asarray([1]),
        component_descriptor_sha256=retrieval_descriptor_sha256(bank),
        retrieval_geometry_sha256=retrieval_geometry_sha256(bank),
        metadata={
            "artifact_type": "v6_anonymous_maplet_pose_vote_bank",
            "representation": (
                "maplet_component_pose_and_region_sufficient_statistics"
            ),
            "uses_query_region_geometry": True,
            "mapping_trajectory_ids": ["mapping"],
        },
    )
    shifted = SurfaceRetrievalMapletBank(
        maplet_ids=bank.maplet_ids,
        centers=bank.centers + np.asarray([[1.0, 0.0, 0.0]]),
        normals=bank.normals,
        extents=bank.extents,
        descriptor_offsets=bank.descriptor_offsets,
        descriptors=bank.descriptors,
        descriptor_weights=bank.descriptor_weights,
        quality_scores=bank.quality_scores,
        descriptor_uncertainties=bank.descriptor_uncertainties,
        metadata={"vfm_layer": "radio_final"},
    )
    with pytest.raises(ValueError, match="geometry lineage"):
        vote_maplet_poses(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            shifted,
            votes,
            maximum_modes=1,
        )
