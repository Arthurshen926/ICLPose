import numpy as np
import torch

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
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
    canonical_maplet_geometry,
)
from feature_extract.vfm.localization_v6.metric_encoder import (
    V6MetricEncoder,
    V6MetricEncoderConfig,
    probabilistic_metric_losses,
)
from feature_extract.vfm.localization_v6.metric_training import (
    analytic_one_step_pose_loss,
    local_correlation_training_loss,
)
from feature_extract.vfm.localization_v6.primitive_contributors import (
    PrimitiveContributorBuffer,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    retrieve_candidate_groups,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.se3_update import (
    projection_jacobian,
    se3_exp,
    solve_correlation_se3_update,
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
    assert report["assignment"] == "gsplat_topk_contributor_source_index"


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


def test_v6_metric_encoder_preserves_stride4_and_trains_probability_heads():
    model = V6MetricEncoder(
        V6MetricEncoderConfig(radio_dim=8, hidden_dim=16, output_dim=8)
    )
    output = model(torch.randn(2, 8, 2, 3), torch.randn(2, 3, 32, 48))
    assert output["fine"].shape == (2, 8, 8, 12)
    labels = torch.zeros_like(output["matchability_logits"])
    labels[:, :, 2:4, 3:6] = 1
    error = torch.ones_like(labels)
    losses = probabilistic_metric_losses(
        output,
        match_labels=labels,
        displacement_error=error,
        displacement_valid=labels.bool(),
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


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


def test_v6_retrieval_preserves_groups_and_transfers_omitted_mass_to_null():
    bank = SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10, 11, 12]),
        centers=np.zeros((3, 3), dtype=np.float32),
        normals=np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (3, 1)),
        extents=np.ones((3, 3), dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1, 2, 3]),
        descriptors=np.asarray([[1, 0], [0.8, 0.2], [-1, 0]], dtype=np.float32),
        descriptor_weights=np.ones(3),
        quality_scores=np.ones(3),
        descriptor_uncertainties=np.zeros(3),
        metadata={"vfm_layer": "radio_final"},
    )
    result = retrieve_candidate_groups(
        np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[20, 30]], dtype=np.float32),
        np.asarray([[5, 7]], dtype=np.float32),
        bank,
        preliminary_candidates=1,
    )
    group = result.groups[0]
    assert group.query_region_xy.tolist() == [20, 30]
    assert group.omitted_probability > 0.0
    assert np.isclose(group.probabilities.sum() + group.null_probability, 1.0)
