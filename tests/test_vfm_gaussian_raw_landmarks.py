import warnings
from typing import Optional

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_raw_landmarks import (
    GaussianTokenContributionConfig,
    GaussianTokenContributionView,
    RawGaussianFeatureAggregationConfig,
    aggregate_raw_vfm_features_from_contribution_visibility,
    VfmGaussianAnchorVoteConfig,
    aggregate_raw_vfm_features_from_token_contributions,
    aggregate_raw_vfm_features_to_gaussian_anchors,
    build_gaussian_token_contribution_view,
    project_gaussian_anchor_map_features,
    sample_gaussian_indices_from_votes,
    subset_gaussian_anchor_map_by_source_indices,
    vote_gaussians_from_token_contribution_views,
    vote_gaussians_from_vfm_token_saliency_configs,
    vote_gaussians_from_vfm_token_saliency,
    _project_xyz_to_grid,
)
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, GaussianVFMSource
from feature_extract.tools.vfm.build_stage_h2_raw_gaussian_anchor_map import _source_geometry_stats


class _ToySelector:
    def encode_rows(self, rows, device="cpu", batch_size=65536):
        values = np.asarray(rows, dtype=np.float32)
        projected = values[:, :2]
        norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-6)
        return projected / norms


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=4, height=4, params=(1.0, 1.0, 2.0, 2.0))


def _source() -> GaussianVFMSource:
    return GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [4.0, 0.0, 4.0], [0.0, 4.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        scale=np.asarray([0.02, 0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([10, 11, 12], dtype=np.int64),
    )


def _view() -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    fmap[:, 2, 3] = np.asarray([0.0, 2.0, 0.0], dtype=np.float32)
    fmap[:, 3, 2] = np.asarray([0.0, 0.0, 3.0], dtype=np.float32)
    return GaussianVFMFeatureView(
        image_id="ref.png",
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def _view_with_feature(image_id: str, feature: np.ndarray) -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray(feature, dtype=np.float32)
    return GaussianVFMFeatureView(
        image_id=image_id,
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def _contribution_view(
    image_id: str,
    feature_map: np.ndarray,
    top_contributor: np.ndarray,
    top_alpha: Optional[np.ndarray] = None,
    alpha_entropy: Optional[np.ndarray] = None,
) -> GaussianTokenContributionView:
    return GaussianTokenContributionView(
        image_id=image_id,
        feature_map=np.asarray(feature_map, dtype=np.float32),
        top_contributor=np.asarray(top_contributor, dtype=np.int64),
        top_alpha=None if top_alpha is None else np.asarray(top_alpha, dtype=np.float32),
        alpha_entropy=None if alpha_entropy is None else np.asarray(alpha_entropy, dtype=np.float32),
    )


def _center_saliency_view() -> GaussianVFMFeatureView:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([3.0, 0.0, 0.0], dtype=np.float32)
    return GaussianVFMFeatureView(
        image_id="ref.png",
        feature_map=fmap,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
    )


def test_raw_vfm_features_are_aggregated_after_gaussian_sampling() -> None:
    anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        _source(),
        sampled_source_indices=np.asarray([0, 1], dtype=np.int64),
        views=[_view()],
        config=RawGaussianFeatureAggregationConfig(min_observations=1, l2_normalize_features=True),
    )

    assert len(anchor_map) == 2
    assert anchor_map.feature_dim == 3
    assert anchor_map.source_types.tolist() == ["gaussian_raw_vfm", "gaussian_raw_vfm"]
    assert anchor_map.source_gaussian_indices.tolist() == [10, 11]
    assert anchor_map.support_counts.tolist() == [1, 1]
    assert np.allclose(anchor_map.features[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(anchor_map.features[1], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))


def test_vfm_token_saliency_votes_for_projected_gaussian_owner() -> None:
    votes, summary = vote_gaussians_from_vfm_token_saliency(
        _source(),
        [_view()],
        VfmGaussianAnchorVoteConfig(top_token_fraction=0.05, min_saliency=0.0),
    )

    selected = sample_gaussian_indices_from_votes(
        _source(),
        votes,
        max_anchors=1,
        min_votes=1,
        nms_voxel_size=0.0,
    )

    assert votes.tolist() == [0, 0, 1]
    assert selected.tolist() == [2]
    assert summary["voted_gaussian_count"] == 1
    assert summary["view_count"] == 1


def test_multi_saliency_vote_configs_match_individual_votes() -> None:
    source = _source()
    views = [_view()]
    configs = {
        "norm_top5": VfmGaussianAnchorVoteConfig(top_token_fraction=0.05, saliency_mode="norm"),
        "norm_top10": VfmGaussianAnchorVoteConfig(top_token_fraction=0.10, saliency_mode="norm"),
        "contrast_top5": VfmGaussianAnchorVoteConfig(top_token_fraction=0.05, saliency_mode="local_contrast"),
    }

    votes_by_name, summary = vote_gaussians_from_vfm_token_saliency_configs(source, views, configs)

    assert sorted(votes_by_name) == sorted(configs)
    assert summary["view_count"] == 1
    for name, config in configs.items():
        single_votes, _single_summary = vote_gaussians_from_vfm_token_saliency(source, views, config)
        assert votes_by_name[name].tolist() == single_votes.tolist()
        assert summary["configs"][name]["voted_gaussian_count"] == int(np.sum(single_votes > 0))


def test_vfm_token_saliency_votes_can_ignore_low_opacity_front_floaters() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.01, 0.9], dtype=np.float32),
        scale=np.asarray([0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([20, 21], dtype=np.int64),
    )

    votes, _summary = vote_gaussians_from_vfm_token_saliency(
        source,
        [_center_saliency_view()],
        VfmGaussianAnchorVoteConfig(top_token_fraction=0.05, min_owner_opacity=0.1),
    )

    assert votes.tolist() == [0, 1]


def test_sampling_can_trade_vote_count_against_opacity() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.1, 0.9], dtype=np.float32),
        scale=np.asarray([0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([30, 31], dtype=np.int64),
    )

    selected = sample_gaussian_indices_from_votes(
        source,
        np.asarray([3, 1], dtype=np.int64),
        max_anchors=1,
        min_votes=1,
        nms_voxel_size=0.0,
        opacity_power=2.0,
    )

    assert selected.tolist() == [1]


def test_raw_gaussian_anchor_map_can_be_projected_with_selector() -> None:
    raw_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        _source(),
        sampled_source_indices=np.asarray([0, 1], dtype=np.int64),
        views=[_view()],
        config=RawGaussianFeatureAggregationConfig(min_observations=1, l2_normalize_features=False),
    )

    projected = project_gaussian_anchor_map_features(raw_map, _ToySelector(), output_dim=2)

    assert projected.feature_dim == 2
    assert projected.source_gaussian_indices.tolist() == [10, 11]
    assert np.allclose(projected.features[0], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.allclose(projected.features[1], np.asarray([0.0, 1.0], dtype=np.float32))
    assert projected.metadata["source_feature_dim"] == 3


def test_owner_visibility_filter_skips_invalid_projections_without_integer_cast_warning() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [np.nan, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.8], dtype=np.float32),
        scale=np.asarray([0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([10, 11], dtype=np.int64),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
            source,
            sampled_source_indices=np.asarray([0, 1], dtype=np.int64),
            views=[_view()],
            config=RawGaussianFeatureAggregationConfig(
                min_observations=1,
                require_token_owner_visibility=True,
            ),
        )

    assert len(caught) == 0
    assert anchor_map.source_gaussian_indices.tolist() == [10]


def test_owner_visibility_aggregation_can_ignore_low_opacity_front_floaters() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.01, 0.9], dtype=np.float32),
        scale=np.asarray([0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([20, 21], dtype=np.int64),
    )

    anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        source,
        sampled_source_indices=np.asarray([0, 1], dtype=np.int64),
        views=[_center_saliency_view()],
        config=RawGaussianFeatureAggregationConfig(
            min_observations=1,
            require_token_owner_visibility=True,
            owner_min_opacity=0.1,
        ),
    )

    assert anchor_map.source_gaussian_indices.tolist() == [21]


def test_aggregation_records_support_views_and_feature_variance() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([0.02], dtype=np.float32),
        gaussian_indices=np.asarray([30], dtype=np.int64),
    )

    anchor_map = aggregate_raw_vfm_features_to_gaussian_anchors(
        source,
        sampled_source_indices=np.asarray([0], dtype=np.int64),
        views=[
            _view_with_feature("a.png", np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
            _view_with_feature("b.png", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        ],
        config=RawGaussianFeatureAggregationConfig(
            min_observations=2,
            l2_normalize_observations=True,
            l2_normalize_features=True,
            require_token_owner_visibility=True,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.observation_image_ids == (("a.png", "b.png"),)
    assert anchor_map.support_counts.tolist() == [2]
    assert anchor_map.visibility_counts.tolist() == [2]
    assert anchor_map.feature_variances[0] > 0.0


def test_contribution_visibility_filters_non_dominant_low_alpha_and_high_entropy_tokens() -> None:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 1, 1] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    fmap[:, 1, 2] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    fmap[:, 2, 1] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    owners = np.full((4, 4), -1, dtype=np.int64)
    owners[1, 1] = 0
    owners[1, 2] = 1
    owners[2, 1] = 2
    top_alpha = np.ones((4, 4), dtype=np.float32)
    top_alpha[1, 2] = 0.2
    entropy = np.zeros((4, 4), dtype=np.float32)
    entropy[2, 1] = 1.5

    anchor_map = aggregate_raw_vfm_features_from_token_contributions(
        _source(),
        sampled_source_indices=np.asarray([0, 1, 2], dtype=np.int64),
        contribution_views=[
            _contribution_view(
                "ref.png",
                fmap,
                owners,
                top_alpha=top_alpha,
                alpha_entropy=entropy,
            )
        ],
        config=RawGaussianFeatureAggregationConfig(
            min_observations=1,
            min_contribution_alpha=0.5,
            max_contribution_entropy=0.5,
        ),
    )

    assert anchor_map.source_gaussian_indices.tolist() == [10]
    assert anchor_map.support_counts.tolist() == [1]
    assert anchor_map.observation_image_ids == (("ref.png",),)
    assert np.allclose(anchor_map.features[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_contribution_visibility_aggregation_records_support_and_variance() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([0.02], dtype=np.float32),
        gaussian_indices=np.asarray([30], dtype=np.int64),
    )
    owners = np.full((4, 4), -1, dtype=np.int64)
    owners[2, 2] = 0
    first = np.zeros((3, 4, 4), dtype=np.float32)
    first[:, 2, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    second = np.zeros((3, 4, 4), dtype=np.float32)
    second[:, 2, 2] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    anchor_map = aggregate_raw_vfm_features_from_token_contributions(
        source,
        sampled_source_indices=np.asarray([0], dtype=np.int64),
        contribution_views=[
            _contribution_view("a.png", first, owners),
            _contribution_view("b.png", second, owners),
        ],
        config=RawGaussianFeatureAggregationConfig(
            min_observations=2,
            l2_normalize_observations=True,
            l2_normalize_features=True,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.support_counts.tolist() == [2]
    assert anchor_map.visibility_counts.tolist() == [2]
    assert anchor_map.observation_image_ids == (("a.png", "b.png"),)
    assert anchor_map.feature_variances[0] > 0.0


def test_projection_to_token_grid_honors_simple_radial_distortion() -> None:
    camera = ColmapCamera(camera_id=2, model_id=2, width=100, height=100, params=(10.0, 50.0, 50.0, 0.1))

    uv, depth = _project_xyz_to_grid(
        np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        np.eye(4, dtype=np.float64),
        camera,
        width=10,
        height=10,
    )

    assert np.allclose(depth, np.asarray([1.0], dtype=np.float64))
    assert np.allclose(uv[0], np.asarray([6.1, 5.0], dtype=np.float64))


def test_gaussian_anchor_map_can_be_subset_by_source_rows_in_requested_order() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [2.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        scale=np.asarray([0.02, 0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([100, 101, 102], dtype=np.int64),
    )
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    owners = np.full((4, 4), -1, dtype=np.int64)
    owners[2, 2] = 1
    union_map = aggregate_raw_vfm_features_from_token_contributions(
        source,
        sampled_source_indices=np.asarray([1], dtype=np.int64),
        contribution_views=[_contribution_view("a.png", fmap, owners)],
        config=RawGaussianFeatureAggregationConfig(min_observations=1),
    )

    subset = subset_gaussian_anchor_map_by_source_indices(
        union_map,
        source,
        sampled_source_indices=np.asarray([2, 1, 0], dtype=np.int64),
        metadata={"subset_name": "toy"},
    )

    assert subset.source_gaussian_indices.tolist() == [101]
    assert subset.metadata["subset_name"] == "toy"
    assert subset.metadata["requested_sampled_gaussian_count"] == 3
    assert subset.metadata["missing_after_union_aggregation"] == 2


def test_gaussian_source_geometry_summary_reports_anisotropy_and_normals() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float64),
        opacity=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        scale=np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        gaussian_indices=np.asarray([10, 11, 12], dtype=np.int64),
        scale_xyz=np.asarray([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0]], dtype=np.float32),
        rotation=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        normal=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    )

    stats = _source_geometry_stats(source, np.asarray([0, 2], dtype=np.int64))

    assert stats["count"] == 2
    assert stats["normal_available"] is True
    assert stats["normal_available_fraction"] == 1.0
    assert np.isclose(stats["opacity"]["median"], 0.5)
    assert np.isclose(stats["scale"]["median"], 3.0)
    assert np.isclose(stats["anisotropy_ratio"]["median"], 2.5)


def test_token_contribution_view_assigns_splat_neighbors_not_only_center_owner() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([0.02], dtype=np.float32),
        gaussian_indices=np.asarray([20], dtype=np.int64),
    )
    view = _center_saliency_view()

    contribution = build_gaussian_token_contribution_view(
        source,
        view,
        GaussianTokenContributionConfig(radius_px=1.25, depth_epsilon=0.01, opacity_threshold=0.1),
    )

    assert contribution.top_contributor[2, 2] == 0
    assert contribution.top_contributor[2, 3] == 0
    assert contribution.top_alpha is not None
    assert contribution.top_alpha[2, 3] > 0.0
    assert contribution.alpha_entropy is not None
    assert contribution.alpha_entropy[2, 3] == 0.0


def test_contribution_votes_use_salient_tokens_and_alpha_gates() -> None:
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([4.0, 0.0, 0.0], dtype=np.float32)
    fmap[:, 1, 1] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    top_contributor = np.full((4, 4), -1, dtype=np.int64)
    top_contributor[2, 2] = 1
    top_contributor[1, 1] = 0
    top_alpha = np.zeros((4, 4), dtype=np.float32)
    top_alpha[2, 2] = 0.8
    top_alpha[1, 1] = 0.2

    votes, summary = vote_gaussians_from_token_contribution_views(
        source_gaussian_count=3,
        contribution_views=[
            GaussianTokenContributionView(
                image_id="ref.png",
                feature_map=fmap,
                top_contributor=top_contributor,
                top_alpha=top_alpha,
            )
        ],
        vote_config=VfmGaussianAnchorVoteConfig(top_token_fraction=0.10, saliency_mode="norm"),
        min_contribution_alpha=0.5,
    )

    assert votes.tolist() == [0, 1, 0]
    assert summary["voted_gaussian_count"] == 1


def test_contribution_visibility_center_sampling_uses_projected_feature() -> None:
    source = GaussianVFMSource(
        xyz=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        opacity=np.asarray([0.9], dtype=np.float32),
        scale=np.asarray([0.02], dtype=np.float32),
        gaussian_indices=np.asarray([30], dtype=np.int64),
    )
    fmap = np.zeros((3, 4, 4), dtype=np.float32)
    fmap[:, 2, 2] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    top_contributor = np.full((4, 4), -1, dtype=np.int64)
    top_contributor[2, 2] = 0

    anchor_map = aggregate_raw_vfm_features_from_contribution_visibility(
        source,
        sampled_source_indices=np.asarray([0], dtype=np.int64),
        contribution_views=[
            GaussianTokenContributionView(
                image_id="ref.png",
                feature_map=fmap,
                top_contributor=top_contributor,
                top_alpha=np.ones((4, 4), dtype=np.float32),
            )
        ],
        camera_views=[_center_saliency_view()],
        config=RawGaussianFeatureAggregationConfig(min_observations=1),
    )

    assert anchor_map.source_gaussian_indices.tolist() == [30]
    assert anchor_map.support_counts.tolist() == [1]
    np.testing.assert_allclose(anchor_map.features[0], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
