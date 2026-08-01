from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.train_v6_global_frame_encoder import (
    _soft_projection_nll,
)
from feature_extract.tools.vfm.train_surface_maplet_mapper import (
    _resolve_prototype_images,
    _split_images_by_trajectory,
)
from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _geometry_consistent_frame_mode_combinations,
    _geometry_balanced_chart_subsets,
    _deduplicated_mode_combinations,
    _likelihood_ranked_frame_mode_combinations,
    _retrieval_metrics,
    _select_oracle_regions,
    _support_balanced_pose_prefix,
)
from feature_extract.tools.vfm.evaluate_v6_radio_frame_source import (
    ALIKE_MATCHABILITY_FLOOR,
    _aggregate_stage_c,
    _alike_matchability_map,
    _complete_regional_pose_distribution,
    _diverse_pose_hypotheses,
    _geometry_conditioned_chart_subset,
    _matchability_pyramid,
    _marginal_frame_log_evidence,
    _multimodal_regional_pose_hypotheses,
    _source_and_heldout_chart_ids,
    _serialized_frame_mode_rows,
    _stage_c_pose_pool,
    _stage_c_pose_log_score,
    _validate_structured_refiner_feature_source,
)
from feature_extract.tools.vfm.replay_v6_stage_c import (
    _frame_matches_from_query,
)
from feature_extract.vfm.localization_v6.atlas_pose_alignment import (
    AtlasAlignmentLevel,
    _direct_rotation_consensus_eligible,
    _axis_rotation_hypotheses,
    _translation_consensus_eligible,
)
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.localization_v6.map_entities import (
    MetricSurfaceChartBank,
    RegionChartIndex,
    RetrievalRegionBank,
    build_region_chart_index,
    compose_metric_region_atlas,
    expand_region_posterior_to_charts,
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    FramePoseHypothesis,
    MapletFrameMatch,
    MapletFrameSearchConfig,
    _Candidate,
    _support_hull_and_control_covariance,
    _select_distinct_candidates,
    frame_control_error_px,
    frame_log_likelihood_ratio,
    frame_matches_to_pose_hypotheses,
    ground_truth_chart_frame,
    estimate_chart_volume_residual,
    factorized_pose_distribution_modes,
    pose_hypotheses_for_mode_sets,
    pose_distribution_consensus_modes,
    regional_frame_log_evidence,
    rescale_maplet_frame_match,
    refine_maplet_frame_matches_with_chart_volume,
    refine_maplet_frame_matches_with_local_flow,
    refine_maplet_frame_matches_with_structured_refiner,
)
from feature_extract.vfm.localization_v6.structured_frame_refiner import (
    StructuredFrameRefiner,
    StructuredFrameRefinerConfig,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    QueryMapletGroup,
    _compress_surface_location_modes,
    aggregate_scene_maplet_evidence,
)


@pytest.mark.parametrize(
    "feature_source_kind",
    ["surface_maplet_mapper", "surface_spatial_projection"],
)
def test_structured_refiner_accepts_typed_learned_feature_transforms(
    feature_source_kind,
):
    _validate_structured_refiner_feature_source(feature_source_kind)


def test_structured_refiner_rejects_untyped_raw_projection():
    with pytest.raises(ValueError, match="learned surface feature transform"):
        _validate_structured_refiner_feature_source("raw_radio_pca")
from feature_extract.vfm.query_to_3d_matching import pnp_pose_error


def _retrieval_bank() -> SurfaceRetrievalMapletBank:
    return SurfaceRetrievalMapletBank(
        maplet_ids=np.asarray([10, 11]),
        centers=np.asarray([[0, 0, 5], [0.3, 0, 5]], dtype=np.float32),
        normals=np.asarray([[0, 0, 1], [0, 0, 1]], dtype=np.float32),
        extents=np.asarray([[0.3, 0.3, 0.02]] * 2, dtype=np.float32),
        tangent_frames=np.asarray([np.eye(3), np.eye(3)], dtype=np.float32),
        descriptor_offsets=np.asarray([0, 1, 2]),
        descriptors=np.eye(2, dtype=np.float32),
        descriptor_weights=np.ones(2, dtype=np.float32),
        quality_scores=np.ones(2, dtype=np.float32),
        descriptor_uncertainties=np.zeros(2, dtype=np.float32),
        metadata={
            "vfm_layer": "radio_final",
            "has_canonical_tangent_frames": True,
        },
    )


def _atlas() -> MapletFeatureAtlasBank:
    size = 8
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing="ij",
    )
    xyz = np.stack([xx * 0.3, yy * 0.3, np.full_like(xx, 5)], axis=2)
    xyz = np.stack([xyz, xyz + np.asarray([0.3, 0, 0])], axis=0)
    feature = np.stack(
        [np.cos(xx * 2.0), np.sin(yy * 2.0), np.ones_like(xx)], axis=0
    )
    features = np.stack([feature, feature], axis=0).astype(np.float32)
    valid = np.ones((2, size, size), dtype=bool)
    return MapletFeatureAtlasBank(
        maplet_ids=np.asarray([10, 11]),
        centers=np.asarray([[0, 0, 5], [0.3, 0, 5]], dtype=np.float32),
        frames=np.asarray([np.eye(3), np.eye(3)], dtype=np.float32),
        extents=np.asarray([[0.3, 0.3, 0.02]] * 2, dtype=np.float32),
        xyz=xyz.astype(np.float32),
        primitive_ids=np.arange(2 * size * size).reshape(2, size, size),
        features=features,
        variance=np.zeros((2, size, size), dtype=np.float32),
        support_count=np.ones((2, size, size), dtype=np.int32),
        valid_mask=valid,
        metadata={
            "artifact_type": "v6_canonical_maplet_feature_atlas",
            "stores_mapping_rgb": False,
        },
    )


def test_surface_mapper_trajectory_split_is_strictly_disjoint():
    images = np.asarray(
        [
            "seq1/frame1.png",
            "seq9/frame1.png",
            "seq11/frame1.png",
            "seq3/frame1.png",
        ],
        dtype=object,
    )
    training, validation = _split_images_by_trajectory(
        images,
        ("seq1", "seq9"),
        ("seq11",),
        ("seq3", "seq5", "seq13"),
    )
    assert training == {"seq1/frame1.png", "seq9/frame1.png"}
    assert validation == {"seq11/frame1.png"}


def test_surface_mapper_validation_prototypes_match_deployed_map():
    training = {
        "seq1/frame1.png",
        "seq2/frame1.png",
        "seq9/frame1.png",
        "seq10/frame1.png",
    }
    validation = {"seq11/frame1.png"}
    prototypes = _resolve_prototype_images(
        training,
        validation,
        ("seq1", "seq2"),
        ("seq3", "seq5", "seq13"),
        trajectory_split=True,
    )
    assert prototypes == {"seq1/frame1.png", "seq2/frame1.png"}


def test_surface_mapper_rejects_nontraining_prototype_trajectory():
    with pytest.raises(ValueError, match="subset"):
        _resolve_prototype_images(
            {"seq1/frame1.png", "seq9/frame1.png"},
            {"seq11/frame1.png"},
            ("seq1", "seq11"),
            ("seq3", "seq5", "seq13"),
            trajectory_split=True,
        )


def test_atlas_alignment_exposes_six_rotation_directions():
    level = AtlasAlignmentLevel(
        name="coarse",
        feature_stride=16,
        correlation_radius=4,
        maximum_translation_step_m=0.08,
        maximum_rotation_step_deg=3.0,
        direct_rotation_step_deg=2.5,
    )
    updates = _axis_rotation_hypotheses(np.eye(4), level)
    assert len(updates) == 6
    deltas = np.stack([update.delta for update in updates])
    assert np.allclose(deltas[:, 3:], 0.0)
    expected = np.deg2rad(2.5)
    assert set(map(tuple, np.round(deltas[:, :3], 12))) == {
        (-round(expected, 12), 0.0, 0.0),
        (round(expected, 12), 0.0, 0.0),
        (0.0, -round(expected, 12), 0.0),
        (0.0, round(expected, 12), 0.0),
        (0.0, 0.0, -round(expected, 12)),
        (0.0, 0.0, round(expected, 12)),
    }


def test_direct_rotation_requires_fixed_surface_chart_consensus():
    common = dict(
        fit_gain=0.2,
        fixed_chart_gain=0.4,
        dynamic_all_chart_gain=0.2,
        heldout_gain=0.1,
        fixed_chart_count=10,
        total_chart_count=12,
        minimum_gain=0.1,
    )
    assert _direct_rotation_consensus_eligible(
        **common, positive_chart_fraction=0.4
    )
    assert not _direct_rotation_consensus_eligible(
        **common, positive_chart_fraction=0.39
    )
    assert not _direct_rotation_consensus_eligible(
        **{**common, "dynamic_all_chart_gain": -0.01},
        positive_chart_fraction=0.8,
    )
    assert not _direct_rotation_consensus_eligible(
        **{**common, "fit_gain": 0.0}, positive_chart_fraction=0.8
    )


def test_metric_translation_requires_disjoint_flow_and_chart_consensus():
    common = dict(
        is_analytic=True,
        heldout_update_consistent=True,
        fixed_chart_gain=0.2,
        dynamic_all_chart_gain=0.1,
        positive_chart_fraction=0.6,
        fixed_chart_count=10,
        total_chart_count=12,
    )
    assert _translation_consensus_eligible(**common)
    assert not _translation_consensus_eligible(
        **{**common, "heldout_update_consistent": False}
    )
    assert not _translation_consensus_eligible(
        **{**common, "dynamic_all_chart_gain": -0.01}
    )


def test_collinear_chart_support_uses_conservative_unknown_hull():
    canonical = np.asarray(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float32
    )
    query = canonical + np.asarray([4.0, 3.0], dtype=np.float32)
    hull, covariance, fraction = _support_hull_and_control_covariance(
        canonical,
        query,
        (16, 16),
        np.eye(2, dtype=np.float32),
        feature_stride=4,
    )
    assert hull is None
    assert covariance.shape == (8, 8)
    assert fraction == 1.0


def test_retrieval_region_rejects_post_pooling_mapper_baseline():
    bank = _retrieval_bank()
    with pytest.raises(ValueError, match="nonlinear mapper"):
        RetrievalRegionBank(
            replace(
                bank,
                metadata={
                    **dict(bank.metadata or {}),
                    "descriptor_construction": (
                        "legacy_pooled_descriptor_mapper_baseline"
                    ),
                },
            )
        )


def test_nearby_surface_modes_merge_mass_and_covariance():
    centers, covariance, probability, null = (
        _compress_surface_location_modes(
            np.asarray([0.4, 0.6]),
            np.asarray([[0.0, 0.0, 4.0], [0.01, 0.0, 4.0]]),
            np.tile(np.eye(3)[None] * 1e-4, (2, 1, 1)),
            maximum_modes=1,
            nms_distance_m=0.02,
        )
    )
    assert probability.tolist() == [1.0]
    assert null == 0.0
    assert np.isclose(centers[0, 0], 0.006)
    assert covariance[0, 0, 0] > 1e-4


def test_scene_evidence_nms_removes_duplicate_overlap_votes():
    groups = (
        QueryMapletGroup(
            query_region_xy=np.asarray([10.0, 10.0]),
            query_region_extent=np.asarray([5.0, 5.0]),
            maplet_ids=np.asarray([1, 2]),
            probabilities=np.asarray([0.6, 0.4]),
            null_probability=0.0,
            omitted_probability=0.0,
        ),
        QueryMapletGroup(
            query_region_xy=np.asarray([11.0, 10.0]),
            query_region_extent=np.asarray([5.0, 5.0]),
            maplet_ids=np.asarray([1]),
            probabilities=np.asarray([0.6]),
            null_probability=0.0,
            omitted_probability=0.0,
        ),
        QueryMapletGroup(
            query_region_xy=np.asarray([30.0, 10.0]),
            query_region_extent=np.asarray([5.0, 5.0]),
            maplet_ids=np.asarray([2]),
            probabilities=np.asarray([0.5]),
            null_probability=0.0,
            omitted_probability=0.0,
        ),
    )
    ids = np.asarray([1, 2])
    legacy = aggregate_scene_maplet_evidence(
        groups, ids, method="legacy_sum"
    )
    deduplicated = aggregate_scene_maplet_evidence(
        groups, ids, method="topq_nms"
    )
    assert legacy[0] > legacy[1]
    assert deduplicated[0] < deduplicated[1]
    assert np.array_equal(
        aggregate_scene_maplet_evidence(
            (), ids, method="block_balanced"
        ),
        np.zeros(2),
    )


def test_region_chart_index_roundtrip_and_composite_atlas(tmp_path: Path):
    regions = RetrievalRegionBank(_retrieval_bank())
    charts = MetricSurfaceChartBank(_atlas())
    index = build_region_chart_index(
        regions,
        charts,
        region_extent_multiplier=3.0,
        maximum_charts_per_region=2,
    )
    assert index.edge_count >= 2
    assert 10 in index.regions_for_charts(np.asarray([10]))
    path = tmp_path / "region-chart.npz"
    index.save_npz(path)
    restored = RegionChartIndex.load_npz(path)
    assert np.array_equal(restored.region_chart_ids, index.region_chart_ids)
    composite = compose_metric_region_atlas(
        regions, charts, restored, 10, resolution=16
    )
    assert len(composite) == 1
    assert composite.maplet_ids.tolist() == [10]
    assert np.any(composite.valid_mask)
    assert not bool(composite.metadata["stores_mapping_rgb"])


def test_identity_initialized_structured_refiner_preserves_frame_geometry():
    atlas = _atlas()
    match = MapletFrameMatch(
        chart_id=10,
        canonical_to_query=np.asarray(
            [[3.0, 0.0, 10.0], [0.0, 3.0, 10.0]], dtype=np.float32
        ),
        query_center_xy=np.asarray([10.0, 10.0], dtype=np.float32),
        scale_xy=np.asarray([3.0, 3.0], dtype=np.float32),
        in_plane_rotation_deg=0.0,
        covariance_xy=np.eye(2, dtype=np.float32),
        score=0.4,
        probability=0.9,
        null_probability=0.1,
        support_fraction=1.0,
        feature_level="coarse",
        feature_stride=16,
    )
    model = StructuredFrameRefiner(
        StructuredFrameRefinerConfig(
            feature_dim=atlas.feature_dim,
            correlation_radius=1,
            hidden_dim=16,
            transformer_layers=1,
            attention_heads=4,
        )
    )
    refined = refine_maplet_frame_matches_with_structured_refiner(
        atlas,
        (match,),
        torch.randn(atlas.feature_dim, 24, 24),
        model,
        maximum_cells=32,
    )
    assert len(refined) == 1
    controls = np.asarray(
        [[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32
    )
    assert np.allclose(
        refined[0].canonical_points_to_query(controls),
        match.canonical_points_to_query(controls),
        atol=1e-4,
    )


def test_region_chart_expansion_normalizes_each_region_probability_mass():
    regions = RetrievalRegionBank(_retrieval_bank())
    charts = MetricSurfaceChartBank(_atlas())
    index = RegionChartIndex(
        region_ids=np.asarray([10, 11]),
        region_offsets=np.asarray([0, 2, 3]),
        region_chart_ids=np.asarray([10, 11, 10]),
        chart_ids=np.asarray([10, 11]),
        chart_offsets=np.asarray([0, 2, 3]),
        chart_region_ids=np.asarray([10, 11, 10]),
    )
    posterior = expand_region_posterior_to_charts(
        np.asarray([10, 11]),
        np.asarray([1.0, 1.0]),
        regions,
        charts,
        index,
        minimum_charts=1,
        maximum_charts=2,
        cumulative_probability=0.95,
    )
    assert np.isclose(np.sum(posterior.region_probability), 1.0)
    assert np.isclose(np.sum(posterior.chart_probability), 1.0)
    assert posterior.selected_chart_ids.size == 2
    assert np.isclose(posterior.selected_probability_mass, 1.0)
    # Region 11 has one outgoing edge and contributes exactly its 0.5 mass;
    # region 10 divides its 0.5 mass across two charts.
    assert posterior.probability_for(10) > 0.5
    assert 0.0 < posterior.probability_for(11) < 0.5


def test_composite_region_preserves_bounded_appearance_modes():
    atlas = _atlas()
    count, channels, height, width = atlas.features.shape
    mode_features = np.stack(
        [atlas.features, -atlas.features], axis=1
    )
    mode_weights = np.full(
        (count, 2, height, width), 0.5, dtype=np.float32
    )
    mode_valid = np.repeat(
        atlas.valid_mask[:, None], 2, axis=1
    )
    directions = np.zeros(
        (count, 2, height, width, 3), dtype=np.float32
    )
    directions[:, 0, ..., 2] = 1.0
    directions[:, 1, ..., 2] = -1.0
    mode_atlas = replace(
        atlas,
        mode_features=mode_features.reshape(
            count, 2, channels, height, width
        ),
        mode_weights=mode_weights,
        mode_view_directions=directions,
        mode_view_covariance=np.zeros(
            (count, 2, height, width, 3, 3), dtype=np.float32
        ),
        mode_variance=np.zeros(
            (count, 2, height, width), dtype=np.float32
        ),
        mode_valid_mask=mode_valid,
    )
    regions = RetrievalRegionBank(_retrieval_bank())
    charts = MetricSurfaceChartBank(mode_atlas)
    index = build_region_chart_index(
        regions,
        charts,
        region_extent_multiplier=3.0,
        maximum_charts_per_region=2,
    )
    composite = compose_metric_region_atlas(
        regions, charts, index, 10, resolution=16
    )
    assert composite.appearance_mode_count == 2
    valid = composite.mode_valid_mask[0]
    assert np.any(valid)
    assert np.allclose(
        np.sum(composite.mode_weights[0], axis=0)[
            np.any(valid, axis=0)
        ],
        1.0,
    )


def test_region_diagnostics_follow_index_not_accidental_equal_ids():
    index = RegionChartIndex(
        region_ids=np.asarray([100, 200]),
        region_offsets=np.asarray([0, 1, 3]),
        region_chart_ids=np.asarray([10, 10, 11]),
        chart_ids=np.asarray([10, 11]),
        chart_offsets=np.asarray([0, 2, 3]),
        chart_region_ids=np.asarray([100, 200, 200]),
    )
    visible_ids = np.asarray([10, 11])
    visible_counts = np.asarray([10, 8])
    assert _select_oracle_regions(
        visible_ids, visible_counts, index, 1
    ).tolist() == [200]
    retrieval = MapletRetrievalResult(
        groups=(),
        ranked_maplet_ids=np.asarray([200]),
        evidence=np.asarray([1.0], dtype=np.float32),
        scene_evidence_aggregation="topq_nms",
    )
    metrics = _retrieval_metrics(
        retrieval, visible_ids, visible_counts, index
    )
    assert metrics["dominant_region_recall_at_1"]
    assert metrics["visible_surface_coverage_at_1"] == 1.0


def test_homography_frame_control_recovers_planar_camera_pose():
    atlas = _atlas()
    camera = ColmapCamera(
        camera_id=0,
        model_id=0,
        width=640,
        height=480,
        params=(500.0, 320.0, 240.0),
    )
    pose = np.eye(4, dtype=np.float64)
    match = ground_truth_chart_frame(
        atlas,
        10,
        pose,
        camera,
        feature_stride=4,
        model="homography",
    )
    assert match is not None
    hypotheses = frame_matches_to_pose_hypotheses(
        atlas, [match], camera
    )
    errors = [pnp_pose_error(value.pose_w2c, pose) for value in hypotheses]
    assert any(
        error.translation_m < 1e-3 and error.rotation_deg < 1e-3
        for error in errors
    )


def test_grouped_mode_sets_do_not_duplicate_individual_chart_poses():
    atlas = _atlas()
    camera = ColmapCamera(
        camera_id=0,
        model_id=0,
        width=640,
        height=480,
        params=(500.0, 320.0, 240.0),
    )
    pose = np.eye(4, dtype=np.float64)
    matches = [
        ground_truth_chart_frame(
            atlas,
            chart_id,
            pose,
            camera,
            feature_stride=4,
            model="affine",
        )
        for chart_id in (10, 11)
    ]
    assert all(match is not None for match in matches)
    hypotheses = pose_hypotheses_for_mode_sets(
        atlas, [matches], camera
    )
    assert hypotheses
    assert all(
        len(hypothesis.source_chart_ids) == 2
        for hypothesis in hypotheses
    )
    assert all(np.isfinite(hypothesis.score) for hypothesis in hypotheses)
    assert all(
        hypothesis.control_model == "chart_factor"
        and hypothesis.seed_model == "EPNP"
        for hypothesis in hypotheses
    )
    assert all(
        hypothesis.score
        <= sum(frame_log_likelihood_ratio(match) for match in matches)
        for hypothesis in hypotheses
    )

    broad_hypotheses = pose_hypotheses_for_mode_sets(
        atlas, [matches], camera, refine_grouped_pose=False
    )
    assert len(broad_hypotheses) == 1
    assert broad_hypotheses[0].control_model == "regional_chart_seed"
    assert broad_hypotheses[0].seed_model == "EPNP_UNREFINED"


def test_three_correlated_chart_factors_recover_pose_without_point_anchors():
    source = _atlas()
    offset = np.asarray([0.0, 0.3, 0.0], dtype=np.float32)
    atlas = MapletFeatureAtlasBank(
        maplet_ids=np.asarray([10, 11, 12]),
        centers=np.concatenate(
            [source.centers, source.centers[:1] + offset[None]]
        ),
        frames=np.concatenate([source.frames, source.frames[:1]]),
        extents=np.concatenate([source.extents, source.extents[:1]]),
        xyz=np.concatenate(
            [source.xyz, source.xyz[:1] + offset[None, None, None]]
        ),
        primitive_ids=np.concatenate(
            [source.primitive_ids, source.primitive_ids[:1]]
        ),
        features=np.concatenate(
            [source.features, source.features[:1]]
        ),
        variance=np.concatenate(
            [source.variance, source.variance[:1]]
        ),
        support_count=np.concatenate(
            [source.support_count, source.support_count[:1]]
        ),
        valid_mask=np.concatenate(
            [source.valid_mask, source.valid_mask[:1]]
        ),
        metadata=source.metadata,
    )
    camera = ColmapCamera(
        camera_id=0,
        model_id=0,
        width=640,
        height=480,
        params=(500.0, 320.0, 240.0),
    )
    pose = np.eye(4, dtype=np.float64)
    matches = [
        ground_truth_chart_frame(
            atlas,
            chart_id,
            pose,
            camera,
            feature_stride=4,
            model="affine",
        )
        for chart_id in (10, 11, 12)
    ]
    hypotheses = pose_hypotheses_for_mode_sets(
        atlas, [matches], camera
    )
    factor_hypotheses = [
        value
        for value in hypotheses
        if value.control_model == "chart_factor"
    ]
    assert factor_hypotheses
    assert any(
        value.seed_model == "SQPNP_CHART_CENTERS"
        for value in factor_hypotheses
    )
    assert not any(
        value.control_model == "diagnostic_chart_centers"
        for value in hypotheses
    )
    errors = [
        pnp_pose_error(value.pose_w2c, pose)
        for value in factor_hypotheses
    ]
    assert any(
        error.translation_m < 1e-3 and error.rotation_deg < 1e-3
        for error in errors
    )


def test_frame_nms_preserves_distinct_affines_at_one_center():
    assert MapletFrameSearchConfig().support_score_power == 0.0

    def candidate(linear, center=(10.0, 10.0), score=1.0):
        return _Candidate(
            score=float(score),
            center_xy=np.asarray(center, dtype=np.float32),
            covariance_xy=np.eye(2, dtype=np.float32),
            width=4.0,
            height=4.0,
            angle_deg=0.0,
            linear=np.asarray(linear, dtype=np.float32),
            support_fraction=1.0,
        )

    identity = candidate(np.eye(2) * 4.0, score=3.0)
    near_duplicate = candidate(np.eye(2) * 4.1, score=2.9)
    rotated = candidate([[0.0, -4.0], [4.0, 0.0]], score=2.8)
    remote = candidate(np.eye(2) * 4.0, center=(20.0, 10.0), score=2.7)
    selected = _select_distinct_candidates(
        [identity, near_duplicate, rotated, remote],
        MapletFrameSearchConfig(
            maximum_modes=4,
            spatial_mode_nms_radius_cells=2.0,
            affine_mode_nms_radius_cells=0.5,
            maximum_modes_per_spatial_cluster=2,
        ),
    )
    assert len(selected) == 3
    assert selected[0] is identity
    assert selected[1] is rotated
    assert selected[2] is remote


def test_global_frame_loss_uses_full_query_hard_negatives():
    query = torch.zeros(2, 3, 3)
    query[:, 1, 2] = torch.tensor([1.0, 0.0])
    query[:, 0, 0] = torch.tensor([0.0, 1.0])
    map_feature = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss, metrics = _soft_projection_nll(
        query,
        map_feature,
        np.asarray([[2.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        (3, 3),
        0.07,
    )
    assert torch.isfinite(loss)
    assert metrics["recall_at_1_radius1"] == 1.0
    assert metrics["positive_wrong_margin"] > 0.9


def test_local_flow_refines_one_correlated_chart_homography():
    atlas = _atlas()
    query = torch.zeros((atlas.feature_dim, 24, 24), dtype=torch.float32)
    query[:, 6:14, 5:13] = torch.from_numpy(atlas.features[0])
    target = MapletFrameMatch(
        chart_id=10,
        canonical_to_query=np.asarray(
            [[3.5, 0.0, 8.5], [0.0, 3.5, 9.5]], dtype=np.float32
        ),
        query_center_xy=np.asarray([8.5, 9.5], dtype=np.float32),
        scale_xy=np.asarray([3.5, 3.5], dtype=np.float32),
        in_plane_rotation_deg=0.0,
        covariance_xy=np.eye(2, dtype=np.float32),
        score=1.0,
        probability=0.9,
        null_probability=0.1,
        support_fraction=1.0,
        feature_level="coarse",
        feature_stride=16,
    )
    initial = replace(
        target,
        canonical_to_query=np.asarray(
            [[3.5, 0.0, 10.5], [0.0, 3.5, 11.5]], dtype=np.float32
        ),
        query_center_xy=np.asarray([10.5, 11.5], dtype=np.float32),
    )
    refined = refine_maplet_frame_matches_with_local_flow(
        atlas,
        [initial],
        query,
        rounds=2,
        radius_cells=3,
        minimum_confidence=0.0,
        temperature=0.04,
        background_similarity=-0.25,
    )
    assert refined
    assert frame_control_error_px(refined[0], target) < (
        frame_control_error_px(initial, target)
    )


def test_chart_volume_refines_one_correlated_regional_transform():
    atlas = _atlas()
    query = torch.zeros((atlas.feature_dim, 24, 24), dtype=torch.float32)
    query[:, 6:14, 5:13] = torch.from_numpy(atlas.features[0])
    target = MapletFrameMatch(
        chart_id=10,
        canonical_to_query=np.asarray(
            [[3.5, 0.0, 8.5], [0.0, 3.5, 9.5]], dtype=np.float32
        ),
        query_center_xy=np.asarray([8.5, 9.5], dtype=np.float32),
        scale_xy=np.asarray([3.5, 3.5], dtype=np.float32),
        in_plane_rotation_deg=0.0,
        covariance_xy=np.eye(2, dtype=np.float32),
        score=1.0,
        probability=0.9,
        null_probability=0.1,
        support_fraction=1.0,
        feature_level="coarse",
        feature_stride=16,
    )
    initial = replace(
        target,
        canonical_to_query=np.asarray(
            [[3.5, 0.0, 10.5], [0.0, 3.5, 11.5]], dtype=np.float32
        ),
        query_center_xy=np.asarray([10.5, 11.5], dtype=np.float32),
    )
    refined = refine_maplet_frame_matches_with_chart_volume(
        atlas,
        [initial],
        query,
        radius_cells=3,
        maximum_cells=64,
        random_proposals=64,
        retained_proposals=6,
        iterations=30,
        minimum_evidence_improvement=-1e-6,
    )
    assert refined
    assert frame_control_error_px(refined[0], target) < (
        frame_control_error_px(initial, target)
    )


def test_chart_volume_detector_score_cannot_create_offset_evidence():
    correlation = torch.zeros((4, 18), dtype=torch.float32)
    canonical = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    matchability = torch.full((4, 9), 0.2)
    # radius=1 has row-major offsets; index 5 is (dx=+1, dy=0).
    matchability[:, 5] = 1.0
    linear, translation, _score, _zero = estimate_chart_volume_residual(
        correlation,
        canonical,
        candidate_matchability=matchability,
        radius_cells=1,
        topk_per_cell=3,
        random_proposals=0,
        retained_proposals=8,
        iterations=0,
    )
    assert np.allclose(linear, 0.0)
    # ALIKE may weight distinctive cells, but a detector peak is not RADIO
    # match evidence and therefore cannot create a displacement on its own,
    # even though it seeds both weak and strong proposal families.
    assert np.allclose(translation, [0.0, 0.0])


def test_detector_guided_branch_does_not_create_radio_evidence():
    correlation = torch.zeros((4, 18), dtype=torch.float32)
    canonical = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    matchability = torch.full((4, 9), 0.2)
    matchability[:, 5] = 1.0
    _linear, _translation, evidence, zero_evidence = (
        estimate_chart_volume_residual(
            correlation,
            canonical,
            candidate_matchability=matchability,
            radius_cells=1,
            topk_per_cell=3,
            random_proposals=0,
            retained_proposals=8,
            iterations=0,
            proposal_selection="detector_guided",
        )
    )
    # This explicit branch may emit a detector-seeded pose proposal, but a
    # flat RADIO volume assigns it no advantage over zero displacement.
    assert evidence == pytest.approx(zero_evidence, abs=1e-7)


def test_alike_detection_prior_is_soft_and_resamples_without_descriptors():
    matchability = _alike_matchability_map(
        np.asarray([[20.0, 10.0]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        image_width=40,
        image_height=20,
        feature_width=10,
        feature_height=5,
        feature_stride=4,
    )
    assert matchability.shape == (5, 10)
    assert float(matchability.min()) >= float(ALIKE_MATCHABILITY_FLOOR)
    assert float(matchability.max()) <= 1.0
    pyramid = _matchability_pyramid(
        matchability,
        {
            "middle": np.zeros((8, 5, 10), dtype=np.float32),
            "fine": np.zeros((8, 10, 20), dtype=np.float32),
        },
    )
    assert pyramid["middle"].shape == (5, 10)
    assert pyramid["fine"].shape == (10, 20)
    assert float(np.min(pyramid["fine"])) >= float(
        ALIKE_MATCHABILITY_FLOOR
    )


def test_noop_chart_refinement_preserves_input_null_odds():
    atlas = _atlas()
    query = torch.zeros((atlas.feature_dim, 24, 24), dtype=torch.float32)
    initial = MapletFrameMatch(
        chart_id=10,
        canonical_to_query=np.asarray(
            [[3.5, 0.0, 8.5], [0.0, 3.5, 9.5]], dtype=np.float32
        ),
        query_center_xy=np.asarray([8.5, 9.5], dtype=np.float32),
        scale_xy=np.asarray([3.5, 3.5], dtype=np.float32),
        in_plane_rotation_deg=0.0,
        covariance_xy=np.eye(2, dtype=np.float32),
        score=0.50,
        probability=0.20,
        null_probability=0.80,
        support_fraction=1.0,
        feature_level="coarse",
        feature_stride=16,
    )
    # The atlas has only 64 valid cells, so this deliberately takes the
    # no-refinement branch.  Re-normalization must not reset null to score 0.
    refined = refine_maplet_frame_matches_with_chart_volume(
        atlas,
        [initial],
        query,
        maximum_cells=65,
        minimum_observations=65,
    )
    assert len(refined) == 1
    assert np.isclose(refined[0].probability, 0.20)
    assert np.isclose(refined[0].null_probability, 0.80)


def test_geometry_balanced_chart_subsets_cover_beyond_score_prefix():
    source = _atlas()
    count = 12
    offsets = np.arange(count, dtype=np.float32)[:, None] * np.asarray(
        [[0.3, 0.0, 0.0]], dtype=np.float32
    )
    atlas = MapletFeatureAtlasBank(
        maplet_ids=np.arange(100, 100 + count, dtype=np.int64),
        centers=source.centers[:1] + offsets,
        frames=np.repeat(source.frames[:1], count, axis=0),
        extents=np.repeat(source.extents[:1], count, axis=0),
        xyz=np.repeat(source.xyz[:1], count, axis=0)
        + offsets[:, None, None],
        primitive_ids=np.arange(
            count * source.height * source.width, dtype=np.int64
        ).reshape(count, source.height, source.width),
        features=np.repeat(source.features[:1], count, axis=0),
        variance=np.repeat(source.variance[:1], count, axis=0),
        support_count=np.repeat(source.support_count[:1], count, axis=0),
        valid_mask=np.repeat(source.valid_mask[:1], count, axis=0),
        metadata=source.metadata,
    )
    ordered = []
    for rank, chart_id in enumerate(atlas.maplet_ids.tolist()):
        match = MapletFrameMatch(
            chart_id=chart_id,
            canonical_to_query=np.asarray(
                [[2.0, 0.0, rank + 2.0], [0.0, 2.0, 4.0]],
                dtype=np.float32,
            ),
            query_center_xy=np.asarray(
                [rank + 2.0, 4.0], dtype=np.float32
            ),
            scale_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            in_plane_rotation_deg=0.0,
            covariance_xy=np.eye(2, dtype=np.float32),
            score=1.0 - 0.01 * rank,
            probability=0.5,
            null_probability=0.5,
            support_fraction=1.0,
            feature_level="coarse",
            feature_stride=16,
        )
        ordered.append((chart_id, (match,)))
    subsets = _geometry_balanced_chart_subsets(
        ordered,
        atlas,
        count=2,
        maximum_subsets=count,
    )
    covered = {index for subset in subsets for index in subset}
    assert covered == set(range(count))
    assert any(10 in subset for subset in subsets)
    exhaustive = _geometry_balanced_chart_subsets(
        ordered,
        atlas,
        count=2,
        maximum_subsets=count * (count - 1) // 2,
    )
    assert len(exhaustive) == count * (count - 1) // 2
    assert set(exhaustive) == {
        (first, second)
        for first in range(count)
        for second in range(first + 1, count)
    }


def test_pose_prefix_covers_support_sets_before_seed_duplicates():
    repeated = [
        SimpleNamespace(source_chart_ids=(10, 11), score=100.0 - rank)
        for rank in range(20)
    ]
    deep = SimpleNamespace(source_chart_ids=(20, 21), score=-100.0)
    selected = _support_balanced_pose_prefix([*repeated, deep], 2)
    assert selected == [repeated[0], deep]


def test_frame_mode_beam_keeps_lower_rank_geometric_consensus():
    def match(chart_id, probability):
        return MapletFrameMatch(
            chart_id=chart_id,
            canonical_to_query=np.asarray(
                [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0]],
                dtype=np.float32,
            ),
            query_center_xy=np.asarray([4.0, 3.0], dtype=np.float32),
            scale_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            in_plane_rotation_deg=0.0,
            covariance_xy=np.eye(2, dtype=np.float32),
            score=1.0,
            probability=probability,
            null_probability=0.1,
            support_fraction=1.0,
            feature_level="coarse",
            feature_stride=16,
        )

    first_top, first_consistent = match(10, 0.9), match(10, 0.1)
    second_top, second_consistent = match(11, 0.9), match(11, 0.1)
    rotation = np.eye(3, dtype=np.float64)
    signatures = {
        id(first_top): ((np.asarray([0.0, 0.0, 0.0]), rotation),),
        id(first_consistent): (
            (np.asarray([5.0, 0.0, 0.0]), rotation),
        ),
        id(second_top): ((np.asarray([10.0, 0.0, 0.0]), rotation),),
        id(second_consistent): (
            (np.asarray([5.1, 0.0, 0.0]), rotation),
        ),
    }
    selected = _geometry_consistent_frame_mode_combinations(
        (
            (first_top, first_consistent),
            (second_top, second_consistent),
        ),
        signatures,
        maximum_combinations=1,
    )
    assert len(selected) == 1
    assert selected[0][0] is first_consistent
    assert selected[0][1] is second_consistent


def test_pair_mode_union_preserves_likelihood_branch_geometry_rejects():
    def match(chart_id, probability):
        return MapletFrameMatch(
            chart_id=chart_id,
            canonical_to_query=np.asarray(
                [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0]], dtype=np.float32
            ),
            query_center_xy=np.asarray([4.0, 3.0], dtype=np.float32),
            scale_xy=np.asarray([2.0, 2.0], dtype=np.float32),
            in_plane_rotation_deg=0.0,
            covariance_xy=np.eye(2, dtype=np.float32),
            score=1.0,
            probability=probability,
            null_probability=0.1,
            support_fraction=1.0,
            feature_level="coarse",
            feature_stride=16,
        )

    first = (match(10, 0.60), match(10, 0.40))
    second = (match(11, 0.60), match(11, 0.40))
    geometry_only = ((first[0], second[0]),)
    likelihood = _likelihood_ranked_frame_mode_combinations(
        (first, second), maximum_combinations=4
    )
    selected = _deduplicated_mode_combinations(
        geometry_only,
        likelihood,
        maximum_combinations=4,
    )
    assert len(selected) == 4
    assert any(
        values[0] is first[1] and values[1] is second[1]
        for values in selected
    )
    assert sum(
        values[0] is first[0] and values[1] is second[0]
        for values in selected
    ) == 1


def test_stage_c_pool_covers_chart_support_before_score_duplicates():
    hypotheses = []
    for rank in range(20):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(rank)
        hypotheses.append(
            SimpleNamespace(
                pose_w2c=pose,
                source_chart_ids=(10, 11),
            )
        )
    deep_pose = np.eye(4, dtype=np.float64)
    deep_pose[0, 3] = 0.5
    deep_support = SimpleNamespace(
        pose_w2c=deep_pose,
        source_chart_ids=(20, 21, 22),
    )
    hypotheses.append(deep_support)
    selected = _diverse_pose_hypotheses(hypotheses, 4)
    assert any(value is deep_support for value in selected)
    regional_first = _diverse_pose_hypotheses(
        hypotheses, 4, minimum_source_charts=3
    )
    assert regional_first[0] is deep_support


def test_stage_c_pool_treats_two_charts_as_regional_seed():
    single = SimpleNamespace(
        pose_w2c=np.eye(4, dtype=np.float64),
        source_chart_ids=(10,),
        score=100.0,
        reprojection_error_px=0.1,
    )
    pair_pose = np.eye(4, dtype=np.float64)
    pair_pose[0, 3] = 2.0
    pair = SimpleNamespace(
        pose_w2c=pair_pose,
        source_chart_ids=(20, 21),
        score=-100.0,
        reprojection_error_px=10.0,
    )
    selected = _stage_c_pose_pool((single, pair), (), (), 1)
    assert selected == [pair]


def test_sparse_stage_c_screen_receives_complete_regional_distribution():
    single = SimpleNamespace(source_chart_ids=(10,))
    regional = [
        SimpleNamespace(source_chart_ids=(20 + rank, 40 + rank))
        for rank in range(300)
    ]
    consensus = SimpleNamespace(source_chart_ids=(50, 51, 52))
    factorized = SimpleNamespace(source_chart_ids=(60, 61))
    selected = _complete_regional_pose_distribution(
        [single, *regional], [consensus], [factorized]
    )
    assert selected[:2] == [consensus, factorized]
    assert selected[2:] == regional
    assert single not in selected


def test_serialized_frame_distribution_replays_without_image_state():
    match = MapletFrameMatch(
        chart_id=17,
        canonical_to_query=np.asarray(
            [[2.0, 0.2, 4.0], [0.1, 3.0, 5.0]], dtype=np.float32
        ),
        query_center_xy=np.asarray([4.0, 5.0], dtype=np.float32),
        scale_xy=np.asarray([2.0, 3.0], dtype=np.float32),
        in_plane_rotation_deg=3.0,
        covariance_xy=np.eye(2, dtype=np.float32),
        score=0.7,
        probability=0.6,
        null_probability=0.2,
        support_fraction=0.8,
        feature_level="coarse",
        feature_stride=16,
        identity_probability=0.9,
    )
    rows = _serialized_frame_mode_rows({17: (match,)})
    restored = _frame_matches_from_query(
        {"m3_retrieved_chart_frame_modes": rows}
    )
    assert tuple(restored) == (17,)
    assert len(restored[17]) == 1
    assert np.allclose(
        restored[17][0].canonical_to_query,
        match.canonical_to_query,
    )
    assert restored[17][0].identity_probability == pytest.approx(0.9)


def test_stage_c_raw_family_keeps_deep_mode_of_top_pair_support():
    hypotheses = []
    target = None
    for cardinality in (2, 3, 4):
        for support_rank in range(4):
            support = tuple(
                cardinality * 100
                + support_rank * 10
                + offset
                for offset in range(cardinality)
            )
            for mode_rank in range(8):
                value = SimpleNamespace(
                    source_chart_ids=support,
                    score=float(100 - support_rank - 0.01 * mode_rank),
                    reprojection_error_px=float(mode_rank),
                )
                hypotheses.append(value)
                if (
                    cardinality == 2
                    and support_rank == 2
                    and mode_rank == 7
                ):
                    target = value
    selected = _multimodal_regional_pose_hypotheses(
        hypotheses,
        64,
        minimum_source_charts=2,
        maximum_modes_per_support=8,
    )
    assert len(selected) == 64
    assert target in selected
    selected_supports = {
        tuple(value.source_chart_ids) for value in selected
    }
    # The finite block interleaves 2/3/4-chart identities rather than letting
    # one cardinality consume the complete raw-mode quota.
    assert {len(value) for value in selected_supports} == {2, 3, 4}


def test_stage_c_chart_subset_adds_se3_geometry_beyond_rank_prefix():
    atlas = SimpleNamespace(
        maplet_ids=np.arange(8, dtype=np.int64),
        centers=np.asarray(
            [
                [0.0, 0.0, 10.0],
                [0.1, 0.0, 10.0],
                [0.2, 0.0, 10.0],
                [0.3, 0.0, 10.0],
                [-4.0, -2.0, 6.0],
                [4.0, 2.0, 18.0],
                [-3.0, 3.0, 14.0],
                [3.0, -3.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )
    selected = _geometry_conditioned_chart_subset(
        atlas,
        np.arange(8, dtype=np.int64),
        (0,),
        np.eye(4, dtype=np.float64),
        limit=4,
    )
    assert selected[0] == 0
    assert len(selected) == 4
    assert sum(int(value) >= 4 for value in selected.tolist()) >= 2


def test_stage_c_consensus_modes_cannot_starve_raw_pose_distribution():
    def hypotheses(count, offset):
        result = []
        for index in range(count):
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = float(offset + index)
            result.append(
                SimpleNamespace(
                    pose_w2c=pose,
                    source_chart_ids=(index, index + 100, index + 200),
                    score=float(count - index),
                    reprojection_error_px=1.0,
                )
            )
        return result

    raw = hypotheses(80, 1000)
    derived = hypotheses(80, 0)
    selected = _stage_c_pose_pool(raw, derived, (), 64)
    raw_identities = {id(value) for value in raw}
    derived_identities = {id(value) for value in derived}
    assert len(selected) == 64
    assert sum(id(value) in raw_identities for value in selected) == 43
    assert sum(id(value) in derived_identities for value in selected) == 21


def test_stage_c_pool_reserves_consensus_and_factorized_families():
    def hypotheses(count, offset):
        result = []
        for index in range(count):
            pose = np.eye(4, dtype=np.float64)
            pose[0, 3] = float(offset + index)
            result.append(
                SimpleNamespace(
                    pose_w2c=pose,
                    source_chart_ids=(index, index + 100, index + 200),
                    score=float(count - index),
                    reprojection_error_px=1.0,
                )
            )
        return result

    raw = hypotheses(128, 1000)
    consensus = hypotheses(32, 0)
    factorized = hypotheses(128, 500)
    selected = _stage_c_pose_pool(raw, consensus, factorized, 96)
    raw_identities = {id(value) for value in raw}
    consensus_identities = {id(value) for value in consensus}
    factorized_identities = {id(value) for value in factorized}
    assert len(selected) == 96
    assert sum(id(value) in raw_identities for value in selected) == 48
    assert sum(id(value) in consensus_identities for value in selected) == 32
    assert sum(id(value) in factorized_identities for value in selected) == 16

    expanded = _stage_c_pose_pool(raw, consensus, factorized, 160)
    assert len(expanded) == 160
    assert sum(id(value) in raw_identities for value in expanded) == 96
    assert sum(id(value) in consensus_identities for value in expanded) == 32
    assert sum(id(value) in factorized_identities for value in expanded) == 32

    production_raw = hypotheses(400, 1000)
    production = _stage_c_pose_pool(
        production_raw, consensus, factorized, 384
    )
    production_raw_identities = {
        id(value) for value in production_raw
    }
    assert len(production) == 384
    assert (
        sum(
            id(value) in production_raw_identities
            for value in production
        )
        == 320
    )
    assert (
        sum(id(value) in consensus_identities for value in production)
        == 32
    )
    assert (
        sum(id(value) in factorized_identities for value in production)
        == 32
    )


def test_stage_c_heldout_charts_exclude_candidate_sources():
    fit, heldout = _source_and_heldout_chart_ids(
        np.asarray([10, 11, 20, 21, 22]),
        (10, 11),
    )
    assert {10, 11}.issubset(set(fit.tolist()))
    assert set(fit.tolist()).isdisjoint(set(heldout.tolist()))
    assert set(fit.tolist()) | set(heldout.tolist()) == {
        10,
        11,
        20,
        21,
        22,
    }
    assert len(heldout) == 2


def test_stage_c_fixed_scoring_split_works_without_candidate_sources():
    fit, heldout = _source_and_heldout_chart_ids(
        np.asarray([10, 11, 20, 21, 22, 23]), ()
    )
    assert len(fit) == 4
    assert len(heldout) == 2
    assert set(fit.tolist()).isdisjoint(set(heldout.tolist()))
    assert set(fit.tolist()) | set(heldout.tolist()) == {
        10,
        11,
        20,
        21,
        22,
        23,
    }


def test_stage_c_score_does_not_count_correlated_solver_supports():
    low_count = SimpleNamespace(mode_support_count=1)
    high_count = SimpleNamespace(mode_support_count=20)
    assert _stage_c_pose_log_score(-4.5, low_count) == -4.5
    assert _stage_c_pose_log_score(-4.5, high_count) == -4.5


def test_pose_consensus_discovers_shared_lower_rank_planar_branch():
    def hypothesis(center_x, support, score):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = -float(center_x)
        return FramePoseHypothesis(
            pose_w2c=pose,
            score=float(score),
            source_chart_ids=tuple(support),
            reprojection_error_px=1.0,
            positive_depth=True,
        )

    values = [
        hypothesis(-10.0, (10, 11), 10.0),
        hypothesis(0.0, (10, 11), 1.0),
        hypothesis(10.0, (20, 21), 10.0),
        hypothesis(0.1, (20, 21), 1.0),
    ]
    modes = pose_distribution_consensus_modes(
        values,
        translation_radius_m=0.5,
        rotation_radius_deg=5.0,
    )
    assert modes
    centers = [
        -value.pose_w2c[:3, :3].T @ value.pose_w2c[:3, 3]
        for value in modes
    ]
    assert min(abs(float(center[0]) - 0.05) for center in centers) < 1e-6


def test_factorized_pose_modes_pair_consensus_rotation_with_member_center():
    consensus_pose = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(2.0)
    consensus_pose[:3, :3] = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    consensus = FramePoseHypothesis(
        pose_w2c=consensus_pose,
        score=3.0,
        source_chart_ids=(10, 11),
        reprojection_error_px=2.0,
        positive_depth=True,
        mode_support_count=2,
        mode_member_count=2,
    )
    member_pose = np.eye(4, dtype=np.float64)
    member_pose[0, 3] = -0.4
    member = FramePoseHypothesis(
        pose_w2c=member_pose,
        score=1.0,
        source_chart_ids=(20, 21),
        reprojection_error_px=3.0,
        positive_depth=True,
    )
    products = factorized_pose_distribution_modes(
        [member], [consensus]
    )
    assert len(products) == 1
    product = products[0]
    np.testing.assert_allclose(
        product.pose_w2c[:3, :3], consensus_pose[:3, :3]
    )
    center = -product.pose_w2c[:3, :3].T @ product.pose_w2c[:3, 3]
    np.testing.assert_allclose(center, [0.4, 0.0, 0.0], atol=1e-12)
    assert product.control_model == "chart_distribution_factorized"


def test_phase_chart_screening_marginalizes_frame_modes_once():
    matches = [
        SimpleNamespace(
            probability=0.2,
            null_probability=0.5,
            identity_probability=0.4,
        ),
        SimpleNamespace(
            probability=0.3,
            null_probability=0.5,
            identity_probability=0.4,
        ),
    ]
    assert np.isclose(
        _marginal_frame_log_evidence(matches),
        np.log(0.4),
    )


def test_pose_distribution_consensus_counts_each_chart_set_once():
    def hypothesis(center_x, support):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = -float(center_x)
        return FramePoseHypothesis(
            pose_w2c=pose,
            score=1.0,
            source_chart_ids=tuple(support),
            reprojection_error_px=1.0,
            positive_depth=True,
        )

    modes = pose_distribution_consensus_modes(
        [
            hypothesis(0.00, (1, 2)),
            hypothesis(0.05, (1, 2)),
            hypothesis(0.20, (2, 3)),
            hypothesis(0.40, (1, 3)),
        ],
        translation_radius_m=0.75,
        rotation_radius_deg=3.0,
    )
    assert modes
    assert modes[0].mode_support_count == 3
    center = (
        -modes[0].pose_w2c[:3, :3].T
        @ modes[0].pose_w2c[:3, 3]
    )
    assert 0.15 < center[0] < 0.25


@pytest.mark.parametrize("projective", [False, True])
def test_frame_lattice_rescale_preserves_physical_controls(projective):
    transform = np.asarray(
        [[5.0, -0.4, 20.25], [0.3, 3.5, 12.75]],
        dtype=np.float32,
    )
    homography = (
        np.asarray(
            [
                [5.0, -0.4, 20.25],
                [0.3, 3.5, 12.75],
                [0.015, -0.01, 1.0],
            ],
            dtype=np.float32,
        )
        if projective
        else None
    )
    control_covariance = np.eye(8, dtype=np.float32) * 7.0
    source = MapletFrameMatch(
        chart_id=12,
        canonical_to_query=transform,
        query_center_xy=transform[:, 2],
        scale_xy=np.linalg.norm(transform[:, :2], axis=0),
        in_plane_rotation_deg=3.0,
        covariance_xy=np.asarray([[0.5, 0.1], [0.1, 0.8]]),
        score=0.7,
        probability=0.6,
        null_probability=0.2,
        support_fraction=0.8,
        feature_level="coarse",
        feature_stride=16,
        canonical_homography=homography,
        support_canonical_hull=np.asarray(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            dtype=np.float32,
        ),
        control_covariance_px=control_covariance,
        identity_probability=0.4,
    )

    target = rescale_maplet_frame_match(
        source,
        feature_stride=8,
        feature_level="middle",
    )
    controls = np.asarray(
        [[-1, -1], [1, -1], [1, 1], [-1, 1], [0.2, -0.3]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        target.canonical_points_to_pixels(controls),
        source.canonical_points_to_pixels(controls),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        target.covariance_xy,
        source.covariance_xy * 4.0,
    )
    np.testing.assert_array_equal(
        target.control_covariance_px,
        source.control_covariance_px,
    )
    assert target.feature_stride == 8
    assert target.feature_level == "middle"


def test_regional_frame_evidence_is_invariant_to_chart_count():
    def match(chart_id):
        return MapletFrameMatch(
            chart_id=chart_id,
            canonical_to_query=np.asarray(
                [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0]]
            ),
            query_center_xy=np.asarray([4.0, 3.0]),
            scale_xy=np.asarray([2.0, 2.0]),
            in_plane_rotation_deg=0.0,
            covariance_xy=np.eye(2),
            score=1.0,
            probability=0.8,
            null_probability=0.1,
            support_fraction=1.0,
            feature_level="coarse",
            feature_stride=16,
            identity_probability=0.5,
        )

    single = regional_frame_log_evidence([match(1)])
    grouped = regional_frame_log_evidence(
        [match(1), match(2), match(3), match(4)]
    )
    assert np.isclose(single, grouped)


def test_stage_c_aggregate_reports_tail_and_initial_errors():
    queries = [
        {
            "stage_c_atlas_alignment": [
                {
                    "initial_translation_m": 1.0,
                    "initial_rotation_deg": 10.0,
                    "final_translation_m": 0.1,
                    "final_rotation_deg": 1.0,
                    "accepted_step_count": 1,
                }
            ]
        },
        {
            "stage_c_atlas_alignment": [
                {
                    "initial_translation_m": 2.0,
                    "initial_rotation_deg": 20.0,
                    "final_translation_m": 0.3,
                    "final_rotation_deg": 3.0,
                    "accepted_step_count": 0,
                }
            ]
        },
    ]

    aggregate = _aggregate_stage_c(queries)

    assert aggregate["top1_translation_median_m"] == pytest.approx(0.2)
    assert aggregate["top1_translation_p90_m"] == pytest.approx(0.28)
    assert aggregate["top1_rotation_p90_deg"] == pytest.approx(2.8)
    assert aggregate["top1_initial_translation_median_m"] == pytest.approx(
        1.5
    )
    assert aggregate["top1_initial_translation_p90_m"] == pytest.approx(1.9)
    assert aggregate["top1_initial_rotation_p90_deg"] == pytest.approx(19.0)
