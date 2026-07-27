from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from feature_extract.tools.vfm.train_v6_global_frame_encoder import (
    _soft_projection_nll,
)
from feature_extract.tools.vfm.evaluate_v6_maplet_frame_alignment import (
    _retrieval_metrics,
    _select_oracle_regions,
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
)
from feature_extract.vfm.localization_v6.maplet_atlas import (
    MapletFeatureAtlasBank,
)
from feature_extract.vfm.localization_v6.maplet_frame_alignment import (
    MapletFrameSearchConfig,
    _Candidate,
    _select_distinct_candidates,
    frame_matches_to_pose_hypotheses,
    ground_truth_chart_frame,
    pose_hypotheses_for_mode_sets,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    QueryMapletGroup,
    _compress_surface_location_modes,
    aggregate_scene_maplet_evidence,
)
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
    assert all(
        np.isclose(
            hypothesis.score,
            sum(match.score for match in matches),
        )
        for hypothesis in hypotheses
    )


def test_three_correlated_chart_centers_recover_pose_without_point_anchors():
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
    center_hypotheses = [
        value
        for value in hypotheses
        if value.control_model == "chart_centers"
    ]
    assert center_hypotheses
    errors = [
        pnp_pose_error(value.pose_w2c, pose)
        for value in center_hypotheses
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
