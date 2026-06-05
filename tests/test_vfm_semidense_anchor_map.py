import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.gaussian_vfm_field import GaussianVFMField, GaussianVFMSource
from feature_extract.vfm.landmark_visibility import LandmarkVisibilityIndex
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex
from feature_extract.vfm.semidense_anchor_map import (
    GaussianConsensusAnchorConfig,
    SemiDenseAnchorMap,
    SemiDenseAnchorConfig,
    build_gaussian_consensus_anchor_map,
    build_sfm_guided_semidense_anchor_map,
    filter_semidense_by_source_visibility,
    semidense_anchor_map_stats,
    write_semidense_anchor_camera_view_visualization,
    write_semidense_anchor_visualizations,
)
from feature_extract.vfm.semidense_stage_e import (
    SemiDensePruningConfig,
    SparsePrimaryFillConfig,
    duplicate_anchor_stats,
    prune_semidense_anchors,
    sparse_primary_fill_matches,
)


def _toy_landmarks() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.asarray([0.01, 0.03], dtype=np.float32),
        observation_counts=np.asarray([5, 4], dtype=np.int64),
        observation_image_ids=(("a.png", "b.png"), ("a.png",)),
        reprojection_errors=np.asarray([0.2, 0.4], dtype=np.float32),
    )


def _toy_gaussians() -> GaussianVFMSource:
    return GaussianVFMSource(
        xyz=np.asarray(
            [
                [0.05, 0.0, 4.0],
                [1.05, 0.0, 4.0],
                [8.0, 0.0, 4.0],
                [0.08, 0.0, 4.0],
            ],
            dtype=np.float64,
        ),
        opacity=np.asarray([0.9, 0.8, 0.9, 0.05], dtype=np.float32),
        scale=np.asarray([0.02, 0.03, 0.02, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([100, 101, 102, 103], dtype=np.int64),
    )


def _toy_gaussian_field() -> GaussianVFMField:
    return GaussianVFMField(
        xyz=np.asarray(
            [
                [0.0, 0.0, 4.0],
                [0.01, 0.0, 4.0],
                [1.0, 0.0, 4.0],
                [2.0, 0.0, 4.0],
                [3.0, 0.0, 4.0],
            ],
            dtype=np.float64,
        ),
        features=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        opacity=np.asarray([0.9, 0.8, 0.05, 0.7, 0.8], dtype=np.float32),
        scale=np.asarray([0.02, 0.02, 0.02, 0.50, 0.02], dtype=np.float32),
        gaussian_indices=np.asarray([100, 101, 102, 103, 104], dtype=np.int64),
        nearest_track_ids=np.asarray([10, 10, 11, 12, 13], dtype=np.int64),
        support_counts=np.asarray([12, 7, 20, 30, 4], dtype=np.int64),
        mean_distances=np.asarray([0.04, 0.05, 0.04, 0.04, 0.08], dtype=np.float32),
        metadata={"source": "toy"},
    )


def test_sfm_guided_semidense_map_keeps_sparse_and_adds_reliable_gaussians() -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )

    assert len(anchor_map) == 4
    assert anchor_map.source_types.tolist() == ["sfm", "sfm", "gaussian_near_sfm", "gaussian_near_sfm"]
    assert anchor_map.source_track_ids.tolist() == [10, 11, 10, 11]
    assert anchor_map.source_gaussian_indices.tolist() == [-1, -1, 100, 101]
    assert np.allclose(anchor_map.features[2], np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.all(anchor_map.quality_scores >= 0.0)
    assert np.all(anchor_map.quality_scores <= 1.0)

    stats = semidense_anchor_map_stats(anchor_map, sparse_landmark_count=2, source_gaussian_count=4)
    assert stats["anchor_count"] == 4
    assert stats["gaussian_near_sfm_count"] == 2
    assert stats["expansion_ratio_vs_sparse"] == 2.0


def test_gaussian_consensus_anchor_map_filters_scores_and_diversifies_gaussians() -> None:
    anchor_map = build_gaussian_consensus_anchor_map(
        _toy_gaussian_field(),
        GaussianConsensusAnchorConfig(
            max_anchors=2,
            min_support=5,
            min_opacity=0.1,
            max_gaussian_scale=0.1,
            max_mean_distance=0.1,
            nms_voxel_size=0.05,
        ),
    )

    assert len(anchor_map) == 1
    assert anchor_map.source_types.tolist() == ["gaussian_consensus"]
    assert anchor_map.source_track_ids.tolist() == [-1]
    assert anchor_map.source_gaussian_indices.tolist() == [100]
    assert anchor_map.support_counts.tolist() == [12]
    assert anchor_map.observation_counts.tolist() == [12]
    assert anchor_map.visibility_counts.tolist() == [12]
    assert anchor_map.quality_scores[0] > 0.0
    assert np.isclose(float(np.linalg.norm(anchor_map.features[0])), 1.0)

    index = anchor_map.to_landmark_index()
    assert index.track_ids.tolist() == [-100000000]
    assert index.observation_image_ids == ((),)


def test_gaussian_consensus_anchor_map_can_inherit_nearest_track_visibility() -> None:
    anchor_map = build_gaussian_consensus_anchor_map(
        _toy_gaussian_field(),
        GaussianConsensusAnchorConfig(
            max_anchors=2,
            min_support=5,
            min_opacity=0.1,
            max_gaussian_scale=0.1,
            max_mean_distance=0.1,
            nms_voxel_size=0.05,
        ),
        nearest_track_observation_image_ids={10: ("a.png", "b.png")},
    )

    assert anchor_map.source_track_ids.tolist() == [10]
    assert anchor_map.observation_image_ids == (("a.png", "b.png"),)
    index = anchor_map.to_landmark_index()
    assert index.track_ids.tolist() == [-100000000]
    assert index.observation_image_ids == (("a.png", "b.png"),)


def test_gaussian_consensus_anchor_map_can_filter_by_keypoint_votes() -> None:
    anchor_map = build_gaussian_consensus_anchor_map(
        _toy_gaussian_field(),
        GaussianConsensusAnchorConfig(
            max_anchors=2,
            min_support=1,
            min_opacity=0.1,
            max_gaussian_scale=1.0,
            max_mean_distance=0.1,
            nms_voxel_size=0.0,
            min_keypoint_votes=2,
            keypoint_vote_weight=1.0,
        ),
        keypoint_vote_counts=np.asarray([0, 3, 0, 4, 1], dtype=np.int64),
    )

    assert len(anchor_map) == 2
    assert set(anchor_map.source_gaussian_indices.tolist()) == {101, 103}
    assert anchor_map.metadata["keypoint_vote_count_stats"]["max"] == 4


def test_stage_h_gaussian_consensus_cli_writes_anchor_map_and_summary(tmp_path) -> None:
    from feature_extract.tools.vfm.build_stage_h_gaussian_consensus_anchor_map import main

    field_path = tmp_path / "field.npz"
    output_npz = tmp_path / "anchor_map.npz"
    summary_json = tmp_path / "summary.json"
    _toy_gaussian_field().save_npz(field_path)

    main(
        [
            "--gaussian_field",
            str(field_path),
            "--output_npz",
            str(output_npz),
            "--summary_json",
            str(summary_json),
            "--max_anchors",
            "2",
            "--min_support",
            "5",
            "--min_opacity",
            "0.1",
            "--max_gaussian_scale",
            "0.1",
            "--max_mean_distance",
            "0.1",
            "--nms_voxel_size",
            "0.05",
        ]
    )

    loaded = build_gaussian_consensus_anchor_map(
        _toy_gaussian_field(),
        GaussianConsensusAnchorConfig(
            max_anchors=2,
            min_support=5,
            min_opacity=0.1,
            max_gaussian_scale=0.1,
            max_mean_distance=0.1,
            nms_voxel_size=0.05,
        ),
    )
    saved = loaded.load_npz(output_npz)
    assert len(saved) == len(loaded)
    summary = __import__("json").loads(summary_json.read_text())
    assert summary["anchor_count"] == 1
    assert summary["candidate_gaussian_count"] == 2
    assert summary["source_gaussian_count"] == 5


def test_semidense_anchor_map_converts_to_landmark_index() -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )

    index = anchor_map.to_landmark_index()

    assert len(index) == 4
    assert index.feature_dim == 2
    assert index.track_ids.tolist()[0:2] == [10, 11]
    assert index.track_ids.tolist()[2:] == [-100000002, -100000003]
    assert index.observation_counts.tolist()[2:] == [1, 1]
    assert index.observation_image_ids[0] == ("a.png", "b.png")
    assert index.observation_image_ids[2] == ("a.png", "b.png")


def test_semidense_visualization_writes_sparse_and_semidense_ply(tmp_path) -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )

    outputs = write_semidense_anchor_visualizations(
        tmp_path,
        sparse_index=_toy_landmarks(),
        semidense_map=anchor_map,
        max_points=0,
        seed=0,
    )

    assert outputs["sparse_pca_ply"].exists()
    assert outputs["semidense_pca_ply"].exists()
    assert outputs["semidense_source_ply"].exists()
    text = outputs["semidense_source_ply"].read_text()
    assert "element vertex 4" in text
    assert "property int anchor_id" in text


def test_camera_view_visualization_projects_sparse_and_semidense_anchors(tmp_path) -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )
    camera = ColmapCamera(camera_id=1, model_id=1, width=160, height=120, params=(80.0, 80.0, 80.0, 60.0))
    pose_w2c = np.eye(4, dtype=np.float64)
    image = np.full((120, 160, 3), 80, dtype=np.uint8)

    output = write_semidense_anchor_camera_view_visualization(
        tmp_path / "camera_view.png",
        sparse_index=_toy_landmarks(),
        semidense_map=anchor_map,
        pose_w2c=pose_w2c,
        camera=camera,
        image_rgb=image,
        max_points=0,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_source_visibility_filter_keeps_gaussian_anchors_from_visible_sfm_tracks() -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )
    visibility = LandmarkVisibilityIndex({"ref.png": frozenset({10})})

    subset, gate = filter_semidense_by_source_visibility(anchor_map, visibility, ["ref.png"])

    assert len(subset) == 2
    assert subset.track_ids.tolist() == [10, -100000002]
    assert gate["full_visible_tracks"] == 1
    assert gate["bank_visible_tracks"] == 2


def test_source_visibility_filter_keeps_non_sfm_anchors_from_observation_images() -> None:
    anchor_map = SemiDenseAnchorMap(
        anchor_ids=np.asarray([1, 2, 3], dtype=np.int64),
        xyz=np.asarray([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [2.0, 0.0, 4.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        source_types=np.asarray(["vfm_2dgs", "vfm_2dgs", "sfm"], dtype=str),
        source_track_ids=np.asarray([-1, -1, 10], dtype=np.int64),
        source_gaussian_indices=np.asarray([100, 101, -1], dtype=np.int64),
        support_counts=np.asarray([2, 3, 4], dtype=np.int64),
        mean_distances=np.asarray([0.0, 0.0, 0.1], dtype=np.float32),
        feature_variances=np.asarray([0.01, 0.02, 0.03], dtype=np.float32),
        observation_counts=np.asarray([2, 3, 4], dtype=np.int64),
        visibility_counts=np.asarray([2, 1, 1], dtype=np.int64),
        quality_scores=np.asarray([0.8, 0.7, 1.0], dtype=np.float32),
        opacity=np.ones((3,), dtype=np.float32),
        scale=np.zeros((3,), dtype=np.float32),
        observation_image_ids=(("ref_a.png", "ref_b.png"), ("ref_c.png",), ("ref_a.png",)),
    )
    visibility = LandmarkVisibilityIndex({"ref_a.png": frozenset({10})})

    subset, gate = filter_semidense_by_source_visibility(anchor_map, visibility, ["ref_b.png"])

    assert subset.track_ids.tolist() == [-100000001]
    assert gate["full_visible_tracks"] == 0
    assert gate["observation_image_visible_anchors"] == 1
    assert gate["bank_visible_tracks"] == 1


def test_semidense_pruning_and_duplicate_stats() -> None:
    anchor_map = build_sfm_guided_semidense_anchor_map(
        _toy_landmarks(),
        _toy_gaussians(),
        SemiDenseAnchorConfig(max_distance=0.15, k_neighbors=1, min_support=1, min_opacity=0.1),
    )

    pruned = prune_semidense_anchors(anchor_map, SemiDensePruningConfig(min_quality=0.1, max_per_source_track=1))
    stats = duplicate_anchor_stats(anchor_map, radius_m=0.2)

    assert len(pruned) == 2
    assert stats["anchor_count"] == 4
    assert stats["max_anchors_per_source"] == 2


def test_sparse_primary_fill_only_adds_semidense_for_unmatched_tokens() -> None:
    from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch

    sparse = [
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([10.0, 10.0]),
            track_id=10,
            xyz=np.asarray([0.0, 0.0, 1.0]),
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.0,
        )
    ]
    semi = [
        QueryTo3DMatch(
            token_index=1,
            xy=np.asarray([10.0, 10.0]),
            track_id=-100,
            xyz=np.asarray([0.0, 0.0, 1.0]),
            similarity=0.95,
            ratio=0.0,
            landmark_variance=0.0,
        ),
        QueryTo3DMatch(
            token_index=2,
            xy=np.asarray([20.0, 10.0]),
            track_id=-101,
            xyz=np.asarray([1.0, 0.0, 1.0]),
            similarity=0.8,
            ratio=0.0,
            landmark_variance=0.0,
        ),
    ]

    merged = sparse_primary_fill_matches(
        sparse,
        semi,
        SparsePrimaryFillConfig(mode="no_sparse", max_semidense_fraction=0.5),
        image_width=100,
        image_height=100,
    )

    assert [match.track_id for match in merged] == [10, -101]
