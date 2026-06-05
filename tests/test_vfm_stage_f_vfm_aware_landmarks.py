import numpy as np

from feature_extract.vfm.vfm_aware_landmarks import (
    aggregate_track_match_reliability,
    anchor_coverage_summary,
    candidate_reference_image_order,
    combine_selected_track_banks,
    filter_selected_track_bank_by_ids,
    fundamental_from_w2c_poses,
    link_vfm_patch_pair_rows,
    reciprocal_token_matches,
    sampson_epipolar_errors_px,
    select_distinctive_tokens,
    project_selected_track_bank,
    triangulate_multiview_dlt,
    triangulate_two_view_dlt,
    token_distinctiveness_scores,
)


def test_token_distinctiveness_marks_duplicate_tokens_as_less_useful():
    features = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    scores = token_distinctiveness_scores(features)

    assert scores.shape == (4,)
    assert scores[0] < 1e-5
    assert scores[1] < 1e-5
    assert scores[2] > scores[0]


def test_select_distinctive_tokens_respects_top_count_and_boundary_mask():
    scores = np.asarray([0.1, 0.9, 0.8, 0.7], dtype=np.float32)
    xy = np.asarray(
        [
            [0.0, 0.0],
            [5.0, 5.0],
            [9.0, 5.0],
            [5.0, 9.0],
        ],
        dtype=np.float64,
    )

    selected = select_distinctive_tokens(
        scores,
        xy,
        image_width=10,
        image_height=10,
        top_count=2,
        min_distance_to_boundary_px=2.0,
    )

    assert selected.tolist() == [1]


def test_anchor_coverage_summary_reports_high_saliency_missing_anchor_ratio():
    saliency = np.asarray([0.9, 0.85, 0.2, 0.8], dtype=np.float32)
    has_anchor = np.asarray([False, True, True, False], dtype=bool)

    summary = anchor_coverage_summary(saliency, has_anchor, top_fractions=(0.5,))

    assert summary["token_count"] == 4
    assert summary["anchor_token_fraction"] == 0.5
    assert summary["top_50pct_anchor_fraction"] == 0.5
    assert summary["top_50pct_missing_anchor_fraction"] == 0.5
    assert summary["mean_saliency_with_anchor"] < summary["mean_saliency_without_anchor"]


def test_candidate_reference_image_order_uses_ranked_unique_topk_references():
    rows = [
        {"record_type": "header"},
        {
            "record_type": "candidate",
            "query_id": "q1.png",
            "reference_image": "r2.png",
            "metadata": {"retrieval_rank": 2},
        },
        {
            "record_type": "candidate",
            "query_id": "q1.png",
            "reference_image": "r1.png",
            "metadata": {"retrieval_rank": 1},
        },
        {
            "record_type": "candidate",
            "query_id": "q1.png",
            "reference_image": "r1.png",
            "metadata": {"retrieval_rank": 3},
        },
        {
            "record_type": "candidate",
            "query_id": "q2.png",
            "reference_image": "r3.png",
            "metadata": {"retrieval_rank": 1},
        },
        {"record_type": "diagnostic", "query_id": "q2.png", "reference_image": "ignored.png"},
    ]

    ordered = candidate_reference_image_order(rows, top_n=2)

    assert ordered == ["r1.png", "r2.png", "r3.png"]


def test_candidate_reference_image_order_can_limit_candidate_queries():
    rows = [
        {"record_type": "candidate", "query_id": "q1.png", "reference_image": "r1.png", "metadata": {"retrieval_rank": 1}},
        {"record_type": "candidate", "query_id": "q2.png", "reference_image": "r2.png", "metadata": {"retrieval_rank": 1}},
        {"record_type": "candidate", "query_id": "q3.png", "reference_image": "r3.png", "metadata": {"retrieval_rank": 1}},
    ]

    ordered = candidate_reference_image_order(rows, top_n=1, max_queries=2)

    assert ordered == ["r1.png", "r2.png"]


def test_reciprocal_token_matches_returns_only_mutual_pairs():
    source = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    matches = reciprocal_token_matches(
        source,
        target,
        source_token_indices=np.asarray([10, 11, 12], dtype=np.int64),
        target_token_indices=np.asarray([20, 21, 22], dtype=np.int64),
        min_similarity=0.95,
    )

    assert [(m.source_token_index, m.target_token_index) for m in matches] == [(10, 20), (11, 21)]
    assert all(m.similarity >= 0.95 for m in matches)


def test_epipolar_errors_are_small_for_points_on_the_same_horizontal_epipolar_line():
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = np.eye(4, dtype=np.float64)
    pose_b[:3, 3] = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    intrinsic = np.eye(3, dtype=np.float64)
    fundamental = fundamental_from_w2c_poses(pose_a, pose_b, intrinsic, intrinsic)

    errors = sampson_epipolar_errors_px(
        np.asarray([[0.0, 0.0], [0.3, 0.0]], dtype=np.float64),
        np.asarray([[0.2, 0.0], [-0.4, 0.0]], dtype=np.float64),
        fundamental,
    )
    off_line = sampson_epipolar_errors_px(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        np.asarray([[0.2, 0.5]], dtype=np.float64),
        fundamental,
    )

    assert float(np.max(errors)) < 1e-6
    assert off_line[0] > 0.1


def test_triangulate_two_view_dlt_recovers_simple_point():
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = np.eye(4, dtype=np.float64)
    pose_b[:3, 3] = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    intrinsic = np.eye(3, dtype=np.float64)

    xyz, reproj_a, reproj_b = triangulate_two_view_dlt(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        np.asarray([[-0.2, 0.0]], dtype=np.float64),
        pose_a,
        pose_b,
        intrinsic,
        intrinsic,
    )

    assert xyz.shape == (1, 3)
    assert np.allclose(xyz[0], np.asarray([0.0, 0.0, 5.0]), atol=1e-5)
    assert reproj_a[0] < 1e-6
    assert reproj_b[0] < 1e-6


def test_link_vfm_patch_pair_rows_connects_shared_observations():
    rows = [
        {
            "source_image_id": "a.png",
            "target_image_id": "b.png",
            "source_token_index": 1,
            "target_token_index": 2,
            "source_xy": [1.0, 1.0],
            "target_xy": [2.0, 1.0],
            "similarity": 0.9,
        },
        {
            "source_image_id": "b.png",
            "target_image_id": "c.png",
            "source_token_index": 2,
            "target_token_index": 3,
            "source_xy": [2.0, 1.0],
            "target_xy": [3.0, 1.0],
            "similarity": 0.8,
        },
        {
            "source_image_id": "x.png",
            "target_image_id": "y.png",
            "source_token_index": 4,
            "target_token_index": 5,
            "source_xy": [4.0, 1.0],
            "target_xy": [5.0, 1.0],
            "similarity": 0.7,
        },
    ]

    tracks = link_vfm_patch_pair_rows(rows, min_observations=3)

    assert len(tracks) == 1
    assert [(obs.image_id, obs.token_index) for obs in tracks[0].observations] == [
        ("a.png", 1),
        ("b.png", 2),
        ("c.png", 3),
    ]
    assert tracks[0].pair_count == 2


def test_triangulate_multiview_dlt_recovers_simple_point():
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = np.eye(4, dtype=np.float64)
    pose_b[:3, 3] = np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)
    pose_c = np.eye(4, dtype=np.float64)
    pose_c[:3, 3] = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    intrinsic = np.eye(3, dtype=np.float64)

    xyz, errors = triangulate_multiview_dlt(
        [np.asarray([0.0, 0.0]), np.asarray([-0.2, 0.0]), np.asarray([0.0, -0.2])],
        [pose_a, pose_b, pose_c],
        [intrinsic, intrinsic, intrinsic],
    )

    assert np.allclose(xyz, np.asarray([0.0, 0.0, 5.0]), atol=1e-5)
    assert float(np.max(errors)) < 1e-6


def test_combine_selected_track_banks_keeps_prefixed_track_ids():
    from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature

    bank_a = SelectedTrackFeatureBank(
        feature_dim=2,
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros((2,), dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("a.png",),
            )
        },
    )
    bank_b = SelectedTrackFeatureBank(
        feature_dim=2,
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([0.0, 1.0], dtype=np.float32),
                variance=np.zeros((2,), dtype=np.float32),
                observation_count=3,
                mean_utility=2.0,
                observation_image_ids=("b.png",),
            )
        },
    )

    combined = combine_selected_track_banks((bank_a, bank_b), track_id_offsets=(0, 100))

    assert sorted(combined.tracks) == [1, 101]
    assert combined.tracks[101].track_id == 101
    assert combined.tracks[101].observation_count == 3


def test_project_selected_track_bank_preserves_metadata_and_normalizes_features():
    import torch
    from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature

    class SliceProjector(torch.nn.Module):
        def forward(self, features):
            return features[:, :2]

    bank = SelectedTrackFeatureBank(
        feature_dim=4,
        tracks={
            7: TrackFeature(
                track_id=7,
                mean_feature=np.asarray([3.0, 4.0, 10.0, 0.0], dtype=np.float32),
                variance=np.ones((4,), dtype=np.float32),
                observation_count=2,
                mean_utility=1.5,
                observation_image_ids=("a.png", "b.png"),
            )
        },
    )

    projected = project_selected_track_bank(bank, SliceProjector(), output_dim=2)

    assert projected.feature_dim == 2
    assert projected.tracks[7].observation_count == 2
    assert projected.tracks[7].observation_image_ids == ("a.png", "b.png")
    assert np.allclose(projected.tracks[7].mean_feature, np.asarray([0.6, 0.8], dtype=np.float32), atol=1e-6)


def test_aggregate_track_match_reliability_computes_per_track_precision():
    rows = [
        {
            "track_id": 10,
            "patch_correct": True,
            "stride_positive_label": True,
            "pnp_inlier": True,
            "gt_reproj_error_px": 4.0,
            "similarity": 0.9,
            "observation_count": 4,
            "landmark_reprojection_error": 1.0,
            "landmark_ambiguity": 0.2,
        },
        {
            "track_id": 10,
            "patch_correct": False,
            "stride_positive_label": True,
            "pnp_inlier": False,
            "gt_reproj_error_px": 20.0,
            "similarity": 0.7,
            "observation_count": 4,
            "landmark_reprojection_error": 1.5,
            "landmark_ambiguity": 0.3,
        },
        {
            "track_id": 11,
            "patch_correct": False,
            "stride_positive_label": False,
            "pnp_inlier": False,
            "gt_reproj_error_px": 40.0,
            "similarity": 0.8,
            "observation_count": 3,
            "landmark_reprojection_error": 3.0,
            "landmark_ambiguity": 0.9,
        },
    ]

    reliability = aggregate_track_match_reliability(rows)

    assert reliability[10]["match_count"] == 2
    assert reliability[10]["patch_precision"] == 0.5
    assert reliability[10]["stride_precision"] == 1.0
    assert reliability[10]["pnp_inlier_rate"] == 0.5
    assert reliability[10]["median_gt_reproj_error_px"] == 12.0
    assert reliability[10]["mean_similarity"] == 0.8
    assert reliability[11]["patch_precision"] == 0.0


def test_filter_selected_track_bank_by_ids_preserves_requested_tracks():
    from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature

    bank = SelectedTrackFeatureBank(
        feature_dim=2,
        tracks={
            1: TrackFeature(
                track_id=1,
                mean_feature=np.asarray([1.0, 0.0], dtype=np.float32),
                variance=np.zeros((2,), dtype=np.float32),
                observation_count=2,
                mean_utility=1.0,
                observation_image_ids=("a.png",),
            ),
            2: TrackFeature(
                track_id=2,
                mean_feature=np.asarray([0.0, 1.0], dtype=np.float32),
                variance=np.ones((2,), dtype=np.float32),
                observation_count=3,
                mean_utility=2.0,
                observation_image_ids=("b.png",),
            ),
        },
    )

    filtered = filter_selected_track_bank_by_ids(bank, {2, 99})

    assert sorted(filtered.tracks) == [2]
    assert filtered.feature_dim == 2
    assert filtered.tracks[2].observation_image_ids == ("b.png",)
