from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_p1_aligned_observation_candidate_layout import (
    _POINT_SOURCE,
    _expand_train_observation_anchor_jitter,
    _select_static_rows,
    _subset_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
)


def _layout() -> CandidatePoseRGBSpatialLayout:
    point_count = 8
    candidate_count = 2
    views = 2
    priors = np.full((point_count, candidate_count), 0.45, dtype=np.float32)
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.arange(point_count, dtype=np.int64),
        query_ids=np.asarray(["query.png"] * point_count),
        split_names=np.asarray(["train"] * point_count),
        xy=np.asarray(
            [[80.0, 80.0], [240.0, 80.0], [400.0, 80.0], [560.0, 80.0], [80.0, 400.0], [240.0, 400.0], [400.0, 400.0], [560.0, 400.0]],
            dtype=np.float32,
        ),
        point_sources=np.asarray([_POINT_SOURCE] * point_count),
        candidate_track_ids=np.tile(np.asarray([[11, 12]], dtype=np.int64), (point_count, 1)),
        candidate_bank_rows=np.tile(np.asarray([[0, 1]], dtype=np.int64), (point_count, 1)),
        candidate_coarse_similarities=np.tile(np.asarray([[0.9, 0.8]], dtype=np.float32), (point_count, 1)),
        candidate_prior_probabilities=priors,
        null_probabilities=np.full((point_count,), 0.1, dtype=np.float32),
        support_image_ids=np.full((point_count, candidate_count, views), "support.png"),
        support_xy=np.full((point_count, candidate_count, views, 2), 256.0, dtype=np.float32),
        support_view_valid=np.ones((point_count, candidate_count, views), dtype=bool),
        support_view_weights=np.full((point_count, candidate_count, views), 0.5, dtype=np.float32),
        support_coverage_counts=np.ones((point_count, candidate_count, views), dtype=np.int32),
        metadata={
            "format": "candidate_pose_rgb_spatial_layout_v1",
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "anchors",
            "maplet_support_index_sha256": "maplet",
            "support_geometry_index_sha256": "geometry",
            "projection_space_id": "projection",
            "descriptor_space_id": "descriptor",
        },
    )


def test_static_observation_subset_is_target_free_and_spatially_fixed() -> None:
    layout = _layout()
    selected = _select_static_rows(
        layout=layout,
        points_per_query=4,
        policy="support_coverage",
        grid_rows=2,
        grid_columns=2,
        coordinate_image_size=(640, 480),
        border_margin_px=48.0,
    )
    assert selected.shape == (4,)
    assert len(np.unique(selected)) == 4
    output = _subset_layout(
        layout,
        selected,
        {
            **layout.metadata,
            "training_only_anchor_layout": True,
            "runtime_scorer_must_not_load_this_layout": True,
        },
    )
    assert output.row_count == 4
    assert np.array_equal(output.source_point_ids, selected)
    assert output.metadata["training_only_anchor_layout"] is True
    assert output.metadata["runtime_scorer_must_not_load_this_layout"] is True


def test_train_observation_jitter_is_deterministic_and_preserves_track_alignment() -> None:
    query_ids = np.asarray(["a.png", "b.png"], dtype="<U5")
    query_xy = np.asarray([[128.0, 96.0], [256.0, 192.0]], dtype=np.float32)
    query_tracks = np.asarray([11, 19], dtype=np.int64)
    first = _expand_train_observation_anchor_jitter(
        query_ids=query_ids,
        query_xy=query_xy,
        query_tracks=query_tracks,
        radius_px=4.0,
        copies=2,
        seed=17,
    )
    second = _expand_train_observation_anchor_jitter(
        query_ids=query_ids,
        query_xy=query_xy,
        query_tracks=query_tracks,
        radius_px=4.0,
        copies=2,
        seed=17,
    )
    ids, xy, tracks = first
    assert np.array_equal(ids, second[0])
    assert np.allclose(xy, second[1])
    assert np.array_equal(tracks, second[2])
    assert ids.shape == (6,)
    assert xy.shape == (6, 2)
    assert tracks.shape == (6,)
    assert np.array_equal(ids[:2], query_ids)
    assert np.allclose(xy[:2], query_xy)
    assert np.array_equal(tracks, np.tile(query_tracks, 3))
    distances = np.linalg.norm(xy[2:] - np.tile(query_xy, (2, 1)), axis=1)
    assert np.all(distances <= 4.0001)
    assert np.any(distances > 0.0)
