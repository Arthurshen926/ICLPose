from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.probe_detector_maplet_geometry import (
    _pose_gate,
    _select_query_rows,
)
from feature_extract.vfm.colmap_tracks import ColmapTrackObservation
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    VIEW_EVIDENCE_NAMES,
    SupportMapletView,
    SupportObservationFeatureStore,
    aggregate_maplet_view_evidence,
    build_support_observation_geometry_index,
    canonical_rows_for_track_candidates,
    load_support_observation_geometry_index_npz,
    save_support_observation_geometry_index_npz,
    score_maplet_support_view,
    select_dense_query_context_neighborhood,
    select_support_maplet_view_indices,
)


def _observation(track_id: int, image_id: str, xy: tuple[float, float]) -> ColmapTrackObservation:
    return ColmapTrackObservation(
        track_id=track_id,
        image_id=image_id,
        point2d_idx=track_id,
        xy=xy,
        xyz=np.asarray([float(track_id), 0.0, 1.0]),
        track_length=2,
        reprojection_error=0.25,
        viewing_ray=np.asarray([0.0, 0.0, 1.0]),
        image_width=100,
        image_height=80,
    )


def test_support_geometry_index_roundtrip_and_feature_join(tmp_path: Path) -> None:
    observations = [
        _observation(3, "b.png", (30.0, 3.0)),
        _observation(2, "a.png", (20.0, 2.0)),
        _observation(1, "a.png", (10.0, 1.0)),
    ]
    index = build_support_observation_geometry_index(observations)
    assert index.image_ids == ("a.png", "b.png")
    assert index.track_ids.tolist() == [1, 2, 3]
    assert index.source_row_indices.tolist() == [2, 1, 0]
    assert index.geometry_rows_for_tracks("a.png", np.asarray([2, 9, 1])).tolist() == [1, -1, 0]

    path = tmp_path / "geometry.npz"
    save_support_observation_geometry_index_npz(index, path, metadata={"source": "test"})
    loaded, metadata = load_support_observation_geometry_index_npz(path)
    assert metadata["source"] == "test"
    assert np.array_equal(loaded.source_row_indices, index.source_row_indices)

    source_tracks = np.asarray([3, 2, 1], dtype=np.int64)
    source_descriptors = np.eye(3, dtype=np.float32)
    store = SupportObservationFeatureStore(
        loaded,
        source_track_ids=source_tracks,
        source_descriptors=source_descriptors,
        source_detector_scores=np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
    )
    view = store.maplet_view("a.png", np.asarray([1, 3, 2], dtype=np.int64))
    assert view.track_ids.tolist() == [1, 2]
    assert np.allclose(view.xy, [[10.0, 1.0], [20.0, 2.0]])
    assert np.allclose(view.detector_scores, [0.1, 0.2])


def test_candidate_tracks_are_resolved_independently_of_prototype_rows() -> None:
    candidates = np.asarray([[30, 10, -1], [20, 30, 10]], dtype=np.int64)
    canonical = np.asarray([20, 10, 30], dtype=np.int64)
    rows = canonical_rows_for_track_candidates(candidates, canonical)
    assert rows.tolist() == [[2, 1, -1], [0, 2, 1]]


def test_maplet_view_geometry_distinguishes_consistent_anchor() -> None:
    descriptors = np.eye(4, dtype=np.float32)
    support = SupportMapletView(
        track_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        xy=np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]], dtype=np.float32),
        descriptors=descriptors,
        detector_scores=np.ones((4,), dtype=np.float32),
        reprojection_errors=np.zeros((4,), dtype=np.float32),
    )
    query_xy = 2.0 * support.xy + np.asarray([100.0, 50.0], dtype=np.float32)
    correct = score_maplet_support_view(
        query_xy=query_xy,
        query_descriptors=descriptors,
        query_anchor_index=0,
        support_view=support,
        anchor_track_id=10,
        expected_maplet_track_count=4,
        min_similarity=0.5,
        geometry_threshold_px=2.0,
    )
    names = {name: index for index, name in enumerate(VIEW_EVIDENCE_NAMES)}
    assert correct[names["context_model_found"]] == 1.0
    assert correct[names["context_inlier_count"]] == 3.0
    assert correct[names["anchor_reprojection_residual_px"]] < 1e-4
    assert correct[names["anchor_geometry_consistent"]] == 1.0

    wrong_xy = query_xy.copy()
    wrong_xy[0] += np.asarray([40.0, 0.0], dtype=np.float32)
    wrong = score_maplet_support_view(
        query_xy=wrong_xy,
        query_descriptors=descriptors,
        query_anchor_index=0,
        support_view=support,
        anchor_track_id=10,
        expected_maplet_track_count=4,
        min_similarity=0.5,
        geometry_threshold_px=2.0,
    )
    assert wrong[names["context_model_found"]] == 1.0
    assert wrong[names["anchor_reprojection_residual_px"]] > 30.0
    assert wrong[names["anchor_geometry_consistent"]] == 0.0

    aggregated, feature_names = aggregate_maplet_view_evidence(np.stack([correct, wrong]))
    assert aggregated.shape == (3 * len(VIEW_EVIDENCE_NAMES),)
    assert len(feature_names) == len(aggregated)


def test_support_view_selection_can_be_query_conditioned() -> None:
    views = [
        SupportMapletView(
            track_ids=np.asarray([10, 11, 12], dtype=np.int64),
            xy=np.zeros((3, 2), dtype=np.float32),
            descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
            detector_scores=np.ones((3,), dtype=np.float32),
            reprojection_errors=np.zeros((3,), dtype=np.float32),
        ),
        SupportMapletView(
            track_ids=np.asarray([10, 11], dtype=np.int64),
            xy=np.zeros((2, 2), dtype=np.float32),
            descriptors=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            detector_scores=np.ones((2,), dtype=np.float32),
            reprojection_errors=np.zeros((2,), dtype=np.float32),
        ),
    ]
    coverage = select_support_maplet_view_indices(
        views,
        query_anchor_descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
        anchor_track_id=10,
        expected_maplet_track_count=3,
        top_k=1,
        strategy="coverage",
    )
    descriptor = select_support_maplet_view_indices(
        views,
        query_anchor_descriptor=np.asarray([0.0, 1.0], dtype=np.float32),
        anchor_track_id=10,
        expected_maplet_track_count=3,
        top_k=1,
        strategy="anchor_similarity",
    )
    assert coverage.tolist() == [0]
    assert descriptor.tolist() == [1]


def test_dense_query_context_prepends_anchor_and_removes_duplicate() -> None:
    xy, descriptors, anchor = select_dense_query_context_neighborhood(
        anchor_xy=np.asarray([10.0, 10.0], dtype=np.float32),
        anchor_descriptor=np.asarray([1.0, 0.0], dtype=np.float32),
        context_xy=np.asarray([[10.5, 10.0], [15.0, 10.0], [20.0, 10.0], [100.0, 100.0]], dtype=np.float32),
        context_descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32),
        context_scores=np.asarray([1.0, 0.5, 0.9, 0.8], dtype=np.float32),
        radius_px=20.0,
        max_points=3,
        duplicate_radius_px=2.0,
    )
    assert anchor == 0
    assert np.allclose(xy, [[10.0, 10.0], [20.0, 10.0], [15.0, 10.0]])
    assert np.allclose(descriptors[0], [1.0, 0.0])


def test_pose_gate_rejects_missing_or_non_finite_error_metrics() -> None:
    baseline = {
        "median_translation_m_success": 0.1,
        "p90_translation_m_success": 0.2,
        "median_rotation_deg_success": 0.3,
        "success_rate": 1.0,
        "recall_25cm_2deg": 1.0,
        "recall_10cm_5deg": 0.5,
        "recall_5cm_5deg": 0.1,
    }
    missing = dict(baseline, median_translation_m_success=None, success_rate=0.0)
    non_finite = dict(baseline, p90_translation_m_success=np.nan)

    assert not _pose_gate(missing, baseline)
    assert not _pose_gate(non_finite, baseline)


def test_query_selector_uses_colmap_dimensions_and_fills_pose_keep_budget() -> None:
    query_ids = np.asarray(["a.png"] * 4 + ["b.png"] * 3)
    xy = np.asarray(
        [
            [100.0, 100.0],
            [1500.0, 100.0],
            [100.0, 900.0],
            [1500.0, 900.0],
            [100.0, 100.0],
            [1500.0, 100.0],
            [100.0, 900.0],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8], dtype=np.float32)
    pose_keep = np.asarray([True, False, False, False, True, True, True])

    selected = _select_query_rows(
        query_ids=query_ids,
        xy=xy,
        detector_scores=scores,
        pose_keep_mask=pose_keep,
        image_sizes_by_id={"a.png": (1920, 1080), "b.png": (1920, 1080)},
        top_k=3,
    )

    assert selected.tolist() == [0, 1, 2, 4, 5, 6]
    assert int(np.sum(pose_keep[selected])) == 4
