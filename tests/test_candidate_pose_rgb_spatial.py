from __future__ import annotations

import json

import numpy as np
import pytest

from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
    CandidatePoseRGBSpatialLayout,
    select_fixed_rgb_support_views,
    load_candidate_pose_rgb_spatial_layout,
    save_candidate_pose_rgb_spatial_layout,
)
from feature_extract.tools.vfm.build_mixed_multiscale_candidate_layout import (
    validate_mixed_layout_exported_splits,
)


def _layout(*, support_weight: np.ndarray | None = None) -> CandidatePoseRGBSpatialLayout:
    weights = (
        np.asarray([[[0.75, 0.25]]], dtype=np.float32)
        if support_weight is None
        else np.asarray(support_weight, dtype=np.float32)
    )
    return CandidatePoseRGBSpatialLayout(
        source_point_ids=np.asarray([17], dtype=np.int64),
        query_ids=np.asarray(["query/frame.png"]),
        split_names=np.asarray(["train"]),
        xy=np.asarray([[128.0, 96.0]], dtype=np.float32),
        point_sources=np.asarray(["alike_high_detail"]),
        candidate_track_ids=np.asarray([[101]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[9]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.8]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.9]], dtype=np.float32),
        null_probabilities=np.asarray([0.1], dtype=np.float32),
        support_image_ids=np.asarray([[["map/frame-a.png", "map/frame-b.png"]]]),
        support_xy=np.asarray([[[[12.0, 24.0], [36.0, 48.0]]]], dtype=np.float32),
        support_view_valid=np.asarray([[[True, True]]]),
        support_view_weights=weights,
        support_coverage_counts=np.asarray([[[7, 3]]], dtype=np.int32),
        metadata={
            "format": CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT,
            "contains_ground_truth": False,
            "contains_target_errors": False,
            "pose_or_ground_truth_used": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "verification_points_sha256": "points-hash",
            "maplet_support_index_sha256": "maplet-hash",
            "support_geometry_index_sha256": "geometry-hash",
            "projection_space_id": "projection-space",
            "descriptor_space_id": "descriptor-space",
        },
    )


def test_rgb_spatial_layout_round_trips_and_preserves_fixed_view_mixture(tmp_path) -> None:
    path = tmp_path / "layout.npz"
    expected = _layout()

    save_candidate_pose_rgb_spatial_layout(expected, path)
    actual = load_candidate_pose_rgb_spatial_layout(path)

    np.testing.assert_array_equal(actual.source_point_ids, expected.source_point_ids)
    np.testing.assert_array_equal(actual.candidate_track_ids, expected.candidate_track_ids)
    np.testing.assert_array_equal(actual.support_image_ids, expected.support_image_ids)
    np.testing.assert_allclose(actual.support_view_weights, expected.support_view_weights)
    assert actual.metadata["verification_points_sha256"] == "points-hash"


def test_rgb_spatial_layout_rejects_non_normalized_valid_view_weights() -> None:
    with pytest.raises(ValueError, match="view weights"):
        _layout(support_weight=np.asarray([[[0.6, 0.6]]], dtype=np.float32))


def test_rgb_spatial_layout_accepts_an_empty_invalid_candidate_slot() -> None:
    layout = _layout()
    CandidatePoseRGBSpatialLayout(
        source_point_ids=layout.source_point_ids,
        query_ids=layout.query_ids,
        split_names=layout.split_names,
        xy=layout.xy,
        point_sources=layout.point_sources,
        candidate_track_ids=np.asarray([[101, -1]], dtype=np.int64),
        candidate_bank_rows=np.asarray([[9, -1]], dtype=np.int64),
        candidate_coarse_similarities=np.asarray([[0.8, 0.0]], dtype=np.float32),
        candidate_prior_probabilities=np.asarray([[0.9, 0.0]], dtype=np.float32),
        null_probabilities=layout.null_probabilities,
        support_image_ids=np.asarray(
            [[["map/frame-a.png", "map/frame-b.png"], ["", ""]]]
        ),
        support_xy=np.asarray(
            [[[[12.0, 24.0], [36.0, 48.0]], [[0.0, 0.0], [0.0, 0.0]]]],
            dtype=np.float32,
        ),
        support_view_valid=np.asarray([[[True, True], [False, False]]]),
        support_view_weights=np.asarray([[[0.75, 0.25], [0.0, 0.0]]], dtype=np.float32),
        support_coverage_counts=np.asarray([[[7, 3], [0, 0]]], dtype=np.int32),
        metadata=layout.metadata,
    )


def test_rgb_spatial_layout_rejects_target_bearing_manifest(tmp_path) -> None:
    path = tmp_path / "layout.npz"
    save_candidate_pose_rgb_spatial_layout(_layout(), path)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["contains_ground_truth"] = True
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    with pytest.raises(ValueError, match="target-free"):
        load_candidate_pose_rgb_spatial_layout(path)


def test_fixed_rgb_support_views_exclude_query_image_and_renormalize_coverage() -> None:
    support_ids, support_xy, support_valid, support_weights, support_coverage = (
        select_fixed_rgb_support_views(
            query_ids=np.asarray(["map/a.png", "query/other.png"]),
            candidate_track_ids=np.asarray([[101], [101]], dtype=np.int64),
            candidate_bank_rows=np.asarray([[0], [0]], dtype=np.int64),
            maplet_support_image_ids=np.asarray(
                ["map/a.png", "map/b.png", "map/c.png"]
            ),
            maplet_support_image_indices=np.asarray([[0, 1, 2]], dtype=np.int64),
            maplet_support_coverage_counts=np.asarray([[9, 3, 1]], dtype=np.int32),
            support_xy_by_image_track={
                ("map/a.png", 101): np.asarray([10.0, 20.0]),
                ("map/b.png", 101): np.asarray([30.0, 40.0]),
                ("map/c.png", 101): np.asarray([50.0, 60.0]),
            },
            support_views_per_candidate=2,
        )
    )

    np.testing.assert_array_equal(
        support_ids[0, 0], np.asarray(["map/b.png", "map/c.png"])
    )
    np.testing.assert_array_equal(support_valid[0, 0], np.asarray([True, True]))
    np.testing.assert_allclose(support_weights[0, 0], np.asarray([0.75, 0.25]))
    np.testing.assert_array_equal(support_coverage[0, 0], np.asarray([3, 1]))
    np.testing.assert_allclose(
        support_xy[0, 0], np.asarray([[30.0, 40.0], [50.0, 60.0]])
    )
    np.testing.assert_array_equal(
        support_ids[1, 0], np.asarray(["map/a.png", "map/b.png"])
    )


def test_mixed_layout_split_validation_allows_target_free_test_points() -> None:
    assert validate_mixed_layout_exported_splits(np.asarray(["test"])) == ("test",)
    assert validate_mixed_layout_exported_splits(
        np.asarray(["train", "validation"])
    ) == ("train", "validation")
    with pytest.raises(ValueError, match="split"):
        validate_mixed_layout_exported_splits(np.asarray(["unknown"]))
