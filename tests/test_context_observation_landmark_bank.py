from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.context_observation_landmark_bank import (
    aggregate_normalized_observations,
    build_context_observation_landmark_index,
    canonical_track_rows,
    sample_spatial_context_descriptors,
    spatial_context_boundary_audit,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)
from feature_extract.vfm.localization.spatial_image_context import SpatialImageContextCache
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex


def _cache() -> SpatialImageContextCache:
    values = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return SpatialImageContextCache(
        image_ids=np.asarray(["a.png", "b.png"]),
        image_sizes=np.asarray([[11, 11], [11, 11]], dtype=np.int64),
        grids={2: values},
        metadata={
            "format": "test_context_v1",
            "pose_or_ground_truth_used": False,
            "spatial_grid_sizes": [2],
        },
    )


def _geometry() -> SupportObservationGeometryIndex:
    return SupportObservationGeometryIndex(
        image_ids=("a.png", "b.png"),
        image_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        source_row_indices=np.asarray([0, 1, 2], dtype=np.int64),
        track_ids=np.asarray([10, 11, 10], dtype=np.int64),
        xy=np.asarray([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float32),
        viewing_rays=np.ones((3, 3), dtype=np.float32),
        reprojection_errors=np.zeros((3,), dtype=np.float32),
    )


def _source() -> LandmarkMapIndex:
    return LandmarkMapIndex(
        track_ids=np.asarray([10, 11], dtype=np.int64),
        xyz=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.asarray([2, 1], dtype=np.int64),
        observation_image_ids=(("a.png", "b.png"), ("a.png",)),
    )


def test_context_observation_bank_samples_full_image_grid_and_preserves_tracks() -> None:
    cache = _cache()
    samples = sample_spatial_context_descriptors(
        cache,
        image_ids=np.asarray(["a.png", "b.png"]),
        xy=np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
        grid_size=2,
    )
    np.testing.assert_allclose(samples, [[1.0, 0.0], [1.0, 0.0]], atol=1e-6)
    assert canonical_track_rows(np.asarray([11, 10, 99]), _source().track_ids).tolist() == [1, 0, -1]

    index, summary = build_context_observation_landmark_index(
        cache=cache,
        geometry=_geometry(),
        source_landmark_index=_source(),
        support_image_ids=("a.png", "b.png"),
        grid_size=2,
    )
    assert index.track_ids.tolist() == [10, 11]
    assert index.observation_counts.tolist() == [2, 1]
    np.testing.assert_allclose(index.features[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(index.features[1], [0.0, 1.0], atol=1e-6)
    assert summary["sampled_observation_count"] == 3
    assert summary["excluded_observation_count"] == 0


def test_aggregate_normalized_observations_reports_track_dispersion() -> None:
    features, counts, variances = aggregate_normalized_observations(
        canonical_rows=np.asarray([0, 0, 1], dtype=np.int64),
        descriptors=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        landmark_count=2,
    )
    np.testing.assert_allclose(features[0], [2**-0.5, 2**-0.5], atol=1e-6)
    assert counts.tolist() == [2, 1]
    assert variances[0] > 0.2
    assert variances[1] == 0.0


def test_context_sampling_can_explicitly_match_border_clamped_source_semantics() -> None:
    cache = _cache()
    count, maximum = spatial_context_boundary_audit(
        cache,
        image_ids=np.asarray(["a.png"]),
        xy=np.asarray([[0.0, 10.25]], dtype=np.float32),
        grid_size=2,
    )
    assert count == 1
    assert maximum == 0.25
    sampled = sample_spatial_context_descriptors(
        cache,
        image_ids=np.asarray(["a.png"]),
        xy=np.asarray([[0.0, 10.25]], dtype=np.float32),
        grid_size=2,
        boundary_mode="border",
    )
    np.testing.assert_allclose(sampled, [[1.0, 0.0]], atol=1e-6)
