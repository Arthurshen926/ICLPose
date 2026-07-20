from __future__ import annotations

import numpy as np
import pytest

from feature_extract.vfm.localization.full_track_support_view_probe import (
    FULL_TRACK_VIEW_STATISTIC_NAMES,
    aggregate_full_track_view_scores,
    build_full_track_candidate_edges,
    build_track_observation_lookup,
)
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)


def _geometry() -> SupportObservationGeometryIndex:
    # Geometry is deliberately image-major rather than track-major.
    return SupportObservationGeometryIndex(
        image_ids=("a.png", "b.png", "c.png"),
        image_offsets=np.asarray([0, 2, 3, 4], dtype=np.int64),
        source_row_indices=np.arange(4, dtype=np.int64),
        track_ids=np.asarray([10, 20, 10, 30], dtype=np.int64),
        xy=np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32),
        viewing_rays=np.ones((4, 3), dtype=np.float32),
        reprojection_errors=np.zeros((4,), dtype=np.float32),
    )


def test_all_positive_candidates_expand_to_every_real_track_observation() -> None:
    lookup = build_track_observation_lookup(_geometry())
    assert lookup.track_ids.tolist() == [10, 20, 30]
    assert lookup.observation_counts.tolist() == [2, 1, 1]
    edges = build_full_track_candidate_edges(
        candidate_track_ids=np.asarray([[10, 20], [30, -1]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.4, 0.5], [0.1, 0.0]], dtype=np.float32),
        lookup=lookup,
    )
    assert edges.candidate_observation_counts.tolist() == [[2, 1], [1, 0]]
    assert edges.edge_candidate_indices.tolist() == [0, 0, 1, 2]
    # Track 10's observations remain in their original image-major geometry
    # rows; no maplet coverage ranking or support cap was applied.
    assert edges.geometry_rows.tolist() == [0, 2, 1, 3]


def test_full_track_summary_keeps_missing_evidence_explicit() -> None:
    lookup = build_track_observation_lookup(_geometry())
    edges = build_full_track_candidate_edges(
        candidate_track_ids=np.asarray([[10, 20]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.5, 0.4]], dtype=np.float32),
        lookup=lookup,
    )
    summary = aggregate_full_track_view_scores(
        edges=edges,
        scores=np.asarray([[0.2, np.nan], [0.8, np.nan], [0.4, np.nan]], dtype=np.float32),
        usable=np.asarray([[True, False], [True, False], [True, False]]),
    )
    assert tuple(summary.statistics) == FULL_TRACK_VIEW_STATISTIC_NAMES
    np.testing.assert_allclose(summary.statistics["uniform_mean_ncc"][0, 0, 0], 0.5)
    np.testing.assert_allclose(summary.statistics["uniform_max_ncc"][0, 0, 0], 0.8)
    np.testing.assert_allclose(summary.statistics["uniform_top4_mean_ncc"][0, 0, 0], 0.5)
    assert summary.usable_counts[0, 0, 0] == 2
    assert summary.usable_fractions[0, 0, 0] == 1.0
    # The second feature never had usable view evidence and remains unknown,
    # rather than becoming a zero or a negative score.
    assert summary.usable_counts[..., 1].sum() == 0
    assert np.isnan(summary.statistics["uniform_mean_ncc"][..., 1]).all()


def test_positive_candidate_missing_from_geometry_is_rejected() -> None:
    lookup = build_track_observation_lookup(_geometry())
    with pytest.raises(ValueError, match="absent from support geometry"):
        build_full_track_candidate_edges(
            candidate_track_ids=np.asarray([[99]], dtype=np.int64),
            candidate_probabilities=np.asarray([[1.0]], dtype=np.float32),
            lookup=lookup,
        )
