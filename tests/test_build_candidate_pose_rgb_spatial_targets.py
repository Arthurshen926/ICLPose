from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_targets import (
    spatial_supervision_from_correct_projections,
    spatial_supervision_from_registered_identity,
)
from feature_extract.vfm.localization.query_observation_identity import (
    RegisteredQueryObservationTargets,
)


def test_spatial_supervision_uses_correct_pose_projection_when_query_observations_are_excluded() -> None:
    offsets, non_dustbin, dustbin = spatial_supervision_from_correct_projections(
        projection_offsets_xy=np.asarray(
            [[[1.0, -2.0], [12.1, 0.0]], [[3.0, 2.0], [0.0, 0.0]]],
            dtype=np.float32,
        ),
        projection_valid=np.asarray([[True, True], [True, False]]),
        spatial_search_radius_px=8.0,
    )

    np.testing.assert_allclose(offsets[0, 0], [1.0, -2.0])
    assert non_dustbin.tolist() == [[True, False], [True, False]]
    assert dustbin.tolist() == [[False, True], [False, True]]


def test_registered_identity_spatial_supervision_rejects_nearby_wrong_candidates() -> None:
    offsets, observed, dustbin, supervised, audit = (
        spatial_supervision_from_registered_identity(
            projection_offsets_xy=np.asarray(
                [
                    [[1.0, 0.0], [0.5, 0.5], [2.0, 0.0]],
                    [[1.0, 1.0], [0.0, 0.0], [1.0, -1.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            projection_valid=np.ones((3, 3), dtype=bool),
            candidate_track_ids=np.asarray(
                [[11, 12, 13], [21, 22, 23], [31, 32, 33]], dtype=np.int64
            ),
            registered_targets=RegisteredQueryObservationTargets(
                track_ids=np.asarray([11, 99, -1], dtype=np.int64),
                distances_px=np.asarray([0.2, 0.4, np.inf], dtype=np.float32),
                supervised=np.asarray([True, True, False]),
            ),
            spatial_search_radius_px=4.0,
        )
    )

    np.testing.assert_allclose(offsets[0, 0], [1.0, 0.0])
    # Row zero has two geometrically nearby tracks, but only exact track 11 is
    # a local positive.  The other two are explicit candidate dustbins.
    assert observed.tolist() == [[True, False, False], [False, False, False], [False, False, False]]
    assert dustbin.tolist() == [[False, True, True], [True, True, True], [False, False, False]]
    assert supervised.tolist() == [[True, True, True], [True, True, True], [False, False, False]]
    assert audit["exact_identity_observed_candidate_count"] == 1
    assert audit["explicit_null_supervised_row_count"] == 1
