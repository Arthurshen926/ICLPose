from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_maplet_support_view_overlay import (
    coverage_prefix_support_slots,
    remap_compact_support_view_probabilities,
)


def test_remap_compact_support_view_probabilities_preserves_original_columns() -> None:
    probabilities = remap_compact_support_view_probabilities(
        selected_rows=np.asarray([0, 1], dtype=np.int64),
        selected_columns=np.asarray([[2, 0, 1], [1, 2, 0]], dtype=np.int64),
        compact_probabilities=np.asarray(
            [
                [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]],
                [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]],
            ],
            dtype=np.float32,
        ),
        proposal_row_count=2,
        candidate_count=3,
    )
    np.testing.assert_allclose(
        probabilities,
        np.asarray(
            [
                [[0.3, 0.7], [0.4, 0.6], [0.2, 0.8]],
                [[0.7, 0.3], [0.5, 0.5], [0.6, 0.4]],
            ],
            dtype=np.float32,
        ),
    )


def test_coverage_prefix_support_slots_cycles_only_when_a_track_lacks_second_view() -> None:
    ids, valid = coverage_prefix_support_slots(
        candidate_track_ids=np.asarray([[10, 20]], dtype=np.int64),
        maplet_track_ids=np.asarray([10, 20], dtype=np.int64),
        support_image_ids=np.asarray(["a.png", "b.png", "c.png"]),
        support_image_indices=np.asarray([[0, 1, 2], [2, -1, -1]], dtype=np.int64),
        support_coverage_counts=np.asarray([[5, 4, 3], [2, 0, 0]], dtype=np.int32),
        view_count=2,
    )
    assert valid.all()
    assert ids.tolist() == [[["a.png", "b.png"], ["c.png", "c.png"]]]


def test_coverage_prefix_support_slots_rejects_missing_support_view() -> None:
    with pytest.raises(ValueError, match="no valid"):
        coverage_prefix_support_slots(
            candidate_track_ids=np.asarray([[10]], dtype=np.int64),
            maplet_track_ids=np.asarray([10], dtype=np.int64),
            support_image_ids=np.asarray(["a.png"]),
            support_image_indices=np.asarray([[-1, -1]], dtype=np.int64),
            support_coverage_counts=np.asarray([[0, 0]], dtype=np.int32),
            view_count=2,
        )
