from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_radio_intermediate_context_cache import (
    _reindex_support_descriptors,
)


class _Geometry:
    def __init__(self, image_ids, offsets, source_rows, tracks, xy):
        self.image_ids = tuple(image_ids)
        self.image_offsets = np.asarray(offsets, dtype=np.int64)
        self.source_row_indices = np.asarray(source_rows, dtype=np.int64)
        self.track_ids = np.asarray(tracks, dtype=np.int64)
        self.xy = np.asarray(xy, dtype=np.float32)


def test_support_reindex_selects_exact_observation_subset() -> None:
    source = _Geometry(
        ["a", "b"],
        [0, 2, 4],
        [0, 1, 2, 3],
        [10, 11, 20, 21],
        [[1, 1], [2, 2], [3, 3], [4, 4]],
    )
    target = _Geometry(
        ["b"], [0, 2], [1, 0], [20, 21], [[3, 3], [4, 4]]
    )
    descriptors = np.asarray(
        [[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32
    )

    output, audit = _reindex_support_descriptors(
        source_descriptors=descriptors,
        source_feature_track_ids=np.asarray([10, 11, 20, 21]),
        source_geometry=source,
        target_feature_track_ids=np.asarray([21, 20]),
        target_geometry=target,
    )

    assert output.tolist() == [[0.0, 2.0], [2.0, 0.0]]
    assert audit["matched_observation_count"] == 2


def test_support_reindex_rejects_coordinate_drift() -> None:
    source = _Geometry(["a"], [0, 1], [0], [10], [[1, 1]])
    target = _Geometry(["a"], [0, 1], [0], [10], [[1.1, 1]])
    with pytest.raises(ValueError, match="coordinates changed"):
        _reindex_support_descriptors(
            source_descriptors=np.asarray([[1, 0]], dtype=np.float32),
            source_feature_track_ids=np.asarray([10]),
            source_geometry=source,
            target_feature_track_ids=np.asarray([10]),
            target_geometry=target,
        )


def test_support_reindex_can_leave_missing_images_for_targeted_extraction() -> None:
    source = _Geometry(["a"], [0, 1], [0], [10], [[1, 1]])
    target = _Geometry(
        ["a", "b"], [0, 1, 2], [0, 1], [10, 20], [[1, 1], [2, 2]]
    )
    output, audit = _reindex_support_descriptors(
        source_descriptors=np.asarray([[1, 0]], dtype=np.float32),
        source_feature_track_ids=np.asarray([10]),
        source_geometry=source,
        target_feature_track_ids=np.asarray([10, 20]),
        target_geometry=target,
        allow_missing=True,
    )
    assert np.isfinite(output[0]).all()
    assert np.isnan(output[1]).all()
    assert audit["missing_observation_count"] == 1
    assert audit["missing_images"] == ["b"]
