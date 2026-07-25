from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.tools.vfm.expand_candidate_pose_rgb_spatial_support_views import (
    _lookup_source_rows,
    _validate_reference_support_prefix,
)


def _reference() -> SimpleNamespace:
    return SimpleNamespace(
        row_count=1,
        candidate_count=2,
        support_view_count=1,
        support_image_ids=np.asarray([[['a.png'], ['b.png']]]),
        support_xy=np.asarray([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32),
        support_view_valid=np.asarray([[[True], [True]]]),
        support_coverage_counts=np.asarray([[[4], [5]]], dtype=np.int32),
    )


def test_source_lookup_preserves_requested_row_order() -> None:
    rows = _lookup_source_rows(
        source_row_indices=np.asarray([12, 5, 9], dtype=np.int64),
        requested_source_ids=np.asarray([9, 12, 5], dtype=np.int64),
    )
    assert rows.tolist() == [2, 0, 1]
    with pytest.raises(ValueError, match="absent"):
        _lookup_source_rows(
            source_row_indices=np.asarray([12, 5, 9], dtype=np.int64),
            requested_source_ids=np.asarray([7], dtype=np.int64),
        )


def test_support_prefix_uses_view_axis_not_xy_axis() -> None:
    reference = _reference()
    ids = np.asarray([[['a.png', 'a2.png'], ['b.png', 'b2.png']]])
    xy = np.asarray(
        [[[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [30.0, 40.0]]]],
        dtype=np.float32,
    )
    valid = np.asarray([[[True, True], [True, True]]])
    coverage = np.asarray([[[4, 2], [5, 1]]], dtype=np.int32)
    _validate_reference_support_prefix(
        reference=reference,
        support_image_ids=ids,
        support_xy=xy,
        support_view_valid=valid,
        support_coverage_counts=coverage,
    )

    xy[0, 1, 0, 1] = 999.0
    with pytest.raises(ValueError, match="support_xy prefix"):
        _validate_reference_support_prefix(
            reference=reference,
            support_image_ids=ids,
            support_xy=xy,
            support_view_valid=valid,
            support_coverage_counts=coverage,
        )
