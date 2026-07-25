from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_hard_pose_pairs import (
    _inside_patch,
    _sample_nonzero_jitter,
)


def test_hard_pose_anchor_jitter_is_deterministic_and_never_center_only() -> None:
    first = _sample_nonzero_jitter(
        count=16, radius_px=4, generator=np.random.default_rng(23)
    )
    second = _sample_nonzero_jitter(
        count=16, radius_px=4, generator=np.random.default_rng(23)
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (16, 2)
    assert not np.any(np.all(first == 0.0, axis=1))
    assert np.max(np.abs(first)) <= 4.0


def test_hard_pose_patch_bounds_rejects_jittered_border_anchor() -> None:
    inside = _inside_patch(
        xy=np.asarray([[20.0, 20.0], [1003.0, 555.0], [19.9, 20.0]], dtype=np.float32),
        width=1024,
        height=576,
        radius_px=20.0,
    )
    np.testing.assert_array_equal(inside, [True, True, False])
    with pytest.raises(ValueError, match="bounds"):
        _inside_patch(xy=np.zeros((3, 3), dtype=np.float32), width=1024, height=576, radius_px=20.0)
