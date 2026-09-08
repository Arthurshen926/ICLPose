import numpy as np
import pytest

from feature_extract.tools.vfm.evaluate_goal_maplet_multiplane_layout_pnp import (
    _layout_pair_groups,
    _unsigned_angle_deg,
)


def test_unsigned_angle_is_sign_and_rotation_invariant():
    left = np.asarray([1.0, 0.0, 0.0])
    right = np.asarray([0.0, 1.0, 0.0])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert _unsigned_angle_deg(left, right) == pytest.approx(90.0)
    assert _unsigned_angle_deg(-(rotation @ left), rotation @ right) == pytest.approx(90.0)


def test_layout_pairs_rank_exact_relative_angle_first():
    provenance = np.asarray([[0, 0, 0]] * 3 + [[1, 1, 0]] * 3 + [[1, 2, 0]] * 3)
    tokens = np.arange(9)
    radio = np.r_[np.ones(3), np.ones(3), np.full(3, 10.0)]
    query = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    maps = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    pairs = _layout_pair_groups(provenance, tokens, radio, query, maps)
    assert len(pairs) == 2
    assert pairs[0]["left_plane"] == 0 and pairs[0]["right_plane"] == 1
    assert pairs[0]["angle_error_deg"] == pytest.approx(0.0)


def test_layout_pairs_reject_nonfinite_evidence():
    with pytest.raises(ValueError, match="nonfinite"):
        _layout_pair_groups(
            np.asarray([[0, 0, 0]] * 6), np.arange(6), np.full(6, np.nan),
            np.asarray([[1.0, 0.0, 0.0]]), np.asarray([[1.0, 0.0, 0.0]]),
        )
