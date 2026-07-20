from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_frozen_loftr_anchor_coverage import (
    rank_band_row_statistics,
)


def test_rank_band_coverage_preserves_fixed_posterior_mass_without_renormalizing() -> None:
    probabilities = np.asarray([[0.30, 0.20, 0.10, 0.10]], dtype=np.float32)
    weights = np.ones((1, 4, 2), dtype=np.float32) * 0.5
    usable = np.asarray([[[True, False], [True, True], [False, False], [True, False]]])
    counts = np.where(usable, 40, 0).astype(np.int32)
    usable_mass = (weights * usable).sum(axis=2)

    stats = rank_band_row_statistics(
        candidate_probabilities=probabilities,
        candidate_view_weights=weights,
        candidate_view_usable=usable,
        candidate_view_pair_match_counts=counts,
        candidate_usable_view_weight_mass=usable_mass,
        start=0,
        stop=4,
    )

    # The raw top-L mass is .7; only .3*.5 + .2 + .1*.5=.4 is observed.
    # Coverage is therefore .4/.7, never a re-softmaxed candidate posterior.
    assert stats["prior_mass"][0] == pytest.approx(0.7)
    assert stats["observed_mass"][0] == pytest.approx(0.4)
    assert stats["coverage"][0] == pytest.approx(0.4 / 0.7)
    assert stats["candidate_any_usable"].tolist() == [True]


def test_rank_band_coverage_rejects_inconsistent_view_mass() -> None:
    probabilities = np.full((1, 2), 0.25, dtype=np.float32)
    weights = np.full((1, 2, 1), 1.0, dtype=np.float32)
    usable = np.ones((1, 2, 1), dtype=bool)
    with pytest.raises(ValueError, match="invalid"):
        rank_band_row_statistics(
            candidate_probabilities=probabilities,
            candidate_view_weights=weights,
            candidate_view_usable=usable,
            candidate_view_pair_match_counts=np.ones((1, 2, 1), dtype=np.int32),
            candidate_usable_view_weight_mass=np.zeros((1, 2), dtype=np.float32),
            start=0,
            stop=2,
        )
