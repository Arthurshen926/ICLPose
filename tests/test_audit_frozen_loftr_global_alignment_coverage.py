from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.audit_frozen_loftr_global_alignment_coverage import (
    _accumulate_rank_buckets,
    _coverage_rows,
    _empty_totals,
    RANK_BUCKETS,
)


def test_coverage_uses_fixed_candidate_rank_columns_and_never_renormalizes_mass() -> None:
    totals = {
        (split, bucket): _empty_totals()
        for split in ("train", "validation")
        for bucket, _start, _stop in RANK_BUCKETS
    }
    probabilities = np.asarray([[0.1, 0.2, 0.0] + [0.0] * 17], dtype=np.float32)
    weights = np.zeros((1, 20, 2), dtype=np.float32)
    weights[0, :2] = 0.5
    usable = np.zeros_like(weights, dtype=bool)
    usable[0, 0, 0] = True
    model_valid = usable.copy()

    _accumulate_rank_buckets(
        totals=totals,
        split="train",
        probabilities=probabilities,
        weights=weights,
        usable=usable,
        model_valid=model_valid,
    )
    rows = {row["rank_bucket"]: row for row in _coverage_rows(totals) if row["split"] == "train"}

    assert rows["rank1_5"]["candidate_count"] == 2
    assert rows["rank1_5"]["candidate_prior_mass"] == 0.30000000447034836
    assert rows["rank1_5"]["candidate_prior_mass_covered"] == 0.10000000149011612
    assert rows["rank1_5"]["candidate_prior_mass_coverage"] == 1.0 / 3.0
    assert rows["rank6_10"]["candidate_count"] == 0
