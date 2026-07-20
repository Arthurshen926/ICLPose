from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_aligned_layout_coverage import (
    summarize_aligned_layout_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    FrozenFulltrackPerViewAppearanceFeatures,
)
from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    ALIGNED_LAYOUT_FEATURE_NAMES,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    scores = np.ones((4, len(ALIGNED_LAYOUT_FEATURE_NAMES)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    # Missing DCT evidence must remain unavailable, not turn into a learned
    # low-score value or an implicit support-count cue.
    valid[0] = False
    scores[0] = np.nan
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/aligned-layout-fixture.npz"),),
        query_ids=np.asarray(["train.png", "validation.png"]),
        split_names=np.asarray(["train", "validation"]),
        source_row_indices=np.asarray([0, 0], dtype=np.int64),
        xy=np.zeros((2, 2), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_probabilities=np.asarray(
            [[0.3, 0.6], [0.3, 0.6]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        candidate_support_observation_counts=np.ones((2, 2), dtype=np.int64),
        edge_candidate_offsets=np.arange(5, dtype=np.int64),
        edge_geometry_rows=np.arange(4, dtype=np.int64),
        edge_profile_scores=scores,
        edge_profile_valid=valid,
        profile_names=ALIGNED_LAYOUT_FEATURE_NAMES,
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_aligned_layout_coverage_keeps_missing_dct_edges_neutral() -> None:
    rows = summarize_aligned_layout_coverage(
        features=_features(),
        families=("fixedprior_fulltrack_perview_aligned_layout_alike_nll",),
    )
    by_key = {(item["split"], item["rank_bucket"]): item for item in rows}
    train = by_key[("train", "rank_1_5")]
    assert train["candidate_count"] == 2
    assert train["candidate_with_usable_aligned_layout_edge_count"] == 1
    assert train["joint_edge_coverage_rate"] == 0.5
    assert train["candidate_prior_mass_coverage_rate"] == 0.6 / 0.9
