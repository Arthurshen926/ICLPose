from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_multiscale_translation_mode_coverage import (
    summarize_multiscale_translation_mode_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    profile_names = FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES
    scores = np.ones((4, len(profile_names)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    # The stored candidate columns are deliberately not rank-sorted.  Missing
    # dense evidence remains unknown and must not masquerade as a low score.
    valid[0, :20] = False
    scores[0, :20] = np.nan
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/translation-mode-fixture.npz"),),
        query_ids=np.asarray(["train.png", "validation.png"]),
        split_names=np.asarray(["train", "validation"]),
        source_row_indices=np.asarray([0, 0], dtype=np.int64),
        xy=np.zeros((2, 2), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_probabilities=np.asarray([[0.3, 0.6], [0.3, 0.6]], dtype=np.float32),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        candidate_support_observation_counts=np.ones((2, 2), dtype=np.int64),
        edge_candidate_offsets=np.arange(5, dtype=np.int64),
        edge_geometry_rows=np.arange(4, dtype=np.int64),
        edge_profile_scores=scores,
        edge_profile_valid=valid,
        profile_names=profile_names,
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_translation_mode_coverage_uses_frozen_rank_and_keeps_missing_unknown() -> None:
    rows = summarize_multiscale_translation_mode_coverage(
        features=_features(),
        families=("fixedprior_fulltrack_perview_translation_mode_final_mixture",),
    )
    by_key = {(item["split"], item["rank_bucket"]): item for item in rows}
    train = by_key[("train", "rank_1_5")]
    assert train["candidate_count"] == 2
    assert train["candidate_with_usable_dense_mode_edge_count"] == 1
    assert train["joint_edge_coverage_rate"] == 0.5
    assert train["candidate_prior_mass_coverage_rate"] == 0.6 / 0.9
