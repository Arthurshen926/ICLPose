from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_region_context_coverage import (
    frozen_candidate_ranks,
    summarize_region_context_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    profiles = FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES
    scores = np.ones((4, len(profiles)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    # This is a boundary-only layout bin.  The pool control must remain usable
    # while the layout family treats this real observation as unknown.
    layout_column = next(
        index
        for index, name in enumerate(profiles)
        if name.startswith("radio_final_")
        and "_region_" in name
        and name.endswith("_cosine")
    )
    valid[1, layout_column] = False
    scores[1, layout_column] = np.nan
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/region-context-fixture.npz"),),
        query_ids=np.asarray(["train.png", "validation.png"]),
        split_names=np.asarray(["train", "validation"]),
        source_row_indices=np.asarray([0, 0], dtype=np.int64),
        xy=np.zeros((2, 2), dtype=np.float32),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        # The stored column order is intentionally not posterior rank order.
        candidate_probabilities=np.asarray(
            [[0.3, 0.6], [0.3, 0.6]], dtype=np.float32
        ),
        null_probabilities=np.asarray([0.1, 0.1], dtype=np.float32),
        candidate_support_observation_counts=np.ones((2, 2), dtype=np.int64),
        edge_candidate_offsets=np.arange(5, dtype=np.int64),
        edge_geometry_rows=np.arange(4, dtype=np.int64),
        edge_profile_scores=scores,
        edge_profile_valid=valid,
        profile_names=profiles,
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_coverage_audit_uses_frozen_probability_rank_and_excludes_missing_layout() -> None:
    features = _features()
    ranks = frozen_candidate_ranks(features)
    assert ranks.tolist() == [[2, 1], [2, 1]]
    rows = summarize_region_context_coverage(
        features=features,
        families=(
            "fixedprior_fulltrack_perview_region_final_mixture",
            "fixedprior_fulltrack_perview_region_final_pool_mixture",
        ),
    )
    by_key = {(item["family"], item["split"], item["rank_bucket"]): item for item in rows}
    layout = by_key[
        (
            "fixedprior_fulltrack_perview_region_final_mixture",
            "train",
            "rank_1_5",
        )
    ]
    pool = by_key[
        (
            "fixedprior_fulltrack_perview_region_final_pool_mixture",
            "train",
            "rank_1_5",
        )
    ]
    assert layout["candidate_count"] == 2
    assert layout["candidate_with_usable_edge_count"] == 1
    assert layout["joint_edge_coverage_rate"] == 0.5
    assert pool["candidate_with_usable_edge_count"] == 2
    assert pool["joint_edge_coverage_rate"] == 1.0
