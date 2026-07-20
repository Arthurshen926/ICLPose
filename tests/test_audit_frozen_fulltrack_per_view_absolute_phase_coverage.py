from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_absolute_phase_coverage import (
    summarize_absolute_phase_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    names = FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES
    scores = np.ones((4, len(names)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    # A missing visual edge remains unknown.  The independently exported
    # position control must not make the visual family appear covered.
    visual_width = len(
        ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE["radio_final_absolute_phase"]
    )
    valid[0, :visual_width] = False
    scores[0, :visual_width] = np.nan
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/absolute-phase-fixture.npz"),),
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
        profile_names=names,
        artifact_metadata=({},),
        compatibility={
            "per_view_edge_feature_semantics": (
                FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_absolute_phase_coverage_keeps_visual_and_position_control_disjoint() -> None:
    rows = summarize_absolute_phase_coverage(
        features=_features(),
        families=(
            "fixedprior_fulltrack_perview_absolute_phase_final_mixture",
            "fixedprior_fulltrack_perview_absolute_phase_position_control_mixture",
        ),
    )
    by_key = {(row["family"], row["split"], row["rank_bucket"]): row for row in rows}
    visual = by_key[
        (
            "fixedprior_fulltrack_perview_absolute_phase_final_mixture",
            "train",
            "rank_1_5",
        )
    ]
    control = by_key[
        (
            "fixedprior_fulltrack_perview_absolute_phase_position_control_mixture",
            "train",
            "rank_1_5",
        )
    ]
    assert visual["candidate_count"] == 2
    assert visual["candidate_with_usable_absolute_phase_edge_count"] == 1
    assert visual["joint_edge_coverage_rate"] == 0.5
    assert control["candidate_with_usable_absolute_phase_edge_count"] == 2
    assert control["joint_edge_coverage_rate"] == 1.0
