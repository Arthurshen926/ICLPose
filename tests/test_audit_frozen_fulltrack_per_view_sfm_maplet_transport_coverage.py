from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_sfm_maplet_transport_coverage import (
    MapletEdgeDiagnostics,
    summarize_sfm_maplet_transport_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES,
    FrozenFulltrackPerViewAppearanceFeatures,
)
from feature_extract.vfm.localization.frozen_fulltrack_sfm_maplet_transport import (
    SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SFM_MAPLET_TRANSPORT_PROFILES,
)


def _features_and_diagnostics() -> tuple[
    FrozenFulltrackPerViewAppearanceFeatures, MapletEdgeDiagnostics
]:
    profile_names = FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES
    scores = np.ones((4, len(profile_names)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    # First train candidate has no final-near maplet.  The value is unknown,
    # not a zero score, and its rank is intentionally two rather than column 1.
    final_near = SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[
        "radio_final_sfm_maplet_near"
    ]
    columns = [profile_names.index(name) for name in final_near]
    valid[0, columns] = False
    scores[0, columns] = np.nan
    features = FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/sfm-maplet-fixture.npz"),),
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
                FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
            )
        },
    )
    usable = np.ones((4, len(SFM_MAPLET_TRANSPORT_PROFILES)), dtype=bool)
    usable[0, 0] = False
    diagnostics = MapletEdgeDiagnostics(
        profile_usable=usable,
        support_quadrant_counts=np.ones((4, len(SFM_MAPLET_TRANSPORT_PROFILES), 4), dtype=np.int64),
    )
    return features, diagnostics


def test_maplet_coverage_uses_frozen_ranks_and_treats_missing_maplets_as_unknown() -> None:
    features, diagnostics = _features_and_diagnostics()
    rows = summarize_sfm_maplet_transport_coverage(
        features=features,
        diagnostics=diagnostics,
        families=("fixedprior_fulltrack_perview_sfm_maplet_final_near_mixture",),
    )
    by_key = {(item["split"], item["rank_bucket"]): item for item in rows}
    train = by_key[("train", "rank_1_5")]
    assert train["candidate_count"] == 2
    assert train["candidate_with_usable_maplet_edge_count"] == 1
    assert train["joint_edge_coverage_rate"] == 0.5
    assert train["all_support_quadrants_present_rate"] == 1.0


def test_maplet_audit_diagnostics_reject_invalid_quadrant_shape() -> None:
    try:
        MapletEdgeDiagnostics(
            profile_usable=np.ones((2, len(SFM_MAPLET_TRANSPORT_PROFILES)), dtype=bool),
            support_quadrant_counts=np.ones((2, len(SFM_MAPLET_TRANSPORT_PROFILES), 3), dtype=np.int64),
        )
    except ValueError as error:
        assert "diagnostics" in str(error)
    else:  # pragma: no cover - explicit failure branch for a shape invariant
        raise AssertionError("invalid SfM-maplet diagnostic shape was accepted")
