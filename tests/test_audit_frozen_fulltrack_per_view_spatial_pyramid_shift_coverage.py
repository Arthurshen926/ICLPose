from __future__ import annotations

from pathlib import Path

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_per_view_spatial_pyramid_shift_coverage import (
    _StreamingArtifactHeader,
    _stream_spatial_pyramid_shift_coverage,
    summarize_spatial_pyramid_shift_coverage,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME,
    FrozenFulltrackPerViewAppearanceFeatures,
)


def _features() -> FrozenFulltrackPerViewAppearanceFeatures:
    profile_names = FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES
    scores = np.ones((4, len(profile_names)), dtype=np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    intermediate = FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME[
        "radio_intermediate_pca256_spatial_pyramid_context9_shift2_bins3"
    ]
    # The first stored column is rank two.  Missing full-crop visual evidence
    # must remain unknown rather than being treated as a low appearance score.
    valid[0, : len(intermediate)] = False
    scores[0, : len(intermediate)] = np.nan
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(Path("/tmp/spatial-pyramid-fixture.npz"),),
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
                FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
            )
        },
    )


def test_spatial_pyramid_coverage_uses_frozen_rank_and_keeps_missing_unknown() -> None:
    rows = summarize_spatial_pyramid_shift_coverage(
        features=_features(),
        families=(
            "fixedprior_fulltrack_perview_spatial_pyramid_intermediate_pca256_mixture",
        ),
    )
    by_key = {(item["split"], item["rank_bucket"]): item for item in rows}
    train = by_key[("train", "rank_1_5")]
    assert train["candidate_count"] == 2
    assert train["candidate_with_usable_spatial_pyramid_edge_count"] == 1
    assert train["joint_edge_coverage_rate"] == 0.5
    assert train["candidate_prior_mass_coverage_rate"] == 0.6 / 0.9


def test_spatial_pyramid_mask_control_uses_a_separate_family() -> None:
    visual = _features()
    control_scores = np.repeat(
        np.nan_to_num(visual.edge_profile_scores[:, :1], nan=0.0),
        len(FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES),
        axis=1,
    )
    control = FrozenFulltrackPerViewAppearanceFeatures(
        **{
            **visual.__dict__,
            "edge_profile_scores": control_scores,
            "edge_profile_valid": np.ones(control_scores.shape, dtype=bool),
            "profile_names": FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
            "compatibility": {
                "per_view_edge_feature_semantics": (
                    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
                )
            },
        }
    )
    rows = summarize_spatial_pyramid_shift_coverage(
        features=control,
        families=(
            "fixedprior_fulltrack_perview_spatial_pyramid_mask_control_intermediate_pca256_mixture",
        ),
        artifact_role="mask_control",
    )
    assert rows
    assert all(item["profile_feature_count"] > 0 for item in rows)


def test_streaming_coverage_reads_only_edge_validity(tmp_path: Path) -> None:
    profile_names = FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES
    intermediate = FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME[
        "radio_intermediate_pca256_spatial_pyramid_context9_shift2_bins3"
    ]
    valid = np.ones((4, len(profile_names)), dtype=bool)
    valid[0, : len(intermediate)] = False
    path = tmp_path / "streaming-validity-only.npz"
    # Deliberately omit edge_profile_scores. The streaming path has no reason
    # to load them for a target-free validity/coverage audit.
    np.savez_compressed(path, edge_profile_valid=valid)
    header = _StreamingArtifactHeader(
        path=path,
        metadata={},
        compatibility={},
        profile_names=profile_names,
        query_ids=np.asarray(["train.png", "validation.png"]),
        split_names=np.asarray(["train", "validation"]),
        source_row_indices=np.asarray([0, 0], dtype=np.int64),
        candidate_track_ids=np.asarray([[10, 11], [12, 13]], dtype=np.int64),
        candidate_probabilities=np.asarray(
            [[0.3, 0.6], [0.3, 0.6]], dtype=np.float32
        ),
        candidate_support_observation_counts=np.ones((2, 2), dtype=np.int64),
        edge_candidate_offsets=np.arange(5, dtype=np.int64),
        edge_count=4,
    )
    rows = _stream_spatial_pyramid_shift_coverage(
        headers=(header,),
        families=(
            "fixedprior_fulltrack_perview_spatial_pyramid_intermediate_pca256_mixture",
        ),
        artifact_role="visual",
    )
    by_key = {(item["split"], item["rank_bucket"]): item for item in rows}
    train = by_key[("train", "rank_1_5")]
    assert train["candidate_count"] == 2
    assert train["candidate_with_usable_spatial_pyramid_edge_count"] == 1
    assert train["candidate_prior_mass_coverage_rate"] == 0.6 / 0.9
