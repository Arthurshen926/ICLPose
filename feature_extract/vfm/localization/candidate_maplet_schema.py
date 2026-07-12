"""Shared feature schema for candidate-maplet training artifacts."""

from __future__ import annotations


CANDIDATE_MAPLET_STATIC_SCHEMA_VERSION = 1

CANDIDATE_MAPLET_STATIC_FEATURE_NAMES = (
    "coarse_score",
    "baseline_score",
    "coarse_rank_fraction",
    "baseline_rank_fraction",
    "coarse_gap_to_row_best",
    "baseline_gap_to_row_best",
    "query_detector_score",
    "query_detector_log10",
    "query_dispersion_log1p",
    "track_observation_count_log1p",
    "track_variance_log1p",
    "track_reprojection_error",
    "track_feature_ambiguity",
    "maplet_neighbor_fraction",
    "maplet_context_radius_log1p",
    "maplet_covisibility_log1p",
    "maplet_feature_variance_log1p",
)


# Query caches and proposal/feature artifacts intentionally differ when a
# trained matcher is applied to a new query set. Everything below defines the
# model's semantic and tensor-shape contract and must remain identical.
CANDIDATE_MAPLET_INFERENCE_COMPATIBILITY_KEYS = (
    "support_feature_cache_sha256",
    "support_geometry_index_sha256",
    "projected_landmark_bank_sha256",
    "maplet_support_index_sha256",
    "query_input_dim",
    "support_input_dim",
    "static_input_dim",
    "candidate_top_k",
    "static_feature_names",
    "positive_threshold_px",
    "assignment_threshold_px",
    "support_view_count",
    "query_radius_px",
    "max_query_nodes",
    "max_support_tracks",
)


def candidate_maplet_inference_manifest_mismatches(
    checkpoint_manifest: dict[str, object],
    inference_manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Compare only fields that must match across different query datasets."""

    return {
        key: {
            "checkpoint": checkpoint_manifest.get(key),
            "inference": inference_manifest.get(key),
        }
        for key in CANDIDATE_MAPLET_INFERENCE_COMPATIBILITY_KEYS
        if checkpoint_manifest.get(key) != inference_manifest.get(key)
    }


def validate_candidate_maplet_static_feature_names(
    feature_names: tuple[str, ...],
    *,
    count: int,
) -> tuple[str, ...]:
    requested = int(count)
    if requested <= 0 or requested > len(feature_names):
        raise ValueError("requested static feature count exceeds the artifact schema")
    shared_count = min(requested, len(CANDIDATE_MAPLET_STATIC_FEATURE_NAMES))
    expected = CANDIDATE_MAPLET_STATIC_FEATURE_NAMES[:shared_count]
    if tuple(feature_names[:shared_count]) != expected:
        raise ValueError("candidate static feature schema or field order differs")
    return tuple(feature_names[:requested])
