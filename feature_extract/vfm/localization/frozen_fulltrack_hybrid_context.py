"""Contracts for a virtual per-view local-plus-absolute appearance probe.

The two source artifacts are produced from the same immutable full-track CSR:
translation modes retain anchor-centred multiscale structure, while the
intermediate absolute-phase profile retains coarse image-coordinate context.
The hybrid is deliberately a manifest, not a copied feature cache.  A loader
must prove the two components have identical frozen rows, candidates, priors,
and support-observation edges before their visual columns can be concatenated.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE,
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES,
)


FULLTRACK_HYBRID_CONTEXT_MANIFEST_FORMAT = (
    "frozen_fulltrack_candidate_per_view_hybrid_context_manifest_v1"
)
FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS = (
    "multiscale_translation_plus_absolute_intermediate_per_real_sfm_observation_v1"
)
FULLTRACK_HYBRID_CONTEXT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_multiscale_translation_plus_absolute_intermediate_v1"
)
HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES = tuple(MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES)
HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES = tuple(
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[
        "radio_intermediate_pca256_absolute_phase"
    ]
)
HYBRID_CONTEXT_PROFILE_NAMES = (
    *HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
    *HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
)

HYBRID_SHARED_ARRAY_KEYS = (
    "verification_query_ids",
    "split_names",
    "verification_source_row_indices",
    "verification_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "candidate_support_observation_counts",
    "edge_candidate_offsets",
    "edge_geometry_rows",
)
HYBRID_SHARED_LINEAGE_KEYS = (
    "source_frozen_appearance_artifact",
    "source_frozen_appearance_artifact_sha256",
    "source_fulltrack_per_view_artifact",
    "source_fulltrack_per_view_artifact_sha256",
    "source_edge_candidate_offsets_sha256",
    "source_edge_geometry_rows_sha256",
    "source_candidate_tracks_sha256",
    "source_candidate_probabilities_sha256",
    "source_null_probabilities_sha256",
    "source_verification_rows_sha256",
    "support_geometry_index_sha256",
)


def hybrid_component_query_id(
    arrays: Mapping[str, np.ndarray], *, context: str
) -> str:
    """Return the only query id accepted for one complete frozen shard."""

    ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    if len(ids) != 192 or len(set(ids.tolist())) != 1:
        raise ValueError(f"{context}: hybrid component is not one complete query shard")
    return str(ids[0])


def _validate_component_contract(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    context: str,
    expected_format: str,
    expected_semantics: str,
    expected_profiles: tuple[str, ...],
) -> None:
    if (
        metadata.get("format") != expected_format
        or metadata.get("per_view_edge_feature_semantics") != expected_semantics
        or tuple(np.asarray(arrays["profile_names"]).astype(str).tolist())
        != expected_profiles
    ):
        raise ValueError(f"{context}: hybrid component contract differs")


def validate_hybrid_component_pair(
    *,
    translation_arrays: Mapping[str, np.ndarray],
    translation_metadata: Mapping[str, Any],
    translation_context: str,
    absolute_arrays: Mapping[str, np.ndarray],
    absolute_metadata: Mapping[str, Any],
    absolute_context: str,
) -> str:
    """Ensure local and absolute components are views of the same frozen CSR."""

    _validate_component_contract(
        arrays=translation_arrays,
        metadata=translation_metadata,
        context=translation_context,
        expected_format=FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
        expected_semantics=FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
        expected_profiles=tuple(MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES),
    )
    _validate_component_contract(
        arrays=absolute_arrays,
        metadata=absolute_metadata,
        context=absolute_context,
        expected_format=FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
        expected_semantics=FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
        expected_profiles=tuple(FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES),
    )
    query_id = hybrid_component_query_id(
        translation_arrays, context=translation_context
    )
    if query_id != hybrid_component_query_id(absolute_arrays, context=absolute_context):
        raise ValueError("hybrid components refer to different query shards")
    for name in HYBRID_SHARED_ARRAY_KEYS:
        if not np.array_equal(translation_arrays[name], absolute_arrays[name]):
            raise ValueError(f"{query_id}: hybrid CSR differs for {name}")
    for name in HYBRID_SHARED_LINEAGE_KEYS:
        translation_value = translation_metadata.get(name)
        absolute_value = absolute_metadata.get(name)
        # Equal ``None`` values would make two unrelated exports look
        # compatible.  Every field below is part of the immutable source
        # lineage, so absence is a contract violation rather than a legacy
        # compatibility case.
        if (
            not isinstance(translation_value, str)
            or not translation_value.strip()
            or not isinstance(absolute_value, str)
            or not absolute_value.strip()
        ):
            raise ValueError(f"{query_id}: hybrid component lacks lineage for {name}")
        if translation_value != absolute_value:
            raise ValueError(f"{query_id}: hybrid component lineage differs for {name}")
    return query_id
