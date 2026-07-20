"""Fixed-mass per-view appearance probes for frozen full-track data.

This is intentionally an S1 diagnostic rather than a production identity or
pose model.  Each real SfM support observation remains a separate edge until a
learned, normalized per-candidate view mixture marginalizes it.  The module
also contains a deliberately constrained raw top-4 overlay: it pools every
real support observation deterministically and can only compare a candidate
with the frozen top-one candidate.  Both paths keep missing observations
neutral rather than making availability an identity feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.fulltrack_aligned_layout_probe import (
    ALIGNED_LAYOUT_ALIKE_FEATURE_NAMES,
    ALIGNED_LAYOUT_FEATURE_NAMES,
    ALIGNED_LAYOUT_RADIO_FINAL_FEATURE_NAMES,
    ALIGNED_LAYOUT_RADIO_INTERMEDIATE_FEATURE_NAMES,
    FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT,
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ALIGNED_LAYOUT_FEATURE_GRANULARITY,
)
from feature_extract.vfm.localization.multiscale_candidate_probe import (
    MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES,
    MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_sfm_maplet_transport import (
    FULLTRACK_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
    FULLTRACK_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY,
    SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SFM_MAPLET_TRANSPORT_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_sparse_maplet_transport import (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT,
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_GRANULARITY,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_FEATURE_GRANULARITY,
    SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
    SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
    SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE,
    SPARSE_MAPLET_TRANSPORT_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_multiscale_translation_mode import (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY,
    MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE,
    MULTISCALE_TRANSLATION_MODE_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_spatial_pyramid_shift import (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY,
    SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE,
    SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE,
    SPATIAL_PYRAMID_SHIFT_PROFILES,
)
from feature_extract.vfm.localization.frozen_fulltrack_absolute_phase import (
    ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE,
    ABSOLUTE_PHASE_VISUAL_PROFILES,
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_ABSOLUTE_PHASE_FEATURE_GRANULARITY,
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES,
)
from feature_extract.vfm.localization.frozen_fulltrack_hybrid_context import (
    FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS,
    FULLTRACK_HYBRID_CONTEXT_FEATURE_GRANULARITY,
    FULLTRACK_HYBRID_CONTEXT_MANIFEST_FORMAT,
    HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
    HYBRID_CONTEXT_PROFILE_NAMES,
    HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
    hybrid_component_query_id,
    validate_hybrid_component_pair,
)


FULLTRACK_PER_VIEW_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_appearance_v1"
)
FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS = (
    "raw_aligned_ncc_per_real_sfm_observation_v1"
)
FULLTRACK_PER_VIEW_RAW_NCC_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_raw_ncc_v1"
)
FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_global_context_v1"
)
FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS = (
    "full_image_radio_final_cosine_per_real_sfm_observation_v1"
)
FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_radio_final_global_context_v1"
)
FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES = (
    "radio_final_global",
    "radio_final_summary",
)
FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_multisource_region_context_v1"
)
FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS = (
    "multisource_landmark_centered_region_cosine_per_real_sfm_observation_v1"
)
FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_multisource_landmark_centered_region_context_v1"
)
FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_PROFILE_NAMES = tuple(
    MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES
)
FULLTRACK_PER_VIEW_REGION_FINAL_LAYOUT_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES
    if not name.endswith("_common_cell_fraction")
)
FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_LAYOUT_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES
    if not name.endswith("_common_cell_fraction")
)
FULLTRACK_PER_VIEW_REGION_ALIKE_LAYOUT_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES
    if not name.endswith("_common_cell_fraction")
)
FULLTRACK_PER_VIEW_REGION_MULTISOURCE_LAYOUT_PROFILE_NAMES = (
    *FULLTRACK_PER_VIEW_REGION_FINAL_LAYOUT_PROFILE_NAMES,
    *FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_LAYOUT_PROFILE_NAMES,
    *FULLTRACK_PER_VIEW_REGION_ALIKE_LAYOUT_PROFILE_NAMES,
)
FULLTRACK_PER_VIEW_REGION_FINAL_POOL_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES
    if name.endswith("_pool_cosine")
)
FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_POOL_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES
    if name.endswith("_pool_cosine")
)
FULLTRACK_PER_VIEW_REGION_ALIKE_POOL_PROFILE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES
    if name.endswith("_pool_cosine")
)
FULLTRACK_PER_VIEW_REGION_MULTISOURCE_POOL_PROFILE_NAMES = (
    *FULLTRACK_PER_VIEW_REGION_FINAL_POOL_PROFILE_NAMES,
    *FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_POOL_PROFILE_NAMES,
    *FULLTRACK_PER_VIEW_REGION_ALIKE_POOL_PROFILE_NAMES,
)
FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT = (
    FULLTRACK_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY = (
    FULLTRACK_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES = tuple(
    name
    for profile in SFM_MAPLET_TRANSPORT_PROFILES
    for name in SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in SFM_MAPLET_TRANSPORT_PROFILES
}
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT = (
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_FEATURE_GRANULARITY = (
    FULLTRACK_SPARSE_MAPLET_TRANSPORT_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES = tuple(
    name
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    for name in SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
}
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT = (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_GRANULARITY = (
    FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES = tuple(
    name
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    for name in SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(
        SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
    )
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
}
FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT = (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY = (
    FULLTRACK_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES = tuple(
    name
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
    for name in MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
}
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES = tuple(
    name
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
    for name in SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
}
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY = (
    FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES = tuple(
    name
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
    for name in SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE[
        profile.name
    ]
)
FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(
        SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
    )
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
}
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_APPEARANCE_FORMAT = (
    FULLTRACK_ABSOLUTE_PHASE_APPEARANCE_FORMAT
)
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_FEATURE_GRANULARITY = (
    FULLTRACK_ABSOLUTE_PHASE_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_PROFILE_NAMES = tuple(
    FULLTRACK_ABSOLUTE_PHASE_PROFILE_NAMES
)
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_VISUAL_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(ABSOLUTE_PHASE_VISUAL_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
}
FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_POSITION_PROFILE_NAMES_BY_NAME = {
    profile.name: tuple(ABSOLUTE_PHASE_POSITION_FEATURE_NAMES_BY_PROFILE[profile.name])
    for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
}
FULLTRACK_PER_VIEW_HYBRID_CONTEXT_APPEARANCE_FORMAT = (
    FULLTRACK_HYBRID_CONTEXT_MANIFEST_FORMAT
)
FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS = (
    FULLTRACK_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS
)
FULLTRACK_PER_VIEW_HYBRID_CONTEXT_FEATURE_GRANULARITY = (
    FULLTRACK_HYBRID_CONTEXT_FEATURE_GRANULARITY
)
FULLTRACK_PER_VIEW_HYBRID_CONTEXT_PROFILE_NAMES = tuple(HYBRID_CONTEXT_PROFILE_NAMES)
FULLTRACK_PER_VIEW_APPEARANCE_FORMAT_BY_EDGE_SEMANTICS: Mapping[str, str] = {
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_APPEARANCE_FORMAT
    ),
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_APPEARANCE_FORMAT
    ),
    FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_HYBRID_CONTEXT_APPEARANCE_FORMAT
    ),
}
FULLTRACK_PER_VIEW_FEATURE_GRANULARITY_BY_EDGE_SEMANTICS: Mapping[str, str] = {
    FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_RAW_NCC_FEATURE_GRANULARITY
    ),
    FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_ALIGNED_LAYOUT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_FEATURE_GRANULARITY
    ),
    FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS: (
        FULLTRACK_PER_VIEW_HYBRID_CONTEXT_FEATURE_GRANULARITY
    ),
}
FULLTRACK_PER_VIEW_MODEL_FORMAT = "frozen_fulltrack_per_view_candidate_probe_model_v1"
FULLTRACK_PER_VIEW_PREDICTION_FORMAT = (
    "frozen_fulltrack_per_view_candidate_probe_predictions_v1"
)
MINIMUM_JOINT_EDGE_COVERAGE = 0.25
MONOTONIC_TOP4_INITIAL_WEIGHT = 0.01
RAW_TOP4_SIGNED_ARCHITECTURE = "monotonic_top1_relative_top4"
RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE = (
    "monotonic_top1_relative_positive_uplift_top4"
)
RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE = (
    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded"
)
RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE = (
    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal"
)
RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE = (
    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_balanced"
)
RAW_TOP4_ARCHITECTURES = frozenset(
    {
        RAW_TOP4_SIGNED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    }
)
RAW_TOPK_AGGREGATIONS = frozenset({"uniform_mean", "lower_envelope"})
PER_VIEW_RESIDUAL_ARCHITECTURES = frozenset({"mlp", "linear"})
RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS: Mapping[str, str] = {
    "uniform_mean": "deterministic_uniform_top4_real_observation_v1",
    "lower_envelope": "deterministic_lower_envelope_top4_real_observations_v1",
}


@dataclass(frozen=True)
class FulltrackPerViewFamily:
    profile_names: tuple[str, ...]
    rank2_hard_pair_weight: float = 1.0
    architecture: str = "sparse_per_view_mixture"
    raw_topk_aggregation: str = "uniform_mean"
    training_seed_key: str | None = None
    calibration_during_train: bool = False
    coarse_top1_stability_weight: float = 0.0
    edge_feature_semantics: str = FULLTRACK_PER_VIEW_RAW_NCC_EDGE_FEATURE_SEMANTICS

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.profile_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or float(self.rank2_hard_pair_weight) < 0.0
            or float(self.coarse_top1_stability_weight) < 0.0
            or self.architecture
            not in {"sparse_per_view_mixture", *RAW_TOP4_ARCHITECTURES}
            or str(self.raw_topk_aggregation) not in RAW_TOPK_AGGREGATIONS
            or str(self.edge_feature_semantics)
            not in FULLTRACK_PER_VIEW_APPEARANCE_FORMAT_BY_EDGE_SEMANTICS
            or (
                self.training_seed_key is not None
                and not str(self.training_seed_key).strip()
            )
            or (
                bool(self.calibration_during_train)
                and self.architecture
                not in {
                    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
                    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
                }
            )
            or (
                float(self.coarse_top1_stability_weight) > 0.0
                and self.architecture
                not in {
                    "sparse_per_view_mixture",
                    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
                }
            )
            or (
                self.architecture
                == RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE
                and float(self.coarse_top1_stability_weight) <= 0.0
            )
        ):
            raise ValueError("full-track per-view family is invalid")
        object.__setattr__(self, "profile_names", names)
        object.__setattr__(self, "raw_topk_aggregation", str(self.raw_topk_aggregation))
        object.__setattr__(
            self,
            "training_seed_key",
            None
            if self.training_seed_key is None
            else str(self.training_seed_key).strip(),
        )
        object.__setattr__(
            self, "calibration_during_train", bool(self.calibration_during_train)
        )
        object.__setattr__(
            self,
            "coarse_top1_stability_weight",
            float(self.coarse_top1_stability_weight),
        )
        object.__setattr__(
            self, "edge_feature_semantics", str(self.edge_feature_semantics)
        )


# These source families are fixed before validation targets are read.  The
# full family deliberately contains only candidate/support-view-conditioned
# real-image evidence; it is not a whole-image retrieval factor.
FULLTRACK_PER_VIEW_FAMILIES: Mapping[str, FulltrackPerViewFamily] = {
    "fixedprior_fulltrack_perview_alike": FulltrackPerViewFamily(
        ("alike_center", "alike_context3", "alike_context5")
    ),
    "fixedprior_fulltrack_perview_radio_intermediate": FulltrackPerViewFamily(
        (
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
        )
    ),
    "fixedprior_fulltrack_perview_multiscale": FulltrackPerViewFamily(
        (
            "radio_final_context5",
            "radio_intermediate_context9",
            "alike_context5",
        )
    ),
    # The full-image factor is deliberately separate from local RGB/NCC.  Its
    # two per-observation descriptors are retained until the learned view
    # mixture, so this probe isolates whether support-view marginalization is
    # more informative than the already-audited early summary aggregation.
    "fixedprior_fulltrack_perview_globalcontext_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_globalcontext_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_GLOBAL_CONTEXT_EDGE_FEATURE_SEMANTICS,
    ),
    # This is the first full-CSR landmark-region probe.  It changes neither
    # candidates nor support edges; every source below is retained per real
    # observation until the learned log-sum-exp view mixture.  The initial
    # sweep deliberately uses identity NLL only, so it tests context capacity
    # before train hard-repeat weighting is allowed to influence the result.
    "fixedprior_fulltrack_perview_region_final_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_FINAL_LAYOUT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_final_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_intermediate_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_LAYOUT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_intermediate_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_alike_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_ALIKE_LAYOUT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_alike_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_multisource_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_MULTISOURCE_LAYOUT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_multisource_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Pool-only controls retain broad landmark-centred context while avoiding
    # boundary-missing spatial bins.  Neither these nor the layout families
    # receive ``common_cell_fraction``: it remains an audit field, not a
    # learnable candidate-identity or support-availability cue.
    "fixedprior_fulltrack_perview_region_final_pool_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_FINAL_POOL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_final_pool_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_intermediate_pool_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_INTERMEDIATE_POOL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_intermediate_pool_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_alike_pool_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_ALIKE_POOL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_alike_pool_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_region_multisource_pool_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_REGION_MULTISOURCE_POOL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_region_multisource_pool_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISOURCE_REGION_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # This maplet branch is intentionally distinct from image-grid region
    # pooling.  Its support descriptors come from neighboring *SfM
    # observations* in the same real support image, and both query/support
    # centre regions are excluded.  Coverage stays an unknown mask rather than
    # a learned identity cue.  These initial families use identity NLL only;
    # hard-repeat loss remains disabled until the frozen evidence passes the
    # held-out gate.
    "fixedprior_fulltrack_perview_sfm_maplet_final_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_final_sfm_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_final_near_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sfm_maplet_final_wide_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_final_sfm_maplet_wide"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_final_wide_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sfm_maplet_intermediate_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_sfm_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_intermediate_near_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sfm_maplet_intermediate_wide_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_sfm_maplet_wide"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_intermediate_wide_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sfm_maplet_multiscale_near_mixture": FulltrackPerViewFamily(
        (
            *FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
                "radio_final_sfm_maplet_near"
            ],
            *FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
                "radio_intermediate_sfm_maplet_near"
            ],
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_multiscale_near_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sfm_maplet_multiscale_wide_mixture": FulltrackPerViewFamily(
        (
            *FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
                "radio_final_sfm_maplet_wide"
            ],
            *FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
                "radio_intermediate_sfm_maplet_wide"
            ],
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sfm_maplet_multiscale_wide_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # V2 keeps centre-excluded maplet appearance but no longer requires every
    # quadrant to contain an SfM neighbour.  The paired topology-only family
    # is fitted independently; neither artifact can authorize pose scoring.
    "fixedprior_fulltrack_perview_sparse_maplet_final_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_final_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_final_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_intermediate_pca256_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_pca256_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_intermediate_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_alike_fpn_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES_BY_NAME[
            "alike_fpn_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_alike_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_multiscale_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_multiscale_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Every visual source has a topology-only counterpart over exactly the
    # same frozen profile availability.  Comparing e.g. ALIKE-near against a
    # six-profile RADIO/ALIKE control would silently change the retained edge
    # population, so such a comparison is not a valid paired shortcut audit.
    "fixedprior_fulltrack_perview_sparse_maplet_topology_control_final_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES_BY_NAME[
            "radio_final_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_topology_control_final_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_topology_control_intermediate_pca256_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_pca256_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_topology_control_intermediate_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_topology_control_alike_fpn_near_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES_BY_NAME[
            "alike_fpn_sparse_maplet_near"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_topology_control_alike_near_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_topology_control_multiscale_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_sparse_maplet_topology_control_perview_v2",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Unlike sparse SfM-neighbor maplets, these profiles obtain context directly
    # from dense real-image maps around every frozen query/support anchor.  No
    # profile receives hard-repeat supervision before its independent audit.
    "fixedprior_fulltrack_perview_translation_mode_final_mixture": FulltrackPerViewFamily(
        (
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "radio_final_mode_context5_shift1"
            ],
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "radio_final_mode_context9_shift2"
            ],
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_translation_mode_final_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_translation_mode_intermediate_pca256_mixture": FulltrackPerViewFamily(
        (
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "radio_intermediate_pca256_mode_context5_shift1"
            ],
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "radio_intermediate_pca256_mode_context9_shift2"
            ],
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "radio_intermediate_pca256_mode_context13_shift3"
            ],
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_translation_mode_intermediate_pca256_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_translation_mode_alike_fpn_mixture": FulltrackPerViewFamily(
        (
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "alike_fpn_mode_context15_shift3"
            ],
            *FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES_BY_NAME[
                "alike_fpn_mode_context31_shift4"
            ],
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_translation_mode_alike_fpn_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_translation_mode_multiscale_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_translation_mode_multiscale_perview_mixture_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Preserve spatial phase inside each candidate-centred context crop.  The
    # values are descriptor correlations only: crop validity is handled as
    # unknown at export time and no overlap, rank, or support-count field is
    # exposed to the learned probe.  As with every S1 family, use identity NLL
    # alone until the frozen hard-repeat audit passes.
    "fixedprior_fulltrack_perview_spatial_pyramid_intermediate_pca256_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_pca256_spatial_pyramid_context9_shift2_bins3"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_intermediate_pca256_perview_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_spatial_pyramid_alike_fpn_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES_BY_NAME[
            "alike_fpn_spatial_pyramid_context15_shift3_bins3"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_alike_fpn_perview_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_spatial_pyramid_multiscale_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_multiscale_perview_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Matched controls contain only the original-crop in-image overlap tensor.
    # They are intentionally distinct artifacts and distinct families, never
    # concatenated with descriptor correlations.  A visual family cannot pass
    # the S1 gate if this control shows the same apparent gain.
    "fixedprior_fulltrack_perview_spatial_pyramid_mask_control_intermediate_pca256_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_pca256_spatial_pyramid_context9_shift2_bins3"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_mask_control_intermediate_pca256_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_spatial_pyramid_mask_control_alike_fpn_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES_BY_NAME[
            "alike_fpn_spatial_pyramid_context15_shift3_bins3"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_mask_control_alike_fpn_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    "fixedprior_fulltrack_perview_spatial_pyramid_mask_control_multiscale_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_spatial_pyramid_mask_control_multiscale_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # This S1 hybrid is intentionally limited to two complementary visual
    # factors from the same immutable CSR: anchor-centred multiscale local
    # structure and coarse intermediate absolute image phase.  No coverage,
    # candidate rank, support count, or geometry field becomes a learnable
    # input, and the view mixture remains downstream of both factors.
    "fixedprior_fulltrack_perview_hybrid_translation_absolute_intermediate_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_HYBRID_CONTEXT_PROFILE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_hybrid_translation_absolute_intermediate_perview_v1",
        edge_feature_semantics=(
            FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    ),
    # Full-image absolute-phase evidence is intentionally isolated from the
    # recentered local-mode probe above.  The visual profiles cannot read mask
    # coverage/count fields; a matched position-only family is fitted and
    # audited separately to catch any framing or anchor-mask confound.
    "fixedprior_fulltrack_perview_absolute_phase_final_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_VISUAL_PROFILE_NAMES_BY_NAME[
            "radio_final_absolute_phase"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_absolute_phase_final_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_absolute_phase_intermediate_pca256_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_VISUAL_PROFILE_NAMES_BY_NAME[
            "radio_intermediate_pca256_absolute_phase"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_absolute_phase_intermediate_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_absolute_phase_alike_fpn_mixture": FulltrackPerViewFamily(
        FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_VISUAL_PROFILE_NAMES_BY_NAME[
            "alike_fpn_absolute_phase"
        ],
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_absolute_phase_alike_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_absolute_phase_multiscale_mixture": FulltrackPerViewFamily(
        tuple(
            name
            for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
            for name in FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_VISUAL_PROFILE_NAMES_BY_NAME[
                profile.name
            ]
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_absolute_phase_multiscale_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_absolute_phase_position_control_mixture": FulltrackPerViewFamily(
        tuple(
            name
            for profile in ABSOLUTE_PHASE_VISUAL_PROFILES
            for name in FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_POSITION_PROFILE_NAMES_BY_NAME[
                profile.name
            ]
        ),
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_absolute_phase_position_control_perview_mixture_v1",
        edge_feature_semantics=FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS,
    ),
    # These retain the all-observation top-4 statistic explicitly instead of
    # asking an unconstrained MLP to rediscover a known raw signal.  The signed
    # variants remain diagnostics; the positive-uplift variant never uses a
    # lower raw score to reinforce the frozen top-one candidate.
    "fixedprior_fulltrack_rawtop4_alike": FulltrackPerViewFamily(
        ("alike_center", "alike_context3", "alike_context5", "alike_context9"),
        architecture=RAW_TOP4_SIGNED_ARCHITECTURE,
    ),
    "fixedprior_fulltrack_rawtop4_radio": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
        ),
        architecture=RAW_TOP4_SIGNED_ARCHITECTURE,
    ),
    "fixedprior_fulltrack_rawtop4_multiscale": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        architecture=RAW_TOP4_SIGNED_ARCHITECTURE,
    ),
    "fixedprior_fulltrack_rawtop4_positive_uplift_alike": FulltrackPerViewFamily(
        ("alike_center", "alike_context3", "alike_context5", "alike_context9"),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
    ),
    # The lower envelope of the top-four real support observations is a
    # deliberately conservative support-consensus probe.  It asks whether an
    # uplift survives more than one visually compatible view, rather than
    # allowing a single unusually similar observation to carry the statistic.
    "fixedprior_fulltrack_rawtop4_positive_uplift_alike_lower_envelope": FulltrackPerViewFamily(
        ("alike_center", "alike_context3", "alike_context5", "alike_context9"),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        raw_topk_aggregation="lower_envelope",
    ),
    # Keep the positive-uplift transformation representation-specific.  These
    # are separate S1 probes, not an implicit concatenation of evidence into a
    # production descriptor or a relaxation of the fixed candidate protocol.
    "fixedprior_fulltrack_rawtop4_positive_uplift_radio": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
        ),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
    ),
    "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
    ),
    # This is a separate, explicitly bounded calibration family.  It shares
    # the unbounded family's seed key so a train-only cap sweep isolates the
    # transform rather than conflating it with random initialization.
    "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        training_seed_key="fixedprior_fulltrack_rawtop4_positive_uplift_multiscale",
    ),
    "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        training_seed_key="fixedprior_fulltrack_rawtop4_positive_uplift_multiscale",
        calibration_during_train=True,
    ),
    # The rank-2 rescue term alone can teach a positive-only overlay to
    # overturn a correct coarse top-one track whenever a repeated support view
    # happens to score highly.  This paired family adds the symmetric
    # train-only constraint: when the immutable coarse top-one is correct, no
    # wrong candidate may overtake it.  It changes neither visual inputs nor
    # inference semantics.
    "fixedprior_fulltrack_rawtop4_positive_uplift_multiscale_tanh_cap_traincal_balanced": FulltrackPerViewFamily(
        (
            "radio_final_center",
            "radio_final_context3",
            "radio_final_context5",
            "radio_intermediate_center",
            "radio_intermediate_context5",
            "radio_intermediate_context9",
            "radio_intermediate_context13",
            "alike_center",
            "alike_context3",
            "alike_context5",
            "alike_context9",
        ),
        rank2_hard_pair_weight=8.0,
        architecture=RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
        training_seed_key="fixedprior_fulltrack_rawtop4_positive_uplift_multiscale",
        calibration_during_train=True,
        coarse_top1_stability_weight=8.0,
    ),
    # S1 absolute-phase probe: compact DCTs retain where aligned local
    # agreement occurs in a fixed query/support crop.  It uses the same frozen
    # global top-20 and every real SfM support observation, but no raw-NCC
    # scalar, support count, coordinate, retrieval, or pose input.
    "fixedprior_fulltrack_perview_aligned_layout_radio_final": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_RADIO_FINAL_FEATURE_NAMES,
        rank2_hard_pair_weight=8.0,
        training_seed_key="fulltrack_aligned_layout_phase_v1",
        coarse_top1_stability_weight=8.0,
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_radio_intermediate": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_RADIO_INTERMEDIATE_FEATURE_NAMES,
        rank2_hard_pair_weight=8.0,
        training_seed_key="fulltrack_aligned_layout_phase_v1",
        coarse_top1_stability_weight=8.0,
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_alike": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_ALIKE_FEATURE_NAMES,
        rank2_hard_pair_weight=8.0,
        training_seed_key="fulltrack_aligned_layout_phase_v1",
        coarse_top1_stability_weight=8.0,
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_multiscale": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_FEATURE_NAMES,
        rank2_hard_pair_weight=8.0,
        training_seed_key="fulltrack_aligned_layout_phase_v1",
        coarse_top1_stability_weight=8.0,
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    # Keep a pure identity-NLL capacity probe beside the older hard-pair and
    # top-one-stability variants.  The latter are useful only after a spatial
    # representation has shown held-out separability; otherwise their losses
    # can obscure whether the DCT layout itself contains absolute evidence.
    "fixedprior_fulltrack_perview_aligned_layout_final_nll": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_RADIO_FINAL_FEATURE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_aligned_layout_nll_v1",
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_intermediate_nll": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_RADIO_INTERMEDIATE_FEATURE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_aligned_layout_nll_v1",
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_alike_nll": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_ALIKE_FEATURE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_aligned_layout_nll_v1",
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
    "fixedprior_fulltrack_perview_aligned_layout_multiscale_nll": FulltrackPerViewFamily(
        ALIGNED_LAYOUT_FEATURE_NAMES,
        rank2_hard_pair_weight=0.0,
        training_seed_key="fulltrack_aligned_layout_nll_v1",
        edge_feature_semantics=FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS,
    ),
}


# This table is the only supported visual/control comparison for sparse
# maplets.  It binds the two models to the same source profile and therefore
# the same target-free valid-edge population before any OOF target is loaded.
FULLTRACK_PER_VIEW_SPARSE_MAPLET_PAIRED_CONTROL_FAMILIES: Mapping[str, str] = {
    "fixedprior_fulltrack_perview_sparse_maplet_final_near_mixture": (
        "fixedprior_fulltrack_perview_sparse_maplet_topology_control_final_near_mixture"
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_intermediate_pca256_near_mixture": (
        "fixedprior_fulltrack_perview_sparse_maplet_topology_control_intermediate_pca256_near_mixture"
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_alike_fpn_near_mixture": (
        "fixedprior_fulltrack_perview_sparse_maplet_topology_control_alike_fpn_near_mixture"
    ),
    "fixedprior_fulltrack_perview_sparse_maplet_multiscale_mixture": (
        "fixedprior_fulltrack_perview_sparse_maplet_topology_control_multiscale_mixture"
    ),
}


@dataclass(frozen=True)
class FrozenFulltrackPerViewAppearanceFeatures:
    """Merged, target-free CSR support-view evidence for fixed candidate rows."""

    paths: tuple[Path, ...]
    query_ids: np.ndarray
    split_names: np.ndarray
    source_row_indices: np.ndarray
    xy: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    candidate_support_observation_counts: np.ndarray
    edge_candidate_offsets: np.ndarray
    edge_geometry_rows: np.ndarray
    edge_profile_scores: np.ndarray
    edge_profile_valid: np.ndarray
    profile_names: tuple[str, ...]
    artifact_metadata: tuple[dict[str, Any], ...]
    compatibility: Mapping[str, Any]

    def __post_init__(self) -> None:
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        candidate = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        counts = np.asarray(self.candidate_support_observation_counts, dtype=np.int64)
        offsets = np.asarray(self.edge_candidate_offsets, dtype=np.int64).reshape(-1)
        geometry_rows = np.asarray(self.edge_geometry_rows, dtype=np.int64).reshape(-1)
        # Full-CSR spatial probes can contain billions of cached edge values.
        # Keep a persisted float16 cache compact here; every numerical consumer
        # explicitly promotes its selected batch to float32 before arithmetic.
        # This avoids a duplicate full-dataset expansion during OOF loading.
        raw_scores = np.asarray(self.edge_profile_scores)
        if raw_scores.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raw_scores = raw_scores.astype(np.float32)
        scores = raw_scores
        valid = np.asarray(self.edge_profile_valid, dtype=bool)
        profile_names = tuple(str(value) for value in self.profile_names)
        row_count = len(query_ids)
        row_keys = tuple(zip(query_ids.tolist(), rows.tolist()))
        score_missingness_valid = True
        if scores.ndim == 2 and valid.shape == scores.shape:
            # Avoid boolean-indexing the whole full-CSR cache at once.  For
            # large S1 visual tensors that temporary can be as large as the
            # cache itself even though this is only an integrity check.
            for start in range(0, len(scores), 262144):
                score_chunk = scores[start : start + 262144]
                valid_chunk = valid[start : start + 262144]
                if (
                    np.any(~np.isfinite(score_chunk[valid_chunk]))
                    or np.any(np.isfinite(score_chunk[~valid_chunk]))
                ):
                    score_missingness_valid = False
                    break
        else:
            score_missingness_valid = False
        if (
            row_count == 0
            # Source-row indices are only unique within a query artifact.  A
            # complete train/validation merge intentionally reuses them for
            # every query image, so identity is the (query_id, source_row)
            # pair rather than source_row alone.
            or len(set(row_keys)) != row_count
            or not (splits.shape == rows.shape == null.shape == (row_count,))
            or xy.shape != (row_count, 2)
            or tracks.ndim != 2
            or candidate.shape != tracks.shape
            or counts.shape != tracks.shape
            or tracks.shape[0] != row_count
            or offsets.shape != (tracks.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(geometry_rows)
            or np.any(offsets[1:] < offsets[:-1])
            or not profile_names
            or len(set(profile_names)) != len(profile_names)
            or scores.shape != valid.shape
            or scores.shape != (len(geometry_rows), len(profile_names))
            or np.any(geometry_rows < 0)
            or np.any(counts < 0)
            or not np.array_equal(np.diff(offsets), counts.reshape(-1))
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(candidate))
            or np.any(~np.isfinite(null))
            or np.any(candidate < 0.0)
            or np.any((candidate > 0.0) & (tracks < 0))
            or np.any((candidate <= 0.0) & (counts > 0))
            or np.any((candidate > 0.0) & (counts <= 0))
            or not score_missingness_valid
            or np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
        ):
            raise ValueError("frozen full-track per-view appearance arrays are invalid")
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "source_row_indices", rows)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_probabilities", candidate)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "candidate_support_observation_counts", counts)
        object.__setattr__(self, "edge_candidate_offsets", offsets)
        object.__setattr__(self, "edge_geometry_rows", geometry_rows)
        object.__setattr__(self, "edge_profile_scores", scores)
        object.__setattr__(self, "edge_profile_valid", valid)
        object.__setattr__(self, "profile_names", profile_names)

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_track_ids.shape[1])


def _load_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_support_observation_counts",
        "profile_names",
        "edge_candidate_offsets",
        "edge_geometry_rows",
        "edge_profile_scores",
        "edge_profile_valid",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: full-track per-view artifact lacks {missing}")
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in required
            if name != "metadata_json"
        }
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    strict = metadata.get("strict_fulltrack_appearance_contract")
    edge_feature_semantics = str(metadata.get("per_view_edge_feature_semantics", ""))
    expected_format = FULLTRACK_PER_VIEW_APPEARANCE_FORMAT_BY_EDGE_SEMANTICS.get(
        edge_feature_semantics
    )
    if expected_format is None:
        raise ValueError(f"{path}: unsupported per-view edge feature semantics")
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != expected_format
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("fixed_candidate_top_k") != 20
        or metadata.get("support_view_source")
        != "all_real_sfm_track_observations_v1"
        or metadata.get("support_view_count_cap") is not None
        or metadata.get("per_view_edges_retained") is not True
        or edge_feature_semantics
        not in FULLTRACK_PER_VIEW_APPEARANCE_FORMAT_BY_EDGE_SEMANTICS
        or not isinstance(strict, Mapping)
        or strict.get("candidate_identity_fixed") is not True
        or strict.get("candidate_posterior_preserved") is not True
        or strict.get("support_reselection") is not False
        or strict.get("all_real_sfm_track_observations_enumerated") is not True
        or strict.get("support_view_count_cap") is not None
        or strict.get("candidate_3d_projection_or_pose_used") is not False
        or strict.get("image_retrieval_or_submap_used") is not False
        or strict.get("render") is not False
        or strict.get("heldout_s0_verification_rows") is not True
    ):
        raise ValueError(f"{path}: per-view artifact violates the frozen protocol")
    if edge_feature_semantics == FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS:
        layout = metadata.get("aligned_layout_contract")
        intermediate = metadata.get("radio_intermediate_projection_override")
        if (
            not isinstance(layout, Mapping)
            or layout.get("representation")
            != "same_relative_cell_cosine_centered_low_frequency_dct_v1"
            or layout.get("missing_evidence")
            != "invalid_edge_omitted_neutral_no_mask_features_v1"
            or layout.get("per_view_order")
            != "all_real_sfm_observations_candidate_major_v1"
            or layout.get("candidate_coordinates_or_pose_used") is not False
            or not isinstance(layout.get("profiles"), list)
            or not layout["profiles"]
            or not isinstance(intermediate, Mapping)
            or not isinstance(intermediate.get("enabled"), bool)
        ):
            raise ValueError(f"{path}: aligned-layout artifact lacks its immutable contract")
    if edge_feature_semantics in {
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    }:
        sparse_maplet = metadata.get("sparse_maplet_transport_contract")
        appearance = metadata.get("appearance_config")
        is_control = (
            edge_feature_semantics
            == FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS
        )
        if (
            not isinstance(sparse_maplet, Mapping)
            or sparse_maplet.get("mode")
            != "partial_center_excluded_sfm_maplet_transport_v2"
            or sparse_maplet.get("support_coordinate_source") != "sfm_observation_xy"
            or sparse_maplet.get("support_neighbor_scope")
            != "same_real_support_image_only"
            or sparse_maplet.get("view_aggregation")
            != "none_before_learned_logsumexp_mixture_v1"
            or sparse_maplet.get("center_descriptor_in_features") is not False
            or sparse_maplet.get("partial_support_quadrants_retained") is not True
            or sparse_maplet.get("minimum_total_neighbors")
            != SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS
            or sparse_maplet.get("maximum_neighbors_per_quadrant")
            != SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT
            or sparse_maplet.get("visual_border_padding")
            != "reflection_from_real_feature_map_v1"
            or sparse_maplet.get("original_crop_topology_control_exported_separately")
            is not True
            or sparse_maplet.get("availability_is_a_visual_feature") is not False
            or not isinstance(sparse_maplet.get("profiles"), list)
            or not sparse_maplet["profiles"]
            or not isinstance(sparse_maplet.get("neighbor_topology_sha256"), Mapping)
            or not isinstance(appearance, Mapping)
            or appearance.get("candidate_specific") is not True
            or appearance.get("per_view") is not True
            or appearance.get("control_only") is not is_control
            or appearance.get("visual_descriptor_values_included") is not (not is_control)
            or appearance.get("topology_or_original_crop_values_included") is not is_control
            or strict.get("candidate_center_descriptor_excluded") is not True
            or strict.get("partial_support_maplet_is_retained_when_total_neighbors_sufficient")
            is not True
            or strict.get("visual_descriptor_values_included") is not (not is_control)
            or strict.get("topology_or_original_crop_values_included") is not is_control
            or strict.get("paired_topology_control_artifact_required") is not True
            or metadata.get("artifact_role")
            != (
                "topology_and_original_crop_control"
                if is_control
                else "visual_descriptor_transport"
            )
        ):
            raise ValueError(f"{path}: sparse-maplet artifact lacks its immutable contract")
    if edge_feature_semantics == FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS:
        spatial = metadata.get("spatial_pyramid_shift_contract")
        if (
            not isinstance(spatial, Mapping)
            or spatial.get("mode")
            != "candidate_specific_multiscale_spatial_pyramid_shift_correlation_v1"
            or spatial.get("support_coordinate_source") != "sfm_observation_xy"
            or spatial.get("view_aggregation")
            != "none_before_learned_logsumexp_mixture_v1"
            or spatial.get("missing_evidence")
            != "invalid_center_anchor_edge_omitted_neutral_v1"
            or spatial.get("visual_border_padding")
            != "reflection_from_real_feature_map_v1"
            or spatial.get("original_crop_mask_control_exported_separately") is not True
            or spatial.get("explicit_availability_or_neighbor_count_feature")
            is not False
            or spatial.get("full_crop_required") is not False
            or spatial.get("center_anchor_required") is not True
            or not isinstance(spatial.get("profiles"), list)
            or not spatial["profiles"]
            or not isinstance(spatial.get("mask_control_profiles"), list)
            or not spatial["mask_control_profiles"]
            or strict.get("incomplete_center_anchor_is_unknown_not_visual_value")
            is not True
            or strict.get("visual_descriptor_values_included") is not True
            or strict.get("original_crop_mask_values_included") is not False
            or strict.get("paired_mask_control_artifact_required") is not True
            or metadata.get("artifact_role") != "visual_descriptor_correlation"
        ):
            raise ValueError(f"{path}: spatial-pyramid artifact lacks its immutable contract")
    if (
        edge_feature_semantics
        == FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS
    ):
        spatial = metadata.get("spatial_pyramid_shift_contract")
        appearance = metadata.get("appearance_config")
        if (
            not isinstance(spatial, Mapping)
            or spatial.get("mode")
            != "candidate_specific_multiscale_spatial_pyramid_shift_correlation_v1"
            or spatial.get("original_crop_mask_control_exported_separately") is not True
            or spatial.get("visual_border_padding")
            != "reflection_from_real_feature_map_v1"
            or not isinstance(spatial.get("mask_control_profiles"), list)
            or not spatial["mask_control_profiles"]
            or not isinstance(appearance, Mapping)
            or appearance.get("control_only") is not True
            or appearance.get("visual_descriptor_values_included") is not False
            or appearance.get("original_crop_mask_values_included") is not True
            or strict.get("visual_descriptor_values_included") is not False
            or strict.get("original_crop_mask_values_included") is not True
            or strict.get("control_only") is not True
            or metadata.get("artifact_role") != "original_crop_mask_overlap_control"
        ):
            raise ValueError(f"{path}: spatial-pyramid mask-control contract is invalid")
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    source_rows = np.asarray(
        arrays["verification_source_row_indices"], dtype=np.int64
    ).reshape(-1)
    if (
        len(query_ids) != 192
        or len(set(query_ids.tolist())) != 1
        or len(set(split_names.tolist())) != 1
        or split_names[0] not in {"train", "validation"}
        or len(np.unique(source_rows)) != 192
    ):
        raise ValueError(f"{path}: per-view artifact is not a complete frozen query shard")
    return arrays, metadata


def _compatibility(metadata: Mapping[str, Any]) -> dict[str, Any]:
    edge_feature_semantics = str(metadata.get("per_view_edge_feature_semantics", ""))
    if edge_feature_semantics not in FULLTRACK_PER_VIEW_APPEARANCE_FORMAT_BY_EDGE_SEMANTICS:
        raise ValueError("per-view artifact has unsupported edge feature semantics")
    result: dict[str, Any] = {
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "profiles": metadata.get("profiles"),
        "appearance_config": metadata.get("appearance_config"),
        "support_geometry_index_sha256": metadata.get("support_geometry_index_sha256"),
        "context_cache_sha256": metadata.get("context_cache_sha256"),
        "per_view_edge_feature_semantics": metadata.get(
            "per_view_edge_feature_semantics"
        ),
        "implementation": metadata.get("implementation"),
    }
    if edge_feature_semantics == FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS:
        # A dimensionality match is not a descriptor-space match.  Keep the
        # exact PCA override and layout definition in the merged compatibility
        # key so a cached raw-NCC/PCA64 or different layout artifact cannot be
        # silently mixed with this full-map PCA256 phase probe.
        result["radio_intermediate_projection_override"] = metadata.get(
            "radio_intermediate_projection_override"
        )
        result["aligned_layout_contract"] = metadata.get("aligned_layout_contract")
        result["source_per_view_contract"] = metadata.get("source_per_view_contract")
    if edge_feature_semantics in {
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS,
        FULLTRACK_PER_VIEW_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS,
    }:
        result["sparse_maplet_transport_contract"] = metadata.get(
            "sparse_maplet_transport_contract"
        )
        result["artifact_role"] = metadata.get("artifact_role")
    if edge_feature_semantics in {
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS,
        FULLTRACK_PER_VIEW_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS,
    }:
        result["spatial_pyramid_shift_contract"] = metadata.get(
            "spatial_pyramid_shift_contract"
        )
        result["artifact_role"] = metadata.get("artifact_role")
    return result


def _resolve_hybrid_manifest_component_path(
    *, manifest_path: Path, value: object
) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _hybrid_manifest_entries_by_query(
    *,
    manifest_path: Path,
    value: object,
    component: str,
) -> dict[str, tuple[Path, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{manifest_path}: hybrid {component} entries are invalid")
    entries: dict[str, tuple[Path, str]] = {}
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{manifest_path}: hybrid {component} entry is invalid")
        query_id = str(entry.get("query_id", "")).strip()
        expected_hash = str(entry.get("sha256", "")).strip()
        if not query_id or not expected_hash or "path" not in entry or query_id in entries:
            raise ValueError(f"{manifest_path}: hybrid {component} entry is incomplete")
        path = _resolve_hybrid_manifest_component_path(
            manifest_path=manifest_path, value=entry["path"]
        )
        if not path.is_file() or file_sha256_short(path) != expected_hash:
            raise ValueError(f"{manifest_path}: hybrid {component} source is stale")
        entries[query_id] = (path, expected_hash)
    return entries


def _hybrid_component_profile_indices(
    *, profile_names: np.ndarray, required_names: Sequence[str], context: str
) -> np.ndarray:
    available = tuple(np.asarray(profile_names).astype(str).tolist())
    index_by_name = {name: index for index, name in enumerate(available)}
    if len(index_by_name) != len(available) or any(
        name not in index_by_name for name in required_names
    ):
        raise ValueError(f"{context}: hybrid component profiles differ")
    return np.asarray([index_by_name[name] for name in required_names], dtype=np.int64)


def _hybrid_manifest_compatibility(
    *, manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": FULLTRACK_PER_VIEW_HYBRID_CONTEXT_APPEARANCE_FORMAT,
        "version": int(manifest["version"]),
        "per_view_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
        "feature_granularity": FULLTRACK_PER_VIEW_HYBRID_CONTEXT_FEATURE_GRANULARITY,
        "hybrid_manifest_sha256": file_sha256_short(manifest_path),
        "translation_component": {
            "format": FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
            "edge_feature_semantics": (
                FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
            ),
            "profile_names": list(HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES),
        },
        "absolute_component": {
            "format": FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
            "edge_feature_semantics": (
                FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS
            ),
            "profile_names": list(HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES),
        },
        "component_query_count": len(manifest["translation_artifacts"]),
    }


def _load_hybrid_manifest_features(
    manifest_path: Path,
) -> FrozenFulltrackPerViewAppearanceFeatures:
    """Join two aligned artifact sets without materializing a copied cache."""

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{manifest_path}: hybrid manifest is unreadable") from error
    expected_contracts = {
        "translation_format": FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT,
        "translation_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS
        ),
        "absolute_format": FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_APPEARANCE_FORMAT,
        "absolute_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_ABSOLUTE_PHASE_EDGE_FEATURE_SEMANTICS
        ),
        "hybrid_edge_feature_semantics": (
            FULLTRACK_PER_VIEW_HYBRID_CONTEXT_EDGE_FEATURE_SEMANTICS
        ),
    }
    expected_protocol = {
        "feature_export_target_free": True,
        "identity_or_pose_targets_loaded": False,
        "fixed_global_top_l": 20,
        "candidate_reselection": False,
        "support_reselection": False,
        "all_real_sfm_support_observations_retained": True,
        "support_view_features_averaged_before_inference": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "virtual_manifest_no_feature_cache_materialized": True,
        "translation_and_absolute_csr_exactly_aligned": True,
    }
    selected = manifest.get("selected_profiles") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("format") != FULLTRACK_PER_VIEW_HYBRID_CONTEXT_APPEARANCE_FORMAT
        or int(manifest.get("version", -1)) != 1
        or manifest.get("component_contracts") != expected_contracts
        or not isinstance(selected, Mapping)
        or tuple(selected.get("translation", ()))
        != tuple(HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES)
        or tuple(selected.get("absolute", ()))
        != tuple(HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES)
        or tuple(selected.get("combined", ())) != tuple(HYBRID_CONTEXT_PROFILE_NAMES)
        or not isinstance(manifest.get("protocol"), Mapping)
        or any(manifest["protocol"].get(key) != value for key, value in expected_protocol.items())
    ):
        raise ValueError(f"{manifest_path}: hybrid manifest contract differs")
    translation_entries = _hybrid_manifest_entries_by_query(
        manifest_path=manifest_path,
        value=manifest.get("translation_artifacts"),
        component="translation",
    )
    absolute_entries = _hybrid_manifest_entries_by_query(
        manifest_path=manifest_path,
        value=manifest.get("absolute_artifacts"),
        component="absolute",
    )
    if set(translation_entries) != set(absolute_entries):
        raise ValueError(f"{manifest_path}: hybrid component query sets differ")
    array_blocks: dict[str, list[np.ndarray]] = {
        "query_ids": [],
        "split_names": [],
        "source_rows": [],
        "xy": [],
        "tracks": [],
        "candidate": [],
        "null": [],
        "counts": [],
        "geometry_rows": [],
        "scores": [],
        "valid": [],
    }
    merged_offsets = [0]
    metadata_rows: list[dict[str, Any]] = []
    row_keys: list[tuple[str, int]] = []
    for expected_query_id in sorted(translation_entries):
        translation_path, translation_hash = translation_entries[expected_query_id]
        absolute_path, absolute_hash = absolute_entries[expected_query_id]
        translation_arrays, translation_metadata = _load_artifact(translation_path)
        absolute_arrays, absolute_metadata = _load_artifact(absolute_path)
        query_id = validate_hybrid_component_pair(
            translation_arrays=translation_arrays,
            translation_metadata=translation_metadata,
            translation_context=str(translation_path),
            absolute_arrays=absolute_arrays,
            absolute_metadata=absolute_metadata,
            absolute_context=str(absolute_path),
        )
        if query_id != expected_query_id or hybrid_component_query_id(
            absolute_arrays, context=str(absolute_path)
        ) != expected_query_id:
            raise ValueError(f"{manifest_path}: hybrid query entry does not match source")
        translation_indices = _hybrid_component_profile_indices(
            profile_names=translation_arrays["profile_names"],
            required_names=HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES,
            context=str(translation_path),
        )
        absolute_indices = _hybrid_component_profile_indices(
            profile_names=absolute_arrays["profile_names"],
            required_names=HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES,
            context=str(absolute_path),
        )
        counts = np.asarray(
            translation_arrays["candidate_support_observation_counts"], dtype=np.int64
        )
        offsets = np.asarray(translation_arrays["edge_candidate_offsets"], dtype=np.int64)
        if (
            offsets.shape != (counts.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(translation_arrays["edge_geometry_rows"])
            or not np.array_equal(np.diff(offsets), counts.reshape(-1))
        ):
            raise ValueError(f"{translation_path}: hybrid CSR edge layout is invalid")
        query_ids = np.asarray(translation_arrays["verification_query_ids"]).astype(str)
        source_rows = np.asarray(
            translation_arrays["verification_source_row_indices"], dtype=np.int64
        )
        row_keys.extend((str(value), int(row)) for value, row in zip(query_ids, source_rows))
        array_blocks["query_ids"].append(query_ids)
        array_blocks["split_names"].append(
            np.asarray(translation_arrays["split_names"]).astype(str)
        )
        array_blocks["source_rows"].append(source_rows)
        array_blocks["xy"].append(
            np.asarray(translation_arrays["verification_xy"], dtype=np.float32)
        )
        array_blocks["tracks"].append(
            np.asarray(translation_arrays["candidate_track_ids"], dtype=np.int64)
        )
        array_blocks["candidate"].append(
            np.asarray(translation_arrays["candidate_probabilities"], dtype=np.float32)
        )
        array_blocks["null"].append(
            np.asarray(translation_arrays["null_probabilities"], dtype=np.float32)
        )
        array_blocks["counts"].append(counts)
        array_blocks["geometry_rows"].append(
            np.asarray(translation_arrays["edge_geometry_rows"], dtype=np.int64)
        )
        array_blocks["scores"].append(
            np.concatenate(
                (
                    np.asarray(translation_arrays["edge_profile_scores"])[
                        :, translation_indices
                    ],
                    np.asarray(absolute_arrays["edge_profile_scores"])[
                        :, absolute_indices
                    ],
                ),
                axis=1,
            ).astype(np.float32, copy=False)
        )
        array_blocks["valid"].append(
            np.concatenate(
                (
                    np.asarray(translation_arrays["edge_profile_valid"], dtype=bool)[
                        :, translation_indices
                    ],
                    np.asarray(absolute_arrays["edge_profile_valid"], dtype=bool)[
                        :, absolute_indices
                    ],
                ),
                axis=1,
            )
        )
        for count in counts.reshape(-1).tolist():
            merged_offsets.append(merged_offsets[-1] + int(count))
        hybrid_metadata = dict(absolute_metadata)
        hybrid_metadata["hybrid_context_component"] = {
            "translation_path": str(translation_path),
            "translation_sha256": translation_hash,
            "absolute_path": str(absolute_path),
            "absolute_sha256": absolute_hash,
            "translation_profile_names": list(HYBRID_CONTEXT_TRANSLATION_PROFILE_NAMES),
            "absolute_profile_names": list(
                HYBRID_CONTEXT_ABSOLUTE_INTERMEDIATE_PROFILE_NAMES
            ),
        }
        metadata_rows.append(hybrid_metadata)
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(f"{manifest_path}: hybrid artifacts overlap query/source rows")
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=(manifest_path,),
        query_ids=np.concatenate(array_blocks["query_ids"]),
        split_names=np.concatenate(array_blocks["split_names"]),
        source_row_indices=np.concatenate(array_blocks["source_rows"]),
        xy=np.concatenate(array_blocks["xy"]),
        candidate_track_ids=np.concatenate(array_blocks["tracks"]),
        candidate_probabilities=np.concatenate(array_blocks["candidate"]),
        null_probabilities=np.concatenate(array_blocks["null"]),
        candidate_support_observation_counts=np.concatenate(array_blocks["counts"]),
        edge_candidate_offsets=np.asarray(merged_offsets, dtype=np.int64),
        edge_geometry_rows=np.concatenate(array_blocks["geometry_rows"]),
        edge_profile_scores=np.concatenate(array_blocks["scores"]),
        edge_profile_valid=np.concatenate(array_blocks["valid"]),
        profile_names=FULLTRACK_PER_VIEW_HYBRID_CONTEXT_PROFILE_NAMES,
        artifact_metadata=tuple(metadata_rows),
        compatibility=_hybrid_manifest_compatibility(
            manifest_path=manifest_path, manifest=manifest
        ),
    )


def load_frozen_fulltrack_per_view_appearance_features(
    paths: Sequence[Path],
) -> FrozenFulltrackPerViewAppearanceFeatures:
    """Merge complete per-query artifacts while rebuilding their CSR offsets.

    A complete S1 export holds a large, dense edge-profile tensor in every
    shard.  Loading all shards into ``array_blocks`` before concatenating them
    temporarily retains two full copies of the dataset, which makes concurrent
    OOF probes needlessly memory-bound.  Validate one shard at a time first,
    then preallocate the merged CSR and fill it one shard at a time.  The
    second read is intentional: it keeps the peak resident set bounded by the
    final merged cache plus a single shard rather than all source shards plus
    the final cache.
    """

    artifact_paths = tuple(Path(path) for path in paths)
    if not artifact_paths or len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("full-track per-view artifact paths must be non-empty and unique")
    manifest_paths = tuple(path for path in artifact_paths if path.suffix == ".json")
    if manifest_paths:
        if len(artifact_paths) != 1 or len(manifest_paths) != 1:
            raise ValueError("a virtual hybrid manifest must be the only appearance input")
        return _load_hybrid_manifest_features(manifest_paths[0])

    headers: list[dict[str, Any]] = []
    reference_compatibility: Mapping[str, Any] | None = None
    profile_names: np.ndarray | None = None
    row_keys: list[tuple[str, int]] = []
    total_rows = 0
    total_edges = 0
    candidate_count: int | None = None
    score_dtype: np.dtype[Any] | None = None
    query_id_width = 1
    split_name_width = 1
    for path in artifact_paths:
        arrays, metadata = _load_artifact(path)
        compatibility = _compatibility(metadata)
        shard_profile_names = np.asarray(arrays["profile_names"]).astype(str)
        shard_tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        shard_scores = np.asarray(arrays["edge_profile_scores"])
        shard_valid = np.asarray(arrays["edge_profile_valid"], dtype=bool)
        shard_geometry = np.asarray(arrays["edge_geometry_rows"], dtype=np.int64)
        shard_rows = len(np.asarray(arrays["verification_query_ids"]).reshape(-1))
        if reference_compatibility is None:
            reference_compatibility = compatibility
            profile_names = shard_profile_names
            candidate_count = int(shard_tracks.shape[1]) if shard_tracks.ndim == 2 else None
            score_dtype = shard_scores.dtype
        if (
            reference_compatibility != compatibility
            or profile_names is None
            or not np.array_equal(shard_profile_names, profile_names)
            or candidate_count is None
            or shard_tracks.shape != (shard_rows, candidate_count)
            or shard_scores.ndim != 2
            or shard_scores.shape != shard_valid.shape
            or shard_scores.shape != (len(shard_geometry), len(profile_names))
            or score_dtype is None
        ):
            raise ValueError(f"{path}: per-view artifact configuration differs")
        score_dtype = np.result_type(score_dtype, shard_scores.dtype)
        if score_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise ValueError(f"{path}: per-view edge scores must be float16 or float32")
        if (
            np.asarray(arrays["candidate_probabilities"]).shape != shard_tracks.shape
            or np.asarray(arrays["candidate_support_observation_counts"]).shape
            != shard_tracks.shape
            or np.asarray(arrays["verification_xy"]).shape != (shard_rows, 2)
            or np.asarray(arrays["null_probabilities"]).shape != (shard_rows,)
        ):
            raise ValueError(f"{path}: per-view artifact row layout is invalid")
        query_ids = np.asarray(arrays["verification_query_ids"]).astype(str)
        source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64)
        counts = np.asarray(arrays["candidate_support_observation_counts"], dtype=np.int64)
        offsets = np.asarray(arrays["edge_candidate_offsets"], dtype=np.int64)
        if (
            offsets.shape != (counts.size + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(arrays["edge_geometry_rows"])
            or not np.array_equal(np.diff(offsets), counts.reshape(-1))
        ):
            raise ValueError(f"{path}: per-view CSR edge layout is invalid")
        keys = list(zip(query_ids.tolist(), source_rows.tolist()))
        row_keys.extend((str(query_id), int(row)) for query_id, row in keys)
        split_names = np.asarray(arrays["split_names"]).astype(str)
        query_id_width = max(query_id_width, max(map(len, query_ids.tolist()), default=1))
        split_name_width = max(split_name_width, max(map(len, split_names.tolist()), default=1))
        headers.append(
            {
                "path": path,
                "metadata": metadata,
                "compatibility": compatibility,
                "row_count": shard_rows,
                "edge_count": len(shard_geometry),
            }
        )
        total_rows += shard_rows
        total_edges += len(shard_geometry)
        del arrays
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("full-track per-view artifacts overlap query/source rows")
    if (
        reference_compatibility is None
        or profile_names is None
        or candidate_count is None
        or score_dtype is None
        or total_rows <= 0
        or total_edges <= 0
    ):
        raise ValueError("full-track per-view artifacts are empty")

    query_ids = np.empty((total_rows,), dtype=f"<U{query_id_width}")
    split_names = np.empty((total_rows,), dtype=f"<U{split_name_width}")
    source_rows = np.empty((total_rows,), dtype=np.int64)
    xy = np.empty((total_rows, 2), dtype=np.float32)
    tracks = np.empty((total_rows, candidate_count), dtype=np.int64)
    candidate = np.empty((total_rows, candidate_count), dtype=np.float32)
    null = np.empty((total_rows,), dtype=np.float32)
    counts = np.empty((total_rows, candidate_count), dtype=np.int64)
    geometry_rows = np.empty((total_edges,), dtype=np.int64)
    scores = np.empty((total_edges, len(profile_names)), dtype=score_dtype)
    valid = np.empty((total_edges, len(profile_names)), dtype=bool)

    row_offset = 0
    edge_offset = 0
    for header in headers:
        path = Path(header["path"])
        arrays, metadata = _load_artifact(path)
        shard_rows = int(header["row_count"])
        shard_edges = int(header["edge_count"])
        row_slice = slice(row_offset, row_offset + shard_rows)
        edge_slice = slice(edge_offset, edge_offset + shard_edges)
        shard_scores = np.asarray(arrays["edge_profile_scores"])
        shard_valid = np.asarray(arrays["edge_profile_valid"], dtype=bool)
        if (
            metadata != header["metadata"]
            or _compatibility(metadata) != reference_compatibility
            or np.asarray(arrays["candidate_track_ids"]).shape
            != (shard_rows, candidate_count)
            or shard_scores.shape != (shard_edges, len(profile_names))
            or shard_valid.shape != shard_scores.shape
        ):
            raise ValueError(f"{path}: per-view artifact changed during streaming merge")
        query_ids[row_slice] = np.asarray(arrays["verification_query_ids"]).astype(str)
        split_names[row_slice] = np.asarray(arrays["split_names"]).astype(str)
        source_rows[row_slice] = np.asarray(
            arrays["verification_source_row_indices"], dtype=np.int64
        )
        xy[row_slice] = np.asarray(arrays["verification_xy"], dtype=np.float32)
        tracks[row_slice] = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        candidate[row_slice] = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
        null[row_slice] = np.asarray(arrays["null_probabilities"], dtype=np.float32)
        counts[row_slice] = np.asarray(
            arrays["candidate_support_observation_counts"], dtype=np.int64
        )
        geometry_rows[edge_slice] = np.asarray(arrays["edge_geometry_rows"], dtype=np.int64)
        scores[edge_slice] = shard_scores
        valid[edge_slice] = shard_valid
        row_offset += shard_rows
        edge_offset += shard_edges
        del arrays
    if row_offset != total_rows or edge_offset != total_edges:
        raise RuntimeError("streaming per-view merge did not fill its allocated CSR")
    merged_offsets = np.empty((total_rows * candidate_count + 1,), dtype=np.int64)
    merged_offsets[0] = 0
    np.cumsum(counts.reshape(-1), dtype=np.int64, out=merged_offsets[1:])
    return FrozenFulltrackPerViewAppearanceFeatures(
        paths=artifact_paths,
        query_ids=query_ids,
        split_names=split_names,
        source_row_indices=source_rows,
        xy=xy,
        candidate_track_ids=tracks,
        candidate_probabilities=candidate,
        null_probabilities=null,
        candidate_support_observation_counts=counts,
        edge_candidate_offsets=merged_offsets,
        edge_geometry_rows=geometry_rows,
        edge_profile_scores=scores,
        edge_profile_valid=valid,
        profile_names=tuple(profile_names.tolist()),
        artifact_metadata=tuple(header["metadata"] for header in headers),
        compatibility=reference_compatibility,
    )


def profile_indices_for_fulltrack_per_view_family(
    family: str, *, profile_names: Sequence[str]
) -> np.ndarray:
    spec = FULLTRACK_PER_VIEW_FAMILIES.get(str(family))
    if spec is None:
        raise ValueError(f"unsupported full-track per-view family: {family!r}")
    source = tuple(str(name) for name in profile_names)
    unavailable = tuple(name for name in spec.profile_names if name not in source)
    if unavailable:
        raise ValueError(
            f"per-view family {family!r} lacks required profiles: {list(unavailable)}"
        )
    indices = np.asarray([source.index(name) for name in spec.profile_names], dtype=np.int64)
    return indices


def fulltrack_per_view_feature_granularity(
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> str:
    """Return the persisted edge granularity after validating its semantics."""

    semantics = str(features.compatibility.get("per_view_edge_feature_semantics", ""))
    granularity = FULLTRACK_PER_VIEW_FEATURE_GRANULARITY_BY_EDGE_SEMANTICS.get(
        semantics
    )
    if granularity is None:
        raise ValueError("full-track per-view feature semantics are unsupported")
    return granularity


@dataclass(frozen=True)
class FulltrackPerViewNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    profile_indices: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.profile_indices, dtype=np.int64).reshape(-1)
        if (
            len(mean) == 0
            or not (mean.shape == scale.shape == indices.shape)
            or len(set(indices.tolist())) != len(indices)
            or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0.0)
            or np.any(indices < 0)
        ):
            raise ValueError("full-track per-view normalizer is invalid")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "profile_indices", indices)


def _edge_indices_for_rows(
    features: FrozenFulltrackPerViewAppearanceFeatures, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate-major edge rows and batch-local candidate IDs."""

    query_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        len(query_rows) == 0
        or np.any((query_rows < 0) | (query_rows >= len(features.query_ids)))
        or len(set(query_rows.tolist())) != len(query_rows)
    ):
        raise ValueError("per-view batch rows are invalid")
    edge_chunks: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    candidate_count = features.candidate_count
    for local_row, global_row in enumerate(query_rows.tolist()):
        first_candidate = int(global_row) * candidate_count
        for column in range(candidate_count):
            candidate_index = first_candidate + column
            start = int(features.edge_candidate_offsets[candidate_index])
            end = int(features.edge_candidate_offsets[candidate_index + 1])
            if end <= start:
                continue
            edge_chunks.append(np.arange(start, end, dtype=np.int64))
            candidate_chunks.append(
                np.full(
                    (end - start,),
                    int(local_row) * candidate_count + column,
                    dtype=np.int64,
                )
            )
    if not edge_chunks:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(edge_chunks), np.concatenate(candidate_chunks)


def _iter_edge_indices_for_rows(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    rows: np.ndarray,
    *,
    row_chunk_size: int = 256,
) -> Sequence[np.ndarray]:
    """Yield bounded candidate-major edge chunks for a unique row set."""

    query_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if (
        len(query_rows) == 0
        or int(row_chunk_size) <= 0
        or np.any((query_rows < 0) | (query_rows >= len(features.query_ids)))
        or len(set(query_rows.tolist())) != len(query_rows)
    ):
        raise ValueError("per-view edge chunk rows are invalid")
    return tuple(
        _edge_indices_for_rows(features, query_rows[start : start + int(row_chunk_size)])[0]
        for start in range(0, len(query_rows), int(row_chunk_size))
    )


def fit_fulltrack_per_view_normalizer(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    profile_indices: np.ndarray,
    train_rows: np.ndarray,
) -> FulltrackPerViewNormalizer:
    indices = np.asarray(profile_indices, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or np.any((indices < 0) | (indices >= len(features.profile_names)))
    ):
        raise ValueError("per-view normalizer profile indices are invalid")
    observed_count = 0
    observed_sum = np.zeros((len(indices),), dtype=np.float64)
    observed_square_sum = np.zeros((len(indices),), dtype=np.float64)
    for edge_indices in _iter_edge_indices_for_rows(features, train_rows):
        if len(edge_indices) == 0:
            continue
        raw = np.asarray(
            features.edge_profile_scores[edge_indices][:, indices], dtype=np.float32
        )
        valid = np.asarray(
            features.edge_profile_valid[edge_indices][:, indices], dtype=bool
        )
        usable = np.all(valid, axis=1)
        if not np.any(usable):
            continue
        observed = raw[usable]
        if np.any(~np.isfinite(observed)):
            raise ValueError("per-view normalizer observed values are non-finite")
        values = observed.astype(np.float64, copy=False)
        observed_count += int(len(values))
        observed_sum += values.sum(axis=0, dtype=np.float64)
        observed_square_sum += np.square(values).sum(axis=0, dtype=np.float64)
    if observed_count <= 0:
        raise ValueError("per-view normalizer has no jointly observed train edges")
    mean = observed_sum / float(observed_count)
    variance = np.maximum(
        observed_square_sum / float(observed_count) - np.square(mean), 0.0
    )
    scale = np.maximum(np.sqrt(variance), 1e-3)
    return FulltrackPerViewNormalizer(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
        profile_indices=indices,
    )


def normalized_fulltrack_per_view_edges(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackPerViewNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized raw evidence and a joint-observation mask.

    An edge lacking any requested profile is not supplied to the model at all.
    This makes missing evidence exactly neutral rather than a learned candidate
    cue.  The model sees no availability/count field.
    """

    raw = np.asarray(
        features.edge_profile_scores[:, normalizer.profile_indices], dtype=np.float32
    )
    valid = np.asarray(
        features.edge_profile_valid[:, normalizer.profile_indices], dtype=bool
    )
    usable = np.all(valid, axis=1)
    if np.any(~np.isfinite(raw[usable])):
        raise ValueError("jointly observed full-track edge values are non-finite")
    normalized = np.zeros_like(raw, dtype=np.float32)
    normalized[usable] = (
        raw[usable] - normalizer.mean[None, :]
    ) / normalizer.scale[None, :]
    return normalized, usable


def _normalized_edge_values(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackPerViewNormalizer,
    edge_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize just one bounded CSR edge selection.

    Fitting and prediction previously allocated a full ``edge_count x profile``
    float matrix for each family.  The probe is sparse by construction, so
    normalizing the selected edge batch is equivalent and keeps OOF fitting
    bounded by the caller's row batch instead of the whole cache.
    """

    rows = np.asarray(edge_rows, dtype=np.int64).reshape(-1)
    if len(rows) == 0:
        return (
            np.zeros((0, len(normalizer.profile_indices)), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    if np.any((rows < 0) | (rows >= len(features.edge_profile_scores))):
        raise ValueError("per-view normalized edge rows are invalid")
    raw = np.asarray(
        features.edge_profile_scores[rows][:, normalizer.profile_indices],
        dtype=np.float32,
    )
    valid = np.asarray(
        features.edge_profile_valid[rows][:, normalizer.profile_indices], dtype=bool
    )
    usable = np.all(valid, axis=1)
    if np.any(~np.isfinite(raw[usable])):
        raise ValueError("jointly observed full-track edge values are non-finite")
    return (
        np.asarray(
            (raw[usable] - normalizer.mean[None, :]) / normalizer.scale[None, :],
            dtype=np.float32,
        ),
        usable,
    )


def fulltrack_per_view_joint_edge_coverage(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    profile_indices: np.ndarray,
    edge_chunk_size: int = 262144,
) -> float:
    """Return joint profile coverage without materializing a whole-family mask."""

    indices = np.asarray(profile_indices, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or int(edge_chunk_size) <= 0
        or np.any((indices < 0) | (indices >= len(features.profile_names)))
    ):
        raise ValueError("per-view coverage profile indices are invalid")
    edge_count = int(len(features.edge_profile_valid))
    if edge_count <= 0:
        raise ValueError("per-view coverage has no edges")
    usable_count = 0
    for start in range(0, edge_count, int(edge_chunk_size)):
        valid = np.asarray(
            features.edge_profile_valid[start : start + int(edge_chunk_size), indices],
            dtype=bool,
        )
        usable_count += int(np.count_nonzero(np.all(valid, axis=1)))
    return float(usable_count) / float(edge_count)


@dataclass(frozen=True)
class FulltrackRawTop4RelativeNormalizer:
    """Target-free scales for monotone candidate-versus-top-one top-4 deltas."""

    scale: np.ndarray
    profile_indices: np.ndarray
    top_k: int = 4
    aggregation: str = "uniform_mean"

    def __post_init__(self) -> None:
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.profile_indices, dtype=np.int64).reshape(-1)
        if (
            len(scale) == 0
            or scale.shape != indices.shape
            or len(set(indices.tolist())) != len(indices)
            or np.any(~np.isfinite(scale))
            or np.any(scale <= 0.0)
            or np.any(indices < 0)
            or int(self.top_k) <= 0
            or str(self.aggregation) not in RAW_TOPK_AGGREGATIONS
        ):
            raise ValueError("full-track raw top-4 normalizer is invalid")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "profile_indices", indices)
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "aggregation", str(self.aggregation))


def fulltrack_per_view_topk_aggregate(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    profile_indices: np.ndarray,
    top_k: int = 4,
    aggregation: str = "uniform_mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate each real support-view list without score imputation.

    ``uniform_mean`` matches the full-track exporter's
    ``uniform_top4_mean_ncc`` definition apart from float16 storage
    quantization.  ``lower_envelope`` is the minimum among the same top-k
    real observations, so it can only be high when every retained support view
    agrees.  Neither mode imputes unavailable observations.
    """

    indices = np.asarray(profile_indices, dtype=np.int64).reshape(-1)
    if (
        len(indices) == 0
        or np.any((indices < 0) | (indices >= len(features.profile_names)))
        or len(set(indices.tolist())) != len(indices)
        or int(top_k) <= 0
        or str(aggregation) not in RAW_TOPK_AGGREGATIONS
    ):
        raise ValueError("raw top-k aggregation configuration is invalid")
    row_count, candidate_count = features.candidate_track_ids.shape
    values = np.full((row_count, candidate_count, len(indices)), np.nan, dtype=np.float32)
    valid = np.zeros(values.shape, dtype=bool)
    scores = np.asarray(features.edge_profile_scores[:, indices], dtype=np.float32)
    usable = np.asarray(features.edge_profile_valid[:, indices], dtype=bool)
    for candidate_flat in range(row_count * candidate_count):
        start = int(features.edge_candidate_offsets[candidate_flat])
        end = int(features.edge_candidate_offsets[candidate_flat + 1])
        if end <= start:
            continue
        row, column = divmod(candidate_flat, candidate_count)
        edge_values = scores[start:end]
        edge_valid = usable[start:end]
        for profile in range(len(indices)):
            selected = edge_values[:, profile][edge_valid[:, profile]]
            if len(selected) == 0:
                continue
            count = min(int(top_k), len(selected))
            if count == len(selected):
                top_values = selected
            else:
                top_values = np.partition(selected, len(selected) - count)[-count:]
            values[row, column, profile] = float(
                np.mean(top_values, dtype=np.float64)
                if str(aggregation) == "uniform_mean"
                else np.min(top_values)
            )
            valid[row, column, profile] = True
    if np.any(~np.isfinite(values[valid])) or np.any(np.isfinite(values[~valid])):
        raise RuntimeError("raw top-4 aggregation lost missing-value semantics")
    return values, valid


def fulltrack_per_view_topk_mean(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    profile_indices: np.ndarray,
    top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible uniform top-k mean helper for existing probes."""

    return fulltrack_per_view_topk_aggregate(
        features,
        profile_indices=profile_indices,
        top_k=top_k,
        aggregation="uniform_mean",
    )


def fit_fulltrack_raw_top4_relative_normalizer(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    *,
    profile_indices: np.ndarray,
    train_rows: np.ndarray,
    top_k: int = 4,
    aggregation: str = "uniform_mean",
) -> FulltrackRawTop4RelativeNormalizer:
    """Fit only target-free scales of common top-one-relative raw evidence."""

    rows = np.asarray(train_rows, dtype=np.int64).reshape(-1)
    if (
        len(rows) == 0
        or len(set(rows.tolist())) != len(rows)
        or np.any((rows < 0) | (rows >= len(features.query_ids)))
    ):
        raise ValueError("raw top-4 normalizer train rows are invalid")
    values, valid = fulltrack_per_view_topk_aggregate(
        features,
        profile_indices=profile_indices,
        top_k=int(top_k),
        aggregation=str(aggregation),
    )
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    top_columns = np.argmax(candidate, axis=1).astype(np.int64)
    top_values = values[np.arange(len(values)), top_columns]
    top_valid = valid[np.arange(len(valid)), top_columns]
    active = candidate > 0.0
    common = valid & top_valid[:, None, :] & active[..., None]
    deltas = values - top_values[:, None, :]
    selected = common[rows]
    selected_deltas = deltas[rows]
    count = selected.sum(axis=(0, 1), dtype=np.int64)
    if np.any(count == 0):
        raise ValueError("raw top-4 normalizer lacks common train support")
    masked = np.where(selected, selected_deltas, 0.0).astype(np.float64, copy=False)
    variance = np.maximum(
        np.square(masked).sum(axis=(0, 1), dtype=np.float64) / count,
        0.0,
    )
    return FulltrackRawTop4RelativeNormalizer(
        scale=np.maximum(np.sqrt(variance), 1e-3).astype(np.float32),
        profile_indices=np.asarray(profile_indices, dtype=np.int64),
        top_k=int(top_k),
        aggregation=str(aggregation),
    )


def normalized_fulltrack_raw_top4_relative_features(
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackRawTop4RelativeNormalizer,
) -> tuple[np.ndarray, np.ndarray]:
    """Return neutral-on-missing candidate deltas to the frozen top-one track."""

    values, valid = fulltrack_per_view_topk_aggregate(
        features,
        profile_indices=normalizer.profile_indices,
        top_k=normalizer.top_k,
        aggregation=normalizer.aggregation,
    )
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    top_columns = np.argmax(candidate, axis=1).astype(np.int64)
    top_values = values[np.arange(len(values)), top_columns]
    top_valid = valid[np.arange(len(valid)), top_columns]
    common = valid & top_valid[:, None, :] & (candidate > 0.0)[..., None]
    output = np.zeros_like(values, dtype=np.float32)
    output[common] = (
        values[common] - np.broadcast_to(top_values[:, None, :], values.shape)[common]
    ) / np.broadcast_to(normalizer.scale[None, None, :], values.shape)[common]
    output[np.arange(len(output)), top_columns] = 0.0
    if np.any(~np.isfinite(output)):
        raise RuntimeError("raw top-4 relative features are non-finite")
    return output, common


def fixed_candidate_conditional_log_priors(
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(features.candidate_probabilities, dtype=np.float32)
    mass = candidate.sum(axis=1, dtype=np.float32)
    expected = 1.0 - np.asarray(features.null_probabilities, dtype=np.float32)
    if np.max(np.abs(mass - expected)) > 2e-5:
        raise ValueError("full-track per-view candidate and null mass disagree")
    output = np.full(candidate.shape, -np.inf, dtype=np.float32)
    active = candidate > 0.0
    conditional = np.divide(
        candidate,
        mass[:, None],
        out=np.zeros_like(candidate),
        where=mass[:, None] > 0.0,
    )
    output[active] = np.log(conditional[active])
    return output, mass


class SparseFulltrackPerViewResidual(nn.Module):
    """Candidate-mass-preserving residual with an order-invariant view mixture."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        residual_architecture: str = "mlp",
        residual_cap: float | None = None,
    ) -> None:
        super().__init__()
        if (
            int(input_dim) <= 0
            or int(hidden_dim) <= 0
            or str(residual_architecture) not in PER_VIEW_RESIDUAL_ARCHITECTURES
            or (
                residual_cap is not None
                and (
                    not np.isfinite(float(residual_cap))
                    or float(residual_cap) <= 0.0
                )
            )
        ):
            raise ValueError("sparse full-track per-view dimensions must be positive")
        self.residual_architecture = str(residual_architecture)
        if self.residual_architecture == "linear":
            self.backbone = nn.Identity()
            output_dim = int(input_dim)
        else:
            self.backbone = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.SiLU(),
            )
            output_dim = int(hidden_dim)
        self.evidence = nn.Linear(output_dim, 1)
        self.view = nn.Linear(output_dim, 1)
        # ``nan`` encodes the legacy unbounded mode in the state dict.  A
        # bounded residual is a log-likelihood-ratio safety contract, not a
        # post-hoc validation scaling operation.
        self.register_buffer(
            "residual_cap_tensor",
            torch.tensor(
                float("nan") if residual_cap is None else float(residual_cap),
                dtype=torch.float32,
            ),
        )
        # At initialization every materialized support view has likelihood
        # ratio one.  Thus the whole model is a bit-exact fixed-posterior
        # fallback before train-only supervision changes it.
        nn.init.zeros_(self.evidence.weight)
        nn.init.zeros_(self.evidence.bias)
        nn.init.zeros_(self.view.weight)
        nn.init.zeros_(self.view.bias)

    def _candidate_residual(
        self,
        edge_features: torch.Tensor,
        edge_candidate_indices: torch.Tensor,
        candidate_total: int,
    ) -> torch.Tensor:
        residual = torch.zeros(
            (int(candidate_total),), dtype=torch.float32, device=edge_features.device
        )
        if int(edge_features.shape[0]) == 0:
            return residual
        if (
            edge_features.ndim != 2
            or edge_candidate_indices.ndim != 1
            or int(edge_candidate_indices.numel()) != int(edge_features.shape[0])
            or int(candidate_total) <= 0
            or bool(torch.any(edge_candidate_indices < 0))
            or bool(torch.any(edge_candidate_indices >= int(candidate_total)))
        ):
            raise ValueError("sparse full-track per-view edge tensors are invalid")
        hidden = self.backbone(edge_features.float())
        evidence = self.evidence(hidden).reshape(-1).float()
        view = self.view(hidden).reshape(-1).float()
        candidates = edge_candidate_indices.to(device=view.device, dtype=torch.long)
        max_view = torch.full_like(residual, -torch.inf)
        max_view.scatter_reduce_(0, candidates, view, reduce="amax", include_self=True)
        view_sum = torch.zeros_like(residual)
        view_sum.scatter_add_(0, candidates, torch.exp(view - max_view[candidates]))
        combined = view + evidence
        max_combined = torch.full_like(residual, -torch.inf)
        max_combined.scatter_reduce_(
            0, candidates, combined, reduce="amax", include_self=True
        )
        combined_sum = torch.zeros_like(residual)
        combined_sum.scatter_add_(
            0, candidates, torch.exp(combined - max_combined[candidates])
        )
        present = view_sum > 0.0
        residual[present] = (
            max_combined[present]
            + torch.log(combined_sum[present].clamp_min(1e-30))
            - max_view[present]
            - torch.log(view_sum[present].clamp_min(1e-30))
        )
        residual_cap = self.residual_cap_tensor.to(
            device=residual.device, dtype=residual.dtype
        )
        if bool(torch.isfinite(residual_cap)):
            residual = residual_cap * torch.tanh(residual / residual_cap)
        return residual

    def forward(
        self,
        *,
        edge_features: torch.Tensor,
        edge_candidate_indices: torch.Tensor,
        candidate_conditional_log_prior: torch.Tensor,
        candidate_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            candidate_conditional_log_prior.ndim != 2
            or candidate_mass.shape != (candidate_conditional_log_prior.shape[0],)
            or bool(torch.any(candidate_mass < 0.0))
        ):
            raise ValueError("full-track per-view candidate priors are invalid")
        batch_size, candidate_count = candidate_conditional_log_prior.shape
        residual = self._candidate_residual(
            edge_features,
            edge_candidate_indices,
            int(batch_size) * int(candidate_count),
        ).reshape(batch_size, candidate_count)
        active = torch.isfinite(candidate_conditional_log_prior)
        logits = candidate_conditional_log_prior + residual
        safe_logits = torch.where(active, logits, torch.full_like(logits, -torch.inf))
        zero_mass = candidate_mass <= 0.0
        safe_logits = torch.where(
            zero_mass[:, None], torch.zeros_like(safe_logits), safe_logits
        )
        conditional = torch.softmax(safe_logits, dim=1)
        conditional = torch.where(active, conditional, torch.zeros_like(conditional))
        candidate = conditional * candidate_mass[:, None]
        return candidate, residual, safe_logits


class MonotonicTop1RelativeTop4Residual(nn.Module):
    """Constrained raw top-4 overlay against the frozen rank-one candidate.

    The model has no candidate bias, candidate index, support count, or
    availability input.  Signed mode is retained as a diagnostic.  Positive
    uplift mode clips negative relative evidence, so only a candidate whose
    raw all-observation top-4 score exceeds the immutable top-one track can
    receive a visual boost.  Missing common evidence is an exact zero delta.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        initial_weight: float = MONOTONIC_TOP4_INITIAL_WEIGHT,
        positive_uplift: bool = False,
        residual_scale: float = 1.0,
        residual_cap: float | None = None,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or not np.isfinite(float(initial_weight)) or float(initial_weight) <= 0.0:
            raise ValueError("monotonic raw top-4 input dimension must be positive")
        # Starting at softplus(0) would inject a large, uncalibrated residual
        # and AdamW decay would keep irrelevant profiles near that value.  A
        # small positive initialization preserves nonzero gradients while the
        # no-decay optimizer below lets a profile be suppressed toward zero.
        initial_logit = float(np.log(np.expm1(float(initial_weight))))
        self.weight_logits = nn.Parameter(
            torch.full((int(input_dim),), initial_logit, dtype=torch.float32)
        )
        self.positive_uplift = bool(positive_uplift)
        # A unit scale and zero cap are the exact unbounded identity transform.
        # Buffers make train-time or post-fit calibration part of checkpoint
        # state instead of an evaluator-side score modification.
        self.register_buffer("residual_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("residual_cap", torch.zeros((), dtype=torch.float32))
        self.set_residual_scale(residual_scale)
        self.set_residual_cap(residual_cap)

    def weights(self) -> torch.Tensor:
        return F.softplus(self.weight_logits)

    def set_residual_cap(self, residual_cap: float | None) -> None:
        if residual_cap is None:
            value = 0.0
        else:
            value = float(residual_cap)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("monotonic raw top-4 residual cap must be positive")
        with torch.no_grad():
            self.residual_cap.fill_(value)

    def set_residual_scale(self, residual_scale: float) -> None:
        value = float(residual_scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("monotonic raw top-4 residual scale must be positive")
        with torch.no_grad():
            self.residual_scale.fill_(value)

    def forward(
        self,
        *,
        relative_features: torch.Tensor,
        candidate_conditional_log_prior: torch.Tensor,
        candidate_mass: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            relative_features.ndim != 3
            or candidate_conditional_log_prior.shape != relative_features.shape[:2]
            or candidate_mass.shape != (relative_features.shape[0],)
            or bool(torch.any(candidate_mass < 0.0))
        ):
            raise ValueError("monotonic raw top-4 tensors are incompatible")
        evidence = relative_features.float()
        if self.positive_uplift:
            evidence = F.relu(evidence)
        residual = (
            torch.sum(evidence * self.weights()[None, None, :], dim=2)
            * self.residual_scale
        )
        if bool(self.residual_cap > 0.0):
            residual = self.residual_cap * torch.tanh(residual / self.residual_cap)
        active = torch.isfinite(candidate_conditional_log_prior)
        logits = candidate_conditional_log_prior + residual
        safe_logits = torch.where(active, logits, torch.full_like(logits, -torch.inf))
        safe_logits = torch.where(
            (candidate_mass <= 0.0)[:, None],
            torch.zeros_like(safe_logits),
            safe_logits,
        )
        conditional = torch.softmax(safe_logits, dim=1)
        conditional = torch.where(active, conditional, torch.zeros_like(conditional))
        candidate = conditional * candidate_mass[:, None]
        return candidate, residual, safe_logits


def _apply_monotonic_top4_postfit_scale(
    model: MonotonicTop1RelativeTop4Residual, *, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bake a train-selected residual scale into the constrained model state."""

    if not np.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("raw top-4 post-fit residual scale must be positive")
    with torch.no_grad():
        raw = model.weights()
        effective = raw * float(scale)
        # Stable inverse softplus for positive, potentially large weights.
        model.weight_logits.copy_(effective + torch.log(-torch.expm1(-effective)))
    return (
        raw.detach().cpu().numpy().astype(np.float32, copy=True),
        model.weights().detach().cpu().numpy().astype(np.float32, copy=True),
    )


def _candidate_membership_nll(logits: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
    if logits.shape != membership.shape or logits.ndim != 2:
        raise ValueError("full-track per-view membership tensors are incompatible")
    if bool(torch.any(membership.sum(dim=1) != 1)):
        raise ValueError("full-track per-view probe needs one retrieved target per row")
    return -torch.logsumexp(
        torch.where(
            membership,
            F.log_softmax(logits, dim=1),
            torch.full_like(logits, -torch.inf),
        ),
        dim=1,
    ).mean()


def _hard_pair_columns(
    candidate_probabilities: np.ndarray, membership: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    labels = np.asarray(membership, dtype=bool)
    if probabilities.shape != labels.shape or np.any(labels.sum(axis=1) != 1):
        raise ValueError("full-track per-view hard-pair inputs are invalid")
    positive = np.argmax(labels, axis=1).astype(np.int64)
    top = np.argmax(probabilities, axis=1).astype(np.int64)
    negative = np.where(top == positive, -1, top).astype(np.int64)
    return positive, negative


def _coarse_top1_stability_loss(
    *,
    logits: torch.Tensor,
    membership: torch.Tensor,
    coarse_top1_columns: torch.Tensor,
) -> torch.Tensor:
    """Keep a train-verified coarse top-one above every wrong candidate.

    The rank-2 rescue loss is evaluated only where the frozen top-one is
    wrong.  This complementary term is evaluated only where that top-one is
    correct, and compares it with the currently strongest wrong candidate.
    It therefore fixes the objective asymmetry without exposing labels during
    prediction or changing the fixed candidate posterior mass.
    """

    if (
        logits.ndim != 2
        or membership.shape != logits.shape
        or membership.dtype != torch.bool
        or coarse_top1_columns.shape != (logits.shape[0],)
        or coarse_top1_columns.dtype != torch.long
        or bool(torch.any(membership.sum(dim=1) != 1))
        or bool(
            torch.any(
                (coarse_top1_columns < 0)
                | (coarse_top1_columns >= logits.shape[1])
            )
        )
    ):
        raise ValueError("coarse top-one stability tensors are incompatible")
    correct_columns = torch.argmax(membership.to(dtype=torch.int64), dim=1)
    stable_rows = coarse_top1_columns == correct_columns
    if not bool(torch.any(stable_rows)):
        return logits.new_zeros(())
    selected_logits = logits[stable_rows]
    selected_membership = membership[stable_rows]
    correct = selected_logits[
        torch.arange(len(selected_logits), device=logits.device),
        correct_columns[stable_rows],
    ]
    strongest_wrong = torch.where(
        selected_membership,
        torch.full_like(selected_logits, -torch.inf),
        selected_logits,
    ).max(dim=1).values
    if not bool(torch.isfinite(correct).all() & torch.isfinite(strongest_wrong).all()):
        raise ValueError("coarse top-one stability pair is not finite")
    return F.softplus(strongest_wrong - correct).mean()


def _batch_edges(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackPerViewNormalizer,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_rows, local_candidates = _edge_indices_for_rows(features, rows)
    if len(edge_rows) == 0:
        return (
            np.zeros((0, len(normalizer.profile_indices)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    edge_values, keep = _normalized_edge_values(
        features=features,
        normalizer=normalizer,
        edge_rows=edge_rows,
    )
    return (
        edge_values,
        np.asarray(local_candidates[keep], dtype=np.int64),
    )


def _cache_target_edges(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackPerViewNormalizer,
    target_rows: np.ndarray,
    row_chunk_size: int = 64,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Cache normalized edges for the repeatedly sampled supervised rows.

    Training revisits the same singleton-identity rows for every epoch.  Keep
    only those rows in memory, while prediction and normalizer fitting remain
    streamable over the full immutable CSR.  Candidate columns are retained
    locally so a shuffled optimizer batch can rebuild its compact scatter IDs
    without re-reading the feature cache.
    """

    rows = np.asarray(target_rows, dtype=np.int64).reshape(-1)
    if (
        len(rows) == 0
        or int(row_chunk_size) <= 0
        or len(set(rows.tolist())) != len(rows)
        or np.any((rows < 0) | (rows >= len(features.query_ids)))
    ):
        raise ValueError("per-view target edge cache rows are invalid")
    values: list[np.ndarray | None] = [None] * len(rows)
    columns: list[np.ndarray | None] = [None] * len(rows)
    candidate_count = int(features.candidate_count)
    for begin in range(0, len(rows), int(row_chunk_size)):
        end = min(begin + int(row_chunk_size), len(rows))
        edge_rows, local_candidates = _edge_indices_for_rows(features, rows[begin:end])
        normalized, keep = _normalized_edge_values(
            features=features,
            normalizer=normalizer,
            edge_rows=edge_rows,
        )
        candidates = np.asarray(local_candidates[keep], dtype=np.int64)
        for local in range(end - begin):
            selection = candidates // candidate_count == local
            values[begin + local] = np.asarray(normalized[selection], dtype=np.float32)
            columns[begin + local] = np.asarray(
                candidates[selection] % candidate_count, dtype=np.int64
            )
    if any(value is None for value in values) or any(value is None for value in columns):
        raise RuntimeError("per-view target edge cache is incomplete")
    return (
        tuple(np.asarray(value, dtype=np.float32) for value in values if value is not None),
        tuple(np.asarray(value, dtype=np.int64) for value in columns if value is not None),
    )


def _cached_batch_edges(
    *,
    cached_values: Sequence[np.ndarray],
    cached_columns: Sequence[np.ndarray],
    selected_target_indices: np.ndarray,
    candidate_count: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one shuffled optimizer batch from a target-only edge cache."""

    selected = np.asarray(selected_target_indices, dtype=np.int64).reshape(-1)
    if (
        len(selected) == 0
        or len(cached_values) != len(cached_columns)
        or int(candidate_count) <= 0
        or int(feature_dim) <= 0
        or np.any((selected < 0) | (selected >= len(cached_values)))
    ):
        raise ValueError("per-view cached batch inputs are invalid")
    value_blocks: list[np.ndarray] = []
    candidate_blocks: list[np.ndarray] = []
    for local_batch_row, target_index in enumerate(selected.tolist()):
        values = np.asarray(cached_values[target_index], dtype=np.float32)
        columns = np.asarray(cached_columns[target_index], dtype=np.int64)
        if values.shape != (len(columns), int(feature_dim)):
            raise ValueError("per-view cached target edge layout is invalid")
        if len(columns) == 0:
            continue
        value_blocks.append(values)
        candidate_blocks.append(columns + int(local_batch_row) * int(candidate_count))
    if not value_blocks:
        return (
            np.zeros((0, int(feature_dim)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.concatenate(value_blocks, axis=0), np.concatenate(candidate_blocks, axis=0)


def fit_fixedprior_fulltrack_per_view_probe(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    family: str,
    train_normalizer_rows: np.ndarray,
    target_train_rows: np.ndarray,
    target_candidate_membership: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    seed: int,
    rank2_hard_pair_weight: float | None = None,
    residual_architecture: str = "mlp",
    residual_cap: float | None = None,
) -> tuple[SparseFulltrackPerViewResidual, FulltrackPerViewNormalizer, dict[str, Any]]:
    spec = FULLTRACK_PER_VIEW_FAMILIES.get(str(family))
    if (
        spec is None
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or int(hidden_dim) <= 0
        or str(residual_architecture) not in PER_VIEW_RESIDUAL_ARCHITECTURES
        or (
            residual_cap is not None
            and (
                not np.isfinite(float(residual_cap))
                or float(residual_cap) <= 0.0
            )
        )
        or (
            rank2_hard_pair_weight is not None
            and float(rank2_hard_pair_weight) <= 0.0
        )
    ):
        raise ValueError("full-track per-view fit configuration is invalid")
    if str(spec.edge_feature_semantics) != str(
        features.compatibility.get("per_view_edge_feature_semantics", "")
    ):
        raise ValueError("per-view family and artifact edge semantics differ")
    target_rows = np.asarray(target_train_rows, dtype=np.int64).reshape(-1)
    membership = np.asarray(target_candidate_membership, dtype=bool)
    if (
        len(target_rows) == 0
        or len(set(target_rows.tolist())) != len(target_rows)
        or np.any((target_rows < 0) | (target_rows >= len(features.query_ids)))
        or membership.shape != (len(target_rows), features.candidate_count)
        or np.any(membership.sum(axis=1) != 1)
    ):
        raise ValueError("full-track per-view train targets are invalid")
    profile_indices = profile_indices_for_fulltrack_per_view_family(
        str(family), profile_names=features.profile_names
    )
    normalizer = fit_fulltrack_per_view_normalizer(
        features,
        profile_indices=profile_indices,
        train_rows=np.asarray(train_normalizer_rows, dtype=np.int64),
    )
    joint_edge_coverage = fulltrack_per_view_joint_edge_coverage(
        features,
        profile_indices=normalizer.profile_indices,
    )
    if joint_edge_coverage < MINIMUM_JOINT_EDGE_COVERAGE:
        raise ValueError(
            "per-view family has insufficient jointly observed edge coverage: "
            f"{joint_edge_coverage:.6f} < {MINIMUM_JOINT_EDGE_COVERAGE:.6f}"
        )
    conditional_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    effective_hard_pair_weight = (
        float(spec.rank2_hard_pair_weight)
        if rank2_hard_pair_weight is None
        else float(rank2_hard_pair_weight)
    )
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = SparseFulltrackPerViewResidual(
        len(normalizer.profile_indices),
        hidden_dim=int(hidden_dim),
        residual_architecture=str(residual_architecture),
        residual_cap=residual_cap,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    cached_values, cached_columns = _cache_target_edges(
        features=features,
        normalizer=normalizer,
        target_rows=target_rows,
    )
    positive_columns, negative_columns = _hard_pair_columns(
        features.candidate_probabilities[target_rows], membership
    )
    coarse_top1_columns = np.argmax(
        features.candidate_probabilities[target_rows], axis=1
    ).astype(np.int64)
    coarse_top1_correct = coarse_top1_columns == positive_columns
    effective_coarse_top1_stability_weight = float(
        spec.coarse_top1_stability_weight
    )
    rng = np.random.default_rng(int(seed))
    final_nll = float("nan")
    final_hard = float("nan")
    final_stability = float("nan")
    final_total = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(target_rows))
        for start in range(0, len(order), int(batch_size)):
            selected = order[start : start + int(batch_size)]
            rows = target_rows[selected]
            edge_values, edge_candidates = _cached_batch_edges(
                cached_values=cached_values,
                cached_columns=cached_columns,
                selected_target_indices=selected,
                candidate_count=features.candidate_count,
                feature_dim=len(normalizer.profile_indices),
            )
            candidate, _residual, logits = model(
                edge_features=torch.from_numpy(edge_values).to(device),
                edge_candidate_indices=torch.from_numpy(edge_candidates).to(device),
                candidate_conditional_log_prior=torch.from_numpy(
                    conditional_log_prior[rows]
                ).to(device),
                candidate_mass=torch.from_numpy(candidate_mass[rows]).to(device),
            )
            del candidate
            membership_tensor = torch.from_numpy(membership[selected]).to(
                device=device, dtype=torch.bool
            )
            identity_loss = _candidate_membership_nll(logits, membership_tensor)
            negative = negative_columns[selected]
            hard_mask = negative >= 0
            if np.any(hard_mask):
                local = torch.from_numpy(np.flatnonzero(hard_mask)).to(device)
                positives = torch.from_numpy(positive_columns[selected][hard_mask]).to(device)
                negatives = torch.from_numpy(negative[hard_mask]).to(device)
                hard_loss = F.softplus(
                    -(logits[local, positives] - logits[local, negatives])
                ).mean()
            else:
                hard_loss = identity_loss.new_zeros(())
            if effective_coarse_top1_stability_weight > 0.0:
                stability_loss = _coarse_top1_stability_loss(
                    logits=logits,
                    membership=membership_tensor,
                    coarse_top1_columns=torch.from_numpy(
                        coarse_top1_columns[selected]
                    ).to(device=device, dtype=torch.long),
                )
            else:
                stability_loss = identity_loss.new_zeros(())
            loss = (
                identity_loss
                + effective_hard_pair_weight * hard_loss
                + effective_coarse_top1_stability_weight * stability_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            final_nll = float(identity_loss.detach().cpu())
            final_hard = float(hard_loss.detach().cpu())
            final_stability = float(stability_loss.detach().cpu())
            final_total = float(loss.detach().cpu())
    model.eval()
    return model, normalizer, {
        "family": str(family),
        "architecture": (
            "sparse_per_view_linear_learned_logsumexp_mixture_v1"
            if str(residual_architecture) == "linear" and residual_cap is None
            else (
                "sparse_per_view_linear_bounded_residual_learned_logsumexp_mixture_v1"
                if str(residual_architecture) == "linear"
                else (
                    "sparse_per_view_mlp_bounded_residual_learned_logsumexp_mixture_v1"
                    if residual_cap is not None
                    else "sparse_per_view_mlp_learned_logsumexp_mixture_v1"
                )
            )
        ),
        "residual_architecture": str(residual_architecture),
        "residual_cap": None if residual_cap is None else float(residual_cap),
        "residual_cap_semantics": (
            "unbounded_legacy_log_likelihood_ratio_v1"
            if residual_cap is None
            else "symmetric_tanh_bounded_log_likelihood_ratio_v1"
        ),
        "profile_names": [
            features.profile_names[index] for index in normalizer.profile_indices.tolist()
        ],
        "profile_count": int(len(normalizer.profile_indices)),
        "joint_edge_coverage": joint_edge_coverage,
        "minimum_joint_edge_coverage": MINIMUM_JOINT_EDGE_COVERAGE,
        "train_identity_row_count": int(len(target_rows)),
        "train_normalizer_row_count": int(len(np.asarray(train_normalizer_rows))),
        "rank2_to_top1_wrong_train_pair_count": int(np.sum(negative_columns >= 0)),
        "rank2_hard_pair_weight": effective_hard_pair_weight,
        "family_default_rank2_hard_pair_weight": float(spec.rank2_hard_pair_weight),
        "coarse_top1_stability_weight": effective_coarse_top1_stability_weight,
        "coarse_top1_correct_train_row_count": int(np.count_nonzero(coarse_top1_correct)),
        "training_objective": (
            "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
            if effective_coarse_top1_stability_weight > 0.0
            else (
                "identity_nll_plus_rank2_rescue_v1"
                if effective_hard_pair_weight > 0.0
                else "identity_nll_v1"
            )
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "last_train_identity_nll": final_nll,
        "last_train_rank2_hard_pair_loss": final_hard,
        "last_train_coarse_top1_stability_loss": final_stability,
        "last_train_total_loss": final_total,
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "support_view_marginalization": "learned_logsumexp_over_retained_real_observation_edges_v1",
        "missing_evidence_semantics": "joint_profile_missing_edge_omitted_neutral_v1",
        "per_view_model": True,
        "zero_residual_reproduces_fixed_posterior": True,
    }


@torch.inference_mode()
def predict_fixedprior_fulltrack_per_view_probe(
    *,
    model: SparseFulltrackPerViewResidual,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackPerViewNormalizer,
    device: torch.device,
    batch_size: int,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(batch_size) <= 0:
        raise ValueError("full-track per-view prediction batch size must be positive")
    conditional_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    selected_rows = (
        np.arange(len(features.query_ids), dtype=np.int64)
        if rows is None
        else np.asarray(rows, dtype=np.int64).reshape(-1)
    )
    if (
        len(selected_rows) == 0
        or len(set(selected_rows.tolist())) != len(selected_rows)
        or np.any((selected_rows < 0) | (selected_rows >= len(features.query_ids)))
    ):
        raise ValueError("full-track per-view prediction rows are invalid")
    candidate_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(selected_rows), int(batch_size)):
        end = min(start + int(batch_size), len(selected_rows))
        batch_rows = selected_rows[start:end]
        edge_values, edge_candidates = _batch_edges(
            features=features,
            normalizer=normalizer,
            rows=batch_rows,
        )
        candidate, residual, _logits = model(
            edge_features=torch.from_numpy(edge_values).to(device),
            edge_candidate_indices=torch.from_numpy(edge_candidates).to(device),
            candidate_conditional_log_prior=torch.from_numpy(
                conditional_log_prior[batch_rows]
            ).to(device),
            candidate_mass=torch.from_numpy(candidate_mass[batch_rows]).to(device),
        )
        candidate_blocks.append(candidate.cpu().numpy().astype(np.float32))
        residual_blocks.append(residual.cpu().numpy().astype(np.float32))
    candidate = np.concatenate(candidate_blocks, axis=0)
    null = np.asarray(features.null_probabilities[selected_rows], dtype=np.float32).copy()
    if (
        np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities[selected_rows])) > 0.0
    ):
        raise RuntimeError("full-track per-view prediction changed fixed posterior mass")
    return candidate, null, np.concatenate(residual_blocks, axis=0)


def zero_residual_fulltrack_per_view_posterior(
    features: FrozenFulltrackPerViewAppearanceFeatures,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(features.candidate_probabilities, dtype=np.float32).copy(),
        np.asarray(features.null_probabilities, dtype=np.float32).copy(),
    )


def fit_fixedprior_fulltrack_rawtop4_probe(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    family: str,
    train_normalizer_rows: np.ndarray,
    target_train_rows: np.ndarray,
    target_candidate_membership: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    rank2_hard_pair_weight: float | None = None,
    postfit_residual_scale: float = 1.0,
    postfit_residual_cap: float | None = None,
) -> tuple[
    MonotonicTop1RelativeTop4Residual,
    FulltrackRawTop4RelativeNormalizer,
    dict[str, Any],
]:
    """Fit nonnegative raw top-4 evidence inside the fixed candidate mass."""

    spec = FULLTRACK_PER_VIEW_FAMILIES.get(str(family))
    target_rows = np.asarray(target_train_rows, dtype=np.int64).reshape(-1)
    membership = np.asarray(target_candidate_membership, dtype=bool)
    if (
        spec is None
        or spec.architecture not in RAW_TOP4_ARCHITECTURES
        or str(spec.edge_feature_semantics)
        != str(features.compatibility.get("per_view_edge_feature_semantics", ""))
        or len(target_rows) == 0
        or len(set(target_rows.tolist())) != len(target_rows)
        or np.any((target_rows < 0) | (target_rows >= len(features.query_ids)))
        or membership.shape != (len(target_rows), features.candidate_count)
        or np.any(membership.sum(axis=1) != 1)
        or int(epochs) <= 0
        or int(batch_size) <= 0
        or float(learning_rate) <= 0.0
        or float(weight_decay) < 0.0
        or not np.isfinite(float(postfit_residual_scale))
        or float(postfit_residual_scale) <= 0.0
        or (
            postfit_residual_cap is not None
            and (
                not np.isfinite(float(postfit_residual_cap))
                or float(postfit_residual_cap) <= 0.0
            )
        )
        or (
            rank2_hard_pair_weight is not None
            and float(rank2_hard_pair_weight) <= 0.0
        )
    ):
        raise ValueError("raw top-4 probe fit configuration is invalid")
    bounded = spec.architecture in {
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    }
    calibration_during_train = bool(spec.calibration_during_train)
    if bounded != (postfit_residual_cap is not None):
        raise ValueError(
            "bounded raw top-4 family must use a cap and unbounded families must not"
        )
    profile_indices = profile_indices_for_fulltrack_per_view_family(
        str(family), profile_names=features.profile_names
    )
    normalizer = fit_fulltrack_raw_top4_relative_normalizer(
        features,
        profile_indices=profile_indices,
        train_rows=np.asarray(train_normalizer_rows, dtype=np.int64),
        aggregation=spec.raw_topk_aggregation,
    )
    relative_features, common = normalized_fulltrack_raw_top4_relative_features(
        features, normalizer
    )
    profile_common_coverage = np.mean(common, axis=(0, 1), dtype=np.float64)
    conditional_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    positive_columns, negative_columns = _hard_pair_columns(
        features.candidate_probabilities[target_rows], membership
    )
    effective_hard_pair_weight = (
        float(spec.rank2_hard_pair_weight)
        if rank2_hard_pair_weight is None
        else float(rank2_hard_pair_weight)
    )
    effective_coarse_top1_stability_weight = float(
        spec.coarse_top1_stability_weight
    )
    coarse_top1_columns = np.argmax(
        features.candidate_probabilities[target_rows], axis=1
    ).astype(np.int64)
    coarse_top1_correct = coarse_top1_columns == positive_columns
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    positive_uplift = spec.architecture in {
        RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    }
    model = MonotonicTop1RelativeTop4Residual(
        int(relative_features.shape[2]),
        positive_uplift=positive_uplift,
        residual_scale=(
            float(postfit_residual_scale) if calibration_during_train else 1.0
        ),
        residual_cap=(postfit_residual_cap if calibration_during_train else None),
    ).to(device)
    # Decoupled weight decay on a softplus logit pulls it toward zero logits,
    # i.e. a weight of about 0.693, rather than toward a zero evidence weight.
    # Keep this constrained calibrator unregularized so an irrelevant profile
    # can remain close to zero.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=0.0
    )
    rng = np.random.default_rng(int(seed))
    final_nll = float("nan")
    final_hard = float("nan")
    final_stability = float("nan")
    final_total = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        order = rng.permutation(len(target_rows))
        for start in range(0, len(order), int(batch_size)):
            selected = order[start : start + int(batch_size)]
            rows = target_rows[selected]
            _candidate, _residual, logits = model(
                relative_features=torch.from_numpy(relative_features[rows]).to(device),
                candidate_conditional_log_prior=torch.from_numpy(
                    conditional_log_prior[rows]
                ).to(device),
                candidate_mass=torch.from_numpy(candidate_mass[rows]).to(device),
            )
            membership_tensor = torch.from_numpy(membership[selected]).to(
                device=device, dtype=torch.bool
            )
            identity_loss = _candidate_membership_nll(logits, membership_tensor)
            negative = negative_columns[selected]
            hard_mask = negative >= 0
            if np.any(hard_mask):
                local = torch.from_numpy(np.flatnonzero(hard_mask)).to(device)
                positives = torch.from_numpy(
                    positive_columns[selected][hard_mask]
                ).to(device)
                negatives = torch.from_numpy(negative[hard_mask]).to(device)
                hard_loss = F.softplus(
                    -(logits[local, positives] - logits[local, negatives])
                ).mean()
            else:
                hard_loss = identity_loss.new_zeros(())
            if effective_coarse_top1_stability_weight > 0.0:
                stability_loss = _coarse_top1_stability_loss(
                    logits=logits,
                    membership=membership_tensor,
                    coarse_top1_columns=torch.from_numpy(
                        coarse_top1_columns[selected]
                    ).to(device=device, dtype=torch.long),
                )
            else:
                stability_loss = identity_loss.new_zeros(())
            loss = (
                identity_loss
                + effective_hard_pair_weight * hard_loss
                + effective_coarse_top1_stability_weight * stability_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            final_nll = float(identity_loss.detach().cpu())
            final_hard = float(hard_loss.detach().cpu())
            final_stability = float(stability_loss.detach().cpu())
            final_total = float(loss.detach().cpu())
    model.eval()
    if calibration_during_train:
        raw_fit_weights = model.weights().detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        effective_weights = (
            raw_fit_weights * float(postfit_residual_scale)
        ).astype(np.float32, copy=False)
    else:
        raw_fit_weights, effective_weights = _apply_monotonic_top4_postfit_scale(
            model, scale=float(postfit_residual_scale)
        )
        model.set_residual_cap(postfit_residual_cap)
    if (
        spec.architecture
        == RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE
    ):
        metadata_architecture = (
            "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_balanced_overlay_v1"
        )
    elif calibration_during_train:
        metadata_architecture = (
            "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_overlay_v1"
        )
    elif bounded:
        metadata_architecture = (
            "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_overlay_v1"
        )
    elif positive_uplift:
        metadata_architecture = "monotonic_top1_relative_positive_uplift_top4_overlay_v1"
    else:
        metadata_architecture = "monotonic_top1_relative_raw_top4_overlay_v1"
    if calibration_during_train:
        candidate_evidence_transform = (
            "relu_positive_relative_uplift_tanh_bounded_traincal_v1"
        )
    elif bounded:
        candidate_evidence_transform = "relu_positive_relative_uplift_tanh_bounded_v1"
    elif positive_uplift:
        candidate_evidence_transform = "relu_positive_relative_uplift_v1"
    else:
        candidate_evidence_transform = "signed_relative_delta_v1"
    training_objective = (
        "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
        if effective_coarse_top1_stability_weight > 0.0
        else "identity_nll_plus_rank2_rescue_v1"
    )
    return model, normalizer, {
        "family": str(family),
        "architecture": metadata_architecture,
        "candidate_evidence_transform": candidate_evidence_transform,
        "profile_names": [
            features.profile_names[index] for index in normalizer.profile_indices.tolist()
        ],
        "profile_count": int(len(normalizer.profile_indices)),
        "profile_common_coverage": {
            features.profile_names[index]: float(coverage)
            for index, coverage in zip(
                normalizer.profile_indices.tolist(), profile_common_coverage.tolist()
            )
        },
        "top_k": int(normalizer.top_k),
        "raw_topk_aggregation": str(normalizer.aggregation),
        "train_identity_row_count": int(len(target_rows)),
        "train_normalizer_row_count": int(len(np.asarray(train_normalizer_rows))),
        "rank2_to_top1_wrong_train_pair_count": int(np.sum(negative_columns >= 0)),
        "rank2_hard_pair_weight": effective_hard_pair_weight,
        "family_default_rank2_hard_pair_weight": float(spec.rank2_hard_pair_weight),
        "coarse_top1_correct_train_pair_count": int(np.sum(coarse_top1_correct)),
        "coarse_top1_stability_weight": effective_coarse_top1_stability_weight,
        "family_default_coarse_top1_stability_weight": float(
            spec.coarse_top1_stability_weight
        ),
        "training_objective": training_objective,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "requested_weight_decay": float(weight_decay),
        "effective_weight_decay": 0.0,
        "initial_monotonic_weight": MONOTONIC_TOP4_INITIAL_WEIGHT,
        "seed": int(seed),
        "last_train_identity_nll": final_nll,
        "last_train_rank2_hard_pair_loss": final_hard,
        "last_train_coarse_top1_stability_loss": final_stability,
        "last_train_total_loss": final_total,
        "raw_fit_monotonic_nonnegative_weights": raw_fit_weights.tolist(),
        "postfit_residual_scale": float(postfit_residual_scale),
        "postfit_scale_applied_after_train": bool(
            not calibration_during_train
            and float(postfit_residual_scale) != 1.0
        ),
        "postfit_residual_cap": (
            None if postfit_residual_cap is None else float(postfit_residual_cap)
        ),
        "postfit_cap_applied_after_train": bool(
            bounded and not calibration_during_train
        ),
        "residual_calibration_applied_during_train": calibration_during_train,
        "monotonic_nonnegative_weights": effective_weights.tolist(),
        "null_handling": "input_null_probability_exactly_preserved_v1",
        "candidate_mass_handling": "input_nonnull_mass_exactly_preserved_v1",
        "support_view_marginalization": RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
            normalizer.aggregation
        ],
        "missing_evidence_semantics": "common_top1_profile_missing_zero_residual_v1",
        "per_view_model": True,
        "zero_residual_reproduces_fixed_posterior": False,
    }


@torch.inference_mode()
def predict_fixedprior_fulltrack_rawtop4_probe(
    *,
    model: MonotonicTop1RelativeTop4Residual,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    normalizer: FulltrackRawTop4RelativeNormalizer,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frozen monotone raw top-4 overlay without moving null mass."""

    if int(batch_size) <= 0:
        raise ValueError("raw top-4 prediction batch size must be positive")
    relative_features, _common = normalized_fulltrack_raw_top4_relative_features(
        features, normalizer
    )
    conditional_log_prior, candidate_mass = fixed_candidate_conditional_log_priors(
        features
    )
    candidate_blocks: list[np.ndarray] = []
    residual_blocks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(features.query_ids), int(batch_size)):
        end = min(start + int(batch_size), len(features.query_ids))
        candidate, residual, _logits = model(
            relative_features=torch.from_numpy(relative_features[start:end]).to(device),
            candidate_conditional_log_prior=torch.from_numpy(
                conditional_log_prior[start:end]
            ).to(device),
            candidate_mass=torch.from_numpy(candidate_mass[start:end]).to(device),
        )
        candidate_blocks.append(candidate.cpu().numpy().astype(np.float32))
        residual_blocks.append(residual.cpu().numpy().astype(np.float32))
    candidate = np.concatenate(candidate_blocks, axis=0)
    null = np.asarray(features.null_probabilities, dtype=np.float32).copy()
    if (
        np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities)) > 0.0
    ):
        raise RuntimeError("raw top-4 prediction changed fixed posterior mass")
    return candidate, null, np.concatenate(residual_blocks, axis=0)
