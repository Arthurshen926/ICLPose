"""Frozen spatial-pyramid shift profiles for full-track S1 appearance probes.

Each profile keeps a candidate-specific, per-real-support-view tensor of
local descriptor correlations.  Unlike the earlier translation-mode summary,
the tensor retains where inside the anchor-centred crop a bounded translation
is supported.  This is a diagnostic identity-evidence representation, not a
measurement update and not a pose-conditioned feature.
"""

from __future__ import annotations

from dataclasses import dataclass


FULLTRACK_SPATIAL_PYRAMID_SHIFT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_spatial_pyramid_shift_v1"
)
FULLTRACK_SPATIAL_PYRAMID_SHIFT_EDGE_FEATURE_SEMANTICS = (
    "multiscale_spatial_pyramid_shift_correlation_per_real_sfm_observation_v1"
)
FULLTRACK_SPATIAL_PYRAMID_SHIFT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_multiscale_spatial_pyramid_shift_correlation_v1"
)
FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_spatial_pyramid_shift_mask_control_v1"
)
FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_EDGE_FEATURE_SEMANTICS = (
    "multiscale_spatial_pyramid_shift_original_crop_mask_control_per_real_sfm_observation_v1"
)
FULLTRACK_SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_multiscale_spatial_pyramid_shift_mask_control_v1"
)


@dataclass(frozen=True)
class SpatialPyramidShiftProfile:
    """One feature source and its fixed local spatial-correlation layout."""

    name: str
    source_name: str
    grid_size: int
    window_size: int
    maximum_shift: int
    spatial_bin_count: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.grid_size) <= 0
            or int(self.window_size) <= 0
            or int(self.window_size) % 2 != 1
            or int(self.window_size) > int(self.grid_size)
            or int(self.maximum_shift) < 0
            or int(self.maximum_shift) >= int(self.window_size)
            or int(self.spatial_bin_count) <= 0
            or int(self.spatial_bin_count) > int(self.window_size)
        ):
            raise ValueError("spatial-pyramid shift profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "maximum_shift", int(self.maximum_shift))
        object.__setattr__(self, "spatial_bin_count", int(self.spatial_bin_count))


# Intermediate tokens carry facade-scale layout while ALIKE/FPN supplies the
# higher-resolution local signal.  Both profiles are frozen independently of
# identity targets, pose labels, retrieval, and render output.
SPATIAL_PYRAMID_SHIFT_PROFILES = (
    SpatialPyramidShiftProfile(
        "radio_intermediate_pca256_spatial_pyramid_context9_shift2_bins3",
        "radio_intermediate_pca256",
        16,
        9,
        2,
        3,
    ),
    SpatialPyramidShiftProfile(
        "alike_fpn_spatial_pyramid_context15_shift3_bins3",
        "alike_fpn",
        32,
        15,
        3,
        3,
    ),
)


def spatial_pyramid_shift_feature_names(
    profile: SpatialPyramidShiftProfile,
) -> tuple[str, ...]:
    """Return the immutable ``(region, shift)`` column order for one profile."""

    radius = int(profile.maximum_shift)
    output: list[str] = []
    for region_row in range(int(profile.spatial_bin_count)):
        for region_column in range(int(profile.spatial_bin_count)):
            for shift_row in range(-radius, radius + 1):
                for shift_column in range(-radius, radius + 1):
                    output.append(
                        f"{profile.name}_region_r{region_row}_c{region_column}"
                        f"_shift_dy{shift_row:+d}_dx{shift_column:+d}_cosine"
                    )
    expected = (
        int(profile.spatial_bin_count) ** 2
        * (2 * int(profile.maximum_shift) + 1) ** 2
    )
    if len(output) != expected or len(set(output)) != expected:
        raise RuntimeError("spatial-pyramid shift feature naming drifted")
    return tuple(output)


def spatial_pyramid_shift_mask_control_feature_names(
    profile: SpatialPyramidShiftProfile,
) -> tuple[str, ...]:
    """Return matched original-crop overlap controls for paired leakage audits."""

    radius = int(profile.maximum_shift)
    output: list[str] = []
    for region_row in range(int(profile.spatial_bin_count)):
        for region_column in range(int(profile.spatial_bin_count)):
            for shift_row in range(-radius, radius + 1):
                for shift_column in range(-radius, radius + 1):
                    output.append(
                        f"{profile.name}_region_r{region_row}_c{region_column}"
                        f"_shift_dy{shift_row:+d}_dx{shift_column:+d}"
                        "_original_crop_overlap_fraction"
                    )
    expected = (
        int(profile.spatial_bin_count) ** 2
        * (2 * int(profile.maximum_shift) + 1) ** 2
    )
    if len(output) != expected or len(set(output)) != expected:
        raise RuntimeError("spatial-pyramid mask-control feature naming drifted")
    return tuple(output)


SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE = {
    profile.name: spatial_pyramid_shift_feature_names(profile)
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
}
SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES = tuple(
    name
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
    for name in SPATIAL_PYRAMID_SHIFT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE = {
    profile.name: spatial_pyramid_shift_mask_control_feature_names(profile)
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
}
SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES = tuple(
    name
    for profile in SPATIAL_PYRAMID_SHIFT_PROFILES
    for name in SPATIAL_PYRAMID_SHIFT_MASK_CONTROL_FEATURE_NAMES_BY_PROFILE[
        profile.name
    ]
)
