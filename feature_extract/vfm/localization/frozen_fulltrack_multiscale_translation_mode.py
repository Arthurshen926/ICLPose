"""Frozen multiscale translation-mode profiles for full-track S1 probes.

Each profile compares a dense query crop with the crop around one real SfM
support observation.  It keeps the resulting bounded translation-mode shape
per support view, so the later diagnostic may marginalize views without
averaging them before candidate scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from feature_extract.vfm.localization.multiscale_candidate_probe import (
    dense_local_translation_mode_feature_names,
)


FULLTRACK_MULTISCALE_TRANSLATION_MODE_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_multiscale_translation_mode_v1"
)
FULLTRACK_MULTISCALE_TRANSLATION_MODE_EDGE_FEATURE_SEMANTICS = (
    "multiscale_dense_translation_mode_per_real_sfm_observation_v1"
)
FULLTRACK_MULTISCALE_TRANSLATION_MODE_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_multiscale_dense_translation_mode_v1"
)


@dataclass(frozen=True)
class MultiscaleTranslationModeProfile:
    """One source-specific crop and bounded candidate-relative translation set."""

    name: str
    source_name: str
    grid_size: int
    window_size: int
    maximum_shift: int

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
        ):
            raise ValueError("multiscale translation-mode profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "maximum_shift", int(self.maximum_shift))


# Final supplies broad semantic context, intermediate PCA256 supplies
# facade-scale structure, and the current-manifest ALIKE grid32 supplies the
# highest-resolution texture cache available in the frozen v3 image contract.
# The profiles are frozen before any train/validation labels are joined.
MULTISCALE_TRANSLATION_MODE_PROFILES = (
    MultiscaleTranslationModeProfile(
        "radio_final_mode_context5_shift1", "radio_final", 16, 5, 1
    ),
    MultiscaleTranslationModeProfile(
        "radio_final_mode_context9_shift2", "radio_final", 16, 9, 2
    ),
    MultiscaleTranslationModeProfile(
        "radio_intermediate_pca256_mode_context5_shift1",
        "radio_intermediate_pca256",
        16,
        5,
        1,
    ),
    MultiscaleTranslationModeProfile(
        "radio_intermediate_pca256_mode_context9_shift2",
        "radio_intermediate_pca256",
        16,
        9,
        2,
    ),
    MultiscaleTranslationModeProfile(
        "radio_intermediate_pca256_mode_context13_shift3",
        "radio_intermediate_pca256",
        16,
        13,
        3,
    ),
    MultiscaleTranslationModeProfile(
        "alike_fpn_mode_context15_shift3", "alike_fpn", 32, 15, 3
    ),
    MultiscaleTranslationModeProfile(
        "alike_fpn_mode_context31_shift4", "alike_fpn", 32, 31, 4
    ),
)


def translation_mode_profile_feature_names(
    profile: MultiscaleTranslationModeProfile,
) -> tuple[str, ...]:
    return tuple(dense_local_translation_mode_feature_names(profile.name))


MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE = {
    profile.name: translation_mode_profile_feature_names(profile)
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
}
MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES = tuple(
    name
    for profile in MULTISCALE_TRANSLATION_MODE_PROFILES
    for name in MULTISCALE_TRANSLATION_MODE_FEATURE_NAMES_BY_PROFILE[profile.name]
)
