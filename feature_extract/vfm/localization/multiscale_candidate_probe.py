"""Target-free multiscale per-view appearance features for frozen probes.

The probe intentionally separates candidate appearance from pose scoring.  It
extracts only query/support image descriptors and deterministic support views;
pose targets are joined by callers outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
    "radio_final_grid4_cosine",
    "radio_intermediate_anchor_cosine",
    "radio_intermediate_context3_pool_cosine",
    "radio_intermediate_context3_grid_cosine",
    "radio_intermediate_context5_pool_cosine",
    "radio_intermediate_context5_grid_cosine",
    "alike_anchor_cosine",
    "alike_context3_pool_cosine",
    "alike_context3_grid_cosine",
    "alike_context5_pool_cosine",
    "alike_context5_grid_cosine",
    "context3_grid_overlap",
    "context5_grid_overlap",
)

# These ablations are fixed before seeing validation/test targets.  In
# particular, the full family deliberately excludes any whole-image retrieval
# descriptor: every non-anchor term is candidate-support-view conditioned.
MULTISCALE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "radio_final_center": ("radio_final_anchor_cosine",),
    "radio_final_context": ("radio_final_grid4_cosine",),
    "radio_intermediate_center": ("radio_intermediate_anchor_cosine",),
    "radio_intermediate_context": (
        "radio_intermediate_context3_pool_cosine",
        "radio_intermediate_context3_grid_cosine",
        "radio_intermediate_context5_pool_cosine",
        "radio_intermediate_context5_grid_cosine",
        "context3_grid_overlap",
        "context5_grid_overlap",
    ),
    "alike_fpn_local": (
        "alike_anchor_cosine",
        "alike_context3_pool_cosine",
        "alike_context3_grid_cosine",
        "alike_context5_pool_cosine",
        "alike_context5_grid_cosine",
        "context3_grid_overlap",
        "context5_grid_overlap",
    ),
    "multiscale_per_view_no_global_retrieval": (
        "radio_final_anchor_cosine",
        "radio_final_grid4_cosine",
        "radio_intermediate_anchor_cosine",
        "radio_intermediate_context3_pool_cosine",
        "radio_intermediate_context3_grid_cosine",
        "radio_intermediate_context5_pool_cosine",
        "radio_intermediate_context5_grid_cosine",
        "alike_anchor_cosine",
        "alike_context3_pool_cosine",
        "alike_context3_grid_cosine",
        "alike_context5_pool_cosine",
        "alike_context5_grid_cosine",
        "context3_grid_overlap",
        "context5_grid_overlap",
    ),
}


def _regional_feature_names(prefix: str, *, region_grid_size: int = 3) -> tuple[str, ...]:
    return (
        f"{prefix}_pool_cosine",
        *(
            f"{prefix}_region_r{row}_c{column}_cosine"
            for row in range(int(region_grid_size))
            for column in range(int(region_grid_size))
        ),
        f"{prefix}_valid_fraction",
    )


def _structured_context_feature_names(prefix: str) -> tuple[str, ...]:
    """Fixed regional-plus-translation-tolerant schema for one feature scale."""

    return (
        *_regional_feature_names(prefix),
        *(
            f"{prefix}_shift_dy{shift_row}_dx{shift_column}_cosine"
            for shift_row in range(-1, 2)
            for shift_column in range(-1, 2)
        ),
        *(
            f"{prefix}_shift_dy{shift_row}_dx{shift_column}_overlap_fraction"
            for shift_row in range(-1, 2)
            for shift_column in range(-1, 2)
        ),
    )


STRUCTURED_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
    "radio_intermediate_anchor_cosine",
    "alike_anchor_cosine",
)

# Keep the existing anchor similarities separate from the new per-view context
# evidence.  The structured experiment is only informative if a context-only
# control improves held-out candidate ranking; otherwise a full-model gain can
# be explained by recalibrating the old coarse descriptor prior.
STRUCTURED_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES = (
    *_structured_context_feature_names("radio_final_grid8_window3"),
    *_structured_context_feature_names("radio_final_grid8_window5"),
    *_structured_context_feature_names("radio_intermediate_context7"),
    *_structured_context_feature_names("radio_intermediate_context11"),
    *_structured_context_feature_names("alike_context7"),
    *_structured_context_feature_names("alike_context11"),
)

STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES = (
    *STRUCTURED_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    *STRUCTURED_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES,
)

STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    # Predeclared attribution controls.  The first is intentionally limited to
    # pre-existing anchor cues; the second excludes them entirely and measures
    # whether candidate-specific real-image context has standalone signal.
    "structured_existing_anchor_control": STRUCTURED_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    "structured_candidate_specific_context_only": STRUCTURED_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES,
    "structured_radio_final_grid8": (
        "radio_final_anchor_cosine",
        *_structured_context_feature_names("radio_final_grid8_window3"),
        *_structured_context_feature_names("radio_final_grid8_window5"),
    ),
    "structured_radio_intermediate_large": (
        "radio_intermediate_anchor_cosine",
        *_structured_context_feature_names("radio_intermediate_context7"),
        *_structured_context_feature_names("radio_intermediate_context11"),
    ),
    "structured_alike_large": (
        "alike_anchor_cosine",
        *_structured_context_feature_names("alike_context7"),
        *_structured_context_feature_names("alike_context11"),
    ),
    "structured_multiscale_per_view_no_global_retrieval": STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
}


# The structured S1b probe above only retains aligned regional averages and a
# 3x3 translation neighbourhood.  The next frozen probe keeps a compact
# candidate-conditioned cross-image cost volume instead.  It deliberately
# uses a 3x3 re-pooled context grid: that is large enough to retain coarse
# facade layout while keeping the full per-view export tractable.
COST_VOLUME_CONTEXT_GRID_SIZE = 3
COST_VOLUME_CONTEXT_SCALE_NAMES = (
    "radio_final_grid8_window5",
    "radio_final_grid8_window7",
    "radio_intermediate_context11",
    "alike_context11",
)


def _cost_volume_scale_feature_names(prefix: str) -> tuple[str, ...]:
    """Stable Hough-cost-volume fields for one candidate-support context scale."""

    context_size = int(COST_VOLUME_CONTEXT_GRID_SIZE)
    if not str(prefix) or context_size <= 0:
        raise ValueError("cost-volume feature naming inputs are invalid")
    shifts = tuple(
        (row, column)
        for row in range(-(context_size - 1), context_size)
        for column in range(-(context_size - 1), context_size)
    )
    return (
        *(
            f"{prefix}_hough_dy{row}_dx{column}_mean_cosine"
            for row, column in shifts
        ),
        *(
            f"{prefix}_hough_dy{row}_dx{column}_attention_mass"
            for row, column in shifts
        ),
        *(
            f"{prefix}_hough_dy{row}_dx{column}_mutual_mass"
            for row, column in shifts
        ),
        *(
            f"{prefix}_hough_dy{row}_dx{column}_valid_pair_fraction"
            for row, column in shifts
        ),
        f"{prefix}_attention_entropy",
        f"{prefix}_valid_pair_fraction",
    )


COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
    "radio_intermediate_anchor_cosine",
    "alike_anchor_cosine",
)
COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for scale in COST_VOLUME_CONTEXT_SCALE_NAMES
    for name in _cost_volume_scale_feature_names(scale)
)
COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES = (
    *COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    *COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES,
)
COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "cost_volume_existing_anchor_control": (
        COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    ),
    "cost_volume_candidate_specific_context_only": (
        COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES
    ),
    "cost_volume_radio_final_context_only": (
        *_cost_volume_scale_feature_names("radio_final_grid8_window5"),
        *_cost_volume_scale_feature_names("radio_final_grid8_window7"),
    ),
    "cost_volume_radio_final_large_context": (
        "radio_final_anchor_cosine",
        *_cost_volume_scale_feature_names("radio_final_grid8_window5"),
        *_cost_volume_scale_feature_names("radio_final_grid8_window7"),
    ),
    "cost_volume_radio_intermediate_large_context": (
        "radio_intermediate_anchor_cosine",
        *_cost_volume_scale_feature_names("radio_intermediate_context11"),
    ),
    "cost_volume_alike_large_context": (
        "alike_anchor_cosine",
        *_cost_volume_scale_feature_names("alike_context11"),
    ),
    "cost_volume_multiscale_per_view_no_global_retrieval": (
        COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    ),
}


def cost_volume_scale_feature_slices() -> Mapping[str, slice]:
    """Return immutable column ranges for every cost-volume context scale."""

    begin = len(COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES)
    output: dict[str, slice] = {}
    for scale in COST_VOLUME_CONTEXT_SCALE_NAMES:
        end = begin + len(_cost_volume_scale_feature_names(scale))
        output[scale] = slice(begin, end)
        begin = end
    if begin != len(COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES):
        raise RuntimeError("cost-volume feature slices drifted")
    return output


# S1e deliberately does not reuse the S1c 3x3 Hough schema. S1c confirmed
# that a compact translation summary can improve an anchor-calibrated model,
# but it could not establish standalone absolute identity evidence. This
# schema keeps the complete 5x5 query-cell by support-cell correlation matrix
# so a frozen probe can inspect which relative facade regions agree instead of
# only retaining a shift-aggregated average.
WIDE_FULL_CORRELATION_CONTEXT_GRID_SIZE = 5
WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES = (
    "radio_final_grid8_window7_wide5",
    "radio_intermediate_context11_wide5",
    "alike_context11_wide5",
)


def _wide_full_correlation_scale_feature_names(prefix: str) -> tuple[str, ...]:
    """Stable unpooled 5x5 query/support correlation fields for one scale."""

    size = int(WIDE_FULL_CORRELATION_CONTEXT_GRID_SIZE)
    if not str(prefix) or size <= 0:
        raise ValueError("wide full-correlation feature naming inputs are invalid")
    cells = tuple((row, column) for row in range(size) for column in range(size))
    return (
        *(
            f"{prefix}_qrow{query_row}_qcol{query_column}"
            f"_srow{support_row}_scol{support_column}_cosine"
            for query_row, query_column in cells
            for support_row, support_column in cells
        ),
        *(f"{prefix}_qrow{row}_qcol{column}_valid" for row, column in cells),
        *(f"{prefix}_srow{row}_scol{column}_valid" for row, column in cells),
        f"{prefix}_pair_valid_fraction",
    )


WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
    "radio_intermediate_anchor_cosine",
    "alike_anchor_cosine",
)
WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for scale in WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES
    for name in _wide_full_correlation_scale_feature_names(scale)
)
WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES = (
    *WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    *WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES,
)
WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FAMILIES: Mapping[
    str, tuple[str, ...]
] = {
    "wide_fullcorr_existing_anchor_control": (
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    ),
    "wide_fullcorr_candidate_specific_context_only": (
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_CONTEXT_FEATURE_NAMES
    ),
    "wide_fullcorr_radio_final_context_only": (
        *_wide_full_correlation_scale_feature_names("radio_final_grid8_window7_wide5"),
    ),
    "wide_fullcorr_radio_final_large_context": (
        "radio_final_anchor_cosine",
        *_wide_full_correlation_scale_feature_names("radio_final_grid8_window7_wide5"),
    ),
    "wide_fullcorr_radio_intermediate_large_context": (
        "radio_intermediate_anchor_cosine",
        *_wide_full_correlation_scale_feature_names("radio_intermediate_context11_wide5"),
    ),
    "wide_fullcorr_alike_large_context": (
        "alike_anchor_cosine",
        *_wide_full_correlation_scale_feature_names("alike_context11_wide5"),
    ),
    "wide_fullcorr_multiscale_per_view_no_global_retrieval": (
        WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES
    ),
}


# S1f is intentionally a separate probe from the local cost-volume paths.
# It compares the query image with only the already-selected support image of
# each fixed candidate/view.  It must never retrieve, reselect, or normalize
# over an image set, otherwise a soft absolute-context factor would silently
# become image retrieval.
GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
    "radio_intermediate_anchor_cosine",
    "alike_anchor_cosine",
)
GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT = "global_context_candidate_probe_features_v1"
GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT = (
    "global_context_support8_candidate_probe_features_v1"
)
GLOBAL_CONTEXT_SOFT_FACTOR_USAGE_BY_FORMAT: Mapping[str, str] = {
    GLOBAL_CONTEXT_FEATURE_ARTIFACT_FORMAT: (
        "fixed_candidate_support_view_soft_radio_final_global_factor_v1"
    ),
    GLOBAL_CONTEXT_SUPPORT8_FEATURE_ARTIFACT_FORMAT: (
        "fixed_maplet_support8_view_soft_radio_final_global_factor_v1"
    ),
}
GLOBAL_CONTEXT_CANDIDATE_PROBE_FEATURE_NAMES = (
    *GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
    "radio_final_global_support_image_cosine",
)
GLOBAL_CONTEXT_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "global_context_existing_anchor_control": (
        GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES
    ),
    "global_context_candidate_specific_context_only": (
        "radio_final_global_support_image_cosine",
    ),
    "global_context_radio_final_soft_factor_with_anchor": (
        *GLOBAL_CONTEXT_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES,
        "radio_final_global_support_image_cosine",
    ),
}
GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FEATURE_NAMES = (
    "radio_final_global_support_image_cosine",
)
GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "global_context_support8_candidate_specific_context_only": (
        "radio_final_global_support_image_cosine",
    ),
}


# S1h keeps each real SfM observation as a separate, fixed support-view
# prototype.  It deliberately exposes a low-capacity spatial pyramid instead
# of another all-pairs local matcher: the question is whether a landmark's
# larger visual neighbourhood supplies absolute facade-phase evidence once
# more than the original two support views are retained.
LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT = (
    "landmark_region_prototype_candidate_probe_features_v1"
)
LANDMARK_REGION_PROTOTYPE_WINDOW_SIZE = 7
LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE = 3
LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
)


def landmark_region_prototype_feature_names(
    prefix: str,
    *,
    region_grid_size: int = LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE,
) -> tuple[str, ...]:
    """Return the fixed low-capacity visual-region prototype schema."""

    size = int(region_grid_size)
    if not str(prefix) or size <= 0:
        raise ValueError("landmark-region prototype feature naming inputs are invalid")
    return (
        f"{prefix}_pool_cosine",
        *(
            f"{prefix}_region_r{row}_c{column}_cosine"
            for row in range(size)
            for column in range(size)
        ),
        f"{prefix}_common_cell_fraction",
    )


LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES = landmark_region_prototype_feature_names(
    "radio_final_grid8_landmark_region7"
)
LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES = (
    *LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    *LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
)
LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "landmark_region_prototype_radio_final_anchor_control": (
        *LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    ),
    "landmark_region_prototype_candidate_specific_context_only": (
        *LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    ),
    "landmark_region_prototype_radio_final_with_anchor": (
        *LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    ),
}


# S1i raises only the spatial sampling resolution of the same real-image
# region-prototype protocol.  The two fixed crop scales retain medium and
# large facade context without falling back to whole-image retrieval.
HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT = (
    "highres_landmark_region_prototype_candidate_probe_features_v1"
)
HIGHRES_LANDMARK_REGION_PROTOTYPE_GRID_SIZE = 16
HIGHRES_LANDMARK_REGION_PROTOTYPE_WINDOW_SIZES = (7, 11)
HIGHRES_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
)
HIGHRES_LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for window_size in HIGHRES_LANDMARK_REGION_PROTOTYPE_WINDOW_SIZES
    for name in landmark_region_prototype_feature_names(
        f"radio_final_grid16_landmark_region{window_size}"
    )
)
HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES = (
    *HIGHRES_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    *HIGHRES_LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
)
HIGHRES_LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES: Mapping[
    str, tuple[str, ...]
] = {
    "highres_landmark_region_prototype_radio_final_anchor_control": (
        *HIGHRES_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    ),
    "highres_landmark_region_prototype_candidate_specific_context_only": (
        *HIGHRES_LANDMARK_REGION_PROTOTYPE_CONTEXT_FEATURE_NAMES,
    ),
    "highres_landmark_region_prototype_radio_final_with_anchor": (
        *HIGHRES_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    ),
}


# S1j keeps all candidate identities and support views fixed, but combines
# semantic RADIO-final, mid-level RADIO-intermediate, and local ALIKE-FPN
# region evidence.  Each source remains available as its own context-only
# family so a fused result cannot be credited to an untested branch.
MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_ARTIFACT_FORMAT = (
    "multisource_landmark_region_prototype_candidate_probe_features_v1"
)
MULTISOURCE_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES = (
    "radio_final_anchor_cosine",
)
MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES = tuple(
    name
    for window_size in (7, 11)
    for name in landmark_region_prototype_feature_names(
        f"radio_final_grid16_landmark_region{window_size}"
    )
)
MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for window_size in (7, 11)
    for name in landmark_region_prototype_feature_names(
        f"radio_intermediate_grid16_landmark_region{window_size}"
    )
)
MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for window_size in (7, 15)
    for name in landmark_region_prototype_feature_names(
        f"alike_grid32_landmark_region{window_size}"
    )
)
MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES = (
    *MULTISOURCE_LANDMARK_REGION_FINAL_CONTEXT_FEATURE_NAMES,
    *MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES,
    *MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES,
)
MULTISOURCE_LANDMARK_REGION_APPEARANCE_CONTEXT_FEATURE_NAMES = tuple(
    name
    for name in MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES
    if not name.endswith("_common_cell_fraction")
)
MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES = (
    *MULTISOURCE_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    *MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES,
)
MULTISOURCE_LANDMARK_REGION_APPEARANCE_FEATURE_NAMES = (
    *MULTISOURCE_LANDMARK_REGION_PROTOTYPE_ANCHOR_FEATURE_NAMES,
    *MULTISOURCE_LANDMARK_REGION_APPEARANCE_CONTEXT_FEATURE_NAMES,
)
MULTISOURCE_LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES: Mapping[
    str, tuple[str, ...]
] = {
    "multisource_landmark_region_radio_intermediate_context_only": (
        *MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES,
    ),
    "multisource_landmark_region_alike_context_only": (
        *MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES,
    ),
    "multisource_landmark_region_candidate_specific_context_only": (
        *MULTISOURCE_LANDMARK_REGION_CONTEXT_FEATURE_NAMES,
    ),
    "multisource_landmark_region_with_anchor": (
        *MULTISOURCE_LANDMARK_REGION_PROTOTYPE_FEATURE_NAMES,
    ),
    # The coverage terms describe crop overlap near image boundaries rather
    # than visual appearance.  S1j's train-only attribution diagnostic found
    # that they are a stronger geometric shortcut than any appearance field,
    # so this fixed family tests the actual candidate-specific visual evidence.
    "multisource_landmark_region_radio_intermediate_appearance_only": tuple(
        name
        for name in MULTISOURCE_LANDMARK_REGION_INTERMEDIATE_CONTEXT_FEATURE_NAMES
        if not name.endswith("_common_cell_fraction")
    ),
    "multisource_landmark_region_alike_appearance_only": tuple(
        name
        for name in MULTISOURCE_LANDMARK_REGION_ALIKE_CONTEXT_FEATURE_NAMES
        if not name.endswith("_common_cell_fraction")
    ),
    "multisource_landmark_region_candidate_specific_appearance_only": (
        *MULTISOURCE_LANDMARK_REGION_APPEARANCE_CONTEXT_FEATURE_NAMES,
    ),
    "multisource_landmark_region_with_anchor_appearance_only": (
        *MULTISOURCE_LANDMARK_REGION_APPEARANCE_FEATURE_NAMES,
    ),
}


# S1k retains dense ALIKE-FPN layout at a finer image grid and summarizes only
# candidate-conditioned local translation modes.  This is intentionally not a
# repeat of S1c/S1e: it neither relies on sparse detector-neighbour contexts
# nor exports an unconstrained all-pairs correlation matrix.
DENSE_ALIKE_LOCAL_MODE_FEATURE_ARTIFACT_FORMAT = (
    "dense_alike_local_mode_candidate_probe_features_v1"
)


def dense_local_translation_mode_feature_names(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_center_cosine",
        f"{prefix}_mean_cosine",
        f"{prefix}_peak_cosine",
        f"{prefix}_peak_minus_center",
        f"{prefix}_peak_minus_second",
        f"{prefix}_peak_probability",
        f"{prefix}_normalized_entropy",
        f"{prefix}_peak_dx_normalized",
        f"{prefix}_peak_dy_normalized",
        f"{prefix}_peak_radius_normalized",
    )


DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES = (
    *dense_local_translation_mode_feature_names("alike_dense_grid64_window15_radius3"),
    *dense_local_translation_mode_feature_names("alike_dense_grid64_window31_radius4"),
)
DENSE_ALIKE_LOCAL_MODE_ANCHOR_FEATURE_NAMES = ("radio_final_anchor_cosine",)
DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES = (
    *DENSE_ALIKE_LOCAL_MODE_ANCHOR_FEATURE_NAMES,
    *DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES,
)
DENSE_ALIKE_LOCAL_MODE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "dense_alike_local_mode_candidate_specific_context_only": (
        *DENSE_ALIKE_LOCAL_MODE_CONTEXT_FEATURE_NAMES,
    ),
    "dense_alike_local_mode_with_anchor": (
        *DENSE_ALIKE_LOCAL_MODE_FEATURE_NAMES,
    ),
}


# S1p keeps a larger RADIO-final grid around the frozen query/support anchors.
# Unlike a pooled image descriptor, every value is a bounded spatial
# translation correlation between the two real-image grids.  This lets the
# probe test whether candidate-specific layout phase contains information that
# the local 5x5 correlation and region-prototype paths discarded.
ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_ARTIFACT_FORMAT = (
    "anchor_aligned_global_layout_candidate_probe_features_v1"
)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_SCALES = (
    ("radio_final_grid16_window7_shift2", 7, 2),
    ("radio_final_grid16_window11_shift3", 11, 3),
    ("radio_final_grid16_window15_shift3", 15, 3),
)


def anchor_aligned_global_layout_scale_feature_names(
    prefix: str, *, maximum_shift: int
) -> tuple[str, ...]:
    shifts = tuple(
        (dy, dx)
        for dy in range(-int(maximum_shift), int(maximum_shift) + 1)
        for dx in range(-int(maximum_shift), int(maximum_shift) + 1)
    )
    return (
        *(f"{prefix}_shift_dy{dy}_dx{dx}_cosine" for dy, dx in shifts),
        *(f"{prefix}_shift_dy{dy}_dx{dx}_overlap" for dy, dx in shifts),
    )


ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES = tuple(
    name
    for prefix, _window_size, maximum_shift in ANCHOR_ALIGNED_GLOBAL_LAYOUT_SCALES
    for name in anchor_aligned_global_layout_scale_feature_names(
        prefix, maximum_shift=int(maximum_shift)
    )
)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_ANCHOR_FEATURE_NAMES = ("radio_final_anchor_cosine",)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES = (
    *ANCHOR_ALIGNED_GLOBAL_LAYOUT_ANCHOR_FEATURE_NAMES,
    *ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES,
)
ANCHOR_ALIGNED_GLOBAL_LAYOUT_CANDIDATE_PROBE_FAMILIES: Mapping[
    str, tuple[str, ...]
] = {
    "anchor_global_layout_context_only": (
        *ANCHOR_ALIGNED_GLOBAL_LAYOUT_CONTEXT_FEATURE_NAMES,
    ),
    "anchor_global_layout_with_anchor": (
        *ANCHOR_ALIGNED_GLOBAL_LAYOUT_FEATURE_NAMES,
    ),
}


# S1q-relative attention established that anchor-relative crops do not provide
# a stable absolute facade phase.  This follow-up is intentionally different:
# it preserves the entire 16x16 image lattice and records where every query
# region transports mass in the fixed candidate support image.  The central
# 3x3 anchor neighbourhood is excluded before any descriptor correlation.
ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_ARTIFACT_FORMAT = (
    "absolute_global_transport_candidate_probe_features_v1"
)
ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE = 3
ABSOLUTE_GLOBAL_TRANSPORT_SCALE_NAMES = (
    "radio_final_grid16",
    "radio_intermediate_grid16",
    "alike_grid16",
)
ABSOLUTE_GLOBAL_TRANSPORT_POSITION_SCALE_NAME = "grid16_position_only"
ABSOLUTE_GLOBAL_TRANSPORT_ANCHOR_FEATURE_NAMES = (
    "radio_final_grid16_anchor_cosine",
)


def absolute_global_transport_scale_feature_names(
    prefix: str,
    *,
    region_grid_size: int = ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
) -> tuple[str, ...]:
    """Stable full-image region-transport fields for one descriptor scale."""

    size = int(region_grid_size)
    if not str(prefix) or size <= 0:
        raise ValueError("absolute global-transport feature naming inputs are invalid")
    regions = tuple((row, column) for row in range(size) for column in range(size))
    return (
        *(
            f"{prefix}_qregion_r{query_row}_c{query_column}"
            f"_sregion_r{support_row}_c{support_column}_attention_mass"
            for query_row, query_column in regions
            for support_row, support_column in regions
        ),
        *(f"{prefix}_qregion_r{row}_c{column}_same_absolute_cosine" for row, column in regions),
        *(f"{prefix}_qregion_r{row}_c{column}_best_cosine" for row, column in regions),
        *(f"{prefix}_qregion_r{row}_c{column}_attention_entropy" for row, column in regions),
        *(f"{prefix}_qregion_r{row}_c{column}_expected_dx" for row, column in regions),
        *(f"{prefix}_qregion_r{row}_c{column}_expected_dy" for row, column in regions),
        *(f"{prefix}_qregion_r{row}_c{column}_query_coverage" for row, column in regions),
        *(f"{prefix}_sregion_r{row}_c{column}_support_coverage" for row, column in regions),
    )


ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES = tuple(
    name
    for scale in ABSOLUTE_GLOBAL_TRANSPORT_SCALE_NAMES
    for name in absolute_global_transport_scale_feature_names(scale)
)
ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES = (
    *absolute_global_transport_scale_feature_names(
        ABSOLUTE_GLOBAL_TRANSPORT_POSITION_SCALE_NAME
    ),
)
ABSOLUTE_GLOBAL_TRANSPORT_FEATURE_NAMES = (
    *ABSOLUTE_GLOBAL_TRANSPORT_ANCHOR_FEATURE_NAMES,
    *ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES,
    *ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES,
)
ABSOLUTE_GLOBAL_TRANSPORT_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    # This matched control receives only the deterministic full-grid masks
    # induced by the frozen query/support anchors.  It can reveal whether an
    # apparent layout gain is merely support-observation position leakage.
    "absolute_global_transport_position_only": (
        *ABSOLUTE_GLOBAL_TRANSPORT_POSITION_FEATURE_NAMES,
    ),
    "absolute_global_transport_context_only": (
        *ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES,
    ),
    "absolute_global_transport_with_anchor": (
        *ABSOLUTE_GLOBAL_TRANSPORT_ANCHOR_FEATURE_NAMES,
        *ABSOLUTE_GLOBAL_TRANSPORT_CONTEXT_FEATURE_NAMES,
    ),
}


def wide_full_correlation_scale_feature_slices() -> Mapping[str, slice]:
    """Return immutable column ranges for S1e wide full-correlation scales."""

    begin = len(WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_ANCHOR_FEATURE_NAMES)
    output: dict[str, slice] = {}
    for scale in WIDE_FULL_CORRELATION_CONTEXT_SCALE_NAMES:
        end = begin + len(_wide_full_correlation_scale_feature_names(scale))
        output[scale] = slice(begin, end)
        begin = end
    if begin != len(WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES):
        raise RuntimeError("wide full-correlation feature slices drifted")
    return output


ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES: Mapping[str, tuple[str, ...]] = {
    **MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    **STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    **COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    **WIDE_FULL_CORRELATION_MULTISCALE_CANDIDATE_PROBE_FAMILIES,
    **GLOBAL_CONTEXT_CANDIDATE_PROBE_FAMILIES,
    **GLOBAL_CONTEXT_SUPPORT8_CANDIDATE_PROBE_FAMILIES,
    **LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES,
    **HIGHRES_LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES,
    **MULTISOURCE_LANDMARK_REGION_PROTOTYPE_CANDIDATE_PROBE_FAMILIES,
    **DENSE_ALIKE_LOCAL_MODE_CANDIDATE_PROBE_FAMILIES,
    **ANCHOR_ALIGNED_GLOBAL_LAYOUT_CANDIDATE_PROBE_FAMILIES,
    **ABSOLUTE_GLOBAL_TRANSPORT_CANDIDATE_PROBE_FAMILIES,
}


def _normalized_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("descriptor values must have shape (N, C)")
    finite = np.isfinite(rows).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(rows), rows, 0.0), axis=1)
    valid = finite & (norms > 1e-8)
    output = np.zeros_like(rows, dtype=np.float32)
    output[valid] = rows[valid] / norms[valid, None]
    return output, valid


def fixed_candidate_support_global_context_cosine(
    query_descriptors: np.ndarray,
    support_descriptors: np.ndarray,
    view_valid: np.ndarray,
) -> np.ndarray:
    """Return only fixed candidate-support-view global RADIO cosine evidence.

    This function deliberately has no candidate-pool operation.  Every output
    value compares one query image with the support image already assigned to
    that candidate/view by the frozen maplet layout.  Invalid or missing image
    descriptors contribute explicit zero evidence; callers retain the view
    validity mask separately for the learned per-view mixture.
    """

    query = np.asarray(query_descriptors, dtype=np.float32)
    support = np.asarray(support_descriptors, dtype=np.float32)
    valid = np.asarray(view_valid, dtype=bool)
    if (
        query.ndim != 2
        or support.ndim != 4
        or valid.ndim != 3
        or support.shape[:3] != valid.shape
        or support.shape[0] != query.shape[0]
        or support.shape[3] != query.shape[1]
    ):
        raise ValueError("global-context query/support descriptor arrays are incompatible")
    normalized_query, query_has_descriptor = _normalized_rows(query)
    normalized_support, support_has_descriptor = _normalized_rows(
        support.reshape(-1, support.shape[-1])
    )
    normalized_support = normalized_support.reshape(support.shape)
    support_has_descriptor = support_has_descriptor.reshape(valid.shape)
    pair_valid = valid & support_has_descriptor & query_has_descriptor[:, None, None]
    scores = np.einsum("nd,nkvd->nkv", normalized_query, normalized_support)
    output = np.zeros(valid.shape, dtype=np.float32)
    output[pair_valid] = scores[pair_valid]
    if np.any(~np.isfinite(output)):
        raise RuntimeError("global-context cosine output is non-finite")
    return output


def _normalized_vector(value: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32, copy=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return a finite cosine or NaN when one side has no descriptor."""

    lhs = _normalized_vector(left)
    rhs = _normalized_vector(right)
    if lhs is None or rhs is None or lhs.shape != rhs.shape:
        return float("nan")
    return float(np.clip(lhs @ rhs, -1.0, 1.0))


@dataclass(frozen=True)
class LocalContextNodes:
    """Anchor-prepended local nodes in one image and one descriptor space."""

    xy: np.ndarray
    alike: np.ndarray
    intermediate: np.ndarray

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float32).reshape(-1, 2)
        alike = np.asarray(self.alike, dtype=np.float32)
        intermediate = np.asarray(self.intermediate, dtype=np.float32)
        if len(xy) == 0:
            raise ValueError("local context needs an anchor node")
        if (
            alike.ndim != 2
            or intermediate.ndim != 2
            or alike.shape[0] != len(xy)
            or intermediate.shape[0] != len(xy)
        ):
            raise ValueError("local context arrays are not node-aligned")
        if not np.isfinite(xy).all():
            raise ValueError("local context coordinates must be finite")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "alike", alike)
        object.__setattr__(self, "intermediate", intermediate)

    @property
    def anchor_xy(self) -> np.ndarray:
        return self.xy[0]


def assemble_local_context_nodes(
    *,
    anchor_xy: np.ndarray,
    anchor_alike: np.ndarray,
    anchor_intermediate: np.ndarray,
    nearby_xy: np.ndarray,
    nearby_alike: np.ndarray,
    nearby_intermediate: np.ndarray,
    nearby_scores: np.ndarray,
    max_nodes: int,
    duplicate_radius_px: float = 2.0,
) -> LocalContextNodes:
    """Deterministically select scored nearby nodes and prepend the anchor.

    Callers perform the radius query (usually through a cached KD-tree).  This
    keeps the selector independent of the representation used to index image
    nodes and makes the descriptor pairing explicit.
    """

    if int(max_nodes) <= 0 or float(duplicate_radius_px) < 0.0:
        raise ValueError("local context selection parameters are invalid")
    center = np.asarray(anchor_xy, dtype=np.float32).reshape(2)
    xy = np.asarray(nearby_xy, dtype=np.float32).reshape(-1, 2)
    alike = np.asarray(nearby_alike, dtype=np.float32)
    intermediate = np.asarray(nearby_intermediate, dtype=np.float32)
    scores = np.asarray(nearby_scores, dtype=np.float32).reshape(-1)
    if (
        alike.ndim != 2
        or intermediate.ndim != 2
        or alike.shape[0] != len(xy)
        or intermediate.shape[0] != len(xy)
        or len(scores) != len(xy)
    ):
        raise ValueError("nearby local context arrays are not aligned")
    distance2 = np.sum((xy - center[None]) ** 2, axis=1)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(scores)
    valid &= distance2 > float(duplicate_radius_px) ** 2
    selected = np.flatnonzero(valid)
    if selected.size:
        order = np.lexsort((selected, -scores[selected]))
        selected = selected[order[: max(int(max_nodes) - 1, 0)]]
    return LocalContextNodes(
        xy=np.concatenate([center[None], xy[selected]], axis=0),
        alike=np.concatenate(
            [np.asarray(anchor_alike, dtype=np.float32).reshape(1, -1), alike[selected]],
            axis=0,
        ),
        intermediate=np.concatenate(
            [
                np.asarray(anchor_intermediate, dtype=np.float32).reshape(1, -1),
                intermediate[selected],
            ],
            axis=0,
        ),
    )


@dataclass(frozen=True)
class ContextDescriptorSummary:
    pooled: np.ndarray
    grid: np.ndarray
    valid_cells: np.ndarray


def resample_context_descriptor_grid(
    summary: ContextDescriptorSummary,
    *,
    grid_size: int = COST_VOLUME_CONTEXT_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool a square context layout into a compact, normalized descriptor grid.

    The operation is deterministic and target-free.  Empty output bins stay
    invalid instead of becoming synthetic negative evidence, which is
    important for sparse ALIKE/intermediate contexts near image boundaries.
    """

    target_size = int(grid_size)
    values = np.asarray(summary.grid, dtype=np.float32)
    valid = np.asarray(summary.valid_cells, dtype=bool).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(valid) or target_size <= 0:
        raise ValueError("context resampling inputs are incompatible")
    source_size = int(round(float(np.sqrt(values.shape[0]))))
    if source_size <= 0 or source_size * source_size != values.shape[0]:
        raise ValueError("context resampling needs a square source grid")
    descriptors, descriptor_valid = _normalized_rows(values)
    source_valid = valid & descriptor_valid
    source_grid = descriptors.reshape(source_size, source_size, descriptors.shape[1])
    source_mask = source_valid.reshape(source_size, source_size)
    output = np.zeros((target_size, target_size, descriptors.shape[1]), dtype=np.float32)
    output_valid = np.zeros((target_size, target_size), dtype=bool)
    row_splits = np.array_split(np.arange(source_size), target_size)
    column_splits = np.array_split(np.arange(source_size), target_size)
    for row, source_rows in enumerate(row_splits):
        for column, source_columns in enumerate(column_splits):
            mask = source_mask[np.ix_(source_rows, source_columns)]
            if not np.any(mask):
                continue
            pooled = np.mean(source_grid[np.ix_(source_rows, source_columns)][mask], axis=0)
            normalized = _normalized_vector(pooled)
            if normalized is None:
                continue
            output[row, column] = normalized
            output_valid[row, column] = True
    return (
        output.reshape(target_size * target_size, descriptors.shape[1]),
        output_valid.reshape(target_size * target_size),
    )


def _cost_volume_hough_indices(
    *, context_grid_size: int, device: torch.device
) -> torch.Tensor:
    """Map every query/support cell pair to an anchor-relative Hough bin."""

    size = int(context_grid_size)
    if size <= 0:
        raise ValueError("cost-volume context grid size must be positive")
    rows, columns = torch.meshgrid(
        torch.arange(size, dtype=torch.long, device=device),
        torch.arange(size, dtype=torch.long, device=device),
        indexing="ij",
    )
    flat_rows = rows.reshape(-1)
    flat_columns = columns.reshape(-1)
    shift_rows = flat_rows[None, :] - flat_rows[:, None] + (size - 1)
    shift_columns = flat_columns[None, :] - flat_columns[:, None] + (size - 1)
    return shift_rows * (2 * size - 1) + shift_columns


def batched_context_cost_volume_hough_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Encode a batched, candidate-conditioned 2-D descriptor cost volume.

    Each output retains four Hough fields over anchor-relative translations:
    raw mean cosine, soft cross-attention mass, mutual-match mass, and valid
    pair coverage.  This is deliberately feature-only: no pose, target, or
    image-retrieval signal participates in the calculation.
    """

    if float(temperature) <= 0.0:
        raise ValueError("cost-volume temperature must be positive")
    if (
        query_grid.ndim != 3
        or support_grid.ndim != 3
        or query_grid.shape != support_grid.shape
        or query_valid.shape != query_grid.shape[:2]
        or support_valid.shape != support_grid.shape[:2]
    ):
        raise ValueError("batched cost-volume tensors are incompatible")
    batch, cell_count, _dimension = query_grid.shape
    grid_size = int(round(float(cell_count) ** 0.5))
    if grid_size <= 0 or grid_size * grid_size != int(cell_count):
        raise ValueError("batched cost-volume needs square context grids")
    query = F.normalize(query_grid.to(dtype=torch.float32), p=2, dim=2)
    support = F.normalize(support_grid.to(dtype=torch.float32), p=2, dim=2)
    pair_valid = query_valid.to(dtype=torch.bool)[:, :, None] & support_valid.to(
        dtype=torch.bool
    )[:, None, :]
    pair_valid_float = pair_valid.to(dtype=query.dtype)
    cosine = torch.bmm(query, support.transpose(1, 2))
    cosine = torch.where(pair_valid, cosine, torch.zeros_like(cosine))
    hough_index = _cost_volume_hough_indices(
        context_grid_size=grid_size, device=query.device
    ).reshape(1, -1).expand(batch, -1)
    hough_count = int((2 * grid_size - 1) ** 2)

    def _scatter(values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((batch, hough_count), dtype=query.dtype, device=query.device)
        return output.scatter_add_(1, hough_index, values.reshape(batch, -1))

    counts = _scatter(pair_valid_float)
    mean_cosine = _scatter(cosine * pair_valid_float) / counts.clamp_min(1.0)
    coverage = counts / float(cell_count * cell_count)

    # A finite sentinel keeps all-missing inputs well-defined.  In ordinary
    # contexts an anchor is valid on both sides, so every row has evidence.
    flat_valid = pair_valid.reshape(batch, -1)
    has_pair = torch.any(flat_valid, dim=1, keepdim=True)
    flat_logits = torch.where(
        flat_valid,
        (cosine / float(temperature)).reshape(batch, -1),
        torch.full((batch, cell_count * cell_count), -1e9, dtype=query.dtype, device=query.device),
    )
    attention = torch.softmax(flat_logits, dim=1)
    attention = torch.where(has_pair, attention, torch.zeros_like(attention))
    attention_mass = _scatter(attention.reshape(batch, cell_count, cell_count))

    row_logits = torch.where(
        pair_valid,
        cosine / float(temperature),
        torch.full_like(cosine, -1e9),
    )
    column_logits = row_logits.transpose(1, 2)
    row_probability = torch.softmax(row_logits, dim=2)
    column_probability = torch.softmax(column_logits, dim=2).transpose(1, 2)
    mutual = row_probability * column_probability * pair_valid_float
    mutual_mass = _scatter(mutual)

    entropy = -torch.sum(
        attention * torch.log(attention.clamp_min(torch.finfo(query.dtype).tiny)), dim=1
    )
    valid_count = torch.sum(pair_valid_float.reshape(batch, -1), dim=1)
    entropy = torch.where(
        valid_count > 1.0,
        entropy / torch.log(valid_count.clamp_min(2.0)),
        torch.zeros_like(entropy),
    )
    valid_fraction = valid_count / float(cell_count * cell_count)
    output = torch.cat(
        [
            mean_cosine,
            attention_mass,
            mutual_mass,
            coverage,
            entropy[:, None],
            valid_fraction[:, None],
        ],
        dim=1,
    )
    if output.shape != (batch, len(_cost_volume_scale_feature_names("scale"))):
        raise RuntimeError("cost-volume Hough feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("cost-volume Hough output is non-finite")
    return output


def batched_wide_context_full_correlation_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
) -> torch.Tensor:
    """Keep every candidate-conditioned 5x5 query/support cosine cell pair.

    Unlike the S1c Hough representation, this does not pool pairs that share a
    relative translation. The output therefore retains the absolute position
    of agreement inside each anchor-relative context crop. Invalid cells are
    zeroed and emitted explicitly as masks, so missing context remains a
    finite, pose-free unknown rather than synthetic negative evidence.
    """

    if (
        query_grid.ndim != 3
        or support_grid.ndim != 3
        or query_grid.shape != support_grid.shape
        or query_valid.shape != query_grid.shape[:2]
        or support_valid.shape != support_grid.shape[:2]
    ):
        raise ValueError("wide full-correlation tensors are incompatible")
    batch, cell_count, _dimension = query_grid.shape
    grid_size = int(round(float(cell_count) ** 0.5))
    expected_size = int(WIDE_FULL_CORRELATION_CONTEXT_GRID_SIZE)
    if grid_size != expected_size or grid_size * grid_size != int(cell_count):
        raise ValueError("wide full-correlation needs the declared square context grid")
    query = F.normalize(query_grid.to(dtype=torch.float32), p=2, dim=2)
    support = F.normalize(support_grid.to(dtype=torch.float32), p=2, dim=2)
    query_mask = query_valid.to(dtype=torch.bool)
    support_mask = support_valid.to(dtype=torch.bool)
    pair_valid = query_mask[:, :, None] & support_mask[:, None, :]
    cosine = torch.bmm(query, support.transpose(1, 2))
    cosine = torch.where(pair_valid, cosine, torch.zeros_like(cosine))
    pair_fraction = pair_valid.to(dtype=cosine.dtype).mean(dim=(1, 2), keepdim=False)
    output = torch.cat(
        [
            cosine.reshape(batch, -1),
            query_mask.to(dtype=cosine.dtype),
            support_mask.to(dtype=cosine.dtype),
            pair_fraction[:, None],
        ],
        dim=1,
    )
    expected_width = len(_wide_full_correlation_scale_feature_names("scale"))
    if output.shape != (batch, expected_width):
        raise RuntimeError("wide full-correlation feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("wide full-correlation output is non-finite")
    return output


def batched_dense_local_translation_mode_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    maximum_shift: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Summarize masked dense local translation modes for each fixed view pair.

    ``query_grid`` and ``support_grid`` are same-sized local crops from dense
    ALIKE feature maps around the frozen query token and the frozen SfM support
    observation.  Each mode compares only corresponding cells after a bounded
    relative translation.  The returned values are compact likelihood-shape
    statistics, never a measurement update or a pose-dependent feature.
    """

    if float(temperature) <= 0.0:
        raise ValueError("dense local-mode temperature must be positive")
    if (
        query_grid.ndim != 4
        or support_grid.ndim != 4
        or query_grid.shape != support_grid.shape
        or query_grid.shape[1] != query_grid.shape[2]
        or query_valid.shape != query_grid.shape[:3]
        or support_valid.shape != support_grid.shape[:3]
    ):
        raise ValueError("dense local-mode tensors are incompatible")
    window_size = int(query_grid.shape[1])
    maximum_shift = int(maximum_shift)
    if window_size <= 0 or maximum_shift < 0 or maximum_shift >= window_size:
        raise ValueError("dense local-mode window/shift configuration is invalid")
    query = F.normalize(query_grid.to(dtype=torch.float32), p=2, dim=3)
    support = F.normalize(support_grid.to(dtype=torch.float32), p=2, dim=3)
    query_mask = query_valid.to(dtype=torch.bool)
    support_mask = support_valid.to(dtype=torch.bool)
    # Treat every batch element as a convolution group.  This is algebraically
    # the same masked dot-product at every bounded translation as the former
    # Python offset loop, but produces all modes in two grouped convolutions.
    # It keeps full candidate/view independence because groups never mix rows.
    batch, _height, _width, dimension = query.shape
    query_weight = (
        query * query_mask[..., None].to(dtype=query.dtype)
    ).permute(0, 3, 1, 2).contiguous()
    support_input = F.pad(
        (support * support_mask[..., None].to(dtype=support.dtype))
        .permute(0, 3, 1, 2)
        .contiguous(),
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch * dimension,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    score_sum = F.conv2d(support_input, query_weight, groups=batch)[0]
    query_mask_weight = query_mask.to(dtype=query.dtype)[:, None]
    support_mask_input = F.pad(
        support_mask.to(dtype=query.dtype)[:, None],
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    count = F.conv2d(support_mask_input, query_mask_weight, groups=batch)[0]
    scores = (score_sum / count.clamp_min(1.0)).reshape(batch, -1)
    valid = (count > 0.0).reshape(batch, -1)
    shifts = [
        (row - maximum_shift, column - maximum_shift)
        for row in range(2 * maximum_shift + 1)
        for column in range(2 * maximum_shift + 1)
    ]
    if not bool(torch.all(torch.any(valid, dim=1))):
        raise RuntimeError("dense local-mode crop has no shared valid cells")
    sentinel = torch.finfo(scores.dtype).min
    masked_scores = torch.where(valid, scores, torch.full_like(scores, sentinel))
    mean_score = (scores * valid.to(dtype=scores.dtype)).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1).to(dtype=scores.dtype)
    peak_score, peak_index = torch.max(masked_scores, dim=1)
    top = torch.topk(masked_scores, k=min(2, masked_scores.shape[1]), dim=1)
    if int(masked_scores.shape[1]) == 1:
        second_score = peak_score
    else:
        second_valid = valid.gather(1, top.indices[:, 1:2])[:, 0]
        second_score = torch.where(second_valid, top.values[:, 1], peak_score)
    center_index = shifts.index((0, 0))
    center_score = torch.where(valid[:, center_index], scores[:, center_index], mean_score)
    logits = torch.where(valid, scores / float(temperature), torch.full_like(scores, -1e9))
    probability = torch.softmax(logits, dim=1)
    entropy = -torch.sum(
        probability * torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny)),
        dim=1,
    )
    valid_count = valid.sum(dim=1)
    entropy = torch.where(
        valid_count > 1,
        entropy / torch.log(valid_count.to(dtype=entropy.dtype).clamp_min(2.0)),
        torch.zeros_like(entropy),
    )
    shift_tensor = torch.tensor(shifts, dtype=scores.dtype, device=scores.device)
    selected_shift = shift_tensor.index_select(0, peak_index)
    denominator = float(max(maximum_shift, 1))
    peak_probability = probability.gather(1, peak_index[:, None])[:, 0]
    output = torch.stack(
        [
            center_score,
            mean_score,
            peak_score,
            peak_score - center_score,
            peak_score - second_score,
            peak_probability,
            entropy,
            selected_shift[:, 1] / denominator,
            selected_shift[:, 0] / denominator,
            torch.linalg.vector_norm(selected_shift, dim=1) / denominator,
        ],
        dim=1,
    )
    if output.shape != (len(query_grid), len(dense_local_translation_mode_feature_names("scale"))):
        raise RuntimeError("dense local-mode feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("dense local-mode features are non-finite")
    return output


def _spatial_pyramid_region_masks(
    *,
    window_size: int,
    spatial_bin_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build deterministic local-region masks in the exported column order."""

    coordinates = torch.arange(window_size, device=device)
    rows, columns = torch.meshgrid(coordinates, coordinates, indexing="ij")
    region_rows = torch.div(
        rows * spatial_bin_count, window_size, rounding_mode="floor"
    )
    region_columns = torch.div(
        columns * spatial_bin_count, window_size, rounding_mode="floor"
    )
    region_indices = (region_rows * spatial_bin_count + region_columns).reshape(-1)
    region_count = spatial_bin_count * spatial_bin_count
    return F.one_hot(region_indices, num_classes=region_count).transpose(0, 1).reshape(
        region_count, window_size, window_size
    ).to(dtype=dtype)


def batched_spatial_pyramid_shift_correlation_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    maximum_shift: int,
    spatial_bin_count: int,
) -> torch.Tensor:
    """Keep candidate-relative translation evidence in fixed local regions.

    The older dense translation-mode probe reduces every crop to peak and
    entropy statistics.  That loses the distinction between, for example, an
    agreement in the upper-left facade context and an otherwise identical
    agreement in the lower-right context.  This function instead emits the
    masked mean cosine for every ``(query-region, relative shift)`` pair.

    It is still a frozen appearance exporter: query/support anchors, support
    views, and shifts are fixed before this function is called.  The caller is
    responsible for treating an incomplete crop as unknown.  In the production
    exporter we require full crop validity, so no availability mask or overlap
    field is supplied to the learned identity model.
    """

    if (
        query_grid.ndim != 4
        or support_grid.ndim != 4
        or query_grid.shape != support_grid.shape
        or query_grid.shape[1] != query_grid.shape[2]
        or query_valid.shape != query_grid.shape[:3]
        or support_valid.shape != query_grid.shape[:3]
    ):
        raise ValueError("spatial-pyramid shift tensors are incompatible")
    window_size = int(query_grid.shape[1])
    maximum_shift = int(maximum_shift)
    spatial_bin_count = int(spatial_bin_count)
    if (
        window_size <= 0
        or maximum_shift < 0
        or maximum_shift >= window_size
        or spatial_bin_count <= 0
        or spatial_bin_count > window_size
    ):
        raise ValueError("spatial-pyramid shift configuration is invalid")

    query = F.normalize(query_grid.to(dtype=torch.float32), p=2, dim=3)
    support = F.normalize(support_grid.to(dtype=torch.float32), p=2, dim=3)
    query_mask = query_valid.to(dtype=torch.bool)
    support_mask = support_valid.to(dtype=torch.bool)
    batch, _height, _width, dimension = query.shape

    # ``conv2d(..., groups=batch)`` evaluates every image pair independently.
    # Giving each group one filter for every fixed spatial bin computes all
    # regional correlations in two convolutions, rather than a Python loop over
    # shifts or candidate views.
    region_count = spatial_bin_count * spatial_bin_count
    region_masks = _spatial_pyramid_region_masks(
        window_size=window_size,
        spatial_bin_count=spatial_bin_count,
        device=query.device,
        dtype=query.dtype,
    )

    query_weight = (
        query * query_mask[..., None].to(dtype=query.dtype)
    ).permute(0, 3, 1, 2)
    query_weight = (
        query_weight[:, None] * region_masks[None, :, None]
    ).reshape(batch * region_count, dimension, window_size, window_size)
    support_input = F.pad(
        (support * support_mask[..., None].to(dtype=support.dtype))
        .permute(0, 3, 1, 2)
        .contiguous(),
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch * dimension,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    score_sum = F.conv2d(support_input, query_weight.contiguous(), groups=batch)[0]

    query_count_weight = (
        query_mask.to(dtype=query.dtype)[:, None] * region_masks[None]
    ).reshape(batch * region_count, 1, window_size, window_size)
    support_mask_input = F.pad(
        support_mask.to(dtype=query.dtype)[:, None],
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    count = F.conv2d(support_mask_input, query_count_weight.contiguous(), groups=batch)[
        0
    ]
    shift_size = 2 * maximum_shift + 1
    score_sum = score_sum.reshape(batch, region_count, shift_size, shift_size)
    count = count.reshape(batch, region_count, shift_size, shift_size)
    values = score_sum / count.clamp_min(1.0)
    values = torch.where(count > 0.0, values, torch.zeros_like(values))
    output = values.reshape(batch, region_count * shift_size * shift_size)
    expected_width = region_count * shift_size * shift_size
    if output.shape != (batch, expected_width):
        raise RuntimeError("spatial-pyramid shift feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("spatial-pyramid shift features are non-finite")
    return output


def batched_spatial_pyramid_shift_overlap_features(
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    maximum_shift: int,
    spatial_bin_count: int,
) -> torch.Tensor:
    """Return only original-crop overlap controls for paired leakage audits.

    This uses the same region/shift layout as
    :func:`batched_spatial_pyramid_shift_correlation_features`, but contains no
    descriptor values.  Production visual models must never consume these
    controls; they exist solely for a matched OOF control experiment.
    """

    if (
        query_valid.ndim != 3
        or support_valid.shape != query_valid.shape
        or query_valid.shape[1] != query_valid.shape[2]
    ):
        raise ValueError("spatial-pyramid overlap masks are incompatible")
    window_size = int(query_valid.shape[1])
    maximum_shift = int(maximum_shift)
    spatial_bin_count = int(spatial_bin_count)
    if (
        window_size <= 0
        or maximum_shift < 0
        or maximum_shift >= window_size
        or spatial_bin_count <= 0
        or spatial_bin_count > window_size
    ):
        raise ValueError("spatial-pyramid overlap configuration is invalid")
    batch = len(query_valid)
    region_count = spatial_bin_count * spatial_bin_count
    masks = _spatial_pyramid_region_masks(
        window_size=window_size,
        spatial_bin_count=spatial_bin_count,
        device=query_valid.device,
        dtype=torch.float32,
    )
    query_weight = (
        query_valid.to(dtype=torch.float32)[:, None] * masks[None]
    ).reshape(batch * region_count, 1, window_size, window_size)
    support_input = F.pad(
        support_valid.to(dtype=torch.float32)[:, None],
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    count = F.conv2d(support_input, query_weight.contiguous(), groups=batch)[0]
    shift_size = 2 * maximum_shift + 1
    count = count.reshape(batch, region_count, shift_size, shift_size)
    region_cell_count = masks.sum(dim=(1, 2)).reshape(1, region_count, 1, 1)
    output = (count / region_cell_count.clamp_min(1.0)).reshape(
        batch, region_count * shift_size * shift_size
    )
    expected_width = region_count * shift_size * shift_size
    if output.shape != (batch, expected_width):
        raise RuntimeError("spatial-pyramid overlap layout drifted")
    if not bool(torch.isfinite(output).all()) or bool(
        torch.any((output < 0.0) | (output > 1.0))
    ):
        raise RuntimeError("spatial-pyramid overlap output is invalid")
    return output


def batched_masked_translation_correlation_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    maximum_shift: int,
) -> torch.Tensor:
    """Return every bounded, masked spatial translation correlation.

    This is intentionally a feature exporter rather than a measurement head:
    query/support anchors and candidate views are fixed before this function
    runs, and the result does not depend on any evaluated camera pose.
    """

    if (
        query_grid.ndim != 4
        or support_grid.ndim != 4
        or query_grid.shape != support_grid.shape
        or query_grid.shape[1] != query_grid.shape[2]
        or query_valid.shape != query_grid.shape[:3]
        or support_valid.shape != support_grid.shape[:3]
    ):
        raise ValueError("masked translation-correlation tensors are incompatible")
    window_size = int(query_grid.shape[1])
    maximum_shift = int(maximum_shift)
    if window_size <= 0 or maximum_shift < 0 or maximum_shift >= window_size:
        raise ValueError("masked translation-correlation window/shift is invalid")
    query = F.normalize(query_grid.to(dtype=torch.float32), p=2, dim=3)
    support = F.normalize(support_grid.to(dtype=torch.float32), p=2, dim=3)
    query_mask = query_valid.to(dtype=torch.bool)
    support_mask = support_valid.to(dtype=torch.bool)
    batch, _height, _width, dimension = query.shape
    query_weight = (
        query * query_mask[..., None].to(dtype=query.dtype)
    ).permute(0, 3, 1, 2).contiguous()
    support_input = F.pad(
        (support * support_mask[..., None].to(dtype=support.dtype))
        .permute(0, 3, 1, 2)
        .contiguous(),
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch * dimension,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    score_sum = F.conv2d(support_input, query_weight, groups=batch)[0]
    query_mask_weight = query_mask.to(dtype=query.dtype)[:, None]
    support_mask_input = F.pad(
        support_mask.to(dtype=query.dtype)[:, None],
        (maximum_shift, maximum_shift, maximum_shift, maximum_shift),
    ).reshape(
        1,
        batch,
        window_size + 2 * maximum_shift,
        window_size + 2 * maximum_shift,
    )
    count = F.conv2d(support_mask_input, query_mask_weight, groups=batch)[0]
    scores = (score_sum / count.clamp_min(1.0)).reshape(batch, -1)
    overlap = (count / float(window_size * window_size)).reshape(batch, -1)
    scores = torch.where(count.reshape(batch, -1) > 0.0, scores, torch.zeros_like(scores))
    output = torch.cat([scores, overlap], dim=1)
    expected_width = 2 * (2 * maximum_shift + 1) ** 2
    if output.shape != (batch, expected_width):
        raise RuntimeError("masked translation-correlation feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("masked translation-correlation features are non-finite")
    return output


def _absolute_global_transport_region_indices(
    *, grid_size: int, region_grid_size: int, device: torch.device
) -> torch.Tensor:
    """Map every full-image grid cell to its fixed absolute region bin."""

    size = int(grid_size)
    region_size = int(region_grid_size)
    if size <= 0 or region_size <= 0 or region_size > size:
        raise ValueError("absolute global-transport region grid is invalid")
    rows, columns = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing="ij",
    )
    region_rows = torch.div(rows * region_size, size, rounding_mode="floor")
    region_columns = torch.div(columns * region_size, size, rounding_mode="floor")
    return (region_rows * region_size + region_columns).reshape(-1)


def batched_absolute_global_transport_features(
    query_grid: torch.Tensor,
    query_valid: torch.Tensor,
    support_grid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    region_grid_size: int = ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Encode target-free full-image candidate-specific visual transport.

    A row represents one frozen query/candidate/support-view edge.  All image
    cells remain in their absolute 16x16 coordinates, while the central
    anchor neighbourhood has already been removed from ``*_valid``.  The
    output therefore retains candidate-conditioned regional transport rather
    than an anchor-relative crop or a global image retrieval score.
    """

    if float(temperature) <= 0.0:
        raise ValueError("absolute global-transport temperature must be positive")
    if (
        query_grid.ndim != 4
        or support_grid.ndim != 4
        or query_grid.shape != support_grid.shape
        or query_grid.shape[1] != query_grid.shape[2]
        or query_valid.shape != query_grid.shape[:3]
        or support_valid.shape != query_grid.shape[:3]
    ):
        raise ValueError("absolute global-transport tensors are incompatible")
    batch, grid_size, _width, _dimension = query_grid.shape
    region_size = int(region_grid_size)
    if region_size <= 0 or region_size > int(grid_size):
        raise ValueError("absolute global-transport region grid is invalid")
    cell_count = int(grid_size) * int(grid_size)
    region_count = region_size * region_size
    query = F.normalize(
        query_grid.reshape(batch, cell_count, _dimension).to(dtype=torch.float32),
        p=2,
        dim=2,
    )
    support = F.normalize(
        support_grid.reshape(batch, cell_count, _dimension).to(dtype=torch.float32),
        p=2,
        dim=2,
    )
    query_mask = query_valid.reshape(batch, cell_count).to(dtype=torch.bool)
    support_mask = support_valid.reshape(batch, cell_count).to(dtype=torch.bool)
    pair_mask = query_mask[:, :, None] & support_mask[:, None, :]
    cosine = torch.bmm(query, support.transpose(1, 2))
    sentinel = torch.finfo(cosine.dtype).min
    masked_cosine = torch.where(pair_mask, cosine, torch.full_like(cosine, sentinel))
    support_count = support_mask.sum(dim=1)
    row_valid = query_mask & (support_count[:, None] > 0)
    attention = torch.softmax(masked_cosine / float(temperature), dim=2)
    attention = torch.where(row_valid[:, :, None], attention, torch.zeros_like(attention))

    region_indices = _absolute_global_transport_region_indices(
        grid_size=int(grid_size), region_grid_size=region_size, device=query.device
    )
    region_one_hot = F.one_hot(region_indices, num_classes=region_count).to(dtype=query.dtype)
    query_weights = query_mask.to(dtype=query.dtype)[:, :, None] * region_one_hot[None]
    support_weights = support_mask.to(dtype=query.dtype)[:, :, None] * region_one_hot[None]
    cells_per_region = region_one_hot.sum(dim=0).clamp_min(1.0)
    query_coverage = query_weights.sum(dim=1) / cells_per_region[None]
    support_coverage = support_weights.sum(dim=1) / cells_per_region[None]

    attention_by_support_region = torch.matmul(attention, region_one_hot)
    query_region_count = query_weights.sum(dim=1).clamp_min(1.0)
    attention_mass = torch.bmm(
        query_weights.transpose(1, 2), attention_by_support_region
    ) / query_region_count[:, :, None]

    def _regional_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.to(dtype=query.dtype)[:, :, None] * region_one_hot[None]
        counts = weights.sum(dim=1).clamp_min(1.0)
        return torch.bmm(weights.transpose(1, 2), values[:, :, None]).squeeze(2) / counts

    diagonal = torch.diagonal(cosine, dim1=1, dim2=2)
    diagonal_valid = query_mask & support_mask
    same_absolute = _regional_mean(
        torch.where(diagonal_valid, diagonal, torch.zeros_like(diagonal)), diagonal_valid
    )
    best, _best_index = torch.max(masked_cosine, dim=2)
    best = torch.where(row_valid, best, torch.zeros_like(best))
    best_cosine = _regional_mean(best, row_valid)
    entropy = -torch.sum(
        attention * torch.log(attention.clamp_min(torch.finfo(attention.dtype).tiny)), dim=2
    )
    entropy = torch.where(
        support_count[:, None] > 1,
        entropy / torch.log(support_count[:, None].to(dtype=entropy.dtype).clamp_min(2.0)),
        torch.zeros_like(entropy),
    )
    entropy = torch.where(row_valid, entropy, torch.zeros_like(entropy))
    regional_entropy = _regional_mean(entropy, row_valid)

    coordinate = (torch.arange(int(grid_size), device=query.device, dtype=query.dtype) + 0.5)
    coordinate = coordinate / float(grid_size) * 2.0 - 1.0
    rows, columns = torch.meshgrid(coordinate, coordinate, indexing="ij")
    xy = torch.stack([columns, rows], dim=-1).reshape(cell_count, 2)
    expected_xy = torch.matmul(attention, xy)
    delta_xy = expected_xy - xy[None]
    expected_dx = _regional_mean(delta_xy[:, :, 0], row_valid)
    expected_dy = _regional_mean(delta_xy[:, :, 1], row_valid)
    output = torch.cat(
        [
            attention_mass.reshape(batch, region_count * region_count),
            same_absolute,
            best_cosine,
            regional_entropy,
            expected_dx,
            expected_dy,
            query_coverage,
            support_coverage,
        ],
        dim=1,
    )
    expected_width = len(absolute_global_transport_scale_feature_names("scale", region_grid_size=region_size))
    if output.shape != (batch, expected_width):
        raise RuntimeError("absolute global-transport feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("absolute global-transport output is non-finite")
    return output


def batched_absolute_global_transport_position_features(
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    *,
    region_grid_size: int = ABSOLUTE_GLOBAL_TRANSPORT_REGION_GRID_SIZE,
) -> torch.Tensor:
    """Matched no-appearance control for the full-grid transport schema.

    Constant descriptors preserve only the frozen center masks and absolute
    image-region geometry.  This must be audited separately from the real
    descriptor transport before treating a layout effect as visual evidence.
    """

    if (
        query_valid.ndim != 3
        or support_valid.ndim != 3
        or query_valid.shape != support_valid.shape
        or query_valid.shape[1] != query_valid.shape[2]
    ):
        raise ValueError("absolute global-transport position masks are incompatible")
    shape = (*query_valid.shape, 1)
    zeros = torch.zeros(shape, dtype=torch.float32, device=query_valid.device)
    return batched_absolute_global_transport_features(
        zeros,
        query_valid,
        zeros,
        support_valid,
        region_grid_size=region_grid_size,
        temperature=1.0,
    )


def summarize_context_descriptors(
    nodes: LocalContextNodes,
    *,
    descriptor_name: str,
    grid_size: int,
    radius_px: float,
) -> ContextDescriptorSummary:
    """Pool a local descriptor set and retain its spatial 3x3/5x5 structure."""

    if int(grid_size) <= 0 or float(radius_px) <= 0.0:
        raise ValueError("context grid size and radius must be positive")
    if descriptor_name not in {"alike", "intermediate"}:
        raise ValueError("unsupported local context descriptor")
    descriptors, valid = _normalized_rows(getattr(nodes, descriptor_name))
    if not bool(valid[0]):
        raise ValueError("local context anchor descriptor must be valid")
    relative = (nodes.xy - nodes.anchor_xy[None]) / float(radius_px)
    inside = np.all(np.abs(relative) <= 1.0 + 1e-6, axis=1) & valid
    # The anchor must always contribute to the central cell even when callers
    # queried a tight neighborhood that omitted other points.
    inside[0] = True
    scaled = (relative + 1.0) * (0.5 * int(grid_size))
    columns = np.clip(np.floor(scaled[:, 0]).astype(np.int64), 0, int(grid_size) - 1)
    rows = np.clip(np.floor(scaled[:, 1]).astype(np.int64), 0, int(grid_size) - 1)
    cells = rows * int(grid_size) + columns
    dimension = int(descriptors.shape[1])
    sums = np.zeros((int(grid_size) ** 2, dimension), dtype=np.float32)
    counts = np.zeros((int(grid_size) ** 2,), dtype=np.int64)
    np.add.at(sums, cells[inside], descriptors[inside])
    np.add.at(counts, cells[inside], 1)
    valid_cells = counts > 0
    grid = np.zeros_like(sums)
    if np.any(valid_cells):
        grid[valid_cells], _ = _normalized_rows(sums[valid_cells])
    pooled, pooled_valid = _normalized_rows(
        np.mean(descriptors[inside], axis=0, keepdims=True)
    )
    if not bool(pooled_valid[0]):
        raise RuntimeError("local descriptor pool unexpectedly became invalid")
    return ContextDescriptorSummary(
        pooled=pooled[0], grid=grid, valid_cells=valid_cells
    )


@dataclass(frozen=True)
class MultiscaleContextSummaries:
    """Reusable local summaries for one image anchor.

    Exporting a candidate group compares one query context with many support
    views.  Keeping these four pose-free summaries separate avoids silently
    changing the feature definition just to improve throughput.
    """

    intermediate3: ContextDescriptorSummary
    intermediate5: ContextDescriptorSummary
    alike3: ContextDescriptorSummary
    alike5: ContextDescriptorSummary


def summarize_multiscale_context(
    context3: LocalContextNodes,
    context5: LocalContextNodes,
    *,
    radius3_px: float,
    radius5_px: float,
) -> MultiscaleContextSummaries:
    """Build the fixed 3x3 and 5x5 summaries for a local image anchor."""

    return MultiscaleContextSummaries(
        intermediate3=summarize_context_descriptors(
            context3,
            descriptor_name="intermediate",
            grid_size=3,
            radius_px=float(radius3_px),
        ),
        intermediate5=summarize_context_descriptors(
            context5,
            descriptor_name="intermediate",
            grid_size=5,
            radius_px=float(radius5_px),
        ),
        alike3=summarize_context_descriptors(
            context3,
            descriptor_name="alike",
            grid_size=3,
            radius_px=float(radius3_px),
        ),
        alike5=summarize_context_descriptors(
            context5,
            descriptor_name="alike",
            grid_size=5,
            radius_px=float(radius5_px),
        ),
    )


def context_summary_similarity(
    query: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
) -> tuple[float, float, float]:
    """Return pooled cosine, aligned-grid cosine, and grid overlap fraction."""

    if query.grid.shape != support.grid.shape:
        raise ValueError("query and support context grids differ")
    common = query.valid_cells & support.valid_cells
    overlap = float(np.mean(common))
    if not np.any(common):
        return cosine_similarity(query.pooled, support.pooled), float("nan"), overlap
    grid_score = np.sum(query.grid[common] * support.grid[common], axis=1)
    return (
        cosine_similarity(query.pooled, support.pooled),
        float(np.clip(np.mean(grid_score), -1.0, 1.0)),
        overlap,
    )


def crop_spatial_grid_context(
    descriptors: np.ndarray,
    *,
    image_size: np.ndarray,
    xy: np.ndarray,
    window_size: int,
) -> ContextDescriptorSummary:
    """Crop an odd local window from a pose-free adaptive image grid.

    ``descriptors`` is indexed in image coordinates, not by a pose hypothesis.
    Out-of-image cells are explicitly invalid rather than replicated, so a
    border crop cannot silently acquire invented context evidence.
    """

    values = np.asarray(descriptors, dtype=np.float32)
    size = np.asarray(image_size, dtype=np.int64).reshape(2)
    center_xy = np.asarray(xy, dtype=np.float32).reshape(2)
    if (
        values.ndim != 3
        or values.shape[0] != values.shape[1]
        or values.shape[0] <= 0
        or values.shape[2] <= 0
        or np.any(size <= 0)
        or not np.isfinite(center_xy).all()
        or int(window_size) <= 0
        or int(window_size) % 2 != 1
    ):
        raise ValueError("spatial-grid context inputs are invalid")
    grid_size = int(values.shape[0])
    width, height = int(size[0]), int(size[1])
    column = int(np.clip(np.floor(center_xy[0] / float(width) * grid_size), 0, grid_size - 1))
    row = int(np.clip(np.floor(center_xy[1] / float(height) * grid_size), 0, grid_size - 1))
    radius = int(window_size) // 2
    output = np.zeros((int(window_size), int(window_size), values.shape[2]), dtype=np.float32)
    valid = np.zeros((int(window_size), int(window_size)), dtype=bool)
    for target_row in range(int(window_size)):
        source_row = row + target_row - radius
        if source_row < 0 or source_row >= grid_size:
            continue
        for target_column in range(int(window_size)):
            source_column = column + target_column - radius
            if source_column < 0 or source_column >= grid_size:
                continue
            descriptor = _normalized_vector(values[source_row, source_column])
            if descriptor is None:
                continue
            output[target_row, target_column] = descriptor
            valid[target_row, target_column] = True
    if not np.any(valid):
        raise RuntimeError("spatial-grid context crop unexpectedly has no valid cells")
    pooled, pooled_valid = _normalized_rows(
        np.mean(output[valid], axis=0, keepdims=True)
    )
    if not bool(pooled_valid[0]):
        raise RuntimeError("spatial-grid context crop has no valid pooled descriptor")
    return ContextDescriptorSummary(
        pooled=pooled[0],
        grid=output.reshape(-1, output.shape[2]),
        valid_cells=valid.reshape(-1),
    )


def context_shift_statistics(
    query: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
    *,
    maximum_shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mask-aware correlation and absolute support for fixed shifts.

    Correlation alone is unsafe for sparse contexts: one shared descriptor can
    produce a perfect cosine.  The companion overlap is normalized by the
    *full* grid area, so it records how much actual context supports each
    shift.  A score is NaN when there is no common cell; its overlap is zero.
    """

    if int(maximum_shift) < 0 or query.grid.shape != support.grid.shape:
        raise ValueError("context shift-correlation inputs are incompatible")
    cell_count, dimension = query.grid.shape
    grid_size = int(round(float(np.sqrt(cell_count))))
    if grid_size <= 0 or grid_size * grid_size != int(cell_count):
        raise ValueError("context shift-correlation needs square grids")
    query_grid = np.asarray(query.grid, dtype=np.float32).reshape(grid_size, grid_size, dimension)
    support_grid = np.asarray(support.grid, dtype=np.float32).reshape(
        grid_size, grid_size, dimension
    )
    query_valid = np.asarray(query.valid_cells, dtype=bool).reshape(grid_size, grid_size)
    support_valid = np.asarray(support.valid_cells, dtype=bool).reshape(grid_size, grid_size)
    scores: list[float] = []
    overlaps: list[float] = []
    for shift_row in range(-int(maximum_shift), int(maximum_shift) + 1):
        for shift_column in range(-int(maximum_shift), int(maximum_shift) + 1):
            query_row_start = max(0, -shift_row)
            query_row_end = min(grid_size, grid_size - shift_row)
            query_column_start = max(0, -shift_column)
            query_column_end = min(grid_size, grid_size - shift_column)
            support_row_start = query_row_start + shift_row
            support_row_end = query_row_end + shift_row
            support_column_start = query_column_start + shift_column
            support_column_end = query_column_end + shift_column
            common = (
                query_valid[query_row_start:query_row_end, query_column_start:query_column_end]
                & support_valid[
                    support_row_start:support_row_end,
                    support_column_start:support_column_end,
                ]
            )
            if not np.any(common):
                scores.append(float("nan"))
                overlaps.append(0.0)
                continue
            query_values = query_grid[
                query_row_start:query_row_end, query_column_start:query_column_end
            ][common]
            support_values = support_grid[
                support_row_start:support_row_end,
                support_column_start:support_column_end,
            ][common]
            scores.append(float(np.clip(np.mean(np.sum(query_values * support_values, axis=1)), -1.0, 1.0)))
            overlaps.append(float(np.sum(common)) / float(cell_count))
    return np.asarray(scores, dtype=np.float32), np.asarray(overlaps, dtype=np.float32)


def context_shift_correlation(
    query: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
    *,
    maximum_shift: int,
) -> np.ndarray:
    """Return only the correlation component of :func:`context_shift_statistics`."""

    return context_shift_statistics(
        query, support, maximum_shift=int(maximum_shift)
    )[0]


def context_shift_feature_names(prefix: str, *, maximum_shift: int) -> tuple[str, ...]:
    """Stable feature names matching :func:`context_shift_correlation`."""

    if not str(prefix) or int(maximum_shift) < 0:
        raise ValueError("context shift-feature naming inputs are invalid")
    return tuple(
        f"{prefix}_shift_dy{shift_row}_dx{shift_column}_cosine"
        for shift_row in range(-int(maximum_shift), int(maximum_shift) + 1)
        for shift_column in range(-int(maximum_shift), int(maximum_shift) + 1)
    )


def context_shift_overlap_feature_names(prefix: str, *, maximum_shift: int) -> tuple[str, ...]:
    """Stable overlap-field names matching :func:`context_shift_statistics`."""

    if not str(prefix) or int(maximum_shift) < 0:
        raise ValueError("context shift-overlap naming inputs are invalid")
    return tuple(
        f"{prefix}_shift_dy{shift_row}_dx{shift_column}_overlap_fraction"
        for shift_row in range(-int(maximum_shift), int(maximum_shift) + 1)
        for shift_column in range(-int(maximum_shift), int(maximum_shift) + 1)
    )


def context_regional_similarity(
    query: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
    *,
    region_grid_size: int = 3,
) -> tuple[np.ndarray, float]:
    """Summarize aligned local layout in fixed 2-D regions.

    Unlike one scalar grid average, this retains where each compatible or
    incompatible part of the surrounding facade occurs.  Regions without a
    shared observed cell remain NaN and are represented by the model's finite
    mask rather than treated as negative evidence.
    """

    if query.grid.shape != support.grid.shape or int(region_grid_size) <= 0:
        raise ValueError("context regional-similarity inputs are incompatible")
    cell_count, dimension = query.grid.shape
    grid_size = int(round(float(np.sqrt(cell_count))))
    if grid_size <= 0 or grid_size * grid_size != int(cell_count):
        raise ValueError("context regional similarity needs square grids")
    query_grid = np.asarray(query.grid, dtype=np.float32).reshape(grid_size, grid_size, dimension)
    support_grid = np.asarray(support.grid, dtype=np.float32).reshape(
        grid_size, grid_size, dimension
    )
    common = np.asarray(query.valid_cells, dtype=bool).reshape(grid_size, grid_size)
    common &= np.asarray(support.valid_cells, dtype=bool).reshape(grid_size, grid_size)
    values: list[float] = []
    row_splits = np.array_split(np.arange(grid_size), int(region_grid_size))
    column_splits = np.array_split(np.arange(grid_size), int(region_grid_size))
    for rows in row_splits:
        for columns in column_splits:
            mask = common[np.ix_(rows, columns)]
            if not np.any(mask):
                values.append(float("nan"))
                continue
            query_values = query_grid[np.ix_(rows, columns)][mask]
            support_values = support_grid[np.ix_(rows, columns)][mask]
            values.append(
                float(
                    np.clip(
                        np.mean(np.sum(query_values * support_values, axis=1)), -1.0, 1.0
                    )
                )
            )
    return np.asarray(values, dtype=np.float32), float(np.mean(common))


def landmark_region_prototype_similarity(
    query: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
    *,
    region_grid_size: int = LANDMARK_REGION_PROTOTYPE_REGION_GRID_SIZE,
) -> tuple[np.ndarray, float]:
    """Compare two landmark-centred crops through pooled spatial regions.

    The existing S1b regional feature uses the mean of same-cell cosine
    similarities.  A visual-region prototype instead first pools the
    descriptor cells inside each fixed spatial bin and then compares the two
    normalized bin descriptors.  This makes each bin a compact appearance
    prototype while retaining the absolute anchor-relative layout.  Missing
    bins remain ``NaN`` so downstream probes can distinguish lack of evidence
    from negative evidence.
    """

    if query.grid.shape != support.grid.shape or int(region_grid_size) <= 0:
        raise ValueError("landmark-region prototype inputs are incompatible")
    cell_count, dimension = query.grid.shape
    grid_size = int(round(float(np.sqrt(cell_count))))
    if grid_size <= 0 or grid_size * grid_size != int(cell_count):
        raise ValueError("landmark-region prototype needs square grids")
    query_grid = np.asarray(query.grid, dtype=np.float32).reshape(
        grid_size, grid_size, dimension
    )
    support_grid = np.asarray(support.grid, dtype=np.float32).reshape(
        grid_size, grid_size, dimension
    )
    query_valid = np.asarray(query.valid_cells, dtype=bool).reshape(grid_size, grid_size)
    support_valid = np.asarray(support.valid_cells, dtype=bool).reshape(
        grid_size, grid_size
    )
    common = query_valid & support_valid
    pooled_query = _normalized_vector(np.mean(query_grid[query_valid], axis=0))
    pooled_support = _normalized_vector(np.mean(support_grid[support_valid], axis=0))
    pooled = (
        float("nan")
        if pooled_query is None or pooled_support is None
        else cosine_similarity(pooled_query, pooled_support)
    )
    values: list[float] = [pooled]
    row_splits = np.array_split(np.arange(grid_size), int(region_grid_size))
    column_splits = np.array_split(np.arange(grid_size), int(region_grid_size))
    for rows in row_splits:
        for columns in column_splits:
            query_mask = query_valid[np.ix_(rows, columns)]
            support_mask = support_valid[np.ix_(rows, columns)]
            query_value = (
                None
                if not np.any(query_mask)
                else _normalized_vector(
                    np.mean(query_grid[np.ix_(rows, columns)][query_mask], axis=0)
                )
            )
            support_value = (
                None
                if not np.any(support_mask)
                else _normalized_vector(
                    np.mean(support_grid[np.ix_(rows, columns)][support_mask], axis=0)
                )
            )
            values.append(
                float("nan")
                if query_value is None or support_value is None
                else cosine_similarity(query_value, support_value)
            )
    return np.asarray(values, dtype=np.float32), float(np.mean(common))


def _regional_context_feature_vector(
    summary: ContextDescriptorSummary,
    support: ContextDescriptorSummary,
) -> np.ndarray:
    pooled, _aligned, _overlap = context_summary_similarity(summary, support)
    regional, valid_fraction = context_regional_similarity(summary, support)
    shift_scores, shift_overlaps = context_shift_statistics(
        summary, support, maximum_shift=1
    )
    return np.concatenate(
        [
            np.asarray([pooled], dtype=np.float32),
            regional.astype(np.float32, copy=False),
            np.asarray([valid_fraction], dtype=np.float32),
            shift_scores,
            shift_overlaps,
        ]
    )


def structured_multiscale_per_view_feature_vector(
    *,
    radio_final_anchor_cosine: float,
    radio_intermediate_anchor_cosine: float,
    alike_anchor_cosine: float,
    query_final_window3: ContextDescriptorSummary,
    support_final_window3: ContextDescriptorSummary,
    query_final_window5: ContextDescriptorSummary,
    support_final_window5: ContextDescriptorSummary,
    query_intermediate7: ContextDescriptorSummary,
    support_intermediate7: ContextDescriptorSummary,
    query_intermediate11: ContextDescriptorSummary,
    support_intermediate11: ContextDescriptorSummary,
    query_alike7: ContextDescriptorSummary,
    support_alike7: ContextDescriptorSummary,
    query_alike11: ContextDescriptorSummary,
    support_alike11: ContextDescriptorSummary,
) -> np.ndarray:
    """Build target-free large-context per-view features with 2-D layout."""

    output = np.concatenate(
        [
            np.asarray(
                [
                    float(radio_final_anchor_cosine),
                    float(radio_intermediate_anchor_cosine),
                    float(alike_anchor_cosine),
                ],
                dtype=np.float32,
            ),
            _regional_context_feature_vector(query_final_window3, support_final_window3),
            _regional_context_feature_vector(query_final_window5, support_final_window5),
            _regional_context_feature_vector(query_intermediate7, support_intermediate7),
            _regional_context_feature_vector(query_intermediate11, support_intermediate11),
            _regional_context_feature_vector(query_alike7, support_alike7),
            _regional_context_feature_vector(query_alike11, support_alike11),
        ]
    ).astype(np.float32, copy=False)
    if output.shape != (len(STRUCTURED_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),):
        raise RuntimeError("structured multiscale feature vector length drifted")
    if not np.isfinite(output[:3]).all():
        raise ValueError("structured multiscale anchor similarities are invalid")
    return output


def cost_volume_multiscale_per_view_feature_vector(
    *,
    radio_final_anchor_cosine: float,
    radio_intermediate_anchor_cosine: float,
    alike_anchor_cosine: float,
    query_final_window5: ContextDescriptorSummary,
    support_final_window5: ContextDescriptorSummary,
    query_final_window7: ContextDescriptorSummary,
    support_final_window7: ContextDescriptorSummary,
    query_intermediate11: ContextDescriptorSummary,
    support_intermediate11: ContextDescriptorSummary,
    query_alike11: ContextDescriptorSummary,
    support_alike11: ContextDescriptorSummary,
    temperature: float = 0.07,
) -> np.ndarray:
    """Return one frozen candidate-support-view multiscale cost-volume vector.

    This CPU helper is used by unit tests and small diagnostics.  Production
    exporters call :func:`batched_context_cost_volume_hough_features` directly
    so the same arithmetic can be evaluated efficiently on a GPU.
    """

    anchors = np.asarray(
        [
            float(radio_final_anchor_cosine),
            float(radio_intermediate_anchor_cosine),
            float(alike_anchor_cosine),
        ],
        dtype=np.float32,
    )
    if not np.isfinite(anchors).all():
        raise ValueError("cost-volume anchor similarities are invalid")
    contexts = (
        (query_final_window5, support_final_window5),
        (query_final_window7, support_final_window7),
        (query_intermediate11, support_intermediate11),
        (query_alike11, support_alike11),
    )
    fields: list[np.ndarray] = [anchors]
    with torch.no_grad():
        for query, support in contexts:
            query_grid, query_valid = resample_context_descriptor_grid(query)
            support_grid, support_valid = resample_context_descriptor_grid(support)
            value = batched_context_cost_volume_hough_features(
                torch.from_numpy(query_grid[None]),
                torch.from_numpy(query_valid[None]),
                torch.from_numpy(support_grid[None]),
                torch.from_numpy(support_valid[None]),
                temperature=float(temperature),
            )[0].cpu().numpy()
            fields.append(value.astype(np.float32, copy=False))
    output = np.concatenate(fields).astype(np.float32, copy=False)
    if output.shape != (len(COST_VOLUME_MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES),):
        raise RuntimeError("multiscale cost-volume feature vector length drifted")
    if not np.isfinite(output).all():
        raise RuntimeError("multiscale cost-volume feature vector is non-finite")
    return output


def multiscale_per_view_feature_vector(
    *,
    radio_final_anchor_cosine: float,
    query_final_grid4: np.ndarray,
    support_final_grid4: np.ndarray,
    query_context3: LocalContextNodes,
    support_context3: LocalContextNodes,
    query_context5: LocalContextNodes,
    support_context5: LocalContextNodes,
    radius3_px: float,
    radius5_px: float,
    query_summaries: MultiscaleContextSummaries | None = None,
    support_summaries: MultiscaleContextSummaries | None = None,
) -> np.ndarray:
    """Build one candidate-support-view feature vector without pose inputs."""

    query_final = np.asarray(query_final_grid4, dtype=np.float32).reshape(-1)
    support_final = np.asarray(support_final_grid4, dtype=np.float32).reshape(-1)
    if query_final.shape != support_final.shape or query_final.size == 0:
        raise ValueError("RADIO-final grid4 descriptors must be non-empty and aligned")
    final_grid4 = cosine_similarity(query_final, support_final)

    summaries_q = (
        summarize_multiscale_context(
            query_context3,
            query_context5,
            radius3_px=float(radius3_px),
            radius5_px=float(radius5_px),
        )
        if query_summaries is None
        else query_summaries
    )
    summaries_s = (
        summarize_multiscale_context(
            support_context3,
            support_context5,
            radius3_px=float(radius3_px),
            radius5_px=float(radius5_px),
        )
        if support_summaries is None
        else support_summaries
    )
    intermediate_pool3, intermediate_grid3, overlap3 = context_summary_similarity(
        summaries_q.intermediate3, summaries_s.intermediate3
    )
    intermediate_pool5, intermediate_grid5, overlap5 = context_summary_similarity(
        summaries_q.intermediate5, summaries_s.intermediate5
    )
    alike_pool3, alike_grid3, _ = context_summary_similarity(
        summaries_q.alike3, summaries_s.alike3
    )
    alike_pool5, alike_grid5, _ = context_summary_similarity(
        summaries_q.alike5, summaries_s.alike5
    )
    return np.asarray(
        (
            float(radio_final_anchor_cosine),
            float(final_grid4),
            cosine_similarity(
                query_context3.intermediate[0], support_context3.intermediate[0]
            ),
            intermediate_pool3,
            intermediate_grid3,
            intermediate_pool5,
            intermediate_grid5,
            cosine_similarity(query_context3.alike[0], support_context3.alike[0]),
            alike_pool3,
            alike_grid3,
            alike_pool5,
            alike_grid5,
            overlap3,
            overlap5,
        ),
        dtype=np.float32,
    )


def feature_indices_for_family(
    family: str,
    *,
    feature_names: Sequence[str] = MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
) -> np.ndarray:
    names = tuple(str(value) for value in feature_names)
    requested = ALL_MULTISCALE_CANDIDATE_PROBE_FAMILIES.get(str(family))
    if requested is None:
        raise ValueError(f"unsupported multiscale probe family: {family}")
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"feature tensor lacks requested family fields: {missing}")
    return np.asarray([names.index(name) for name in requested], dtype=np.int64)


@dataclass(frozen=True)
class PerViewFeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    feature_indices: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        indices = np.asarray(self.feature_indices, dtype=np.int64).reshape(-1)
        if len(mean) == 0 or mean.shape != scale.shape or len(indices) != len(mean):
            raise ValueError("per-view normalizer arrays are incompatible")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("per-view normalizer is invalid")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "feature_indices", indices)


def fit_per_view_feature_normalizer(
    features: np.ndarray,
    view_valid: np.ndarray,
    *,
    feature_indices: np.ndarray,
    train_groups: np.ndarray,
    batch_size: int = 256,
) -> PerViewFeatureNormalizer:
    values = np.asarray(features)
    valid_views = np.asarray(view_valid, dtype=bool)
    groups = np.asarray(train_groups, dtype=bool).reshape(-1)
    indices = np.asarray(feature_indices, dtype=np.int64).reshape(-1)
    if values.ndim != 4 or values.shape[:3] != valid_views.shape or values.shape[0] != len(groups):
        raise ValueError("per-view feature arrays are incompatible")
    if (
        indices.size == 0
        or np.any((indices < 0) | (indices >= values.shape[3]))
        or int(batch_size) <= 0
    ):
        raise ValueError("per-view feature indices are invalid")
    rows = np.flatnonzero(groups)
    if rows.size == 0:
        raise ValueError("per-view feature normalizer has no train groups")
    counts = np.zeros((len(indices),), dtype=np.int64)
    sums = np.zeros((len(indices),), dtype=np.float64)
    squared_sums = np.zeros((len(indices),), dtype=np.float64)
    # Do not materialize all selected high-dimensional features.  The wide
    # full-correlation schema can exceed tens of GB after float conversion.
    for begin in range(0, len(rows), int(batch_size)):
        batch_rows = rows[begin : begin + int(batch_size)]
        selected = np.asarray(
            values[batch_rows][..., indices], dtype=np.float32
        )
        selected_valid = valid_views[batch_rows][..., None]
        finite = np.isfinite(selected)
        mask = selected_valid & finite
        masked = np.where(mask, selected, 0.0).astype(np.float64, copy=False)
        counts += np.sum(mask, axis=(0, 1, 2), dtype=np.int64)
        sums += np.sum(masked, axis=(0, 1, 2), dtype=np.float64)
        squared_sums += np.sum(
            masked * masked, axis=(0, 1, 2), dtype=np.float64
        )
    mean = np.zeros((len(indices),), dtype=np.float32)
    scale = np.ones((len(indices),), dtype=np.float32)
    present = counts > 0
    if np.any(present):
        mean64 = np.zeros_like(sums)
        mean64[present] = sums[present] / counts[present]
        variance = np.zeros_like(sums)
        variance[present] = np.maximum(
            squared_sums[present] / counts[present] - mean64[present] ** 2,
            0.0,
        )
        mean[present] = mean64[present].astype(np.float32)
        scale[present] = np.maximum(
            np.sqrt(variance[present]).astype(np.float32), 1e-3
        )
    return PerViewFeatureNormalizer(
        mean=mean, scale=scale, feature_indices=indices
    )


def normalized_per_view_model_input(
    features: np.ndarray,
    normalizer: PerViewFeatureNormalizer,
) -> np.ndarray:
    values = np.asarray(features)
    if values.ndim != 4 or np.any(
        normalizer.feature_indices >= values.shape[3]
    ):
        raise ValueError("per-view model input is incompatible with normalizer")
    selected = np.asarray(
        values[..., normalizer.feature_indices], dtype=np.float32
    )
    finite = np.isfinite(selected)
    standardized = (np.where(finite, selected, normalizer.mean) - normalizer.mean) / normalizer.scale
    return np.concatenate(
        [standardized.astype(np.float32), finite.astype(np.float32)], axis=-1
    )


def aggregate_per_view_logits(
    logits: torch.Tensor, view_valid: torch.Tensor
) -> torch.Tensor:
    """Marginalize support-view logits with an order-invariant uniform mixture."""

    if logits.ndim != 3 or logits.shape != view_valid.shape:
        raise ValueError("per-view logits and masks must have shape (Q, L, V)")
    valid = view_valid.to(dtype=torch.bool)
    masked = torch.where(valid, logits, torch.full_like(logits, -torch.inf))
    counts = valid.sum(dim=2)
    output = torch.logsumexp(masked, dim=2) - torch.log(
        counts.clamp_min(1).to(dtype=logits.dtype)
    )
    return torch.where(counts > 0, output, torch.full_like(output, -torch.inf))


class PerViewLinearCandidateProbe(nn.Module):
    """Small candidate-group classifier with a separate explicit null logit."""

    def __init__(self, input_dim: int, *, use_base_prior_residual: bool = False) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("per-view probe input dimension must be positive")
        self.linear = nn.Linear(int(input_dim), 1)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.use_base_prior_residual = bool(use_base_prior_residual)
        if self.use_base_prior_residual:
            # With a fixed candidate/null prior passed to forward(), exact zero
            # visual offsets must reproduce the baseline probability mass.
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        features: torch.Tensor,
        view_valid: torch.Tensor,
        base_candidate_log_probability: torch.Tensor | None = None,
        base_null_log_probability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[:3] != view_valid.shape:
            raise ValueError("per-view probe tensors are incompatible")
        if (base_candidate_log_probability is None) != (base_null_log_probability is None):
            raise ValueError("candidate and null base priors must be supplied together")
        if self.use_base_prior_residual != (base_candidate_log_probability is not None):
            raise ValueError("per-view probe base-prior mode differs from forward inputs")
        per_view = self.linear(features).squeeze(-1)
        candidates = aggregate_per_view_logits(per_view, view_valid)
        null = self.null_logit.expand(len(candidates), 1)
        if base_candidate_log_probability is not None:
            if (
                base_candidate_log_probability.shape != candidates.shape
                or base_null_log_probability.shape != (len(candidates),)
            ):
                raise ValueError("per-view probe base prior tensors are incompatible")
            candidates = candidates + base_candidate_log_probability
            null = null + base_null_log_probability[:, None]
        return torch.cat([candidates, null], dim=1), per_view


def aggregate_per_view_learned_mixture(
    evidence_logits: torch.Tensor,
    view_logits: torch.Tensor,
    view_valid: torch.Tensor,
) -> torch.Tensor:
    """Marginalize view-specific evidence with a learned, normalized mixture.

    The view weights are normalized independently for every candidate.  This
    preserves the per-view likelihood factorization instead of averaging view
    embeddings before scoring them.  A zero evidence head therefore evaluates
    to an exact zero residual for every supported candidate, regardless of the
    learned view weights.
    """

    if (
        evidence_logits.ndim != 3
        or view_logits.shape != evidence_logits.shape
        or view_valid.shape != evidence_logits.shape
    ):
        raise ValueError("per-view learned-mixture tensors are incompatible")
    valid = view_valid.to(dtype=torch.bool)
    counts = valid.sum(dim=2)
    masked_view = torch.where(valid, view_logits, torch.full_like(view_logits, -torch.inf))
    normalizer = torch.logsumexp(masked_view, dim=2)
    safe_normalizer = torch.where(counts > 0, normalizer, torch.zeros_like(normalizer))
    log_weights = masked_view - safe_normalizer[:, :, None]
    mixture = torch.where(
        valid,
        evidence_logits + log_weights,
        torch.full_like(evidence_logits, -torch.inf),
    )
    output = torch.logsumexp(mixture, dim=2)
    return torch.where(counts > 0, output, torch.full_like(output, -torch.inf))


class PerViewMLPCandidateProbe(nn.Module):
    """Small nonlinear residual probe with an explicit per-view mixture.

    This is deliberately a capacity control for the frozen S1 feature schema,
    not a new source of image evidence.  It lets the experiment distinguish a
    linear fusion failure from a representation failure before generating a
    much larger context artifact.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        use_base_prior_residual: bool = False,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("per-view MLP probe dimensions must be positive")
        self.backbone = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.evidence = nn.Linear(int(hidden_dim), 1)
        self.view = nn.Linear(int(hidden_dim), 1)
        self.null_logit = nn.Parameter(torch.zeros(()))
        self.use_base_prior_residual = bool(use_base_prior_residual)
        self.hidden_dim = int(hidden_dim)
        if self.use_base_prior_residual:
            # A zero evidence residual plus normalized view mixture leaves the
            # supplied candidate/null probability distribution unchanged.
            nn.init.zeros_(self.evidence.weight)
            nn.init.zeros_(self.evidence.bias)
            nn.init.zeros_(self.view.weight)
            nn.init.zeros_(self.view.bias)

    def forward(
        self,
        features: torch.Tensor,
        view_valid: torch.Tensor,
        base_candidate_log_probability: torch.Tensor | None = None,
        base_null_log_probability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[:3] != view_valid.shape:
            raise ValueError("per-view MLP probe tensors are incompatible")
        if (base_candidate_log_probability is None) != (base_null_log_probability is None):
            raise ValueError("candidate and null base priors must be supplied together")
        if self.use_base_prior_residual != (base_candidate_log_probability is not None):
            raise ValueError("per-view MLP base-prior mode differs from forward inputs")
        hidden = self.backbone(features)
        per_view = self.evidence(hidden).squeeze(-1)
        view_logits = self.view(hidden).squeeze(-1)
        candidates = aggregate_per_view_learned_mixture(per_view, view_logits, view_valid)
        null = self.null_logit.expand(len(candidates), 1)
        if base_candidate_log_probability is not None:
            if (
                base_candidate_log_probability.shape != candidates.shape
                or base_null_log_probability.shape != (len(candidates),)
            ):
                raise ValueError("per-view MLP base prior tensors are incompatible")
            candidates = candidates + base_candidate_log_probability
            null = null + base_null_log_probability[:, None]
        return torch.cat([candidates, null], dim=1), per_view


def _build_per_view_candidate_probe(
    *,
    architecture: str,
    input_dim: int,
    hidden_dim: int,
    use_base_prior_residual: bool,
) -> nn.Module:
    if str(architecture) == "linear":
        return PerViewLinearCandidateProbe(
            int(input_dim), use_base_prior_residual=bool(use_base_prior_residual)
        )
    if str(architecture) == "mlp":
        return PerViewMLPCandidateProbe(
            int(input_dim),
            hidden_dim=int(hidden_dim),
            use_base_prior_residual=bool(use_base_prior_residual),
        )
    raise ValueError(f"unsupported per-view probe architecture: {architecture}")


def set_membership_negative_log_likelihood(
    log_probability: torch.Tensor, target_membership: torch.Tensor
) -> torch.Tensor:
    """Negative log total mass assigned to each valid candidate/null set."""

    if log_probability.shape != target_membership.shape or log_probability.ndim != 2:
        raise ValueError("set-membership log probabilities and targets differ")
    selected = torch.where(
        target_membership.to(dtype=torch.bool),
        log_probability,
        torch.full_like(log_probability, -torch.inf),
    )
    return -torch.logsumexp(selected, dim=1).mean()


def _base_prior_log_probabilities(
    candidate_probabilities: np.ndarray,
    null_probabilities: np.ndarray,
    *,
    view_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and log a fixed candidate-plus-null prior for residual fitting."""

    candidate = np.asarray(candidate_probabilities, dtype=np.float32)
    null = np.asarray(null_probabilities, dtype=np.float32).reshape(-1)
    valid = np.asarray(view_valid, dtype=bool)
    if candidate.shape != valid.shape[:2] or null.shape != (len(candidate),):
        raise ValueError("per-view probe base prior arrays are incompatible")
    if (
        np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(candidate < 0.0)
        or np.any(null <= 0.0)
        or np.any(candidate[~np.any(valid, axis=2)] != 0.0)
    ):
        raise ValueError("per-view probe base prior contains invalid mass")
    mass = candidate.sum(axis=1, dtype=np.float64) + null.astype(np.float64)
    if np.max(np.abs(mass - 1.0)) > 1e-4:
        raise ValueError("per-view probe base prior does not conserve mass")
    log_candidate = np.full(candidate.shape, -np.inf, dtype=np.float32)
    positive = candidate > 0.0
    log_candidate[positive] = np.log(candidate[positive])
    return log_candidate, np.log(null).astype(np.float32)


def train_per_view_candidate_probe(
    *,
    features: np.ndarray,
    view_valid: np.ndarray,
    targets: np.ndarray | None = None,
    train_groups: np.ndarray,
    target_membership: np.ndarray | None = None,
    base_candidate_probabilities: np.ndarray | None = None,
    base_null_probabilities: np.ndarray | None = None,
    family: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    architecture: str = "linear",
    hidden_dim: int = 64,
    feature_names: Sequence[str] = MULTISCALE_CANDIDATE_PROBE_FEATURE_NAMES,
    precompute_normalized_model_input: bool = False,
    cache_precomputed_train_input_on_device: bool = False,
) -> tuple[nn.Module, PerViewFeatureNormalizer, dict[str, object]]:
    """Fit a group-softmax probe using hard targets or set-valued memberships.

    A detector point can legitimately be close to several SfM tracks.  Soft
    cross-entropy would incorrectly force an arbitrary uniform distribution
    among those tracks.  The set-valued route instead maximizes the *total*
    probability of any geometrically valid candidate.  The hard-index route
    remains available for unit tests and exact-track diagnostics.
    """

    if int(epochs) <= 0 or int(batch_size) <= 0 or float(learning_rate) <= 0.0:
        raise ValueError("per-view probe optimization parameters are invalid")
    raw = np.asarray(features)
    valid = np.asarray(view_valid, dtype=bool)
    groups = np.asarray(train_groups, dtype=bool).reshape(-1)
    if raw.ndim != 4 or raw.shape[:3] != valid.shape or raw.shape[0] != len(groups):
        raise ValueError("per-view probe training arrays are incompatible")
    if (targets is None) == (target_membership is None):
        raise ValueError("provide exactly one of hard targets or target membership")
    if (base_candidate_probabilities is None) != (base_null_probabilities is None):
        raise ValueError("candidate and null base probabilities must be supplied together")
    target_mode: str
    if targets is not None:
        labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        if labels.size == 0 or len(labels) != len(groups) or np.any(
            (labels < 0) | (labels > raw.shape[1])
        ):
            raise ValueError("per-view probe targets must index candidates or null")
        membership = np.zeros((len(labels), raw.shape[1] + 1), dtype=bool)
        membership[np.arange(len(labels)), labels] = True
        target_mode = "hard_candidate_or_null"
    else:
        raw_membership = np.asarray(target_membership)
        if raw_membership.shape != (len(groups), raw.shape[1] + 1):
            raise ValueError("per-view target membership must align with candidates plus null")
        if not np.issubdtype(raw_membership.dtype, np.bool_):
            numeric = np.asarray(raw_membership, dtype=np.float32)
            if np.any(~np.isfinite(numeric)) or np.any((numeric != 0.0) & (numeric != 1.0)):
                raise ValueError("per-view target membership must be binary and finite")
        membership = raw_membership.astype(bool, copy=False)
        target_mode = "set_membership_candidate_or_null"
    if np.any(np.sum(membership, axis=1) == 0):
        raise ValueError("every per-view group needs a target candidate set or null")
    candidate_target = membership[:, :-1]
    supported_candidate = np.any(valid, axis=2)
    unsupported_positive = candidate_target & ~supported_candidate
    if np.any(unsupported_positive):
        raise ValueError("a positive target candidate has no real support view")
    indices = feature_indices_for_family(str(family), feature_names=feature_names)
    normalizer = fit_per_view_feature_normalizer(
        raw,
        valid,
        feature_indices=indices,
        train_groups=groups,
        batch_size=min(int(batch_size), 256),
    )
    # Large full-correlation feature schemas can be tens of GB after float
    # conversion, so this remains opt-in.  For compact frozen probes, however,
    # normalizing the exact same rows in every epoch is pure CPU copy overhead.
    prepared_features = (
        normalized_per_view_model_input(raw, normalizer)
        if bool(precompute_normalized_model_input)
        else None
    )
    if bool(cache_precomputed_train_input_on_device) and prepared_features is None:
        raise ValueError(
            "device-resident train input requires precompute_normalized_model_input"
        )
    use_base_prior_residual = base_candidate_probabilities is not None
    if use_base_prior_residual:
        base_candidate_log, base_null_log = _base_prior_log_probabilities(
            np.asarray(base_candidate_probabilities),
            np.asarray(base_null_probabilities),
            view_valid=valid,
        )
    fit_groups = np.flatnonzero(groups)
    if len(fit_groups) == 0:
        raise ValueError("per-view probe has no supported train groups")
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = _build_per_view_candidate_probe(
        architecture=str(architecture),
        input_dim=int(len(normalizer.feature_indices) * 2),
        hidden_dim=int(hidden_dim),
        use_base_prior_residual=use_base_prior_residual,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    valid_tensor = torch.from_numpy(valid)
    target_tensor = torch.from_numpy(membership)
    base_candidate_tensor = (
        None if not use_base_prior_residual else torch.from_numpy(base_candidate_log)
    )
    base_null_tensor = None if not use_base_prior_residual else torch.from_numpy(base_null_log)
    cached_train_features = None
    cached_train_valid = None
    cached_train_target = None
    cached_train_base_candidate = None
    cached_train_base_null = None
    if bool(cache_precomputed_train_input_on_device):
        # Keep only train rows on device.  This preserves the exact existing
        # batch objective while removing repeated host indexing and PCIe copies
        # for wide frozen visual schemas.
        cached_train_features = torch.from_numpy(prepared_features[fit_groups]).to(device)
        cached_train_valid = valid_tensor[fit_groups].to(device)
        cached_train_target = target_tensor[fit_groups].to(device=device, dtype=torch.bool)
        if base_candidate_tensor is not None:
            cached_train_base_candidate = base_candidate_tensor[fit_groups].to(device)
            cached_train_base_null = base_null_tensor[fit_groups].to(device)
    last_loss = float("nan")
    model.train()
    for _epoch in range(int(epochs)):
        for begin in range(0, len(fit_groups), int(batch_size)):
            if begin == 0:
                order = (
                    rng.permutation(len(fit_groups))
                    if cached_train_features is not None
                    else rng.permutation(fit_groups)
                )
            batch_rows = order[begin : begin + int(batch_size)]
            if cached_train_features is not None:
                batch = torch.as_tensor(batch_rows, dtype=torch.long, device=device)
                logits, _ = model(
                    cached_train_features[batch],
                    cached_train_valid[batch],
                    None
                    if cached_train_base_candidate is None
                    else cached_train_base_candidate[batch],
                    None if cached_train_base_null is None else cached_train_base_null[batch],
                )
                batch_target = cached_train_target[batch]
            else:
                batch = torch.as_tensor(batch_rows, dtype=torch.long)
                batch_features = (
                    prepared_features[batch_rows]
                    if prepared_features is not None
                    else normalized_per_view_model_input(raw[batch_rows], normalizer)
                )
                logits, _ = model(
                    torch.from_numpy(batch_features).to(device),
                    valid_tensor[batch].to(device),
                    None
                    if base_candidate_tensor is None
                    else base_candidate_tensor[batch].to(device),
                    None if base_null_tensor is None else base_null_tensor[batch].to(device),
                )
                batch_target = target_tensor[batch].to(device=device, dtype=torch.bool)
            log_probability = F.log_softmax(logits, dim=1)
            loss = set_membership_negative_log_likelihood(
                log_probability, batch_target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    model.eval()
    return model, normalizer, {
        "family": str(family),
        "target_mode": target_mode,
        "training_objective": "set_log_mass_nll_over_target_membership_v1",
        "use_base_prior_residual": bool(use_base_prior_residual),
        "architecture": str(architecture),
        "hidden_dim": (None if str(architecture) == "linear" else int(hidden_dim)),
        "train_group_count": int(np.sum(groups)),
        "fit_group_count": int(len(fit_groups)),
        "unsupported_positive_train_group_count": int(
            np.sum(groups & np.any(unsupported_positive, axis=1))
        ),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "last_train_loss": last_loss,
        "precompute_normalized_model_input": bool(precompute_normalized_model_input),
        "cache_precomputed_train_input_on_device": bool(
            cache_precomputed_train_input_on_device
        ),
    }


@torch.inference_mode()
def predict_per_view_candidate_probe(
    model: nn.Module,
    *,
    features: np.ndarray,
    view_valid: np.ndarray,
    normalizer: PerViewFeatureNormalizer,
    device: torch.device,
    batch_size: int,
    base_candidate_probabilities: np.ndarray | None = None,
    base_null_probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate probabilities, null probability, and per-view logits."""

    if int(batch_size) <= 0:
        raise ValueError("per-view prediction batch size must be positive")
    raw = np.asarray(features)
    valid = np.asarray(view_valid, dtype=bool)
    if raw.ndim != 4 or raw.shape[:3] != valid.shape:
        raise ValueError("per-view prediction arrays are incompatible")
    use_base_prior_residual = base_candidate_probabilities is not None
    if not hasattr(model, "use_base_prior_residual"):
        raise ValueError("per-view prediction model lacks base-prior mode")
    if use_base_prior_residual != bool(model.use_base_prior_residual) or (
        (base_candidate_probabilities is None) != (base_null_probabilities is None)
    ):
        raise ValueError("per-view prediction base-prior mode is incompatible with model")
    if use_base_prior_residual:
        base_candidate_log, base_null_log = _base_prior_log_probabilities(
            np.asarray(base_candidate_probabilities),
            np.asarray(base_null_probabilities),
            view_valid=valid,
        )
    candidates: list[np.ndarray] = []
    nulls: list[np.ndarray] = []
    views: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(raw), int(batch_size)):
        end = min(start + int(batch_size), len(raw))
        batch_values = normalized_per_view_model_input(raw[start:end], normalizer)
        logits, per_view = model(
            torch.from_numpy(batch_values).to(device),
            torch.from_numpy(valid[start:end]).to(device),
            None
            if not use_base_prior_residual
            else torch.from_numpy(base_candidate_log[start:end]).to(device),
            None
            if not use_base_prior_residual
            else torch.from_numpy(base_null_log[start:end]).to(device),
        )
        probability = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        candidates.append(probability[:, :-1])
        nulls.append(probability[:, -1])
        views.append(per_view.cpu().numpy().astype(np.float32))
    return (
        np.concatenate(candidates, axis=0),
        np.concatenate(nulls, axis=0),
        np.concatenate(views, axis=0),
    )


def train_per_view_linear_probe(
    *,
    features: np.ndarray,
    view_valid: np.ndarray,
    targets: np.ndarray | None = None,
    train_groups: np.ndarray,
    target_membership: np.ndarray | None = None,
    base_candidate_probabilities: np.ndarray | None = None,
    base_null_probabilities: np.ndarray | None = None,
    family: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[PerViewLinearCandidateProbe, PerViewFeatureNormalizer, dict[str, object]]:
    """Compatibility wrapper for the original linear S1 probe."""

    model, normalizer, metadata = train_per_view_candidate_probe(
        features=features,
        view_valid=view_valid,
        targets=targets,
        train_groups=train_groups,
        target_membership=target_membership,
        base_candidate_probabilities=base_candidate_probabilities,
        base_null_probabilities=base_null_probabilities,
        family=family,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        architecture="linear",
    )
    if not isinstance(model, PerViewLinearCandidateProbe):
        raise RuntimeError("linear probe factory returned an unexpected model")
    return model, normalizer, metadata


def predict_per_view_linear_probe(
    model: PerViewLinearCandidateProbe,
    *,
    features: np.ndarray,
    view_valid: np.ndarray,
    normalizer: PerViewFeatureNormalizer,
    device: torch.device,
    batch_size: int,
    base_candidate_probabilities: np.ndarray | None = None,
    base_null_probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for the original linear S1 probe."""

    return predict_per_view_candidate_probe(
        model,
        features=features,
        view_valid=view_valid,
        normalizer=normalizer,
        device=device,
        batch_size=batch_size,
        base_candidate_probabilities=base_candidate_probabilities,
        base_null_probabilities=base_null_probabilities,
    )
