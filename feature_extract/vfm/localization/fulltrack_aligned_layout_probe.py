"""Pose-free per-view absolute-layout evidence for full-track probes.

The existing full-track path stores one NCC scalar per support observation.  This
module keeps a compact, location-aware representation instead: matching cells
from query and support crops are compared at the same relative crop position,
then encoded with a low-frequency 2-D DCT.  It retains facade phase without
serializing an unbounded all-pairs correlation volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_aligned_layout_v1"
)
FULLTRACK_ALIGNED_LAYOUT_APPEARANCE_VERSION = (
    "frozen_fulltrack_candidate_per_view_aligned_layout_v1"
)
FULLTRACK_ALIGNED_LAYOUT_EDGE_FEATURE_SEMANTICS = (
    "aligned_spatial_dct_per_real_sfm_observation_v1"
)
FULLTRACK_ALIGNED_LAYOUT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_aligned_spatial_dct_v1"
)


@dataclass(frozen=True)
class AlignedLayoutProfile:
    """One fixed real-image context scale and its retained DCT coefficients."""

    name: str
    source_name: str
    window_size: int
    dct_size: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.window_size) <= 0
            or int(self.window_size) % 2 != 1
            or int(self.dct_size) <= 0
            or int(self.dct_size) > int(self.window_size)
        ):
            raise ValueError("aligned-layout profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "dct_size", int(self.dct_size))


# These scales are fixed before validation labels are read.  The final branch
# captures coarse semantic context, intermediate retains facade structure, and
# ALIKE contributes high-frequency local detail.  All are centered on the
# immutable query/support observation anchors.
ALIGNED_LAYOUT_PROFILES = (
    AlignedLayoutProfile(
        name="radio_final_aligned_layout7",
        source_name="radio_final",
        window_size=7,
        dct_size=3,
    ),
    AlignedLayoutProfile(
        # The intermediate cache is a 16x16 map.  A 13x13 crop leaves too
        # little real-image support near ordinary observation boundaries and
        # would make the multiscale mixture fail its fixed minimum coverage
        # gate.  9x9 retains a distinct structural scale while preserving
        # enough target-free support observations for a neutral-missing model.
        name="radio_intermediate_aligned_layout9",
        source_name="radio_intermediate",
        window_size=9,
        dct_size=4,
    ),
    AlignedLayoutProfile(
        name="alike_aligned_layout9",
        source_name="alike",
        window_size=9,
        dct_size=4,
    ),
)


def aligned_layout_profile_feature_names(
    profile: AlignedLayoutProfile,
) -> tuple[str, ...]:
    """Return stable DCT coordinate names for one fixed profile."""

    return tuple(
        f"{profile.name}__dct_r{row}_c{column}"
        for row in range(int(profile.dct_size))
        for column in range(int(profile.dct_size))
    )


ALIGNED_LAYOUT_FEATURE_NAMES_BY_PROFILE = {
    profile.name: aligned_layout_profile_feature_names(profile)
    for profile in ALIGNED_LAYOUT_PROFILES
}
ALIGNED_LAYOUT_FEATURE_NAMES = tuple(
    name
    for profile in ALIGNED_LAYOUT_PROFILES
    for name in ALIGNED_LAYOUT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
ALIGNED_LAYOUT_RADIO_FINAL_FEATURE_NAMES = ALIGNED_LAYOUT_FEATURE_NAMES_BY_PROFILE[
    "radio_final_aligned_layout7"
]
ALIGNED_LAYOUT_RADIO_INTERMEDIATE_FEATURE_NAMES = (
    ALIGNED_LAYOUT_FEATURE_NAMES_BY_PROFILE["radio_intermediate_aligned_layout9"]
)
ALIGNED_LAYOUT_ALIKE_FEATURE_NAMES = ALIGNED_LAYOUT_FEATURE_NAMES_BY_PROFILE[
    "alike_aligned_layout9"
]


def _orthonormal_dct_basis(
    *, size: int, component_count: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Construct the first ``component_count`` orthonormal DCT-II vectors."""

    length = int(size)
    count = int(component_count)
    if length <= 0 or count <= 0 or count > length:
        raise ValueError("DCT basis dimensions are invalid")
    coordinates = torch.arange(length, dtype=dtype, device=device)
    frequencies = torch.arange(count, dtype=dtype, device=device)[:, None]
    basis = torch.cos(
        torch.pi
        * (coordinates[None, :] + 0.5)
        * frequencies
        / float(length)
    )
    scale = torch.full(
        (count,), np.sqrt(2.0 / float(length)), dtype=dtype, device=device
    )
    scale[0] = np.sqrt(1.0 / float(length))
    return basis * scale[:, None]


def aligned_spatial_dct_features(
    *,
    query_patches: torch.Tensor,
    support_patches: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    dct_size: int,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode same-relative-cell visual agreement without a mask feature.

    Invalid cells are filled with the observed edge mean *after* centering, so
    crop-boundary geometry cannot be used as an implicit appearance cue.  The
    first coefficient stores the aligned mean cosine; the remaining
    coefficients encode the absolute layout of deviations around that mean.
    ``usable`` is emitted separately so callers can keep missing evidence
    neutral rather than treating its zero placeholder as a negative match.
    """

    query = torch.as_tensor(query_patches)
    support = torch.as_tensor(support_patches)
    qmask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    smask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    if (
        query.ndim != 4
        or support.shape != query.shape
        or qmask.shape != query.shape[:1] + query.shape[2:]
        or smask.shape != qmask.shape
        or query.shape[2] != query.shape[3]
        or int(dct_size) <= 0
        or int(dct_size) > int(query.shape[2])
        or not 0.0 < float(minimum_support_fraction) <= 1.0
        or not 0.0 < float(minimum_overlap_fraction) <= 1.0
    ):
        raise ValueError("aligned-layout DCT inputs are invalid")
    query = F.normalize(query.float(), p=2, dim=1)
    support = F.normalize(support.float(), p=2, dim=1)
    pair_valid = qmask & smask
    pair_weight = pair_valid.to(dtype=query.dtype)
    cell_count = float(query.shape[2] * query.shape[3])
    query_fraction = qmask.to(dtype=query.dtype).sum(dim=(1, 2)) / cell_count
    support_fraction = smask.to(dtype=query.dtype).sum(dim=(1, 2)) / cell_count
    overlap_fraction = pair_weight.sum(dim=(1, 2)) / cell_count
    usable = (
        (query_fraction >= float(minimum_support_fraction))
        & (support_fraction >= float(minimum_support_fraction))
        & (overlap_fraction >= float(minimum_overlap_fraction))
    )
    cosine = torch.sum(query * support, dim=1)
    observed_count = pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
    mean = (cosine * pair_weight).sum(dim=(1, 2)) / observed_count
    centered = torch.where(
        pair_valid,
        cosine - mean[:, None, None],
        torch.zeros_like(cosine),
    )
    basis = _orthonormal_dct_basis(
        size=int(query.shape[2]),
        component_count=int(dct_size),
        device=query.device,
        dtype=query.dtype,
    )
    coefficients = torch.einsum("byx,uy,vx->buv", centered, basis, basis)
    output = coefficients.reshape(len(query), -1)
    output = output.clone()
    output[:, 0] = mean
    output = torch.where(usable[:, None], output, torch.zeros_like(output))
    if output.shape != (len(query), int(dct_size) ** 2) or not bool(
        torch.isfinite(output).all()
    ):
        raise RuntimeError("aligned-layout DCT output is invalid")
    return output, usable
