"""Pose-free SfM-maplet transport features for frozen full-track probes.

Each candidate/support-observation edge is represented by the context *around*
the observed landmark, rather than by the landmark centre descriptor itself.
The support side is a sparse maplet of real same-image SfM observations; the
query side is a dense four-quadrant context around the frozen query point.

The compact Hough representation retains relative quadrant agreement while
allowing a one-quadrant cross-view translation.  It is deliberately a frozen
S1 feature extractor: it receives neither poses nor labels, and it never
selects candidates or support observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


FULLTRACK_SFM_MAPLET_TRANSPORT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_sfm_maplet_transport_v1"
)
FULLTRACK_SFM_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS = (
    "center_excluded_sfm_maplet_quadrant_transport_per_real_sfm_observation_v1"
)
FULLTRACK_SFM_MAPLET_TRANSPORT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_center_excluded_sfm_maplet_quadrant_transport_v1"
)

_QUADRANT_GRID_SIZE = 2
_Hough_GRID_SIZE = 2 * _QUADRANT_GRID_SIZE - 1
_Hough_SHIFT_ROWS = tuple(range(-(_QUADRANT_GRID_SIZE - 1), _QUADRANT_GRID_SIZE))
_Hough_SHIFT_COLUMNS = _Hough_SHIFT_ROWS


@dataclass(frozen=True)
class SfmMapletTransportProfile:
    """One source/radius pair with a center-excluded quadrant context."""

    name: str
    source_name: str
    radius_cells: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.radius_cells) < 2
        ):
            raise ValueError("SfM-maplet transport profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "radius_cells", int(self.radius_cells))

    @property
    def window_size(self) -> int:
        return 2 * int(self.radius_cells) + 1


# RADIO final and intermediate are deliberately tested independently at a
# near facade scale and a larger contextual scale.  ALIKE is withheld from
# this first maplet probe: its local texture was already tested extensively,
# while this feature family is intended to test structural context.
SFM_MAPLET_TRANSPORT_PROFILES = (
    SfmMapletTransportProfile(
        name="radio_final_sfm_maplet_near", source_name="radio_final", radius_cells=4
    ),
    SfmMapletTransportProfile(
        name="radio_final_sfm_maplet_wide", source_name="radio_final", radius_cells=7
    ),
    SfmMapletTransportProfile(
        name="radio_intermediate_sfm_maplet_near",
        source_name="radio_intermediate",
        radius_cells=4,
    ),
    SfmMapletTransportProfile(
        name="radio_intermediate_sfm_maplet_wide",
        source_name="radio_intermediate",
        radius_cells=7,
    ),
)


def sfm_maplet_transport_profile_feature_names(
    profile: SfmMapletTransportProfile,
) -> tuple[str, ...]:
    """Return the learned, coverage-free Hough fields for one profile."""

    fields = ("mean_cosine", "attention_mass", "mutual_mass")
    return tuple(
        f"{profile.name}_hough_dy{dy}_dx{dx}_{field}"
        for field in fields
        for dy in _Hough_SHIFT_ROWS
        for dx in _Hough_SHIFT_COLUMNS
    )


SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE = {
    profile.name: sfm_maplet_transport_profile_feature_names(profile)
    for profile in SFM_MAPLET_TRANSPORT_PROFILES
}
SFM_MAPLET_TRANSPORT_FEATURE_NAMES = tuple(
    name
    for profile in SFM_MAPLET_TRANSPORT_PROFILES
    for name in SFM_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
)


def _quadrant_masks(
    *, window_size: int, device: torch.device
) -> torch.Tensor:
    """Return the four center-excluded quadrant masks of an odd patch."""

    window = int(window_size)
    if window < 5 or window % 2 != 1:
        raise ValueError("quadrant context window must be odd and at least five")
    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long, device=device)
    rows, columns = torch.meshgrid(offsets, offsets, indexing="ij")
    return torch.stack(
        (
            (rows < 0) & (columns < 0),
            (rows < 0) & (columns > 0),
            (rows > 0) & (columns < 0),
            (rows > 0) & (columns > 0),
        ),
        dim=0,
    )


def pool_center_excluded_query_quadrants(
    *,
    patches: torch.Tensor,
    valid: torch.Tensor,
    minimum_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool an odd dense crop into four non-central, fixed relative regions."""

    values = torch.as_tensor(patches)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    fraction = float(minimum_fraction)
    if (
        values.ndim != 4
        or values.shape[0] == 0
        or values.shape[2] != values.shape[3]
        or mask.shape != (values.shape[0], values.shape[2], values.shape[3])
        or not 0.0 < fraction <= 1.0
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("query quadrant pooling inputs are invalid")
    masks = _quadrant_masks(window_size=int(values.shape[2]), device=values.device)
    cell_count = masks.reshape(4, -1).sum(dim=1)
    if torch.any(cell_count <= 0):  # pragma: no cover - guarded by window validation
        raise RuntimeError("quadrant mask is empty")
    selected = mask[:, None] & masks[None]
    counts = selected.reshape(values.shape[0], 4, -1).sum(dim=2)
    pooled = torch.einsum(
        "bchw,bqhw->bqc", values, selected.to(dtype=values.dtype)
    ) / counts.clamp_min(1).to(dtype=values.dtype)[..., None]
    pooled = F.normalize(pooled, p=2, dim=2, eps=1e-8)
    required = torch.ceil(cell_count.to(dtype=torch.float32) * fraction).to(dtype=torch.long)
    usable = counts >= required[None]
    return pooled, usable, counts


def pool_sparse_support_quadrants(
    *,
    descriptors: torch.Tensor,
    present: torch.Tensor,
    minimum_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool fixed same-image SfM neighbors within each relative quadrant."""

    values = torch.as_tensor(descriptors)
    mask = torch.as_tensor(present, dtype=torch.bool, device=values.device)
    minimum = int(minimum_neighbors)
    if (
        values.ndim != 4
        or values.shape[0] == 0
        or values.shape[1] != 4
        or mask.shape != values.shape[:3]
        or minimum <= 0
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("sparse support quadrant pooling inputs are invalid")
    counts = mask.sum(dim=2)
    pooled = (values * mask[..., None].to(dtype=values.dtype)).sum(dim=2)
    pooled = pooled / counts.clamp_min(1).to(dtype=values.dtype)[..., None]
    pooled = F.normalize(pooled, p=2, dim=2, eps=1e-8)
    return pooled, counts >= minimum, counts


def _hough_indices(*, device: torch.device) -> torch.Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(_QUADRANT_GRID_SIZE, dtype=torch.long, device=device),
        torch.arange(_QUADRANT_GRID_SIZE, dtype=torch.long, device=device),
        indexing="ij",
    )
    flat_rows = rows.reshape(-1)
    flat_columns = columns.reshape(-1)
    shift_rows = flat_rows[None, :] - flat_rows[:, None] + (_QUADRANT_GRID_SIZE - 1)
    shift_columns = (
        flat_columns[None, :] - flat_columns[:, None] + (_QUADRANT_GRID_SIZE - 1)
    )
    return shift_rows * _Hough_GRID_SIZE + shift_columns


def batched_sfm_maplet_quadrant_transport_features(
    *,
    query_quadrants: torch.Tensor,
    query_valid: torch.Tensor,
    support_quadrants: torch.Tensor,
    support_valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return coverage-free 2x2 Hough features for frozen maplet context.

    The returned fields are only appearance agreement statistics.  Valid-pair
    coverage and entropy are intentionally excluded because they describe
    observation availability rather than candidate identity.
    """

    query = torch.as_tensor(query_quadrants)
    support = torch.as_tensor(
        support_quadrants, dtype=query.dtype, device=query.device
    )
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    support_mask = torch.as_tensor(
        support_valid, dtype=torch.bool, device=query.device
    )
    if (
        query.ndim != 3
        or support.shape != query.shape
        or query.shape[0] == 0
        or query.shape[1] != _QUADRANT_GRID_SIZE**2
        or query_mask.shape != query.shape[:2]
        or support_mask.shape != query.shape[:2]
        or not float(temperature) > 0.0
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(support).all())
    ):
        raise ValueError("SfM-maplet quadrant transport inputs are invalid")
    query = F.normalize(query.to(dtype=torch.float32), p=2, dim=2, eps=1e-8)
    support = F.normalize(support.to(dtype=torch.float32), p=2, dim=2, eps=1e-8)
    pair_valid = query_mask[:, :, None] & support_mask[:, None, :]
    pair_valid_float = pair_valid.to(dtype=query.dtype)
    cosine = torch.bmm(query, support.transpose(1, 2))
    cosine = torch.where(pair_valid, cosine, torch.zeros_like(cosine))
    batch = int(query.shape[0])
    hough_index = _hough_indices(device=query.device).reshape(1, -1).expand(batch, -1)
    hough_count = _Hough_GRID_SIZE**2

    def scatter(values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((batch, hough_count), dtype=query.dtype, device=query.device)
        return output.scatter_add_(1, hough_index, values.reshape(batch, -1))

    counts = scatter(pair_valid_float)
    mean_cosine = scatter(cosine * pair_valid_float) / counts.clamp_min(1.0)
    flat_valid = pair_valid.reshape(batch, -1)
    has_pair = torch.any(flat_valid, dim=1, keepdim=True)
    sentinel = torch.full_like(cosine, -1e9)
    flat_logits = torch.where(pair_valid, cosine / float(temperature), sentinel).reshape(batch, -1)
    attention = torch.softmax(flat_logits, dim=1)
    attention = torch.where(has_pair, attention, torch.zeros_like(attention))
    attention_mass = scatter(attention.reshape_as(cosine))
    row_logits = torch.where(pair_valid, cosine / float(temperature), sentinel)
    column_logits = row_logits.transpose(1, 2)
    row_probability = torch.softmax(row_logits, dim=2)
    column_probability = torch.softmax(column_logits, dim=2).transpose(1, 2)
    mutual_mass = scatter(row_probability * column_probability * pair_valid_float)
    output = torch.cat((mean_cosine, attention_mass, mutual_mass), dim=1)
    if output.shape != (
        batch,
        len(sfm_maplet_transport_profile_feature_names(SFM_MAPLET_TRANSPORT_PROFILES[0])),
    ):
        raise RuntimeError("SfM-maplet transport feature layout drifted")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("SfM-maplet transport features are non-finite")
    return output
