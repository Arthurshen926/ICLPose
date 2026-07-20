"""Partial-coverage, center-excluded SfM-maplet transport primitives.

The first maplet probe required a populated support maplet in every one of
four quadrants.  That turns ordinary sparse SfM coverage into an accidental
all-or-nothing gate: most real candidate/support edges become unknown before
their visual context can be tested.  This v2 representation keeps the same
target-free, per-real-support-view maplet topology, but retains every partial
maplet with a minimum total neighbour count.  A paired topology-only control
is exported separately so observation density, quadrant availability, and
query-border overlap cannot silently become identity evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


FULLTRACK_SPARSE_MAPLET_TRANSPORT_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_sparse_maplet_transport_v2"
)
FULLTRACK_SPARSE_MAPLET_TRANSPORT_EDGE_FEATURE_SEMANTICS = (
    "candidate_specific_partial_sfm_maplet_transport_per_real_sfm_observation_v2"
)
FULLTRACK_SPARSE_MAPLET_TRANSPORT_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_partial_sfm_maplet_transport_v2"
)
FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_APPEARANCE_FORMAT = (
    "frozen_fulltrack_candidate_per_view_sparse_maplet_topology_control_v2"
)
FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_EDGE_FEATURE_SEMANTICS = (
    "candidate_specific_partial_sfm_maplet_topology_control_per_real_sfm_observation_v2"
)
FULLTRACK_SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_GRANULARITY = (
    "sparse_per_real_support_view_partial_sfm_maplet_topology_control_v2"
)

SPARSE_MAPLET_QUADRANT_COUNT = 4
SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT = 4
SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS = 4
SPARSE_MAPLET_TEMPERATURE = 0.07


@dataclass(frozen=True)
class SparseMapletTransportProfile:
    """One source-specific, centre-excluded sparse maplet scale."""

    name: str
    source_name: str
    grid_size: int
    radius_cells: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or int(self.grid_size) <= 0
            or int(self.radius_cells) < 2
            or 2 * int(self.radius_cells) + 1 > int(self.grid_size)
        ):
            raise ValueError("sparse-maplet transport profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(self, "radius_cells", int(self.radius_cells))

    @property
    def window_size(self) -> int:
        return 2 * int(self.radius_cells) + 1


# The RADIO branches carry facade-scale context.  ALIKE/FPN contributes a
# high-resolution maplet branch, but remains a distinct profile so an OOF
# audit can attribute any gain to one source rather than a mixed feature bag.
SPARSE_MAPLET_TRANSPORT_PROFILES = (
    SparseMapletTransportProfile(
        "radio_final_sparse_maplet_near", "radio_final", 16, 4
    ),
    SparseMapletTransportProfile(
        "radio_final_sparse_maplet_wide", "radio_final", 16, 7
    ),
    SparseMapletTransportProfile(
        "radio_intermediate_pca256_sparse_maplet_near",
        "radio_intermediate_pca256",
        16,
        4,
    ),
    SparseMapletTransportProfile(
        "radio_intermediate_pca256_sparse_maplet_wide",
        "radio_intermediate_pca256",
        16,
        7,
    ),
    SparseMapletTransportProfile("alike_fpn_sparse_maplet_near", "alike_fpn", 32, 8),
    SparseMapletTransportProfile("alike_fpn_sparse_maplet_wide", "alike_fpn", 32, 15),
)


def _quadrant_masks(*, window_size: int, device: torch.device) -> torch.Tensor:
    """Return the four centre-excluded quadrant masks of one odd crop."""

    window = int(window_size)
    if window < 5 or window % 2 != 1:
        raise ValueError("sparse-maplet quadrant window is invalid")
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


def _hough_indices(*, device: torch.device) -> torch.Tensor:
    """Map query/support quadrant pairs to their relative 3x3 shift bin."""

    rows, columns = torch.meshgrid(
        torch.arange(2, dtype=torch.long, device=device),
        torch.arange(2, dtype=torch.long, device=device),
        indexing="ij",
    )
    flat_rows = rows.reshape(-1)
    flat_columns = columns.reshape(-1)
    shifts_row = flat_rows[None, :] - flat_rows[:, None] + 1
    shifts_column = flat_columns[None, :] - flat_columns[:, None] + 1
    return shifts_row * 3 + shifts_column


def _hough_pair_counts(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    indices = _hough_indices(device=device).reshape(-1)
    return torch.bincount(indices, minlength=9).to(dtype=dtype)


def sparse_maplet_transport_feature_names(profile: SparseMapletTransportProfile) -> tuple[str, ...]:
    """Return stable appearance-only Hough columns for one maplet profile."""

    fields = ("mean_cosine", "attention_mass", "mutual_mass")
    return tuple(
        f"{profile.name}_hough_dy{dy:+d}_dx{dx:+d}_{field}"
        for field in fields
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )


def sparse_maplet_topology_control_feature_names(
    profile: SparseMapletTransportProfile,
) -> tuple[str, ...]:
    """Return matched, descriptor-free topology and border-control columns."""

    output: list[str] = []
    for field in ("pair_valid_fraction", "pair_neighbor_fraction"):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                output.append(f"{profile.name}_hough_dy{dy:+d}_dx{dx:+d}_{field}")
    for quadrant in range(SPARSE_MAPLET_QUADRANT_COUNT):
        output.append(
            f"{profile.name}_query_quadrant_{quadrant}_original_crop_overlap_fraction"
        )
    for quadrant in range(SPARSE_MAPLET_QUADRANT_COUNT):
        output.append(
            f"{profile.name}_support_quadrant_{quadrant}_neighbor_fraction"
        )
    return tuple(output)


SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE = {
    profile.name: sparse_maplet_transport_feature_names(profile)
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
}
SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE = {
    profile.name: sparse_maplet_topology_control_feature_names(profile)
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
}
SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES = tuple(
    name
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    for name in SPARSE_MAPLET_TRANSPORT_FEATURE_NAMES_BY_PROFILE[profile.name]
)
SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES = tuple(
    name
    for profile in SPARSE_MAPLET_TRANSPORT_PROFILES
    for name in SPARSE_MAPLET_TOPOLOGY_CONTROL_FEATURE_NAMES_BY_PROFILE[profile.name]
)


def pool_sparse_maplet_query_quadrants(
    *,
    patches: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool a dense crop into four non-central quadrants.

    ``valid`` is intentionally caller-controlled: visual export uses reflected
    real-map values everywhere, while the paired control receives the original
    in-bounds mask.  The returned coverage is diagnostic/control-only.
    """

    values = torch.as_tensor(patches)
    mask = torch.as_tensor(valid, dtype=torch.bool, device=values.device)
    if (
        values.ndim != 4
        or values.shape[1] != values.shape[2]
        or mask.shape != values.shape[:3]
        or values.shape[0] == 0
        or values.shape[3] == 0
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("sparse-maplet query quadrant inputs are invalid")
    masks = _quadrant_masks(window_size=int(values.shape[1]), device=values.device)
    selected = mask[:, None] & masks[None]
    counts = selected.reshape(values.shape[0], 4, -1).sum(dim=2)
    possible = masks.reshape(4, -1).sum(dim=1).to(dtype=values.dtype)
    pooled = torch.einsum("bhwc,bqhw->bqc", values, selected.to(dtype=values.dtype))
    pooled = pooled / counts.clamp_min(1).to(dtype=values.dtype)[..., None]
    pooled = F.normalize(pooled, p=2, dim=2, eps=1e-8)
    coverage = counts.to(dtype=values.dtype) / possible[None]
    return pooled, counts > 0, coverage


def sparse_maplet_quadrant_transport_features(
    *,
    query_quadrants: torch.Tensor,
    query_valid: torch.Tensor,
    support_quadrants: torch.Tensor,
    support_valid: torch.Tensor,
    temperature: float = SPARSE_MAPLET_TEMPERATURE,
) -> torch.Tensor:
    """Compute appearance-only partial-maplet transport features.

    Missing quadrant pairs contribute no visual score.  This is not treated as
    evidence against a candidate; the caller emits the result only if the
    target-free minimum *total* support-neighbour condition is met, and the
    matched topology-control artifact exposes the same missing pattern.
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
        or query.shape[1] != SPARSE_MAPLET_QUADRANT_COUNT
        or query_mask.shape != query.shape[:2]
        or support_mask.shape != query.shape[:2]
        or float(temperature) <= 0.0
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(support).all())
    ):
        raise ValueError("sparse-maplet transport inputs are invalid")
    query = F.normalize(query.to(dtype=torch.float32), p=2, dim=2, eps=1e-8)
    support = F.normalize(support.to(dtype=torch.float32), p=2, dim=2, eps=1e-8)
    pair_valid = query_mask[:, :, None] & support_mask[:, None, :]
    pair_valid_float = pair_valid.to(dtype=query.dtype)
    cosine = torch.bmm(query, support.transpose(1, 2))
    cosine = torch.where(pair_valid, cosine, torch.zeros_like(cosine))
    batch = int(query.shape[0])
    hough_index = _hough_indices(device=query.device).reshape(1, -1).expand(batch, -1)

    def scatter(values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((batch, 9), dtype=query.dtype, device=query.device)
        return output.scatter_add_(1, hough_index, values.reshape(batch, -1))

    counts = scatter(pair_valid_float)
    mean_cosine = scatter(cosine * pair_valid_float) / counts.clamp_min(1.0)
    flat_valid = pair_valid.reshape(batch, -1)
    has_pair = torch.any(flat_valid, dim=1, keepdim=True)
    logits = torch.where(
        flat_valid,
        (cosine / float(temperature)).reshape(batch, -1),
        torch.full((batch, 16), -1e9, dtype=query.dtype, device=query.device),
    )
    attention = torch.softmax(logits, dim=1)
    attention = torch.where(has_pair, attention, torch.zeros_like(attention))
    attention_mass = scatter(attention.reshape_as(cosine))
    row_logits = torch.where(
        pair_valid, cosine / float(temperature), torch.full_like(cosine, -1e9)
    )
    column_logits = row_logits.transpose(1, 2)
    row_probability = torch.softmax(row_logits, dim=2)
    column_probability = torch.softmax(column_logits, dim=2).transpose(1, 2)
    mutual_mass = scatter(row_probability * column_probability * pair_valid_float)
    output = torch.cat((mean_cosine, attention_mass, mutual_mass), dim=1)
    if output.shape != (batch, 27) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("sparse-maplet transport feature layout drifted")
    return output


def sparse_maplet_topology_control_features(
    *,
    query_original_coverage: torch.Tensor,
    support_neighbor_counts: torch.Tensor,
    maximum_neighbors_per_quadrant: int = SPARSE_MAPLET_MAX_NEIGHBORS_PER_QUADRANT,
) -> torch.Tensor:
    """Emit only availability and crop-overlap controls for one profile."""

    query_coverage = torch.as_tensor(query_original_coverage, dtype=torch.float32)
    support_counts = torch.as_tensor(
        support_neighbor_counts, dtype=torch.float32, device=query_coverage.device
    )
    maximum = int(maximum_neighbors_per_quadrant)
    if (
        query_coverage.ndim != 2
        or support_counts.shape != query_coverage.shape
        or query_coverage.shape[0] == 0
        or query_coverage.shape[1] != SPARSE_MAPLET_QUADRANT_COUNT
        or maximum <= 0
        or not bool(torch.isfinite(query_coverage).all())
        or not bool(torch.isfinite(support_counts).all())
        or bool(torch.any(query_coverage < 0.0))
        or bool(torch.any(query_coverage > 1.0))
        or bool(torch.any(support_counts < 0.0))
        or bool(torch.any(support_counts > float(maximum)))
    ):
        raise ValueError("sparse-maplet topology control inputs are invalid")
    batch = int(query_coverage.shape[0])
    hough_index = _hough_indices(device=query_coverage.device).reshape(1, -1).expand(
        batch, -1
    )
    hough_counts = _hough_pair_counts(
        device=query_coverage.device, dtype=query_coverage.dtype
    )

    def scatter(values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            (batch, 9), dtype=query_coverage.dtype, device=query_coverage.device
        )
        return output.scatter_add_(1, hough_index, values.reshape(batch, -1))

    support_fraction = support_counts / float(maximum)
    pair_valid = (
        (query_coverage > 0.0).to(dtype=query_coverage.dtype)[:, :, None]
        * (support_counts > 0.0).to(dtype=query_coverage.dtype)[:, None, :]
    )
    pair_neighbor = query_coverage[:, :, None] * support_fraction[:, None, :]
    pair_valid = scatter(pair_valid) / hough_counts[None]
    pair_neighbor = scatter(pair_neighbor) / hough_counts[None]
    output = torch.cat((pair_valid, pair_neighbor, query_coverage, support_fraction), dim=1)
    if output.shape != (batch, 26) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("sparse-maplet topology control feature layout drifted")
    return output


def sparse_maplet_support_usable(
    support_neighbor_counts: torch.Tensor,
    *,
    minimum_total_neighbors: int = SPARSE_MAPLET_MINIMUM_TOTAL_NEIGHBORS,
) -> torch.Tensor:
    """Return the target-free minimum-support condition for one maplet edge."""

    counts = torch.as_tensor(support_neighbor_counts)
    minimum = int(minimum_total_neighbors)
    if (
        counts.ndim != 2
        or counts.shape[1] != SPARSE_MAPLET_QUADRANT_COUNT
        or minimum <= 0
        or bool(torch.any(counts < 0))
    ):
        raise ValueError("sparse-maplet support usability inputs are invalid")
    return counts.sum(dim=1) >= minimum
