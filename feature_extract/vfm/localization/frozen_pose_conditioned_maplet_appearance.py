"""Frozen candidate-specific 3-D maplet appearance likelihood primitives.

This is deliberately a diagnostic P1 representation, not a PnP input.  It
keeps the global top-L candidate posterior and support-view posterior fixed,
then uses only *centre-excluded* SfM neighbours of each support observation.
For a tested pose those neighbouring 3-D tracks are projected into the query
feature map.  Therefore the score asks whether a candidate's surrounding
real-image maplet, rather than only its central landmark, is compatible with
the hypothesised pose.

The module contains no pose targets, no image retrieval, and no rendering.
Target-side rank/pose conclusions belong in a separate audit program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    SupportObservationGeometryIndex,
)


FROZEN_POSE_CONDITIONED_MAPLET_APPEARANCE_VERSION = (
    "frozen_candidate_specific_center_excluded_sfm_maplet_pose_likelihood_v1"
)


@dataclass(frozen=True)
class FrozenMapletAppearanceProfile:
    """One source-specific, centre-excluded support-neighbour topology."""

    name: str
    source_name: str
    topology_key: str
    grid_size: int
    radius_cells: int

    def __post_init__(self) -> None:
        if (
            not str(self.name)
            or not str(self.source_name)
            or not str(self.topology_key)
            or int(self.grid_size) <= 1
            or int(self.radius_cells) < 2
        ):
            raise ValueError("frozen maplet appearance profile is invalid")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "topology_key", str(self.topology_key))
        object.__setattr__(self, "grid_size", int(self.grid_size))
        object.__setattr__(self, "radius_cells", int(self.radius_cells))


# Each topology has four fixed quadrants and up to four real SfM observations
# per quadrant.  The scorer may use one or two slots per quadrant as a frozen
# cost/coverage ablation, but never changes the topology based on a pose.
FROZEN_MAPLET_APPEARANCE_PROFILES = (
    FrozenMapletAppearanceProfile(
        "radio_final_near", "radio_final", "grid16_radius4", 16, 4
    ),
    FrozenMapletAppearanceProfile(
        "radio_final_wide", "radio_final", "grid16_radius7", 16, 7
    ),
    FrozenMapletAppearanceProfile(
        "radio_intermediate_near", "radio_intermediate", "grid16_radius4", 16, 4
    ),
    FrozenMapletAppearanceProfile(
        "radio_intermediate_wide", "radio_intermediate", "grid16_radius7", 16, 7
    ),
    FrozenMapletAppearanceProfile("alike_near", "alike", "grid32_radius8", 32, 8),
    FrozenMapletAppearanceProfile("alike_wide", "alike", "grid32_radius15", 32, 15),
)


@dataclass(frozen=True)
class FrozenCandidateMapletEvidenceLayout:
    """Frozen held-out candidate groups and their fixed support views."""

    verification_source_rows: np.ndarray
    verification_xy: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    null_probabilities: np.ndarray
    support_view_probabilities: np.ndarray
    support_image_ids: np.ndarray
    support_view_valid: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.verification_source_rows, dtype=np.int64).reshape(-1)
        query_xy = np.asarray(self.verification_xy, dtype=np.float32).reshape(-1, 2)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        probabilities = np.asarray(self.candidate_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        view_probabilities = np.asarray(self.support_view_probabilities, dtype=np.float32)
        support_ids = np.asarray(self.support_image_ids).astype(str)
        view_valid = np.asarray(self.support_view_valid, dtype=bool)
        if (
            len(rows) == 0
            or len(np.unique(rows)) != len(rows)
            or query_xy.shape != (len(rows), 2)
            or tracks.ndim != 2
            or tracks.shape[0] != len(rows)
            or probabilities.shape != tracks.shape
            or null.shape != (len(rows),)
            or view_probabilities.ndim != 3
            or view_probabilities.shape[:2] != tracks.shape
            or support_ids.shape != view_probabilities.shape
            or view_valid.shape != view_probabilities.shape
            or np.any(~np.isfinite(query_xy))
            or np.any(~np.isfinite(probabilities))
            or np.any(~np.isfinite(null))
            or np.any(~np.isfinite(view_probabilities))
            or np.any(probabilities < 0.0)
            or np.any(null < 0.0)
            or np.any(view_probabilities < 0.0)
            or np.any(view_valid & (support_ids == ""))
        ):
            raise ValueError("frozen candidate maplet evidence layout is invalid")
        candidate_valid = tracks >= 0
        if (
            np.any((probabilities > 0.0) & ~candidate_valid)
            or np.any(candidate_valid & ~np.any(view_valid, axis=2))
            or np.any(np.abs(probabilities.sum(axis=1, dtype=np.float64) + null - 1.0) > 2e-4)
            or np.any(
                np.abs(
                    view_probabilities.sum(axis=2, dtype=np.float64)
                    - candidate_valid.astype(np.float64)
                )
                > 2e-4
            )
            or np.any((~view_valid) & (view_probabilities != 0.0))
        ):
            raise ValueError("frozen candidate/support-view posterior is inconsistent")
        object.__setattr__(self, "verification_source_rows", rows)
        object.__setattr__(self, "verification_xy", query_xy)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_probabilities", probabilities)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "support_view_probabilities", view_probabilities)
        object.__setattr__(self, "support_image_ids", support_ids)
        object.__setattr__(self, "support_view_valid", view_valid)

    @property
    def point_count(self) -> int:
        return int(self.candidate_track_ids.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_track_ids.shape[1])

    @property
    def support_view_count(self) -> int:
        return int(self.support_view_probabilities.shape[2])


@dataclass(frozen=True)
class FrozenCandidateMapletProfileLayout:
    """One profile's real support-neighbour geometry for frozen candidates."""

    profile: FrozenMapletAppearanceProfile
    anchor_geometry_rows: np.ndarray
    neighbor_track_ids: np.ndarray
    neighbor_xyz: np.ndarray
    neighbor_support_xy: np.ndarray
    neighbor_valid: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_geometry_rows, dtype=np.int64)
        tracks = np.asarray(self.neighbor_track_ids, dtype=np.int64)
        xyz = np.asarray(self.neighbor_xyz, dtype=np.float32)
        xy = np.asarray(self.neighbor_support_xy, dtype=np.float32)
        valid = np.asarray(self.neighbor_valid, dtype=bool)
        if (
            anchors.ndim != 3
            or tracks.ndim != 5
            or tracks.shape[:3] != anchors.shape
            or tracks.shape[3] != 4
            or tracks.shape[4] <= 0
            or xyz.shape != (*tracks.shape, 3)
            or xy.shape != (*tracks.shape, 2)
            or valid.shape != tracks.shape
            or np.any(anchors < -1)
            or np.any(tracks < -1)
            or np.any(valid & (tracks < 0))
            or np.any(~np.isfinite(xyz[valid]))
            or np.any(~np.isfinite(xy[valid]))
        ):
            raise ValueError("frozen candidate maplet profile layout is invalid")
        object.__setattr__(self, "anchor_geometry_rows", anchors)
        object.__setattr__(self, "neighbor_track_ids", tracks)
        object.__setattr__(self, "neighbor_xyz", xyz)
        object.__setattr__(self, "neighbor_support_xy", xy)
        object.__setattr__(self, "neighbor_valid", valid)

    @property
    def slots_per_quadrant(self) -> int:
        return int(self.neighbor_valid.shape[-1])


def _resolve_canonical_rows(
    requested_track_ids: np.ndarray, canonical_track_ids: np.ndarray
) -> np.ndarray:
    """Resolve physical IDs exactly, retaining ``-1`` for missing tracks."""

    requested = np.asarray(requested_track_ids, dtype=np.int64)
    canonical = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    if canonical.size == 0 or np.unique(canonical).size != len(canonical):
        raise ValueError("canonical landmark tracks must be unique and non-empty")
    result = np.full(requested.shape, -1, dtype=np.int64)
    valid = requested >= 0
    if not np.any(valid):
        return result
    order = np.argsort(canonical, kind="stable")
    sorted_tracks = canonical[order]
    positions = np.searchsorted(sorted_tracks, requested[valid])
    in_range = positions < len(sorted_tracks)
    safe_positions = np.minimum(positions, len(sorted_tracks) - 1)
    found = in_range & (sorted_tracks[safe_positions] == requested[valid])
    values = np.full(len(positions), -1, dtype=np.int64)
    values[found] = order[safe_positions[found]]
    result[valid] = values
    return result


def build_frozen_candidate_maplet_profile_layout(
    *,
    evidence: FrozenCandidateMapletEvidenceLayout,
    profile: FrozenMapletAppearanceProfile,
    neighbor_topology: np.ndarray,
    support_geometry: SupportObservationGeometryIndex,
    canonical_track_ids: np.ndarray,
    canonical_xyz: np.ndarray,
) -> FrozenCandidateMapletProfileLayout:
    """Attach only fixed, centre-excluded real SfM neighbours to candidates."""

    topology = np.asarray(neighbor_topology, dtype=np.int64)
    canonical = np.asarray(canonical_track_ids, dtype=np.int64).reshape(-1)
    xyz_bank = np.asarray(canonical_xyz, dtype=np.float32).reshape(-1, 3)
    if (
        topology.ndim != 3
        or topology.shape[0] != len(support_geometry)
        or topology.shape[1] != 4
        or topology.shape[2] <= 0
        or np.any(topology < -1)
        or xyz_bank.shape != (len(canonical), 3)
        or np.any(~np.isfinite(xyz_bank))
    ):
        raise ValueError("frozen maplet topology or landmark bank is invalid")
    candidates = evidence.candidate_track_ids
    view_ids = evidence.support_image_ids
    view_valid = evidence.support_view_valid
    anchors = np.full(view_ids.shape, -1, dtype=np.int64)
    flat_valid = view_valid.reshape(-1)
    flat_ids = view_ids.reshape(-1)
    flat_tracks = np.repeat(candidates[:, :, None], evidence.support_view_count, axis=2).reshape(-1)
    flat_anchors = anchors.reshape(-1)
    for image_id in np.unique(flat_ids[flat_valid]).tolist():
        rows = np.flatnonzero(flat_valid & (flat_ids == str(image_id)))
        geometry_rows = support_geometry.geometry_rows_for_tracks(
            str(image_id), flat_tracks[rows]
        )
        if np.any(geometry_rows < 0):
            raise ValueError("frozen candidate support view lacks its SfM anchor observation")
        flat_anchors[rows] = geometry_rows
    if np.any(view_valid & (anchors < 0)):
        raise RuntimeError("frozen candidate maplet anchor resolution drifted")

    safe_anchors = anchors.clip(min=0)
    neighbor_rows = topology[safe_anchors]
    neighbor_present = (anchors[..., None, None] >= 0) & (neighbor_rows >= 0)
    safe_neighbors = neighbor_rows.clip(min=0)
    neighbor_tracks = support_geometry.track_ids[safe_neighbors]
    neighbor_xy = support_geometry.xy[safe_neighbors]
    bank_rows = _resolve_canonical_rows(neighbor_tracks, canonical)
    neighbor_valid = neighbor_present & (bank_rows >= 0)
    neighbor_xyz = np.zeros((*neighbor_tracks.shape, 3), dtype=np.float32)
    if np.any(neighbor_valid):
        neighbor_xyz[neighbor_valid] = xyz_bank[bank_rows[neighbor_valid]]
    neighbor_tracks = np.where(neighbor_valid, neighbor_tracks, -1).astype(np.int64)
    neighbor_xy = np.where(neighbor_valid[..., None], neighbor_xy, 0.0).astype(np.float32)

    center_tracks = candidates[..., None, None]
    if np.any(neighbor_valid & (neighbor_tracks == center_tracks)):
        raise RuntimeError("centre track leaked into a centre-excluded maplet")
    return FrozenCandidateMapletProfileLayout(
        profile=profile,
        anchor_geometry_rows=anchors,
        neighbor_track_ids=neighbor_tracks,
        neighbor_xyz=neighbor_xyz,
        neighbor_support_xy=neighbor_xy,
        neighbor_valid=neighbor_valid,
    )


def select_maplet_slots(
    layout: FrozenCandidateMapletProfileLayout, *, slots_per_quadrant: int
) -> FrozenCandidateMapletProfileLayout:
    """Use a fixed prefix of each topology quadrant for a declared ablation."""

    slots = int(slots_per_quadrant)
    if not 0 < slots <= layout.slots_per_quadrant:
        raise ValueError("maplet slots per quadrant exceed the frozen topology")
    return FrozenCandidateMapletProfileLayout(
        profile=layout.profile,
        anchor_geometry_rows=layout.anchor_geometry_rows,
        neighbor_track_ids=layout.neighbor_track_ids[..., :slots],
        neighbor_xyz=layout.neighbor_xyz[..., :slots, :],
        neighbor_support_xy=layout.neighbor_support_xy[..., :slots, :],
        neighbor_valid=layout.neighbor_valid[..., :slots],
    )


def pool_maplet_neighbor_log_ratios(
    *,
    neighbor_log_ratios: torch.Tensor,
    neighbor_active: torch.Tensor,
    neighbor_present: torch.Tensor,
    minimum_active_neighbors_per_quadrant: int,
    minimum_active_quadrants: int,
    quadrant_reduction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool fixed neighbour evidence without selecting a pose-specific subset.

    Returns ``(view_log_ratios, view_usable, active_quadrant_counts)``.  A
    view becomes usable only if it explains a predeclared number of distinct
    support-maplet quadrants.  Missing slots are not filled by a query-centre
    fallback and are never treated as a new correspondence selection.
    """

    ratios = torch.as_tensor(neighbor_log_ratios)
    active = torch.as_tensor(neighbor_active, dtype=torch.bool, device=ratios.device)
    present = torch.as_tensor(neighbor_present, dtype=torch.bool, device=ratios.device)
    min_neighbors = int(minimum_active_neighbors_per_quadrant)
    min_quadrants = int(minimum_active_quadrants)
    reduction = str(quadrant_reduction)
    if (
        ratios.ndim != 6
        or ratios.shape[4] != 4
        or active.shape != ratios.shape
        or present.shape != ratios.shape[1:]
        or min_neighbors <= 0
        or min_quadrants <= 0
        or min_quadrants > 4
        or reduction not in {"mean", "median"}
        or not bool(torch.isfinite(ratios).all())
        or torch.any(active & ~present[None])
    ):
        raise ValueError("maplet neighbour likelihood tensors are invalid")
    counts = active.sum(dim=5)
    quadrant_usable = counts >= min_neighbors
    quadrant_values = (ratios * active.to(dtype=ratios.dtype)).sum(dim=5) / counts.clamp_min(1).to(
        dtype=ratios.dtype
    )
    # Put unusable quadrants at +inf so a sorted median can select only valid
    # values.  Mean uses the same explicit mask below.
    active_quadrants = quadrant_usable.sum(dim=4)
    view_usable = active_quadrants >= min_quadrants
    if reduction == "mean":
        view_values = (
            quadrant_values * quadrant_usable.to(dtype=ratios.dtype)
        ).sum(dim=4) / active_quadrants.clamp_min(1).to(dtype=ratios.dtype)
    else:
        ordered = torch.sort(
            torch.where(
                quadrant_usable,
                quadrant_values,
                torch.full_like(quadrant_values, torch.inf),
            ),
            dim=4,
        ).values
        # lower median avoids a synthetic value for an even number of valid
        # quadrants and is deterministic for a fixed topology.
        median_index = ((active_quadrants - 1) // 2).clamp_min(0)
        view_values = torch.gather(ordered, 4, median_index[..., None]).squeeze(4)
    view_values = torch.where(view_usable, view_values, torch.zeros_like(view_values))
    return view_values, view_usable, active_quadrants


def fixed_candidate_maplet_group_log_ratios(
    *,
    view_log_ratios: torch.Tensor,
    view_usable: torch.Tensor,
    support_view_probabilities: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
    missing_view_ratio: float,
    max_log_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize frozen candidate/view latent variables with explicit null.

    A support view without enough projected held-out maplet evidence receives
    the declared ``missing_view_ratio``.  It cannot silently vanish from the
    candidate denominator, which prevents a wrong pose from improving its
    score simply by projecting difficult evidence off-image or into a held-out
    PnP-fit neighbourhood.
    """

    logs = torch.as_tensor(view_log_ratios)
    usable = torch.as_tensor(view_usable, dtype=torch.bool, device=logs.device)
    view_weights = torch.as_tensor(
        support_view_probabilities, dtype=logs.dtype, device=logs.device
    )
    candidates = torch.as_tensor(candidate_probabilities, dtype=logs.dtype, device=logs.device)
    null = torch.as_tensor(null_probabilities, dtype=logs.dtype, device=logs.device)
    missing = float(missing_view_ratio)
    cap = float(max_log_ratio)
    if (
        logs.ndim != 4
        or usable.shape != logs.shape
        or view_weights.shape != logs.shape[1:]
        or candidates.shape != logs.shape[1:3]
        or null.shape != (logs.shape[1],)
        or not np.isfinite([missing, cap]).all()
        or not 0.0 < missing <= 1.0
        or cap <= 0.0
        or torch.any(view_weights < 0.0)
        or torch.any(candidates < 0.0)
        or torch.any(null < 0.0)
        or not bool(torch.isfinite(logs).all())
        or torch.any(
            torch.abs(view_weights.sum(dim=2) - (candidates > 0.0).to(logs.dtype))
            > 2e-4
        )
        or torch.any(torch.abs(candidates.sum(dim=1) + null - 1.0) > 2e-4)
    ):
        raise ValueError("frozen candidate maplet mixture tensors are invalid")
    ratio = torch.exp(torch.clamp(logs, min=-cap, max=cap))
    ratio = torch.where(usable, ratio, torch.full_like(ratio, missing))
    candidate_ratio = torch.sum(ratio * view_weights[None], dim=3)
    group_ratio = null[None] + torch.sum(candidate_ratio * candidates[None], dim=2)
    return torch.log(group_ratio.clamp_min(torch.finfo(logs.dtype).tiny)), candidate_ratio


def summarize_frozen_group_log_ratios(
    *,
    group_log_ratios: torch.Tensor,
    query_xy: np.ndarray | torch.Tensor,
    image_width: int,
    image_height: int,
) -> dict[str, torch.Tensor]:
    """Return fixed point and 2x2 block-robust summaries for P1 ranking."""

    values = torch.as_tensor(group_log_ratios)
    xy = torch.as_tensor(query_xy, dtype=values.dtype, device=values.device)
    if (
        values.ndim != 2
        or values.shape[1] == 0
        or xy.shape != (values.shape[1], 2)
        or int(image_width) <= 1
        or int(image_height) <= 1
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(xy).all())
    ):
        raise ValueError("frozen group log-ratio summaries are invalid")
    point_count = int(values.shape[1])
    ordered = torch.sort(values, dim=1).values
    worst_count = max(1, int(np.ceil(point_count * 0.25)))
    block_x = (xy[:, 0] >= float(image_width) * 0.5).to(dtype=torch.long)
    block_y = (xy[:, 1] >= float(image_height) * 0.5).to(dtype=torch.long)
    blocks = block_y * 2 + block_x
    block_scores: list[torch.Tensor] = []
    for block in range(4):
        rows = torch.nonzero(blocks == block, as_tuple=False).reshape(-1)
        if len(rows) == 0:
            # A fixed empty block is neutral, rather than being removed by a
            # tested pose.  The caller also records the static block coverage.
            block_scores.append(torch.zeros((values.shape[0],), dtype=values.dtype, device=values.device))
        else:
            block_scores.append(values.index_select(1, rows).mean(dim=1))
    return {
        "mean": values.mean(dim=1),
        "median": values.median(dim=1).values,
        "worst_quartile_mean": ordered[:, :worst_count].mean(dim=1),
        "spatial_median_of_means_2x2": torch.stack(block_scores, dim=1).median(dim=1).values,
    }


def deterministic_support_descriptor_derangement(
    *, image_ids: Sequence[str] | np.ndarray, image_sizes: np.ndarray
) -> dict[str, str]:
    """Map every support image to another same-geometry image deterministically.

    This is an appearance-only paired control: support coordinates, candidate
    IDs, maplet topology, pose projections, and all denominators stay fixed.
    The mapping is restricted to equal-size images so sampling geometry cannot
    become a shortcut.
    """

    ids = np.asarray(image_ids).astype(str).reshape(-1)
    sizes = np.asarray(image_sizes, dtype=np.int64).reshape(-1, 2)
    if (
        len(ids) == 0
        or len(set(ids.tolist())) != len(ids)
        or sizes.shape != (len(ids), 2)
        or np.any(sizes <= 1)
    ):
        raise ValueError("support descriptor derangement inputs are invalid")
    mapping: dict[str, str] = {}
    for size in sorted({(int(width), int(height)) for width, height in sizes.tolist()}):
        rows = np.flatnonzero((sizes[:, 0] == size[0]) & (sizes[:, 1] == size[1]))
        ordered = sorted(ids[rows].tolist())
        if len(ordered) < 2:
            raise ValueError("support descriptor control needs two equal-size real images")
        # Rotation by one is a derangement independent of target/pose data.
        for index, image_id in enumerate(ordered):
            mapping[str(image_id)] = str(ordered[(index + 1) % len(ordered)])
    if len(mapping) != len(ids) or any(source == target for source, target in mapping.items()):
        raise RuntimeError("support descriptor derangement has a fixed point")
    return mapping
