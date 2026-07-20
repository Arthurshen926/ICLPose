"""Pose-free landmark-centred region-layout appearance evidence.

The direct aligned-NCC probe compares corresponding descriptor cells.  This
companion probe deliberately relaxes that requirement: it pools a larger real
image crop into fixed relative regions before comparing query and support.
It retains candidate and support-view identity, but removes a small
cross-view cell misregistration as a confounder.  No pose, target, candidate
reselection, or descriptor averaging is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


FROZEN_MULTISCALE_CANDIDATE_REGION_LAYOUT_VERSION = (
    "frozen_multiscale_candidate_region_layout_v1"
)


@dataclass(frozen=True)
class RegionLayoutAppearance:
    """Per-view raw large-context similarities and their availability masks."""

    scores: torch.Tensor
    usable: torch.Tensor
    region_pair_count: torch.Tensor

    def __post_init__(self) -> None:
        scores = torch.as_tensor(self.scores)
        usable = torch.as_tensor(self.usable, dtype=torch.bool, device=scores.device)
        count = torch.as_tensor(
            self.region_pair_count, dtype=torch.int64, device=scores.device
        )
        if (
            scores.ndim != 2
            or scores.shape[0] == 0
            or scores.shape[1] != 4
            or usable.shape != scores.shape
            or count.shape != (scores.shape[0],)
            or bool(torch.any(count < 0))
            or not bool(torch.isfinite(scores[usable]).all())
        ):
            raise ValueError("region-layout appearance tensors are invalid")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "usable", usable)
        object.__setattr__(self, "region_pair_count", count)


REGION_LAYOUT_STATISTIC_NAMES = (
    "global_pool_cosine",
    "layout_mean_cosine",
    "layout_min_cosine",
    "global_layout_mean_cosine",
)


def _masked_pool(
    *, patches: torch.Tensor, mask: torch.Tensor, row_indices: torch.Tensor, column_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = patches[:, :, row_indices[:, None], column_indices[None, :]]
    selected_mask = mask[:, row_indices[:, None], column_indices[None, :]]
    count = selected_mask.sum(dim=(1, 2))
    pooled = (values * selected_mask[:, None].to(dtype=values.dtype)).sum(dim=(2, 3))
    pooled = pooled / count.clamp_min(1).to(dtype=values.dtype)[:, None]
    return F.normalize(pooled, p=2, dim=1, eps=1e-8), count > 0


def region_layout_similarity(
    *,
    query_patches: torch.Tensor,
    support_patches: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    region_grid_size: int = 3,
) -> RegionLayoutAppearance:
    """Compare global and 2-D region-pooled crops at fixed anchors.

    The four scores are global crop cosine, mean corresponding-region cosine,
    minimum corresponding-region cosine, and a fixed 50/50 global/layout
    blend.  Invalid crop cells are excluded rather than padded.  A partially
    visible crop remains an explicit partial observation when it has at least
    one common region; callers can inspect ``region_pair_count`` rather than
    confusing that condition with a negative visual score.
    """

    query = torch.as_tensor(query_patches)
    support = torch.as_tensor(support_patches, dtype=query.dtype, device=query.device)
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    support_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    region_count = int(region_grid_size)
    if (
        query.ndim != 4
        or support.shape != query.shape
        or query.shape[0] == 0
        or query.shape[1] <= 0
        or query.shape[2] != query.shape[3]
        or query_mask.shape != (query.shape[0], query.shape[2], query.shape[3])
        or support_mask.shape != query_mask.shape
        or region_count <= 0
        or region_count > query.shape[2]
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(support).all())
    ):
        raise ValueError("region-layout similarity inputs are invalid")
    window = int(query.shape[2])
    all_indices = torch.arange(window, dtype=torch.long, device=query.device)
    query_global, query_global_valid = _masked_pool(
        patches=query,
        mask=query_mask,
        row_indices=all_indices,
        column_indices=all_indices,
    )
    support_global, support_global_valid = _masked_pool(
        patches=support,
        mask=support_mask,
        row_indices=all_indices,
        column_indices=all_indices,
    )
    global_valid = query_global_valid & support_global_valid
    global_score = torch.sum(query_global * support_global, dim=1)

    query_regions: list[torch.Tensor] = []
    support_regions: list[torch.Tensor] = []
    region_valid: list[torch.Tensor] = []
    partitions = torch.tensor_split(all_indices, region_count)
    for rows in partitions:
        for columns in partitions:
            query_region, query_present = _masked_pool(
                patches=query,
                mask=query_mask,
                row_indices=rows,
                column_indices=columns,
            )
            support_region, support_present = _masked_pool(
                patches=support,
                mask=support_mask,
                row_indices=rows,
                column_indices=columns,
            )
            query_regions.append(query_region)
            support_regions.append(support_region)
            region_valid.append(query_present & support_present)
    query_region_tensor = torch.stack(query_regions, dim=1)
    support_region_tensor = torch.stack(support_regions, dim=1)
    region_valid_tensor = torch.stack(region_valid, dim=1)
    region_scores = torch.sum(query_region_tensor * support_region_tensor, dim=2)
    paired_count = region_valid_tensor.sum(dim=1)
    layout_present = paired_count > 0
    layout_mean = (
        region_scores * region_valid_tensor.to(dtype=query.dtype)
    ).sum(dim=1) / paired_count.clamp_min(1).to(dtype=query.dtype)
    layout_min = torch.where(
        region_valid_tensor,
        region_scores,
        torch.full_like(region_scores, torch.inf),
    ).min(dim=1).values
    layout_mean = torch.where(
        layout_present, layout_mean, torch.full_like(layout_mean, torch.nan)
    )
    layout_min = torch.where(
        layout_present, layout_min, torch.full_like(layout_min, torch.nan)
    )
    combined_present = global_valid & layout_present
    combined = 0.5 * global_score + 0.5 * layout_mean
    scores = torch.stack((global_score, layout_mean, layout_min, combined), dim=1)
    usable = torch.stack(
        (global_valid, layout_present, layout_present, combined_present), dim=1
    )
    scores = torch.where(usable, scores, torch.full_like(scores, torch.nan))
    return RegionLayoutAppearance(
        scores=scores,
        usable=usable,
        region_pair_count=paired_count,
    )
