"""Target-free direct appearance evidence for frozen landmark candidates.

This module intentionally only compares an anchor-centred query descriptor
patch with the corresponding patch from one fixed support observation.  It
does not project a landmark through a pose, form a spatial density, select a
candidate, or use a target.  Those separations make it suitable for testing
whether the available real-image feature spaces contain absolute candidate
identity information before fitting any learned likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


FROZEN_MULTISCALE_CANDIDATE_APPEARANCE_VERSION = (
    "frozen_multiscale_candidate_absolute_appearance_v1"
)


@dataclass(frozen=True)
class DirectPatchAppearance:
    """Masked aligned-NCC score and its explicit evidence coverage."""

    score: torch.Tensor
    overlap_fraction: torch.Tensor
    support_fraction: torch.Tensor
    usable: torch.Tensor

    def __post_init__(self) -> None:
        score = torch.as_tensor(self.score)
        overlap = torch.as_tensor(self.overlap_fraction)
        support = torch.as_tensor(self.support_fraction)
        usable = torch.as_tensor(self.usable, dtype=torch.bool)
        if (
            score.ndim != 1
            or overlap.shape != score.shape
            or support.shape != score.shape
            or usable.shape != score.shape
            or not bool(torch.isfinite(overlap).all())
            or not bool(torch.isfinite(support).all())
            or bool(torch.any(overlap < 0.0))
            or bool(torch.any(overlap > 1.0))
            or bool(torch.any(support < 0.0))
            or bool(torch.any(support > 1.0))
            or not bool(torch.isfinite(score[usable]).all())
        ):
            raise ValueError("direct patch appearance tensors are invalid")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "overlap_fraction", overlap)
        object.__setattr__(self, "support_fraction", support)
        object.__setattr__(self, "usable", usable)


def aligned_patch_ncc(
    *,
    query_patches: torch.Tensor,
    support_patches: torch.Tensor,
    query_valid: torch.Tensor,
    support_valid: torch.Tensor,
    minimum_support_fraction: float,
    minimum_overlap_fraction: float,
) -> DirectPatchAppearance:
    """Score aligned, L2-normalized descriptor patches without a pose path.

    ``ImageGridFeatureSource.context_patches_torch`` supplies patches whose
    descriptor channel is L2-normalized at every sampled spatial cell.  The
    mean per-cell dot product below is therefore an aligned normalized
    cross-correlation over only the real image cells shared by query and
    support.  A low-coverage crop is represented as unavailable, not padded
    with a synthetic score or a geometry-dependent fallback.
    """

    query = torch.as_tensor(query_patches)
    support = torch.as_tensor(support_patches, dtype=query.dtype, device=query.device)
    query_mask = torch.as_tensor(query_valid, dtype=torch.bool, device=query.device)
    support_mask = torch.as_tensor(support_valid, dtype=torch.bool, device=query.device)
    minimum_support = float(minimum_support_fraction)
    minimum_overlap = float(minimum_overlap_fraction)
    if (
        query.ndim != 4
        or support.shape != query.shape
        or query.shape[0] == 0
        or query.shape[1] <= 0
        or query.shape[2] != query.shape[3]
        or query_mask.shape != (query.shape[0], query.shape[2], query.shape[3])
        or support_mask.shape != query_mask.shape
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(support).all())
        or not (0.0 < minimum_support <= 1.0)
        or not (0.0 < minimum_overlap <= 1.0)
    ):
        raise ValueError("aligned patch NCC inputs are invalid")

    overlap_mask = query_mask & support_mask
    cell_count = float(query.shape[2] * query.shape[3])
    overlap_count = overlap_mask.sum(dim=(1, 2))
    support_count = support_mask.sum(dim=(1, 2))
    overlap_fraction = overlap_count.to(dtype=query.dtype) / cell_count
    support_fraction = support_count.to(dtype=query.dtype) / cell_count
    usable = (
        (support_fraction >= minimum_support)
        & (overlap_fraction >= minimum_overlap)
    )
    per_cell_cosine = torch.sum(query * support, dim=1)
    score = (per_cell_cosine * overlap_mask.to(dtype=query.dtype)).sum(dim=(1, 2))
    score = score / overlap_count.clamp_min(1).to(dtype=query.dtype)
    score = torch.where(usable, score, torch.full_like(score, torch.nan))
    return DirectPatchAppearance(
        score=score,
        overlap_fraction=overlap_fraction,
        support_fraction=support_fraction,
        usable=usable,
    )


def coverage_weighted_view_appearance(
    *,
    view_scores: torch.Tensor,
    view_usable: torch.Tensor,
    view_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marginalize fixed support views without averaging their descriptors.

    The return values are ``(coverage_weighted_mean, max_score,
    usable_weight_mass)``.  They are raw features, not probabilities.  A
    candidate with no usable view has NaN scores and zero usable mass so a
    later calibrated model can learn an explicit unknown state instead of
    treating unavailable visual evidence as negative evidence.
    """

    scores = torch.as_tensor(view_scores)
    usable = torch.as_tensor(view_usable, dtype=torch.bool, device=scores.device)
    weights = torch.as_tensor(view_weights, dtype=scores.dtype, device=scores.device)
    if (
        scores.ndim != 3
        or usable.shape != scores.shape
        or weights.shape != scores.shape[:2]
        or scores.shape[0] == 0
        or scores.shape[2] == 0
        or not bool(torch.isfinite(weights).all())
        or bool(torch.any(weights < 0.0))
        or not bool(torch.isfinite(scores[usable]).all())
    ):
        raise ValueError("view appearance aggregation inputs are invalid")
    masked_weights = weights[..., None] * usable.to(dtype=scores.dtype)
    usable_mass = masked_weights.sum(dim=1)
    weighted = torch.where(usable, scores, torch.zeros_like(scores))
    mean = (weighted * weights[..., None]).sum(dim=1) / usable_mass.clamp_min(1e-12)
    minimum = torch.finfo(scores.dtype).min
    maximum = torch.where(usable, scores, torch.full_like(scores, minimum)).max(dim=1).values
    available = usable_mass > 0.0
    mean = torch.where(available, mean, torch.full_like(mean, torch.nan))
    maximum = torch.where(available, maximum, torch.full_like(maximum, torch.nan))
    return mean, maximum, usable_mass
