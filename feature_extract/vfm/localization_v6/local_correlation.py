"""Local maplet-render/query correlation distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.localization_v6.atlas_renderer import (
    RenderedMapletAtlases,
)


@dataclass(frozen=True)
class CorrelationDistribution:
    pixel_xy: np.ndarray
    xyz: np.ndarray
    maplet_ids: np.ndarray
    offsets_xy: np.ndarray
    probabilities: np.ndarray
    null_probability: np.ndarray
    mean_displacement: np.ndarray
    covariance: np.ndarray
    entropy: np.ndarray
    matchability: np.ndarray
    surface_ids: np.ndarray | None = None
    valid_offset_count: np.ndarray | None = None


@dataclass(frozen=True)
class LocalCorrelationQueryCache:
    """Reusable query tensors for many rendered-pose comparisons."""

    feature_shape: tuple[int, int, int]
    radius: int
    background_samples: int
    query: torch.Tensor
    patches: torch.Tensor
    background_query: torch.Tensor


def build_local_correlation_query_cache(
    query_feature: np.ndarray,
    *,
    radius: int,
    background_samples: int = 512,
    device: str = "cuda",
) -> LocalCorrelationQueryCache:
    """Precompute the pose-invariant RADIO query neighbourhood tensor."""

    feature = np.asarray(query_feature, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError("query feature must have shape [C,H,W]")
    search_radius = int(radius)
    if search_radius < 0:
        raise ValueError("radius must be non-negative")
    torch_device = torch.device(
        device
        if torch.cuda.is_available() or not str(device).startswith("cuda")
        else "cpu"
    )
    query = F.normalize(
        torch.as_tensor(feature, dtype=torch.float32, device=torch_device),
        dim=0,
    )
    kernel = 2 * search_radius + 1
    patches = F.unfold(
        query[None], kernel_size=kernel, padding=search_radius
    )[0].reshape(feature.shape[0], kernel * kernel, -1)
    query_flat = query.reshape(feature.shape[0], -1)
    background_count = min(
        max(int(background_samples), 1), int(query_flat.shape[1])
    )
    background_indices = torch.linspace(
        0,
        int(query_flat.shape[1]) - 1,
        background_count,
        dtype=torch.float64,
        device=torch_device,
    ).round().to(torch.long)
    return LocalCorrelationQueryCache(
        feature_shape=tuple(int(value) for value in feature.shape),
        radius=search_radius,
        background_samples=background_count,
        query=query,
        patches=patches,
        background_query=query_flat[:, background_indices],
    )


def local_correlation_distribution(
    rendered: RenderedMapletAtlases,
    query_feature: np.ndarray,
    *,
    radius: int,
    temperature: float = 0.07,
    query_matchability: np.ndarray | None = None,
    null_logit: float = 0.0,
    uncertainty_temperature_scale: float = 2.0,
    uncertainty_null_scale: float = 2.0,
    maximum_points: int = 8192,
    background_samples: int = 512,
    device: str = "cuda",
    query_cache: LocalCorrelationQueryCache | None = None,
    displacement_prior_sigma_cells: float | None = None,
) -> CorrelationDistribution:
    """Keep the complete local displacement posterior, including null."""

    feature = np.asarray(query_feature, dtype=np.float32)
    if feature.shape != rendered.feature.shape:
        raise ValueError("query and rendered feature maps must have identical shape")
    search_radius = int(radius)
    if search_radius < 0:
        raise ValueError("radius must be non-negative")
    rows_y, rows_x = np.nonzero(rendered.mask)
    if rows_y.size > int(maximum_points) > 0:
        # Deterministic spatially uniform subsampling.
        selected = np.linspace(
            0, rows_y.size - 1, int(maximum_points), dtype=np.int64
        )
        rows_y, rows_x = rows_y[selected], rows_x[selected]
    point_count = int(rows_y.size)
    offsets = np.asarray(
        [
            (float(dx), float(dy))
            for dy in range(-search_radius, search_radius + 1)
            for dx in range(-search_radius, search_radius + 1)
        ],
        dtype=np.float32,
    )
    if point_count == 0:
        return CorrelationDistribution(
            pixel_xy=np.zeros((0, 2), dtype=np.float32),
            xyz=np.zeros((0, 3), dtype=np.float32),
            maplet_ids=np.zeros((0,), dtype=np.int64),
            offsets_xy=offsets,
            probabilities=np.zeros((0, offsets.shape[0]), dtype=np.float32),
            null_probability=np.zeros((0,), dtype=np.float32),
            mean_displacement=np.zeros((0, 2), dtype=np.float32),
            covariance=np.zeros((0, 2, 2), dtype=np.float32),
            entropy=np.zeros((0,), dtype=np.float32),
            matchability=np.zeros((0,), dtype=np.float32),
            surface_ids=np.zeros((0,), dtype=np.int64),
            valid_offset_count=np.zeros((0,), dtype=np.int64),
        )
    if query_cache is None:
        cache = build_local_correlation_query_cache(
            feature,
            radius=search_radius,
            background_samples=background_samples,
            device=device,
        )
    else:
        cache = query_cache
        expected_background = min(
            max(int(background_samples), 1),
            int(feature.shape[1] * feature.shape[2]),
        )
        if (
            tuple(cache.feature_shape) != tuple(feature.shape)
            or int(cache.radius) != search_radius
            or int(cache.background_samples) != expected_background
        ):
            raise ValueError(
                "local-correlation query cache configuration differs"
            )
    query = cache.query
    torch_device = query.device
    render_values = F.normalize(
        torch.as_tensor(
            rendered.feature[:, rows_y, rows_x].T,
            dtype=torch.float32,
            device=torch_device,
        ),
        dim=1,
    )
    kernel = 2 * search_radius + 1
    patches = cache.patches
    linear = torch.as_tensor(
        rows_y * feature.shape[2] + rows_x,
        dtype=torch.long,
        device=torch_device,
    )
    candidates = patches[:, :, linear].permute(2, 1, 0)
    candidates = F.normalize(candidates, dim=2)
    background_count = int(cache.background_samples)
    background_query = cache.background_query
    atlas_uncertainty = torch.as_tensor(
        np.asarray(rendered.uncertainty[rows_y, rows_x], dtype=np.float32),
        device=torch_device,
    ).clamp(0.0, 1.0)
    effective_temperature = max(float(temperature), 1e-4) * (
        1.0 + float(uncertainty_temperature_scale) * atlas_uncertainty
    )
    if rendered.mode_feature is None:
        logits = torch.sum(render_values[:, None] * candidates, dim=2) / (
            effective_temperature[:, None]
        )
        background_logits = (
            render_values @ background_query
        ) / effective_temperature[:, None]
        background_log_partition = torch.logsumexp(
            background_logits, dim=1
        ) - np.log(float(background_count))
    else:
        if rendered.mode_log_prior is None:
            raise ValueError("rendered modes require mode_log_prior")
        mode_values = torch.as_tensor(
            rendered.mode_feature[:, :, rows_y, rows_x].transpose(2, 0, 1),
            dtype=torch.float32,
            device=torch_device,
        )
        mode_values = F.normalize(mode_values, dim=2)
        mode_prior = torch.as_tensor(
            rendered.mode_log_prior[:, rows_y, rows_x].T,
            dtype=torch.float32,
            device=torch_device,
        )
        mode_similarity = torch.einsum(
            "nkc,noc->nko", mode_values, candidates
        ) / effective_temperature[:, None, None]
        # Do not choose an appearance mode before observing the query.  This
        # is p(delta|q) ∝ Σ_k p(k|view) p(q_delta|mode_k).
        logits = torch.logsumexp(
            mode_similarity + mode_prior[:, :, None], dim=1
        )
        background_mode_similarity = torch.einsum(
            "nkc,cs->nks", mode_values, background_query
        ) / effective_temperature[:, None, None]
        background_logits = torch.logsumexp(
            background_mode_similarity + mode_prior[:, :, None],
            dim=1,
        )
        background_log_partition = torch.logsumexp(
            background_logits, dim=1
        ) - np.log(float(background_count))
    offset_tensor = torch.as_tensor(offsets, device=torch_device)
    point_x = torch.as_tensor(rows_x, device=torch_device)[:, None]
    point_y = torch.as_tensor(rows_y, device=torch_device)[:, None]
    candidate_x = point_x + offset_tensor[None, :, 0]
    candidate_y = point_y + offset_tensor[None, :, 1]
    valid = (
        (candidate_x >= 0)
        & (candidate_x < feature.shape[2])
        & (candidate_y >= 0)
        & (candidate_y < feature.shape[1])
    )
    valid_count = torch.clamp(valid.sum(dim=1), min=1).to(logits.dtype)
    # Convert cosine scores into a query-adaptive match/background density
    # ratio. RADIO projections occupy a strongly anisotropic cone (unrelated
    # descriptors can have cosine around 0.6), so the old random-unit-vector
    # partition made virtually every point non-null. Under background, the
    # local mean likelihood ratio is now one regardless of descriptor bias or
    # search-window size; only above-background correlation favours a match.
    logits = (
        logits
        - torch.log(valid_count)[:, None]
        - background_log_partition[:, None]
    )
    if displacement_prior_sigma_cells is not None:
        sigma = float(displacement_prior_sigma_cells)
        if sigma <= 0.0:
            raise ValueError("displacement_prior_sigma_cells must be positive")
        spatial_log_prior = -0.5 * torch.sum(offset_tensor * offset_tensor, dim=1) / (sigma * sigma)
        logits = logits + spatial_log_prior[None]
    logits = torch.where(valid, logits, torch.full_like(logits, -1e4))
    if query_matchability is None:
        cell_matchability = torch.ones(
            (point_count,), dtype=logits.dtype, device=torch_device
        )
    else:
        match_map = np.asarray(query_matchability, dtype=np.float32)
        if match_map.shape != rendered.mask.shape:
            raise ValueError("query_matchability shape differs")
        match_tensor = torch.as_tensor(
            match_map[None, None], dtype=torch.float32, device=torch_device
        )
        match_patches = F.unfold(
            match_tensor, kernel_size=kernel, padding=search_radius
        )[0]
        candidate_matchability = match_patches[:, linear].T.clamp(1e-4, 1.0)
        # ALIKE is detector-only evidence, not a descriptor likelihood.  It
        # may decide which rendered cells are sufficiently distinctive to
        # trust, but it must not choose one displacement inside a RADIO
        # correlation window.  Candidate-wise addition here previously let a
        # detector peak create optical flow even when every RADIO score was
        # identical.  Collapse the valid search patch to one reliability per
        # rendered cell, matching the chart-volume probability semantics.
        candidate_matchability = torch.where(
            valid,
            candidate_matchability,
            torch.zeros_like(candidate_matchability),
        )
        cell_matchability = torch.amax(candidate_matchability, dim=1)
    # Pairwise null cannot be predicted from the query image alone: it depends
    # on the rendered map feature and its complete candidate correlations.
    # Query-only unmatchability already enters every offset above.  Keep a
    # separate calibrated pair-null prior instead of reusing a spatial head.
    null_logits = torch.full(
        (point_count, 1),
        float(null_logit),
        dtype=torch.float32,
        device=torch_device,
    )
    null_logits = null_logits + (
        float(uncertainty_null_scale) * atlas_uncertainty[:, None]
    )
    joint = torch.softmax(torch.cat([logits, null_logits], dim=1), dim=1)
    probability = joint[:, :-1]
    null_probability = joint[:, -1]
    visible_mass = torch.clamp(probability.sum(dim=1), min=1e-8)
    conditional = probability / visible_mass[:, None]
    mean = conditional @ offset_tensor
    residual = offset_tensor[None] - mean[:, None]
    covariance = torch.einsum(
        "nk,nki,nkj->nij", conditional, residual, residual
    )
    entropy = -torch.sum(
        joint * torch.log(torch.clamp(joint, min=1e-8)), dim=1
    )
    return CorrelationDistribution(
        pixel_xy=np.stack([rows_x, rows_y], axis=1).astype(np.float32),
        xyz=np.asarray(rendered.xyz[rows_y, rows_x], dtype=np.float32),
        maplet_ids=np.asarray(rendered.maplet_id[rows_y, rows_x], dtype=np.int64),
        offsets_xy=offsets,
        probabilities=probability.detach().cpu().numpy().astype(np.float32),
        null_probability=null_probability.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        mean_displacement=mean.detach().cpu().numpy().astype(np.float32),
        covariance=covariance.detach().cpu().numpy().astype(np.float32),
        entropy=entropy.detach().cpu().numpy().astype(np.float32),
        matchability=cell_matchability.detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        surface_ids=(
            np.asarray(rendered.surface_id[rows_y, rows_x], dtype=np.int64)
            if rendered.surface_id is not None
            else None
        ),
        valid_offset_count=(
            valid_count.detach().cpu().numpy().astype(np.int64)
        ),
    )
