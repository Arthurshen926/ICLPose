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
    device: str = "cuda",
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
        )
    torch_device = torch.device(
        device
        if torch.cuda.is_available() or not str(device).startswith("cuda")
        else "cpu"
    )
    query = F.normalize(
        torch.as_tensor(feature, dtype=torch.float32, device=torch_device),
        dim=0,
    )
    render_values = F.normalize(
        torch.as_tensor(
            rendered.feature[:, rows_y, rows_x].T,
            dtype=torch.float32,
            device=torch_device,
        ),
        dim=1,
    )
    kernel = 2 * search_radius + 1
    patches = F.unfold(
        query[None], kernel_size=kernel, padding=search_radius
    )
    patches = patches[0].reshape(feature.shape[0], kernel * kernel, -1)
    linear = torch.as_tensor(
        rows_y * feature.shape[2] + rows_x,
        dtype=torch.long,
        device=torch_device,
    )
    candidates = patches[:, :, linear].permute(2, 1, 0)
    candidates = F.normalize(candidates, dim=2)
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
    # Convert cosine scores into an approximate match/non-match density ratio.
    # Without the offset prior and the random-unit-vector partition term, the
    # maximum of many unrelated candidates almost always overwhelms null.
    negative_log_partition = 0.5 / (
        feature.shape[0] * effective_temperature * effective_temperature
    )
    logits = (
        logits
        - torch.log(valid_count)[:, None]
        - negative_log_partition[:, None]
    )
    logits = torch.where(valid, logits, torch.full_like(logits, -1e4))
    if query_matchability is None:
        candidate_matchability = torch.ones_like(logits)
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
        # Matchability belongs to the candidate query location.  A value
        # sampled only at the rendered centre is constant over offsets and
        # therefore cancels from the displacement posterior.
        logits = logits + torch.log(candidate_matchability)
    candidate_matchability = torch.where(
        valid, candidate_matchability, torch.zeros_like(candidate_matchability)
    )
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
    matchability = torch.sum(conditional * candidate_matchability, dim=1)
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
        matchability=matchability.detach().cpu().numpy().astype(np.float32),
        surface_ids=(
            np.asarray(rendered.surface_id[rows_y, rows_x], dtype=np.int64)
            if rendered.surface_id is not None
            else None
        ),
    )
