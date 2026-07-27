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


def local_correlation_distribution(
    rendered: RenderedMapletAtlases,
    query_feature: np.ndarray,
    *,
    radius: int,
    temperature: float = 0.07,
    query_matchability: np.ndarray | None = None,
    query_null_probability: np.ndarray | None = None,
    null_logit: float = 0.0,
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
    logits = torch.sum(render_values[:, None] * candidates, dim=2) / max(
        float(temperature), 1e-4
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
    logits = torch.where(valid, logits, torch.full_like(logits, -1e4))
    if query_matchability is None:
        matchability = torch.ones(
            (point_count,), dtype=torch.float32, device=torch_device
        )
    else:
        match_map = np.asarray(query_matchability, dtype=np.float32)
        if match_map.shape != rendered.mask.shape:
            raise ValueError("query_matchability shape differs")
        matchability = torch.as_tensor(
            match_map[rows_y, rows_x], device=torch_device
        ).clamp(1e-4, 1.0)
        logits = logits + torch.log(matchability[:, None])
    if query_null_probability is None:
        null_logits = torch.full(
            (point_count, 1),
            float(null_logit),
            dtype=torch.float32,
            device=torch_device,
        )
    else:
        null_map = np.asarray(query_null_probability, dtype=np.float32)
        if null_map.shape != rendered.mask.shape:
            raise ValueError("query_null_probability shape differs")
        null_values = torch.as_tensor(
            null_map[rows_y, rows_x], device=torch_device
        ).clamp(1e-5, 1.0 - 1e-5)
        null_logits = torch.logit(null_values)[:, None] + float(null_logit)
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
        matchability=matchability.detach().cpu().numpy().astype(np.float32),
    )
