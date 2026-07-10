from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RGBPatchMeasurementPrediction:
    logits: torch.Tensor
    offsets_xy: torch.Tensor
    dustbin_logit: torch.Tensor
    coarse_logits: torch.Tensor | None = None
    coarse_offsets_xy: torch.Tensor | None = None
    coarse_mode_offset_xy: torch.Tensor | None = None
    local_log_probs: torch.Tensor | None = None
    mean_offset_xy: torch.Tensor | None = None
    cov_2x2: torch.Tensor | None = None
    mode_offset_xy: torch.Tensor | None = None
    dustbin_probability: torch.Tensor | None = None
    target_is_dustbin: torch.Tensor | None = None
    epe_px: torch.Tensor | None = None
    direct_mean_offset_xy: torch.Tensor | None = None
    direct_log_sigma_xy: torch.Tensor | None = None
    gated_mean_offset_xy: torch.Tensor | None = None
    gate_logit: torch.Tensor | None = None
    gate_probability: torch.Tensor | None = None


def local_offset_grid(*, search_radius_px: float, step_px: float, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    radius = float(search_radius_px)
    step = float(step_px)
    if radius < 0.0 or step <= 0.0:
        raise ValueError("search_radius_px must be non-negative and step_px must be positive")
    values = torch.arange(-radius, radius + 0.5 * step, step, device=device, dtype=dtype)
    dx, dy = torch.meshgrid(values, values, indexing="xy")
    return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)


def _normalise_xy(xy: torch.Tensor, *, image_width: int, image_height: int) -> torch.Tensor:
    x = xy[..., 0]
    y = xy[..., 1]
    if int(image_width) > 1:
        x_norm = 2.0 * x / float(int(image_width) - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x)
    if int(image_height) > 1:
        y_norm = 2.0 * y / float(int(image_height) - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y)
    return torch.stack([x_norm, y_norm], dim=-1)


def crop_rgb_window(
    images: torch.Tensor,
    centers_xy: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    centers = centers_xy.to(device=images.device, dtype=images.dtype).reshape(int(images.shape[0]), 2)
    offsets = local_offset_grid(search_radius_px=float(radius_px), step_px=float(step_px), device=images.device, dtype=images.dtype)
    side = int(round((2.0 * float(radius_px)) / float(step_px))) + 1
    xy = centers[:, None, :] + offsets[None, :, :]
    grid = _normalise_xy(xy.reshape(int(images.shape[0]), side, side, 2), image_width=int(image_width), image_height=int(image_height))
    patch = F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return patch, offsets.detach().cpu()


def crop_rgb_windows_by_owner(
    images: torch.Tensor,
    owner_indices: torch.Tensor,
    centers_xy: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop many windows while keeping one full RGB tensor per unique owner."""

    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    owners = owner_indices.to(device=images.device, dtype=torch.long).reshape(-1)
    centers = centers_xy.to(device=images.device, dtype=images.dtype).reshape(-1, 2)
    if int(owners.numel()) != int(centers.shape[0]):
        raise ValueError("owner_indices and centers_xy must contain the same number of rows")
    if bool(torch.any(owners < 0)) or bool(torch.any(owners >= int(images.shape[0]))):
        raise ValueError("owner_indices contains an out-of-range image owner")
    offsets = local_offset_grid(
        search_radius_px=float(radius_px),
        step_px=float(step_px),
        device=images.device,
        dtype=images.dtype,
    )
    side = int(round((2.0 * float(radius_px)) / float(step_px))) + 1
    if int(owners.numel()) == 0:
        return (
            torch.zeros((0, int(images.shape[1]), side, side), dtype=torch.float32, device=images.device),
            offsets.detach().cpu(),
        )
    grouped_patches = []
    grouped_rows = []
    for owner_id in torch.unique(owners, sorted=True).tolist():
        rows = torch.nonzero(owners == int(owner_id), as_tuple=False).reshape(-1)
        owner_centers = centers[rows]
        xy = owner_centers[:, None, :] + offsets[None, :, :]
        packed_grid = _normalise_xy(
            xy.reshape(1, int(rows.numel()) * side, side, 2),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        packed = F.grid_sample(
            images[int(owner_id) : int(owner_id) + 1].float(),
            packed_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        patches = packed[0].reshape(int(images.shape[1]), int(rows.numel()), side, side).permute(1, 0, 2, 3)
        grouped_patches.append(patches)
        grouped_rows.append(rows)
    row_order = torch.cat(grouped_rows, dim=0)
    patches_by_group = torch.cat(grouped_patches, dim=0)
    return patches_by_group[torch.argsort(row_order)].contiguous(), offsets.detach().cpu()


def crop_rgb_window_with_source_from_output_affine(
    images: torch.Tensor,
    centers_xy: torch.Tensor,
    source_from_output_2x2: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    batch = int(images.shape[0])
    centers = centers_xy.to(device=images.device, dtype=images.dtype).reshape(batch, 2)
    affine = source_from_output_2x2.to(device=images.device, dtype=images.dtype).reshape(batch, 2, 2)
    offsets = local_offset_grid(search_radius_px=float(radius_px), step_px=float(step_px), device=images.device, dtype=images.dtype)
    side = int(round((2.0 * float(radius_px)) / float(step_px))) + 1
    source_offsets = torch.einsum("bij,kj->bki", affine, offsets)
    xy = centers[:, None, :] + source_offsets
    grid = _normalise_xy(xy.reshape(batch, side, side, 2), image_width=int(image_width), image_height=int(image_height))
    patch = F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return patch, offsets.detach().cpu()


def crop_rgb_window_with_source_from_output_homography(
    images: torch.Tensor,
    output_centers_xy: torch.Tensor,
    source_from_output_3x3: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    batch = int(images.shape[0])
    centers = output_centers_xy.to(device=images.device, dtype=images.dtype).reshape(batch, 2)
    homography = source_from_output_3x3.to(device=images.device, dtype=images.dtype).reshape(batch, 3, 3)
    offsets = local_offset_grid(search_radius_px=float(radius_px), step_px=float(step_px), device=images.device, dtype=images.dtype)
    side = int(round((2.0 * float(radius_px)) / float(step_px))) + 1
    output_xy = centers[:, None, :] + offsets[None, :, :]
    ones = torch.ones((batch, int(offsets.shape[0]), 1), device=images.device, dtype=images.dtype)
    output_h = torch.cat([output_xy, ones], dim=2)
    source_h = torch.einsum("bij,bkj->bki", homography, output_h)
    source_xy = source_h[..., :2] / source_h[..., 2:3].clamp_min(1e-8)
    grid = _normalise_xy(source_xy.reshape(batch, side, side, 2), image_width=int(image_width), image_height=int(image_height))
    patch = F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return patch, offsets.detach().cpu()


def _xy_for_batch(offsets_xy: torch.Tensor, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    offsets = offsets_xy.to(device=device, dtype=dtype)
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(int(batch_size), -1, -1)
    if offsets.ndim != 3 or int(offsets.shape[0]) != int(batch_size) or int(offsets.shape[2]) != 2:
        raise ValueError("offsets_xy must have shape (K,2) or (B,K,2)")
    return offsets


def _sample_weight_for_batch(
    sample_weight: torch.Tensor | None,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if sample_weight is None:
        return None
    weight = sample_weight.to(device=device, dtype=dtype).reshape(int(batch_size))
    if int(weight.numel()) != int(batch_size):
        raise ValueError("sample_weight must contain one value per batch row")
    return weight.clamp_min(0.0)


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if int(values.numel()) == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    if sample_weight is None:
        return torch.mean(values)
    weight = sample_weight.to(device=values.device, dtype=values.dtype).reshape(values.shape)
    denom = torch.sum(weight)
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return torch.sum(values * weight) / denom.clamp_min(1e-12)


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = torch.where(target > 0.5, torch.full_like(loss, float(positive_weight)), torch.ones_like(loss))
    return loss * weight


def _continuous_probability_at_target(log_probs: torch.Tensor, offsets_xy: torch.Tensor, target_delta_xy: torch.Tensor) -> torch.Tensor:
    batch = int(log_probs.shape[0])
    offsets = _xy_for_batch(offsets_xy, batch, device=log_probs.device, dtype=log_probs.dtype)
    target = target_delta_xy.to(device=log_probs.device, dtype=log_probs.dtype).reshape(batch, 2)
    probs = torch.exp(log_probs)
    out = []
    for row_idx in range(batch):
        row_xy = offsets[row_idx]
        xs = torch.unique(row_xy[:, 0], sorted=True)
        ys = torch.unique(row_xy[:, 1], sorted=True)
        if int(xs.numel() * ys.numel()) != int(row_xy.shape[0]):
            distances = torch.linalg.norm(row_xy - target[row_idx].reshape(1, 2), dim=1)
            out.append(probs[row_idx, torch.argmin(distances)])
            continue
        grid = probs[row_idx].reshape(int(ys.numel()), int(xs.numel()))
        tx, ty = target[row_idx, 0], target[row_idx, 1]
        x1_idx = torch.searchsorted(xs, tx).clamp(1, int(xs.numel()) - 1)
        y1_idx = torch.searchsorted(ys, ty).clamp(1, int(ys.numel()) - 1)
        x0_idx = x1_idx - 1
        y0_idx = y1_idx - 1
        x0, x1 = xs[x0_idx], xs[x1_idx]
        y0, y1 = ys[y0_idx], ys[y1_idx]
        wx = ((tx - x0) / torch.clamp(x1 - x0, min=1e-8)).clamp(0.0, 1.0)
        wy = ((ty - y0) / torch.clamp(y1 - y0, min=1e-8)).clamp(0.0, 1.0)
        p00 = grid[y0_idx, x0_idx]
        p10 = grid[y0_idx, x1_idx]
        p01 = grid[y1_idx, x0_idx]
        p11 = grid[y1_idx, x1_idx]
        out.append(p00 * (1.0 - wx) * (1.0 - wy) + p10 * wx * (1.0 - wy) + p01 * (1.0 - wx) * wy + p11 * wx * wy)
    return torch.stack(out).clamp_min(1e-12)


def continuous_offset_nll_with_dustbin(
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    target_delta_xy: torch.Tensor,
    *,
    dustbin_logit: torch.Tensor,
    search_radius_px: float,
    covariance_floor_px2: float = 1e-4,
    epe_weight: float = 0.0,
    dustbin_bce_weight: float = 0.0,
    dustbin_positive_weight: float = 1.0,
    target_is_dustbin: torch.Tensor | None = None,
    target_heatmap_sigma_px: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, RGBPatchMeasurementPrediction]:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B,K)")
    batch = int(logits.shape[0])
    target = target_delta_xy.to(device=logits.device, dtype=logits.dtype).reshape(batch, 2)
    radius = float(search_radius_px)
    geometry_in_window = (torch.abs(target[:, 0]) <= radius) & (torch.abs(target[:, 1]) <= radius)
    if target_is_dustbin is None:
        dustbin_target_bool = ~geometry_in_window
    else:
        dustbin_target_bool = target_is_dustbin.to(device=logits.device).reshape(batch).bool()
    valid_target_bool = ~dustbin_target_bool
    spatial_log_probs = F.log_softmax(logits, dim=1)
    dustbin_logits = dustbin_logit.to(device=logits.device, dtype=logits.dtype).reshape(batch)
    offsets = _xy_for_batch(offsets_xy, batch, device=logits.device, dtype=logits.dtype)
    if float(target_heatmap_sigma_px) > 0.0:
        sigma2 = float(target_heatmap_sigma_px) ** 2
        dist2 = torch.sum((offsets - target[:, None, :]) ** 2, dim=2)
        target_probs = F.softmax(-0.5 * dist2 / max(sigma2, 1e-12), dim=1).detach()
        spatial_nll = -torch.sum(target_probs * spatial_log_probs, dim=1)
    else:
        p_target = _continuous_probability_at_target(spatial_log_probs, offsets_xy, target)
        spatial_nll = -torch.log(p_target)
    weights = _sample_weight_for_batch(sample_weight, batch, device=logits.device, dtype=logits.dtype)
    loss = _weighted_mean(spatial_nll[valid_target_bool], None if weights is None else weights[valid_target_bool])
    dustbin_target = dustbin_target_bool.to(device=logits.device, dtype=logits.dtype)
    validity_bce_rows = _weighted_bce_with_logits(
        dustbin_logits,
        dustbin_target,
        positive_weight=float(dustbin_positive_weight),
    )
    validity_bce = _weighted_mean(validity_bce_rows, weights)
    loss = loss + float(dustbin_bce_weight) * validity_bce
    spatial_probs = torch.exp(spatial_log_probs)
    conditional = spatial_probs / spatial_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean = torch.sum(conditional[..., None] * offsets, dim=1)
    centered = offsets - mean[:, None, :]
    cov = torch.einsum("bk,bki,bkj->bij", conditional, centered, centered)
    cov = cov + torch.eye(2, device=logits.device, dtype=logits.dtype).unsqueeze(0) * float(covariance_floor_px2)
    mode_idx = torch.argmax(spatial_probs, dim=1)
    row_idx = torch.arange(batch, device=logits.device)
    mode = offsets[row_idx, mode_idx]
    epe = torch.linalg.norm(mean - target, dim=1)
    if float(epe_weight) > 0.0:
        epe_values = epe[valid_target_bool] if torch.any(valid_target_bool) else epe
        epe_weights = None if weights is None else (weights[valid_target_bool] if torch.any(valid_target_bool) else weights)
        loss = loss + float(epe_weight) * _weighted_mean(epe_values, epe_weights)
    pred = RGBPatchMeasurementPrediction(
        logits=logits,
        offsets_xy=offsets_xy,
        dustbin_logit=dustbin_logit,
        local_log_probs=spatial_log_probs,
        mean_offset_xy=mean,
        cov_2x2=cov,
        mode_offset_xy=mode,
        dustbin_probability=torch.sigmoid(dustbin_logits),
        target_is_dustbin=dustbin_target_bool,
        epe_px=epe,
    )
    return loss, pred


def likelihood_moments_from_logits(
    logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    *,
    covariance_floor_px2: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log-probabilities, mean, covariance, and mode for a local offset volume."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (B,K)")
    batch = int(logits.shape[0])
    spatial_log_probs = F.log_softmax(logits, dim=1)
    spatial_probs = torch.exp(spatial_log_probs)
    conditional = spatial_probs / spatial_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    offsets = _xy_for_batch(offsets_xy, batch, device=logits.device, dtype=logits.dtype)
    mean = torch.sum(conditional[..., None] * offsets, dim=1)
    centered = offsets - mean[:, None, :]
    cov = torch.einsum("bk,bki,bkj->bij", conditional, centered, centered)
    cov = cov + torch.eye(2, device=logits.device, dtype=logits.dtype).unsqueeze(0) * float(covariance_floor_px2)
    mode_idx = torch.argmax(spatial_probs, dim=1)
    row_idx = torch.arange(batch, device=logits.device)
    mode = offsets[row_idx, mode_idx]
    return spatial_log_probs, mean, cov, mode


def residual_delta_gaussian_nll(
    mean_offset_xy: torch.Tensor,
    log_sigma_xy: torch.Tensor,
    target_delta_xy: torch.Tensor,
    *,
    search_radius_px: float,
    dustbin_logit: torch.Tensor | None = None,
    target_is_dustbin: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
    dustbin_positive_weight: float = 1.0,
) -> tuple[torch.Tensor, RGBPatchMeasurementPrediction]:
    mean = mean_offset_xy.reshape(int(mean_offset_xy.shape[0]), 2)
    log_sigma = log_sigma_xy.reshape(int(mean.shape[0]), 2).clamp(-5.0, 3.0)
    target = target_delta_xy.to(device=mean.device, dtype=mean.dtype).reshape(int(mean.shape[0]), 2)
    radius = float(search_radius_px)
    geometry_in_window = (torch.abs(target[:, 0]) <= radius) & (torch.abs(target[:, 1]) <= radius)
    if target_is_dustbin is None:
        dustbin_target_bool = ~geometry_in_window
    else:
        dustbin_target_bool = target_is_dustbin.to(device=mean.device).reshape(int(mean.shape[0])).bool()
    sigma = torch.exp(log_sigma).clamp_min(1e-4)
    residual = (target - mean) / sigma
    gaussian_nll = 0.5 * torch.sum(residual * residual + 2.0 * log_sigma, dim=1)
    if dustbin_logit is not None:
        dustbin_logits = dustbin_logit.to(device=mean.device, dtype=mean.dtype).reshape(int(mean.shape[0]))
        dustbin_target = dustbin_target_bool.to(device=mean.device, dtype=mean.dtype)
        validity_bce = _weighted_bce_with_logits(
            dustbin_logits,
            dustbin_target,
            positive_weight=float(dustbin_positive_weight),
        )
        loss_rows = torch.where(~dustbin_target_bool, gaussian_nll + validity_bce, validity_bce)
    else:
        loss_rows = gaussian_nll
    weights = _sample_weight_for_batch(sample_weight, int(mean.shape[0]), device=mean.device, dtype=mean.dtype)
    epe = torch.linalg.norm(mean - target, dim=1)
    cov = torch.diag_embed(sigma * sigma)
    pred = RGBPatchMeasurementPrediction(
        logits=torch.empty((int(mean.shape[0]), 0), device=mean.device, dtype=mean.dtype),
        offsets_xy=torch.empty((0, 2), device=mean.device, dtype=mean.dtype),
        dustbin_logit=torch.zeros((int(mean.shape[0]),), device=mean.device, dtype=mean.dtype) if dustbin_logit is None else dustbin_logit,
        mean_offset_xy=mean,
        cov_2x2=cov,
        target_is_dustbin=dustbin_target_bool,
        epe_px=epe,
        direct_mean_offset_xy=mean,
        direct_log_sigma_xy=log_sigma,
    )
    return _weighted_mean(loss_rows, weights), pred


def _grid_side(radius_px: float, step_px: float) -> int:
    return int(round((2.0 * float(radius_px)) / float(step_px))) + 1


def _candidate_offsets_for_feature_grid(
    candidate_offsets_xy: torch.Tensor,
    *,
    batch_size: int,
    search_radius_px: float,
    feature_step_px: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = candidate_offsets_xy.to(device=device, dtype=dtype)
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(int(batch_size), -1, -1)
    if offsets.ndim != 3 or int(offsets.shape[0]) != int(batch_size) or int(offsets.shape[2]) != 2:
        raise ValueError("candidate_offsets_xy must have shape (K,2) or (B,K,2)")
    radius = float(search_radius_px)
    step = float(feature_step_px)
    if step <= 0.0:
        raise ValueError("feature_step_px must be positive")
    grid_xy = (offsets + radius) / step
    rounded = torch.round(grid_xy)
    if torch.max(torch.abs(grid_xy - rounded)).detach().cpu().item() > 1e-4:
        raise ValueError("candidate offsets must lie on the feature grid")
    side = int(round((2.0 * radius) / step)) + 1
    if torch.any(rounded < -1e-4) or torch.any(rounded > float(side - 1) + 1e-4):
        raise ValueError("candidate offsets fall outside the search feature grid")
    indices = (rounded[..., 1].long() * int(side) + rounded[..., 0].long()).reshape(int(batch_size), -1)
    return offsets, indices


def _template_search_cost_volume_logits_for_offsets(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    candidate_offsets_xy: torch.Tensor,
    search_radius_px: float,
    context_radius_px: float,
    feature_step_px: float,
    temperature: torch.Tensor | float = 10.0,
    template_scale_factors: Sequence[float] = (1.0,),
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_features.ndim != 4 or render_features.ndim != 4:
        raise ValueError("query_features and render_features must have shape (B,C,H,W)")
    if int(query_features.shape[0]) != int(render_features.shape[0]) or int(query_features.shape[1]) != int(render_features.shape[1]):
        raise ValueError("query/render feature batch and channel dimensions must match")
    batch = int(query_features.shape[0])
    feature_step = float(feature_step_px)
    if feature_step <= 0.0:
        raise ValueError("feature_step_px must be positive")
    search_steps = int(round(float(search_radius_px) / feature_step))
    context_steps = int(round(float(context_radius_px) / feature_step))
    crop_steps = int(round((float(search_radius_px) + float(context_radius_px)) / feature_step))
    expected_side = 2 * crop_steps + 1
    if int(query_features.shape[2]) != expected_side or int(query_features.shape[3]) != expected_side:
        raise ValueError("query_features spatial shape does not match search/context radius and feature step")
    if int(render_features.shape[2]) != expected_side or int(render_features.shape[3]) != expected_side:
        raise ValueError("render_features spatial shape does not match search/context radius and feature step")
    template_side = 2 * context_steps + 1
    if template_side <= 0:
        raise ValueError("context_radius_px is too small for the feature step")
    offsets = _xy_for_batch(
        candidate_offsets_xy,
        batch_size=batch,
        device=query_features.device,
        dtype=query_features.dtype,
    )
    if torch.any(torch.abs(offsets[..., 0]) > float(search_radius_px) + 1e-4) or torch.any(
        torch.abs(offsets[..., 1]) > float(search_radius_px) + 1e-4
    ):
        raise ValueError("candidate offsets fall outside the search radius")
    scale = torch.as_tensor(temperature, device=query_features.device, dtype=query_features.dtype).clamp_min(1.0)
    scale_values = tuple(float(value) for value in template_scale_factors)
    if not scale_values or any(value <= 0.0 for value in scale_values):
        raise ValueError("template_scale_factors must contain positive values")
    center = crop_steps
    yy, xx = torch.meshgrid(
        torch.arange(-context_steps, context_steps + 1, device=query_features.device, dtype=query_features.dtype),
        torch.arange(-context_steps, context_steps + 1, device=query_features.device, dtype=query_features.dtype),
        indexing="ij",
    )
    width = int(query_features.shape[3])
    height = int(query_features.shape[2])
    logits_by_scale = []
    for template_scale in scale_values:
        source_side = int(round(float(template_side) * float(template_scale)))
        source_side = max(1, min(int(expected_side), source_side))
        if source_side % 2 == 0:
            source_side = source_side + 1 if source_side < int(expected_side) else source_side - 1
        half = source_side // 2
        render_template = render_features[:, :, center - half : center + half + 1, center - half : center + half + 1]
        if int(render_template.shape[2]) != template_side or int(render_template.shape[3]) != template_side:
            render_template = F.interpolate(render_template, size=(template_side, template_side), mode="bilinear", align_corners=True)
        render_flat = F.normalize(render_template.reshape(batch, -1), dim=1)
        candidate_logits = []
        for candidate_index in range(int(offsets.shape[1])):
            center_x = float(center) + offsets[:, candidate_index, 0] / feature_step
            center_y = float(center) + offsets[:, candidate_index, 1] / feature_step
            sample_x = center_x[:, None, None] + xx[None, :, :]
            sample_y = center_y[:, None, None] + yy[None, :, :]
            if width > 1:
                grid_x = 2.0 * sample_x / float(width - 1) - 1.0
            else:
                grid_x = torch.zeros_like(sample_x)
            if height > 1:
                grid_y = 2.0 * sample_y / float(height - 1) - 1.0
            else:
                grid_y = torch.zeros_like(sample_y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            query_window = F.grid_sample(
                query_features.float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            query_flat = F.normalize(query_window.reshape(batch, -1), dim=1)
            candidate_logits.append(torch.sum(query_flat * render_flat, dim=1))
        logits_by_scale.append(torch.stack(candidate_logits, dim=1))
    logits = torch.stack(logits_by_scale, dim=0).max(dim=0).values
    return logits * scale, offsets


def template_search_cost_volume_logits(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
    feature_step_px: float | None = None,
    output_step_px: float | None = None,
    temperature: torch.Tensor | float = 10.0,
    template_scale_factors: Sequence[float] = (1.0,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slide a render template over the query search patch and score each offset.

    Both inputs are feature maps extracted from crops with radius
    search_radius_px + context_radius_px. The render template is the central
    context window; every query candidate uses the same-sized context window
    shifted by one search offset.
    """

    offsets = local_offset_grid(
        search_radius_px=float(search_radius_px),
        step_px=float(step_px if output_step_px is None else output_step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    logits, _offsets = _template_search_cost_volume_logits_for_offsets(
        query_features,
        render_features,
        candidate_offsets_xy=offsets,
        search_radius_px=float(search_radius_px),
        context_radius_px=float(context_radius_px),
        feature_step_px=float(step_px if feature_step_px is None else feature_step_px),
        temperature=temperature,
        template_scale_factors=template_scale_factors,
    )
    return logits, offsets.detach().cpu()


def coarse_to_fine_template_search_cost_volume_logits(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    coarse_search_radius_px: float,
    coarse_step_px: float,
    fine_search_radius_px: float,
    fine_step_px: float,
    context_radius_px: float,
    feature_step_px: float,
    temperature: torch.Tensor | float = 10.0,
    template_scale_factors: Sequence[float] = (1.0,),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    full_search_radius = float(coarse_search_radius_px) + float(fine_search_radius_px)
    coarse_offsets_grid = local_offset_grid(
        search_radius_px=float(coarse_search_radius_px),
        step_px=float(coarse_step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    coarse_logits, _coarse_offsets = _template_search_cost_volume_logits_for_offsets(
        query_features,
        render_features,
        candidate_offsets_xy=coarse_offsets_grid,
        search_radius_px=full_search_radius,
        context_radius_px=float(context_radius_px),
        feature_step_px=float(feature_step_px),
        temperature=temperature,
        template_scale_factors=template_scale_factors,
    )
    coarse_mode = coarse_offsets_grid[torch.argmax(coarse_logits, dim=1)]
    fine_local_offsets = local_offset_grid(
        search_radius_px=float(fine_search_radius_px),
        step_px=float(fine_step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    fine_offsets = coarse_mode[:, None, :] + fine_local_offsets[None, :, :]
    fine_logits, fine_offsets = _template_search_cost_volume_logits_for_offsets(
        query_features,
        render_features,
        candidate_offsets_xy=fine_offsets,
        search_radius_px=full_search_radius,
        context_radius_px=float(context_radius_px),
        feature_step_px=float(feature_step_px),
        temperature=temperature,
        template_scale_factors=template_scale_factors,
    )
    return fine_logits, fine_offsets, coarse_logits, coarse_offsets_grid.detach().cpu()


def cost_volume_quality_features(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (B,K)")
    values = logits.float()
    count = int(values.shape[1])
    if count < 2:
        raise ValueError("cost volume must contain at least two offsets")
    mean = torch.mean(values, dim=1)
    std = torch.std(values, dim=1, unbiased=False).clamp_min(1e-6)
    top2 = torch.topk(values, k=2, dim=1).values
    top1 = top2[:, 0]
    top2_value = top2[:, 1]
    probs = F.softmax(values - mean[:, None], dim=1)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=1)
    entropy_norm = entropy / max(1e-6, math.log(float(count)))
    top1_centered = top1 - mean
    top_gap = top1 - top2_value
    return torch.stack(
        [
            top1_centered / std,
            top_gap / std,
            torch.log1p(std),
            entropy_norm,
            torch.max(probs, dim=1).values,
            top1_centered,
        ],
        dim=1,
    )


class TexturePatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 32,
        hidden_dim: int | None = None,
        input_mode: str = "rgb",
        encoder_arch: str = "simple",
    ) -> None:
        super().__init__()
        out = int(feature_dim)
        hidden = int(hidden_dim) if hidden_dim is not None else max(out, 16)
        if out <= 0 or hidden <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        mode = str(input_mode)
        if mode not in {"rgb", "rgb_graygrad", "norm_graygrad"}:
            raise ValueError("input_mode must be 'rgb', 'rgb_graygrad', or 'norm_graygrad'")
        arch = str(encoder_arch).strip().lower() or "simple"
        if arch not in {"simple", "fpn"}:
            raise ValueError("encoder_arch must be 'simple' or 'fpn'")
        self.input_mode = mode
        self.encoder_arch = arch
        self.input_channels = 3 if mode in {"rgb", "norm_graygrad"} else 6
        if arch == "simple":
            self.net = nn.Sequential(
                nn.Conv2d(self.input_channels, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, out, 1),
            )
            self.stem = None
            self.down1 = None
            self.down2 = None
            self.lateral0 = None
            self.lateral1 = None
            self.lateral2 = None
            self.fpn_out = None
        else:
            self.net = None
            self.stem = nn.Sequential(
                nn.Conv2d(self.input_channels, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
            )
            self.down1 = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
            )
            self.down2 = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(1, hidden),
                nn.GELU(),
            )
            self.lateral0 = nn.Conv2d(hidden, out, 1)
            self.lateral1 = nn.Conv2d(hidden, out, 1)
            self.lateral2 = nn.Conv2d(hidden, out, 1)
            self.fpn_out = nn.Sequential(
                nn.Conv2d(out, out, 3, padding=1),
                nn.GroupNorm(1, out),
                nn.GELU(),
                nn.Conv2d(out, out, 1),
            )

    def _prepare_input(self, patch: torch.Tensor) -> torch.Tensor:
        values = patch.float()
        if self.input_mode == "rgb":
            return values
        gray = 0.2989 * values[:, 0:1] + 0.5870 * values[:, 1:2] + 0.1140 * values[:, 2:3]
        if self.input_mode == "norm_graygrad":
            mean = torch.mean(gray, dim=(2, 3), keepdim=True)
            std = torch.std(gray, dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
            gray = (gray - mean) / std
        dx = F.pad(gray[:, :, :, 2:] - gray[:, :, :, :-2], (1, 1, 0, 0)) * 0.5
        dy = F.pad(gray[:, :, 2:, :] - gray[:, :, :-2, :], (0, 0, 1, 1)) * 0.5
        if self.input_mode == "norm_graygrad":
            return torch.cat([gray, dx, dy], dim=1)
        return torch.cat([values, gray, dx, dy], dim=1)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        values = self._prepare_input(patch)
        if self.encoder_arch == "simple":
            return F.normalize(self.net(values), dim=1)
        f0 = self.stem(values)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        p0 = self.lateral0(f0)
        p1 = F.interpolate(self.lateral1(f1), size=f0.shape[-2:], mode="bilinear", align_corners=True)
        p2 = F.interpolate(self.lateral2(f2), size=f0.shape[-2:], mode="bilinear", align_corners=True)
        return F.normalize(self.fpn_out(p0 + p1 + p2), dim=1)


class RGBPatchMeasurementBranch(nn.Module):
    """High-resolution local RGB patch measurement branch for fixed render anchors."""

    def __init__(
        self,
        *,
        search_radius_px: float,
        context_radius_px: float,
        step_px: float,
        coarse_search_radius_px: float | None = None,
        coarse_step_px: float | None = None,
        feature_dim: int = 32,
        hidden_dim: int | None = None,
        input_mode: str = "rgb",
        encoder_arch: str = "simple",
        template_scale_factors: Sequence[float] = (1.0,),
        condition_on_prior_scale: bool = False,
        prior_scale_expert_centers_px: Sequence[float] = (),
        prior_scale_expert_projection: bool = False,
        prior_scale_expert_gate: str = "soft",
    ) -> None:
        super().__init__()
        self.search_radius_px = float(search_radius_px)
        self.context_radius_px = float(context_radius_px)
        self.step_px = float(step_px)
        if self.search_radius_px < 0.0 or self.context_radius_px < 0.0 or self.step_px <= 0.0:
            raise ValueError("search/context radii and step are invalid")
        if (coarse_search_radius_px is None) != (coarse_step_px is None):
            raise ValueError("coarse_search_radius_px and coarse_step_px must be provided together")
        self.coarse_search_radius_px = None if coarse_search_radius_px is None else float(coarse_search_radius_px)
        self.coarse_step_px = None if coarse_step_px is None else float(coarse_step_px)
        if self.coarse_search_radius_px is not None:
            if self.coarse_search_radius_px < 0.0 or self.coarse_step_px is None or self.coarse_step_px <= 0.0:
                raise ValueError("coarse search radius and step are invalid")
            if self.coarse_step_px < self.step_px:
                raise ValueError("coarse_step_px must be >= fine step_px")
        self.input_mode = str(input_mode)
        self.encoder_arch = str(encoder_arch).strip().lower() or "simple"
        scale_values = tuple(float(value) for value in template_scale_factors)
        if not scale_values or any(value <= 0.0 for value in scale_values):
            raise ValueError("template_scale_factors must contain positive values")
        self.template_scale_factors = scale_values
        self.condition_on_prior_scale = bool(condition_on_prior_scale)
        expert_centers = tuple(float(value) for value in prior_scale_expert_centers_px if float(value) > 0.0)
        self.prior_scale_expert_centers_px = expert_centers
        self.prior_scale_expert_projection = bool(prior_scale_expert_projection)
        gate = str(prior_scale_expert_gate)
        if gate not in {"soft", "hard"}:
            raise ValueError("prior_scale_expert_gate must be 'soft' or 'hard'")
        self.prior_scale_expert_gate = gate
        self.encoder = TexturePatchEncoder(
            feature_dim=int(feature_dim),
            hidden_dim=hidden_dim,
            input_mode=self.input_mode,
            encoder_arch=self.encoder_arch,
        )
        pair_dim = int(feature_dim) * 4
        hidden = int(hidden_dim) if hidden_dim is not None else max(int(feature_dim) * 2, 32)
        self.delta_head = nn.Sequential(
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.logit_scale = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        grid_count = _grid_side(self.search_radius_px, self.step_px) ** 2
        if self.condition_on_prior_scale:
            self.prior_scale_logit_bias = nn.Sequential(
                nn.Linear(3, hidden),
                nn.GELU(),
                nn.Linear(hidden, int(grid_count)),
            )
        else:
            self.prior_scale_logit_bias = None
        if expert_centers:
            self.register_buffer("prior_scale_expert_centers", torch.tensor(expert_centers, dtype=torch.float32))
            self.prior_scale_expert_logit_bias = nn.Parameter(torch.zeros((len(expert_centers), int(grid_count)), dtype=torch.float32))
            if self.prior_scale_expert_projection:
                self.prior_scale_expert_query_projections = nn.ModuleList(
                    [nn.Conv2d(int(feature_dim), int(feature_dim), kernel_size=1, bias=False) for _ in expert_centers]
                )
                self.prior_scale_expert_render_projections = nn.ModuleList(
                    [nn.Conv2d(int(feature_dim), int(feature_dim), kernel_size=1, bias=False) for _ in expert_centers]
                )
                for projection in list(self.prior_scale_expert_query_projections) + list(self.prior_scale_expert_render_projections):
                    nn.init.dirac_(projection.weight)
            else:
                self.prior_scale_expert_query_projections = None
                self.prior_scale_expert_render_projections = None
            gate_sigma = max(min(expert_centers), float(self.step_px), 1e-3)
            self.prior_scale_expert_gate_sigma_px = float(gate_sigma)
        else:
            self.register_buffer("prior_scale_expert_centers", torch.empty((0,), dtype=torch.float32))
            self.prior_scale_expert_logit_bias = None
            self.prior_scale_expert_query_projections = None
            self.prior_scale_expert_render_projections = None
            self.prior_scale_expert_gate_sigma_px = 1.0
        self.dustbin_head = nn.Sequential(
            nn.Linear(pair_dim + 6, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.dustbin_head[-1].bias, -4.0)
        self.measurement_gate_head = nn.Sequential(
            nn.Linear(pair_dim + 12, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.measurement_gate_head[-1].bias, -1.5)

    @property
    def crop_radius_px(self) -> float:
        return float(self.measurement_search_radius_px + self.context_radius_px)

    @property
    def use_coarse_to_fine(self) -> bool:
        return self.coarse_search_radius_px is not None and self.coarse_step_px is not None

    @property
    def measurement_search_radius_px(self) -> float:
        if self.coarse_search_radius_px is None:
            return float(self.search_radius_px)
        return float(self.coarse_search_radius_px + self.search_radius_px)

    def _prior_scale_features(self, prior_scale_px: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = prior_scale_px.to(device=device, dtype=dtype).reshape(int(batch_size), 1)
        normalised = (scale / max(float(self.search_radius_px), 1e-6)).clamp(0.0, 4.0)
        return torch.cat([normalised, normalised * normalised, torch.log1p(normalised)], dim=1)

    def _prior_scale_expert_bias(self, prior_scale_px: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.prior_scale_expert_logit_bias is None:
            return None
        weights = self._prior_scale_expert_weights(prior_scale_px, batch_size=batch_size, device=device, dtype=dtype)
        return weights @ self.prior_scale_expert_logit_bias.to(device=device, dtype=dtype)

    def _prior_scale_expert_weights(self, prior_scale_px: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scale = prior_scale_px.to(device=device, dtype=dtype).reshape(int(batch_size), 1)
        centers = self.prior_scale_expert_centers.to(device=device, dtype=dtype).reshape(1, -1)
        dist2 = (scale - centers) ** 2
        if self.prior_scale_expert_gate == "hard":
            best = torch.argmin(dist2, dim=1)
            return F.one_hot(best, num_classes=int(centers.shape[1])).to(device=device, dtype=dtype)
        return F.softmax(-0.5 * dist2 / max(float(self.prior_scale_expert_gate_sigma_px) ** 2, 1e-12), dim=1)

    def forward_from_patches(
        self,
        query_patch: torch.Tensor,
        render_patch: torch.Tensor,
        prior_scale_px: torch.Tensor | None = None,
    ) -> RGBPatchMeasurementPrediction:
        if query_patch.ndim != 4 or render_patch.ndim != 4:
            raise ValueError("query_patch and render_patch must have shape (B,3,H,W)")
        if int(query_patch.shape[0]) != int(render_patch.shape[0]):
            raise ValueError("query_patch and render_patch must share batch size")
        qfeat = self.encoder(query_patch)
        rfeat = self.encoder(render_patch)
        query_pool = torch.mean(qfeat.flatten(2), dim=2)
        render_pool = torch.mean(rfeat.flatten(2), dim=2)
        pair = torch.cat([query_pool, render_pool, query_pool - render_pool, query_pool * render_pool], dim=1)
        delta_raw = self.delta_head(pair)
        direct_mean = torch.tanh(delta_raw[:, :2]) * float(self.measurement_search_radius_px)
        direct_log_sigma = delta_raw[:, 2:].clamp(-5.0, 3.0)
        scale = torch.clamp(self.logit_scale, min=1.0, max=100.0)
        coarse_logits = None
        coarse_offsets = None
        coarse_mode = None
        if self.prior_scale_expert_query_projections is not None and self.prior_scale_expert_render_projections is not None:
            if prior_scale_px is None:
                prior_scale_px = torch.zeros((int(query_patch.shape[0]),), device=query_patch.device, dtype=query_patch.dtype)
            weights = self._prior_scale_expert_weights(
                prior_scale_px,
                batch_size=int(query_patch.shape[0]),
                device=qfeat.device,
                dtype=qfeat.dtype,
            )
            expert_logits = []
            expert_coarse_logits = []
            offsets = None
            for query_projection, render_projection in zip(self.prior_scale_expert_query_projections, self.prior_scale_expert_render_projections):
                if self.use_coarse_to_fine:
                    logits_i, offsets_i, coarse_logits_i, coarse_offsets_i = coarse_to_fine_template_search_cost_volume_logits(
                        F.normalize(query_projection(qfeat), dim=1),
                        F.normalize(render_projection(rfeat), dim=1),
                        coarse_search_radius_px=float(self.coarse_search_radius_px),
                        coarse_step_px=float(self.coarse_step_px),
                        fine_search_radius_px=float(self.search_radius_px),
                        fine_step_px=float(self.step_px),
                        context_radius_px=float(self.context_radius_px),
                        feature_step_px=float(self.step_px),
                        temperature=scale,
                        template_scale_factors=self.template_scale_factors,
                    )
                    expert_coarse_logits.append(coarse_logits_i)
                    coarse_offsets = coarse_offsets_i
                else:
                    logits_i, offsets_i = template_search_cost_volume_logits(
                        F.normalize(query_projection(qfeat), dim=1),
                        F.normalize(render_projection(rfeat), dim=1),
                        search_radius_px=self.search_radius_px,
                        context_radius_px=self.context_radius_px,
                        step_px=self.step_px,
                        temperature=scale,
                        template_scale_factors=self.template_scale_factors,
                    )
                expert_logits.append(logits_i)
                offsets = offsets_i
            logits = torch.sum(torch.stack(expert_logits, dim=1) * weights[:, :, None], dim=1)
            if expert_coarse_logits:
                coarse_logits = torch.sum(torch.stack(expert_coarse_logits, dim=1) * weights[:, :, None], dim=1)
            if offsets is None:
                raise RuntimeError("prior scale expert projection produced no logits")
        else:
            if self.use_coarse_to_fine:
                logits, offsets, coarse_logits, coarse_offsets = coarse_to_fine_template_search_cost_volume_logits(
                    qfeat,
                    rfeat,
                    coarse_search_radius_px=float(self.coarse_search_radius_px),
                    coarse_step_px=float(self.coarse_step_px),
                    fine_search_radius_px=float(self.search_radius_px),
                    fine_step_px=float(self.step_px),
                    context_radius_px=float(self.context_radius_px),
                    feature_step_px=float(self.step_px),
                    temperature=scale,
                    template_scale_factors=self.template_scale_factors,
                )
            else:
                logits, offsets = template_search_cost_volume_logits(
                    qfeat,
                    rfeat,
                    search_radius_px=self.search_radius_px,
                    context_radius_px=self.context_radius_px,
                    step_px=self.step_px,
                    temperature=scale,
                    template_scale_factors=self.template_scale_factors,
                )
        if coarse_logits is not None and coarse_offsets is not None:
            coarse_offsets_device = coarse_offsets.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
            coarse_mode = coarse_offsets_device[torch.argmax(coarse_logits, dim=1)]
        if self.prior_scale_logit_bias is not None:
            if prior_scale_px is None:
                prior_scale_px = torch.zeros((int(query_patch.shape[0]),), device=query_patch.device, dtype=query_patch.dtype)
            prior_features = self._prior_scale_features(
                prior_scale_px,
                batch_size=int(query_patch.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = logits + self.prior_scale_logit_bias(prior_features)
        if self.prior_scale_expert_logit_bias is not None:
            if prior_scale_px is None:
                prior_scale_px = torch.zeros((int(query_patch.shape[0]),), device=query_patch.device, dtype=query_patch.dtype)
            expert_bias = self._prior_scale_expert_bias(
                prior_scale_px,
                batch_size=int(query_patch.shape[0]),
                device=logits.device,
                dtype=logits.dtype,
            )
            if expert_bias is not None:
                logits = logits + expert_bias
        quality = cost_volume_quality_features(logits)
        _log_probs, likelihood_mean, _likelihood_cov, _likelihood_mode = likelihood_moments_from_logits(logits, offsets)
        radius = max(float(self.measurement_search_radius_px), 1e-6)
        gate_features = torch.cat(
            [
                pair,
                quality.to(device=pair.device, dtype=pair.dtype),
                likelihood_mean.to(device=pair.device, dtype=pair.dtype) / radius,
                direct_mean.to(device=pair.device, dtype=pair.dtype) / radius,
                direct_log_sigma.to(device=pair.device, dtype=pair.dtype),
            ],
            dim=1,
        )
        gate_logit = self.measurement_gate_head(gate_features).reshape(int(query_patch.shape[0]))
        gate_probability = torch.sigmoid(gate_logit)
        gated_mean = gate_probability[:, None] * likelihood_mean.to(device=pair.device, dtype=pair.dtype)
        dustbin = self.dustbin_head(torch.cat([pair, quality.to(device=pair.device, dtype=pair.dtype)], dim=1)).reshape(int(query_patch.shape[0]))
        return RGBPatchMeasurementPrediction(
            logits=logits,
            offsets_xy=offsets,
            dustbin_logit=dustbin,
            coarse_logits=coarse_logits,
            coarse_offsets_xy=coarse_offsets,
            coarse_mode_offset_xy=coarse_mode,
            local_log_probs=_log_probs,
            mean_offset_xy=likelihood_mean,
            cov_2x2=_likelihood_cov,
            mode_offset_xy=_likelihood_mode,
            direct_mean_offset_xy=direct_mean,
            direct_log_sigma_xy=direct_log_sigma,
            gated_mean_offset_xy=gated_mean,
            gate_logit=gate_logit,
            gate_probability=gate_probability,
        )

    def forward(
        self,
        *,
        query_images: torch.Tensor,
        render_images: torch.Tensor,
        query_centers_xy: torch.Tensor,
        render_anchor_xy: torch.Tensor,
        image_width: int,
        image_height: int,
        prior_scale_px: torch.Tensor | None = None,
    ) -> RGBPatchMeasurementPrediction:
        query_patch, _ = crop_rgb_window(
            query_images,
            query_centers_xy,
            radius_px=self.crop_radius_px,
            step_px=self.step_px,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        render_patch, _ = crop_rgb_window(
            render_images,
            render_anchor_xy,
            radius_px=self.crop_radius_px,
            step_px=self.step_px,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        return self.forward_from_patches(query_patch, render_patch, prior_scale_px=prior_scale_px)
