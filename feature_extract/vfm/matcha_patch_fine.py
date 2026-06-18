"""RGB patch-correlation fine head for RADIO-adapted MATCHA."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _normalise_grid_xy(x: torch.Tensor, y: torch.Tensor, *, width: int, height: int) -> torch.Tensor:
    if int(width) > 1:
        x_norm = 2.0 * x / float(int(width) - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x)
    if int(height) > 1:
        y_norm = 2.0 * y / float(int(height) - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y)
    return torch.stack([x_norm, y_norm], dim=-1)


def _select_pair_images(images: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    if int(images.shape[0]) == 1:
        return images.expand(int(pairs.numel()), -1, -1, -1)
    return images[pairs]


def extract_query_source_patch(
    images: torch.Tensor,
    pair_indices: torch.Tensor,
    query_xy: torch.Tensor,
    *,
    patch_size: int = 32,
) -> torch.Tensor:
    """Crop fixed-size query RGB patches centered on continuous image coordinates."""

    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B, 3, H, W)")
    size = int(patch_size)
    if size <= 0:
        raise ValueError("patch_size must be positive")
    pairs = pair_indices.long().reshape(-1).clamp(0, int(images.shape[0]) - 1)
    xy = query_xy.to(device=images.device, dtype=images.dtype).reshape(-1, 2)
    if int(pairs.numel()) != int(xy.shape[0]):
        raise ValueError("pair_indices and query_xy must contain the same number of entries")
    height, width = int(images.shape[2]), int(images.shape[3])
    offsets = torch.arange(size, dtype=images.dtype, device=images.device) - float(size // 2)
    y = xy[:, 1, None, None] + offsets[None, :, None]
    x = xy[:, 0, None, None] + offsets[None, None, :]
    x = x.expand(-1, size, size)
    y = y.expand(-1, size, size)
    grid = _normalise_grid_xy(x, y, width=width, height=height)
    return F.grid_sample(_select_pair_images(images, pairs), grid, mode="bilinear", padding_mode="border", align_corners=True)


def extract_render_cell_patch(
    images: torch.Tensor,
    pair_indices: torch.Tensor,
    render_cell_indices: torch.Tensor,
    *,
    render_grid_hw: tuple[int, int],
    patch_size: int = 32,
) -> torch.Tensor:
    """Crop each render coarse cell and resize it to a fixed RGB patch."""

    if images.ndim != 4 or int(images.shape[1]) != 3:
        raise ValueError("images must have shape (B, 3, H, W)")
    size = int(patch_size)
    if size <= 0:
        raise ValueError("patch_size must be positive")
    grid_h, grid_w = int(render_grid_hw[0]), int(render_grid_hw[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("render_grid_hw must contain positive dimensions")
    pairs = pair_indices.long().reshape(-1).clamp(0, int(images.shape[0]) - 1)
    cells = render_cell_indices.long().reshape(-1).clamp(0, grid_h * grid_w - 1)
    if int(pairs.numel()) != int(cells.numel()):
        raise ValueError("pair_indices and render_cell_indices must contain the same number of entries")
    height, width = int(images.shape[2]), int(images.shape[3])
    rows = torch.div(cells, grid_w, rounding_mode="floor").to(dtype=images.dtype, device=images.device)
    cols = torch.remainder(cells, grid_w).to(dtype=images.dtype, device=images.device)
    cell_w = float(width) / float(grid_w)
    cell_h = float(height) / float(grid_h)
    sample = torch.arange(size, dtype=images.dtype, device=images.device)
    x = cols[:, None, None] * cell_w + ((sample[None, None, :] + 0.5) / float(size)) * cell_w - 0.5
    y = rows[:, None, None] * cell_h + ((sample[None, :, None] + 0.5) / float(size)) * cell_h - 0.5
    x = x.expand(-1, size, size)
    y = y.expand(-1, size, size)
    grid = _normalise_grid_xy(x, y, width=width, height=height)
    return F.grid_sample(_select_pair_images(images, pairs), grid, mode="bilinear", padding_mode="border", align_corners=True)


def explicit_token_cost_volume_logits(
    source_token: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    temperature: torch.Tensor | float = 10.0,
) -> torch.Tensor:
    """Score a target local window by explicit normalized dot-product correlation."""

    source = source_token.reshape(int(source_token.shape[0]), -1)
    target = target_tokens.reshape(int(target_tokens.shape[0]), int(target_tokens.shape[1]), -1)
    if int(source.shape[0]) != int(target.shape[0]):
        raise ValueError("source_token and target_tokens must have the same batch size")
    if int(source.shape[1]) != int(target.shape[2]):
        raise ValueError("source_token and target_tokens channel counts must match")
    source = F.normalize(source, dim=1)
    target = F.normalize(target, dim=2)
    logits = torch.sum(target * source[:, None, :], dim=2)
    return logits * temperature


def explicit_patch_cost_volume_logits(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    offset_bins: int = 8,
    temperature: torch.Tensor | float = 10.0,
) -> torch.Tensor:
    """Build 8x8 local-window logits from source-center to target-token correlation."""

    if source_features.ndim != 4 or target_features.ndim != 4:
        raise ValueError("source_features and target_features must have shape (B, C, H, W)")
    if int(source_features.shape[0]) != int(target_features.shape[0]):
        raise ValueError("source_features and target_features must have the same batch size")
    if int(source_features.shape[1]) != int(target_features.shape[1]):
        raise ValueError("source_features and target_features channel counts must match")
    bins = int(offset_bins)
    if bins <= 0:
        raise ValueError("offset_bins must be positive")
    source_grid = F.adaptive_avg_pool2d(source_features, (bins, bins))
    target_grid = F.adaptive_avg_pool2d(target_features, (bins, bins))
    center = min(bins - 1, bins // 2)
    source_token = source_grid[:, :, center, center]
    target_tokens = target_grid.flatten(2).transpose(1, 2).contiguous()
    return explicit_token_cost_volume_logits(source_token, target_tokens, temperature=temperature)


class PatchCorrelationFineHead(nn.Module):
    """Pair-conditioned 8x8 offset classifier from explicit local cost volume."""

    def __init__(
        self,
        *,
        context_dim: int,
        hidden_dim: int = 128,
        patch_size: int = 32,
        offset_bins: int = 8,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.context_dim = int(context_dim)
        self.patch_size = int(patch_size)
        self.offset_bins = int(offset_bins)
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.offset_bins <= 0:
            raise ValueError("offset_bins must be positive")
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2 if hidden >= 2 else hidden, 3, padding=1),
            nn.GroupNorm(1, hidden // 2 if hidden >= 2 else hidden),
            nn.GELU(),
            nn.Conv2d(hidden // 2 if hidden >= 2 else hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        self.query_context_proj = nn.Linear(self.context_dim, hidden)
        self.render_context_proj = nn.Linear(self.context_dim, hidden)
        self.query_proj = nn.Linear(hidden, hidden)
        self.render_proj = nn.Linear(hidden, hidden)
        self.correlation_logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(
        self,
        query_images: torch.Tensor,
        render_images: torch.Tensor,
        pair_indices: torch.Tensor,
        query_xy: torch.Tensor,
        render_cell_indices: torch.Tensor,
        *,
        render_grid_hw: tuple[int, int],
        query_context: torch.Tensor,
        render_context: torch.Tensor,
    ) -> torch.Tensor:
        pairs = pair_indices.long().reshape(-1)
        qctx = query_context.reshape(int(pairs.numel()), -1)
        rctx = render_context.reshape(int(pairs.numel()), -1)
        if int(qctx.shape[1]) != self.context_dim or int(rctx.shape[1]) != self.context_dim:
            raise ValueError("query_context and render_context channel counts must match context_dim")
        query_patch = extract_query_source_patch(query_images, pairs, query_xy, patch_size=self.patch_size)
        render_patch = extract_render_cell_patch(
            render_images,
            pairs,
            render_cell_indices,
            render_grid_hw=render_grid_hw,
            patch_size=self.patch_size,
        )
        query_feat = self.rgb_encoder(query_patch)
        render_feat = self.rgb_encoder(render_patch)
        bins = int(self.offset_bins)
        query_grid = F.adaptive_avg_pool2d(query_feat, (bins, bins))
        render_grid = F.adaptive_avg_pool2d(render_feat, (bins, bins))
        center = min(bins - 1, bins // 2)
        query_token = query_grid[:, :, center, center]
        render_tokens = render_grid.flatten(2).transpose(1, 2).contiguous()
        query_hidden = self.query_proj(query_token) + self.query_context_proj(qctx)
        render_hidden = self.render_proj(render_tokens) + self.render_context_proj(rctx)[:, None, :]
        scale = torch.clamp(torch.exp(self.correlation_logit_scale), min=1.0, max=100.0)
        return explicit_token_cost_volume_logits(query_hidden, render_hidden, temperature=scale)
