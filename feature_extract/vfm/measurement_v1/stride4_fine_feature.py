from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Stride4FineFeatureExtractor(nn.Module):
    """Lightweight whole-image stride-4 fine feature pyramid."""

    def __init__(self, *, output_dim: int = 64, base_dim: int = 32) -> None:
        super().__init__()
        out = int(output_dim)
        base = int(base_dim)
        if out <= 0 or base <= 0:
            raise ValueError("output_dim and base_dim must be positive")
        self.net = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1),
            nn.GroupNorm(1, base),
            nn.GELU(),
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            nn.GroupNorm(1, base),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(1, base),
            nn.GELU(),
            nn.Conv2d(base, out, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError("image must have shape (B,3,H,W)")
        return F.normalize(self.net(image.float()), dim=1)


def _normalise_xy(xy: torch.Tensor, *, image_width: int, image_height: int) -> torch.Tensor:
    x = xy[..., 0]
    y = xy[..., 1]
    x_norm = 2.0 * x / max(float(image_width - 1), 1.0) - 1.0
    y_norm = 2.0 * y / max(float(image_height - 1), 1.0) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def _sample_features(feature_map: torch.Tensor, xy: torch.Tensor, *, image_width: int, image_height: int) -> torch.Tensor:
    grid = _normalise_xy(xy, image_width=image_width, image_height=image_height).reshape(int(feature_map.shape[0]), -1, 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


def _offset_grid(*, radius_px: float, step_px: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if float(radius_px) < 0.0 or float(step_px) <= 0.0:
        raise ValueError("radius_px must be non-negative and step_px must be positive")
    values = torch.arange(
        -float(radius_px),
        float(radius_px) + 0.5 * float(step_px),
        float(step_px),
        device=device,
        dtype=dtype,
    )
    dx, dy = torch.meshgrid(values, values, indexing="xy")
    return torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=1)


def crop_feature_window(
    feature_map: torch.Tensor,
    centers_xy: torch.Tensor,
    *,
    radius_px: float,
    step_px: float,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape (B,C,H,W)")
    batch = int(feature_map.shape[0])
    offsets = _offset_grid(radius_px=float(radius_px), step_px=float(step_px), device=feature_map.device, dtype=feature_map.dtype)
    side = int(round((2.0 * float(radius_px)) / float(step_px))) + 1
    if int(offsets.shape[0]) != side * side:
        raise ValueError("radius_px and step_px do not form a square sampling grid")
    centers = centers_xy.to(device=feature_map.device, dtype=feature_map.dtype).reshape(batch, 1, 2)
    xy = centers + offsets.reshape(1, -1, 2)
    grid = _normalise_xy(xy.reshape(batch, side, side, 2), image_width=int(image_width), image_height=int(image_height))
    patch = F.grid_sample(feature_map.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return patch, offsets.detach().cpu()


def local_correlation_logits(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    query_centers_xy: torch.Tensor,
    render_anchor_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    step_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_features.ndim != 4 or render_features.ndim != 4:
        raise ValueError("feature maps must have shape (B,C,H,W)")
    if int(query_features.shape[0]) != int(render_features.shape[0]):
        raise ValueError("query and render feature maps must share batch size")
    offsets = _offset_grid(
        radius_px=float(search_radius_px),
        step_px=float(step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    xy = offsets + query_centers_xy.to(query_features.device, query_features.dtype).reshape(-1, 1, 2)
    render_desc = _sample_features(
        render_features,
        render_anchor_xy.to(render_features.device, render_features.dtype).reshape(-1, 1, 2),
        image_width=int(image_width),
        image_height=int(image_height),
    )[:, 0, :]
    query_desc = _sample_features(query_features, xy, image_width=int(image_width), image_height=int(image_height))
    logits = torch.sum(F.normalize(query_desc, dim=2) * F.normalize(render_desc, dim=1)[:, None, :], dim=2)
    xy_out = xy.detach().cpu()
    if int(xy_out.shape[0]) == 1:
        xy_out = xy_out[0]
    return logits, xy_out


def local_template_correlation_logits(
    query_features: torch.Tensor,
    render_features: torch.Tensor,
    *,
    query_centers_xy: torch.Tensor,
    render_anchor_xy: torch.Tensor,
    image_width: int,
    image_height: int,
    search_radius_px: float,
    context_radius_px: float,
    step_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score query search candidates using a render context template.

    Coordinates are expressed in the original image pixel frame. The feature
    maps may be lower resolution; ``grid_sample`` handles the normalized
    coordinate transform so long as the map spans the same image extent.
    """

    if query_features.ndim != 4 or render_features.ndim != 4:
        raise ValueError("feature maps must have shape (B,C,H,W)")
    if int(query_features.shape[0]) != int(render_features.shape[0]):
        raise ValueError("query and render feature maps must share batch size")
    if int(query_features.shape[1]) != int(render_features.shape[1]):
        raise ValueError("query and render feature maps must share channel count")
    search_offsets = _offset_grid(
        radius_px=float(search_radius_px),
        step_px=float(step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    context_offsets = _offset_grid(
        radius_px=float(context_radius_px),
        step_px=float(step_px),
        device=query_features.device,
        dtype=query_features.dtype,
    )
    batch = int(query_features.shape[0])
    query_centers = query_centers_xy.to(query_features.device, query_features.dtype).reshape(batch, 1, 1, 2)
    render_anchors = render_anchor_xy.to(render_features.device, render_features.dtype).reshape(batch, 1, 2)
    render_xy = render_anchors + context_offsets.reshape(1, -1, 2)
    render_desc = _sample_features(
        render_features,
        render_xy,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    query_xy = query_centers + search_offsets.reshape(1, -1, 1, 2) + context_offsets.reshape(1, 1, -1, 2)
    query_desc = _sample_features(
        query_features,
        query_xy.reshape(batch, -1, 2),
        image_width=int(image_width),
        image_height=int(image_height),
    ).reshape(batch, int(search_offsets.shape[0]), int(context_offsets.shape[0]), int(query_features.shape[1]))
    render_desc = F.normalize(render_desc, dim=2)
    query_desc = F.normalize(query_desc, dim=3)
    logits = torch.mean(torch.sum(query_desc * render_desc[:, None, :, :], dim=3), dim=2)
    sample_xy = query_centers_xy.to(query_features.device, query_features.dtype).reshape(batch, 1, 2) + search_offsets.reshape(1, -1, 2)
    xy_out = sample_xy.detach().cpu()
    if int(xy_out.shape[0]) == 1:
        xy_out = xy_out[0]
    return logits, xy_out
