"""Pose-observability diagnostics for localization features."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _resize_map(tensor: torch.Tensor, hw: tuple[int, int], *, mode: str) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4:
        raise ValueError("map tensor must have shape (B,1,H,W) or (B,H,W)")
    if tuple(tensor.shape[-2:]) != tuple(hw):
        tensor = F.interpolate(tensor.float(), size=hw, mode=mode, align_corners=False if mode == "bilinear" else None)
    return tensor.float()


def _intrinsics_components(
    intrinsics: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    if intrinsics.ndim == 3 and intrinsics.shape[1] == 1:
        intrinsics = intrinsics[:, 0]
    if intrinsics.ndim == 4 and intrinsics.shape[1] == 1:
        intrinsics = intrinsics[:, 0]
    if intrinsics.ndim == 2 and intrinsics.shape[-1] == 4:
        fx, fy, cx, cy = [intrinsics[:, idx] for idx in range(4)]
    elif intrinsics.ndim == 3 and intrinsics.shape[-2:] == (3, 3):
        fx, fy, cx, cy = intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    else:
        raise ValueError(f"unsupported intrinsics shape {tuple(intrinsics.shape)}")
    if fx.shape[0] != batch_size:
        raise ValueError(f"intrinsics batch {fx.shape[0]} does not match feature batch {batch_size}")
    return fx, fy, cx, cy


def _feature_gradients(feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = F.pad(feature.float(), (1, 1, 1, 1), mode="replicate")
    grad_u = 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])
    grad_v = 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])
    return grad_u, grad_v


def feature_pose_fisher_stats(
    feature: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    damping: float = 1.0e-3,
    normalize_channels: bool = False,
) -> Dict[str, torch.Tensor]:
    """Approximate feature-metric pose Fisher information.

    The approximation follows J = dF/du · du/dxi.  It is meant as a lightweight
    diagnostic for whether a feature map has pose-observable spatial/channel
    variation under the current depth and intrinsics.  It does not replace a
    full differentiable renderer Jacobian.
    """
    if feature.ndim != 4:
        raise ValueError("feature must have shape (B,C,H,W)")
    feature = feature.float()
    if normalize_channels:
        feature = F.normalize(feature, dim=1, eps=1.0e-6)
    bsz, _channels, height, width = feature.shape
    device = feature.device
    dtype = feature.dtype
    depth_f = _resize_map(depth.to(device=device), (height, width), mode="bilinear").to(dtype=dtype)
    valid = torch.isfinite(depth_f) & (depth_f > 1.0e-6)
    if mask is not None:
        valid = valid & (_resize_map(mask.to(device=device), (height, width), mode="nearest") > 0.5)

    fx, fy, cx, cy = _intrinsics_components(intrinsics, batch_size=bsz, device=device, dtype=dtype)
    fx = fx.clamp(min=1.0e-6).view(bsz, 1, 1)
    fy = fy.clamp(min=1.0e-6).view(bsz, 1, 1)
    cx = cx.view(bsz, 1, 1)
    cy = cy.view(bsz, 1, 1)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs[None].expand(bsz, -1, -1)
    ys = ys[None].expand(bsz, -1, -1)
    z = depth_f[:, 0].clamp(min=1.0e-6)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    du_dxyz = torch.stack(
        [
            fx.expand_as(z) / z,
            torch.zeros_like(z),
            -fx.expand_as(z) * x / (z * z),
        ],
        dim=-1,
    )
    dv_dxyz = torch.stack(
        [
            torch.zeros_like(z),
            fy.expand_as(z) / z,
            -fy.expand_as(z) * y / (z * z),
        ],
        dim=-1,
    )
    dxyz_dxi = torch.stack(
        [
            torch.stack([torch.ones_like(z), torch.zeros_like(z), torch.zeros_like(z), torch.zeros_like(z), z, -y], dim=-1),
            torch.stack([torch.zeros_like(z), torch.ones_like(z), torch.zeros_like(z), -z, torch.zeros_like(z), x], dim=-1),
            torch.stack([torch.zeros_like(z), torch.zeros_like(z), torch.ones_like(z), y, -x, torch.zeros_like(z)], dim=-1),
        ],
        dim=-2,
    )
    du_dxi = torch.einsum("bhwc,bhwcd->bhwd", du_dxyz, dxyz_dxi)
    dv_dxi = torch.einsum("bhwc,bhwcd->bhwd", dv_dxyz, dxyz_dxi)

    grad_u, grad_v = _feature_gradients(feature)
    jac = grad_u.permute(0, 2, 3, 1)[..., None] * du_dxi[:, :, :, None, :]
    jac = jac + grad_v.permute(0, 2, 3, 1)[..., None] * dv_dxi[:, :, :, None, :]
    weight = valid[:, 0].to(dtype=dtype)
    valid_count = weight.flatten(1).sum(dim=1).clamp(min=1.0)
    jac = jac * weight[:, :, :, None, None].sqrt()
    fisher = torch.einsum("bhwci,bhwcj->bij", jac, jac) / valid_count[:, None, None]
    eye = torch.eye(6, device=device, dtype=dtype)[None]
    damped = fisher + float(damping) * eye
    sign, logabsdet = torch.linalg.slogdet(damped.float())
    logdet = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, -float("inf")))
    eigvals = torch.linalg.eigvalsh(damped.float())
    condition = eigvals[:, -1] / eigvals[:, 0].clamp(min=1.0e-12)
    trace_inv = torch.diagonal(torch.linalg.inv(damped.float()), dim1=-2, dim2=-1).sum(dim=-1)
    return {
        "logdet": logdet.mean(),
        "trace": torch.diagonal(fisher.float(), dim1=-2, dim2=-1).sum(dim=-1).mean(),
        "trace_inv": trace_inv.mean(),
        "condition": condition.mean(),
        "valid_frac": weight.flatten(1).mean(dim=1).mean(),
    }
