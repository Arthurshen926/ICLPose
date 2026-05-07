"""
ConcatPoseNet: Pose Estimation with Optional RAFT-style GRU Refinement
=======================================================================
Two modes:
  1. **Concat mode** (default): Concatenate query+rendered -> CNN -> one-shot
     flow + confidence -> geometry solver.
  2. **GRU mode** (use_gru=True): RAFT-style iterative refinement using
     warp-guided local correlation + ConvGRU for much better flow.

Both modes default to depth-aware WLS for the final 6-DoF update.  The older
MLP translation/rotation heads remain available for ablations.
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Dict, List, Optional, Tuple

from pose_refine.models.depth_aware_matcher import (
    DepthAwareLocalFlowHead,
    DepthAwareLocalMatcher,
)
from pose_refine.utils.geometry_solver import compute_image_jacobian, diff_pose_solve


# ======================================================================
#  Correlation Functions (adapted from MSFlowPoseNet)
# ======================================================================

def local_correlation(fmap1: torch.Tensor, fmap2: torch.Tensor,
                      radius: int = 4) -> torch.Tensor:
    """Local correlation: dot-product in (2r+1)^2 neighborhood.

    Args:
        fmap1, fmap2: (B, C, H, W) L2-normalized features
        radius: search window radius
    Returns:
        corr: (B, (2r+1)^2, H, W)
    """
    B, C, H, W = fmap1.shape
    radius = int(radius)
    fmap2_pad = F.pad(fmap2, [radius] * 4, mode='constant', value=0)
    corrs = []
    for dy in range(-radius, radius + 1):
        y0 = dy + radius
        for dx in range(-radius, radius + 1):
            x0 = dx + radius
            sampled = fmap2_pad[:, :, y0:y0 + H, x0:x0 + W]
            corrs.append((fmap1 * sampled).sum(dim=1))
    return torch.stack(corrs, dim=1).contiguous()


def guided_local_correlation(fmap1: torch.Tensor, fmap2: torch.Tensor,
                             flow: torch.Tensor, radius: int = 4,
                             checkpoint_offsets: bool = False,
                             ) -> torch.Tensor:
    """Local correlation centered at each pixel's current flow estimate."""
    B, C, H, W = fmap1.shape
    device = fmap1.device
    use_checkpoint = (
        bool(checkpoint_offsets)
        and torch.is_grad_enabled()
        and (fmap1.requires_grad or fmap2.requires_grad)
    )
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow.dtype),
        torch.arange(W, device=device, dtype=flow.dtype),
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)
    base_x = grid_x + flow[:, 0]
    base_y = grid_y + flow[:, 1]

    corrs = []

    def _sample_corr(
        fmap1_arg: torch.Tensor,
        fmap2_arg: torch.Tensor,
        sample_grid_arg: torch.Tensor,
    ) -> torch.Tensor:
        sampled = F.grid_sample(
            fmap2_arg,
            sample_grid_arg,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        return (fmap1_arg * sampled).sum(dim=1)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sample_x = (base_x + dx) / max(W - 1, 1) * 2.0 - 1.0
            sample_y = (base_y + dy) / max(H - 1, 1) * 2.0 - 1.0
            sample_grid = torch.stack([sample_x, sample_y], dim=-1)
            if use_checkpoint:
                corrs.append(
                    torch_checkpoint(
                        _sample_corr,
                        fmap1,
                        fmap2,
                        sample_grid,
                        use_reentrant=False,
                    )
                )
            else:
                corrs.append(
                    _sample_corr(fmap1, fmap2, sample_grid)
            )
    return torch.stack(corrs, dim=1).contiguous()


def soft_argmax_flow_from_correlation(
    corr: torch.Tensor,
    radius: int = 4,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Convert a rendered-centered local correlation volume to subpixel flow."""
    B, channels, H, W = corr.shape
    window = 2 * int(radius) + 1
    expected_channels = window * window
    if channels != expected_channels:
        raise ValueError(
            f"corr has {channels} channels, expected {expected_channels} "
            f"for radius={radius}"
        )
    offsets = torch.arange(-int(radius), int(radius) + 1, device=corr.device, dtype=corr.dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx = dx.reshape(1, expected_channels, 1, 1)
    dy = dy.reshape(1, expected_channels, 1, 1)
    probs = torch.softmax(corr.float() / max(float(temperature), 1e-6), dim=1).to(corr.dtype)
    return torch.cat(
        [
            (probs * dx).sum(dim=1, keepdim=True),
            (probs * dy).sum(dim=1, keepdim=True),
        ],
        dim=1,
    ).reshape(B, 2, H, W)


def correlation_confidence_from_probs(
    probs: torch.Tensor,
    radius: int = 4,
    *,
    mode: str = "max",
    variance_scale: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate WLS confidence from a local-correlation probability volume.

    ``max`` preserves the legacy peak-probability behavior. ``variance`` is
    useful when the correct match is represented by a precise soft/subpixel
    distribution whose maximum probability is not high enough for peak-based
    gating.
    """
    B, channels, H, W = probs.shape
    radius = int(radius)
    window = 2 * radius + 1
    expected_channels = window * window
    if channels != expected_channels:
        raise ValueError(
            f"probs has {channels} channels, expected {expected_channels} "
            f"for radius={radius}"
        )
    mode_key = str(mode or "max").lower()
    if mode_key in {"max", "peak", "probmax", "softmax_max"}:
        return probs.max(dim=1, keepdim=True).values
    if mode_key in {"uniform", "one", "ones", "none"}:
        return torch.ones(B, 1, H, W, device=probs.device, dtype=probs.dtype)
    if mode_key in {"entropy", "negentropy"}:
        entropy = -(probs.clamp(min=eps) * probs.clamp(min=eps).log()).sum(dim=1, keepdim=True)
        max_entropy = math.log(float(expected_channels))
        return (1.0 - entropy / max(max_entropy, eps)).clamp(min=0.0, max=1.0).to(probs.dtype)
    if mode_key in {"variance", "var", "soft_variance"}:
        offsets = torch.arange(-radius, radius + 1, device=probs.device, dtype=probs.dtype)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        dx = dx.reshape(1, expected_channels, 1, 1)
        dy = dy.reshape(1, expected_channels, 1, 1)
        mean_x = (probs * dx).sum(dim=1, keepdim=True)
        mean_y = (probs * dy).sum(dim=1, keepdim=True)
        variance = (probs * ((dx - mean_x).square() + (dy - mean_y).square())).sum(
            dim=1,
            keepdim=True,
        )
        return torch.exp(-variance / max(float(variance_scale), eps)).clamp(
            min=0.0,
            max=1.0,
        ).to(probs.dtype)
    raise ValueError(
        "correlation confidence mode must be one of "
        "{'max', 'uniform', 'entropy', 'variance'}, "
        f"got {mode!r}"
    )


def _as_pool_factors(pool_factors) -> Tuple[int, ...]:
    if pool_factors is None:
        return (4, 2)
    if isinstance(pool_factors, int):
        factors = (int(pool_factors),)
    else:
        factors = tuple(int(v) for v in pool_factors)
    factors = tuple(v for v in factors if v > 1)
    return tuple(sorted(set(factors), reverse=True))


def coarse_to_fine_correlation_flow(
    fmap1: torch.Tensor,
    fmap2: torch.Tensor,
    *,
    radius: int = 4,
    coarse_radius: Optional[int] = None,
    pool_factors=(4, 2),
    temperature: float = 0.05,
    confidence_mode: str = "max",
    confidence_variance_scale: float = 0.5,
    checkpoint_offsets: bool = False,
) -> Dict[str, torch.Tensor]:
    """Estimate rendered->query flow with a coarse-to-fine correlation pyramid."""
    if fmap1.shape != fmap2.shape:
        raise ValueError(
            f"fmap1 and fmap2 must have the same shape, got {tuple(fmap1.shape)} and {tuple(fmap2.shape)}"
        )
    radius = int(radius)
    coarse_radius = int(coarse_radius if coarse_radius is not None else radius)
    factors = _as_pool_factors(pool_factors)
    H, W = fmap1.shape[-2:]

    flow_full = None
    confidence_full = None

    for factor in factors:
        if H // factor < 2 or W // factor < 2:
            continue
        r_s = F.normalize(F.avg_pool2d(fmap1, factor), dim=1)
        q_s = F.normalize(F.avg_pool2d(fmap2, factor), dim=1)
        hs, ws = r_s.shape[-2:]
        if flow_full is None:
            corr = local_correlation(r_s, q_s, coarse_radius)
            flow_s = soft_argmax_flow_from_correlation(
                corr,
                radius=coarse_radius,
                temperature=temperature,
            )
        else:
            flow_s = F.interpolate(flow_full, size=(hs, ws), mode="bilinear", align_corners=False) / float(factor)
            corr = guided_local_correlation(
                r_s,
                q_s,
                flow_s,
                radius=radius,
                checkpoint_offsets=checkpoint_offsets,
            )
            flow_s = flow_s + soft_argmax_flow_from_correlation(
                corr,
                radius=radius,
                temperature=temperature,
            )
        probs = torch.softmax(corr.float() / max(float(temperature), 1e-6), dim=1)
        confidence_s = correlation_confidence_from_probs(
            probs,
            radius=coarse_radius if flow_full is None else radius,
            mode=confidence_mode,
            variance_scale=confidence_variance_scale,
        ).to(fmap1.dtype)
        flow_full = F.interpolate(flow_s, size=(H, W), mode="bilinear", align_corners=False) * float(factor)
        confidence_full = F.interpolate(confidence_s, size=(H, W), mode="bilinear", align_corners=False)

    if flow_full is None:
        corr = local_correlation(fmap1, fmap2, radius)
        flow_full = soft_argmax_flow_from_correlation(corr, radius=radius, temperature=temperature)
        probs = torch.softmax(corr.float() / max(float(temperature), 1e-6), dim=1)
        confidence_full = correlation_confidence_from_probs(
            probs,
            radius=radius,
            mode=confidence_mode,
            variance_scale=confidence_variance_scale,
        ).to(fmap1.dtype)

    corr_final = guided_local_correlation(
        fmap1,
        fmap2,
        flow_full,
        radius=radius,
        checkpoint_offsets=checkpoint_offsets,
    )
    flow_full = flow_full + soft_argmax_flow_from_correlation(
        corr_final,
        radius=radius,
        temperature=temperature,
    )
    probs_final = torch.softmax(corr_final.float() / max(float(temperature), 1e-6), dim=1)
    confidence_final = correlation_confidence_from_probs(
        probs_final,
        radius=radius,
        mode=confidence_mode,
        variance_scale=confidence_variance_scale,
    ).to(fmap1.dtype)
    if confidence_full is not None:
        confidence_final = torch.maximum(confidence_final, confidence_full.to(confidence_final.dtype))

    return {
        "flow": flow_full,
        "confidence": confidence_final.expand(-1, 2, -1, -1).contiguous(),
        "corr": corr_final,
    }


# ======================================================================
#  Building Blocks
# ======================================================================

class ConvGRU(nn.Module):
    """Convolutional GRU cell for iterative flow refinement."""
    def __init__(self, hidden_dim: int = 128, input_dim: int = 64):
        super().__init__()
        self.conv_zr = nn.Conv2d(
            hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
        self.conv_q = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        zr = torch.sigmoid(self.conv_zr(hx))
        z, r = zr.chunk(2, dim=1)
        q = torch.tanh(self.conv_q(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class ResBlock(nn.Module):
    """BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 + skip."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CrossAttentionMatcher(nn.Module):
    """Cross-attention for global feature matching.

    Enriches query features with global context from rendered features,
    helping disambiguate self-similar regions (e.g. corridors) where
    local correlation alone is ambiguous.
    """

    def __init__(self, dim: int = 32, num_heads: int = 4,
                 num_layers: int = 1, downsample: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.downsample = downsample
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm_q': nn.LayerNorm(dim),
                'norm_k': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(
                    dim, num_heads, dropout=dropout, batch_first=True),
                'norm_ffn': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, q_feat: torch.Tensor,
                r_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q_feat.shape

        if self.downsample > 1:
            q_ds = F.avg_pool2d(q_feat, self.downsample)
            r_ds = F.avg_pool2d(r_feat, self.downsample)
        else:
            q_ds, r_ds = q_feat, r_feat

        # (B, C, Hd, Wd) -> (B, N, C)
        q = q_ds.flatten(2).permute(0, 2, 1)
        k = r_ds.flatten(2).permute(0, 2, 1)

        for layer in self.layers:
            q_n = layer['norm_q'](q)
            k_n = layer['norm_k'](k)
            attn_out, _ = layer['attn'](q_n, k_n, k_n)
            q = q + attn_out
            q = q + layer['ffn'](layer['norm_ffn'](q))

        Hd, Wd = q_ds.shape[2], q_ds.shape[3]
        out = q.permute(0, 2, 1).reshape(B, C, Hd, Wd)

        if self.downsample > 1:
            out = F.interpolate(out, size=(H, W),
                                mode='bilinear', align_corners=False)

        return q_feat + out


class CoarsePoseStage(nn.Module):
    """Explicit coarse pose regressor over query/map coarse features."""

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 256,
        pool_hw: int = 4,
        use_transformer: bool = True,
        num_heads: int = 4,
        num_layers: int = 1,
        downsample: int = 2,
        dropout: float = 0.1,
        use_fsm: bool = True,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_fsm = use_fsm
        self.pool_hw = int(pool_hw)

        if use_transformer:
            self.cross_attn = CrossAttentionMatcher(
                dim=feature_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                downsample=downsample,
                dropout=dropout,
            )

        extra_channels = 2 + (1 if use_fsm else 0)
        feat_ch = hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_dim * 2 + extra_channels, hidden_dim, 5, padding=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, feat_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            ResBlock(feat_ch),
            ResBlock(feat_ch),
        )
        self.pool = nn.AdaptiveAvgPool2d(self.pool_hw)
        self.head = nn.Sequential(
            nn.Linear(feat_ch * self.pool_hw * self.pool_hw, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 6),
        )

    @staticmethod
    def _make_positional_channels(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        u = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        v = torch.linspace(-1, 1, height, device=device, dtype=dtype)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        return torch.stack([uu, vv], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(
        self,
        query_coarse: torch.Tensor,
        rendered_coarse: torch.Tensor,
        fsm_spatial_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query_coarse is None:
            raise ValueError('query_coarse is required for coarse-stage refinement')
        if rendered_coarse is None:
            raise ValueError('rendered_coarse is required for coarse-stage refinement')

        if rendered_coarse.shape[-2:] != query_coarse.shape[-2:]:
            rendered_coarse = F.interpolate(
                rendered_coarse,
                size=query_coarse.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        q_feat = F.normalize(query_coarse, dim=1)
        r_feat = F.normalize(rendered_coarse, dim=1)
        if self.use_transformer:
            q_feat = F.normalize(self.cross_attn(q_feat, r_feat), dim=1)

        B, _, H, W = q_feat.shape
        x_parts = [
            q_feat,
            r_feat,
            self._make_positional_channels(B, H, W, q_feat.device, q_feat.dtype),
        ]
        if self.use_fsm:
            if fsm_spatial_conf is None:
                conf = torch.ones(B, 1, H, W, device=q_feat.device, dtype=q_feat.dtype)
            else:
                conf = fsm_spatial_conf
                if conf.ndim == 3:
                    conf = conf.unsqueeze(1)
                if conf.shape[-2:] != (H, W):
                    conf = F.interpolate(conf, size=(H, W), mode='bilinear', align_corners=False)
                conf = conf.to(device=q_feat.device, dtype=q_feat.dtype)
            x_parts.append(conf)

        feat = self.encoder(torch.cat(x_parts, dim=1))
        pooled = self.pool(feat).flatten(1)
        delta_xi = self.head(pooled)
        return {
            'delta_xi': delta_xi,
            'coarse_features': feat,
        }


# ======================================================================
#  ConcatPoseNet
# ======================================================================

class ConcatPoseNet(nn.Module):
    """Pose correction: concat (one-shot) or GRU (iterative) mode."""

    BASE_INTRINSICS: Optional[Dict[str, float]] = None
    IMG_HW: Optional[Tuple[int, int]] = None

    def __init__(
        self,
        feature_dim: int = 64,
        coarse_feature_dim: Optional[int] = None,
        hidden_dim: int = 256,
        irls_iters: int = 0,
        robust_kernel: str = 'huber',
        use_coarse: bool = False,
        # GRU mode
        use_gru: bool = False,
        use_corr_wls: bool = False,
        gru_iters: int = 8,
        local_radius: int = 4,
        proj_dim: int = 32,
        corr_wls_temperature: float = 0.04,
        corr_wls_conf_mode: str = "max",
        corr_wls_conf_variance_scale: float = 0.5,
        coarse_flow_init: bool = False,
        coarse_pool_factor: int = 4,
        use_multiscale_corr: bool = False,
        multiscale_corr_pool_factors=(4, 2),
        multiscale_corr_coarse_radius: Optional[int] = None,
        # Projection mode: 'separate' (legacy), 'shared' (shared+GroupNorm)
        proj_mode: str = 'separate',
        # Full WLS: derive both rotation and translation from flow (no MLP trans)
        full_wls: bool = True,
        # Keep coarse features out of the fine metric head unless an ablation
        # explicitly asks for the legacy coarse-conditioned MLP heads.
        use_coarse_in_fine_head: bool = True,
        # Flow head init: 'zero' (legacy RAFT) or 'kaiming' or 'small' (std=0.01)
        flow_init: str = 'zero',
        # Detach trans: stop gradient from MLP trans back to GRU (forces flow learning)
        detach_trans: bool = False,
        # Detach WLS rot: use WLS rotation for pose update but block gradient through it
        detach_wls_rot: bool = False,
        # Rotation mode: 'wls' (WLS solver), 'mlp' (MLP head), 'hybrid' (WLS+MLP)
        rot_mode: str = 'wls',
        # Cross-attention for global matching
        use_cross_attention: bool = False,
        cross_attn_heads: int = 4,
        cross_attn_layers: int = 1,
        cross_attn_downsample: int = 2,
        cross_attn_dropout: float = 0.1,
        use_two_stage_refine: bool = False,
        coarse_only_first_iter: bool = True,
        coarse_stage_hidden_dim: int = 256,
        coarse_stage_use_transformer: bool = True,
        coarse_stage_heads: int = 4,
        coarse_stage_layers: int = 1,
        coarse_stage_downsample: int = 2,
        coarse_stage_dropout: float = 0.1,
        coarse_stage_pool_hw: int = 4,
        coarse_stage_use_fsm: bool = True,
        pose_update_scale: float = 1.0,
        local_matcher_enabled: bool = False,
        local_matcher_hidden_dim: int = 64,
        local_matcher_zero_init: bool = True,
        local_matcher_residual_scale: float = 1.0,
        local_matcher_context_mode: str = "basic",
        checkpoint_guided_corr: bool = False,
        local_flow_head_enabled: bool = False,
        local_flow_head_hidden_dim: int = 64,
        local_flow_head_zero_init: bool = True,
        local_flow_head_max_flow: Optional[float] = None,
        local_flow_head_base_flow_mode: str = "none",
        local_flow_head_base_temperature: float = 0.05,
        local_flow_head_context_mode: str = "basic",
        corr_wls_use_local_flow_head: bool = False,
        pose_update_trans_scale: float = 1.0,
        pose_update_rot_scale: float = 1.0,
        pose_update_trans_scale_after_first: Optional[float] = None,
        pose_update_rot_scale_after_first: Optional[float] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.coarse_feature_dim = int(coarse_feature_dim or feature_dim)
        self.irls_iters = irls_iters
        self.robust_kernel = robust_kernel
        self.use_coarse = use_coarse
        self.use_gru = use_gru
        self.use_corr_wls = use_corr_wls
        self.gru_iters = gru_iters
        self.local_radius = local_radius
        self.corr_wls_temperature = float(corr_wls_temperature)
        self.corr_wls_conf_mode = str(corr_wls_conf_mode or "max")
        self.corr_wls_conf_variance_scale = float(corr_wls_conf_variance_scale)
        self.coarse_flow_init = coarse_flow_init
        self.coarse_pool_factor = coarse_pool_factor
        self.use_multiscale_corr = bool(use_multiscale_corr)
        self.multiscale_corr_pool_factors = _as_pool_factors(multiscale_corr_pool_factors)
        self.multiscale_corr_coarse_radius = (
            None if multiscale_corr_coarse_radius is None else int(multiscale_corr_coarse_radius)
        )
        self.proj_mode = proj_mode
        self.full_wls = full_wls
        self.use_coarse_in_fine_head = use_coarse_in_fine_head
        self.flow_init = flow_init
        self.detach_trans = detach_trans
        self.detach_wls_rot = detach_wls_rot
        self.rot_mode = rot_mode
        self.use_cross_attention = use_cross_attention
        self.use_two_stage_refine = use_two_stage_refine
        self.coarse_only_first_iter = coarse_only_first_iter
        self.pose_update_scale = float(pose_update_scale)
        self.pose_update_trans_scale = float(pose_update_trans_scale)
        self.pose_update_rot_scale = float(pose_update_rot_scale)
        self.pose_update_trans_scale_after_first = (
            None
            if pose_update_trans_scale_after_first is None
            else float(pose_update_trans_scale_after_first)
        )
        self.pose_update_rot_scale_after_first = (
            None
            if pose_update_rot_scale_after_first is None
            else float(pose_update_rot_scale_after_first)
        )
        self.local_matcher_enabled = bool(local_matcher_enabled)
        self.local_matcher_zero_init = bool(local_matcher_zero_init)
        self.checkpoint_guided_corr = bool(checkpoint_guided_corr)
        self.local_flow_head_enabled = bool(local_flow_head_enabled)
        self.local_flow_head_zero_init = bool(local_flow_head_zero_init)
        self.corr_wls_use_local_flow_head = bool(corr_wls_use_local_flow_head)
        self.external_local_corr_projector = None
        self.external_local_corr_projector_bypass_pose_proj = False

        if self.use_corr_wls and not self.full_wls:
            raise ValueError("use_corr_wls requires full_wls=True")

        self.coarse_pose_stage = None
        if self.use_two_stage_refine:
            self.coarse_pose_stage = CoarsePoseStage(
                feature_dim=feature_dim,
                hidden_dim=coarse_stage_hidden_dim,
                pool_hw=coarse_stage_pool_hw,
                use_transformer=coarse_stage_use_transformer,
                num_heads=coarse_stage_heads,
                num_layers=coarse_stage_layers,
                downsample=coarse_stage_downsample,
                dropout=coarse_stage_dropout,
                use_fsm=coarse_stage_use_fsm,
            )

        in_channels = feature_dim * 2 + 3   # 64+64+1(depth)+2(pos) = 131
        feat_ch = hidden_dim // 2            # 128

        if use_gru or use_corr_wls:
            # -- Projection components for local matching --
            corr_channels = (2 * local_radius + 1) ** 2   # 81

            if proj_mode == 'identity':
                pass
            elif proj_mode == 'shared':
                # Shared projection: same weights for query and render features.
                # Uses GroupNorm (batch-independent) instead of BatchNorm.
                self.proj_shared = nn.Sequential(
                    nn.Conv2d(feature_dim, proj_dim, 1, bias=False),
                    nn.GroupNorm(min(8, proj_dim), proj_dim),
                )
            elif proj_mode == 'shared_linear':
                # Shared linear projection without normalization; when
                # feature_dim == proj_dim it is initialized as an identity map
                # so training starts from the raw DCFF descriptor geometry.
                self.proj_shared = nn.Conv2d(feature_dim, proj_dim, 1, bias=False)
            else:
                # Legacy separate projections (kept for backward compatibility)
                self.proj_query = nn.Sequential(
                    nn.Conv2d(feature_dim, proj_dim, 1, bias=False),
                    nn.BatchNorm2d(proj_dim),
                )
                self.proj_render = nn.Sequential(
                    nn.Conv2d(feature_dim, proj_dim, 1, bias=False),
                    nn.BatchNorm2d(proj_dim),
                )
            self.local_matcher = (
                DepthAwareLocalMatcher(
                    radius=int(local_radius),
                    hidden_dim=int(local_matcher_hidden_dim),
                    zero_init=bool(local_matcher_zero_init),
                    residual_scale=float(local_matcher_residual_scale),
                    context_mode=str(local_matcher_context_mode),
                )
                if self.local_matcher_enabled
                else None
            )
            self.local_flow_head = (
                DepthAwareLocalFlowHead(
                    radius=int(local_radius),
                    hidden_dim=int(local_flow_head_hidden_dim),
                    zero_init=bool(local_flow_head_zero_init),
                    max_flow=local_flow_head_max_flow,
                    base_flow_mode=str(local_flow_head_base_flow_mode),
                    base_temperature=float(local_flow_head_base_temperature),
                    context_mode=str(local_flow_head_context_mode),
                )
                if self.local_flow_head_enabled
                else None
            )
        else:
            self.local_matcher = None
            self.local_flow_head = None

        if use_gru:
            # -- GRU mode components --
            # Context encoder -> GRU hidden state init
            self.context_encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                ResBlock(hidden_dim),
                nn.Conv2d(hidden_dim, feat_ch, 1),
                nn.ReLU(inplace=True),
            )

            # Correlation encoder: corr(81) + flow(2) + conf(1) -> 64
            self.corr_encoder = nn.Sequential(
                nn.Conv2d(corr_channels + 3, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.GELU(),
            )

            self.gru = ConvGRU(hidden_dim=feat_ch, input_dim=64)

            self.gru_flow_head = nn.Sequential(
                nn.Conv2d(feat_ch, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 1),   # 2 (flow) + 1 (confidence)
            )

            # Coarse flow init: predict initial flow at downsampled resolution
            if coarse_flow_init:
                coarse_corr_ch = (2 * local_radius + 1) ** 2
                self.coarse_flow_head = nn.Sequential(
                    nn.Conv2d(coarse_corr_ch, 64, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(32, 2, 1),
                )

            # Cross-attention for global matching in GRU mode
            if use_cross_attention:
                self.cross_attn = CrossAttentionMatcher(
                    dim=proj_dim,
                    num_heads=cross_attn_heads,
                    num_layers=cross_attn_layers,
                    downsample=cross_attn_downsample,
                    dropout=cross_attn_dropout,
                )
        elif use_corr_wls:
            # -- Lightweight depth-aware correlation consumer --
            # No dense CNN/GRU head: projected DCFF features must themselves
            # produce a local match peak; depth/Jacobian converts it to SE(3).
            pass
        else:
            # -- Concat mode components --
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 5, padding=2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                ResBlock(hidden_dim),
                ResBlock(hidden_dim),
                nn.Conv2d(hidden_dim, feat_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(feat_ch),
                nn.ReLU(inplace=True),
                ResBlock(feat_ch),
                ResBlock(feat_ch),
                ResBlock(feat_ch),
            )
            self.flow_head = nn.Conv2d(feat_ch, 2, 1)
            self.confidence_head = nn.Sequential(
                nn.Conv2d(feat_ch, 2, 1),
                nn.Sigmoid(),
            )

        # -- Shared: Translation regression head (skip if full_wls) --
        coarse_context_dim = 128 if use_coarse and use_coarse_in_fine_head else 0
        if use_coarse and use_coarse_in_fine_head:
            self.coarse_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.coarse_feature_dim, coarse_context_dim),
                nn.ReLU(inplace=True),
            )

        if not full_wls:
            self.trans_pool = nn.AdaptiveAvgPool2d(4)
            self.trans_fc = nn.Sequential(
                nn.Linear(feat_ch * 16 + coarse_context_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )

        # -- MLP rotation head (when rot_mode='mlp' or 'hybrid') --
        if rot_mode in ('mlp', 'hybrid'):
            self.rot_pool = nn.AdaptiveAvgPool2d(4)
            coarse_ctx_dim = 128 if use_coarse and use_coarse_in_fine_head else 0
            self.rot_fc = nn.Sequential(
                nn.Linear(feat_ch * 16 + coarse_ctx_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Flow head initialization
        if self.flow_init == 'zero':
            if self.use_gru:
                nn.init.zeros_(self.gru_flow_head[-1].weight)
                nn.init.zeros_(self.gru_flow_head[-1].bias)
            elif hasattr(self, 'flow_head'):
                nn.init.zeros_(self.flow_head.weight)
                nn.init.zeros_(self.flow_head.bias)
        elif self.flow_init == 'small':
            # Small non-zero init: breaks zero-gradient trap without destabilizing WLS
            target = self.gru_flow_head[-1] if self.use_gru else getattr(self, 'flow_head', None)
            if target is not None:
                nn.init.normal_(target.weight, std=0.01)
                nn.init.zeros_(target.bias)
        # 'kaiming' leaves the default kaiming init from the loop above
        if not self.full_wls:
            nn.init.zeros_(self.trans_fc[-1].weight)
            nn.init.zeros_(self.trans_fc[-1].bias)
        if self.rot_mode in ('mlp', 'hybrid'):
            nn.init.zeros_(self.rot_fc[-1].weight)
            nn.init.zeros_(self.rot_fc[-1].bias)
        if self.coarse_pose_stage is not None:
            nn.init.zeros_(self.coarse_pose_stage.head[-1].weight)
            nn.init.zeros_(self.coarse_pose_stage.head[-1].bias)
        if self.local_matcher is not None and self.local_matcher_zero_init:
            nn.init.zeros_(self.local_matcher.refine[-1].weight)
            if self.local_matcher.refine[-1].bias is not None:
                nn.init.zeros_(self.local_matcher.refine[-1].bias)
        if self.local_flow_head is not None and self.local_flow_head_zero_init:
            nn.init.zeros_(self.local_flow_head.predict[-1].weight)
            if self.local_flow_head.predict[-1].bias is not None:
                nn.init.zeros_(self.local_flow_head.predict[-1].bias)
        if (
            self.proj_mode == 'shared_linear'
            and hasattr(self, 'proj_shared')
            and isinstance(self.proj_shared, nn.Conv2d)
            and self.proj_shared.weight.shape[0] == self.proj_shared.weight.shape[1]
        ):
            nn.init.zeros_(self.proj_shared.weight)
            eye = torch.eye(
                self.proj_shared.weight.shape[0],
                device=self.proj_shared.weight.device,
                dtype=self.proj_shared.weight.dtype,
            )
            self.proj_shared.weight.data[:, :, 0, 0].copy_(eye)

    def _scale_intrinsics(self, target_h: int, target_w: int) -> Dict[str, float]:
        if self.BASE_INTRINSICS is None:
            raise RuntimeError("BASE_INTRINSICS not set")
        orig_h, orig_w = self.IMG_HW
        sx = target_w / orig_w
        sy = target_h / orig_h
        return {
            'fx': self.BASE_INTRINSICS['fx'] * sx,
            'fy': self.BASE_INTRINSICS['fy'] * sy,
            'cx': self.BASE_INTRINSICS['cx'] * sx,
            'cy': self.BASE_INTRINSICS['cy'] * sy,
        }

    def _make_context_input(self, query_fine, rendered_fine, depth, B, H, W, device):
        depth_4d = depth.unsqueeze(1) if depth.ndim == 3 else depth
        d_max = depth_4d.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        depth_norm = depth_4d / d_max
        u = torch.linspace(-1, 1, W, device=device)
        v = torch.linspace(-1, 1, H, device=device)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        pos = torch.stack([uu, vv]).unsqueeze(0).expand(B, -1, -1, -1)
        # Normalize both feature maps to unit norm for consistent magnitudes
        q_ctx = F.normalize(query_fine, dim=1)
        r_ctx = F.normalize(rendered_fine, dim=1)
        return torch.cat([q_ctx, r_ctx, depth_norm, pos], dim=1)

    def _solve_and_regress(self, flow, confidence, depth, intrinsics,
                           feat_for_trans, query_coarse, irls_iters,
                           robust_kernel, rendered_coarse=None):
        with torch.cuda.amp.autocast(enabled=False):
            flow_f = flow.float()
            conf_f = confidence.float()
            depth_s = depth.float() if depth.ndim == 3 else depth.squeeze(1).float()
            Ju, Jv, valid = compute_image_jacobian(depth_s, intrinsics)
            _irls = irls_iters if irls_iters is not None else self.irls_iters
            _kern = robust_kernel if robust_kernel is not None else self.robust_kernel
            delta_xi_full = diff_pose_solve(
                flow_f, conf_f, Ju, Jv, valid,
                irls_iters=_irls, robust_kernel=_kern,
            )
        if self.full_wls:
            delta_xi = delta_xi_full
            if self.rot_mode in ('mlp', 'hybrid'):
                # Keep translation from depth-WLS while allowing a learned
                # rotation correction; WLS rotation is weak under noisy query
                # flow and was previously forced by full_wls=True.
                rot_feat = self.rot_pool(feat_for_trans).flatten(1)
                if (
                    self.use_coarse
                    and self.use_coarse_in_fine_head
                    and query_coarse is not None
                ):
                    coarse_input = query_coarse
                    if rendered_coarse is not None:
                        if rendered_coarse.shape[-2:] != query_coarse.shape[-2:]:
                            rendered_coarse = F.interpolate(
                                rendered_coarse,
                                query_coarse.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )
                        coarse_input = query_coarse - rendered_coarse
                    coarse_ctx = self.coarse_encoder(coarse_input)
                    rot_feat = torch.cat([rot_feat, coarse_ctx], dim=1)
                if self.rot_mode == 'hybrid':
                    rot_delta = delta_xi_full[:, 3:].float().detach() + self.rot_fc(rot_feat)
                else:
                    rot_delta = self.rot_fc(rot_feat)
                delta_xi = torch.cat([delta_xi_full[:, :3].float(), rot_delta.float()], dim=1)
        else:
            # Translation from MLP
            feat_input = feat_for_trans.detach() if self.detach_trans else feat_for_trans
            trans_feat = self.trans_pool(feat_input).flatten(1)
            coarse_ctx = None
            if self.use_coarse and self.use_coarse_in_fine_head and query_coarse is not None:
                coarse_ctx = self.coarse_encoder(query_coarse)
                trans_feat = torch.cat([trans_feat, coarse_ctx], dim=1)
            trans_delta = self.trans_fc(trans_feat)

            # Rotation: choose source based on rot_mode
            if self.rot_mode == 'mlp':
                rot_feat = self.rot_pool(feat_input).flatten(1)
                if coarse_ctx is not None:
                    rot_feat = torch.cat([rot_feat, coarse_ctx], dim=1)
                rot_delta = self.rot_fc(rot_feat)
            elif self.rot_mode == 'hybrid':
                # WLS rotation (detached) + MLP residual
                wls_rot = delta_xi_full[:, 3:].float().detach()
                rot_feat = self.rot_pool(feat_input).flatten(1)
                if coarse_ctx is not None:
                    rot_feat = torch.cat([rot_feat, coarse_ctx], dim=1)
                rot_delta = wls_rot + self.rot_fc(rot_feat)
            else:  # 'wls' (default)
                wls_rot = delta_xi_full[:, 3:].float()
                if self.detach_wls_rot:
                    wls_rot = wls_rot.detach()
                rot_delta = wls_rot

            delta_xi = torch.cat([trans_delta.float(), rot_delta.float()], dim=1)
        return delta_xi, delta_xi_full

    # -- Forward dispatch --

    def should_run_coarse_stage(self, outer_iter: int = 0) -> bool:
        if not self.use_two_stage_refine or self.coarse_pose_stage is None:
            return False
        return outer_iter == 0 or not self.coarse_only_first_iter

    def forward_coarse_stage(
        self,
        query_coarse: torch.Tensor,
        rendered_coarse: torch.Tensor,
        fsm_spatial_conf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.coarse_pose_stage is None:
            raise RuntimeError('Coarse stage requested but use_two_stage_refine is disabled')
        return self.coarse_pose_stage(
            query_coarse,
            rendered_coarse,
            fsm_spatial_conf=fsm_spatial_conf,
        )

    def forward_fine_stage(self, query_fine, rendered_fine, depth, intrinsics,
                           irls_iters=None, robust_kernel=None, query_coarse=None,
                           rendered_coarse=None):
        return self.forward(
            query_fine,
            rendered_fine,
            depth,
            intrinsics,
            irls_iters=irls_iters,
            robust_kernel=robust_kernel,
            query_coarse=query_coarse,
            rendered_coarse=rendered_coarse,
        )

    def forward(self, query_fine, rendered_fine, depth, intrinsics,
                irls_iters=None, robust_kernel=None, query_coarse=None,
                rendered_coarse=None):
        if self.use_corr_wls:
            return self._forward_corr_wls(
                query_fine, rendered_fine, depth, intrinsics,
                irls_iters, robust_kernel, query_coarse, rendered_coarse=rendered_coarse)
        elif self.use_gru:
            return self._forward_gru(
                query_fine, rendered_fine, depth, intrinsics,
                irls_iters, robust_kernel, query_coarse, rendered_coarse=rendered_coarse)
        else:
            return self._forward_concat(
                query_fine, rendered_fine, depth, intrinsics,
                irls_iters, robust_kernel, query_coarse)

    def _forward_concat(self, query_fine, rendered_fine, depth, intrinsics,
                        irls_iters, robust_kernel, query_coarse):
        B, _, H, W = query_fine.shape
        x = self._make_context_input(
            query_fine, rendered_fine, depth, B, H, W, query_fine.device)
        feat = self.encoder(x)
        flow = self.flow_head(feat)
        confidence = self.confidence_head(feat)
        delta_xi, delta_xi_full = self._solve_and_regress(
            flow, confidence, depth, intrinsics, feat,
            query_coarse, irls_iters, robust_kernel)
        return {
            'delta_xi': delta_xi,
            'delta_xi_full': delta_xi_full,
            'flow': flow,
            'confidence': confidence,
        }

    def _project_for_local_corr(self, query_fine, rendered_fine):
        external_projector = getattr(self, 'external_local_corr_projector', None)
        if external_projector is not None:
            if hasattr(external_projector, 'project_query'):
                q_external = external_projector.project_query(query_fine)
                r_external = external_projector.project_render(rendered_fine)
            else:
                q_external = external_projector(query_fine)
                r_external = external_projector(rendered_fine)
            if bool(getattr(self, 'external_local_corr_projector_bypass_pose_proj', False)):
                return F.normalize(q_external, dim=1), F.normalize(r_external, dim=1)
            query_fine = q_external
            rendered_fine = r_external

        if self.proj_mode == 'identity':
            q_proj = F.normalize(query_fine, dim=1)
            r_proj = F.normalize(rendered_fine, dim=1)
        elif self.proj_mode in ('shared', 'shared_linear'):
            q_proj = F.normalize(self.proj_shared(query_fine), dim=1)
            r_proj = F.normalize(self.proj_shared(rendered_fine), dim=1)
        else:
            q_proj = F.normalize(self.proj_query(query_fine), dim=1)
            r_proj = F.normalize(self.proj_render(rendered_fine), dim=1)
        if self.use_cross_attention:
            q_proj = self.cross_attn(q_proj, r_proj)
            q_proj = F.normalize(q_proj, dim=1)
        return q_proj, r_proj

    @staticmethod
    def _depth_valid_mask(depth: Optional[torch.Tensor], target_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
        if depth is None:
            return None
        depth_f = depth.float()
        if depth_f.ndim == 3:
            depth_f = depth_f.unsqueeze(1)
        if depth_f.shape[-2:] != target_hw:
            depth_f = F.interpolate(depth_f, target_hw, mode='nearest')
        return (depth_f > 0.05).float()

    def _apply_local_matcher(
        self,
        corr: torch.Tensor,
        depth: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        intrinsics=None,
    ) -> torch.Tensor:
        matcher = getattr(self, 'local_matcher', None)
        if matcher is None:
            return corr
        if valid_mask is None:
            valid_mask = self._depth_valid_mask(depth, corr.shape[-2:])
        if "intrinsics" in inspect.signature(matcher.forward).parameters:
            return matcher(corr, depth=depth, valid_mask=valid_mask, intrinsics=intrinsics)
        return matcher(corr, depth=depth, valid_mask=valid_mask)

    def _local_flow_head_init(
        self,
        corr: torch.Tensor,
        depth: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
        intrinsics=None,
    ) -> Optional[dict]:
        head = getattr(self, 'local_flow_head', None)
        if head is None:
            return None
        if valid_mask is None:
            valid_mask = self._depth_valid_mask(depth, corr.shape[-2:])
        return head(corr, depth=depth, valid_mask=valid_mask, intrinsics=intrinsics)

    def _forward_corr_wls(self, query_fine, rendered_fine, depth, intrinsics,
                          irls_iters, robust_kernel, query_coarse, rendered_coarse=None):
        if query_fine.shape[-2:] != rendered_fine.shape[-2:]:
            query_fine = F.interpolate(
                query_fine,
                rendered_fine.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        q_proj, r_proj = self._project_for_local_corr(query_fine, rendered_fine)
        use_local_flow_head = (
            bool(getattr(self, 'corr_wls_use_local_flow_head', False))
            and getattr(self, 'local_flow_head', None) is not None
        )
        if self.use_multiscale_corr:
            corr_result = coarse_to_fine_correlation_flow(
                r_proj,
                q_proj,
                radius=self.local_radius,
                coarse_radius=self.multiscale_corr_coarse_radius,
                pool_factors=self.multiscale_corr_pool_factors,
                temperature=self.corr_wls_temperature,
                confidence_mode=self.corr_wls_conf_mode,
                confidence_variance_scale=self.corr_wls_conf_variance_scale,
                checkpoint_offsets=self.checkpoint_guided_corr,
            )
            corr = self._apply_local_matcher(corr_result["corr"], depth, intrinsics=intrinsics)
            if use_local_flow_head:
                flow_init = self._local_flow_head_init(corr, depth, intrinsics=intrinsics)
                flow = corr_result["flow"] + flow_init["flow"]
                confidence = flow_init.get("confidence")
                if confidence is None:
                    confidence = torch.ones_like(flow[:, :1])
                confidence = confidence.to(flow.dtype).expand(-1, 2, -1, -1).contiguous()
            elif corr is corr_result["corr"]:
                flow = corr_result["flow"]
                confidence = corr_result["confidence"]
            else:
                flow = corr_result["flow"] + soft_argmax_flow_from_correlation(
                    corr,
                    radius=self.local_radius,
                    temperature=self.corr_wls_temperature,
                )
                probs = torch.softmax(
                    corr.float() / max(float(self.corr_wls_temperature), 1e-6),
                    dim=1,
                )
                confidence = correlation_confidence_from_probs(
                    probs,
                    radius=self.local_radius,
                    mode=self.corr_wls_conf_mode,
                    variance_scale=self.corr_wls_conf_variance_scale,
                ).to(flow.dtype)
                confidence = confidence.expand(-1, 2, -1, -1).contiguous()
        else:
            corr = local_correlation(r_proj, q_proj, self.local_radius)
            corr = self._apply_local_matcher(corr, depth, intrinsics=intrinsics)
            if use_local_flow_head:
                flow_init = self._local_flow_head_init(corr, depth, intrinsics=intrinsics)
                flow = flow_init["flow"]
                confidence = flow_init.get("confidence")
                if confidence is None:
                    confidence = torch.ones_like(flow[:, :1])
                confidence = confidence.to(flow.dtype).expand(-1, 2, -1, -1).contiguous()
            else:
                flow = soft_argmax_flow_from_correlation(
                    corr,
                    radius=self.local_radius,
                    temperature=self.corr_wls_temperature,
                )
                probs = torch.softmax(
                    corr.float() / max(float(self.corr_wls_temperature), 1e-6),
                    dim=1,
                )
                confidence = correlation_confidence_from_probs(
                    probs,
                    radius=self.local_radius,
                    mode=self.corr_wls_conf_mode,
                    variance_scale=self.corr_wls_conf_variance_scale,
                ).to(flow.dtype)
                confidence = confidence.expand(-1, 2, -1, -1).contiguous()

        delta_xi, delta_xi_full = self._solve_and_regress(
            flow, confidence, depth, intrinsics, r_proj,
            query_coarse, irls_iters, robust_kernel, rendered_coarse=rendered_coarse)
        return {
            'delta_xi': delta_xi,
            'delta_xi_full': delta_xi_full,
            'flow': flow,
            'confidence': confidence,
            'corr': corr,
        }

    def _forward_gru(self, query_fine, rendered_fine, depth, intrinsics,
                     irls_iters, robust_kernel, query_coarse, rendered_coarse=None):
        B, _, H, W = query_fine.shape
        device = query_fine.device

        # Project features for correlation (L2-normalize).
        #
        # Flow in this module is defined on rendered/current pixels:
        #   rendered pixel p -> query pixel p + flow[p]
        # Therefore local correlation must be centered on rendered features and
        # search in the query feature map.  The earlier query-centered ordering
        # produced the inverse/query-indexed match, which is especially harmful
        # for translation because the downstream WLS/PnP solvers consume
        # rendered-indexed depth.
        q_proj, r_proj = self._project_for_local_corr(query_fine, rendered_fine)

        # Context -> initial GRU hidden state
        ctx_input = self._make_context_input(
            query_fine, rendered_fine, depth, B, H, W, device)
        h = self.context_encoder(ctx_input)

        init_corr = None
        flow_init = None
        init_flow = None
        init_confidence = None
        # Coarse flow initialization: predict flow at downsampled resolution
        # then upsample to full resolution to seed the GRU
        if self.use_multiscale_corr:
            corr_result = coarse_to_fine_correlation_flow(
                r_proj,
                q_proj,
                radius=self.local_radius,
                coarse_radius=self.multiscale_corr_coarse_radius,
                pool_factors=self.multiscale_corr_pool_factors,
                temperature=self.corr_wls_temperature,
                confidence_mode=self.corr_wls_conf_mode,
                confidence_variance_scale=self.corr_wls_conf_variance_scale,
                checkpoint_offsets=self.checkpoint_guided_corr,
            )
            flow = corr_result["flow"]
            init_corr = corr_result["corr"]
            init_confidence = corr_result["confidence"][:, :1]
        elif self.coarse_flow_init:
            pf = self.coarse_pool_factor
            q_ds = F.avg_pool2d(q_proj, pf)            # (B, 32, H/pf, W/pf)
            r_ds = F.avg_pool2d(r_proj, pf)
            q_ds = F.normalize(q_ds, dim=1)
            r_ds = F.normalize(r_ds, dim=1)
            coarse_corr = local_correlation(r_ds, q_ds, self.local_radius)
            coarse_flow = self.coarse_flow_head(coarse_corr)   # (B, 2, H/pf, W/pf)
            # Upsample to full resolution and scale displacement by pool factor
            flow = F.interpolate(
                coarse_flow, size=(H, W), mode='bilinear', align_corners=False
            ) * pf
        else:
            init_corr = local_correlation(r_proj, q_proj, self.local_radius)
            init_corr = self._apply_local_matcher(init_corr, depth, intrinsics=intrinsics)
            flow_init = self._local_flow_head_init(init_corr, depth, intrinsics=intrinsics)
            if flow_init is None:
                flow = torch.zeros(B, 2, H, W, device=device, dtype=q_proj.dtype)
            else:
                flow = flow_init['flow']
                init_flow = flow_init['flow']
                init_confidence = flow_init['confidence']

        conf = torch.full((B, 1, H, W), 0.5, device=device, dtype=q_proj.dtype)
        if init_confidence is not None:
            conf = init_confidence
        elif flow_init is not None:
            conf = flow_init['confidence']
        flow_preds: List[torch.Tensor] = []

        for i in range(self.gru_iters):
            if i == 0 and not self.coarse_flow_init and init_corr is not None:
                corr = init_corr
            else:
                corr = guided_local_correlation(
                    r_proj,
                    q_proj,
                    flow.detach(),
                    self.local_radius,
                    checkpoint_offsets=self.checkpoint_guided_corr,
                )
            corr = self._apply_local_matcher(corr, depth, intrinsics=intrinsics)

            inp = torch.cat([corr, flow, conf], dim=1)
            inp_encoded = self.corr_encoder(inp)
            h = self.gru(h, inp_encoded)

            out = self.gru_flow_head(h)
            delta_flow = out[:, :2]
            conf = torch.sigmoid(out[:, 2:3])
            flow = flow + delta_flow
            flow_preds.append(flow)

        if self.gru_iters <= 0:
            flow_preds.append(flow)

        confidence = conf.expand(-1, 2, -1, -1).contiguous()

        delta_xi, delta_xi_full = self._solve_and_regress(
            flow, confidence, depth, intrinsics, h,
            query_coarse, irls_iters, robust_kernel, rendered_coarse=rendered_coarse)

        result = {
            'delta_xi': delta_xi,
            'delta_xi_full': delta_xi_full,
            'flow': flow,
            'confidence': confidence,
            'flow_preds': flow_preds,
        }
        if init_flow is not None:
            result['init_flow'] = init_flow
        if init_confidence is not None:
            result['init_confidence'] = init_confidence
        if init_corr is not None:
            result['init_corr'] = init_corr
        return result

    # -- GT flow computation --

    @staticmethod
    def compute_gt_flow(pose_init, pose_gt, depth, target_hw, intrinsics=None):
        B, H, W = depth.shape
        device = depth.device
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth
        Y = (v_coords - cy) / fy * depth
        Z = depth
        pts = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = pose_gt @ torch.linalg.inv(pose_init)
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)

        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow_gt = torch.cat([
            u_gt - u_coords.unsqueeze(1),
            v_gt - v_coords.unsqueeze(1),
        ], dim=1)

        valid = (
            (depth.unsqueeze(1) > 0.05)
            & (Z_gt_raw > 0.1)
            & (u_gt > -0.5) & (u_gt < W - 0.5)
            & (v_gt > -0.5) & (v_gt < H - 0.5)
        ).float()
        flow_gt = flow_gt * valid

        tH, tW = target_hw
        if tH != H or tW != W:
            sx, sy = tW / W, tH / H
            flow_gt = F.interpolate(
                flow_gt, (tH, tW), mode='bilinear', align_corners=False)
            flow_gt[:, 0] *= sx
            flow_gt[:, 1] *= sy
            valid = F.interpolate(valid, (tH, tW), mode='nearest')

        return flow_gt, valid
