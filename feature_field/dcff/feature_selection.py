"""
Feature Selection Module (FSM) — learns to select and refine fine/coarse features.

Three components:
  1. Spatial confidence gate: Alpha/depth-based per-pixel confidence
  2. Channel selection: Gumbel-Softmax routing between fine and coarse streams
  3. Cross-attention fusion: Fine queries coarse for semantic context

Integrated after DeferredCascadedRenderer.forward(), before loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax


class SpatialConfidenceGate(nn.Module):
    """Per-pixel confidence from alpha and depth signals.
    
    Learns which pixels are reliable (foreground, high alpha) vs
    unreliable (background, low alpha). Used to mask losses.
    """

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(
        self,
        alpha: torch.Tensor,
        depth: torch.Tensor,
        normals: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute per-pixel confidence in [0, 1].
        
        Args:
            alpha: [B, 1, H, W] rendered opacity
            depth: [B, 1, H, W] rendered depth (for edge detection)
            normals: [B, 3, H, W] optional surface normals
        """
        B, _, H, W = alpha.shape
        
        # Edge-aware depth: compute Laplacian-like depth gradient
        if depth.numel() > 0:
            d_up = F.interpolate(depth, (H * 2, W * 2), mode='bilinear', align_corners=False)
            depth_edge = F.avg_pool2d(
                torch.abs(d_up[:, :, ::2, ::2] - depth), 3, padding=1, stride=1
            )
        else:
            depth_edge = torch.zeros_like(alpha)

        # Depth uncertainty proxy (high variance near edges)
        depth_std = F.avg_pool2d(
            (depth - F.avg_pool2d(depth, 3, padding=1, stride=1)).abs(),
            3, padding=1, stride=1,
        )

        # Confidence inputs: alpha (foreground), depth edge (ignore edge pixels)
        conf_input = torch.cat([alpha, depth_edge, depth_std], dim=1)
        
        # Learn confidence: high alpha + low edge → confident
        conf = self.net(conf_input).sigmoid()
        
        # Penalize edge regions (high depth edge → lower confidence)
        edge_mask = torch.exp(-depth_edge * 5.0)
        conf = conf * edge_mask
        
        return conf.clamp_(0.0, 1.0)


class ChannelSelector(nn.Module):
    """Lightweight channel routing between fine and coarse streams.
    
    Uses global average pooling to predict per-head routing weights,
    then applies channel-wise selection via broadcasting.
    """

    def __init__(self, feature_dim: int = 64, num_heads: int = 4, routing_mode: str = 'categorical'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.routing_mode = routing_mode

        self.fine_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        self.coarse_proj = nn.Conv2d(feature_dim, feature_dim, 1)
        out_dim = num_heads if routing_mode == 'categorical' else num_heads * 2
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, out_dim),
        )
        nn.init.zeros_(self.routing[-1].weight)
        nn.init.zeros_(self.routing[-1].bias)

    def forward(
        self,
        fine: torch.Tensor,
        coarse: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route between fine and coarse features.
        
        Args:
            fine: [B, C, H, W]
            coarse: [B, C, H, W]
            temperature: Gumbel-Softmax temperature
            hard: if True, use argmax
            
        Returns:
            fused: [B, C, H, W]
            gate: [B, num_heads] routing weights
        """
        B, C, H, W = fine.shape
        fine_p = self.fine_proj(fine)
        coarse_p = self.coarse_proj(coarse)

        routing_logits = self.routing(fine_p)
        if self.routing_mode == 'categorical':
            if self.training:
                gate = gumbel_softmax(routing_logits, tau=temperature, dim=-1, hard=hard)
            else:
                gate = torch.softmax(routing_logits / max(temperature, 1e-6), dim=-1)
                if hard:
                    hard_idx = gate.argmax(dim=-1, keepdim=True)
                    hard_gate = torch.zeros_like(gate).scatter_(-1, hard_idx, 1.0)
                    gate = hard_gate

            gate_expanded = gate.view(B, self.num_heads, 1, 1, 1)
            fine_reshaped = fine_p.view(B, self.num_heads, self.head_dim, H, W)
            coarse_reshaped = coarse_p.view(B, self.num_heads, self.head_dim, H, W)
            fused_heads = fine_reshaped * gate_expanded + coarse_reshaped * (1 - gate_expanded)
        else:
            logits = routing_logits.view(B, self.num_heads, 2)
            if self.training:
                gate2 = gumbel_softmax(logits, tau=temperature, dim=-1, hard=hard)
            else:
                gate2 = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                if hard:
                    hard_idx = gate2.argmax(dim=-1, keepdim=True)
                    gate2 = torch.zeros_like(gate2).scatter_(-1, hard_idx, 1.0)

            gate = gate2[..., 0]
            gate_expanded = gate.view(B, self.num_heads, 1, 1, 1)
            fine_reshaped = fine_p.view(B, self.num_heads, self.head_dim, H, W)
            coarse_reshaped = coarse_p.view(B, self.num_heads, self.head_dim, H, W)
            fused_heads = fine_reshaped * gate_expanded + coarse_reshaped * (1 - gate_expanded)

        fused = fused_heads.reshape(B, C, H, W)
        return fused, gate


class CrossAttentionFusion(nn.Module):
    """Efficient fusion: fine features enriched with coarse context via pooling.

    Uses global average pooled coarse features as context, concatenated with fine
    features, then fused through MLP. This avoids O(HW) memory overhead of
    full cross-attention while still capturing coarse-to-fine relationships.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(4, feature_dim)
        mlp_hidden = int(feature_dim * mlp_ratio)
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, mlp_hidden, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden, feature_dim, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        """Fuse fine and coarse features efficiently.
        
        Args:
            fine: [B, C, H, W] fine geometric features
            coarse: [B, C, H, W] coarse semantic features
            
        Returns:
            fused: [B, C, H, W] fused features
        """
        fine_norm = self.norm(fine)
        # Global average pooled coarse context [B, C, 1, 1]
        coarse_ctx = F.adaptive_avg_pool2d(coarse, 1)
        # Broadcast context to spatial dimensions
        coarse_ctx = coarse_ctx.expand_as(fine_norm)
        # Concatenate fine + coarse context
        fused_input = torch.cat([fine_norm, coarse_ctx], dim=1)
        # MLP fusion
        delta = self.fusion(fused_input)
        return fine + self.residual_scale * delta


class FeatureSelectionModule(nn.Module):
    """Feature Selection Module: spatial gate + channel select + cross-fusion.
    
    Placed after DeferredCascadedRenderer.forward(), applies three refinement
    stages before features are used for loss computation:
    
    1. Spatial confidence weighting (per-pixel reliability)
    2. Fine-Coarse channel routing (Gumbel-Softmax)
    3. Cross-attention fusion (fine queries coarse)
    
    Args:
        feature_dim: Feature channel dimension (default 64)
        hidden_dim: Hidden dimension for sub-modules
        num_heads: Number of attention heads
        use_channel_select: Enable channel selection (default True)
        use_cross_attn: Enable cross-attention fusion (default True)
        use_spatial_conf: Enable spatial confidence gating (default True)
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 32,
        num_heads: int = 4,
        channel_routing_mode: str = 'categorical',
        use_channel_select: bool = True,
        use_cross_attn: bool = True,
        use_spatial_conf: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_channel_select = use_channel_select
        self.use_cross_attn = use_cross_attn
        self.use_spatial_conf = use_spatial_conf

        if use_spatial_conf:
            self.spatial_gate = SpatialConfidenceGate(feature_dim, hidden_dim)

        if use_channel_select:
            self.channel_selector = ChannelSelector(
                feature_dim,
                max(1, num_heads),
                routing_mode=channel_routing_mode,
            )

        if use_cross_attn:
            self.cross_attn = CrossAttentionFusion(feature_dim, max(1, num_heads))

    def forward(
        self,
        fine_features: torch.Tensor,
        coarse_features: torch.Tensor,
        alpha: torch.Tensor,
        depth: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> dict:
        """Apply feature selection to fine/coarse features.
        
        Args:
            fine_features: [B, C, H, W] fine geometric features
            coarse_features: [B, C, H, W] coarse semantic features
            alpha: [B, 1, H, W] rendered alpha for spatial gating
            depth: [B, 1, H, W] rendered depth for spatial gating
            temperature: Gumbel-Softmax temperature
            
        Returns:
            dict with:
                'fine_features': [B, C, H, W] refined fine features
                'coarse_features': [B, C, H, W] refined coarse features
                'spatial_confidence': [B, 1, H, W] spatial confidence map
                'channel_weights': [B, num_heads] per-head routing weights
        """
        fine = fine_features
        coarse = coarse_features
        
        # 1. Spatial confidence gating
        spatial_conf = None
        if self.use_spatial_conf:
            spatial_conf = self.spatial_gate(alpha, depth)
            fine = fine * spatial_conf
            coarse = coarse * spatial_conf

        # 2. Channel selection (fine vs coarse routing)
        channel_weights = None
        if self.use_channel_select:
            fine, channel_weights = self.channel_selector(
                fine, coarse, temperature=temperature, hard=hard
            )

        # 3. Cross-attention fusion
        if self.use_cross_attn:
            fine = self.cross_attn(fine, coarse)

        return {
            'fine_features': fine,
            'coarse_features': coarse,
            'spatial_confidence': spatial_conf,
            'channel_weights': channel_weights,
        }
